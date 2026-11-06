import torch
import logging
import json
import os
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.glue.model.adapter_builder import AdapterModel
try:
    from datasets import load_metric
except ImportError:
    from evaluate import load as load_metric
import numpy as np

logger = logging.getLogger(__name__)


class GLUETrainer(GeneralTorchTrainer):
    def get_model_para(self):
        model_parameters = super().get_model_para()
        if str(self.cfg.federate.method).lower() == 'cenmag_new':
            from federatedscope.glue.cenmag.magnitude import \
                effective_magnitude_state
            cenmag_state = getattr(self.ctx.model, '_cenmag_state', None)
            if cenmag_state is None:
                raise RuntimeError('CenMag state is unavailable during upload')
            effective = effective_magnitude_state(self.ctx.model)
            for key, value in effective.items():
                model_parameters[key] = value.detach().clone()
            cenmag_state.validate_communicated_state(
                model_parameters, 'CenMag client upload')
        return model_parameters

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                from federatedscope.glue.cenmag.model_utils import is_4bit_model
                if is_4bit_model(ctx.model):
                    raise RuntimeError(
                        'Whole-model half conversion is unsafe for a 4-bit model')
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed


            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            from federatedscope.glue.cenmag.model_utils import is_4bit_model
            if not is_4bit_model(ctx.model):
                ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                if ctx.cfg.llm.adapter.args[0].get('adapter_method', '') == "vera":

                    vera_params = [param for name, param in ctx.model.named_parameters() if "vera" in name and param.requires_grad]
                    other_params = [param for name, param in ctx.model.named_parameters() if "vera" not in name and param.requires_grad]
                    optimizer_grouped_parameters = [
                        {'params': vera_params, 'lr': ctx.cfg.train.optimizer.lr},
                        {'params': other_params, 'lr': ctx.cfg.train.vera.lr_c}
                    ]
                    from transformers import AdamW, get_linear_schedule_with_warmup
                    ctx.optimizer = AdamW(optimizer_grouped_parameters, no_deprecation_warning=True)
                    ctx.scheduler = get_linear_schedule_with_warmup(
                                    ctx.optimizer,
                                    num_warmup_steps=0.06 * ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num,
                                    num_training_steps=ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num
                    )
                else:
                    optimizer_input = ctx.model
                    if str(ctx.cfg.federate.method).lower() == 'cenmag_new':
                        from federatedscope.glue.cenmag.magnitude import \
                            cenmag_optimizer_parameters
                        optimizer_input = cenmag_optimizer_parameters(ctx.model)
                    ctx.optimizer = get_optimizer(
                        optimizer_input,
                        **ctx.cfg[ctx.cur_mode].optimizer)
                    ctx.scheduler = get_scheduler(
                        ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.magnitude_regular_sample_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.total_loss_sample_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)
        if ctx.cur_mode == MODE.TRAIN and os.environ.get('CENMAG_TRACE_DIR'):
            from federatedscope.glue.cenmag.trace import write_trainer_trace
            write_trainer_trace(ctx, 'before_training')

    def _hook_on_batch_forward_regularizer(self, ctx):
        if str(ctx.cfg.federate.method).lower() != 'cenmag_new':
            return super()._hook_on_batch_forward_regularizer(ctx)
        from federatedscope.glue.cenmag.magnitude import \
            magnitude_regularization_loss
        regularization = magnitude_regularization_loss(
            ctx.model, float(ctx.cfg.cenmag.lambda_m))
        if not bool(torch.isfinite(regularization).all()):
            raise FloatingPointError(
                'CenMag encountered a non-finite magnitude regularizer')
        ctx.loss_regular = CtxVar(regularization, LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(
            ctx.loss_batch + regularization, LIFECYCLE.BATCH)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        preds = outputs.logits.argmax(dim=-1)
        loss = outputs.loss
        if str(ctx.cfg.federate.method).lower() == 'cenmag_new' and not \
                bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                'CenMag encountered a non-finite training loss; refusing to '
                'skip the batch or continue')
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if str(ctx.cfg.federate.method).lower() == 'cenmag_new':
                cenmag_state = getattr(ctx.model, '_cenmag_state', None)
                if cenmag_state is not None and not \
                        cenmag_state.last_gradient_validation:
                    from federatedscope.glue.cenmag.magnitude import (
                        binding_modules, magnitude_ownership_audit)
                    groups = {
                        'lora_A': [],
                        'lora_B': [],
                        'client_local_magnitude': [],
                        'classifier': [],
                    }
                    for name, parameter in ctx.model.named_parameters():
                        for marker in groups:
                            if marker in name and parameter.requires_grad:
                                groups[marker].append((name, parameter.grad))
                                break
                    summary = {}
                    for marker, entries in groups.items():
                        if not entries:
                            raise RuntimeError(
                                f'CenMag found no trainable {marker} parameters')
                        missing = [name for name, gradient in entries
                                   if gradient is None]
                        non_finite = [
                            name for name, gradient in entries
                            if gradient is not None and
                            not torch.isfinite(gradient).all()
                        ]
                        if missing or non_finite:
                            raise RuntimeError(
                                f'CenMag gradient invariant failed for {marker}; '
                                f'missing={missing}, non_finite={non_finite}')
                        summary[marker] = {
                            'tensor_count': len(entries),
                            'all_finite': True,
                        }
                    magnitude_modules = binding_modules(ctx.model)
                    local_grad_norms = [
                        float(module.client_local_magnitude.grad.float().norm().item())
                        for module in magnitude_modules.values()
                        if module.client_local_magnitude.grad is not None
                    ]
                    if len(local_grad_norms) != len(magnitude_modules) or not any(
                            norm > 0.0 for norm in local_grad_norms):
                        raise RuntimeError(
                            'CenMag local magnitude received no non-zero task gradient')
                    if any(getattr(module.global_magnitude, 'grad', None) is not None
                           for module in magnitude_modules.values()):
                        raise RuntimeError('CenMag global magnitude received a gradient')
                    ownership = magnitude_ownership_audit(
                        ctx.model, ctx.optimizer)
                    required_ownership = {
                        'all_global_frozen': True,
                        'all_local_trainable': True,
                        'all_native_frozen': True,
                        'local_in_optimizer': True,
                        'global_in_optimizer': False,
                        'native_in_optimizer': False,
                        'duplicate_trainable_effective_magnitude': False,
                    }
                    for key, expected in required_ownership.items():
                        if ownership[key] != expected:
                            raise RuntimeError(
                                f'CenMag ownership invariant failed: {key}='
                                f'{ownership[key]}, expected {expected}')
                    optimizer_names = []
                    name_by_id = {
                        id(parameter): name
                        for name, parameter in ctx.model.named_parameters()
                    }
                    for group_index, group in enumerate(ctx.optimizer.param_groups):
                        optimizer_names.append({
                            'group': group_index,
                            'parameters': [name_by_id.get(id(parameter), '<unknown>')
                                           for parameter in group['params']],
                        })
                    cenmag_state.last_gradient_validation = {
                        'groups': summary,
                        'magnitude_ownership': ownership,
                        'local_gradient_norm_min': min(local_grad_norms),
                        'local_gradient_norm_max': max(local_grad_norms),
                        'optimizer_parameter_groups': optimizer_names,
                    }
                    logger.info(
                        'CENMAG_GRADIENT_OWNERSHIP %s',
                        json.dumps(cenmag_state.last_gradient_validation,
                                   sort_keys=True))

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            if getattr(getattr(ctx.cfg, 'offline', None), 'diagnostics', False):
                from federatedscope.glue.cenmag.observability import gradient_summary
                logger.info('CENMAG_GRADIENT_DIAGNOSTIC %s', json.dumps(gradient_summary(ctx.model)))
            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()
        if os.environ.get('CENMAG_TRACE_DIR') and int(ctx.cur_batch_i) == 0 and int(ctx.cur_epoch_i) == 0:
            from federatedscope.glue.cenmag.trace import write_trainer_trace
            write_trainer_trace(ctx, 'after_first_step')

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        regular_loss = ctx.get("loss_regular", None)
        if regular_loss is not None:
            ctx.loss_regular_total += float(
                regular_loss.detach().item()
                if torch.is_tensor(regular_loss) else regular_loss)
        if str(ctx.cfg.federate.method).lower() == 'cenmag_new':
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                if regular_loss is None or ctx.get("loss_task", None) is None:
                    raise RuntimeError(
                        'CenMag training batch is missing its task or '
                        'magnitude-regularization loss')
                ctx.magnitude_regular_sample_total += float(
                    regular_loss.detach().item()) * ctx.batch_size
                ctx.total_loss_sample_total += float(
                    ctx.loss_task.detach().item()) * ctx.batch_size
                cenmag_state = getattr(ctx.model, '_cenmag_state', None)
                if cenmag_state is not None and \
                        'first_training_batch' not in cenmag_state.last_memory:
                    from federatedscope.glue.cenmag.model_utils import \
                        cuda_memory_snapshot
                    cenmag_state.last_memory['first_training_batch'] = \
                        cuda_memory_snapshot('first_training_batch')
            else:
                # The regularizer hook is deliberately absent during
                # evaluation.  Report zero regularization and make the total
                # evaluation loss equal the task loss without creating a
                # phantom trainable magnitude objective in eval mode.
                ctx.total_loss_sample_total += float(
                    ctx.loss_batch.detach().item()) * ctx.batch_size
        # cache label for evaluate, use extend not append
        ctx.ys_true.extend(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.extend(ctx.y_pred.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss
        }
        if str(ctx.cfg.federate.method).lower() == 'cenmag_new':
            denominator = max(1.0, float(ctx.num_samples))
            eval_results.update({
                f'{ctx.cur_split}_task_avg_loss': float(avg_loss),
                f'{ctx.cur_split}_magnitude_regularization_avg_loss': float(
                    ctx.magnitude_regular_sample_total / denominator),
                f'{ctx.cur_split}_total_avg_loss': float(
                    ctx.total_loss_sample_total / denominator),
            })

        from federatedscope.glue.cenmag.offline import enabled
        if enabled(ctx.cfg):
            from federatedscope.glue.cenmag.offline import load_local_metric
            glue_metric = load_local_metric(ctx.cfg)
        else:
            glue_metric = load_metric('glue', ctx.cfg.data.type.split('@')[0], trust_remote_code=True)
        eval_metric = glue_metric.compute(predictions=ctx.ys_pred, references=ctx.ys_true)
        for k, v in eval_metric.items():
            eval_results[f'{ctx.cur_split}_{k}'] = v
        if str(ctx.cfg.federate.method).lower() == 'cenmag_new':
            # Expose class-collapse diagnostics in every client evaluation log.
            # Accuracy alone cannot distinguish a useful model from predicting
            # one majority class on the highly non-IID MNLI shards.
            true_labels = np.asarray(ctx.ys_true, dtype=np.int64).reshape(-1)
            predicted_labels = np.asarray(
                ctx.ys_pred, dtype=np.int64).reshape(-1)
            model_config = getattr(ctx.model, 'config', None)
            if model_config is None and hasattr(ctx.model, 'model'):
                model_config = getattr(ctx.model.model, 'config', None)
            observed_max = max(
                int(true_labels.max()) if true_labels.size else -1,
                int(predicted_labels.max()) if predicted_labels.size else -1)
            num_labels = int(getattr(model_config, 'num_labels',
                                     observed_max + 1))
            if num_labels <= observed_max or num_labels <= 0:
                raise RuntimeError(
                    'CenMag could not determine a valid label count for '
                    'prediction diagnostics')
            true_counts = np.bincount(
                true_labels, minlength=num_labels).astype(np.int64)
            predicted_counts = np.bincount(
                predicted_labels, minlength=num_labels).astype(np.int64)
            for class_id in range(num_labels):
                eval_results[
                    f'{ctx.cur_split}_true_count_class_{class_id}'] = int(
                        true_counts[class_id])
                eval_results[
                    f'{ctx.cur_split}_pred_count_class_{class_id}'] = int(
                        predicted_counts[class_id])
            predicted_total = int(predicted_counts.sum())
            eval_results[f'{ctx.cur_split}_pred_unique_classes'] = int(
                np.count_nonzero(predicted_counts))
            eval_results[f'{ctx.cur_split}_pred_majority_fraction'] = (
                float(predicted_counts.max() / predicted_total)
                if predicted_total else 0.0)

        if getattr(getattr(ctx.cfg, 'offline', None), 'diagnostics', False):
            from federatedscope.glue.cenmag.observability import prediction_summary
            logger.info('CENMAG_PREDICTION_DIAGNOSTIC %s', json.dumps(
                prediction_summary(ctx.ys_pred, ctx.ys_true, ctx.cfg.data.num_labels)))
        setattr(ctx, 'eval_metrics', eval_results)


        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_glue_trainer(trainer_type):
    if trainer_type == 'gluetrainer':
        trainer_builder = GLUETrainer
        return trainer_builder


register_trainer('gluetrainer', call_glue_trainer)
