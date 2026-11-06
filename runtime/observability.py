"""Non-mutating observation of canonical forward/backward and completed rounds."""
from pathlib import Path
import copy
import math
import time
import numpy as np
import torch
from platform_tools.common import append, atomic
from deep_tools import Instrumentation, group, norm
from boundary import original
from federatedscope.glue.cenmag.magnitude import binding_modules, effective_magnitude_state


def summarize(logits, labels, sample_ids, client, round_index, loss):
    logits = logits.float().cpu()
    labels = labels.long().cpu()
    finite = torch.isfinite(logits).all(-1)
    # Invalid output rows are counted, never silently reinterpreted as valid logits.
    good_logits, good_labels = logits[finite], labels[finite]
    predictions = good_logits.argmax(-1)
    probabilities = good_logits.softmax(-1)
    cm = torch.bincount(good_labels * 3 + predictions, minlength=9).reshape(3, 3)
    support = torch.bincount(labels, minlength=3)
    counts = cm.sum(0)
    n = len(labels)
    valid_n = int(finite.sum())
    return dict(client=int(client), round_index=int(round_index), completed_round=int(round_index) + 1,
                evaluation_id=f'completed_{int(round_index)+1:06d}_client_{client}',
                examples=n, valid_examples=valid_n, correct=int(cm.diag().sum()),
                accuracy=int(cm.diag().sum()) / n, loss=loss,
                true_class_counts=support.tolist(), predicted_class_counts=counts.tolist(),
                confusion_matrix=cm.tolist(),
                per_class_recall=[float(cm[k, k] / support[k]) if support[k] else None for k in range(3)],
                prediction_entropy=float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)).mean()) if valid_n else 0.,
                entropy_definition='mean Shannon entropy of softmax logits in nats over finite rows',
                dominant_prediction_fraction=float(counts.max()) / valid_n if valid_n else 0.,
                nonfinite_logits=n - valid_n, invalid_output_fraction=(n - valid_n) / n,
                sample_ids_sha256=__import__('hashlib').sha256(np.asarray(sample_ids, dtype='<i8').tobytes()).hexdigest())


class Observer:
    def __init__(self, out, runner, operations, mode='light'):
        self.out, self.runner, self.operations, self.mode = Path(out), runner, operations, mode
        self.steps, self.evaluations, self.aggregations = [], [], []
        self.deep = Instrumentation(out, runner, False)
        self.overhead_seconds = 0.
        (self.out / 'logits').mkdir()
        self.fit_times = {}
        self.fit_summaries = {}

    def forensic_round(self, index):
        return self.mode == 'forensic' or (self.mode != 'off' and (int(index) + 1) % self.operations['forensic_every'] == 0)

    def attach(self):
        for cid, client in self.runner.client.items():
            trainer = client.trainer
            forward, backward = trainer._hook_on_batch_forward, trainer._hook_on_batch_backward

            def fit_start(ctx, client=client):
                client._research_fit_start = time.perf_counter()
                self.deep.enabled = self.forensic_round(client.state)
                if self.deep.enabled:
                    self.deep.probe(client, 'before_local', client.state)

            def fit_end(ctx, client=client):
                self.fit_times[client.ID] = time.perf_counter() - client._research_fit_start
                self.fit_summaries[client.ID] = dict(metrics=copy.deepcopy(ctx.eval_metrics),
                    actual_processed_examples=int(ctx.num_samples), seconds=self.fit_times[client.ID])
                if self.deep.enabled:
                    state = client.cenmag_state
                    self.deep.local_a[client.ID] = state.snapshot(ctx.model, state.keys.lora_a, fp32=False)
                    self.deep.local_effective[client.ID] = {k: v.detach().cpu().clone() for k, v in effective_magnitude_state(ctx.model).items()}
                    self.deep.probe(client, 'after_local', client.state)

            def observed_backward(ctx, client=client, backward=backward):
                # Scientific computation is called exactly once, unchanged.
                if self.mode == 'off':
                    return backward(ctx)
                selected = int(ctx.cur_batch_i) == 0 or (int(ctx.cur_batch_i) + 1) % self.operations['gradient_sample_every'] == 0
                if not selected:
                    result = backward(ctx)
                    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(torch.isfinite(p).all())
                           for p in ctx.model.parameters() if p.requires_grad):
                        raise FloatingPointError('Nonfinite or missing gradient/parameter at unsampled local step')
                    return result
                start = time.perf_counter()
                groups = {name: [p for n, p in ctx.model.named_parameters() if p.requires_grad and group(n) == name]
                          for name in ['lora_A', 'lora_B', 'client_local_magnitude', 'classifier']}
                expected = {id(p) for values in groups.values() for p in values}
                actual = {id(p) for g in ctx.optimizer.param_groups for p in g['params']}
                if expected != actual or any(not values for values in groups.values()):
                    raise RuntimeError('Unexpected parameter/optimizer ownership')
                if type(ctx.optimizer).__name__ != 'SGD' or ctx.optimizer.state or ctx.scheduler is not None:
                    raise RuntimeError('Unexpected persistent optimizer or scheduler')
                before = {g: torch.cat([p.detach().flatten() for p in values]) for g, values in groups.items()}
                before_seconds = time.perf_counter() - start
                backward(ctx)
                start = time.perf_counter()
                rows = {}
                for g, values in groups.items():
                    if any(p.grad is None for p in values):
                        raise RuntimeError('Missing expected parameter gradient: ' + g)
                    gradient = torch.cat([p.grad.detach().flatten() for p in values])
                    after = torch.cat([p.detach().flatten() for p in values])
                    rows[g] = dict(gradient_norm=float(gradient.norm()), update_norm=float((after - before[g]).norm()),
                                   parameter_norm_before=float(before[g].norm()),
                                   gradient_finite=bool(torch.isfinite(gradient).all()), parameter_finite=bool(torch.isfinite(after).all()))
                residual = before['client_local_magnitude']
                # Exact derivative of lambda/(2*p_m) * sum(residual^2).
                regularization_gradient = residual * float(ctx.cfg.cenmag.lambda_m) / residual.numel()
                combined_gradient = torch.cat([p.grad.detach().flatten() for p in groups['client_local_magnitude']])
                task_gradient = combined_gradient - regularization_gradient
                rg, tg = float(regularization_gradient.norm()), float(task_gradient.norm())
                row = dict(round_index=int(client.state), completed_round=int(client.state) + 1,
                           client=client.ID, local_step=int(ctx.cur_batch_i), groups=rows,
                           task_loss=float(ctx.loss_batch.detach()), regularization_loss=float(ctx.loss_regular.detach()),
                           total_loss=float(ctx.loss_task.detach()), lr=[float(g['lr']) for g in ctx.optimizer.param_groups],
                           optimizer=dict(type='SGD', momentum=0., weight_decay=0.),
                           regularization_gradient_norm=rg, magnitude_task_gradient_norm=tg,
                           regularization_to_task_gradient_ratio=rg / tg if tg > 0 else None,
                           ratio_scope='private magnitude parameters; analytic regularizer subtracted from total gradient',
                           sample_ids=ctx.data_batch['sample_id'].tolist(), batch_samples=int(ctx.batch_size),
                           center_norm=norm([b.global_magnitude for b in binding_modules(ctx.model).values()]),
                           residual_norm_before=float(residual.norm()))
                row['observer_seconds'] = before_seconds + time.perf_counter() - start
                self.overhead_seconds += row['observer_seconds']
                # Correctness failures preserve earlier coherent generations. JSON forbids NaN.
                finite = all(v['gradient_finite'] and v['parameter_finite'] for v in rows.values())
                if not finite:
                    append(self.out / 'errors.jsonl', dict(kind='nonfinite_gradient_or_parameter', client=client.ID,
                           round_index=int(client.state), phase='local_optimization'))
                    raise FloatingPointError('Nonfinite gradient or parameter; recover preceding coherent state')
                self.steps.append(row)
                append(self.out / 'steps.jsonl', row)

            def eval_start(ctx):
                ctx._research_eval_start = time.perf_counter()
                ctx._research_eval_rows = []

            def eval_forward(ctx, forward=forward):
                captured = []
                handle = ctx.model.register_forward_hook(lambda module, args, output: captured.append(output.logits.detach()))
                try:
                    forward(ctx)
                finally:
                    handle.remove()
                logits = captured[0].float().cpu()
                if logits.shape != (int(ctx.batch_size), 3):
                    raise RuntimeError('Unexpected canonical classifier output shape')
                ctx._research_eval_rows.append((ctx.data_batch['sample_id'].cpu(), ctx.y_true.cpu(), logits))

            def eval_end(ctx, client=client):
                start = time.perf_counter()
                blocks = ctx._research_eval_rows
                ids, labels, logits = [torch.cat([b[i] for b in blocks]) for i in range(3)]
                row = summarize(logits, labels, ids.tolist(), client.ID, client.state,
                                float(ctx.eval_metrics['val_avg_loss']))
                row['loss_definition'] = 'canonical trainer sample-weighted mean task loss at original forward precision'
                row['duration_seconds'] = time.perf_counter() - ctx._research_eval_start
                row['training_examples_available'] = len(self.runner.data[client.ID]['train'].dataset)
                # Compressed numeric evidence only; never raw MNLI text.
                name = f'round_{int(client.state)+1:06d}_client_{client.ID}.npz'
                np.savez_compressed(self.out / 'logits' / name, sample_id=ids.numpy(), labels=labels.numpy(),
                                    logits=logits.numpy(), prediction=logits.argmax(-1).numpy())
                row['logits_file'] = 'logits/' + name
                self.evaluations.append(row)
                if self.deep.enabled:
                    wanted = [int(s['sample_id']) for s in self.runner.data[client.ID]['probe']]
                    lookup = {int(v): i for i, v in enumerate(ids)}
                    subset = torch.tensor([lookup[v] for v in wanted])
                    probe = summarize(logits[subset], labels[subset], wanted, client.ID, client.state, None)
                    probe.update(stage='final_evaluation', round=int(client.state), sample_ids=wanted,
                                 logits=logits[subset].tolist(), prediction=logits[subset].argmax(-1).tolist())
                    self.deep.probes.append(probe)
                self.overhead_seconds += time.perf_counter() - start
                if row['nonfinite_logits']:
                    raise FloatingPointError('Nonfinite evaluation output')

            trainer.register_hook_in_train(fit_start, 'on_fit_start', insert_pos=-1)
            trainer.register_hook_in_train(fit_end, 'on_fit_end', insert_pos=-1)
            trainer.replace_hook_in_train(observed_backward, 'on_batch_backward', '_hook_on_batch_backward')
            trainer.register_hook_in_eval(eval_start, 'on_fit_start', insert_pos=-1)
            trainer.replace_hook_in_eval(eval_forward, 'on_batch_forward', '_hook_on_batch_forward')
            trainer.register_hook_in_eval(eval_end, 'on_fit_end', insert_pos=-1)

    def after_aggregation(self, old_center, old_a, round_index, uploads):
        state = self.runner.server.cenmag_state
        total = sum(v[0] for v in uploads.values())
        a_error = max(float((state.global_a[k] - sum(uploads[i][1][k].cpu() * uploads[i][0] / total for i in uploads)).abs().max()) for k in state.global_a)
        center_error = max(float((state.global_magnitude[k] - sum(uploads[i][1][k].cpu().double() * uploads[i][0] / total for i in uploads)).abs().max()) for k in state.global_magnitude)
        if max(a_error, center_error) >= 1e-6:
            raise RuntimeError('Independent aggregation reference mismatch')
        record = dict(round_index=int(round_index), completed_round=int(round_index)+1,
                      processed_examples={i: int(v[0]) for i, v in uploads.items()}, weights=state.last_aggregation_weights,
                      independent_A_max_abs_error=a_error, independent_center_max_abs_error=center_error,
                      validation=copy.deepcopy(state.last_aggregation_validation))
        self.aggregations.append(record)
        append(self.out / 'aggregation.jsonl', record)
        if self.deep.enabled:
            self.deep.after_aggregation(old_center, old_a, round_index, uploads)

    def flush_round(self):
        for row in self.evaluations:
            append(self.out / 'evaluation.jsonl', row)
        for row in self.deep.probes:
            append(self.out / 'probes.jsonl', row)
        for row in self.deep.aggregations:
            append(self.out / 'stage_aggregation.jsonl', row)
        self.steps.clear()
        self.evaluations.clear()
        self.aggregations.clear()
        self.deep.probes.clear()
        self.deep.aggregations.clear()
        self.deep.local_a.clear()
        self.deep.local_effective.clear()
