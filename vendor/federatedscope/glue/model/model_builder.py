import logging
from importlib.metadata import version

import torch
from federatedscope.glue.model.adapter_builder import AdapterModel


logger = logging.getLogger(__name__)


def _is_cenmag(config):
    return str(config.federate.method).lower() == 'cenmag_new'


def get_model_from_huggingface(model_name, config):
    from transformers import AutoModelForSequenceClassification

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model


    kwargs['num_labels'] = config.data.num_labels

    if _is_cenmag(config):
        from transformers import BitsAndBytesConfig
        from federatedscope.glue.cenmag.hardware import \
            assert_cenmag_cuda, verify_nf4_execution

        if int(config.device) != 0:
            raise RuntimeError(
                'Canonical CenMag single-GPU configuration requires logical device 0')
        hardware = assert_cenmag_cuda(torch, int(config.device))
        logger.info('CenMag CUDA eligibility: %s', hardware)
        expected_bnb = str(config.cenmag.quantization.bitsandbytes_version)
        try:
            actual_bnb = version('bitsandbytes')
        except Exception as error:
            raise RuntimeError(
                'CenMag-FedSA requires the pinned bitsandbytes package') from error
        if actual_bnb != expected_bnb:
            raise RuntimeError(
                f'CenMag-FedSA requires bitsandbytes {expected_bnb}, '
                f'found {actual_bnb}')
        dtype_name = str(config.cenmag.quantization.compute_dtype)
        if dtype_name != 'float16':
            raise RuntimeError(
                f'Unsupported canonical CenMag compute dtype {dtype_name}')
        logger.info('CenMag NF4 kernel gate: %s', verify_nf4_execution(
            int(config.device), expected_version=expected_bnb))
        compute_dtype = torch.float16
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            # RoBERTa's sequence-classification head is newly initialized.
            # Keeping it out of the 4-bit replacement pass prevents
            # Transformers from trying to initialize a uint8 Params4bit
            # tensor with normal_(), while preserving the intended
            # trainable shared/sequential classifier lifecycle.
            llm_int8_skip_modules=['classifier'],
            bnb_4bit_quant_type=str(config.cenmag.quantization.type),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(
                config.cenmag.quantization.double_quantization),
        )
        kwargs['device_map'] = {'': int(config.device)}
        kwargs['torch_dtype'] = compute_dtype
        kwargs['low_cpu_mem_usage'] = True

    if _is_cenmag(config):
        torch.cuda.reset_peak_memory_stats()
    from federatedscope.glue.cenmag.offline import enabled
    if enabled(config):
        model_name = str(config.offline.model_dir)
        kwargs['local_files_only'] = True
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               **kwargs)
    if _is_cenmag(config):
        from federatedscope.glue.cenmag.model_utils import cuda_memory_snapshot
        setattr(model, '_cenmag_base_quantized_memory',
                cuda_memory_snapshot('base_quantized_model'))
    return model


def get_llm(config):
    model_name, model_hub = config.model.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    if _is_cenmag(config):
        from peft import prepare_model_for_kbit_training
        if not getattr(model, 'is_loaded_in_4bit', False):
            raise RuntimeError(
                'CenMag model loader did not produce a 4-bit Hugging Face model')
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(
                config.cenmag.quantization.gradient_checkpointing),
        )

    cenmag_base_memory = getattr(model, '_cenmag_base_quantized_memory', {}) \
        if _is_cenmag(config) else {}
    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    if _is_cenmag(config):
        from federatedscope.glue.cenmag.magnitude import \
            install_cenmag_magnitude_parametrizations
        install_cenmag_magnitude_parametrizations(
            model,
            expected_tensors=int(config.cenmag.expected_magnitude_tensors),
        )
        from federatedscope.glue.cenmag.model_utils import \
            assert_frozen_quantized_backbone, assert_real_nf4_backbone
        from federatedscope.glue.cenmag.state import \
            discover_cenmag_parameter_keys
        summary = assert_frozen_quantized_backbone(model)
        summary['real_nf4'] = assert_real_nf4_backbone(model)
        keys = discover_cenmag_parameter_keys(
            model,
            expected_magnitude_tensors=int(
                config.cenmag.expected_magnitude_tensors),
        )
        summary.update({
            'lora_a_tensors': len(keys.lora_a),
            'lora_b_tensors': len(keys.lora_b),
            'magnitude_tensors': len(keys.magnitude),
            'classifier_tensors': len(keys.classifier),
        })
        from federatedscope.glue.cenmag.model_utils import cuda_memory_snapshot
        setattr(model, '_cenmag_base_quantized_memory',
                cenmag_base_memory)
        setattr(model, '_cenmag_qdora_memory',
                cuda_memory_snapshot('qdora_model'))
        setattr(model, '_cenmag_model_summary', summary)
        logger.info('CenMag QDoRA model summary: %s', summary)

    # for FFA-LoRA & FFA-VeRA
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name or "vera_lambda_d" in name:
                param.requires_grad = False

    # save intial lora parameters, for local traning
    if config.federate.method == "local":
        if config.llm.adapter.args[0].get('adapter_method', '') == "vera":
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'vera' in name}
        else:
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'lora' in name}
        torch.save(initial_lora_params, config.federate.save_to + '.init')
    return model
