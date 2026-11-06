"""Configuration and fail-fast scope checks for CenMag-FedSA."""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


METHOD_NAME = "cenmag_new"


def extend_cenmag_cfg(cfg):
    cfg.cenmag = CN()
    cfg.cenmag.enabled = False
    cfg.cenmag.magnitude_policy = "centered"
    cfg.cenmag.full_participation_only = True
    cfg.cenmag.centering_tolerance = 1e-6
    cfg.cenmag.validation_mode = True
    cfg.cenmag.expected_magnitude_tensors = -1
    cfg.cenmag.checkpoint_version = 6
    cfg.cenmag.deterministic_round_seeding = True
    cfg.cenmag.lambda_m = 1.0

    cfg.cenmag.quantization = CN()
    cfg.cenmag.quantization.enabled = False
    cfg.cenmag.quantization.bits = 4
    cfg.cenmag.quantization.type = "nf4"
    cfg.cenmag.quantization.compute_dtype = "float16"
    cfg.cenmag.quantization.double_quantization = True
    cfg.cenmag.quantization.gradient_checkpointing = True
    cfg.cenmag.quantization.bitsandbytes_version = "0.50.0"

    cfg.register_cfg_check_fun(assert_cenmag_cfg)


def assert_cenmag_cfg(cfg):
    if str(cfg.federate.method).lower() != METHOD_NAME:
        return

    adapter_args = dict(cfg.llm.adapter.args[0])
    failures = []
    if not cfg.cenmag.enabled:
        failures.append("cenmag.enabled must be true")
    if int(cfg.cenmag.checkpoint_version) != 6:
        failures.append("cenmag.checkpoint_version must be 6")
    if not cfg.cenmag.deterministic_round_seeding:
        failures.append(
            "deterministic_round_seeding must remain enabled for exact resume")
    if float(cfg.cenmag.lambda_m) < 0.0:
        failures.append("cenmag.lambda_m must be non-negative")
    if cfg.cenmag.magnitude_policy != "centered":
        failures.append("magnitude_policy must be centered")
    if not cfg.cenmag.full_participation_only:
        failures.append("full_participation_only must remain true")
    if int(cfg.federate.client_num) != 3:
        failures.append("the first CenMag implementation requires 3 clients")
    if int(cfg.federate.sample_client_num) not in [-1, 3]:
        failures.append("partial participation is unsupported")
    if float(cfg.federate.sample_client_rate) not in [-1.0, 1.0]:
        failures.append("partial participation by sample rate is unsupported")
    if cfg.federate.mode != "standalone":
        failures.append("standalone mode is required")
    if not cfg.federate.share_local_model:
        failures.append("share_local_model must be true")
    if cfg.federate.online_aggr or cfg.asyn.use:
        failures.append("only synchronous offline aggregation is supported")
    if cfg.federate.ignore_weight:
        failures.append(
            "federate.ignore_weight must be false so A and magnitude use "
            "the same sample-size weights")
    if int(cfg.federate.process_num) != 1:
        failures.append("single-process execution is required")
    if len(cfg.train.data_para_dids) != 0:
        failures.append("DataParallel is unsupported; data_para_dids must be empty")
    if cfg.llm.deepspeed.use:
        failures.append("DeepSpeed/distributed execution is unsupported")
    if int(cfg.model.model_num_per_trainer) != 1:
        failures.append("one model per trainer is required")
    if float(cfg.federate.unseen_clients_rate) != 0.0:
        failures.append("unseen clients are unsupported")
    if cfg.federate.use_ss:
        failures.append("secret sharing is unsupported")
    if cfg.quantization.method != "none":
        failures.append("FederatedScope message quantization is unsupported")
    if not cfg.cenmag.quantization.enabled or int(
            cfg.cenmag.quantization.bits) != 4:
        failures.append("real 4-bit backbone quantization is required")
    if cfg.cenmag.quantization.compute_dtype != "float16":
        failures.append("the one-T4 path requires float16 compute")
    if cfg.cenmag.quantization.type != "nf4":
        failures.append("the one-T4 path requires NF4 quantization")
    if not cfg.cenmag.quantization.double_quantization:
        failures.append(
            "the pinned initial path requires tested double quantization")
    if not cfg.cenmag.quantization.gradient_checkpointing:
        failures.append(
            "gradient checkpointing must remain enabled for logical "
            "batch size 128 on one T4")
    if not cfg.llm.adapter.use:
        failures.append("PEFT adapter use must be enabled")
    if adapter_args.get("adapter_package") != "peft":
        failures.append("adapter_package must be peft")
    if adapter_args.get("adapter_method") != "lora":
        failures.append("adapter_method must be lora")
    if not adapter_args.get("use_dora", False):
        failures.append("PEFT use_dora must be true")
    if sorted(adapter_args.get("target_modules", [])) != ["query", "value"]:
        failures.append("target_modules must remain query/value")
    local_parameters = set(cfg.personalization.local_param)
    if not {"lora_B", "classifier"}.issubset(local_parameters):
        failures.append("lora_B and classifier must retain baseline filtering")
    if "client_local_magnitude" not in local_parameters:
        failures.append(
            "client_local_magnitude must be filtered from communication")
    if cfg.train.is_enable_half:
        failures.append(
            "whole-model half mode must be disabled for the quantized CenMag path")
    if failures:
        raise ValueError("Invalid CenMag configuration: " + "; ".join(failures))


register_config("cenmag", extend_cenmag_cfg)
