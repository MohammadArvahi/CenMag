"""Model-level validation helpers for the opt-in CenMag path."""

from __future__ import annotations

import os
from typing import Any, Dict


def is_4bit_model(model: Any) -> bool:
    """Return True only when the live module tree contains bnb Linear4bit."""
    if model is None:
        return False
    try:
        if any(module.__class__.__name__ == "Linear4bit"
               for module in model.modules()):
            return True
    except Exception:
        return False
    return False


def count_linear4bit_modules(model: Any) -> int:
    return sum(module.__class__.__name__ == "Linear4bit"
               for module in model.modules())


def assert_single_visible_t4(torch_module: Any = None) -> Dict[str, Any]:
    """Require the experiment's single visible CUDA device to be a Tesla T4."""
    if torch_module is None:
        import torch as torch_module

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    failures = []
    if cuda_visible_devices != "0":
        failures.append(
            "CUDA_VISIBLE_DEVICES must be exactly '0' for the one-T4 experiment")
    if not torch_module.cuda.is_available():
        failures.append("CUDA is unavailable")

    device_count = int(torch_module.cuda.device_count())
    if device_count != 1:
        failures.append(
            f"exactly one CUDA device must be visible, found {device_count}")

    name = None
    capability = None
    supported_architectures = []
    if device_count > 0:
        name = str(torch_module.cuda.get_device_name(0))
        capability = tuple(torch_module.cuda.get_device_capability(0))
        normalized_name = " ".join(name.lower().split())
        if normalized_name not in {"tesla t4", "nvidia tesla t4"}:
            failures.append(f"visible CUDA device 0 is not a Tesla T4: {name}")
        if capability != (7, 5):
            failures.append(
                f"Tesla T4 compute capability must be 7.5, found {capability}")
        get_arch_list = getattr(torch_module.cuda, "get_arch_list", None)
        if callable(get_arch_list):
            supported_architectures = list(get_arch_list())
            if supported_architectures and "sm_75" not in supported_architectures:
                failures.append(
                    "the active PyTorch CUDA build does not include sm_75")

    if failures:
        raise RuntimeError("Invalid CenMag T4 runtime: " + "; ".join(failures))
    return {
        "cuda_visible_devices": cuda_visible_devices,
        "visible_device_count": device_count,
        "visible_device_index": 0,
        "gpu_name": name,
        "compute_capability": list(capability),
        "torch_supported_architectures": supported_architectures,
    }


def trainable_dtype_summary(model: Any) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = str(parameter.dtype)
            summary[key] = summary.get(key, 0) + parameter.numel()
    return summary


def cuda_memory_snapshot(stage: str) -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"stage": stage, "status": "NOT_RUN", "reason": "CUDA unavailable"}
        synchronization_error = None
        try:
            torch.cuda.synchronize()
        except Exception as error:
            synchronization_error = repr(error)
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        result = {
            "stage": stage,
            "status": "RECORDED" if synchronization_error is None else "ERROR",
            "allocated_bytes": allocated,
            "allocated_mib": allocated / 1024**2,
            "allocated_gib": allocated / 1024**3,
            "reserved_bytes": reserved,
            "reserved_mib": reserved / 1024**2,
            "reserved_gib": reserved / 1024**3,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "peak_allocated_gib": peak_allocated / 1024**3,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mib": peak_reserved / 1024**2,
            "peak_reserved_gib": peak_reserved / 1024**3,
        }
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                status_lines = handle.readlines()
            rss_line = next(
                line for line in status_lines if line.startswith("VmRSS:"))
            rss_kib = int(rss_line.split()[1])
            result["host_process_rss_bytes"] = rss_kib * 1024
            result["host_process_rss_gib"] = rss_kib / 1024**2
        except Exception:
            result["host_process_rss_bytes"] = None
            result["host_process_rss_gib"] = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                memory_lines = handle.readlines()
            available_line = next(
                line for line in memory_lines
                if line.startswith("MemAvailable:"))
            available_kib = int(available_line.split()[1])
            result["host_available_bytes"] = available_kib * 1024
            result["host_available_gib"] = available_kib / 1024**2
        except Exception:
            result["host_available_bytes"] = None
            result["host_available_gib"] = None
        if synchronization_error is not None:
            result["synchronization_error"] = synchronization_error
        return result
    except Exception as error:
        return {
            "stage": stage,
            "status": "ERROR",
            "error": repr(error),
        }


def assert_frozen_quantized_backbone(model: Any) -> Dict[str, Any]:
    """Validate the required QDoRA ownership without assuming PEFT key depth."""
    linear4bit_count = count_linear4bit_modules(model)
    if linear4bit_count <= 0:
        raise RuntimeError(
            "CenMag requires real bitsandbytes Linear4bit modules; none found")

    quantized_classifier_modules = [
        name for name, module in model.named_modules()
        if "classifier" in name.lower()
        and module.__class__.__name__ == "Linear4bit"
    ]
    if quantized_classifier_modules:
        raise RuntimeError(
            "CenMag classifier must remain full precision and trainable; "
            "found 4-bit classifier modules: "
            f"{quantized_classifier_modules[:20]}")

    allowed_trainable = (
        "lora_A",
        "lora_B",
        "lora_magnitude_vector",
        "classifier",
    )
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(keyword in name for keyword in allowed_trainable)
    ]
    if unexpected:
        raise RuntimeError(
            "CenMag found unexpected trainable backbone parameters: "
            f"{unexpected[:20]}")

    invalid_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and str(parameter.dtype) not in {
            "torch.float16", "torch.float32"
        }
    ]
    if invalid_dtypes:
        raise RuntimeError(
            "CenMag trainable parameters use unsupported canonical dtypes: "
            f"{invalid_dtypes[:20]}")

    return {
        "linear4bit_count": linear4bit_count,
        "classifier_linear4bit_count": 0,
        "trainable_dtype_numel": trainable_dtype_summary(model),
    }


def assert_real_nf4_backbone(model: Any) -> Dict[str, Any]:
    """Reject CPU/full-precision fallback in the real bitsandbytes model."""
    modules = [
        module for module in model.modules()
        if module.__class__.__name__ == "Linear4bit"
    ]
    if not modules:
        raise RuntimeError("CenMag found no real bitsandbytes Linear4bit modules")

    failures = []
    devices = set()
    quant_types = set()
    for index, module in enumerate(modules):
        weight = getattr(module, "weight", None)
        if weight is None or weight.__class__.__name__ != "Params4bit":
            failures.append(f"module {index} does not own Params4bit weights")
            continue
        devices.add(str(weight.device))
        if weight.device.type != "cuda":
            failures.append(f"module {index} weight is on {weight.device}, not CUDA")
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is None:
            failures.append(f"module {index} has no materialized quantization state")
            continue
        quant_type = str(getattr(quant_state, "quant_type", "")).lower()
        quant_types.add(quant_type)
        if quant_type != "nf4":
            failures.append(f"module {index} quantization type is {quant_type!r}")
        if not bool(getattr(quant_state, "nested", False)):
            failures.append(
                f"module {index} does not use nested/double quantization")

    if failures:
        raise RuntimeError(
            "CenMag real NF4 backbone validation failed: " + "; ".join(
                failures[:20]))
    return {
        "linear4bit_count": len(modules),
        "weight_class": "Params4bit",
        "weight_devices": sorted(devices),
        "quantization_types": sorted(quant_types),
        "double_quantization": True,
        "full_precision_fallback": False,
    }
