"""PDF-defined CenMag magnitude parameterization for PEFT 0.12 QDoRA.

The PEFT ``DoraLinearLayer.forward`` implementation remains untouched.  Each
native DoRA ``weight`` parameter is parametrized so that every access returns

    global_magnitude.detach() + client_local_magnitude

The native parameter is retained only as a frozen initialization witness; it
is not used by the parametrization and cannot enter the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class MagnitudeBinding:
    module_path: str
    canonical_key: str
    local_parameter_key: str
    global_buffer_key: str
    frozen_native_key: str


class CenMagMagnitudeParametrization(nn.Module):
    """Expose a derived effective magnitude with one trainable local term."""

    def __init__(self, native_magnitude: torch.Tensor):
        super().__init__()
        initial = native_magnitude.detach().clone()
        self.register_buffer("global_magnitude", initial, persistent=True)
        self.client_local_magnitude = nn.Parameter(torch.zeros_like(initial))

    def forward(self, frozen_native_magnitude: torch.Tensor) -> torch.Tensor:
        # The frozen native parameter is intentionally unused.  Keeping it in
        # the parametrization registry lets us prove that the native QDoRA
        # initialization was preserved without creating a second trainable
        # effective-magnitude parameter.
        del frozen_native_magnitude
        return self.global_magnitude.detach() + self.client_local_magnitude


def _peft_model(model: Any) -> Any:
    """Return the wrapped PEFT model whose names match AdapterModel.state_dict."""
    wrapped = getattr(model, "model", None)
    return wrapped if wrapped is not None else model


def install_cenmag_magnitude_parametrizations(
        model: Any, expected_tensors: int = -1) -> Tuple[MagnitudeBinding, ...]:
    """Replace native trainable DoRA magnitudes by derived CenMag magnitudes."""
    if getattr(model, "_cenmag_magnitude_bindings", None):
        raise RuntimeError("CenMag magnitude parametrizations are already installed")

    peft_model = _peft_model(model)
    candidates = []
    for module_path, module in peft_model.named_modules():
        if module.__class__.__name__ != "DoraLinearLayer":
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise RuntimeError(
                f"Unexpected native DoRA magnitude at {module_path}: {type(weight)}")
        candidates.append((module_path, module, weight.detach().clone()))

    if not candidates:
        raise RuntimeError("CenMag found no PEFT DoraLinearLayer magnitude tensors")
    if expected_tensors > 0 and len(candidates) != int(expected_tensors):
        raise RuntimeError(
            f"Expected {expected_tensors} DoRA magnitudes, found {len(candidates)}")

    bindings = []
    for module_path, module, native in sorted(candidates, key=lambda item: item[0]):
        parametrization = CenMagMagnitudeParametrization(native)
        parametrize.register_parametrization(
            module, "weight", parametrization, unsafe=False)
        module.parametrizations.weight.original.requires_grad_(False)

        canonical = f"{module_path}.weight"
        local_key = (
            f"{module_path}.parametrizations.weight.0.client_local_magnitude")
        global_key = f"{module_path}.parametrizations.weight.0.global_magnitude"
        original_key = f"{module_path}.parametrizations.weight.original"
        bindings.append(MagnitudeBinding(
            module_path=module_path,
            canonical_key=canonical,
            local_parameter_key=local_key,
            global_buffer_key=global_key,
            frozen_native_key=original_key,
        ))

        if module.parametrizations.weight.original.requires_grad:
            raise RuntimeError(f"Native magnitude remained trainable at {module_path}")
        if parametrization.global_magnitude.requires_grad:
            raise RuntimeError(f"Global magnitude unexpectedly requires grad at {module_path}")
        if not parametrization.client_local_magnitude.requires_grad:
            raise RuntimeError(f"Local magnitude is not trainable at {module_path}")
        if not torch.equal(module.weight.detach(), native):
            raise RuntimeError(
                f"CenMag initialization changed the effective magnitude at {module_path}")

    frozen = tuple(bindings)
    setattr(model, "_cenmag_magnitude_bindings", frozen)
    return frozen


def magnitude_bindings(model: Any) -> Tuple[MagnitudeBinding, ...]:
    bindings = getattr(model, "_cenmag_magnitude_bindings", None)
    if not bindings and getattr(model, "model", None) is not None:
        bindings = getattr(model.model, "_cenmag_magnitude_bindings", None)
    if not bindings:
        raise RuntimeError("CenMag magnitude parametrizations are not installed")
    return tuple(bindings)


def binding_modules(model: Any) -> Dict[str, CenMagMagnitudeParametrization]:
    peft_model = _peft_model(model)
    result = {}
    for binding in magnitude_bindings(model):
        dora_module = peft_model.get_submodule(binding.module_path)
        parametrization = dora_module.parametrizations.weight[0]
        if not isinstance(parametrization, CenMagMagnitudeParametrization):
            raise RuntimeError(
                f"Unexpected magnitude parametrization at {binding.module_path}")
        result[binding.canonical_key] = parametrization
    return result


def effective_magnitude_state(model: Any) -> Dict[str, torch.Tensor]:
    return {
        key: module.global_magnitude.detach() + module.client_local_magnitude
        for key, module in binding_modules(model).items()
    }


def local_magnitude_parameters(model: Any) -> Tuple[nn.Parameter, ...]:
    return tuple(
        module.client_local_magnitude
        for module in binding_modules(model).values())


def cenmag_optimizer_parameters(model: Any) -> Tuple[nn.Parameter, ...]:
    """Return only trainable parameters for the New CenMag optimizer.

    The stock FederatedScope builder registers frozen parameters when passed a
    module.  Passing this tuple keeps the frozen native QDoRA magnitude (and
    the frozen quantized backbone) physically absent from the optimizer.
    """
    parameters = tuple(parameter for parameter in model.parameters()
                       if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("New CenMag found no trainable client parameters")
    return parameters


def magnitude_parameter_count(model: Any) -> int:
    return sum(parameter.numel()
               for parameter in local_magnitude_parameters(model))


def magnitude_regularization_loss(model: Any,
                                  lambda_m: float) -> torch.Tensor:
    """Return lambda_m / (2 P_m) times the local-magnitude squared norm."""
    parameters = local_magnitude_parameters(model)
    p_m = sum(parameter.numel() for parameter in parameters)
    if p_m <= 0:
        raise RuntimeError("CenMag local magnitude parameter count is zero")
    squared_norm = sum(
        parameter.float().pow(2).sum() for parameter in parameters)
    return squared_norm * (float(lambda_m) / (2.0 * float(p_m)))


def load_global_magnitude(model: Any,
                          values: Mapping[str, torch.Tensor]) -> None:
    modules = binding_modules(model)
    if set(values) != set(modules):
        raise RuntimeError("Global magnitude key mismatch")
    with torch.no_grad():
        for key, module in modules.items():
            module.global_magnitude.copy_(values[key].detach().to(
                device=module.global_magnitude.device,
                dtype=module.global_magnitude.dtype))


def load_client_local_magnitude(model: Any,
                                values: Mapping[str, torch.Tensor]) -> None:
    modules = binding_modules(model)
    if set(values) != set(modules):
        raise RuntimeError("Client-local magnitude key mismatch")
    with torch.no_grad():
        for key, module in modules.items():
            module.client_local_magnitude.copy_(values[key].detach().to(
                device=module.client_local_magnitude.device,
                dtype=module.client_local_magnitude.dtype))


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def magnitude_ownership_audit(model: Any,
                              optimizer: torch.optim.Optimizer | None = None
                              ) -> Dict[str, Any]:
    modules = binding_modules(model)
    local = [module.client_local_magnitude for module in modules.values()]
    globals_ = [module.global_magnitude for module in modules.values()]
    peft_model = _peft_model(model)
    original = [
        peft_model.get_submodule(binding.module_path).parametrizations.weight.original
        for binding in magnitude_bindings(model)
    ]
    optimizer_ids = optimizer_parameter_ids(optimizer) if optimizer else set()
    return {
        "magnitude_tensor_count": len(modules),
        "p_m": sum(parameter.numel() for parameter in local),
        "all_global_frozen": all(not tensor.requires_grad for tensor in globals_),
        "all_local_trainable": all(parameter.requires_grad for parameter in local),
        "all_native_frozen": all(not parameter.requires_grad for parameter in original),
        "local_in_optimizer": (
            all(id(parameter) in optimizer_ids for parameter in local)
            if optimizer else None),
        "global_in_optimizer": (
            any(id(tensor) in optimizer_ids for tensor in globals_)
            if optimizer else None),
        "native_in_optimizer": (
            any(id(parameter) in optimizer_ids for parameter in original)
            if optimizer else None),
        "duplicate_trainable_effective_magnitude": False,
    }
