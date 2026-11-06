"""Server and client state for the PDF-defined CenMag algorithm."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch

from federatedscope.glue.cenmag.magnitude import (
    binding_modules, effective_magnitude_state, load_client_local_magnitude,
    load_global_magnitude, magnitude_bindings)


TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class CenMagParameterKeys:
    lora_a: Tuple[str, ...]
    lora_b: Tuple[str, ...]
    magnitude: Tuple[str, ...]
    local_magnitude: Tuple[str, ...]
    global_magnitude_buffers: Tuple[str, ...]
    frozen_native_magnitude: Tuple[str, ...]
    classifier: Tuple[str, ...]

    @property
    def communicated(self) -> Tuple[str, ...]:
        return self.lora_a + self.magnitude


def _sorted_matching(keys: Iterable[str], marker: str) -> Tuple[str, ...]:
    return tuple(sorted(key for key in keys if marker in key))


def discover_cenmag_parameter_keys(
        model: Any, expected_magnitude_tensors: int = -1) -> CenMagParameterKeys:
    """Discover and validate PEFT A/B/DoRA/classifier state names at runtime."""
    state = model.state_dict()
    state_keys = tuple(state.keys())
    named_parameters = dict(model.named_parameters())
    named_parameter_keys = set(named_parameters)

    lora_a = _sorted_matching(state_keys, "lora_A")
    lora_b = _sorted_matching(state_keys, "lora_B")
    bindings = magnitude_bindings(model)
    magnitude = tuple(binding.canonical_key for binding in bindings)
    local_magnitude = tuple(binding.local_parameter_key for binding in bindings)
    global_magnitude_buffers = tuple(
        binding.global_buffer_key for binding in bindings)
    frozen_native_magnitude = tuple(
        binding.frozen_native_key for binding in bindings)
    classifier = _sorted_matching(state_keys, "classifier")

    if not lora_a:
        raise RuntimeError("CenMag discovered no lora_A tensors")
    if not lora_b:
        raise RuntimeError("CenMag discovered no lora_B tensors")
    if not magnitude:
        raise RuntimeError(
            "CenMag discovered no PEFT DoRA lora_magnitude_vector tensors")
    if not classifier:
        raise RuntimeError("CenMag discovered no trainable classifier tensors")
    if len(lora_a) != len(lora_b) or len(lora_b) != len(magnitude):
        raise RuntimeError(
            "Unexpected PEFT tensor counts: "
            f"A={len(lora_a)}, B={len(lora_b)}, magnitude={len(magnitude)}")
    if expected_magnitude_tensors > 0 and len(magnitude) != expected_magnitude_tensors:
        raise RuntimeError(
            f"Expected {expected_magnitude_tensors} DoRA magnitude tensors, "
            f"found {len(magnitude)}")

    discovered = set(lora_a + lora_b + local_magnitude + classifier)
    named_parameter_matches = {
        key: [
            name for name in named_parameter_keys
            if name == key or name.endswith("." + key)
        ]
        for key in discovered
    }
    missing_named_parameters = sorted(
        key for key, matches in named_parameter_matches.items() if not matches)
    ambiguous_named_parameters = {
        key: matches
        for key, matches in named_parameter_matches.items() if len(matches) > 1
    }
    if missing_named_parameters or ambiguous_named_parameters:
        raise RuntimeError(
            "CenMag state keys do not map one-to-one to live named parameters: "
            f"missing={missing_named_parameters}, "
            f"ambiguous={ambiguous_named_parameters}")
    non_trainable = sorted(
        key for key, matches in named_parameter_matches.items()
        if not named_parameters[matches[0]].requires_grad)
    if non_trainable:
        raise RuntimeError(
            f"CenMag discovered required non-trainable tensors: {non_trainable}")

    effective = effective_magnitude_state(model)
    # PEFT 0.12 DoraLinearLayer stores one output magnitude per B matrix.
    for magnitude_key in magnitude:
        b_key = magnitude_key.replace("lora_magnitude_vector", "lora_B")
        if b_key not in state:
            raise RuntimeError(
                f"No corresponding lora_B tensor for {magnitude_key}")
        magnitude_tensor = effective[magnitude_key]
        b_tensor = state[b_key]
        if magnitude_tensor.ndim != 1 or b_tensor.ndim != 2:
            raise RuntimeError(
                f"Unexpected DoRA/B ranks for {magnitude_key}: "
                f"magnitude={tuple(magnitude_tensor.shape)}, "
                f"B={tuple(b_tensor.shape)}")
        if magnitude_tensor.shape[0] != b_tensor.shape[0]:
            raise RuntimeError(
                f"PEFT output orientation mismatch for {magnitude_key}: "
                f"magnitude={tuple(magnitude_tensor.shape)}, "
                f"B={tuple(b_tensor.shape)}")

    modules = binding_modules(model)
    if any(module.global_magnitude.requires_grad for module in modules.values()):
        raise RuntimeError("CenMag global magnitude must be frozen")
    if any(not module.client_local_magnitude.requires_grad
           for module in modules.values()):
        raise RuntimeError("CenMag client-local magnitude must be trainable")

    return CenMagParameterKeys(
        lora_a=lora_a,
        lora_b=lora_b,
        magnitude=magnitude,
        local_magnitude=local_magnitude,
        global_magnitude_buffers=global_magnitude_buffers,
        frozen_native_magnitude=frozen_native_magnitude,
        classifier=classifier,
    )


def _cpu_fp32_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _clone_tensor_dict(tensors: Mapping[str, torch.Tensor]) -> TensorDict:
    return {key: value.detach().cpu().clone() for key, value in tensors.items()}


def tensor_dict_nbytes(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size()
               for value in tensors.values())


def derive_client_round_seed(base_seed: int, server_round: int,
                             client_id: int) -> int:
    """Return a stable per-round/client seed independent of process history."""
    material = (
        f"cenmag-fedsa-v1|{int(base_seed)}|{int(server_round)}|{int(client_id)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


class CenMagState:
    """Server-owned CPU snapshots independent from the live shared model."""

    VERSION = 5

    def __init__(self,
                 model: Any,
                 client_ids: Sequence[int],
                 tolerance: float = 1e-6,
                 validation_mode: bool = True,
                 expected_magnitude_tensors: int = -1):
        self.client_ids = tuple(sorted(int(client_id)
                                       for client_id in client_ids))
        if not self.client_ids:
            raise ValueError("CenMag requires at least one client")
        self.tolerance = float(tolerance)
        self.validation_mode = bool(validation_mode)
        self.keys = discover_cenmag_parameter_keys(
            model, expected_magnitude_tensors=expected_magnitude_tensors)
        initial_state = model.state_dict()
        self._state_shapes = {
            key: tuple(value.shape) for key, value in initial_state.items()
        }
        initial_effective = effective_magnitude_state(model)
        self._state_shapes.update({
            key: tuple(value.shape) for key, value in initial_effective.items()
        })
        self.global_a = {
            key: initial_state[key].detach().cpu().clone()
            for key in self.keys.lora_a
        }
        self.global_magnitude = {
            key: _cpu_fp32_clone(initial_effective[key])
            for key in self.keys.magnitude
        }
        self.client_local_magnitudes = {
            client_id: {
                key: torch.zeros_like(value)
                for key, value in self.global_magnitude.items()
            }
            for client_id in self.client_ids
        }
        local_keys = self.keys.lora_b + self.keys.classifier
        self.client_local_states = {
            client_id: {
                key: initial_state[key].detach().cpu().clone()
                for key in local_keys
            }
            for client_id in self.client_ids
        }
        self.client_local_rounds = {
            client_id: 0 for client_id in self.client_ids
        }
        self.resume_client_local_states = {
            client_id: _clone_tensor_dict(tensors)
            for client_id, tensors in self.client_local_states.items()
        }
        self.resume_client_local_rounds = dict(self.client_local_rounds)
        self.resume_local_state = {
            key: initial_state[key].detach().cpu().clone()
            for key in local_keys
        }
        self.resume_round = 0
        self.last_centering = {
            "max_abs_weighted_residual_mean": 0.0,
            "mean_abs_weighted_residual_mean": 0.0,
        }
        self.last_aggregation_weights: Dict[int, float] = {}
        self.last_aggregation_validation: Dict[str, Any] = {}
        self.last_communication: Dict[str, Any] = {}
        self.last_gradient_validation: Dict[str, Any] = {}
        self.last_memory: Dict[str, Any] = {
            "base_quantized_model": copy.deepcopy(
                getattr(model, "_cenmag_base_quantized_memory", {})),
            "qdora_model": copy.deepcopy(
                getattr(model, "_cenmag_qdora_memory", {})),
            "clients": {},
            "round_history": [],
        }
        self.last_client_start: Dict[int, TensorDict] = {}
        self.last_evaluation_start: Dict[int, TensorDict] = {}
        self.last_client_local_start: Dict[int, TensorDict] = {}
        self.last_client_local_end: Dict[int, TensorDict] = {}
        self.last_evaluation_local_start: Dict[int, TensorDict] = {}
        self.last_rng_schedule: Dict[str, Any] = {}
        self.assert_local_magnitude_isolation()
        self.assert_client_local_isolation()
        self.assert_detached_from_live_model(model)
        self.assert_initialization()

    def _validate_tensor_dict(self, tensors: Mapping[str, torch.Tensor],
                              expected_keys: Sequence[str], label: str) -> None:
        if set(tensors) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(tensors))
            extra = sorted(set(tensors) - set(expected_keys))
            raise RuntimeError(
                f"{label} key mismatch; missing={missing}, extra={extra}")
        for key in expected_keys:
            if key not in self._state_shapes:
                raise RuntimeError(f"{label} contains unknown state key {key}")
            if tuple(tensors[key].shape) != self._state_shapes[key]:
                raise RuntimeError(
                    f"{label} shape mismatch for {key}: "
                    f"{tuple(tensors[key].shape)} versus "
                    f"{self._state_shapes[key]}")

    def assert_initialization(self) -> None:
        for client_id in self.client_ids:
            effective = self.effective_magnitude(client_id)
            for key in self.keys.magnitude:
                if not torch.equal(effective[key], self.global_magnitude[key]):
                    raise RuntimeError(
                        f"Client {client_id} initial magnitude is not global magnitude")

    def assert_local_magnitude_isolation(self) -> None:
        seen_pointers = set()
        global_pointers = {
            value.untyped_storage().data_ptr()
            for value in self.global_magnitude.values()
        }
        for client_id in self.client_ids:
            for key, value in self.client_local_magnitudes[client_id].items():
                if value.device.type != "cpu" or value.requires_grad:
                    raise RuntimeError(
                        f"Client-local magnitude is not detached CPU state for client "
                        f"{client_id}, key {key}")
                pointer = value.untyped_storage().data_ptr()
                if pointer in global_pointers or pointer in seen_pointers:
                    raise RuntimeError(
                        f"Client-local magnitude storage is aliased for client "
                        f"{client_id}, key {key}")
                seen_pointers.add(pointer)

    def assert_client_local_isolation(self) -> None:
        """Require one detached, non-aliased B/head snapshot per client."""
        expected_clients = set(self.client_ids)
        local_keys = self.keys.lora_b + self.keys.classifier
        seen_pointers = set()
        for label, states, rounds in [
                ("client-local", self.client_local_states,
                 self.client_local_rounds),
                ("resume client-local", self.resume_client_local_states,
                 self.resume_client_local_rounds)]:
            if set(states) != expected_clients:
                raise RuntimeError(
                    f"CenMag {label} state IDs do not match")
            if set(rounds) != expected_clients:
                raise RuntimeError(
                    f"CenMag {label} round IDs do not match")
            for client_id in self.client_ids:
                tensors = states[client_id]
                self._validate_tensor_dict(
                    tensors, local_keys, f"{label} state {client_id}")
                if int(rounds[client_id]) < 0:
                    raise RuntimeError(
                        f"Negative {label} round for client {client_id}")
                for key, value in tensors.items():
                    if value.device.type != "cpu" or value.requires_grad:
                        raise RuntimeError(
                            f"{label} state is not detached CPU state for "
                            f"client {client_id}, key {key}")
                    pointer = value.untyped_storage().data_ptr()
                    if pointer in seen_pointers:
                        raise RuntimeError(
                            f"{label} storage is aliased for client "
                            f"{client_id}, key {key}")
                    seen_pointers.add(pointer)

    def assert_detached_from_live_model(self, model: Any) -> None:
        live = model.state_dict()
        live_effective = effective_magnitude_state(model)
        persistent_groups = {
            "global A": self.global_a,
            "global magnitude": self.global_magnitude,
            "resume local state": self.resume_local_state,
        }
        for client_id in self.client_ids:
            persistent_groups[f"client-local state {client_id}"] = \
                self.client_local_states[client_id]
            persistent_groups[f"resume client-local state {client_id}"] = \
                self.resume_client_local_states[client_id]
            persistent_groups[f"client-local magnitude {client_id}"] = \
                self.client_local_magnitudes[client_id]
        for label, tensors in persistent_groups.items():
            for key, persistent in tensors.items():
                if persistent.device.type != "cpu" or persistent.requires_grad:
                    raise RuntimeError(
                        f"{label} is not detached CPU state for {key}")
                live_tensor = live_effective[key] if key in live_effective \
                    else live[key]
                if live_tensor.device.type == "cpu" and \
                        live_tensor.untyped_storage().data_ptr() == \
                        persistent.untyped_storage().data_ptr():
                    raise RuntimeError(
                        f"{label} aliases the live model tensor for {key}")
        for key, persistent in self.global_magnitude.items():
            if persistent.dtype != torch.float32:
                raise RuntimeError(
                    f"Global magnitude is not FP32 persistent state for {key}")
        live_cpu_pointers = {
            value.untyped_storage().data_ptr()
            for value in live_effective.values() if value.device.type == "cpu"
        }
        for client_id in self.client_ids:
            for key, local_magnitude in self.client_local_magnitudes[
                    client_id].items():
                if local_magnitude.untyped_storage().data_ptr() in live_cpu_pointers:
                    raise RuntimeError(
                        f"Client {client_id} local magnitude aliases live PEFT state "
                        f"for {key}")

        live_local_pointers = {
            live[key].untyped_storage().data_ptr()
            for key in self.keys.lora_b + self.keys.classifier
            if live[key].device.type == "cpu"
        }
        for client_id in self.client_ids:
            for label, states in [
                    ("local", self.client_local_states),
                    ("resume local", self.resume_client_local_states)]:
                for key, local_value in states[client_id].items():
                    if local_value.untyped_storage().data_ptr() in \
                            live_local_pointers:
                        raise RuntimeError(
                            f"Client {client_id} {label} state aliases live "
                            f"model state for {key}")

    def effective_magnitude(self, client_id: int) -> TensorDict:
        if client_id not in self.client_local_magnitudes:
            raise KeyError(f"Unknown CenMag client {client_id}")
        return {
            key: self.global_magnitude[key] +
            self.client_local_magnitudes[client_id][key]
            for key in self.keys.magnitude
        }

    def snapshot(self, model: Any, keys: Sequence[str], fp32: bool = False) -> TensorDict:
        state = model.state_dict()
        effective = effective_magnitude_state(model)
        available = set(state) | set(effective)
        missing = sorted(set(keys) - available)
        if missing:
            raise RuntimeError(f"Live model is missing state keys {missing}")
        selected = {
            key: effective[key] if key in effective else state[key]
            for key in keys
        }
        if fp32:
            return {key: _cpu_fp32_clone(selected[key]) for key in keys}
        return {
            key: selected[key].detach().cpu().clone() for key in keys
        }

    def _load(self, model: Any, tensors: Mapping[str, torch.Tensor]) -> None:
        current = model.state_dict()
        prepared = {}
        for key, value in tensors.items():
            if key not in current:
                raise RuntimeError(f"Cannot load missing live model key {key}")
            if tuple(value.shape) != tuple(current[key].shape):
                raise RuntimeError(
                    f"Cannot load shape {tuple(value.shape)} into {key} "
                    f"with shape {tuple(current[key].shape)}")
            prepared[key] = value.detach().to(device=current[key].device,
                                              dtype=current[key].dtype)
        model.load_state_dict(prepared, strict=False)

    def load_global_magnitude(self, model: Any) -> None:
        load_global_magnitude(model, self.global_magnitude)

    def load_neutral_local_magnitude(self, model: Any) -> None:
        load_client_local_magnitude(model, {
            key: torch.zeros_like(value)
            for key, value in self.global_magnitude.items()
        })

    def load_global_a(self, model: Any) -> None:
        self._load(model, self.global_a)

    def capture_global_a(self, model: Any) -> None:
        self.global_a = self.snapshot(model, self.keys.lora_a, fp32=False)
        self.assert_detached_from_live_model(model)

    def capture_resume_local_state(self, model: Any,
                                   server_round: int) -> None:
        if int(server_round) < 0:
            raise ValueError("CenMag resume round must be non-negative")
        server_round = int(server_round)
        if any(int(self.client_local_rounds[client_id]) != server_round
               for client_id in self.client_ids):
            raise RuntimeError(
                "CenMag cannot freeze client-local states from different "
                f"round boundaries: expected {server_round}, got "
                f"{self.client_local_rounds}")
        self.resume_client_local_states = {
            client_id: _clone_tensor_dict(tensors)
            for client_id, tensors in self.client_local_states.items()
        }
        self.resume_client_local_rounds = dict(self.client_local_rounds)
        self.resume_local_state = self.snapshot(
            model, self.keys.lora_b + self.keys.classifier, fp32=False)
        self.resume_round = server_round
        self.assert_client_local_isolation()
        self.assert_detached_from_live_model(model)

    def load_resume_local_state(self, model: Any) -> None:
        self._load(model, self.resume_local_state)

    def load_client_local_state(self, model: Any, client_id: int,
                                evaluation: bool = False) -> TensorDict:
        if client_id not in self.client_local_states:
            raise KeyError(f"Unknown CenMag client {client_id}")
        expected = self.client_local_states[client_id]
        self._load(model, expected)
        loaded = self.snapshot(
            model, self.keys.lora_b + self.keys.classifier, fp32=False)
        current = model.state_dict()
        for key in expected:
            representable = expected[key].to(dtype=current[key].dtype)
            if not torch.equal(loaded[key], representable):
                raise RuntimeError(
                    f"Client {client_id} local state was not loaded for {key}")
        target = self.last_evaluation_local_start if evaluation else \
            self.last_client_local_start
        target[client_id] = _clone_tensor_dict(loaded)
        return _clone_tensor_dict(expected)

    def capture_client_local_state(self, model: Any, client_id: int,
                                   next_server_round: int) -> None:
        if client_id not in self.client_local_states:
            raise KeyError(f"Unknown CenMag client {client_id}")
        next_server_round = int(next_server_round)
        if next_server_round < 0:
            raise ValueError("CenMag client-local round must be non-negative")
        captured = self.snapshot(
            model, self.keys.lora_b + self.keys.classifier, fp32=False)
        captured_local_magnitude = {
            key: _cpu_fp32_clone(module.client_local_magnitude)
            for key, module in binding_modules(model).items()
        }
        self.client_local_states[client_id] = _clone_tensor_dict(captured)
        self.client_local_magnitudes[client_id] = _clone_tensor_dict(
            captured_local_magnitude)
        self.client_local_rounds[client_id] = next_server_round
        self.last_client_local_end[client_id] = _clone_tensor_dict(captured)
        self.assert_client_local_isolation()
        self.assert_local_magnitude_isolation()
        self.assert_detached_from_live_model(model)

    def load_effective_magnitude(self, model: Any, client_id: int) -> TensorDict:
        effective = self.effective_magnitude(client_id)
        load_global_magnitude(model, self.global_magnitude)
        load_client_local_magnitude(
            model, self.client_local_magnitudes[client_id])
        self.last_client_start[client_id] = self.snapshot(
            model, self.keys.magnitude, fp32=True)
        return effective

    def assert_loaded(self, model: Any, expected: Mapping[str, torch.Tensor],
                      label: str) -> None:
        current_effective = effective_magnitude_state(model)
        live = self.snapshot(model, self.keys.magnitude, fp32=True)
        for key in self.keys.magnitude:
            representable_expected = expected[key].to(
                dtype=current_effective[key].dtype).float().cpu()
            if not torch.allclose(live[key], representable_expected, rtol=0.0,
                                  atol=self.tolerance):
                difference = (live[key] -
                              representable_expected).abs().max().item()
                raise RuntimeError(
                    f"{label} magnitude was not loaded for {key}; max diff={difference}")

    def validate_communicated_state(self, state: Mapping[str, torch.Tensor],
                                    label: str) -> None:
        expected = set(self.keys.communicated)
        actual = set(state)
        if actual != expected:
            raise RuntimeError(
                f"{label} must contain exactly A+magnitude; "
                f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    def assert_broadcast_global_magnitude(
            self, state: Mapping[str, torch.Tensor]) -> None:
        self.validate_communicated_state(state, "CenMag server broadcast")
        for key in self.keys.magnitude:
            incoming = _cpu_fp32_clone(state[key])
            representable_global = self.global_magnitude[key].to(
                dtype=state[key].dtype).float().cpu()
            if not torch.allclose(incoming,
                                  representable_global,
                                  rtol=0.0,
                                  atol=self.tolerance):
                difference = (incoming - representable_global).abs().max().item()
                raise RuntimeError(
                    f"Broadcast magnitude is not explicit global state for {key}; "
                    f"max diff={difference}")

    def update_from_client_magnitudes(
            self, client_magnitudes: Mapping[int, Mapping[str, torch.Tensor]],
            sample_sizes: Mapping[int, int]) -> Dict[str, float]:
        expected_clients = set(self.client_ids)
        if set(client_magnitudes) != expected_clients or set(
                sample_sizes) != expected_clients:
            raise RuntimeError(
                "CenMag aggregation requires full participation from exactly "
                f"{self.client_ids}; magnitude clients={sorted(client_magnitudes)}, "
                f"weight clients={sorted(sample_sizes)}")
        if any(int(sample_sizes[client_id]) <= 0 for client_id in self.client_ids):
            raise RuntimeError('CenMag aggregation received no positive samples for at least one client')
        total_samples = sum(int(sample_sizes[client_id])
                            for client_id in self.client_ids)
        if total_samples <= 0:
            raise RuntimeError("CenMag aggregation received no positive samples")
        weights = {
            client_id: float(sample_sizes[client_id]) / float(total_samples)
            for client_id in self.client_ids
        }
        self.last_aggregation_weights = copy.deepcopy(weights)
        for client_id in self.client_ids:
            self._validate_tensor_dict(client_magnitudes[client_id],
                                       self.keys.magnitude,
                                       f"client {client_id} magnitude")

        client_fp64 = {
            client_id: {
                key: client_magnitudes[client_id][key].detach().to(
                    device="cpu", dtype=torch.float64).clone()
                for key in self.keys.magnitude
            }
            for client_id in self.client_ids
        }
        new_global_fp64: TensorDict = {}
        for key in self.keys.magnitude:
            accumulator = torch.zeros_like(
                client_fp64[self.client_ids[0]][key], dtype=torch.float64)
            for client_id in self.client_ids:
                accumulator.add_(client_fp64[client_id][key],
                                 alpha=weights[client_id])
            new_global_fp64[key] = accumulator

        new_local_fp64 = {
            client_id: {
                key: client_fp64[client_id][key] - new_global_fp64[key]
                for key in self.keys.magnitude
            }
            for client_id in self.client_ids
        }

        client_values_finite = all(
            bool(torch.isfinite(client_magnitudes[client_id][key]).all())
            for client_id in self.client_ids
            for key in self.keys.magnitude)
        global_values_finite = all(bool(torch.isfinite(value).all())
                                   for value in new_global_fp64.values())
        local_values_finite = all(
            bool(torch.isfinite(new_local_fp64[client_id][key]).all())
            for client_id in self.client_ids
            for key in self.keys.magnitude)
        if not (client_values_finite and global_values_finite and
                local_values_finite):
            raise RuntimeError(
                "CenMag magnitude aggregation produced or received NaN/Inf")

        pre_rebase_errors = [
            (new_global_fp64[key] + new_local_fp64[client_id][key] -
             client_fp64[client_id][key]).abs().max()
            for client_id in self.client_ids
            for key in self.keys.magnitude
        ]

        # PDF-defined paired numerical recentering.  Moving the center and all
        # local components in opposite directions preserves every effective
        # client magnitude exactly in FP64 before storage casting.
        weighted_mean_fp64: TensorDict = {}
        for key in self.keys.magnitude:
            accumulator = torch.zeros_like(new_global_fp64[key])
            for client_id in self.client_ids:
                accumulator.add_(new_local_fp64[client_id][key],
                                 alpha=weights[client_id])
            weighted_mean_fp64[key] = accumulator

        before_correction = {
            client_id: {
                key: new_global_fp64[key] + new_local_fp64[client_id][key]
                for key in self.keys.magnitude
            }
            for client_id in self.client_ids
        }
        for key in self.keys.magnitude:
            new_global_fp64[key].add_(weighted_mean_fp64[key])
        for client_id in self.client_ids:
            for key in self.keys.magnitude:
                new_local_fp64[client_id][key].sub_(weighted_mean_fp64[key])

        correction_preservation_errors = [
            (new_global_fp64[key] + new_local_fp64[client_id][key] -
             before_correction[client_id][key]).abs().max()
            for client_id in self.client_ids
            for key in self.keys.magnitude
        ]
        post_center_fp64: TensorDict = {}
        for key in self.keys.magnitude:
            accumulator = torch.zeros_like(new_global_fp64[key])
            for client_id in self.client_ids:
                accumulator.add_(new_local_fp64[client_id][key],
                                 alpha=weights[client_id])
            post_center_fp64[key] = accumulator

        new_global = {
            key: value.to(dtype=torch.float32)
            for key, value in new_global_fp64.items()
        }
        new_local = {
            client_id: {
                key: value.to(dtype=torch.float32)
                for key, value in new_local_fp64[client_id].items()
            }
            for client_id in self.client_ids
        }
        post_center_fp32: TensorDict = {}
        for key in self.keys.magnitude:
            accumulator = torch.zeros_like(new_global[key])
            for client_id in self.client_ids:
                accumulator.add_(new_local[client_id][key],
                                 alpha=weights[client_id])
            post_center_fp32[key] = accumulator
        all_abs = torch.cat([
            value.abs().reshape(-1) for value in post_center_fp32.values()
        ])
        reconstruction_errors = [
            (new_global[key] + new_local[client_id][key] -
             _cpu_fp32_clone(client_magnitudes[client_id][key])).abs().max()
            for client_id in self.client_ids
            for key in self.keys.magnitude
        ]
        p_m = sum(value.numel() for value in new_global.values())
        dispersion = sum(
            weights[client_id] * sum(
                float(value.double().pow(2).sum().item())
                for value in new_local[client_id].values())
            for client_id in self.client_ids) / float(p_m)
        stats = {
            "max_abs_weighted_residual_mean": float(all_abs.max().item()),
            "mean_abs_weighted_residual_mean": float(all_abs.mean().item()),
            "weighted_residual_mean_l2": float(torch.sqrt(sum(
                value.double().pow(2).sum()
                for value in post_center_fp32.values())).item()),
            "client_magnitude_dispersion": float(dispersion),
            "p_m": int(p_m),
        }
        if self.validation_mode and stats[
                "max_abs_weighted_residual_mean"] > self.tolerance:
            raise RuntimeError(
                "CenMag residual centering invariant failed: "
                f"{stats['max_abs_weighted_residual_mean']} > {self.tolerance}")

        aggregation_validation = {
            "client_ids": list(self.client_ids),
            "client_count": len(self.client_ids),
            "magnitude_tensor_count": len(self.keys.magnitude),
            "all_client_magnitudes_finite": client_values_finite,
            "all_global_magnitudes_finite": global_values_finite,
            "all_residuals_finite": local_values_finite,
            "max_abs_residual_formula_error_before_centering": float(
                torch.stack(pre_rebase_errors).max().item()),
            "max_abs_paired_correction_preservation_error": float(
                torch.stack(correction_preservation_errors).max().item()),
            "max_abs_weighted_local_mean_fp64": float(torch.cat([
                value.abs().reshape(-1)
                for value in post_center_fp64.values()
            ]).max().item()),
            "max_abs_effective_reconstruction_error_after_centering": float(
                torch.stack(reconstruction_errors).max().item()),
        }
        if (self.validation_mode and aggregation_validation[
                "max_abs_effective_reconstruction_error_after_centering"] >
                self.tolerance):
            raise RuntimeError(
                "CenMag effective magnitude reconstruction failed after "
                "floating-point recentering: "
                f"{aggregation_validation}")

        self.global_magnitude = _clone_tensor_dict(new_global)
        self.client_local_magnitudes = {
            client_id: _clone_tensor_dict(new_local[client_id])
            for client_id in self.client_ids
        }
        self.last_centering = stats
        self.last_aggregation_validation = aggregation_validation
        self.assert_local_magnitude_isolation()
        return stats

    def export_state(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "client_ids": list(self.client_ids),
            "global_a": _clone_tensor_dict(self.global_a),
            "global_magnitude": _clone_tensor_dict(self.global_magnitude),
            "client_local_magnitudes": {
                client_id: _clone_tensor_dict(local_magnitude)
                for client_id, local_magnitude in
                self.client_local_magnitudes.items()
            },
            "client_local_states": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in
                self.resume_client_local_states.items()
            },
            "client_local_rounds": {
                client_id: int(server_round)
                for client_id, server_round in
                self.resume_client_local_rounds.items()
            },
            "last_centering": copy.deepcopy(self.last_centering),
            "last_aggregation_weights": copy.deepcopy(
                self.last_aggregation_weights),
            "last_aggregation_validation": copy.deepcopy(
                self.last_aggregation_validation),
            "last_communication": copy.deepcopy(self.last_communication),
            "last_gradient_validation": copy.deepcopy(
                self.last_gradient_validation),
            "last_memory": copy.deepcopy(self.last_memory),
            "last_client_start": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in self.last_client_start.items()
            },
            "last_evaluation_start": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in self.last_evaluation_start.items()
            },
            "last_client_local_start": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in self.last_client_local_start.items()
            },
            "last_client_local_end": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in self.last_client_local_end.items()
            },
            "last_evaluation_local_start": {
                client_id: _clone_tensor_dict(tensors)
                for client_id, tensors in
                self.last_evaluation_local_start.items()
            },
            "last_rng_schedule": copy.deepcopy(self.last_rng_schedule),
            "resume_local_state": _clone_tensor_dict(
                self.resume_local_state),
            "resume_round": int(self.resume_round),
        }

    def validate_exported_state(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("version", -1)) != self.VERSION:
            raise RuntimeError(
                f"Unsupported CenMag state version {payload.get('version')}")
        if tuple(sorted(int(item) for item in payload["client_ids"])) != self.client_ids:
            raise RuntimeError(
                f"Checkpoint client IDs {payload['client_ids']} do not match "
                f"configured clients {self.client_ids}")
        self._validate_tensor_dict(payload["global_a"], self.keys.lora_a,
                                   "checkpoint global A")
        self._validate_tensor_dict(payload["global_magnitude"],
                                   self.keys.magnitude,
                                   "checkpoint global magnitude")
        local_keys = self.keys.lora_b + self.keys.classifier
        self._validate_tensor_dict(payload["resume_local_state"], local_keys,
                                   "checkpoint resume local state")
        resume_round = int(payload["resume_round"])
        if resume_round < 0:
            raise RuntimeError("Checkpoint resume round must be non-negative")
        local_magnitude_payload = {
            int(client_id): local_magnitude
            for client_id, local_magnitude in
            payload["client_local_magnitudes"].items()
        }
        if set(local_magnitude_payload) != set(self.client_ids):
            raise RuntimeError(
                "Checkpoint client-local magnitude IDs do not match")
        for client_id in self.client_ids:
            self._validate_tensor_dict(local_magnitude_payload[client_id],
                                       self.keys.magnitude,
                                       f"checkpoint local magnitude {client_id}")
        client_local_payload = {
            int(client_id): tensors
            for client_id, tensors in payload["client_local_states"].items()
        }
        if set(client_local_payload) != set(self.client_ids):
            raise RuntimeError(
                "Checkpoint client-local state IDs do not match")
        client_local_rounds = {
            int(client_id): int(server_round)
            for client_id, server_round in
            payload["client_local_rounds"].items()
        }
        if set(client_local_rounds) != set(self.client_ids):
            raise RuntimeError(
                "Checkpoint client-local round IDs do not match")
        for client_id in self.client_ids:
            self._validate_tensor_dict(
                client_local_payload[client_id], local_keys,
                f"checkpoint client-local state {client_id}")
            if client_local_rounds[client_id] != resume_round:
                raise RuntimeError(
                    "Checkpoint client-local round does not match resume "
                    f"boundary for client {client_id}")
        persistent_groups = {
            "checkpoint global A": payload["global_a"],
            "checkpoint global magnitude": payload["global_magnitude"],
            "checkpoint resume local state": payload["resume_local_state"],
        }
        for client_id in self.client_ids:
            persistent_groups[f"checkpoint local magnitude {client_id}"] = \
                local_magnitude_payload[client_id]
            persistent_groups[f"checkpoint client-local state {client_id}"] = \
                client_local_payload[client_id]
        for label, tensors in persistent_groups.items():
            for key, value in tensors.items():
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"{label} contains non-tensor {key}")
                if value.is_floating_point() and not bool(
                        torch.isfinite(value).all()):
                    raise RuntimeError(f"{label} contains non-finite {key}")
        for key, value in payload["global_magnitude"].items():
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"Checkpoint global magnitude is not FP32 for {key}")
        for client_id in self.client_ids:
            for key, value in local_magnitude_payload[client_id].items():
                if value.dtype != torch.float32:
                    raise RuntimeError(
                        f"Checkpoint local magnitude {client_id}/{key} is not FP32")
        schedule = payload.get("last_rng_schedule", {})
        if schedule:
            required_schedule = {
                "base_seed", "server_round", "client_order",
                "client_seeds", "completed_clients",
            }
            if set(schedule) != required_schedule:
                raise RuntimeError("Checkpoint RNG schedule key mismatch")
            order = [int(item) for item in schedule["client_order"]]
            if order != list(self.client_ids):
                raise RuntimeError("Checkpoint RNG client order mismatch")
            seeds = {int(key): int(value)
                     for key, value in schedule["client_seeds"].items()}
            if set(seeds) != set(self.client_ids):
                raise RuntimeError("Checkpoint RNG client seed IDs mismatch")
            for client_id in self.client_ids:
                expected = derive_client_round_seed(
                    int(schedule["base_seed"]),
                    int(schedule["server_round"]), client_id)
                if seeds[client_id] != expected:
                    raise RuntimeError(
                        f"Checkpoint RNG seed mismatch for client {client_id}")
            completed = [int(item) for item in schedule["completed_clients"]]
            if len(completed) != len(set(completed)) or not set(
                    completed).issubset(set(self.client_ids)):
                raise RuntimeError("Checkpoint completed RNG client list invalid")

    def load_exported_state(self, payload: Mapping[str, Any]) -> None:
        self.validate_exported_state(payload)
        resume_round = int(payload["resume_round"])
        local_magnitude_payload = {
            int(client_id): local_magnitude
            for client_id, local_magnitude in
            payload["client_local_magnitudes"].items()
        }
        client_local_payload = {
            int(client_id): tensors
            for client_id, tensors in payload["client_local_states"].items()
        }
        self.global_a = {
            key: value.detach().cpu().clone()
            for key, value in payload["global_a"].items()
        }
        self.global_magnitude = {
            key: _cpu_fp32_clone(value)
            for key, value in payload["global_magnitude"].items()
        }
        self.client_local_magnitudes = {
            client_id: {
                key: _cpu_fp32_clone(value)
                for key, value in local_magnitude_payload[client_id].items()
            }
            for client_id in self.client_ids
        }
        self.client_local_states = {
            client_id: {
                key: value.detach().cpu().clone()
                for key, value in client_local_payload[client_id].items()
            }
            for client_id in self.client_ids
        }
        self.client_local_rounds = {
            int(client_id): int(server_round)
            for client_id, server_round in
            payload["client_local_rounds"].items()
        }
        self.resume_client_local_states = {
            client_id: _clone_tensor_dict(tensors)
            for client_id, tensors in self.client_local_states.items()
        }
        self.resume_client_local_rounds = dict(self.client_local_rounds)
        self.last_centering = copy.deepcopy(payload.get("last_centering", {}))
        self.last_aggregation_weights = {
            int(client_id): float(weight)
            for client_id, weight in payload.get(
                "last_aggregation_weights", {}).items()
        }
        self.last_aggregation_validation = copy.deepcopy(
            payload.get("last_aggregation_validation", {}))
        self.last_communication = copy.deepcopy(
            payload.get("last_communication", {}))
        self.last_gradient_validation = copy.deepcopy(
            payload.get("last_gradient_validation", {}))
        self.last_memory = copy.deepcopy(payload.get("last_memory", {}))
        self.last_client_start = {
            int(client_id): _clone_tensor_dict(tensors)
            for client_id, tensors in payload.get("last_client_start", {}).items()
        }
        self.last_evaluation_start = {
            int(client_id): _clone_tensor_dict(tensors)
            for client_id, tensors in payload.get(
                "last_evaluation_start", {}).items()
        }
        self.last_client_local_start = {
            int(client_id): _clone_tensor_dict(tensors)
            for client_id, tensors in payload.get(
                "last_client_local_start", {}).items()
        }
        self.last_client_local_end = {
            int(client_id): _clone_tensor_dict(tensors)
            for client_id, tensors in payload.get(
                "last_client_local_end", {}).items()
        }
        self.last_evaluation_local_start = {
            int(client_id): _clone_tensor_dict(tensors)
            for client_id, tensors in payload.get(
                "last_evaluation_local_start", {}).items()
        }
        self.last_rng_schedule = copy.deepcopy(
            payload.get("last_rng_schedule", {}))
        self.resume_local_state = {
            key: value.detach().cpu().clone()
            for key, value in payload["resume_local_state"].items()
        }
        self.resume_round = resume_round
        self.assert_local_magnitude_isolation()
        self.assert_client_local_isolation()
