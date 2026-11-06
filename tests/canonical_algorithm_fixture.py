"""Mandatory correctness and synthetic smoke tests for PDF-defined CenMag."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from federatedscope.glue.cenmag.aggregator import (
    CHECKPOINT_VERSION, CenMagAggregator)
from federatedscope.glue.cenmag.magnitude import (
    binding_modules, cenmag_optimizer_parameters, effective_magnitude_state,
    install_cenmag_magnitude_parametrizations,
    local_magnitude_parameters, magnitude_ownership_audit,
    magnitude_regularization_loss)
from federatedscope.glue.cenmag.state import CenMagState


class DoraLinearLayer(torch.nn.Module):
    """Small PEFT-shaped stand-in; the class name is intentionally exact."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(1.0, 2.0, width, dtype=torch.float32))


class ToyPeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.ModuleDict({
            "default": torch.nn.Linear(4, 2, bias=False),
        })
        self.block.lora_B = torch.nn.ModuleDict({
            "default": torch.nn.Linear(2, 4, bias=False),
        })
        self.block.lora_magnitude_vector = torch.nn.ModuleDict({
            "default": DoraLinearLayer(4),
        })
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        direction = self.block.lora_B["default"](
            self.block.lora_A["default"](inputs))
        magnitude = self.block.lora_magnitude_vector["default"].weight
        return self.classifier(direction * magnitude)


class AdapterLikeModel(torch.nn.Module):
    """Match AdapterModel trainable-only state and prefixed parameter names."""

    def __init__(self):
        super().__init__()
        self.model = ToyPeftModel()
        install_cenmag_magnitude_parametrizations(self, expected_tensors=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def state_dict(self, *args, **kwargs):
        trainable = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        return {
            key: value for key, value in self.model.state_dict(
                *args, **kwargs).items() if key in trainable
        }

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict=False)


def fake_config():
    return SimpleNamespace(
        seed=123,
        model=SimpleNamespace(type="toy@huggingface_llm"),
        federate=SimpleNamespace(client_num=3),
        data=SimpleNamespace(
            type="mnli@glue", splitter="lda",
            splitter_args=[{"alpha": 0.5}],
        ),
        dataloader=SimpleNamespace(
            batch_size=128, shuffle=True, drop_last=False, num_workers=0),
        train=SimpleNamespace(
            local_update_steps=10,
            optimizer={
                "type": "SGD", "lr": 0.02, "momentum": 0.0,
                "weight_decay": 0.0,
            },
        ),
        llm=SimpleNamespace(
            tok_len=128,
            adapter=SimpleNamespace(args=[{
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "target_modules": ["query"],
                "use_dora": True,
            }]),
        ),
        cenmag=SimpleNamespace(
            checkpoint_version=CHECKPOINT_VERSION,
            deterministic_round_seeding=True,
            centering_tolerance=1e-6,
            validation_mode=True,
            expected_magnitude_tensors=1,
            lambda_m=1.0,
            quantization=SimpleNamespace(
                enabled=True,
                bits=4,
                type="nf4",
                compute_dtype="float16",
                double_quantization=True,
                gradient_checkpointing=True,
                bitsandbytes_version="0.50.0",
            ),
        ),
    )


def canonical_magnitude_key() -> str:
    return "block.lora_magnitude_vector.default.weight"


def make_state(client_ids=(1, 2, 3)):
    model = AdapterLikeModel()
    state = CenMagState(
        model, client_ids, validation_mode=True,
        expected_magnitude_tensors=1)
    setattr(model, "_cenmag_state", state)
    return model, state


def test_01_to_05_gradient_and_optimizer_ownership():
    """Tests 1-5: frozen global, trainable local, optimizer, gradients, duplicate."""
    torch.manual_seed(7)
    model, _ = make_state()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=0.02)
    modules = binding_modules(model)
    assert all(not module.global_magnitude.requires_grad
               for module in modules.values())
    assert all(module.client_local_magnitude.requires_grad
               for module in modules.values())

    audit = magnitude_ownership_audit(model, optimizer)
    assert audit["all_global_frozen"] is True
    assert audit["all_local_trainable"] is True
    assert audit["all_native_frozen"] is True
    assert audit["local_in_optimizer"] is True
    assert audit["global_in_optimizer"] is False
    assert audit["native_in_optimizer"] is False
    assert audit["duplicate_trainable_effective_magnitude"] is False

    inputs = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    task_loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    total_loss = task_loss + magnitude_regularization_loss(model, 1.0)
    total_loss.backward()
    local_gradients = [parameter.grad
                       for parameter in local_magnitude_parameters(model)]
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in local_gradients)
    assert any(float(gradient.norm().item()) > 0.0
               for gradient in local_gradients)
    assert all(getattr(module.global_magnitude, "grad", None) is None
               for module in modules.values())
    trainable_magnitude_names = [
        name for name, parameter in model.named_parameters()
        if "magnitude" in name and parameter.requires_grad
    ]
    assert len(trainable_magnitude_names) == len(modules)
    assert all("client_local_magnitude" in name
               for name in trainable_magnitude_names)


def test_06_and_12_regularizer_value_and_lambda_zero():
    """Tests 6 and 12: exact normalized L2 value and exact zero at lambda=0."""
    model, _ = make_state()
    parameters = local_magnitude_parameters(model)
    with torch.no_grad():
        for index, parameter in enumerate(parameters, start=1):
            parameter.copy_(torch.arange(
                1, parameter.numel() + 1,
                device=parameter.device, dtype=parameter.dtype) * index)
    p_m = sum(parameter.numel() for parameter in parameters)
    independent = sum(parameter.float().pow(2).sum()
                      for parameter in parameters) / (2.0 * p_m)
    implemented = magnitude_regularization_loss(model, 1.0)
    assert torch.equal(implemented, independent)
    assert magnitude_regularization_loss(model, 0.0).item() == 0.0


def test_07_to_10_aggregation_rebase_and_paired_recentering():
    """Tests 7-10: weighted center, rebase, centering, preservation."""
    _, state = make_state()
    key = canonical_magnitude_key()
    trained = {
        1: {key: torch.tensor([1.0, 2.0, 3.0, 4.0])},
        2: {key: torch.tensor([2.0, 4.0, 6.0, 8.0])},
        3: {key: torch.tensor([5.0, 7.0, 9.0, 11.0])},
    }
    sizes = {1: 1, 2: 2, 3: 3}
    expected = (trained[1][key] + 2 * trained[2][key] +
                3 * trained[3][key]) / 6.0
    stats = state.update_from_client_magnitudes(trained, sizes)
    assert torch.allclose(state.global_magnitude[key], expected, atol=1e-7,
                          rtol=0.0)
    for client_id in state.client_ids:
        reconstructed = (state.global_magnitude[key] +
                         state.client_local_magnitudes[client_id][key])
        assert torch.allclose(reconstructed, trained[client_id][key],
                              atol=state.tolerance, rtol=0.0)
    weighted_local = sum(
        (sizes[client_id] / 6.0) *
        state.client_local_magnitudes[client_id][key]
        for client_id in state.client_ids)
    assert torch.allclose(weighted_local, torch.zeros_like(weighted_local),
                          atol=state.tolerance, rtol=0.0)
    validation = state.last_aggregation_validation
    assert validation[
        "max_abs_paired_correction_preservation_error"] <= state.tolerance
    assert validation[
        "max_abs_effective_reconstruction_error_after_centering"] <= \
        state.tolerance
    assert stats["max_abs_weighted_residual_mean"] <= state.tolerance


def test_11_b_and_classifier_are_local_and_never_communicated():
    """Test 11: B_i and C_i are absent from the exact A+effective upload."""
    model, state = make_state()
    live = model.state_dict()
    payload = {
        key: live[key].detach().clone() for key in state.keys.lora_a
    }
    payload.update({
        key: value.detach().clone()
        for key, value in effective_magnitude_state(model).items()
    })
    state.validate_communicated_state(payload, "synthetic client upload")
    assert set(payload) == set(state.keys.lora_a + state.keys.magnitude)
    assert not set(state.keys.lora_b).intersection(payload)
    assert not set(state.keys.classifier).intersection(payload)
    assert not set(state.keys.local_magnitude).intersection(payload)


def test_personalized_client_load_supports_derived_non_state_dict_magnitude():
    """The canonical effective magnitude is derived, not a state-dict entry."""
    model, state = make_state()
    key = canonical_magnitude_key()
    assert key not in model.state_dict()
    state.client_local_magnitudes[2][key].copy_(
        torch.tensor([0.25, -0.5, 0.75, -1.0]))
    expected = state.effective_magnitude(2)
    state.load_effective_magnitude(model, 2)
    state.assert_loaded(model, expected, "derived magnitude client load")
    assert torch.allclose(
        effective_magnitude_state(model)[key].detach().cpu(),
        expected[key], atol=state.tolerance, rtol=0.0)


def test_production_optimizer_input_excludes_frozen_native_magnitude():
    """The trainer wiring must omit every frozen magnitude parameter."""
    model, _ = make_state()
    optimizer = torch.optim.SGD(
        cenmag_optimizer_parameters(model), lr=0.02)
    ownership = magnitude_ownership_audit(model, optimizer)
    assert ownership["local_in_optimizer"] is True
    assert ownership["global_in_optimizer"] is False
    assert ownership["native_in_optimizer"] is False


def test_synthetic_forward_round_evaluation_and_checkpoint_resume(tmp_path):
    """Smallest complete local smoke: train, aggregate, eval, save and load."""
    torch.manual_seed(19)
    model, state = make_state()
    key = canonical_magnitude_key()
    uploads = {}
    sample_sizes = {}
    evaluation_inputs = torch.randn(5, 4)

    for client_id in state.client_ids:
        state.load_client_local_state(model, client_id)
        state.load_effective_magnitude(model, client_id)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], lr=0.02)
        inputs = torch.randn(4, 4)
        targets = torch.tensor([
            client_id % 3, (client_id + 1) % 3,
            (client_id + 2) % 3, client_id % 3,
        ])
        optimizer.zero_grad()
        task = torch.nn.functional.cross_entropy(model(inputs), targets)
        regular = magnitude_regularization_loss(model, 1.0)
        (task + regular).backward()
        optimizer.step()
        uploads[client_id] = {
            key: value.detach().cpu().clone()
            for key, value in effective_magnitude_state(model).items()
        }
        sample_sizes[client_id] = len(inputs)
        state.capture_client_local_state(model, client_id, 1)

    state.update_from_client_magnitudes(uploads, sample_sizes)
    state.capture_global_a(model)
    state.capture_resume_local_state(model, 1)
    state.load_client_local_state(model, 1, evaluation=True)
    state.load_effective_magnitude(model, 1)
    with torch.no_grad():
        logits = model(evaluation_inputs)
    assert logits.shape == (5, 3)
    assert torch.isfinite(logits).all()

    exported = state.export_state()
    checkpoint_path = tmp_path / "synthetic_state.pt"
    torch.save(exported, checkpoint_path)
    loaded_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    restored_model, restored = make_state()
    restored.load_exported_state(loaded_payload)
    for client_id in state.client_ids:
        assert torch.equal(
            restored.client_local_magnitudes[client_id][key],
            state.client_local_magnitudes[client_id][key])
    restored.load_effective_magnitude(restored_model, 1)
    assert torch.allclose(
        effective_magnitude_state(restored_model)[key].detach().cpu(),
        restored.effective_magnitude(1)[key], atol=1e-7, rtol=0.0)


def test_corruption_safe_checkpoint_contains_new_ownership_state(tmp_path):
    model = AdapterLikeModel()
    aggregator = CenMagAggregator(model, "cpu", fake_config())
    state = aggregator.cenmag_state
    key = canonical_magnitude_key()
    trained = {
        1: {key: torch.tensor([1.0, 2.0, 3.0, 4.0])},
        2: {key: torch.tensor([2.0, 3.0, 4.0, 5.0])},
        3: {key: torch.tensor([4.0, 5.0, 6.0, 7.0])},
    }
    state.update_from_client_magnitudes(trained, {1: 2, 2: 3, 3: 5})
    for client_id in state.client_ids:
        state.load_client_local_state(model, client_id)
        state.load_effective_magnitude(model, client_id)
        with torch.no_grad():
            model.model.block.lora_B["default"].weight.add_(client_id / 10)
            model.model.classifier.weight.add_(client_id / 20)
        state.capture_client_local_state(model, client_id, 1)
    state.capture_global_a(model)
    state.capture_resume_local_state(model, 1)

    result = aggregator.save_model(tmp_path / "latest.pt", cur_round=1)
    assert result["server_round"] == 1
    payload = torch.load(
        tmp_path / "round_000001.pt", map_location="cpu",
        weights_only=False)
    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["method"] == "cenmag_new"
    assert set(payload["client_local_magnitudes"]) == {1, 2, 3}
    assert payload["metadata"]["lambda_m"] == 1.0

    restored_model = AdapterLikeModel()
    restored = CenMagAggregator(restored_model, "cpu", fake_config())
    assert restored.load_model(tmp_path / "latest.pt") == 1
    for client_id in state.client_ids:
        assert torch.equal(
            restored.cenmag_state.client_local_magnitudes[client_id][key],
            state.client_local_magnitudes[client_id][key])
        assert restored.cenmag_state.client_local_rounds[client_id] == 1
