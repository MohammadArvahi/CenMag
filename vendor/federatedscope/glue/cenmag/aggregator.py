"""Canonical CenMag checkpointing; remaining boundary defects are documented."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.glue.cenmag.state import CenMagState


CHECKPOINT_VERSION = 6
METHOD_NAME = "cenmag_new"
MANIFEST_VERSION = 1
NUMPY_RNG_ENCODING = "numpy_random_state_primitives_v1"


def _plain(value: Any) -> Any:
    """Convert configuration nodes to stable JSON-like metadata."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return {str(key): _plain(item) for key, item in dict(value).items()}
    except Exception:
        return str(value)


def checkpoint_file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_dict_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor keys, dtypes, shapes, and exact CPU values."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """Capture every process-global RNG used by the supported runner."""
    cuda_states = []
    cuda_device_count = 0
    if torch.cuda.is_available():
        cuda_device_count = int(torch.cuda.device_count())
        cuda_states = [state.detach().cpu().clone()
                       for state in torch.cuda.get_rng_state_all()]
    numpy_state = np.random.get_state()
    numpy_payload = {
        "encoding": NUMPY_RNG_ENCODING,
        "numpy_version": str(np.__version__),
        "bit_generator": str(numpy_state[0]),
        # Never pickle NumPy arrays in a cross-process checkpoint. Kaggle's
        # already-running notebook kernel can retain a different NumPy ABI
        # after dependency resolution than the fresh training subprocess.
        "keys": [int(value) for value in numpy_state[1].tolist()],
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    return {
        "python": random.getstate(),
        "numpy": numpy_payload,
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_states,
        "cuda_device_count": cuda_device_count,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
    }


def _validate_rng_state(payload: Mapping[str, Any]) -> None:
    required = {
        "python", "numpy", "torch_cpu", "torch_cuda",
        "cuda_device_count", "cudnn_benchmark", "cudnn_deterministic",
        "deterministic_algorithms",
    }
    if set(payload) != required:
        raise RuntimeError(
            "CenMag checkpoint RNG key mismatch; "
            f"missing={sorted(required-set(payload))}, "
            f"extra={sorted(set(payload)-required)}")
    if not isinstance(payload["torch_cpu"], torch.Tensor) or \
            payload["torch_cpu"].dtype != torch.uint8:
        raise RuntimeError("CenMag checkpoint has invalid Torch CPU RNG state")
    numpy_payload = payload["numpy"]
    numpy_required = {
        "encoding", "numpy_version", "bit_generator", "keys", "position",
        "has_gauss", "cached_gaussian",
    }
    if not isinstance(numpy_payload, Mapping) or \
            set(numpy_payload) != numpy_required:
        raise RuntimeError("CenMag checkpoint has invalid NumPy RNG payload")
    if numpy_payload["encoding"] != NUMPY_RNG_ENCODING or \
            numpy_payload["bit_generator"] != "MT19937" or \
            not isinstance(numpy_payload["numpy_version"], str):
        raise RuntimeError("CenMag checkpoint NumPy RNG encoding mismatch")
    numpy_keys = numpy_payload["keys"]
    if not isinstance(numpy_keys, list) or len(numpy_keys) != 624 or any(
            isinstance(value, bool) or not isinstance(value, int) or
            value < 0 or value > 2**32 - 1 for value in numpy_keys):
        raise RuntimeError("CenMag checkpoint has invalid NumPy RNG keys")
    position = numpy_payload["position"]
    has_gauss = numpy_payload["has_gauss"]
    cached_gaussian = numpy_payload["cached_gaussian"]
    if isinstance(position, bool) or not isinstance(position, int) or \
            position < 0 or position > 624 or \
            isinstance(has_gauss, bool) or has_gauss not in (0, 1) or \
            not isinstance(cached_gaussian, (int, float)) or \
            not np.isfinite(float(cached_gaussian)):
        raise RuntimeError("CenMag checkpoint has invalid NumPy RNG metadata")
    cuda_states = payload["torch_cuda"]
    if len(cuda_states) != int(payload["cuda_device_count"]):
        raise RuntimeError("CenMag checkpoint CUDA RNG count is inconsistent")
    if any(not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
           for state in cuda_states):
        raise RuntimeError("CenMag checkpoint has invalid CUDA RNG state")


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    """Restore RNG state only after the checkpoint has fully validated."""
    _validate_rng_state(payload)
    checkpoint_cuda_count = int(payload["cuda_device_count"])
    current_cuda_count = int(torch.cuda.device_count()) \
        if torch.cuda.is_available() else 0
    if checkpoint_cuda_count != current_cuda_count:
        raise RuntimeError(
            "CenMag checkpoint CUDA RNG device count mismatch; "
            f"checkpoint={checkpoint_cuda_count}, runtime={current_cuda_count}")
    random.setstate(payload["python"])
    numpy_payload = payload["numpy"]
    np.random.set_state((
        numpy_payload["bit_generator"],
        np.asarray(numpy_payload["keys"], dtype=np.uint32),
        int(numpy_payload["position"]),
        int(numpy_payload["has_gauss"]),
        float(numpy_payload["cached_gaussian"]),
    ))
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if checkpoint_cuda_count:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["torch_cuda"]])
    torch.backends.cudnn.benchmark = bool(payload["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(payload["cudnn_deterministic"])
    deterministic_target = bool(payload["deterministic_algorithms"])
    if torch.are_deterministic_algorithms_enabled() != deterministic_target:
        torch.use_deterministic_algorithms(deterministic_target)


def cenmag_metadata(config: Any) -> Dict[str, Any]:
    adapter_args = dict(config.llm.adapter.args[0])
    return {
        "model_identifier": str(config.model.type),
        "client_num": int(config.federate.client_num),
        "rank": int(adapter_args["r"]),
        "lora_alpha": int(adapter_args["lora_alpha"]),
        "lora_dropout": float(adapter_args["lora_dropout"]),
        "target_modules": sorted(list(adapter_args["target_modules"])),
        "use_dora": bool(adapter_args.get("use_dora", False)),
        "lambda_m": float(config.cenmag.lambda_m),
        "configured_checkpoint_version": int(config.cenmag.checkpoint_version),
        "quantization": {
            "enabled": bool(config.cenmag.quantization.enabled),
            "bits": int(config.cenmag.quantization.bits),
            "type": str(config.cenmag.quantization.type),
            "compute_dtype": str(config.cenmag.quantization.compute_dtype),
            "double_quantization": bool(
                config.cenmag.quantization.double_quantization),
            "gradient_checkpointing": bool(
                config.cenmag.quantization.gradient_checkpointing),
            "bitsandbytes_version": str(
                config.cenmag.quantization.bitsandbytes_version),
        },
        "experiment": {
            "seed": int(config.seed),
            "data_type": str(config.data.type),
            "data_splitter": str(config.data.splitter),
            "data_splitter_args": _plain(config.data.splitter_args),
            "partition_seed": 12345,
            "batch_size": int(config.dataloader.batch_size),
            "dataloader_shuffle": bool(config.dataloader.shuffle),
            "dataloader_drop_last": bool(config.dataloader.drop_last),
            "dataloader_num_workers": int(config.dataloader.num_workers),
            "local_update_steps": int(config.train.local_update_steps),
            "optimizer": _plain(config.train.optimizer),
            "sequence_length": int(config.llm.tok_len),
            "full_participation": True,
            "deterministic_round_seeding": bool(
                config.cenmag.deterministic_round_seeding),
        },
    }


class CenMagAggregator(ClientsAvgAggregator):
    """Retain FedAvg math while owning explicit CenMag persistent state."""

    def __init__(self, model=None, device="cpu", config=None):
        super().__init__(model=model, device=device, config=config)
        self.loaded_round = None
        self.last_saved_path = None
        self.cenmag_state = CenMagState(
            model=model,
            client_ids=range(1, int(config.federate.client_num) + 1),
            tolerance=float(config.cenmag.centering_tolerance),
            validation_mode=bool(config.cenmag.validation_mode),
            expected_magnitude_tensors=int(
                config.cenmag.expected_magnitude_tensors),
        )
        # Standalone FedSA deliberately shares this model object. The attached
        # coordinator contains only detached CPU snapshots, never live tensors.
        setattr(model, "_cenmag_state", self.cenmag_state)

    def _canonical_model_state(self) -> Dict[str, torch.Tensor]:
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
            if key not in self.cenmag_state.keys.local_magnitude
        }
        for key, value in self.cenmag_state.global_a.items():
            state[key] = value.detach().cpu().clone()
        for key, value in self.cenmag_state.global_magnitude.items():
            state[key] = value.detach().cpu().clone()
        for key, value in self.cenmag_state.resume_local_state.items():
            state[key] = value.detach().cpu().clone()
        return state

    def _build_checkpoint(self, cur_round: int) -> Dict[str, Any]:
        resume_round = int(self.cenmag_state.resume_round)
        return {
            "version": CHECKPOINT_VERSION,
            "method": METHOD_NAME,
            "checkpoint_uuid": str(uuid.uuid4()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "cur_round": resume_round,
            "server_round": resume_round,
            "last_completed_round": resume_round - 1,
            "requested_server_state": int(cur_round),
            "resident_state_round": resume_round,
            "model": self._canonical_model_state(),
            "global_a": {
                key: value.detach().cpu().clone()
                for key, value in self.cenmag_state.global_a.items()
            },
            "global_magnitude": {
                key: value.detach().cpu().clone()
                for key, value in self.cenmag_state.global_magnitude.items()
            },
            "client_local_magnitudes": {
                client_id: {
                    key: value.detach().cpu().clone()
                    for key, value in local_magnitude.items()
                }
                for client_id, local_magnitude in
                self.cenmag_state.client_local_magnitudes.items()
            },
            "client_local_states": {
                client_id: {
                    key: value.detach().cpu().clone()
                    for key, value in tensors.items()
                }
                for client_id, tensors in
                self.cenmag_state.resume_client_local_states.items()
            },
            "client_local_rounds": {
                client_id: int(server_round)
                for client_id, server_round in
                self.cenmag_state.resume_client_local_rounds.items()
            },
            "resident_local_state": {
                key: value.detach().cpu().clone()
                for key, value in self.cenmag_state.resume_local_state.items()
            },
            "cenmag_state": self.cenmag_state.export_state(),
            "rng_state": capture_rng_state(),
            "resume_policy": {
                "optimizer": "recreated_per_client_routine",
                "scheduler": "recreated_per_client_routine",
                "dataloader": "recreated_then_deterministically_seeded_per_round_client",
                "client_order": "ascending_client_id_full_participation",
                "round_seed_scheme": "sha256(base_seed,server_round,client_id)",
                "local_state": (
                    "client_keyed_lora_B_classifier_and_local_magnitude"),
            },
            "metadata": cenmag_metadata(self.cfg),
        }

    def _validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected = cenmag_metadata(self.cfg)
        if dict(metadata) != expected:
            raise RuntimeError(
                "CenMag checkpoint metadata does not match the active "
                f"configuration. expected={expected}, actual={dict(metadata)}")

    @staticmethod
    def _assert_finite_tensor_dict(tensors: Mapping[str, torch.Tensor],
                                   label: str) -> None:
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"{label} contains non-tensor value for {key}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{label} contains non-finite tensor {key}")

    @staticmethod
    def _assert_equal_tensor_dict(left: Mapping[str, torch.Tensor],
                                  right: Mapping[str, torch.Tensor],
                                  label: str) -> None:
        if set(left) != set(right):
            raise RuntimeError(f"{label} key mismatch")
        for key in left:
            if not torch.equal(left[key], right[key].to(dtype=left[key].dtype)):
                raise RuntimeError(f"{label} tensor mismatch for {key}")

    @classmethod
    def _assert_all_floating_tensors_finite(cls, value: Any,
                                            path: str = "checkpoint") -> None:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"CenMag checkpoint contains non-finite {path}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                cls._assert_all_floating_tensors_finite(
                    item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                cls._assert_all_floating_tensors_finite(
                    item, f"{path}[{index}]")

    def _validate_checkpoint_payload(self, checkpoint: Mapping[str, Any]) -> None:
        required = {
            "version", "method", "checkpoint_uuid", "saved_at_utc",
            "cur_round", "server_round", "last_completed_round",
            "requested_server_state", "resident_state_round", "model",
            "global_a", "global_magnitude", "client_local_magnitudes",
            "client_local_states", "client_local_rounds",
            "resident_local_state", "cenmag_state", "rng_state",
            "resume_policy", "metadata",
        }
        if set(checkpoint) != required:
            raise RuntimeError(
                "CenMag checkpoint top-level key mismatch; "
                f"missing={sorted(required-set(checkpoint))}, "
                f"extra={sorted(set(checkpoint)-required)}")
        if int(checkpoint["version"]) != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Unsupported CenMag checkpoint version {checkpoint['version']}")
        if checkpoint["method"] != METHOD_NAME:
            raise RuntimeError(
                f"Expected method {METHOD_NAME}, found {checkpoint['method']}")
        self._assert_all_floating_tensors_finite(checkpoint)
        self._validate_metadata(checkpoint["metadata"])

        server_round = int(checkpoint["server_round"])
        if server_round < 0 or int(checkpoint["cur_round"]) != server_round or \
                int(checkpoint["resident_state_round"]) != server_round or \
                int(checkpoint["last_completed_round"]) != server_round - 1:
            raise RuntimeError("CenMag checkpoint round boundary is inconsistent")
        state_payload = checkpoint["cenmag_state"]
        if int(state_payload.get("resume_round", -1)) != server_round:
            raise RuntimeError(
                "Checkpoint server round does not match resident local state")

        keys = self.cenmag_state.keys
        expected_model_keys = set(
            keys.lora_a + keys.lora_b + keys.magnitude + keys.classifier)
        if set(checkpoint["model"]) != expected_model_keys:
            raise RuntimeError(
                "CenMag canonical model key mismatch; "
                f"missing={sorted(expected_model_keys-set(checkpoint['model']))}, "
                f"extra={sorted(set(checkpoint['model'])-expected_model_keys)}")
        self.cenmag_state.validate_exported_state(state_payload)
        self.cenmag_state._validate_tensor_dict(
            checkpoint["global_a"], keys.lora_a, "checkpoint global A")
        self.cenmag_state._validate_tensor_dict(
            checkpoint["global_magnitude"], keys.magnitude,
            "checkpoint global magnitude")
        local_keys = keys.lora_b + keys.classifier
        self.cenmag_state._validate_tensor_dict(
            checkpoint["resident_local_state"], local_keys,
            "checkpoint resident local state")
        local_magnitudes = {
            int(key): value for key, value in
            checkpoint["client_local_magnitudes"].items()
        }
        if set(local_magnitudes) != set(self.cenmag_state.client_ids):
            raise RuntimeError(
                "Checkpoint client-local magnitude IDs do not match")
        for client_id, tensors in local_magnitudes.items():
            self.cenmag_state._validate_tensor_dict(
                tensors, keys.magnitude,
                f"checkpoint local magnitude client {client_id}")
        client_local_states = {
            int(key): value
            for key, value in checkpoint["client_local_states"].items()
        }
        client_local_rounds = {
            int(key): int(value)
            for key, value in checkpoint["client_local_rounds"].items()
        }
        if set(client_local_states) != set(self.cenmag_state.client_ids):
            raise RuntimeError(
                "Checkpoint client-local state IDs do not match")
        if set(client_local_rounds) != set(self.cenmag_state.client_ids):
            raise RuntimeError(
                "Checkpoint client-local round IDs do not match")
        for client_id, tensors in client_local_states.items():
            self.cenmag_state._validate_tensor_dict(
                tensors, local_keys,
                f"checkpoint client-local state {client_id}")
            if client_local_rounds[client_id] != server_round:
                raise RuntimeError(
                    "Checkpoint client-local round does not match server "
                    f"round for client {client_id}")

        tensor_groups = {
            "canonical model": checkpoint["model"],
            "global A": checkpoint["global_a"],
            "global magnitude": checkpoint["global_magnitude"],
            "resident local state": checkpoint["resident_local_state"],
        }
        for client_id, tensors in local_magnitudes.items():
            tensor_groups[f"local magnitude client {client_id}"] = tensors
        for client_id, tensors in client_local_states.items():
            tensor_groups[f"client-local state {client_id}"] = tensors
        for label, tensors in tensor_groups.items():
            self._assert_finite_tensor_dict(tensors, label)
        for key, value in checkpoint["global_magnitude"].items():
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"Checkpoint global magnitude is not FP32 for {key}")
        for client_id, tensors in local_magnitudes.items():
            for key, value in tensors.items():
                if value.dtype != torch.float32:
                    raise RuntimeError(
                        f"Checkpoint local magnitude {client_id}/{key} is not FP32")

        self._assert_equal_tensor_dict(
            checkpoint["global_a"], state_payload["global_a"],
            "top-level versus state global A")
        self._assert_equal_tensor_dict(
            checkpoint["global_magnitude"], state_payload["global_magnitude"],
            "top-level versus state global magnitude")
        self._assert_equal_tensor_dict(
            checkpoint["resident_local_state"],
            state_payload["resume_local_state"],
            "top-level versus state resident local state")
        for client_id in local_magnitudes:
            state_local_magnitudes = {
                int(key): value for key, value in
                state_payload["client_local_magnitudes"].items()
            }
            self._assert_equal_tensor_dict(
                local_magnitudes[client_id],
                state_local_magnitudes[client_id],
                f"top-level versus state local magnitude client {client_id}")
        state_client_locals = {
            int(key): value for key, value in
            state_payload["client_local_states"].items()
        }
        state_client_local_rounds = {
            int(key): int(value) for key, value in
            state_payload["client_local_rounds"].items()
        }
        if client_local_rounds != state_client_local_rounds:
            raise RuntimeError(
                "Top-level versus state client-local round mismatch")
        for client_id in client_local_states:
            self._assert_equal_tensor_dict(
                client_local_states[client_id],
                state_client_locals[client_id],
                f"top-level versus state client-local client {client_id}")
        self._assert_equal_tensor_dict(
            {key: checkpoint["model"][key] for key in keys.lora_a},
            checkpoint["global_a"], "canonical global A")
        self._assert_equal_tensor_dict(
            {key: checkpoint["model"][key] for key in keys.magnitude},
            checkpoint["global_magnitude"], "canonical global magnitude")
        self._assert_equal_tensor_dict(
            {key: checkpoint["model"][key] for key in local_keys},
            checkpoint["resident_local_state"],
            "canonical resident local state")
        _validate_rng_state(checkpoint["rng_state"])

    @staticmethod
    def _atomic_write_bytes(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False)
        temp_path = Path(handle.name)
        try:
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def _atomic_write_json(cls, destination: Path,
                           payload: Mapping[str, Any]) -> None:
        cls._atomic_write_bytes(
            destination,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"))

    def _atomic_torch_save(self, destination: Path,
                           checkpoint: Mapping[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False)
        temp_path = Path(handle.name)
        try:
            with handle:
                torch.save(dict(checkpoint), handle)
                handle.flush()
                os.fsync(handle.fileno())
            if temp_path.stat().st_size <= 0:
                raise RuntimeError("CenMag checkpoint temporary file is empty")
            loaded = self._read_checkpoint(temp_path, verify_manifest=False)
            if loaded["checkpoint_uuid"] != checkpoint["checkpoint_uuid"]:
                raise RuntimeError("CenMag checkpoint temporary validation mismatch")
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _manifest_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.parent / "checkpoint_manifest.json"

    def _load_manifest(self, checkpoint_path: Path,
                       required: bool) -> Dict[str, Any] | None:
        manifest_path = self._manifest_path(checkpoint_path)
        if not manifest_path.exists():
            if required:
                raise RuntimeError(
                    f"CenMag checkpoint manifest is missing: {manifest_path}")
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(
                f"CenMag checkpoint manifest is unreadable: {manifest_path}") \
                from error
        if int(manifest.get("version", -1)) != MANIFEST_VERSION:
            raise RuntimeError("Unsupported CenMag checkpoint manifest version")
        return manifest

    def _verify_manifest_hash(self, checkpoint_path: Path) -> None:
        manifest = self._load_manifest(checkpoint_path, required=True)
        entry = manifest.get("files", {}).get(checkpoint_path.name)
        if entry is None:
            raise RuntimeError(
                f"CenMag checkpoint is not listed in its manifest: "
                f"{checkpoint_path.name}")
        actual_size = checkpoint_path.stat().st_size
        actual_sha = checkpoint_file_sha256(checkpoint_path)
        if actual_size != int(entry["bytes"]) or actual_sha != entry["sha256"]:
            previous = manifest.get("previous")
            hint = (f" Previous known-good checkpoint is {previous}; select it "
                    "explicitly." if previous else "")
            raise RuntimeError(
                "CenMag checkpoint size/SHA mismatch; the file is corrupted or "
                f"truncated.{hint}")

    def _read_checkpoint(self, path: os.PathLike[str] | str,
                         verify_manifest: bool) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise ValueError(f"The file {checkpoint_path} does NOT exist")
        if checkpoint_path.stat().st_size <= 0:
            raise RuntimeError("CenMag checkpoint is empty")
        if verify_manifest:
            self._verify_manifest_hash(checkpoint_path)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f"CenMag checkpoint is unreadable or corrupted: "
                f"{checkpoint_path}") from error
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("CenMag checkpoint payload is not a mapping")
        checkpoint = dict(checkpoint)
        self._validate_checkpoint_payload(checkpoint)
        return checkpoint

    def validate_checkpoint_file(self, path: os.PathLike[str] | str,
                                 verify_manifest: bool = True) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        checkpoint = self._read_checkpoint(
            checkpoint_path, verify_manifest=verify_manifest)
        return {
            "status": "PASS",
            "path": str(checkpoint_path),
            "bytes": int(checkpoint_path.stat().st_size),
            "sha256": checkpoint_file_sha256(checkpoint_path),
            "version": int(checkpoint["version"]),
            "server_round": int(checkpoint["server_round"]),
            "last_completed_round": int(checkpoint["last_completed_round"]),
            "checkpoint_uuid": checkpoint["checkpoint_uuid"],
        }

    def save_model(self, path, cur_round=-1):
        if self.model is None:
            raise RuntimeError("Cannot save CenMag checkpoint without a model")
        save_started = time.perf_counter()
        checkpoint = self._build_checkpoint(cur_round=int(cur_round))
        checkpoint_dir = Path(path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        server_round = int(checkpoint["server_round"])
        round_path = checkpoint_dir / f"round_{server_round:06d}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        manifest_path = checkpoint_dir / "checkpoint_manifest.json"

        old_manifest = self._load_manifest(round_path, required=False) or {
            "version": MANIFEST_VERSION,
            "method": METHOD_NAME,
            "files": {},
            "latest": None,
            "previous": None,
        }
        old_latest = old_manifest.get("latest")
        self._atomic_torch_save(round_path, checkpoint)
        round_validation = self.validate_checkpoint_file(
            round_path, verify_manifest=False)
        round_bytes = round_path.read_bytes()
        self._atomic_write_bytes(latest_path, round_bytes)
        latest_validation = self.validate_checkpoint_file(
            latest_path, verify_manifest=False)
        if latest_validation["sha256"] != round_validation["sha256"]:
            raise RuntimeError("CenMag latest checkpoint copy SHA mismatch")

        previous = old_manifest.get("previous")
        if old_latest and old_latest != round_path.name:
            previous = old_latest
        entry = {
            "bytes": round_validation["bytes"],
            "sha256": round_validation["sha256"],
            "server_round": server_round,
            "last_completed_round": server_round - 1,
            "checkpoint_uuid": checkpoint["checkpoint_uuid"],
            "validated": True,
        }
        old_files = dict(old_manifest.get("files", {}))
        files = {}
        if previous and previous in old_files:
            files[previous] = dict(old_files[previous])
        files[round_path.name] = dict(entry)
        files[latest_path.name] = dict(entry)
        manifest = {
            "version": MANIFEST_VERSION,
            "method": METHOD_NAME,
            "latest": round_path.name,
            "previous": previous,
            "files": files,
        }
        self._atomic_write_json(manifest_path, manifest)
        self._verify_manifest_hash(round_path)
        self._verify_manifest_hash(latest_path)
        # Prune only after the new numbered file, latest copy, and manifest
        # have all passed validation. Orphaned older files from an interrupted
        # cleanup are harmless and will be removed by the next successful save.
        retained_numbered = {
            name for name in (round_path.name, previous) if name
        }
        for candidate in checkpoint_dir.glob("round_*.pt"):
            if candidate.name not in retained_numbered:
                candidate.unlink()
        self.last_saved_path = str(round_path)
        io_path = checkpoint_dir / "checkpoint_io.json"
        io_payload = {
            "server_round": server_round,
            "last_completed_round": server_round - 1,
            "checkpoint": round_path.name,
            "bytes": round_validation["bytes"],
            "sha256": round_validation["sha256"],
            "write_validate_seconds": time.perf_counter() - save_started,
            "latest": latest_path.name,
            "previous": previous,
        }
        self._atomic_write_json(io_path, io_payload)
        round_validation["write_validate_seconds"] = io_payload[
            "write_validate_seconds"]
        return round_validation

    def load_model(self, path):
        if self.model is None:
            raise RuntimeError("Cannot load CenMag checkpoint without a model")
        checkpoint = self._read_checkpoint(path, verify_manifest=True)

        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.cenmag_state.load_exported_state(checkpoint["cenmag_state"])
        self.cenmag_state.load_global_a(self.model)
        self.cenmag_state.load_global_magnitude(self.model)
        self.cenmag_state.load_neutral_local_magnitude(self.model)
        self.cenmag_state.load_resume_local_state(self.model)
        self.cenmag_state.assert_detached_from_live_model(self.model)

        live_global_a = self.cenmag_state.snapshot(
            self.model, self.cenmag_state.keys.lora_a, fp32=False)
        self._assert_equal_tensor_dict(
            live_global_a, self.cenmag_state.global_a,
            "restored live global A")
        live_global_magnitude = self.cenmag_state.snapshot(
            self.model, self.cenmag_state.keys.magnitude, fp32=True)
        live_effective_dtype = {
            key: value.dtype for key, value in
            self.cenmag_state.snapshot(
                self.model, self.cenmag_state.keys.magnitude,
                fp32=False).items()
        }
        representable_global = {
            key: value.to(dtype=live_effective_dtype[key]).float()
            for key, value in self.cenmag_state.global_magnitude.items()
        }
        self._assert_equal_tensor_dict(
            live_global_magnitude, representable_global,
            "restored live global magnitude")

        restore_rng_state(checkpoint["rng_state"])
        self.loaded_round = int(checkpoint["server_round"])
        if os.environ.get('CENMAG_TRACE_DIR'):
            from federatedscope.glue.cenmag.trace import write_loaded_trace
            write_loaded_trace(self, checkpoint)
        audit_path = os.environ.get("CENMAG_RESUME_AUDIT_PATH", "").strip()
        if audit_path:
            local_magnitude_hashes = {
                str(client_id): tensor_dict_sha256(local_magnitude)
                for client_id, local_magnitude in
                self.cenmag_state.client_local_magnitudes.items()
            }
            client_local_hashes = {
                str(client_id): tensor_dict_sha256(tensors)
                for client_id, tensors in
                self.cenmag_state.client_local_states.items()
            }
            audit = {
                "status": "PASS",
                "checkpoint_path": str(Path(path).resolve()),
                "checkpoint_sha256": checkpoint_file_sha256(path),
                "checkpoint_version": int(checkpoint["version"]),
                "server_next_round": self.loaded_round,
                "last_completed_round": int(
                    checkpoint["last_completed_round"]),
                "global_a_sha256": tensor_dict_sha256(
                    self.cenmag_state.global_a),
                "global_magnitude_sha256": tensor_dict_sha256(
                    self.cenmag_state.global_magnitude),
                "client_local_magnitude_sha256": local_magnitude_hashes,
                "client_local_state_sha256": client_local_hashes,
                "client_local_rounds": {
                    str(client_id): int(server_round)
                    for client_id, server_round in
                    self.cenmag_state.client_local_rounds.items()
                },
                "resident_local_state_sha256": tensor_dict_sha256(
                    self.cenmag_state.resume_local_state),
                "rng_restored": True,
                "training_started": False,
            }
            self._atomic_write_json(Path(audit_path), audit)
        return self.loaded_round
