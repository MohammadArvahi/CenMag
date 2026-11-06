"""CenMag workers with explicit client-keyed local B/classifier state."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch

from federatedscope.core.message import Message
from federatedscope.core.workers import Client, Server
from federatedscope.glue.cenmag.aggregator import CenMagAggregator
from federatedscope.glue.cenmag.model_utils import cuda_memory_snapshot
from federatedscope.glue.cenmag.state import (
    derive_client_round_seed, tensor_dict_nbytes)


logger = logging.getLogger(__name__)


class CenMagClient(Client):
    """Load each client's local magnitude, LoRA-B, and classifier state."""

    def __init__(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None and len(args) > 5:
            model = args[5]
        super().__init__(*args, **kwargs)
        if self._cfg is None:
            self.cenmag_state = None
            return
        self.cenmag_state = getattr(model, "_cenmag_state", None)
        if self.cenmag_state is None:
            raise RuntimeError(
                "CenMagClient did not receive the server-owned CenMag state; "
                "share_local_model=True is required")

    def _prepare_personalized_message(self, message: Message,
                                      evaluation: bool = False) -> Message:
        if not isinstance(message.content, dict):
            raise RuntimeError("CenMag requires a dictionary model message")
        self.cenmag_state.assert_broadcast_global_magnitude(message.content)
        model = self.trainer.ctx.model
        self.cenmag_state.load_client_local_state(
            model, self.ID, evaluation=evaluation)
        effective = self.cenmag_state.load_effective_magnitude(model, self.ID)
        self.cenmag_state.assert_loaded(
            model, effective, f"client {self.ID} start")
        if evaluation:
            self.cenmag_state.last_evaluation_start[self.ID] = \
                self.cenmag_state.snapshot(
                    model, self.cenmag_state.keys.magnitude, fp32=True)

        # The server did communicate global magnitude.  It is removed only from
        # the baseline trainer.update call because the effective client value is
        # already loaded explicitly above.  Global A remains in the message.
        personalized = copy.copy(message)
        personalized.content = {
            key: value
            for key, value in message.content.items()
            if key not in self.cenmag_state.keys.magnitude
        }
        if set(personalized.content) != set(self.cenmag_state.keys.lora_a):
            raise RuntimeError(
                "CenMag personalized message must apply exactly global A")
        return personalized

    def callback_funcs_for_model_para(self, message: Message):
        if "ss" in message.msg_type:
            raise RuntimeError("CenMag does not support secret sharing")
        schedule = self.cenmag_state.last_rng_schedule
        if not schedule or int(schedule.get("server_round", -1)) != int(
                message.state):
            raise RuntimeError(
                "CenMag client received no matching deterministic RNG schedule")
        client_seeds = {
            int(client_id): int(seed)
            for client_id, seed in schedule["client_seeds"].items()
        }
        seed = derive_client_round_seed(
            int(self._cfg.seed), int(message.state), int(self.ID))
        if client_seeds.get(int(self.ID)) != seed:
            raise RuntimeError(
                f"CenMag RNG schedule mismatch for client {self.ID}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        personalized = self._prepare_personalized_message(message)
        if os.environ.get('CENMAG_TRACE_DIR'):
            self.trainer.ctx._cenmag_trace_client_id = int(self.ID)
            self.trainer.ctx._cenmag_trace_round = int(message.state)
        if self.cenmag_state.validation_mode:
            print(
                f"CENMAG_REAL_CLIENT_TRAINING_START round={message.state} "
                f"client={self.ID}",
                flush=True,
            )
        before = cuda_memory_snapshot(
            f"round_{message.state}_client_{self.ID}_before_update")
        result = super().callback_funcs_for_model_para(personalized)
        self.cenmag_state.capture_client_local_state(
            self.trainer.ctx.model, self.ID, int(message.state) + 1)
        after = cuda_memory_snapshot(
            f"round_{message.state}_client_{self.ID}_after_update")
        self.cenmag_state.last_memory.setdefault("clients", {})[self.ID] = {
            "before": before,
            "after": after,
        }
        completed = schedule["completed_clients"]
        if int(self.ID) in completed:
            raise RuntimeError(
                f"CenMag client {self.ID} trained twice in round {message.state}")
        completed.append(int(self.ID))
        return result

    def callback_funcs_for_evaluate(self, message: Message):
        return super().callback_funcs_for_evaluate(
            self._prepare_personalized_message(message, evaluation=True))


class CenMagServer(Server):
    """Aggregate effective magnitudes and rebase client-local components."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._cfg is None:
            self.cenmag_state = None
            return
        if not isinstance(self.aggregator, CenMagAggregator):
            raise RuntimeError("CenMagServer requires CenMagAggregator")
        self.cenmag_state = self.aggregator.cenmag_state
        if self.aggregator.loaded_round is not None:
            self.state = int(self.aggregator.loaded_round)
            logger.info("CenMag restored next server round %d", self.state)
        self._assert_supported_runtime()

    def _assert_supported_runtime(self) -> None:
        cfg = self._cfg
        failures = []
        if cfg.federate.mode != "standalone":
            failures.append("standalone mode is required")
        if int(cfg.federate.client_num) != 3:
            failures.append("exactly 3 clients are required")
        if int(cfg.federate.sample_client_num) != int(cfg.federate.client_num):
            failures.append("full participation is required")
        if not cfg.federate.share_local_model:
            failures.append("share_local_model=True is required")
        if cfg.federate.online_aggr:
            failures.append("online aggregation is unsupported")
        if cfg.federate.ignore_weight:
            failures.append(
                "ignore_weight=True is unsupported; CenMag uses the baseline "
                "sample-size weights for A and magnitude")
        if cfg.asyn.use:
            failures.append("asynchronous aggregation is unsupported")
        if int(cfg.model.model_num_per_trainer) != 1:
            failures.append("one model per trainer is required")
        if cfg.federate.unseen_clients_rate != 0:
            failures.append("unseen clients are unsupported")
        if cfg.federate.use_ss:
            failures.append("secret sharing is unsupported")
        if cfg.quantization.method != "none":
            failures.append("FederatedScope message quantization is unsupported")
        if failures:
            raise RuntimeError("Invalid CenMag runtime: " + "; ".join(failures))

    def _global_payload(self) -> Dict[str, torch.Tensor]:
        self.cenmag_state.load_global_a(self.model)
        self.cenmag_state.load_global_magnitude(self.model)
        live = self.model.state_dict()
        live_effective = self.cenmag_state.snapshot(
            self.model, self.cenmag_state.keys.magnitude, fp32=False)
        payload = {
            key: self.cenmag_state.global_a[key].to(
                device=live[key].device, dtype=live[key].dtype).clone()
            for key in self.cenmag_state.keys.lora_a
        }
        for key in self.cenmag_state.keys.magnitude:
            payload[key] = self.cenmag_state.global_magnitude[key].to(
                device=live_effective[key].device,
                dtype=live_effective[key].dtype).clone()
        self.cenmag_state.validate_communicated_state(
            payload, "CenMag server broadcast")
        return payload

    def broadcast_model_para(self,
                             msg_type="model_para",
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if msg_type not in ["model_para", "evaluate"]:
            raise RuntimeError(
                f"CenMag only supports model/evaluation broadcasts, got {msg_type}")
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, "unseen")
        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == "model_para":
                self.sampler.change_state(receiver, "working")

        receiver = sorted(int(client_id) for client_id in receiver)
        if msg_type == "model_para" and set(receiver) != set(
                self.cenmag_state.client_ids):
            raise RuntimeError(
                "CenMag rejected partial participation; selected clients="
                f"{list(receiver)}")

        if msg_type == "model_para":
            base_seed = int(self._cfg.seed)
            self.cenmag_state.last_rng_schedule = {
                "base_seed": base_seed,
                "server_round": int(self.state),
                "client_order": list(receiver),
                "client_seeds": {
                    client_id: derive_client_round_seed(
                        base_seed, int(self.state), client_id)
                    for client_id in receiver
                },
                "completed_clients": [],
            }

        if msg_type == "model_para" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.cenmag_state.last_memory["round_start"] = \
                cuda_memory_snapshot(f"round_{self.state}_start")
        if msg_type == "model_para":
            self.cenmag_state.last_memory["round_started_at_unix"] = time.time()

        payload = self._global_payload()
        a_payload = {
            key: payload[key] for key in self.cenmag_state.keys.lora_a
        }
        magnitude_payload = {
            key: payload[key] for key in self.cenmag_state.keys.magnitude
        }
        recipient_count = len(receiver)
        direction_bytes = tensor_dict_nbytes(a_payload) * recipient_count
        magnitude_bytes = tensor_dict_nbytes(
            magnitude_payload) * recipient_count
        server_to_clients = self.cenmag_state.last_communication.setdefault(
            "server_to_clients", {})
        server_to_clients[msg_type] = {
            "message_type": msg_type,
            "recipient_count": recipient_count,
            "tensor_keys": sorted(payload),
            "a_tensor_bytes": direction_bytes,
            "magnitude_tensor_bytes": magnitude_bytes,
            "total_tensor_bytes": direction_bytes + magnitude_bytes,
        }

        round_number = self.state - 1 if msg_type == "evaluate" else self.state
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(round_number, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=payload))
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, "seen")

    def _perform_federated_aggregation(self):
        train_buffer = self.msg_buffer["train"][self.state]
        expected_clients = set(self.cenmag_state.client_ids)
        if set(train_buffer) != expected_clients:
            raise RuntimeError(
                "CenMag aggregation requires all configured clients; "
                f"expected={sorted(expected_clients)}, "
                f"received={sorted(train_buffer)}")

        client_magnitudes = {}
        sample_sizes = {}
        upload_a_bytes = 0
        upload_magnitude_bytes = 0
        for client_id in self.cenmag_state.client_ids:
            sample_size, state = train_buffer[client_id]
            if not isinstance(state, dict):
                raise RuntimeError("CenMag requires one dictionary model upload")
            self.cenmag_state.validate_communicated_state(
                state, f"client {client_id} upload")
            sample_sizes[client_id] = int(sample_size)
            client_magnitudes[client_id] = {
                key: state[key].detach().cpu().clone()
                for key in self.cenmag_state.keys.magnitude
            }
            upload_a_bytes += tensor_dict_nbytes({
                key: state[key] for key in self.cenmag_state.keys.lora_a
            })
            upload_magnitude_bytes += tensor_dict_nbytes({
                key: state[key] for key in self.cenmag_state.keys.magnitude
            })

        # Preserve the existing FedSA/FedAvg implementation for A only. The
        # effective client magnitudes are deliberately kept out of that path
        # and aggregated once, explicitly in CPU FP32 below. Clone the A
        # tensors because ClientsAvgAggregator mutates the first upload while
        # constructing its result.
        original_train_buffer = train_buffer
        a_only_buffer = {
            client_id: (
                original_train_buffer[client_id][0],
                {
                    key: original_train_buffer[client_id][1][key].detach().clone()
                    for key in self.cenmag_state.keys.lora_a
                },
            )
            for client_id in self.cenmag_state.client_ids
        }
        self.msg_buffer["train"][self.state] = a_only_buffer
        try:
            aggregated_num = super()._perform_federated_aggregation()
        finally:
            self.msg_buffer["train"][self.state] = original_train_buffer

        self.cenmag_state.capture_global_a(self.model)
        stats = self.cenmag_state.update_from_client_magnitudes(
            client_magnitudes, sample_sizes)
        self.cenmag_state.load_global_magnitude(self.model)
        magnitude_diagnostics = {
            "round": int(self.state),
            "p_m": int(stats["p_m"]),
            "global_magnitude_l2": float(torch.sqrt(sum(
                value.double().pow(2).sum()
                for value in self.cenmag_state.global_magnitude.values()
            )).item()),
            "client_magnitude_dispersion": float(
                stats["client_magnitude_dispersion"]),
            "max_rebasing_error": float(
                self.cenmag_state.last_aggregation_validation[
                    "max_abs_effective_reconstruction_error_after_centering"]),
            "weighted_local_mean_l2": float(
                stats["weighted_residual_mean_l2"]),
            "clients": {},
        }
        for client_id in self.cenmag_state.client_ids:
            local_values = self.cenmag_state.client_local_magnitudes[client_id]
            effective_values = self.cenmag_state.effective_magnitude(client_id)
            magnitude_diagnostics["clients"][str(client_id)] = {
                "sample_count": int(sample_sizes[client_id]),
                "aggregation_weight": float(
                    self.cenmag_state.last_aggregation_weights[client_id]),
                "local_magnitude_l2": float(torch.sqrt(sum(
                    value.double().pow(2).sum()
                    for value in local_values.values())).item()),
                "effective_magnitude_l2": float(torch.sqrt(sum(
                    value.double().pow(2).sum()
                    for value in effective_values.values())).item()),
            }
        self.cenmag_state.last_aggregation_validation[
            "magnitude_diagnostics"] = copy.deepcopy(magnitude_diagnostics)
        expected_local_round = int(self.state) + 1
        local_rounds = {
            int(client_id): int(server_round)
            for client_id, server_round in
            self.cenmag_state.client_local_rounds.items()
        }
        if any(server_round != expected_local_round
               for server_round in local_rounds.values()):
            raise RuntimeError(
                "CenMag client-local states are not aligned with the completed "
                f"round boundary {expected_local_round}: {local_rounds}")
        # The standalone runner can execute the next broadcast on the shared
        # model before evaluation responses trigger a checkpoint.  Capture the
        # non-communicated resident state now, while it is aligned with the
        # newly aggregated server state, so a later save cannot serialize
        # speculative next-round mutations from the live model.
        self.cenmag_state.capture_resume_local_state(self.model,
                                                     self.state + 1)

        self.cenmag_state.last_communication["clients_to_server"] = {
            "client_count": len(self.cenmag_state.client_ids),
            "tensor_keys": sorted(
                self.cenmag_state.keys.lora_a +
                self.cenmag_state.keys.magnitude),
            "a_tensor_bytes": upload_a_bytes,
            "magnitude_tensor_bytes": upload_magnitude_bytes,
            "total_tensor_bytes": upload_a_bytes + upload_magnitude_bytes,
            "client_order": list(self.cenmag_state.last_rng_schedule.get(
                "completed_clients", [])),
        }
        if self.cenmag_state.last_rng_schedule.get(
                "completed_clients") != list(self.cenmag_state.client_ids):
            raise RuntimeError(
                "CenMag clients did not complete in deterministic ascending order")
        self.cenmag_state.last_communication["aggregation_calls"] = int(
            self.cenmag_state.last_communication.get(
                "aggregation_calls", 0)) + 1
        round_memory = cuda_memory_snapshot(
            f"round_{self.state}_after_aggregation")
        round_memory["round"] = int(self.state)
        started_at = self.cenmag_state.last_memory.get(
            "round_started_at_unix")
        if started_at is not None:
            round_memory["duration_seconds"] = max(
                0.0, time.time() - float(started_at))
        history = self.cenmag_state.last_memory.setdefault("round_history", [])
        history.append(round_memory)
        del history[:-10]
        logger.info(
            "CenMag round %d centered residuals: max_abs=%g mean_abs=%g; "
            "upload A=%d B, magnitude=%d B",
            self.state, stats["max_abs_weighted_residual_mean"],
            stats["mean_abs_weighted_residual_mean"], upload_a_bytes,
            upload_magnitude_bytes)
        logger.info("CENMAG_MAGNITUDE_DIAGNOSTICS %s",
                    json.dumps(magnitude_diagnostics, sort_keys=True))
        return aggregated_num
