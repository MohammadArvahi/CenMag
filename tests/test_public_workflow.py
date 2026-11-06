"""CPU checks for the published configuration, checkpoint, and protocol paths."""
import copy
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor'), str(ROOT / 'runtime')]

import torch
import canonical_algorithm_fixture as fixture
from boundary import digest
from federatedscope.glue.cenmag.state import derive_client_round_seed
from platform_tools.artifacts import Retention, event_package
from platform_tools.common import atomic, read, sha
from platform_tools.identity import make
from protocol import Protocol, summarize
from public_config import configuration, input_assets
from public_run import read_source
from recovery import Validator, checkpoint_path, extend, import_window, recovery_window

CLIENTS = [1, 2, 3]


def metric(n, dominant=0.5, invalid=False, accuracy=0.91):
    clients = [dict(client=c, accuracy=accuracy, dominant_prediction_fraction=dominant if c == 1 else 0.5,
                    true_class_counts=[6, 3, 1], nonfinite_logits=int(invalid and c == 1))
               for c in CLIENTS]
    return dict(completed_round=n, weighted_validation_accuracy=accuracy, clients=clients)


def synthetic_boundary(validator, n, protocol, out):
    """Use the actual saver and model tensor contract with generated CPU tensors."""
    aggregator = validator.aggregator
    aggregator._frozen = None
    aggregator._defer_save = False
    state = aggregator.cenmag_state
    model = {key: torch.zeros(row['shape'], dtype=getattr(torch, row['dtype'].split('.')[-1]))
             for key, row in validator.shapes.items()}
    shared = {key: model[key] for key in state.keys.lora_a}
    center = {key: model[key] for key in state.keys.magnitude}
    residuals = {c: {key: torch.full_like(value, (c - 2) * 0.01) for key, value in center.items()} for c in CLIENTS}
    local = {key: model[key] for key in state.keys.lora_b + state.keys.classifier}
    local_states = {c: {key: torch.full_like(value, c - 1) for key, value in local.items()} for c in CLIENTS}
    schedule = dict(base_seed=1, server_round=n - 1, client_order=CLIENTS, completed_clients=CLIENTS,
                    client_seeds={c: derive_client_round_seed(1, n - 1, c) for c in CLIENTS})
    state.load_exported_state(dict(version=state.VERSION, client_ids=CLIENTS, global_a=shared,
                                   global_magnitude=center, client_local_magnitudes=residuals,
                                   client_local_states=local_states, client_local_rounds=dict.fromkeys(CLIENTS, n),
                                   resume_local_state=local, resume_round=n, last_rng_schedule=schedule,
                                   last_aggregation_weights=dict.fromkeys(CLIENTS, 1 / 3)))
    aggregator._canonical_model_state = lambda: model
    runtime = dict(next_round=n, eval_completed_clients=CLIENTS, server_timestamp=0.,
                   history_results={}, best_results={}, sampler_client_state=[0] * 4,
                   processed_examples=dict.fromkeys(CLIENTS, n * 128 * 10),
                   client_states=dict.fromkeys(CLIENTS, n - 1), optimizer_policy='stateless SGD',
                   scheduler=None, dataloader_policy='per-round seed', early_stop_policy='disabled patience=0',
                   queue_empty=True, server_cache_empty=True,
                   research_platform=dict(public_protocol=protocol.snapshot(),
                                          forensic_evidence=dict(metrics=metric(n))))
    aggregator.capture_complete_boundary(n, runtime=runtime, barrier_complete=True)
    return Path(aggregator.save_model(out / 'checkpoints/requested.pt', n)['path'])


class PublicWorkflowTests(unittest.TestCase):
    def test_reference_cpu_fixtures(self):
        for name, function in vars(fixture).items():
            if name.startswith('test_'):
                with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                    function(Path(directory)) if inspect.signature(function).parameters else function()

    def test_arbitrary_targets_preserve_science(self):
        assets = input_assets(ROOT / 'inputs')
        signatures = []
        for n in (1, 37, 1703, 1_000_047):
            with self.subTest(rounds=n):
                _, effective = configuration(n, ROOT / 'runs/check', assets)
                self.assertEqual(effective['federate'].pop('total_round_num'), n)
                signatures.append(effective)
        self.assertTrue(all(config == signatures[0] for config in signatures))
        for invalid in (0, -7, True, 1.5):
            with self.assertRaises(ValueError):
                configuration(invalid, ROOT / 'runs/check', assets)

    def test_safety_checks_apply_without_a_stage_boundary(self):
        protocol = Protocol()
        self.assertEqual(protocol.observe(summarize(metric(1, invalid=True))), 'nonfinite_metrics')
        protocol = Protocol()
        for n in range(1, 29):
            protocol.observe(summarize(metric(n, dominant=1 if n >= 26 else 0.5)))
        self.assertEqual(protocol.stop, 'validation_instability')
        protocol = Protocol()
        for n in range(1, 36):
            protocol.observe(summarize(metric(n, dominant=1, accuracy=0.62)))
        self.assertEqual(protocol.stop, 'persistent_single_class')
        with self.assertRaisesRegex(ValueError, 'nonterminal'):
            protocol.observe(summarize(metric(36)))

    def test_retention_and_extending_the_latest_checkpoint(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, extended, resumed = [root / name for name in ('started', 'extended', 'resumed')]
            for folder in (parent, extended, resumed):
                folder.mkdir()
            assets = input_assets(root)
            identity_assets = {'test': 'synthetic'}
            execution = dict(torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                             gpus=[], packages={}, python=sys.version)
            initial_cfg, _ = configuration(14, parent, assets)
            next_cfg, _ = configuration(29, extended, assets)
            before, after = Validator(initial_cfg, identity_assets), Validator(next_cfg, identity_assets)
            before.aggregator.cfg = initial_cfg
            before.aggregator.runtime_identity = make(before.cfg, identity_assets, execution)
            target = make(after.cfg, identity_assets, execution)

            retention = Retention(parent, 10, 16 * 1024**3, 0, 2)
            protocol = Protocol()
            for n in range(1, 15):
                protocol.observe(summarize(metric(n)))
                latest = synthetic_boundary(before, n, protocol, parent)
                self.assertEqual(latest, checkpoint_path(parent))
                if n == 1:
                    atomic(parent / 'checkpoints/best.json', {'current': latest.parent.name})
                if 2 <= n <= 5:
                    event_package(parent, dict(completed_round=n), [latest], {})
                if n == 13:
                    previous_pointer = read(parent / 'checkpoints/latest.json')
                retention.prune()
            self.assertEqual(len(list((parent / 'checkpoints').glob('boundary_*'))), 11)
            self.assertEqual(len(list((parent / 'events').glob('round_*'))), 2)

            # Simulate an interrupted pointer write and an incomplete last ledger line.
            atomic(parent / 'checkpoints/latest.json', previous_pointer)
            atomic(parent / 'session.json', {'target': 14})
            atomic(parent / 'rounds_before.json', protocol.rows[:-1])
            (parent / 'rounds.jsonl').write_text('{"completed_round":')
            source_target, _, checkpoint, payload, verified = read_source(parent, assets, identity_assets)
            self.assertEqual(source_target, 14)
            self.assertEqual(checkpoint, latest)
            self.assertEqual(verified.snapshot(), protocol.snapshot())
            self.assertEqual(list(recovery_window(latest, payload)), list(range(5, 15)))

            parent_hash = sha(latest)
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                after(latest)
            wrong = copy.deepcopy(target)
            wrong['scientific']['configuration']['grad']['grad_clip'] = 1.
            with self.assertRaises(RuntimeError):
                extend(payload, wrong)
            wrong_assets = Validator(initial_cfg, {'test': 'different'})
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                wrong_assets(latest)
            imported = import_window(latest, payload, extended, before, after, target)
            result = after(imported)
            self.assertEqual(result['server_round'], 14)
            self.assertEqual(digest({k: v for k, v in payload.items() if k != 'boundary_contract'}),
                             digest({k: v for k, v in result.items() if k != 'boundary_contract'}))
            self.assertEqual(sha(latest), parent_hash)
            self.assertEqual(result['boundary_contract']['identity'], target)
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                before(imported)

            again = import_window(imported, result, resumed, after, after)
            self.assertEqual(sha(again), sha(imported))
            with again.open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'Corrupted checkpoint bytes'):
                checkpoint_path(resumed)
            recovery_window(imported, result)[6].unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing or inconsistent recovery rounds'):
                recovery_window(imported, result)
            altered = list(protocol.rows[:-1])
            altered[0] = dict(altered[0], accuracy=0.90)
            atomic(parent / 'rounds_before.json', altered)
            with self.assertRaisesRegex(RuntimeError, 'Round history does not match'):
                read_source(parent, assets, identity_assets)

    def test_checkpoint_round_names_can_exceed_six_digits(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rounds = 1_000_047
            cfg, _ = configuration(rounds, out, input_assets(out))
            validator = Validator(cfg, {})
            validator.aggregator.cfg = cfg
            validator.aggregator.runtime_identity = make(validator.cfg, {}, dict(
                torch=str(torch.__version__), cuda_runtime=torch.version.cuda, gpus=[], packages={}, python=sys.version,
            ))
            state = synthetic_boundary(validator, rounds, Protocol(), out)
            self.assertEqual(checkpoint_path(out), state)
            self.assertEqual(validator(state)['server_round'], rounds)


if __name__ == '__main__':
    unittest.main()
