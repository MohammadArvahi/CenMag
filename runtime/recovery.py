"""Verify and import local Schema 7 recovery generations."""
from pathlib import Path
import copy
import json
import os
import re
import shutil

import torch

from boundary import BoundaryAggregator, IMPLEMENTATION_ID, digest
from federatedscope.glue.cenmag.aggregator import cenmag_metadata
from federatedscope.glue.cenmag.state import CenMagParameterKeys, CenMagState
from platform_tools.artifacts import checked_generation
from platform_tools.common import RELEASE, atomic, json_hash, read, sha, sync_dir
from platform_tools.identity import compatible, make
from protocol import Protocol, summarize

GENERATION = re.compile(r'boundary_(?P<round>[0-9]{6,})_[0-9a-f-]{36}')


class Validator:
    """Check identity, the tensor contract, and the original state invariants."""

    def __init__(self, cfg, assets, environment=None):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.clear_aux_info()
        self.assets = assets
        self.environment = environment
        self.shapes = read(RELEASE / 'manifests/checkpoint_tensors.json')
        self.metadata = cenmag_metadata(cfg)
        state = object.__new__(CenMagState)
        state.client_ids = (1, 2, 3)
        state.tolerance = cfg.cenmag.centering_tolerance
        keys = sorted(self.shapes)
        state.keys = CenMagParameterKeys(
            lora_a=tuple(k for k in keys if 'lora_A' in k),
            lora_b=tuple(k for k in keys if 'lora_B' in k),
            magnitude=tuple(k for k in keys if 'lora_magnitude_vector' in k),
            classifier=tuple(k for k in keys if 'classifier' in k),
            local_magnitude=(), global_magnitude_buffers=(), frozen_native_magnitude=(),
        )
        state._state_shapes = {key: tuple(row['shape']) for key, row in self.shapes.items()}
        self.aggregator = object.__new__(BoundaryAggregator)
        self.aggregator.cenmag_state = state
        self.aggregator._validate_metadata = self.validate_metadata

    def validate_metadata(self, metadata):
        if metadata != self.metadata:
            raise RuntimeError('Checkpoint metadata differs from the training configuration')

    def identity_for(self, execution):
        return make(self.cfg, self.assets, self.environment or dict(
            torch=execution['torch'], cuda_runtime=execution['cuda'],
            gpus=execution['gpus'], packages=execution['packages'], python=execution['python'],
        ))

    def __call__(self, path, active_identity=None):
        manifest = checked_generation(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        contract = payload['boundary_contract']
        identity = contract['identity']
        expected = self.identity_for(identity['execution'])
        # The live executor can additionally check the identity of its active GPU.
        self.aggregator.runtime_identity = active_identity or expected
        if identity != expected:
            raise RuntimeError('Checkpoint source, configuration, inputs or execution identity differs')
        compatible(identity, expected)
        if set(payload['model']) != set(self.shapes):
            raise RuntimeError('Checkpoint parameter names differ')
        for key, tensor in payload['model'].items():
            row = self.shapes[key]
            if list(tensor.shape) != row['shape'] or str(tensor.dtype) != row['dtype']:
                raise RuntimeError(f'Checkpoint shape or dtype differs for {key}')
        self.aggregator._validate_checkpoint_payload(payload)
        n = payload['server_round']
        if (type(n) is not int or not 1 <= n <= self.cfg.federate.total_round_num
                or manifest['server_round'] != n or manifest['boundary_id'] != contract['boundary_id']
                or manifest.get('implementation_id') != IMPLEMENTATION_ID
                or path.parent.name != f"boundary_{n:06d}_{contract['boundary_id']}"):
            raise RuntimeError('Checkpoint generation does not match the completed round')
        return payload


def checkpoint_path(session):
    root = session / 'checkpoints'
    pointer = root / 'latest.json'
    if root.is_symlink() or pointer.is_symlink():
        raise RuntimeError('Checkpoint directory and pointer must be local regular entries')
    generations = [
        (int(match['round']), path) for path in root.glob('boundary_*')
        if (match := GENERATION.fullmatch(path.name))
    ]
    if not generations:
        raise RuntimeError('No complete checkpoint generation found')
    n = max(number for number, _ in generations)
    latest = [path for number, path in generations if number == n]
    if len(latest) != 1:
        raise RuntimeError('Ambiguous newest checkpoint generation')
    path = latest[0] / 'state.pt'
    manifest = checked_generation(path)
    if pointer.is_file():
        record = read(pointer)
        current = record['current']
        match = GENERATION.fullmatch(current) if isinstance(current, str) else None
        # A crash can occur after the generation rename and before the pointer update.
        if not match or int(match['round']) not in (n - 1, n):
            raise RuntimeError('Checkpoint pointer is inconsistent with the newest generation')
        if int(match['round']) == n and (path.parent.name != current or record['manifest'] != manifest):
            raise RuntimeError('Checkpoint pointer differs from the newest generation')
    elif pointer.exists() or n != 1:
        raise RuntimeError('Missing or invalid checkpoint pointer')
    return path


def history(session, payload):
    """Reconcile the final ledger row with its atomically committed checkpoint."""
    n = payload['server_round']
    platform = payload['boundary_contract']['runtime']['research_platform']
    rows = read(session / 'rounds_before.json', [])
    ledger = session / 'rounds.jsonl'
    if ledger.exists():
        with ledger.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n') and len(rows) == n - 1:
                        break
                    raise
                if record['completed_round'] > n:
                    raise RuntimeError('Metrics advance past the selected checkpoint')
                rows.append(summarize(record))
    if len(rows) == n - 1:
        rows.append(summarize(platform['forensic_evidence']['metrics']))
    protocol = Protocol(rows)
    if protocol.snapshot() != platform['public_protocol'] or len(rows) != n:
        raise RuntimeError('Round history does not match the checkpoint')
    return protocol


def required_rounds(payload):
    n = payload['server_round']
    rolling = read(RELEASE / 'configs/operations.json')['rolling_states']
    needed = set(range(max(1, n - rolling + 1), n + 1))
    state = payload['boundary_contract']['runtime']['research_platform']
    event = state.get('pending_event')
    if event:
        end = event['completed_round']
        needed.update(range(max(1, end - 2), end + 1))
    needed.update(p + 1 for p in state.get('pending_after', []) if p + 1 <= n)
    return needed


def recovery_window(current, payload):
    needed = required_rounds(payload)
    selected = {}
    for path in sorted(current.parent.parent.glob('boundary_*/state.pt')):
        match = GENERATION.fullmatch(path.parent.name)
        if not match or int(match['round']) not in needed:
            continue
        if (path.parent / 'manifest.json').is_symlink():
            raise RuntimeError('Checkpoint manifest must be a regular file')
        row = checked_generation(path)
        n = row['server_round']
        if n != int(match['round']) or n in selected:
            raise RuntimeError('Ambiguous or inconsistent checkpoint generation')
        selected[n] = path
    if set(selected) != needed or selected[payload['server_round']] != current:
        raise RuntimeError(f'Missing or inconsistent recovery rounds: {sorted(needed - set(selected))}')
    return dict(sorted(selected.items()))


def extend(payload, target_identity):
    """Apply the single permitted identity change while preserving the signed state."""
    contract = payload['boundary_contract']
    original = contract['identity']
    expected = copy.deepcopy(original)
    cfg = expected['scientific']['configuration']['federate']
    rounds = target_identity['scientific']['configuration']['federate']['total_round_num']
    if type(rounds) is not int or rounds <= payload['server_round']:
        raise RuntimeError('The new target must exceed the completed round')
    cfg['total_round_num'] = rounds
    expected['scientific_sha256'] = json_hash(expected['scientific'])
    compatible(expected, target_identity)
    updated = dict(contract, identity=expected)
    updated['contract_digest'] = digest({k: v for k, v in updated.items() if k != 'contract_digest'})
    return dict(payload, boundary_contract=updated)


def import_window(current, payload, out, validate, target_validate, new_target_identity=None):
    """Copy the window and revalidate its destination before starting the engine."""
    source = current.parent.parent.parent
    selected = recovery_window(current, payload)
    root = out / 'checkpoints'
    root.mkdir()
    for path in selected.values():
        original = validate(path)
        target = root / path.parent.name
        target.mkdir()
        state_file = target / 'state.pt'
        if new_target_identity is None:
            shutil.copyfile(path, state_file)
            shutil.copyfile(path.parent / 'manifest.json', target / 'manifest.json')
        else:
            converted = extend(original, new_target_identity)
            torch.save(converted, state_file)
            manifest = dict(checked_generation(path), bytes=state_file.stat().st_size, sha256=sha(state_file))
            atomic(target / 'manifest.json', manifest)
        for copied in (state_file, target / 'manifest.json'):
            with copied.open('rb') as handle:
                os.fsync(handle.fileno())
        sync_dir(target)
        restored = target_validate(state_file)
        before = original['boundary_contract']
        after = restored['boundary_contract']
        # The original validator has already recomputed each state digest.
        excluded = {'identity', 'contract_digest'}
        if digest({k: v for k, v in before.items() if k not in excluded}) != digest(
            {k: v for k, v in after.items() if k not in excluded}
        ):
            raise RuntimeError('Imported checkpoint changed training state')
    latest = root / current.parent.name / 'state.pt'
    ordered = list(selected.values())
    atomic(root / 'latest.json', dict(current=latest.parent.name,
               previous=ordered[-2].parent.name if len(ordered) > 1 else None,
               manifest=checked_generation(latest)))

    # Event recovery may require the final evaluation logits from the parent.
    history_dir = out / 'history' / 'logits'
    history_dir.mkdir(parents=True)
    for n in selected:
        pattern = f'round_{n:06d}_client_*.npz'
        for folder in (source / 'history' / 'logits', source / 'logits'):
            for path in folder.glob(pattern):
                if path.is_symlink():
                    raise RuntimeError('Evaluation output must be a regular file')
                dest = history_dir / path.name
                if dest.exists():
                    if sha(dest) != sha(path):
                        raise RuntimeError('Conflicting parent evaluation output')
                else:
                    shutil.copyfile(path, dest)
    atomic(out / 'lineage.json', dict(parent_checkpoint_sha256=sha(current),
               parent_completed_round=payload['server_round'], horizon_extended=new_target_identity is not None))
    return latest
