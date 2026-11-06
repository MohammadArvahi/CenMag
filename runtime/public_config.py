"""Load the fixed MNLI training settings and verify the inputs."""
from pathlib import Path
import copy
import hashlib
import json
import platform
import re

from platform_tools.common import RELEASE, read, sha, verify_inventory

SCIENCE_SHA256 = '9fb412314676b0165879cfe02f11dbfc70742f3961c59a85db7b6e75cb45a7ad'


def configuration(rounds, out, assets, resume=''):
    from execute import make_cfg
    from platform_tools.identity import normalize_config

    if type(rounds) is not int or rounds <= 0:
        raise ValueError('The training target must be a positive integer')
    cfg = make_cfg(rounds, out, assets, resume)
    cfg.freeze(inform=False, save=False)
    clean = copy.deepcopy(cfg)
    clean.clear_aux_info()
    effective = normalize_config(clean)
    science = copy.deepcopy(effective)
    del science['federate']['total_round_num']
    fingerprint = hashlib.sha256(json.dumps(science, sort_keys=True, default=str).encode()).hexdigest()
    if fingerprint != SCIENCE_SHA256:
        raise RuntimeError('Effective MNLI configuration differs from the locked scientific settings')
    return cfg, effective


def input_assets(root):
    return {name: str(root / name) for name in ('model', 'dataset', 'metric', 'cache')}


def verify_inputs(assets):
    plan = read(RELEASE / 'manifests/direct_download_plan.json')
    result = {}
    for name in ('model', 'dataset', 'metric'):
        records = {
            Path(item['relative_path']).relative_to(name).as_posix():
                {'bytes': item['bytes'], 'sha256': item['sha256']}
            for item in plan['download_records'] if item['group'] == name
        }
        verify_inventory(assets[name], records)
        result[name] = dict(revision=plan['revisions'][name], files=records)
    cache = Path(assets['cache'])
    cache_records = {
        name: dict(bytes=(cache / name).stat().st_size, **record) if 'bytes' not in record else record
        for name, record in plan['reference_cache'].items()
    }
    verify_inventory(cache, cache_records)
    partition = read(cache / 'partition.json')
    if partition != plan['reference_partition']:
        raise RuntimeError('MNLI partition differs from the reference')
    descriptor = dict(
        dataset='mnli', labels=['entailment', 'neutral', 'contradiction'], num_labels=3,
        source=dict(repository='nyu-mll/glue', configuration='mnli', revision=plan['revisions']['dataset']),
        partition=partition, partition_sha256=sha(cache / 'partition.json'),
        cache=cache_records,
        tokenizer=dict(backbone='FacebookAI/roberta-large', maximum_length=128,
                       input_fields=['premise', 'hypothesis']),
        evaluation='official labeled validation; sample-weighted accuracy; no hidden test',
    )
    result.update(cache=cache_records, dataset_contract=descriptor)
    return result


def dependency_versions():
    import importlib.metadata
    import torch

    lock = (RELEASE / 'configs/local_reference.lock').read_text()
    required = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)', lock, flags=re.M))
    actual = {d.metadata['Name'].lower().replace('_', '-'): d.version
              for d in importlib.metadata.distributions()}
    mismatches = {name: (expected, actual.get(name.lower().replace('_', '-')))
                  for name, expected in required.items()
                  if actual.get(name.lower().replace('_', '-')) != expected}
    if (platform.python_version() != '3.12.14' or torch.__version__ != '2.10.0+cu128'
            or torch.version.cuda != '12.8' or len(required) != 77 or mismatches):
        raise RuntimeError(f'Use the locked CPython and CUDA environment. Package mismatches: {mismatches}')
    return required
