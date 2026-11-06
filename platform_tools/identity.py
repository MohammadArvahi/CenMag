"""Scientific equality is mandatory; cross-device replay is explicitly recorded."""
from .common import RELEASE, sha, json_hash


def normalize_config(cfg):
    from boundary import safe_state
    result = safe_state(cfg)
    result.pop('outdir', None)
    for node, fields in [('federate', ['save_to', 'restore_from', 'save_freq']),
                         ('data', ['root']), ('llm.cache', ['model']),
                         ('offline', ['model_dir', 'dataset_dir', 'metric_dir'])]:
        d = result
        for part in node.split('.'):
            d = d[part]
        for field in fields:
            d.pop(field, None)
    return result


def make(cfg, assets, environment):
    release = sha(RELEASE / 'RELEASE_MANIFEST.json')
    scientific = dict(algorithm='cenmag', configuration=normalize_config(cfg),
                      config_sha256=sha(RELEASE / 'configs/mnli.json'),
                      release_sha256=release, assets=assets, schema=7,
                      boundary_semantics='N completed; next zero-based training index N')
    execution = dict(torch=environment['torch'], cuda=environment['cuda_runtime'],
                     gpus=environment['gpus'], packages=environment['packages'],
                     python=environment['python'], release_manifest_sha256=release)
    return dict(scientific=scientific, scientific_sha256=json_hash(scientific),
                execution=execution, execution_sha256=json_hash(execution))


def compatible(saved, active, replay=False):
    for identity in [saved, active]:
        for kind in ['scientific', 'execution']:
            if kind + '_sha256' in identity and identity[kind + '_sha256'] != json_hash(identity[kind]):
                raise RuntimeError('Identity digest is inconsistent: ' + kind)
    if saved.get('scientific') != active.get('scientific'):
        raise RuntimeError('Scientific source/configuration/asset identity mismatch')
    if saved.get('execution') != active.get('execution') and not replay:
        raise RuntimeError('Execution environment differs; only explicit forensic replay may cross devices')
    return dict(scientific_equal=True, execution_equal=saved['execution'] == active['execution'],
                bitwise_equivalence_claim=False, replay=bool(replay))
