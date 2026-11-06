"""Read-only offline model/dataset inputs; all cache/output writes are separate."""
from pathlib import Path
import os

LABEL_NAMES = ['entailment', 'neutral', 'contradiction']
SPLITS = ('train', 'validation_matched', 'validation_mismatched', 'test_matched', 'test_mismatched')

def enabled(config):
    return bool(getattr(getattr(config, 'offline', None), 'enabled', False))

def require_assets(model_dir, dataset_dir, metric_dir):
    model, data, metric = Path(model_dir), Path(dataset_dir), Path(metric_dir)
    required = [model / n for n in ['config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']]
    if not (model / 'model.safetensors').is_file() and not (model / 'pytorch_model.bin').is_file():
        required.append(model / 'model.safetensors')
    required += [data / 'mnli' / f'{s}-00000-of-00001.parquet' for s in SPLITS]
    # The retained dataset snapshot stores its shards in a nested mnli directory.
    if not (data / 'mnli').is_dir():
        required = [p if p.parent != data / 'mnli' else data / p.name for p in required]
    required.append(metric / 'glue.py')
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing offline CenMag assets: ' + ', '.join(missing))
    return {'model_dir': str(model.resolve()), 'dataset_dir': str(data.resolve()),
            'metric_dir': str(metric.resolve())}

def configure_environment(cache_dir):
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ[key] = '1'
    os.environ['HF_HOME'] = str(cache / 'hf')
    os.environ['HF_DATASETS_CACHE'] = str(cache / 'datasets')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_local_mnli(config):
    from federatedscope.glue.cenmag.dataset_compat import enable_numpy2_dataset_compat
    enable_numpy2_dataset_compat()
    from datasets import ClassLabel, load_dataset
    directory = Path(config.offline.dataset_dir)
    if (directory / 'mnli').is_dir():
        directory /= 'mnli'
    paths = {s: str(directory / f'{s}-00000-of-00001.parquet') for s in SPLITS}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing offline MNLI shard: {path}')
    dataset = load_dataset('parquet', data_files=paths, cache_dir=config.data.root)
    label = dataset['train'].features['label']
    if isinstance(label, ClassLabel):
        if label.names != LABEL_NAMES:
            raise RuntimeError(f'Offline MNLI label mapping mismatch: {label.names}')
    else:
        # Pinned Parquet preserves integer IDs; attach official GLUE label names
        # without remapping any example. Both negative-label test shards remain
        # untouched; the historical protocol does not evaluate official test.
        for split in ['train', 'validation_matched', 'validation_mismatched']:
            dataset[split] = dataset[split].cast_column('label', ClassLabel(names=LABEL_NAMES))
    return dataset

def load_local_metric(config, cache_dir=None):
    # The retained metric script subclasses evaluate.Metric, so the legacy
    # datasets.load_metric API cannot construct it. The GLUE accuracy formula
    # and pinned script bytes are preserved.
    from federatedscope.glue.cenmag.dataset_compat import enable_numpy2_dataset_compat
    enable_numpy2_dataset_compat()
    import evaluate
    path = Path(config.offline.metric_dir) / 'glue.py'
    if not path.is_file():
        raise FileNotFoundError(f'Missing offline GLUE metric: {path}')
    if cache_dir is None:
        cache_dir = str(Path(config.data.root).parent / 'metrics')
    return evaluate.load(str(path), config.data.type.split('@')[0], cache_dir=cache_dir)
