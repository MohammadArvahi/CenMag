"""Stage measurements, including CPU RSS when CUDA is unavailable."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

def snapshot(stage, model=None, state=None):
    from federatedscope.glue.cenmag.model_utils import cuda_memory_snapshot
    result = cuda_memory_snapshot(stage)
    try:
        import psutil
        result['cpu_rss_bytes'] = psutil.Process().memory_info().rss
        result['cpu_available_bytes'] = psutil.virtual_memory().available
    except ImportError:
        result['cpu_rss_bytes'] = None
    if model is not None:
        result['model_parameter_storage_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    if state is not None:
        from federatedscope.glue.cenmag.state import tensor_dict_nbytes
        result['client_private_snapshot_bytes'] = sum(tensor_dict_nbytes(t) for t in state.client_local_states.values())
        result['client_residual_snapshot_bytes'] = sum(tensor_dict_nbytes(t) for t in state.client_local_magnitudes.values())
    return result

@contextmanager
def measure(stage, output, model=None, state=None):
    started = time.perf_counter()
    before = snapshot(stage + '_before', model, state)
    try:
        yield
    finally:
        after = snapshot(stage + '_after', model, state)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'stage': stage, 'seconds': time.perf_counter() - started,
                                     'before': before, 'after': after}) + '\n')
