"""Opt-in research validation traces, with no RNG or parameter mutations."""
import os
from pathlib import Path
import torch

def _directory():
    value = os.environ.get('CENMAG_TRACE_DIR', '')
    if not value:
        return None
    root = Path(__file__).resolve().parents[3]
    directory = Path(value).resolve()
    if root not in directory.parents:
        raise ValueError('CenMag validation traces must stay inside paper_code')
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    return value

def write_trainer_trace(ctx, stage):
    directory = _directory()
    if directory is None:
        return
    from federatedscope.glue.cenmag.aggregator import capture_rng_state
    from federatedscope.glue.cenmag.magnitude import binding_modules
    client = int(getattr(ctx, '_cenmag_trace_client_id', -1))
    round_number = int(getattr(ctx, '_cenmag_trace_round', -1))
    payload = {
        'round': round_number, 'client': client, 'stage': stage,
        'model': _clone(ctx.model.state_dict()),
        'global_magnitude': {k: _clone(m.global_magnitude) for k, m in binding_modules(ctx.model).items()},
        'gradients': {k: _clone(p.grad) for k, p in ctx.model.named_parameters() if p.requires_grad},
        'optimizer': _clone(ctx.optimizer.state_dict()),
        'scheduler': _clone(ctx.scheduler.state_dict()) if ctx.scheduler is not None else None,
        'rng': capture_rng_state(),
    }
    torch.save(payload, directory / f'round_{round_number:06d}_client_{client}_{stage}.pt')

def write_loaded_trace(aggregator, checkpoint):
    directory = _directory()
    if directory is None:
        return
    from federatedscope.glue.cenmag.aggregator import capture_rng_state
    state = aggregator.cenmag_state
    fields = ['global_a', 'global_magnitude', 'client_local_states', 'client_local_magnitudes', 'client_local_rounds']
    payload = {'state': {k: _clone(getattr(state, k)) for k in fields},
               'rng': capture_rng_state(), 'server_round': aggregator.loaded_round}
    torch.save(payload, directory / 'immediate_loaded_state.pt')
