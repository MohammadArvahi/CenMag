"""CUDA eligibility and real NF4 execution checks; no GPU-name allowlist."""
from __future__ import annotations
from importlib.metadata import version

def validate_capability(info):
    failures = []
    if not info['cuda_available']:
        failures.append('CUDA is unavailable')
    if info['visible_device_count'] != 1 or info['device_index'] != 0:
        failures.append('canonical single-device execution requires one visible CUDA device at logical index 0')
    capability = tuple(info['compute_capability'] or (0, 0))
    if capability < (6, 0):
        failures.append('NF4 requires compute capability >= 6.0 in bitsandbytes 0.50.0')
    arch = 'sm_' + ''.join(str(x) for x in capability)
    if arch not in info['torch_supported_architectures']:
        failures.append(f'active PyTorch wheel has no native {arch} support; validate a compatible build before running')
    try:
        cuda_version = tuple(int(x) for x in (info.get('torch_cuda_version') or '0.0').split('.')[:2])
    except ValueError:
        cuda_version = (0, 0)
    if capability[0] >= 10 and cuda_version < (12, 8):
        failures.append('Blackwell execution requires CUDA >= 12.8 on this supported wheel path')
    if failures:
        raise RuntimeError('Invalid CenMag CUDA runtime: ' + '; '.join(failures))
    return dict(info, eligibility='PASS', quantization_execution='NOT_RUN')

def assert_cenmag_cuda(torch_module=None, device_index=0):
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    count = int(cuda.device_count()) if available else 0
    info = {
        'cuda_available': available, 'visible_device_count': count,
        'device_index': int(device_index),
        'gpu_name': str(cuda.get_device_name(device_index)) if count else None,
        'compute_capability': list(cuda.get_device_capability(device_index)) if count else None,
        'torch_supported_architectures': list(cuda.get_arch_list()) if count else [],
        'torch_cuda_version': torch_module.version.cuda,
    }
    return validate_capability(info)

def verify_nf4_execution(device_index=0, expected_version='0.50.0'):
    """Execute forward/backward with materialized NF4 + nested quantization.

    Forking Torch RNG preserves initialization and training random streams.
    This gate is a kernel test, not a model/round or memory feasibility result.
    """
    import torch
    import bitsandbytes as bnb
    if version('bitsandbytes') != expected_version:
        raise RuntimeError(f'Expected bitsandbytes {expected_version}, found {version("bitsandbytes")}')
    assert_cenmag_cuda(torch, device_index)
    device = f'cuda:{device_index}'
    with torch.random.fork_rng(devices=[device_index]):
        layer = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type='nf4').to(device)
        inputs = torch.randn(2, 32, device=device, dtype=torch.float16, requires_grad=True)
        outputs = layer(inputs)
        outputs.float().square().mean().backward()
        torch.cuda.synchronize(device_index)
        state = layer.weight.quant_state
        if (layer.weight.__class__.__name__ != 'Params4bit' or
            layer.weight.device.type != 'cuda' or state is None or
            str(state.quant_type).lower() != 'nf4' or not bool(state.nested) or
            not bool(torch.isfinite(outputs).all()) or inputs.grad is None or
            not bool(torch.isfinite(inputs.grad).all())):
            raise RuntimeError('Real NF4/double-quantization forward/backward gate failed')
    return {'status': 'PASS', 'nf4': True, 'double_quantization': True,
            'forward_backward': True, 'device_index': device_index}
