"""Read-only capability/dependency inventory; no installation or GPU workload."""
from pathlib import Path
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from .common import read, sha, RELEASE, verify_inventory, json_hash

CORE = {'transformers': '4.42.3', 'peft': '0.12.0', 'bitsandbytes': '0.50.0',
        'numpy': '1.26.4', 'datasets': '2.20.0', 'evaluate': '0.4.2',
        'accelerate': '0.33.0', 'tokenizers': '0.19.1', 'pyarrow': '17.0.0'}


def inspect(root, cuda=False):
    packages = {}
    for distribution in importlib.metadata.distributions():
        if distribution.metadata['Name']:
            packages.setdefault(distribution.metadata['Name'].lower().replace('_', '-'), distribution.version)
    from public_config import dependency_versions
    required = {n.lower().replace('_', '-'): v for n, v in dependency_versions().items()}
    mismatches = {k: dict(expected=v, actual=packages.get(k)) for k, v in required.items() if packages.get(k) != v}
    import psutil
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  cpu_count=os.cpu_count(), affinity_count=len(os.sched_getaffinity(0)),
                  memory=psutil.virtual_memory()._asdict(), disk=shutil.disk_usage(root)._asdict(),
                  packages=packages, core_mismatches=mismatches, gpu_probe_requested=cuda)
    result['packages_sha256'] = json_hash(packages)
    if cuda:
        import torch
        result['torch'] = str(torch.__version__)
        result['cuda_runtime'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        result['visible_devices'] = torch.cuda.device_count()
        result['gpus'] = [dict(name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i)),
                              total_bytes=torch.cuda.get_device_properties(i).total_memory)
                          for i in range(torch.cuda.device_count())]
        result['torch_architectures'] = torch.cuda.get_arch_list()
        try:
            result['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            result['driver'] = 'unavailable'
    return result


def verify_release():
    manifest = read(RELEASE / 'RELEASE_MANIFEST.json')
    if not manifest:
        raise RuntimeError('Release manifest not generated')
    verify_inventory(RELEASE, manifest['files'])
    return dict(status='PASS', files=len(manifest['files']),
                release_manifest_sha256=sha(RELEASE / 'RELEASE_MANIFEST.json'))
