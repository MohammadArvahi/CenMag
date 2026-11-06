"""Explicit isolated CPython 3.12 + pinned CUDA wheels for a Python-3.13 host.

Everything is installed under a new scratch prefix. No host package, CUDA driver,
account setting or global environment is replaced. No GPU workload is executed.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from platform_tools.common import RELEASE, read, sha, atomic


def setup(input_root, prefix, execute=False):
    input_root, prefix = Path(input_root).resolve(), Path(prefix).resolve()
    profile = read(RELEASE/'manifests/isolated_runtime.json')
    archive = input_root/profile['interpreter']['filename']
    if sha(archive) != profile['interpreter']['sha256']:
        raise RuntimeError('Portable CPython archive hash mismatch')
    for wheel in profile['wheels']:
        if sha(input_root/'wheels'/wheel['filename']) != wheel['sha256']:
            raise RuntimeError('Pinned runtime wheel mismatch: ' + wheel['filename'])
    if prefix.exists():
        raise RuntimeError('A new isolated scratch prefix is mandatory')
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('Profile requires Linux x86-64')
    probe = prefix.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < 12*1024**3:
        raise RuntimeError('Need at least 12 GiB scratch capacity for pinned runtime installation')
    python = prefix/'python/bin/python3.12'
    venv_python = prefix/'env/bin/python'
    commands = [[str(python), '-m', 'venv', str(prefix/'env')],
                [str(venv_python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--only-binary=:all:',
                 '--require-hashes', '--find-links', str(input_root/'wheels'), '-r', str(RELEASE/'configs/local_reference.lock')],
                [str(venv_python), '-m', 'pip', 'check']]
    plan = dict(profile=profile['profile'], status='DRY_RUN', commands=commands,
                interpreter_archive_sha256=profile['interpreter']['sha256'],
                install_prefix=str(prefix), model_loading=False, gpu_execution=False, global_changes=False)
    if not execute:
        return plan
    prefix.mkdir(parents=True)
    # Python's data filter rejects escaping paths/devices/unsafe links.
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(prefix, filter='data')
    version = subprocess.check_output([str(python), '-c', 'import platform;print(platform.python_version())'], text=True).strip()
    if version != profile['interpreter']['version']:
        raise RuntimeError('Extracted interpreter version differs from pinned archive')
    log_path = prefix/'installation.log'
    with log_path.open('x') as log:
        for command in commands:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError('Isolated runtime command failed; inspect ' + str(log_path))
    plan.update(status='INSTALLED_CPU_DEPENDENCIES_CHECKED_GPU_VALIDATION_REQUIRED', python=str(venv_python),
                interpreter_version=version)
    atomic(prefix/'runtime_installation.json', plan)
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print(json.dumps(setup(args.input, args.prefix, args.execute), indent=2))
