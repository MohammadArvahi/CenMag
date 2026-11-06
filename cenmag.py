"""Prepare, train, resume, and report CenMag on MNLI."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def positive_rounds(text):
    rounds = int(text)
    if rounds <= 0:
        raise argparse.ArgumentTypeError('rounds must be a positive integer')
    return rounds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'start', 'resume'):
        command = commands.add_parser(name)
        command.add_argument('--inputs', type=Path, default=Path('inputs'))
        if name != 'check':
            command.add_argument('--out', type=Path, required=True)
            command.add_argument('--rounds', required=True, type=positive_rounds,
                                 help='Total completed rounds to reach')
        if name != 'start':
            command.add_argument('--from', dest='source', type=Path, required=name == 'resume',
                                 help='A session directory created by this release')
    metrics = commands.add_parser('metrics')
    metrics.add_argument('--run', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith('CENMAG_') and k != 'PYTHONPATH'}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', CENMAG_KEEP_CHECKPOINTS='3', WANDB_MODE='disabled')
    if args.command in ('start', 'resume'):
        environment.setdefault('CUDA_VISIBLE_DEVICES', '0')
        cache = args.out.resolve() / 'cache'
        environment.update(HF_HOME=str(cache / 'huggingface'), XDG_CACHE_HOME=str(cache / 'xdg'),
                           CUDA_CACHE_PATH=str(cache / 'cuda'), TRITON_CACHE_DIR=str(cache / 'triton'),
                           TMPDIR=str(cache / 'tmp'),
                           CENMAG_RUN_CHECKPOINT_ROOT=str(args.out.resolve() / 'checkpoints'))
    else:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    os.execve(sys.executable, [sys.executable, '-B', str(ROOT / 'runtime/public_run.py'), *sys.argv[1:]], environment)


if __name__ == '__main__':
    main()
