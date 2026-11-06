"""Small, durable filesystem primitives. No network or credentials."""
from pathlib import Path
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time

RELEASE = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append(path, record):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def inventory(root, exclude=()):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError('Symlink is not a portable artifact: ' + relative)
        if path.is_file() and relative not in exclude:
            result[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def verify_inventory(root, records):
    root = Path(root).resolve()
    for name, expected in records.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Unsafe artifact path: ' + name)
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise RuntimeError('Artifact hash/size mismatch: ' + name)
    return True


@contextlib.contextmanager
def exclusive(path):
    """Kernel lock, never guessed stale from a PID or elapsed time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another local session owns this experiment') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
