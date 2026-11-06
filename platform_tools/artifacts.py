"""Local checkpoint generations and protected event packages."""
from pathlib import Path
import os
import re
import shutil
import tempfile
from .common import atomic, append, read, sha, inventory, verify_inventory, sync_dir, utc


def checked_generation(path):
    path = Path(path)
    if (path.name != 'state.pt' or not path.is_file() or path.is_symlink()
            or path.parent.is_symlink() or (path.parent / 'manifest.json').is_symlink()):
        raise RuntimeError('Expected a regular state.pt generation')
    manifest = read(path.parent / 'manifest.json')
    if not manifest or manifest.get('file') != 'state.pt' or manifest.get('schema') != 7:
        raise RuntimeError('Missing Schema 7 generation manifest')
    if manifest['bytes'] != path.stat().st_size or manifest['sha256'] != sha(path):
        raise RuntimeError('Corrupted checkpoint bytes')
    return manifest


def discover(root, validator=None):
    """Newest eligible state; a corrupt latest pointer/generation is not fatal."""
    root = Path(root)
    candidates, rejected = [], []
    for path in root.glob('boundary_*/state.pt'):
        try:
            m = checked_generation(path)
            if validator:
                validator(path)
            candidates.append((m['server_round'], str(path), m))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    candidates.sort(reverse=True)
    # Multiple states of the same completed round may be divergent branches.
    if candidates:
        newest = [c for c in candidates if c[0] == candidates[0][0]]
        if len({c[2]['boundary_id'] for c in newest}) > 1:
            raise RuntimeError('Ambiguous branches at newest boundary; select one explicitly')
    return dict(eligible=[c[1] for c in candidates], rejected=rejected)


class Retention:
    def __init__(self, run, rolling, budget_bytes, reserve_bytes, event_limit):
        if rolling < 10:
            raise ValueError('At least ten complete rolling states are mandatory')
        self.run = Path(run)
        self.root = self.run / 'checkpoints'
        self.rolling = rolling
        self.budget = budget_bytes
        self.reserve = reserve_bytes
        self.event_limit = event_limit

    def capacity(self, estimated_checkpoint_bytes=128 * 1024**2):
        used = sum(p.stat().st_size for p in self.run.rglob('*') if p.is_file())
        free = shutil.disk_usage(self.run).free
        # Reserve a checkpoint, an event copy of three states, and export space.
        needed = used + estimated_checkpoint_bytes * 8 + self.reserve
        return dict(used_bytes=used, free_bytes=free, reserve_bytes=needed,
                    budget_bytes=self.budget, safe=free >= needed and used + needed <= self.budget)

    def prune(self):
        self.root.mkdir(parents=True, exist_ok=True)
        protected = set()
        generations = []
        for directory in self.root.glob('boundary_*'):
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                m = checked_generation(directory / 'state.pt')
            except (RuntimeError, ValueError, KeyError, OSError):
                continue  # Unknown/corrupt material is evidence, never generic clutter.
            generations.append((int(m['server_round']), directory))
        generations.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        protected.update(d.name for _, d in generations[:self.rolling])
        best = read(self.root / 'best.json', {}).get('current')
        if isinstance(best, str):
            protected.add(best)
        deleted = []
        for n, directory in generations:
            if directory.name not in protected:
                deleted.append(dict(name=directory.name, completed_round=n,
                                    bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())))
                shutil.rmtree(directory)
        sync_dir(self.root)
        self.prune_diagnostics({n for n, directory in generations if directory.name in protected})
        append(self.run / 'storage.jsonl', dict(utc=utc(), deleted=deleted,
               protected=sorted(protected), capacity=self.capacity(), external_verified=False))

    def prune_diagnostics(self, retained_rounds):
        events = []
        for path in (self.run / 'events').glob('round_*'):
            match = re.fullmatch(r'round_([0-9]+)_([0-9]+)', path.name)
            if match and path.is_dir() and not path.is_symlink():
                events.append((int(match[1]), int(match[2]), path))
        for _, _, path in sorted(events, reverse=True)[self.event_limit:]:
            verify_artifact(path)
            shutil.rmtree(path)
        if events:
            sync_dir(self.run / 'events')

        for folder in (self.run / 'logits', self.run / 'history' / 'logits'):
            for path in folder.glob('round_*_client_*.npz'):
                match = re.fullmatch(r'round_([0-9]+)_client_[0-9]+\.npz', path.name)
                if match and int(match[1]) not in retained_rounds and path.is_file() and not path.is_symlink():
                    path.unlink()


def event_package(run, event, checkpoint_paths, evidence):
    """Separate immutable package per event round; no text or backbone copies."""
    run = Path(run)
    event_root = run / 'events'
    event_root.mkdir(exist_ok=True)
    event_id = f"round_{event['completed_round']:06d}_{event.get('sequence', 0):04d}"
    destination = event_root / event_id
    if destination.exists():
        raise RuntimeError('Event packages cannot be overwritten')
    pending = Path(tempfile.mkdtemp(prefix='.pending_', dir=event_root))
    try:
        checkpoint_rounds = set()
        for path in checkpoint_paths:
            path = Path(path)
            checkpoint_rounds.add(checked_generation(path)['server_round'])
            shutil.copytree(path.parent, pending / 'checkpoints' / path.parent.name)
        # Include available numeric evidence inside the immutable package itself,
        # including preceding-session logits after a cross-session continuation.
        available_logits = []
        for completed in sorted(checkpoint_rounds):
            pattern = f'round_{completed:06d}_client_*.npz'
            local = list((run/'logits').glob(pattern))
            candidates = local if local else list((run/'history').rglob(pattern))
            by_name = {}
            for source in sorted(candidates):
                fingerprint = sha(source)
                name = source.name
                if name in by_name and by_name[name] == fingerprint:
                    continue
                if name in by_name:
                    name = fingerprint[:12] + '_' + name
                by_name[name] = fingerprint
                (pending/'logits').mkdir(exist_ok=True)
                shutil.copy2(source, pending/'logits'/name)
                available_logits.append(dict(completed_round=completed, file='logits/'+name,
                    sha256=fingerprint, original_relative_path=source.relative_to(run).as_posix()))
        atomic(pending / 'event.json', dict(event=event, evidence=evidence,
               checkpoint_names=[Path(p).parent.name for p in checkpoint_paths],
               available_logits=available_logits, immutable=True, external_verified=False))
        atomic(pending / 'artifact.json', dict(schema=1, kind='forensic_event', files=inventory(pending)))
        verify_inventory(pending, read(pending / 'artifact.json')['files'])
        for p in pending.rglob('*'):
            if p.is_file():
                p.chmod(0o444)
        os.replace(pending, destination)
        sync_dir(event_root)
        return destination
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def verify_artifact(path, expected_manifest_sha256=None, receipt=None, external_reference=None):
    path = Path(path)
    manifest_path = path / 'artifact.json'
    if not manifest_path.is_file():
        raise RuntimeError('Missing artifact manifest; file existence is not persistence verification')
    actual = sha(manifest_path)
    if expected_manifest_sha256 and actual != expected_manifest_sha256:
        raise RuntimeError('External artifact manifest differs from recorded export')
    manifest = read(manifest_path)
    verify_inventory(path, manifest['files'])
    actual_files = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
    if actual_files != set(manifest['files']) | {'artifact.json'}:
        raise RuntimeError('Unmanifested or missing files in external artifact')
    result = dict(status='PASS', artifact_sha256=actual, verified_utc=utc(),
                  path=str(path.resolve()), external_reference=external_reference,
                  external_verified=bool(expected_manifest_sha256 and external_reference))
    if receipt:
        if not result['external_verified']:
            raise RuntimeError('Receipt requires expected hash and pinned external version reference')
        # Place outside immutable imported artifact.
        if Path(receipt).resolve().is_relative_to(path.resolve()):
            raise RuntimeError('Verification receipt must not mutate imported artifact')
        atomic(receipt, result)
    return result
