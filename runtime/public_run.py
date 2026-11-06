"""Run the pinned engine in an isolated local session."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import fcntl
import json
import logging
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]

from cenmag import parse_args
from platform_tools.common import atomic, exclusive, read
from public_config import configuration, dependency_versions, input_assets, verify_inputs


@contextmanager
def source_lock(path):
    with (path / '.session.lock').open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def read_source(source, assets, identity):
    from recovery import Validator, checkpoint_path, history

    target = read(source / 'session.json')['target']
    cfg, _ = configuration(target, source, assets)
    validator = Validator(cfg, identity)
    path = checkpoint_path(source)
    payload = validator(path)
    return target, validator, path, payload, history(source, payload)


def session_class(protocol):
    from platform_tools.session import Session
    from protocol import summarize

    class LocalSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from storage_policy import available_capacity
            self.retention.capacity = lambda estimated_checkpoint_bytes=128 * 1024**2: available_capacity(
                self.out, self.operations['free_reserve_bytes'], estimated_checkpoint_bytes
            )

        def before_dispatch(self, server):
            if int(server.state) >= self.stop_at:
                from boundary import ControlledStop
                raise ControlledStop('requested_session_boundary')
            return super().before_dispatch(server)

        def prepare_boundary(self, server):
            super().prepare_boundary(server)
            if reason := protocol.observe(summarize(self.metrics)):
                self.reason = reason

        def snapshot(self):
            return dict(super().snapshot(), public_protocol=protocol.snapshot())

    return LocalSession


def launch(args, assets, asset_identity):
    import torch
    from federatedscope.glue.cenmag.hardware import assert_cenmag_cuda
    from recovery import Validator, import_window
    from protocol import Protocol

    out = args.out.resolve()
    source = args.source.resolve() if getattr(args, 'source', None) else None
    with source_lock(source) if source else nullcontext():
        if source:
            if read(source / 'result.json', {}).get('numerical_failure'):
                raise RuntimeError('Source session stopped on a numerical invariant')
            previous, source_validate, path, payload, protocol = read_source(source, assets, asset_identity)
            if protocol.stop:
                raise RuntimeError(f'Protocol stopped at round {payload["server_round"]}: {protocol.stop}')
            completed = payload['server_round']
        else:
            completed, protocol = 0, Protocol()
        target = args.rounds
        if target <= completed:
            raise ValueError(f'The training target must be larger than the completed round ({completed})')
        assert_cenmag_cuda(torch, 0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Select exactly one visible CUDA device')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        cfg, _ = configuration(target, out, assets)
        target_validate = Validator(cfg, asset_identity)
        checkpoint = ''
        if source:
            new_target_identity = None
            if previous != target:
                new_target_identity = target_validate.identity_for(payload['boundary_contract']['identity']['execution'])
            checkpoint = import_window(path, payload, out, source_validate, target_validate, new_target_identity)
            atomic(out / 'rounds_before.json', protocol.rows)
            cfg, _ = configuration(target, out, assets, checkpoint)
        atomic(out / 'session.json', dict(completed_before=completed, target=target))
        atomic(out / 'assets.json', assets)
        args = SimpleNamespace(out=out, assets=out / 'assets.json', checkpoint=str(checkpoint),
                               stop_at=target, diagnostics='light', replay=False)
        import data_runtime
        import execute
        import mnli
        import observability
        data_runtime.load_data = mnli.load_data
        observability.summarize = mnli.summarize
        result = execute.run(args, cfg=cfg, asset_identity=asset_identity, checkpoint_validator=target_validate,
                             session_class=session_class(protocol))
        atomic(out / 'protocol.json', protocol.snapshot())
        return result


def main():
    args = parse_args()
    if args.command in ('start', 'resume'):
        out = args.out.resolve()
        source = args.source.resolve() if getattr(args, 'source', None) else None
        if source and out.is_relative_to(source):
            raise ValueError('Create a new session outside the source directory')
        out.mkdir(parents=True, exist_ok=False)
        with exclusive(out / '.session.lock'):
            (out / 'cache' / 'tmp').mkdir(parents=True)
            return dispatch(args)
    return dispatch(args)


def dispatch(args):
    from platform_tools.environment import verify_release
    verify_release()
    dependency_versions()
    if args.command == 'metrics':
        import torch
        from recovery import checkpoint_path, checked_generation, history
        run = args.run.resolve()
        with source_lock(run):
            path = checkpoint_path(run)
            checked_generation(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            protocol = history(run, payload)
            print(json.dumps(dict(**protocol.snapshot(), rounds=protocol.rows), indent=2))
        return
    assets = input_assets(args.inputs.resolve())
    identity = verify_inputs(assets)
    if args.command == 'check':
        configuration(1, ROOT / 'runs/check', assets)
        if args.source:
            from recovery import recovery_window
            source = args.source.resolve()
            with source_lock(source):
                target, validator, path, payload, protocol = read_source(source, assets, identity)
                window = recovery_window(path, payload)
                for generation in window.values():
                    validator(generation)
                print(json.dumps(dict(status='PASS', target=target, window=list(window), **protocol.snapshot())))
        else:
            print(json.dumps(dict(status='PASS', partitions=3, training=False)))
        return
    launch(args, assets, identity)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(f'CenMag: {error}')
