"""Round-boundary session guard and forensic promotion, independent of accuracy."""
from pathlib import Path
import copy
import os
import time
import psutil
from .common import atomic, append, read, utc
from .artifacts import Retention, discover, event_package, checked_generation, verify_artifact
from .detector import Detector

_CURRENT = None


def current():
    if _CURRENT is None:
        raise RuntimeError('Protected session controller was not installed')
    return _CURRENT


class Session:
    def __init__(self, out, observer, operations, stop_at, identity, state=None):
        global _CURRENT
        _CURRENT = self
        self.out, self.observer, self.operations = Path(out), observer, operations
        self.stop_at, self.identity = stop_at, identity
        self.reason = ''
        self.started = time.monotonic()
        self.round_started = time.monotonic()
        self.round_times = []
        self.save_times = []
        self.detector = Detector(operations.get('thresholds'), (state or {}).get('detector'))
        self.pending_after = (state or {}).get('pending_after', [])
        self.pending_event = (state or {}).get('pending_event')
        self.saved_evidence = (state or {}).get('forensic_evidence', {})
        self.events = []
        self.last_event_signature = (state or {}).get('last_event_signature', [])
        self.metrics = None
        self.retention = Retention(out, operations['rolling_states'], operations['storage_budget_bytes'],
                                   operations['free_reserve_bytes'], operations['retained_event_packages'])

    def stopped(self):
        request = read(self.out / 'stop_request.json', {})
        if request:
            self.reason = self.reason or request.get('reason', 'manual_boundary_stop')
        if os.environ.get('CENMAG_STOP_REQUESTED'):
            self.reason = self.reason or 'signal_boundary_stop'
        return bool(self.reason)

    def remaining(self):
        return self.operations['session_seconds'] - (time.monotonic() - self.started)

    def estimate(self):
        round_seconds = max(self.round_times[-5:], default=self.operations['initial_round_estimate_seconds'])
        checkpoint = max(self.save_times[-5:], default=self.operations['checkpoint_estimate_seconds'])
        return (round_seconds * self.operations['round_time_safety_factor'] + checkpoint
                + self.operations['export_estimate_seconds'] + self.operations['shutdown_reserve_seconds'])

    def before_dispatch(self, server):
        from boundary import ControlledStop
        self.recover_forensic_publication()
        if self.remaining() < self.estimate():
            self.reason = 'session_time_budget'
        if not self.retention.capacity()['safe']:
            self.reason = self.reason or 'evidence_storage_budget'
        if self.stopped():
            raise ControlledStop(self.reason)
        self.round_started = time.monotonic()
        atomic(self.out / 'progress.json', dict(phase='dispatch', next_round=int(server.state),
               unix=time.time(), remaining_seconds=self.remaining(), next_round_reserve_seconds=self.estimate()))

    def recover_forensic_publication(self):
        """A checkpoint may commit just before event packaging is interrupted."""
        if not self.pending_event and not self.pending_after:
            return
        candidates = discover(self.out/'checkpoints')['eligible']
        rounds = {checked_generation(p)['server_round']: p for p in candidates}
        def existing(n, sequence):
            name = f'round_{n:06d}_{sequence:04d}'
            targets = [self.out/'events'/name] + list((self.out/'history').glob('**/events/'+name))
            for target in targets:
                if target.is_dir() and n in rounds:
                    recorded = read(target/'event.json', {})
                    if Path(rounds[n]).parent.name in recorded.get('checkpoint_names', []):
                        verify_artifact(target)
                        return target
            return None
        if self.pending_event:
            n = self.pending_event['completed_round']
            if existing(n, 0):
                pass
            elif n in rounds:
                paths = [rounds[k] for k in sorted(rounds) if n-2 <= k <= n]
                event_package(self.out, self.pending_event, paths,
                              dict(**self.saved_evidence, recovered_after_publication_interruption=True))
            else:
                raise RuntimeError('Pending forensic event is missing its coherent checkpoint')
            self.pending_event = None
        for previous in self.pending_after[:]:
            n = previous + 1
            if existing(n, previous):
                self.pending_after.remove(previous)
            elif n in rounds:
                event_package(self.out, dict(kind='event_followup', completed_round=n,
                    sequence=previous, preceding_event_round=previous), [rounds[n]],
                    dict(**self.saved_evidence, recovered_after_publication_interruption=True))
                self.pending_after.remove(previous)
        self.retention.prune()

    def prepare_boundary(self, server):
        n = int(server.state)
        rows = self.observer.evaluations
        if sorted(int(r['client']) for r in rows) != [1, 2, 3] or any(r['completed_round'] != n for r in rows):
            raise RuntimeError('Observer evaluation/client boundary mismatch')
        expected = read(Path(__file__).resolve().parents[1] / 'manifests/partition_reference.json')['clients']
        if any(r['examples'] != expected[str(r['client'])]['val']['count'] for r in rows):
            raise RuntimeError('Full canonical validation sample count mismatch')
        self.events = self.detector.observe(n, rows, self.observer.steps, self.observer.deep.probes)
        significant = [e for e in self.events if e['kind'] != 'warmup_single_class']
        signature = sorted({f"{e['kind']}:{e['client']}:{e['measurements'].get('stage', '')}" for e in significant})
        newly_triggered = bool(set(signature) - set(self.last_event_signature))
        self.last_event_signature = signature
        # One immutable evidence package per event round, with all individual triggers.
        self.bundle_event = dict(kind='diagnostic_event', completed_round=n, sequence=0,
                                 triggers=significant, phase='completed_round_evaluation',
                                 heuristic=True, causal_claim=False) if newly_triggered else None
        if self.bundle_event:
            self.pending_after.append(n)
        self.round_times.append(time.monotonic() - self.round_started)
        self.metrics = dict(schema=1, completed_round=n, evaluation_id=f'completed_{n:06d}',
                            weighted_validation_accuracy=sum(r['correct'] for r in rows) / sum(r['examples'] for r in rows),
                            mean_client_accuracy=sum(r['accuracy'] for r in rows) / len(rows),
                            validation_examples=sum(r['examples'] for r in rows), clients=copy.deepcopy(rows),
                            client_participation=[1, 2, 3], aggregation_weights=server.cenmag_state.last_aggregation_weights,
                            processed_examples_cumulative=copy.deepcopy(server._processed_examples),
                            processed_examples_this_round=self.observer.aggregations[-1]['processed_examples'],
                            training_seconds=sum(self.observer.fit_times.values()),
                            local_training=copy.deepcopy(self.observer.fit_summaries),
                            evaluation_seconds=sum(r['duration_seconds'] for r in rows),
                            round_seconds=self.round_times[-1],
                            optimization=copy.deepcopy(self.observer.steps),
                            observer_seconds_cumulative=self.observer.overhead_seconds,
                            forensic_seconds_cumulative=self.observer.deep.overhead_seconds,
                            resource=dict(rss_bytes=psutil.Process().memory_info().rss,
                                          available_ram_bytes=psutil.virtual_memory().available),
                            anomalies=self.events)
        import torch
        if torch.cuda.is_available():
            self.metrics['resource'].update(allocated_gpu_bytes=torch.cuda.memory_allocated(),
                reserved_gpu_bytes=torch.cuda.memory_reserved(), peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
        if n >= self.stop_at:
            self.reason = 'requested_session_boundary'
        if self.remaining() < self.estimate():
            self.reason = self.reason or 'session_time_budget'
        self.stopped()

    def snapshot(self):
        return dict(detector=self.detector.snapshot(), pending_after=list(self.pending_after),
                    pending_event=getattr(self, 'bundle_event', self.pending_event),
                    forensic_evidence=dict(metrics=self.metrics, identity=self.identity,
                        stages=self.observer.deep.probes if self.observer else [],
                        stage_aggregation=self.observer.deep.aggregations if self.observer else []),
                    operations=self.operations, last_event_signature=self.last_event_signature,
                    last_completed_round=self.metrics['completed_round'] if self.metrics else None)

    def commit(self, server, best_requests):
        n = int(server.state)
        start = time.perf_counter()
        # Publish the complete state before advancing the round ledger.
        result = server.aggregator.save_model(server._cfg.federate.save_to, n)
        path = Path(result['path'])
        manifest = checked_generation(path)
        self.save_times.append(time.perf_counter() - start)
        if best_requests:
            atomic(self.out / 'checkpoints/best.json', dict(current=path.parent.name, boundary=n))
        atomic(self.out / 'checkpoints/recovery.json', dict(current=path.parent.name, completed_round=n,
               checkpoint_sha256=manifest['sha256'], local_verified=True, external_verified=False))
        candidates = discover(self.out / 'checkpoints')['eligible']
        evidence = dict(metrics=self.metrics, stages=self.observer.deep.probes,
                        stage_aggregation=self.observer.deep.aggregations,
                        identity=self.identity, optimizer={'type': 'SGD', 'lr': 0.02, 'state': {}},
                        runtime_policy='canonical per-fit reset under client-round seed')
        related = sorted([p for p in candidates if n-2 <= checked_generation(p)['server_round'] <= n],
                         key=lambda p: checked_generation(p)['server_round'])
        if self.bundle_event:
            package = event_package(self.out, self.bundle_event, related, evidence)
            append(self.out / 'event_timeline.jsonl', dict(completed_round=n, package=str(package.relative_to(self.out)),
                   triggers=self.events, external_verified=False))
        # After-state is a separate immutable package; original event never changes.
        for previous in self.pending_after[:]:
            if previous < n:
                event_package(self.out, dict(kind='event_followup', completed_round=n, sequence=previous,
                              preceding_event_round=previous), [path], evidence)
                self.pending_after.remove(previous)
        self.metrics.update(checkpoint_id=path.parent.name, checkpoint_bytes=manifest['bytes'],
                            checkpoint_seconds=self.save_times[-1], checkpoint_sha256=manifest['sha256'],
                            checkpoint_status='LOCAL_VERIFIED', external_verified=False)
        append(self.out / 'rounds.jsonl', self.metrics)
        self.observer.flush_round()
        atomic(self.out / 'progress.json', dict(phase='completed_boundary', completed_round=n,
               local_verified_round=n, external_verified_round=None, checkpoint=str(path),
               reason=self.reason, unix=time.time(), remaining_seconds=self.remaining()))
        self.retention.prune()
        if not self.retention.capacity(manifest['bytes'])['safe']:
            self.reason = self.reason or 'evidence_storage_budget'
        # Worker completes core lifecycle, then raises ControlledStop if requested.
