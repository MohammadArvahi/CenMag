"""Prospectively configured diagnostic heuristics, never an optimizer controller."""
from collections import deque
import math

DEFAULTS = dict(majority=0.99, accuracy_drop=0.15, entropy_drop=0.4,
                distribution_tv=0.5, client_accuracy_spread=0.4,
                learned_accuracy=0.60, warmup_rounds=10, persistent_rounds=3,
                recent_window=5, gradient_explosion=1e3, gradient_vanishing=1e-12,
                update_jump_ratio=10.0, stage_logit_change=3.0,
                loss_prediction_inconsistency=0.2)


class Detector:
    def __init__(self, config=None, state=None):
        self.config = {**DEFAULTS, **(config or {})}
        self.history = {int(k): deque(v, maxlen=self.config['recent_window'])
                        for k, v in (state or {}).get('history', {}).items()}
        self.streak = {int(k): v for k, v in (state or {}).get('streak', {}).items()}
        self.learned = {int(k): v for k, v in (state or {}).get('learned', {}).items()}
        self.active = {int(k): v for k, v in (state or {}).get('active', {}).items()}

    def snapshot(self):
        return dict(history={k: list(v) for k, v in self.history.items()},
                    streak=self.streak, learned=self.learned, active=self.active)

    def observe(self, completed, rows, optimization=(), stages=()):
        events = []
        def emit(kind, cid, **evidence):
            events.append(dict(kind=kind, client=cid, completed_round=completed,
                               phase='completed_round_evaluation', heuristic=True,
                               causal_claim=False, measurements=evidence))
        for row in rows:
            cid = int(row['client'])
            history = self.history.setdefault(cid, deque(maxlen=self.config['recent_window']))
            previous = history[-1] if history else None
            recent = sum(v['accuracy'] for v in history) / len(history) if history else None
            row['accuracy_delta_previous'] = row['accuracy'] - previous['accuracy'] if previous else None
            row['accuracy_delta_recent_window'] = row['accuracy'] - recent if recent is not None else None
            near = row['dominant_prediction_fraction'] >= self.config['majority']
            drop = recent is not None and recent - row['accuracy'] >= self.config['accuracy_drop']
            learned = self.learned.get(cid, False)
            self.streak[cid] = self.streak.get(cid, 0) + 1 if near else 0
            if near or drop:
                if learned and drop:
                    kind = 'post_learning_collapse_candidate'
                elif completed <= self.config['warmup_rounds'] and not learned:
                    kind = 'warmup_single_class'
                elif self.streak[cid] >= self.config['persistent_rounds']:
                    kind = 'persistent_degeneration_candidate'
                else:
                    kind = 'temporary_instability'
                emit(kind, cid, majority=row['dominant_prediction_fraction'],
                     accuracy=row['accuracy'], recent_accuracy=recent, consecutive=self.streak[cid])
                self.active[cid] = True
            elif self.active.get(cid):
                emit('recovery_candidate', cid, accuracy=row['accuracy'])
                self.active[cid] = False
            if previous:
                entropy_drop = previous['prediction_entropy'] - row['prediction_entropy']
                if entropy_drop >= self.config['entropy_drop']:
                    emit('entropy_drop', cid, entropy_drop=entropy_drop)
                a, b = previous['predicted_class_counts'], row['predicted_class_counts']
                tv = 0.5 * sum(abs(x / sum(a) - y / sum(b)) for x, y in zip(a, b))
                if tv >= self.config['distribution_tv']:
                    emit('prediction_distribution_shift', cid, total_variation=tv)
            if row.get('nonfinite_logits', 0):
                emit('nonfinite_outputs', cid, count=row['nonfinite_logits'])
            self.learned[cid] = learned or (not near and row['accuracy'] >= self.config['learned_accuracy'])
            history.append({k: row[k] for k in ['accuracy', 'prediction_entropy', 'predicted_class_counts']})
        if len(rows) > 1:
            spread = max(r['accuracy'] for r in rows) - min(r['accuracy'] for r in rows)
            if spread >= self.config['client_accuracy_spread']:
                emit('personalized_client_divergence', None, accuracy_spread=spread)
        for step in optimization:
            for name, group in step.get('groups', {}).items():
                gn = group['gradient_norm']
                if not group['gradient_finite'] or not math.isfinite(gn):
                    emit('nonfinite_gradient', step['client'], group=name)
                elif gn >= self.config['gradient_explosion'] or gn <= self.config['gradient_vanishing']:
                    emit('gradient_scale', step['client'], group=name, gradient_norm=gn)
                pn = group.get('parameter_norm_before', 0)
                if pn and group['update_norm'] / pn >= self.config['update_jump_ratio']:
                    emit('parameter_update_jump', step['client'], group=name,
                         relative_update=group['update_norm'] / pn)
            client_row = next((r for r in rows if r['client'] == step['client']), None)
            if client_row and client_row['dominant_prediction_fraction'] >= self.config['majority'] and step['task_loss'] <= self.config['loss_prediction_inconsistency']:
                emit('loss_prediction_inconsistency', step['client'], task_loss=step['task_loss'])
        for probe in stages:
            if probe.get('logit_change_max_abs', 0) >= self.config['stage_logit_change']:
                emit('stage_output_change', probe['client'], stage=probe['stage'],
                     maximum_logit_change=probe['logit_change_max_abs'])
        return events
