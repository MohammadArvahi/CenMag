"""Round history and numerical and validation safety checks."""
import hashlib
import math
import statistics

from platform_tools.common import json_hash


def is_finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def record_nonfinite(record):
    """Detect invalid recorded loss, accuracy, gradients, parameters or logits."""
    if not is_finite(record.get("weighted_validation_accuracy")):
        return True
    for client in record.get("clients", []):
        if not is_finite(client.get("accuracy")) or client.get("nonfinite_logits", 0) or client.get("invalid_output_fraction", 0):
            return True
    for row in record.get("optimization", []):
        for group in row.get("groups", {}).values():
            if group.get("gradient_finite") is False or group.get("parameter_finite") is False or not is_finite(group.get("gradient_norm")):
                return True
    for entry in (record.get("local_training") or {}).values():
        metrics = (entry or {}).get("metrics", {})
        for key in ("train_loss", "train_avg_loss"):
            if key in metrics and not is_finite(metrics[key]):
                return True
    return False


def summarize(record):
    def finite(x):
        return x if is_finite(x) else None

    clients = sorted(record['clients'], key=lambda row: row['client'])
    if [row['client'] for row in clients] != [1, 2, 3]:
        raise ValueError('Round record must cover all three clients')
    return dict(round=record['completed_round'], accuracy=finite(record['weighted_validation_accuracy']),
                nonfinite=record_nonfinite(record),
                clients=[dict(accuracy=finite(row['accuracy']),
                              dominant=finite(row['dominant_prediction_fraction']),
                              majority=max(row['true_class_counts']) / sum(row['true_class_counts']))
                         for row in clients])


class Protocol:
    """Verify continuous round history and record the first safety stop."""

    def __init__(self, records=()):
        self.rows = []
        self.digest = hashlib.sha256(b'').hexdigest()
        self.stop = None
        self._learned_at = None
        self._flagged = 0
        self._instability_streaks = [0, 0, 0]
        self._single_class_streaks = [0, 0, 0]
        for record in records:
            self.observe(record)

    def observe(self, record):
        n = len(self.rows) + 1
        if record['round'] != n or self.stop:
            raise ValueError('Expected the next round of a nonterminal run')
        self.rows.append(record)
        self.digest = hashlib.sha256((self.digest + json_hash(record)).encode()).hexdigest()
        if record['nonfinite']:
            self.stop = 'nonfinite_metrics'
            return self.stop

        if self._learned_at is None and n >= 5 and all(row['accuracy'] >= 0.85 for row in self.rows[-5:]):
            self._learned_at = n
        if n <= 25:
            return None

        for i, client in enumerate(record['clients']):
            single = client['dominant'] >= 0.98 and client['majority'] <= 0.90
            self._single_class_streaks[i] = self._single_class_streaks[i] + 1 if single else 0
            if self._learned_at is None or n <= self._learned_at:
                continue
            previous = [row['clients'][i]['accuracy'] for row in self.rows[max(0, n - 11):n - 1]]
            drop = len(previous) >= 5 and client['accuracy'] <= statistics.median(previous) - 0.25
            flagged = single or drop
            self._flagged += int(flagged)
            self._instability_streaks[i] = self._instability_streaks[i] + 1 if flagged else 0

        if self._flagged >= 4 or max(self._instability_streaks) >= 3:
            self.stop = 'validation_instability'
        elif max(self._single_class_streaks) >= 10:
            self.stop = 'persistent_single_class'
        return self.stop

    def snapshot(self):
        return dict(completed_round=len(self.rows), digest=self.digest, stop=self.stop)
