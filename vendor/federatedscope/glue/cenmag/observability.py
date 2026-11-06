"""Optional scalar diagnostics; does not modify parameters, gradients or RNG."""
from __future__ import annotations
from collections import Counter
import math

def prediction_summary(predictions, labels, num_classes=3):
    predicted, true = list(predictions), list(labels)
    if len(predicted) != len(true) or not true:
        raise ValueError('Predictions and labels must be nonempty and equal length')
    if any(isinstance(x, bool) or int(x) != x or not 0 <= x < num_classes for x in predicted + true):
        raise ValueError('Class IDs are outside the declared label mapping')
    n = len(true)
    pc, tc = Counter(predicted), Counter(true)
    accuracy = sum(p == t for p, t in zip(predicted, true)) / n
    majority = max(tc.values()) / n
    return {'samples': n, 'accuracy': accuracy, 'majority_baseline': majority,
            'gain_over_majority': accuracy - majority, 'single_class': len(pc) == 1,
            'true_counts': [tc[i] for i in range(num_classes)],
            'prediction_counts': [pc[i] for i in range(num_classes)]}

def aggregate_accuracy(records):
    rows = list(records)
    if not rows or any(r['samples'] <= 0 or not 0 <= r['accuracy'] <= 1 for r in rows):
        raise ValueError('Invalid client metric records')
    total = sum(r['samples'] for r in rows)
    return {'samples': total,
            'weighted_accuracy': sum(r['samples'] * r['accuracy'] for r in rows) / total,
            'macro_client_accuracy': sum(r['accuracy'] for r in rows) / len(rows),
            'minimum_client_accuracy': min(r['accuracy'] for r in rows),
            'weighted_majority_baseline': sum(r['samples'] * r['majority_baseline'] for r in rows) / total}

def accuracy_drop(previous, current, threshold=0.10):
    if not 0 <= threshold <= 1 or not all(0 <= x <= 1 for x in [previous, current]):
        raise ValueError('Accuracy/threshold outside [0,1]')
    return {'drop': previous - current, 'sudden_drop': previous - current >= threshold}

def gradient_summary(model):
    import torch
    groups = {name: [] for name in ['lora_A', 'lora_B', 'client_local_magnitude', 'classifier']}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        for marker in groups:
            if marker in name:
                gradient = parameter.grad.detach().float()
                groups[marker].append((float(gradient.norm().item()), bool(torch.isfinite(gradient).all())))
                break
    return {name: {'tensor_count': len(values),
                   'l2': math.sqrt(sum(n * n for n, _ in values)),
                   'all_finite': all(finite for _, finite in values)}
            for name, values in groups.items()}
