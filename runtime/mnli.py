"""Prepared MNLI loader and evaluation summary."""
from pathlib import Path


def load_data(cache, cfg, evaluation_counts=None):
    """All labels and sample membership come from immutable prepared tensors."""
    import torch
    from torch.utils.data import Subset
    from data_runtime import Rows, array_hash, probe_indices
    from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
    if evaluation_counts is not None:
        raise ValueError('New manual runs require full labeled validation')
    payload = torch.load(Path(cache) / 'data.pt', weights_only=True, map_location='cpu')
    data, selection = {}, {}
    for cid, part in payload['partition'].items():
        data[cid] = {s: get_dataloader(Subset(Rows(payload[s]), ix.tolist()), cfg, split=s)
                     for s, ix in part.items()}
        probe = probe_indices(part['val'], payload['val']['label'], 32)
        data[cid]['probe'] = [Rows(payload['val'])[int(i)] for i in probe]
        selection[cid] = dict(train_count=len(part['train']), val_count=len(part['val']),
            train_indices_sha256=array_hash(part['train']), val_indices_sha256=array_hash(part['val']),
            val_indices=part['val'].tolist(), val_class_counts=torch.bincount(payload['val']['label'][part['val']], minlength=cfg.data.num_labels).tolist(),
            probe_indices=probe.tolist(), probe_indices_sha256=array_hash(probe))
    return data, selection



def summarize(logits, labels, sample_ids, client, round_index, loss):
    import torch
    import numpy as np
    logits, labels = logits.float().cpu(), labels.long().cpu()
    if logits.ndim != 2 or logits.shape[0] != len(labels) or logits.shape[1] not in (2, 3):
        raise ValueError('Invalid task logits shape')
    k = logits.shape[1]
    if len(labels) and (int(labels.min()) < 0 or int(labels.max()) >= k):
        raise ValueError('Label outside dataset mapping')
    finite = torch.isfinite(logits).all(-1)
    good_logits, good_labels = logits[finite], labels[finite]
    pred = good_logits.argmax(-1)
    probability = good_logits.softmax(-1)
    cm = torch.bincount(good_labels * k + pred, minlength=k*k).reshape(k, k)
    support = torch.bincount(labels, minlength=k); counts = cm.sum(0)
    n, valid = len(labels), int(finite.sum())
    return dict(client=int(client), round_index=int(round_index), completed_round=int(round_index)+1,
        evaluation_id=f'completed_{int(round_index)+1:06d}_client_{client}', examples=n, valid_examples=valid,
        correct=int(cm.diag().sum()), accuracy=int(cm.diag().sum()) / n if n else None, loss=loss,
        true_class_counts=support.tolist(), predicted_class_counts=counts.tolist(), confusion_matrix=cm.tolist(),
        per_class_recall=[float(cm[j,j] / support[j]) if support[j] else None for j in range(k)],
        prediction_entropy=float((-(probability * probability.clamp_min(1e-30).log()).sum(-1)).mean()) if valid else 0.,
        entropy_definition='mean Shannon entropy of softmax logits in nats over finite rows',
        dominant_prediction_fraction=float(counts.max()) / valid if valid else 0., nonfinite_logits=n-valid,
        invalid_output_fraction=(n-valid)/n if n else 0.,
        sample_ids_sha256=__import__('hashlib').sha256(np.asarray(sample_ids, dtype='<i8').tobytes()).hexdigest())

