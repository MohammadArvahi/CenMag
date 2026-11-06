_DIAGNOSTIC_KEY_FRAGMENTS = (
    '_true_count_class_',
    '_pred_count_class_',
    '_pred_unique_classes',
    '_pred_majority_fraction',
    '_task_avg_loss',
    '_magnitude_regularization_avg_loss',
    '_total_avg_loss',
)


def filter_server_metrics(metrics, enabled):
    """Keep client diagnostics out of generic best-metric aggregation.

    The diagnostics remain in each client's ``Results_raw`` log.  FedScope's
    generic server monitor only accepts metrics registered in its metric
    calculator, so passing class counts or CenMag's decomposed loss telemetry
    to it raises a ``KeyError``.  Canonical ``*_avg_loss`` and ``*_accuracy``
    remain available for weighted aggregation and best-result selection.
    """
    if not enabled:
        return metrics
    return {
        key: value
        for key, value in metrics.items()
        if not any(fragment in key
                   for fragment in _DIAGNOSTIC_KEY_FRAGMENTS)
    }
