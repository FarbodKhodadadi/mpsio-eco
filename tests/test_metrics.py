import numpy as np
from mpsio_eco.metrics import compute_metrics


def test_binary_metrics_return_macro_scores_and_specificity():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    metrics, per_class, cm = compute_metrics(y, pred, prob, ["N", "A"])
    assert cm.shape == (2, 2)
    assert 0 <= metrics["accuracy"] <= 1
    assert np.isfinite(metrics["macro_roc_auc_ovr"])
    assert "specificity" in per_class.columns


def test_multiclass_metrics_keep_all_class_rows():
    y = np.array([0, 1, 2, 0, 1, 2])
    pred = y.copy()
    prob = np.eye(3)[y] * 0.8 + 0.2 / 3
    metrics, per_class, cm = compute_metrics(y, pred, prob, ["A", "B", "C"])
    assert metrics["accuracy"] == 1.0
    assert len(per_class) == 3
    assert cm.shape == (3, 3)
