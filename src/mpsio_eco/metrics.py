from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n_classes = len(class_names)
    labels = np.arange(n_classes)
    if probabilities.shape != (len(y_true), n_classes):
        raise ValueError(
            f"probabilities must have shape {(len(y_true), n_classes)}, "
            f"got {probabilities.shape}"
        )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        output_dict=True, zero_division=0,
    )
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    y_onehot = np.eye(n_classes, dtype=np.int64)[y_true]
    try:
        if n_classes == 2:
            metrics["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            metrics["macro_roc_auc_ovr"] = float(
                roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = float("nan")
    try:
        metrics["macro_auprc"] = float(average_precision_score(y_onehot, probabilities, average="macro"))
        metrics["weighted_auprc"] = float(average_precision_score(y_onehot, probabilities, average="weighted"))
    except ValueError:
        metrics["macro_auprc"] = float("nan")
        metrics["weighted_auprc"] = float("nan")

    per_class = []
    total = int(cm.sum())
    for index, name in enumerate(class_names):
        tp = int(cm[index, index])
        fn = int(cm[index, :].sum() - tp)
        fp = int(cm[:, index].sum() - tp)
        tn = total - tp - fn - fp
        try:
            ap = float(average_precision_score(y_onehot[:, index], probabilities[:, index]))
        except ValueError:
            ap = float("nan")
        row = report[name]
        per_class.append({
            "class": name,
            "support": int(row["support"]),
            "precision": float(row["precision"]),
            "recall_sensitivity": float(row["recall"]),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "f1": float(row["f1-score"]),
            "auprc": ap,
        })
    return metrics, pd.DataFrame(per_class), cm


def save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.65), max(5, len(class_names) * 0.55)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True, cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
