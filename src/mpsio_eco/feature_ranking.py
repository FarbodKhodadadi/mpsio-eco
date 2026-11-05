from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .metrics import compute_metrics, save_confusion_matrix

DEFAULT_TOP_K = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)


def run_feature_ranking(
    latent_file: str | Path,
    output_dir: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    ranking_estimators: int = 400,
    pruning_estimators: int = 250,
    random_state: int = 42,
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
) -> pd.DataFrame:

    latent_file = Path(latent_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not latent_file.exists():
        raise FileNotFoundError(latent_file)
    with np.load(latent_file, allow_pickle=False) as data:
        required = {
            "train_latents", "train_labels", "validation_latents", "validation_labels",
            "test_latents", "test_labels",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Latent archive is missing arrays: {sorted(missing)}")
        x_train = data["train_latents"].astype(np.float32)
        y_train = data["train_labels"].astype(np.int64)
        x_val = data["validation_latents"].astype(np.float32)
        y_val = data["validation_labels"].astype(np.int64)
        x_test = data["test_latents"].astype(np.float32)
        y_test = data["test_labels"].astype(np.int64)
        saved_class_names = data["class_names"].astype(str).tolist() if "class_names" in data.files else None

    if x_train.ndim != 2 or x_val.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Latent arrays must be 2-D [samples, dimensions]")
    if not (x_train.shape[1] == x_val.shape[1] == x_test.shape[1]):
        raise ValueError("Train/validation/test latent dimensions do not match")
    n_features = x_train.shape[1]
    if class_names is None:
        if saved_class_names is not None:
            class_names = saved_class_names
        else:
            max_id = int(max(y_train.max(), y_val.max(), y_test.max()))
            class_names = [str(i) for i in range(max_id + 1)]

    ranker = RandomForestClassifier(
        n_estimators=ranking_estimators,
        class_weight="balanced_subsample",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    ranker.fit(x_train, y_train)
    importances = ranker.feature_importances_
    order = np.argsort(importances)[::-1]
    ranking = pd.DataFrame({
        "rank": np.arange(1, n_features + 1),
        "feature_index": order,
        "feature_name": [f"latent_{index:02d}" for index in order],
        "importance": importances[order],
    })
    ranking.to_csv(output_dir / "random_forest_feature_ranking.csv", index=False)

    rows = []
    selected_best_k: int | None = None
    best_val_macro_f1 = -np.inf
    for k in top_k_values:
        k = int(k)
        if k < 1 or k > n_features:
            continue
        selected = order[:k]
        clf = RandomForestClassifier(
            n_estimators=pruning_estimators,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(x_train[:, selected], y_train)
        pred = clf.predict(x_val[:, selected])
        accuracy = float(accuracy_score(y_val, pred))
        macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, pred, average="weighted", zero_division=0))
        rows.append({
            "retained_features": k,
            "pruned_features": n_features - k,
            "retained_percent": 100.0 * k / n_features,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
        })
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            selected_best_k = k

    pruning = pd.DataFrame(rows)
    pruning.to_csv(output_dir / "feature_ranking_pruning_validation.csv", index=False)
    if not pruning.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pruning["retained_features"], pruning["validation_accuracy"], marker="o", label="Accuracy")
        ax.plot(pruning["retained_features"], pruning["validation_macro_f1"], marker="s", label="Macro-F1")
        ax.plot(pruning["retained_features"], pruning["validation_weighted_f1"], marker="^", label="Weighted-F1")
        ax.set_xlabel("Retained latent dimensions")
        ax.set_ylabel("Validation score")
        ax.set_title("Random-Forest performance after latent-feature pruning")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "feature_ranking_pruning_curve.png", dpi=300)
        plt.close(fig)

    if selected_best_k is not None and selected_best_k >= 2:
        selected = order[:selected_best_k]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train[:, selected])
        x_train_scaled = scaler.transform(x_train[:, selected])
        x_test_scaled = scaler.transform(x_test[:, selected])
        pca_top = PCA(n_components=2, random_state=random_state).fit(x_train_scaled)
        z_train = pca_top.transform(x_train_scaled)
        z_test = pca_top.transform(x_test_scaled)
        fig, ax = plt.subplots(figsize=(9, 7))
        for class_id, name in enumerate(class_names):
            mask = y_train == class_id
            if mask.any():
                ax.scatter(z_train[mask, 0], z_train[mask, 1], s=12, alpha=0.3, label=f"train: {name}")
            test_mask = y_test == class_id
            if test_mask.any():
                ax.scatter(z_test[test_mask, 0], z_test[test_mask, 1], s=24, marker="x", label=f"test: {name}")
        ax.set_title(f"PCA after RF feature ranking (top {selected_best_k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_after_rf_feature_ranking.png", dpi=300)
        plt.close(fig)
        pd.DataFrame({
            "feature_index": selected,
            "feature_name": [f"latent_{index:02d}" for index in selected],
            "importance": importances[selected],
        }).to_csv(output_dir / "pca_after_rf_selected_features.csv", index=False)

    final_rf = RandomForestClassifier(
        n_estimators=pruning_estimators,
        class_weight="balanced_subsample",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    final_rf.fit(x_train, y_train)
    test_pred = final_rf.predict(x_test)
    test_prob_raw = final_rf.predict_proba(x_test)
    test_prob = np.zeros((len(x_test), len(class_names)), dtype=np.float64)
    test_prob[:, final_rf.classes_.astype(int)] = test_prob_raw
    metrics, per_class, cm = compute_metrics(y_test, test_pred, test_prob, class_names)
    pd.DataFrame([metrics]).to_csv(output_dir / "random_forest_full_latents_test_metrics.csv", index=False)
    per_class.to_csv(output_dir / "random_forest_full_latents_per_class.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "random_forest_full_latents_confusion_matrix.csv")
    save_confusion_matrix(cm, class_names, output_dir / "random_forest_full_latents_confusion_matrix.png")

    if n_features >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        train_2d = pca.fit_transform(x_train)
        test_2d = pca.transform(x_test)
        fig, ax = plt.subplots(figsize=(9, 7))
        for class_id, name in enumerate(class_names):
            mask = y_train == class_id
            if mask.any():
                ax.scatter(train_2d[mask, 0], train_2d[mask, 1], s=12, alpha=0.35, label=name)
        ax.set_title("PCA of Alpha attention-derived training embeddings")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_attention_latents_train.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        for class_id, name in enumerate(class_names):
            mask = y_test == class_id
            if mask.any():
                ax.scatter(test_2d[mask, 0], test_2d[mask, 1], s=20, alpha=0.7, label=name)
        ax.set_title("PCA of Alpha attention-derived test embeddings")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_attention_latents_test.png", dpi=300)
        plt.close(fig)

    with (output_dir / "feature_ranking_metadata.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Input latent dimensions: {n_features}\n")
        handle.write(f"Validation-selected k (by macro-F1): {selected_best_k}\n")
        handle.write("Final RF test score is evaluated on all dimensions; test data did not select k.\n")
        handle.write("Feature importance is model-specific predictive importance, not physiological interpretation.\n")

    return pruning
