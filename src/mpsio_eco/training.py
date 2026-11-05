from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .data import ArrayECGDataset, ECGData
from .losses import SeparationIndexLoss
from .metrics import compute_metrics, save_confusion_matrix
from .models import build_model, count_trainable_parameters
from .seed import DEFAULT_SEEDS, get_device, set_seed
from .splitting import FoldIndices, make_cv_splits, make_seeded_holdout


DEFAULT_MODELS = ("alpha", "alpha_ce", "beta", "gamma", "delta", "epsilon")


def _make_train_loader(
    dataset: ArrayECGDataset,
    indices: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    sampling: str,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": generator,
    }
    if sampling == "weighted":
        fold_y = labels[indices]
        counts = np.bincount(fold_y, minlength=int(labels.max()) + 1)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = weights[fold_y]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(subset, sampler=sampler, **common)
    if sampling != "none":
        raise ValueError("sampling must be 'none' or 'weighted'")
    return DataLoader(subset, shuffle=True, **common)


def _make_eval_loader(
    dataset: ArrayECGDataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )


def _loss_for_batch(
    model: nn.Module,
    model_name: str,
    criterion: nn.Module,
    separation_loss: SeparationIndexLoss,
    x: torch.Tensor,
    y: torch.Tensor,
    separation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, embedding = model(x, return_embedding=True)
    ce = criterion(logits, y)
    si = separation_loss(embedding, y) if model_name == "alpha" else ce.new_tensor(0.0)
    total = ce + separation_weight * si if model_name == "alpha" else ce
    return total, logits, si


def _train_one_fold(
    data: ECGData,
    split: FoldIndices,
    model_name: str,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    separation_weight: float,
    sampling: str,
    num_workers: int,
    device: torch.device,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    set_seed(split.seed + split.fold * 1009)
    dataset = ArrayECGDataset(data.signals, data.labels)
    train_loader = _make_train_loader(
        dataset, split.train, data.labels, batch_size, sampling,
        split.seed + split.fold, num_workers, device,
    )
    train_eval_loader = _make_eval_loader(dataset, split.train, batch_size, num_workers, device)
    val_loader = _make_eval_loader(dataset, split.validation, batch_size, num_workers, device)
    test_loader = _make_eval_loader(dataset, split.test, batch_size, num_workers, device)

    model = build_model(model_name, len(data.class_names), **(model_kwargs or {})).to(device)
    criterion = nn.CrossEntropyLoss()
    separation_loss = SeparationIndexLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_total: list[float] = []
        train_ce: list[float] = []
        train_si: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, si = _loss_for_batch(
                model, model_name, criterion, separation_loss,
                batch_x, batch_y, separation_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss: model={model_name}, seed={split.seed}, "
                    f"fold={split.fold}, epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total.append(float(loss.detach().cpu()))
            train_ce.append(float((loss.detach() - separation_weight * si.detach()).cpu())
                             if model_name == "alpha" else float(loss.detach().cpu()))
            train_si.append(float(si.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                loss, _, _ = _loss_for_batch(
                    model, model_name, criterion, separation_loss,
                    batch_x, batch_y, separation_weight,
                )
                val_losses.append(float(loss.cpu()))
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_total))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cross_entropy": float(np.mean(train_ce)),
            "train_separation_index": float(np.mean(train_si)),
            "validation_loss": val_loss,
        }
        history.append(row)
        print(
            f"{model_name} | seed={split.seed} fold={split.fold} | "
            f"epoch={epoch:02d}/{epochs} train={train_loss:.5f} val={val_loss:.5f}"
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        latent_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        prob_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device, non_blocking=True)
                logits, embedding = model(bx, return_embedding=True)
                probabilities = torch.softmax(logits, dim=1)
                latent_parts.append(embedding.cpu().numpy())
                label_parts.append(by.numpy())
                prob_parts.append(probabilities.cpu().numpy())
                pred_parts.append(logits.argmax(dim=1).cpu().numpy())
        return (
            np.concatenate(latent_parts), np.concatenate(label_parts),
            np.concatenate(prob_parts), np.concatenate(pred_parts),
        )

    train_latents, train_labels, _, _ = collect(train_eval_loader)
    val_latents, val_labels, val_probs, val_preds = collect(val_loader)
    test_latents, test_labels, test_probs, test_preds = collect(test_loader)

    metrics, per_class, cm = compute_metrics(
        test_labels, test_preds, test_probs, data.class_names,
    )
    fold_dir = output_dir / model_name / f"seed_{split.seed}" / f"fold_{split.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    _plot_history(history, fold_dir / "training_curves.png", model_name)
    per_class.to_csv(fold_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=data.class_names, columns=data.class_names).to_csv(fold_dir / "confusion_matrix.csv")
    save_confusion_matrix(cm, data.class_names, fold_dir / "confusion_matrix.png")
    pd.DataFrame({
        "true_label": test_labels,
        "true_class": [data.class_names[int(i)] for i in test_labels],
        "predicted_label": test_preds,
        "predicted_class": [data.class_names[int(i)] for i in test_preds],
        "max_probability": test_probs.max(axis=1),
    }).to_csv(fold_dir / "test_predictions.csv", index=False)
    np.savez_compressed(
        fold_dir / "latent_arrays.npz",
        train_latents=train_latents, train_labels=train_labels,
        validation_latents=val_latents, validation_labels=val_labels,
        test_latents=test_latents, test_labels=test_labels,
        test_probabilities=test_probs, test_predictions=test_preds,
        class_names=np.asarray(data.class_names, dtype=str),
    )

    checkpoint = {
        "model_name": model_name,
        "num_classes": len(data.class_names),
        "class_names": list(data.class_names),
        "model_kwargs": model_kwargs or {},
        "model_state_dict": best_state,
        "seed": split.seed,
        "fold": split.fold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, fold_dir / "best_model.pt")
    with (fold_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "model": model_name,
            "seed": split.seed,
            "fold": split.fold,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "trainable_parameters": count_trainable_parameters(model),
            "n_train": len(split.train),
            "n_validation": len(split.validation),
            "n_test": len(split.test),
            "class_names": list(data.class_names),
            "separation_weight": separation_weight if model_name == "alpha" else None,
            "sampling": sampling,
        }, handle, indent=2)

    result = {
        "model": model_name,
        "seed": split.seed,
        "fold": split.fold,
        "best_epoch": best_epoch,
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        **metrics,
        "checkpoint": str((fold_dir / "best_model.pt").as_posix()),
    }
    return result


def _plot_history(history: list[dict[str, float]], output_path: Path, model_name: str) -> None:
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], label="Training")
    ax.plot(frame["epoch"], frame["validation_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name.upper()} training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_experiments(
    data: ECGData,
    output_dir: str | Path,
    models: Sequence[str] = DEFAULT_MODELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    protocol: str = "cv",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    separation_weight: float = 1.0,
    sampling: str = "weighted",
    num_workers: int = 0,
    device_name: str = "auto",
    allow_record_level: bool = False,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run one or more model variants under CV or repeated seeded holdout.

    ``cv`` runs 5 folds for each seed. ``seeded_holdout`` runs one 64/16/20
    split per seed; it is useful when reproducing the manuscript's matched
    five-seed comparison on a fixed held-out protocol.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device_name)
    if protocol not in {"cv", "seeded_holdout"}:
        raise ValueError("protocol must be 'cv' or 'seeded_holdout'")
    results: list[dict[str, Any]] = []
    for model_name in models:
        model_name = model_name.strip().lower().replace("-", "_")
        for seed in seeds:
            if protocol == "cv":
                splits = make_cv_splits(data.labels, data.groups, int(seed), 5,
                                        allow_record_level=allow_record_level)
            else:
                splits = [make_seeded_holdout(data.labels, data.groups, int(seed),
                                              allow_record_level=allow_record_level)]
            for split in splits:
                result = _train_one_fold(
                    data, split, model_name, output_dir,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    separation_weight=separation_weight, sampling=sampling,
                    num_workers=num_workers, device=device, model_kwargs=model_kwargs,
                )
                results.append(result)
                pd.DataFrame(results).to_csv(output_dir / "metrics_all_runs.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "metrics_all_runs.csv", index=False)
    numeric_metrics = [
        "accuracy", "macro_precision", "weighted_precision", "macro_recall",
        "weighted_recall", "macro_f1", "weighted_f1", "macro_roc_auc_ovr",
        "macro_auprc", "weighted_auprc",
    ]
    summary = results_df.groupby("model")[numeric_metrics].agg(["mean", "std"])
    summary.to_csv(output_dir / "metrics_summary.csv")
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "protocol": protocol,
            "models": list(models),
            "seeds": [int(s) for s in seeds],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "separation_weight": separation_weight,
            "sampling": sampling,
            "device": str(device),
            "class_names": list(data.class_names),
            "patient_groups_available": data.groups is not None,
        }, handle, indent=2)
    return results_df
