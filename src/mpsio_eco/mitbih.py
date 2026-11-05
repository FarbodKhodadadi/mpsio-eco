from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd

from .data import ArrayECGDataset
from .losses import CenterMarginLoss, SeparationIndexLoss, SupervisedContrastiveLoss
from .metrics import compute_metrics, save_confusion_matrix
from .models import AlphaECGNet
from .seed import get_device, set_seed


@dataclass
class MITBIHBeats:
    signals: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    class_names: list[str]


def _read_signal_column(path: Path, signal_column: int = 1) -> np.ndarray:
    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) <= signal_column:
                continue
            try:
                values.append(float(row[signal_column]))
            except (ValueError, TypeError):
                continue
    signal = np.asarray(values, dtype=np.float32)
    if signal.size == 0:
        raise ValueError(f"No numeric samples found in {path} column {signal_column}")
    if not np.isfinite(signal).all():
        finite = np.isfinite(signal)
        if finite.sum() < 2:
            raise ValueError(f"Too few finite samples in {path}")
        positions = np.arange(len(signal))
        signal[~finite] = np.interp(positions[~finite], positions[finite], signal[finite])
    return signal


def load_mitbih_beats(
    data_dir: str | Path,
    retained_symbols: Sequence[str] = ("N", "V", "F"),
    window_size: int = 180,
    signal_column: int = 1,
    min_support: int = 1,
    normalize_beats: bool = False,
) -> MITBIHBeats:
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No MIT-BIH CSV files found in {data_dir}")
    retained = list(retained_symbols)
    if len(retained) < 2:
        raise ValueError("At least two retained beat symbols are required")

    signal_files: dict[str, Path] = {p.stem.casefold(): p for p in csv_files}
    annotation_files: dict[str, Path] = {p.stem.casefold(): p for p in data_dir.glob("*.txt")}
    pairs = sorted(set(signal_files).intersection(annotation_files))
    if not pairs:
        raise FileNotFoundError("No matching CSV/TXT files share the same filename stem")
    if set(signal_files) != set(annotation_files):
        warnings.warn(
            f"Found {len(signal_files)} signal CSVs and {len(annotation_files)} annotation TXT files; "
            f"processing {len(pairs)} matched stems.", RuntimeWarning,
        )

    beats: list[np.ndarray] = []
    symbols: list[str] = []
    record_ids: list[str] = []
    for record_stem in pairs:
        signal_path = signal_files[record_stem]
        annotation_path = annotation_files[record_stem]
        signal = _read_signal_column(signal_path, signal_column=signal_column)
        with annotation_path.open("r", encoding="utf-8-sig") as handle:
            lines = handle.readlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                position = int(parts[1])
            except ValueError:
                continue
            symbol = parts[2].strip()
            if symbol not in retained:
                continue
            start = position - window_size
            end = position + window_size
            if start < 0 or end > len(signal):
                continue
            segment = signal[start:end].copy()
            if len(segment) != 2 * window_size:
                continue
            if normalize_beats:
                std = float(segment.std())
                if std < 1e-8:
                    continue
                segment = (segment - segment.mean()) / (std + 1e-8)
            beats.append(segment)
            symbols.append(symbol)
            record_ids.append(record_stem)

    if not beats:
        raise RuntimeError("No annotated beat windows were extracted")
    raw_counts = {symbol: symbols.count(symbol) for symbol in retained}
    keep = [symbol for symbol in retained if raw_counts.get(symbol, 0) >= min_support]
    if len(keep) < 2:
        raise ValueError(f"Fewer than two classes meet min_support={min_support}: {raw_counts}")
    mask = np.asarray([symbol in keep for symbol in symbols], dtype=bool)
    x = np.stack(beats)[mask].astype(np.float32)[:, None, :]
    kept_symbols = np.asarray(symbols)[mask]
    groups = np.asarray(record_ids)[mask]
    mapping = {symbol: index for index, symbol in enumerate(keep)}
    y = np.asarray([mapping[symbol] for symbol in kept_symbols], dtype=np.int64)
    print("MIT-BIH beat counts:", {symbol: int(np.sum(y == i)) for i, symbol in enumerate(keep)})
    print(f"Beat array: {x.shape}; source records: {len(np.unique(groups))}")
    return MITBIHBeats(x, y, groups, keep)


def make_record_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.15,
    attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    g = np.asarray(groups).astype(str)
    if len(np.unique(g)) < 3:
        raise ValueError("At least three source records are needed for grouped train/val/test splits")
    for attempt in range(attempts):
        current_seed = seed + attempt
        outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=current_seed)
        trainval, test = next(outer.split(np.zeros(len(y)), y, groups=g))
        val_relative = validation_fraction / (1.0 - test_fraction)
        inner = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=current_seed + 101)
        train_rel, val_rel = next(inner.split(np.zeros(len(trainval)), y[trainval], groups=g[trainval]))
        train = trainval[train_rel]
        val = trainval[val_rel]
        def all_classes(index: np.ndarray) -> bool:
            return set(np.unique(y[index]).tolist()) == set(np.unique(y).tolist())
        if all(all_classes(index) for index in (train, val, test)):
            if set(g[train]) & set(g[val]) or set(g[train]) & set(g[test]) or set(g[val]) & set(g[test]):
                raise RuntimeError("Source-record leakage detected")
            return train, val, test
    raise RuntimeError(
        "Could not create record-independent splits containing every retained class. "
        "Reduce the number of retained classes or review per-record class support."
    )


def _collect_eval(model, loader, device):
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x, return_embedding=True)
            all_y.append(y.numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_prob.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_prob)


def run_mitbih_experiment(
    beats: MITBIHBeats,
    output_dir: str | Path,
    *,
    seed: int = 456,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_recipe: str = "supcon_center",
    device_name: str = "auto",
) -> dict[str, float | str]:
    
    if loss_recipe not in {"supcon_center", "mpsio", "ce"}:
        raise ValueError("loss_recipe must be supcon_center, mpsio or ce")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_idx, val_idx, test_idx = make_record_group_split(beats.labels, beats.groups, seed)
    dataset = ArrayECGDataset(beats.signals, beats.labels)
    train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset, test_idx.tolist())
    device = get_device(device_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")

    model = AlphaECGNet(num_classes=len(beats.class_names)).to(device)
    ce_loss = nn.CrossEntropyLoss()
    si_loss = SeparationIndexLoss()
    supcon_loss = SupervisedContrastiveLoss(temperature=0.1)
    center_loss = CenterMarginLoss(margin=1.0, normalize=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, embedding = model(x, return_embedding=True)
            ce = ce_loss(logits, y)
            if loss_recipe == "supcon_center":
                loss = ce + 0.5 * supcon_loss(embedding, y) + 0.2 * center_loss(embedding, y)
            elif loss_recipe == "mpsio":
                loss = ce + si_loss(embedding, y)
            else:
                loss = ce
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at MIT-BIH epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, embedding = model(x, return_embedding=True)
                ce = ce_loss(logits, y)
                if loss_recipe == "supcon_center":
                    loss = ce + 0.5 * supcon_loss(embedding, y) + 0.2 * center_loss(embedding, y)
                elif loss_recipe == "mpsio":
                    loss = ce + si_loss(embedding, y)
                else:
                    loss = ce
                val_losses.append(float(loss.cpu()))
        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_mean, "validation_loss": val_mean})
        print(f"MIT-BIH epoch={epoch:02d}/{epochs} train={train_mean:.5f} val={val_mean:.5f}")
        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No valid MIT-BIH checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_true, y_pred, y_prob = _collect_eval(model, test_loader, device)
    metrics, per_class, cm = compute_metrics(y_true, y_pred, y_prob, beats.class_names)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=beats.class_names, columns=beats.class_names).to_csv(output_dir / "confusion_matrix.csv")
    save_confusion_matrix(cm, beats.class_names, output_dir / "confusion_matrix.png")
    torch.save({
        "model_name": "alpha",
        "model_state_dict": best_state,
        "class_names": beats.class_names,
        "loss_recipe": loss_recipe,
        "seed": seed,
    }, output_dir / "mitbih_model.pt")
    groups_train = set(beats.groups[train_idx])
    groups_val = set(beats.groups[val_idx])
    groups_test = set(beats.groups[test_idx])
    if groups_train & groups_val or groups_train & groups_test or groups_val & groups_test:
        raise RuntimeError("MIT-BIH source-record leakage detected after training")
    pd.DataFrame([{
        **metrics,
        "loss_recipe": loss_recipe,
        "seed": seed,
        "n_train_beats": len(train_idx),
        "n_validation_beats": len(val_idx),
        "n_test_beats": len(test_idx),
        "n_train_records": len(groups_train),
        "n_validation_records": len(groups_val),
        "n_test_records": len(groups_test),
    }]).to_csv(output_dir / "test_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        class_names=np.asarray(beats.class_names),
    )
    return metrics
