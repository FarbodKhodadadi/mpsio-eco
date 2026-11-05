from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


@dataclass
class ECGData:
    signals: np.ndarray  # [N,1,T], float32
    labels: np.ndarray  # [N], int64
    class_names: list[str]
    groups: Optional[np.ndarray]
    manifest: pd.DataFrame


class ArrayECGDataset(Dataset):

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(signals, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected signals [N,1,T], got {x.shape}")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("labels must be one-dimensional and match signals")
        self.signals = torch.from_numpy(np.ascontiguousarray(x))
        self.labels = torch.from_numpy(np.ascontiguousarray(y))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.signals[index], self.labels[index]


def _read_numeric_signal(
    path: Path,
    lead_index: int = 0,
    expected_length: Optional[int] = 5000,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path)
        if raw.ndim == 1:
            signal = raw.astype(np.float64)
        elif raw.ndim == 2:
            if raw.shape[0] < raw.shape[1]:
                raw = raw.T
            if lead_index >= raw.shape[1]:
                raise ValueError(f"lead_index={lead_index} unavailable in {path}")
            signal = raw[:, lead_index].astype(np.float64)
        else:
            raise ValueError(f"Unsupported NPY dimensions {raw.shape} in {path}")
    elif suffix in {".csv", ".txt"}:
        sep = "," if suffix == ".csv" else None
        frame = pd.read_csv(path, sep=sep, engine="python")
        frame = frame.loc[:, [c for c in frame.columns if not str(c).strip().lower().startswith("unnamed:")]]
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] <= lead_index:
            frame = pd.read_csv(path, sep=sep, engine="python", header=None)
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.dropna(axis=1, how="all")
        if numeric.shape[1] <= lead_index:
            raise ValueError(f"Numeric lead {lead_index} is absent in {path}")
        signal = numeric.iloc[:, lead_index].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported recording extension '{suffix}' for {path}")

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 2:
        raise ValueError(f"Recording is empty or too short: {path}")
    signal[~np.isfinite(signal)] = np.nan
    if np.isnan(signal).any():
        signal = pd.Series(signal).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Could not repair NaN/Inf values in {path}")
    if expected_length is not None and len(signal) != expected_length:
        signal = resample(signal, int(expected_length))
    std = float(np.std(signal))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError(f"Constant or near-constant ECG signal: {path}")
    signal = (signal - float(np.mean(signal))) / (std + 1e-8)
    return signal.astype(np.float32)


def load_manifest(
    manifest_path: str | Path,
    data_root: str | Path = ".",
    classes: Optional[Sequence[str]] = None,
    lead_index: int = 0,
    expected_length: Optional[int] = 5000,
) -> ECGData:
    """Load records described by a manifest.

    Required columns: ``recording_path`` and ``label``.
    Recommended column: ``patient_id`` (used for grouped, patient-level splits).
    Optional column: ``record_id``.

    ``classes`` specifies the exact labels to retain and their output order. It
    is required for the two-/five-class tasks unless all labels are intended.
    """
    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path.resolve()}")
    df = pd.read_csv(manifest_path)
    required = {"recording_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["recording_path", "label"]).copy()
    df["recording_path"] = df["recording_path"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    if classes:
        selected = [str(c).strip() for c in classes]
        if len(selected) != len(set(selected)):
            raise ValueError("The requested class list contains duplicates")
        observed = set(df["label"])
        absent = [name for name in selected if name not in observed]
        if absent:
            raise ValueError(f"Requested classes absent from manifest: {absent}")
        df = df[df["label"].isin(selected)].copy()
        label_to_id = {name: index for index, name in enumerate(selected)}
        class_names = selected
        y = df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(df["label"].to_numpy()).astype(np.int64)
        class_names = [str(name) for name in encoder.classes_]

    if len(class_names) < 2:
        raise ValueError(f"At least two classes are needed; found {class_names}")
    if df.empty:
        raise ValueError("No records remain after applying the class filter")

    signals: list[np.ndarray] = []
    retained_rows: list[int] = []
    skipped: list[tuple[str, str]] = []
    for row_index, record in enumerate(df.itertuples(index=False)):
        values = record._asdict()
        rel_path = Path(values["recording_path"])
        path = rel_path if rel_path.is_absolute() else data_root / rel_path
        try:
            signal = _read_numeric_signal(path, lead_index=lead_index, expected_length=expected_length)
        except Exception as exc:
            skipped.append((str(path), str(exc)))
            continue
        signals.append(signal)
        retained_rows.append(row_index)

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} unreadable record(s); first examples: {skipped[:3]}",
            RuntimeWarning,
        )
    if not signals:
        raise RuntimeError("No ECG recordings could be loaded from the manifest")

    kept_df = df.iloc[retained_rows].reset_index(drop=True)
    if classes:
        label_to_id = {name: index for index, name in enumerate(class_names)}
        y_kept = kept_df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    else:
        enc = LabelEncoder().fit(kept_df["label"].to_numpy())
        class_names = [str(name) for name in enc.classes_]
        y_kept = enc.transform(kept_df["label"].to_numpy()).astype(np.int64)

    X = np.stack(signals, axis=0)[:, None, :].astype(np.float32)
    if len(X) != len(y_kept):
        raise RuntimeError("Internal signal/label alignment failure")

    groups = None
    if "patient_id" in kept_df.columns:
        raw_groups = kept_df["patient_id"]
        if raw_groups.notna().all() and (raw_groups.astype(str).str.strip() != "").all():
            groups = raw_groups.astype(str).to_numpy()

    kept_df["encoded_label"] = y_kept
    print(f"Loaded ECGs: {len(X):,}")
    print(f"Signal shape: {X.shape}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print("Class counts:", {name: int(np.sum(y_kept == i)) for i, name in enumerate(class_names)})
    if groups is None:
        warnings.warn(
            "No patient_id/record_id groups were found. Patient-independent splitting "
            "cannot be guaranteed. Add a patient_id column before publication.",
            RuntimeWarning,
        )
    return ECGData(signals=X, labels=y_kept, class_names=class_names, groups=groups, manifest=kept_df)


def build_manifest_from_workbook(
    label_workbook: str | Path,
    ecg_directory: str | Path,
    output_csv: str | Path,
    filename_column: str = "FileName",
    label_column: str = "Rhythm",
    patient_column: Optional[str] = None,
) -> pd.DataFrame:
    """Match ``Data/ECGData/*.csv`` files to labels in an Excel workbook.

    This is the adapter for the supplied ``Diagnostics.xlsx`` workflow. It does
    not infer patient IDs: pass a real patient identifier column when available.
    """
    label_workbook = Path(label_workbook)
    ecg_directory = Path(ecg_directory)
    output_csv = Path(output_csv)
    if not label_workbook.exists():
        raise FileNotFoundError(label_workbook)
    if not ecg_directory.exists():
        raise FileNotFoundError(ecg_directory)
    table = pd.read_excel(label_workbook)
    lookup = {str(col).strip().casefold(): col for col in table.columns}
    for required in (filename_column, label_column):
        if required.strip().casefold() not in lookup:
            raise ValueError(f"Column {required!r} not found; available: {list(table.columns)}")
    file_col = lookup[filename_column.strip().casefold()]
    label_col = lookup[label_column.strip().casefold()]
    patient_col = None
    if patient_column:
        if patient_column.strip().casefold() not in lookup:
            raise ValueError(f"Patient column {patient_column!r} not found")
        patient_col = lookup[patient_column.strip().casefold()]

    selected_cols = [file_col, label_col] + ([patient_col] if patient_col else [])
    table = table[selected_cols].dropna(subset=[file_col, label_col]).copy()

    def stem(value: object) -> str:
        return Path(str(value).replace("\\", "/")).stem.strip().casefold()

    label_map: dict[str, tuple[str, Optional[str]]] = {}
    for row in table.itertuples(index=False, name=None):
        key = stem(row[0])
        label = str(row[1]).strip()
        patient = str(row[2]).strip() if patient_col and pd.notna(row[2]) else None
        if key in label_map and label_map[key][0] != label:
            raise ValueError(f"Conflicting labels for workbook filename stem {key!r}")
        label_map[key] = (label, patient)

    rows = []
    for path in sorted(ecg_directory.glob("*.csv")):
        key = stem(path.name)
        if key not in label_map:
            continue
        label, patient = label_map[key]
        rows.append({
            "recording_path": str(path.resolve()),
            "label": label,
            "patient_id": patient if patient is not None else "",
            "record_id": key,
        })
    if not rows:
        raise ValueError("No CSV files matched workbook FileName values by filename stem")
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    if result["patient_id"].eq("").all():
        warnings.warn(
            "The workbook did not provide a patient identifier. The resulting manifest "
            "cannot support patient-independent partitions until patient_id is added.",
            RuntimeWarning,
        )
    print(f"Wrote {len(result):,} matched records to {output_csv}")
    return result
