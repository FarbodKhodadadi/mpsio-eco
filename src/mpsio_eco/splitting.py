from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class FoldIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    fold: int
    seed: int


def _assert_disjoint_groups(groups: Optional[np.ndarray], *indices: np.ndarray) -> None:
    if groups is None:
        return
    group_sets = [set(groups[index].tolist()) for index in indices]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            overlap = group_sets[i].intersection(group_sets[j])
            if overlap:
                raise RuntimeError(f"Patient/record leakage: {len(overlap)} group(s) overlap")


def make_cv_splits(
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    seed: int,
    n_splits: int = 5,
    allow_record_level: bool = False,
) -> Iterator[FoldIndices]:
    y = np.asarray(labels, dtype=np.int64)
    if groups is None:
        if not allow_record_level:
            raise ValueError(
                "Patient IDs are required for the default patient-independent protocol. "
                "Add patient_id to manifest.csv or explicitly pass --allow-record-level-split."
            )
        warnings.warn("Using record-level rather than patient-level splitting.", RuntimeWarning)
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (trainval_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y), 1):
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            train_rel, val_rel = next(inner.split(np.zeros(len(trainval_idx)), y[trainval_idx]))
            train_idx = trainval_idx[train_rel]
            val_idx = trainval_idx[val_rel]
            yield FoldIndices(train_idx, val_idx, test_idx, fold, seed)
        return

    groups = np.asarray(groups).astype(str)
    if len(groups) != len(y):
        raise ValueError("groups and labels have different lengths")
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique patient groups; found {len(unique_groups)}")

    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (trainval_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(y)), y, groups), 1
    ):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
        train_rel, val_rel = next(
            inner.split(
                np.zeros(len(trainval_idx)),
                y[trainval_idx],
                groups[trainval_idx],
            )
        )
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        _assert_disjoint_groups(groups, train_idx, val_idx, test_idx)
        yield FoldIndices(train_idx, val_idx, test_idx, fold, seed)


def make_seeded_holdout(
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    seed: int,
    allow_record_level: bool = False,
) -> FoldIndices:
    """Return one approximately 64/16/20 split for one random seed."""
    return next(make_cv_splits(labels, groups, seed, n_splits=5,
                               allow_record_level=allow_record_level))
