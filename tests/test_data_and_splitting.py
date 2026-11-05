from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mpsio_eco.data import load_manifest
from mpsio_eco.splitting import make_cv_splits


def test_load_manifest_reads_csv_and_encodes_requested_classes(tmp_path):
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    rows = []
    rng = np.random.default_rng(2)
    for i in range(20):
        name = f"record_{i:02d}.csv"
        t = np.linspace(0, 2 * np.pi, 500)
        signal = np.sin(t * (1 + i % 2)) + rng.normal(0, 0.01, size=len(t))
        pd.DataFrame({"lead_I": signal}).to_csv(signal_dir / name, index=False)
        rows.append({"recording_path": f"signals/{name}", "label": "A" if i % 2 == 0 else "B",
                     "patient_id": f"patient_{i:02d}"})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    data = load_manifest(manifest, data_root=tmp_path, classes=["A", "B"], expected_length=500)
    assert data.signals.shape == (20, 1, 500)
    assert data.labels.shape == (20,)
    assert data.groups is not None


def test_grouped_cv_keeps_patient_ids_disjoint():
    y = np.tile(np.array([0, 1, 2, 0, 1, 2]), 10)
    groups = np.array([f"p{i}" for i in range(len(y))])
    splits = list(make_cv_splits(y, groups, seed=23, n_splits=5))
    assert len(splits) == 5
    for split in splits:
        train = set(groups[split.train])
        val = set(groups[split.validation])
        test = set(groups[split.test])
        assert not train & val
        assert not train & test
        assert not val & test


def test_record_level_split_requires_explicit_override():
    y = np.tile(np.array([0, 1]), 10)
    with pytest.raises(ValueError):
        list(make_cv_splits(y, None, seed=1, n_splits=5))
    splits = list(make_cv_splits(y, None, seed=1, n_splits=5, allow_record_level=True))
    assert len(splits) == 5
