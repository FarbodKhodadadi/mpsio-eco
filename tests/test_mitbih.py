import numpy as np
import pytest

from mpsio_eco.mitbih import load_mitbih_beats, make_record_group_split


def test_mitbih_loader_and_record_disjoint_split(tmp_path):
    time = np.arange(1200, dtype=float)
    signal = np.sin(time / 17.0)
    for record in range(12):
        csv_path = tmp_path / f"record_{record:02d}.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("sample,signal\n")
            for idx, value in enumerate(signal):
                handle.write(f"{idx},{value}\n")
        txt_path = tmp_path / f"record_{record:02d}.txt"
        txt_path.write_text(
            "annotation_index sample symbol\n"
            "0 250 N\n"
            "1 500 V\n"
            "2 850 F\n",
            encoding="utf-8",
        )

    beats = load_mitbih_beats(
        tmp_path, retained_symbols=("N", "V", "F"), window_size=40,
        signal_column=1, min_support=1,
    )
    assert beats.signals.shape == (36, 1, 80)
    assert beats.class_names == ["N", "V", "F"]
    assert set(np.unique(beats.labels)) == {0, 1, 2}

    train, val, test = make_record_group_split(beats.labels, beats.groups, seed=12)
    groups_train = set(beats.groups[train])
    groups_val = set(beats.groups[val])
    groups_test = set(beats.groups[test])
    assert not groups_train & groups_val
    assert not groups_train & groups_test
    assert not groups_val & groups_test
    for part in (train, val, test):
        assert set(np.unique(beats.labels[part])) == {0, 1, 2}


def test_mitbih_requires_matching_csv_and_annotation(tmp_path):
    (tmp_path / "only_signal.csv").write_text("sample,signal\n0,0\n1,1\n")
    with pytest.raises(FileNotFoundError):
        load_mitbih_beats(tmp_path)
