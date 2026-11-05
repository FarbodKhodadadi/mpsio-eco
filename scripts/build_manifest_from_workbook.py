from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.data import build_manifest_from_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/Diagnostics.xlsx", help="Excel label workbook")
    parser.add_argument("--ecg-dir", default="data/ECGData", help="Directory containing ECG CSV files")
    parser.add_argument("--output", default="data/manifest.csv")
    parser.add_argument("--filename-column", default="FileName")
    parser.add_argument("--label-column", default="Rhythm")
    parser.add_argument("--patient-column", default=None, help="Optional patient ID column in workbook")
    args = parser.parse_args()
    build_manifest_from_workbook(
        args.labels, args.ecg_dir, args.output,
        filename_column=args.filename_column,
        label_column=args.label_column,
        patient_column=args.patient_column,
    )

if __name__ == "__main__":
    main()
