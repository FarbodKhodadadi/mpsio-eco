# Data directory

Raw ECG files and patient-level metadata are intentionally not distributed in
this repository. Do not commit identifiable patient data, restricted datasets,
large binary waveforms, or credentials to GitHub.

## Manifest schema

Create `data/manifest.csv` with at least these columns:

| Column | Required | Meaning |
|---|---:|---|
| `recording_path` | Yes | Path to one ECG CSV or NPY file, relative to `--data-root` or absolute |
| `label` | Yes | Final single target class for this record |
| `patient_id` | Strongly recommended | Patient-level grouping ID for leakage-controlled splitting |
| `record_id` | Optional | Source recording ID for traceability |

See `manifest_template.csv` for the header. CSV files should contain numeric
lead columns; the training pipeline selects `--lead-index` (default 0),
resamples to 5,000 samples by default, interpolates missing values and
standardizes each recording independently. NPY files may be 1-D (one lead) or
2-D (time × leads).

## Supplied workbook adapter

For the supplied workbook layout (`Diagnostics.xlsx`, columns `FileName` and
`Rhythm`, and `Data/ECGData/*.csv`), run:

```bash
python scripts/build_manifest_from_workbook.py \
  --labels data/Diagnostics.xlsx \
  --ecg-dir data/ECGData \
  --output data/manifest.csv
```

When the workbook contains a real patient-ID column, pass it using
`--patient-column PatientID` (replace with the exact column name). If it does
not, add patient IDs to the manifest from an authorized source before using the
patient-independent protocol.

## PTB-XL preparation

The final manuscript identifies PTB-XL as the primary dataset. This repository
requires a manifest that already encodes the manuscript's selected single-lead
records and final task labels. The uploaded code did not include the complete
PTB-XL label-mapping/preprocessing cells, nor the exact 2- and 5-class label
lists. Therefore, the repository deliberately does not guess the multi-label
SCP-ECG-to-11-class mapping. Prepare a manifest from your licensed/local PTB-XL
copy and document the mapping used in the paper before claiming exact
reproducibility.
