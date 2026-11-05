from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_comparison(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    proposed_name: str = "MPSIO-ECO",
    baseline_name: str = "Alpha-CE",
) -> dict[str, float | int | str]:
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if proposed.ndim != 1 or baseline.ndim != 1 or len(proposed) != len(baseline):
        raise ValueError("Paired arrays must be 1-D and have equal lengths")
    if len(proposed) < 2:
        raise ValueError("At least two matched runs are needed")
    if not (np.isfinite(proposed).all() and np.isfinite(baseline).all()):
        raise ValueError("Paired accuracy arrays contain non-finite values")
    differences = proposed - baseline
    sd = float(differences.std(ddof=1))
    cohens_dz = float(differences.mean() / sd) if sd > 0 else float("nan")
    t_result = ttest_rel(proposed, baseline)
    try:
        w_result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        wilcoxon_p = float(w_result.pvalue)
        wilcoxon_statistic = float(w_result.statistic)
    except ValueError:
        wilcoxon_p = float("nan")
        wilcoxon_statistic = float("nan")
    return {
        "proposed_model": proposed_name,
        "baseline_model": baseline_name,
        "n_paired_runs": int(len(proposed)),
        "proposed_mean_accuracy": float(proposed.mean()),
        "proposed_sd_accuracy": float(proposed.std(ddof=1)),
        "baseline_mean_accuracy": float(baseline.mean()),
        "baseline_sd_accuracy": float(baseline.std(ddof=1)),
        "mean_paired_difference": float(differences.mean()),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p_value": float(t_result.pvalue),
        "cohens_dz": cohens_dz,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p,
    }


def analyze_results_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    proposed_model: str = "alpha",
    baseline_model: str = "alpha_ce",
    metric: str = "accuracy",
) -> dict[str, float | int | str]:
    frame = pd.read_csv(input_csv)
    required = {"seed", "model", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input CSV requires columns {sorted(required)}")
    proposed = (frame[frame["model"] == proposed_model]
                .groupby("seed", as_index=False)[metric].mean()
                .rename(columns={metric: "proposed"}))
    baseline = (frame[frame["model"] == baseline_model]
                .groupby("seed", as_index=False)[metric].mean()
                .rename(columns={metric: "baseline"}))
    paired = proposed.merge(baseline, on="seed", how="inner", validate="one_to_one").sort_values("seed")
    if len(paired) < 2:
        raise ValueError("Could not find at least two matched seeds for both model names")
    result = paired_comparison(
        paired["proposed"].to_numpy(), paired["baseline"].to_numpy(),
        proposed_name=proposed_model, baseline_name=baseline_model,
    )
    result["metric"] = metric
    result["matched_seeds"] = ",".join(map(str, paired["seed"].tolist()))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_values.csv", index=False)
    with (output_dir / "paired_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
