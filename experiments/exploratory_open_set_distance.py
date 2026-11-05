"""
Optional notebook-derived open-set latent-distance analysis.

"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def nearest_center_distance(x: np.ndarray, centers: dict[int, np.ndarray]):
    class_ids = sorted(centers)
    distances = np.stack([np.linalg.norm(x - centers[c], axis=1) for c in class_ids], axis=1)
    nearest_col = distances.argmin(axis=1)
    return distances.min(axis=1), np.asarray(class_ids)[nearest_col]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-latents", required=True, help="Training/validation/test latent_arrays.npz")
    parser.add_argument("--unknown-latents", required=True, help="NPZ containing array 'latents' for held-out class")
    parser.add_argument("--output", default="outputs/open_set_exploratory")
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    with np.load(args.known_latents, allow_pickle=False) as known:
        x_train = known["train_latents"]
        y_train = known["train_labels"]
        x_val = known["validation_latents"]
        x_test = known["test_latents"]
        y_test = known["test_labels"]
    with np.load(args.unknown_latents, allow_pickle=False) as unknown:
        if "latents" not in unknown.files:
            raise ValueError("Unknown-latents NPZ must contain an array named 'latents'")
        x_unknown = unknown["latents"]

    scaler = StandardScaler().fit(x_train)
    z_train = scaler.transform(x_train)
    z_val = scaler.transform(x_val)
    z_test = scaler.transform(x_test)
    z_unknown = scaler.transform(x_unknown)
    class_centers = {int(c): z_train[y_train == c].mean(axis=0) for c in np.unique(y_train)}
    val_distance, _ = nearest_center_distance(z_val, class_centers)
    threshold = float(np.percentile(val_distance, args.percentile))
    test_distance, test_nearest = nearest_center_distance(z_test, class_centers)
    unknown_distance, unknown_nearest = nearest_center_distance(z_unknown, class_centers)
    test_rejected = test_distance > threshold
    unknown_detected = unknown_distance > threshold

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {"split": "known_test", "n": len(test_distance), "mean_nearest_center_distance": float(test_distance.mean()),
         "rejected_as_unknown": int(test_rejected.sum()), "rejection_rate": float(test_rejected.mean())},
        {"split": "heldout_unknown", "n": len(unknown_distance), "mean_nearest_center_distance": float(unknown_distance.mean()),
         "rejected_as_unknown": int(unknown_detected.sum()), "rejection_rate": float(unknown_detected.mean())},
    ]
    pd.DataFrame(rows).to_csv(output / "open_set_summary.csv", index=False)
    pd.DataFrame({"distance": test_distance, "threshold": threshold, "predicted_known_class": test_nearest,
                  "true_known_class": y_test, "rejected": test_rejected}).to_csv(output / "known_test_distances.csv", index=False)
    pd.DataFrame({"distance": unknown_distance, "threshold": threshold,
                  "nearest_known_class": unknown_nearest, "detected_unknown": unknown_detected}).to_csv(
                      output / "unknown_distances.csv", index=False)
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold_percentile": args.percentile, "threshold_from_validation": threshold,
                   "note": "Exploratory only; not a reported manuscript experiment."}, f, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
