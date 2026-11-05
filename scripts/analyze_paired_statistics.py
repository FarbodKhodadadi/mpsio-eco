from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.statistics import analyze_results_csv, paired_comparison
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="CSV with seed, model and accuracy columns")
    parser.add_argument("--output", default="outputs/statistics")
    parser.add_argument("--proposed", default="alpha")
    parser.add_argument("--baseline", default="alpha_ce")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--use-paper-example", action="store_true",
                        help="Use the five paired accuracy values printed in the manuscript")
    args = parser.parse_args()
    if args.use_paper_example:
        proposed = np.array([0.8272, 0.8310, 0.8371, 0.8408, 0.8343])
        baseline = np.array([0.8265, 0.8165, 0.8233, 0.8223, 0.8193])
        from mpsio_eco.statistics import paired_comparison
        result = paired_comparison(proposed, baseline, proposed_name="Alpha", baseline_name="Alpha-CE")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        import json
        with open(Path(args.output) / "paired_statistics_paper_example.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    elif args.input:
        result = analyze_results_csv(args.input, args.output, args.proposed, args.baseline, args.metric)
    else:
        parser.error("Provide --input or --use-paper-example")
    print(result)

if __name__ == "__main__":
    main()
