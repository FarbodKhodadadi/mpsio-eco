"""Run RF importance ranking and validation-only pruning on saved Alpha latents."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.feature_ranking import run_feature_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-file", required=True, help="Path to fold's latent_arrays.npz")
    parser.add_argument("--output", default="outputs/feature_ranking")
    parser.add_argument("--class-names", default=None, help="Comma-separated output class names in encoded order")
    parser.add_argument("--ranking-estimators", type=int, default=400)
    parser.add_argument("--pruning-estimators", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    class_names = [x.strip() for x in args.class_names.split(",")] if args.class_names else None
    table = run_feature_ranking(
        args.latent_file, args.output, class_names=class_names,
        ranking_estimators=args.ranking_estimators,
        pruning_estimators=args.pruning_estimators,
        random_state=args.random_state,
    )
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
