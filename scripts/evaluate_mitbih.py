from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.mitbih import load_mitbih_beats, run_mitbih_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder with matched MIT-BIH CSV and TXT files")
    parser.add_argument("--output", default="outputs/mitbih")
    parser.add_argument("--symbols", default="N,V,F")
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--signal-column", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--normalize-beats", action="store_true")
    parser.add_argument("--loss-recipe", choices=["supcon_center", "mpsio", "ce"], default="supcon_center")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    beats = load_mitbih_beats(
        args.data_dir, retained_symbols=[x.strip() for x in args.symbols.split(",")],
        window_size=args.window_size, signal_column=args.signal_column,
        min_support=args.min_support, normalize_beats=args.normalize_beats,
    )
    metrics = run_mitbih_experiment(
        beats, args.output, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, loss_recipe=args.loss_recipe,
        device_name=args.device,
    )
    print("MIT-BIH test metrics:", metrics)

if __name__ == "__main__":
    main()
