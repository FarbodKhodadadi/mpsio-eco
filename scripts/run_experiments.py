from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.data import load_manifest
from mpsio_eco.seed import DEFAULT_SEEDS
from mpsio_eco.training import DEFAULT_MODELS, run_experiments


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--task-config", default=None, help="YAML file containing task classes")
    parser.add_argument("--classes", default=None, help="Comma-separated exact manifest label names")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="alpha,alpha_ce,beta,gamma,delta,epsilon")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--protocol", choices=["cv", "seeded_holdout"], default="cv")
    parser.add_argument("--expected-classes", type=int, default=None)
    parser.add_argument("--output", default="outputs/main_experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--separation-weight", type=float, default=1.0)
    parser.add_argument("--sampling", choices=["weighted", "none"], default="weighted")
    parser.add_argument("--lead-index", type=int, default=0)
    parser.add_argument("--expected-length", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-record-level-split", action="store_true",
                        help="Allow record-level splits when no patient_id is available (not recommended).")
    args = parser.parse_args()

    selected_classes = _parse_list(args.classes) if args.classes else None
    task_name = "custom_task"
    if args.task_config:
        with open(args.task_config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        task_name = str(config.get("name", Path(args.task_config).stem))
        if selected_classes is None and config.get("classes"):
            selected_classes = [str(x) for x in config["classes"]]
        if args.expected_classes is None and config.get("expected_num_classes") is not None:
            args.expected_classes = int(config["expected_num_classes"])

    data = load_manifest(
        args.manifest,
        data_root=args.data_root,
        classes=selected_classes,
        lead_index=args.lead_index,
        expected_length=args.expected_length,
    )
    if args.expected_classes is not None and len(data.class_names) != args.expected_classes:
        raise SystemExit(
            f"Task {task_name} expected {args.expected_classes} classes, "
            f"but the manifest contains {len(data.class_names)}: {data.class_names}"
        )
    result_dir = Path(args.output) / task_name
    results = run_experiments(
        data=data,
        output_dir=result_dir,
        models=_parse_list(args.models),
        seeds=[int(x) for x in _parse_list(args.seeds)],
        protocol=args.protocol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        separation_weight=args.separation_weight,
        sampling=args.sampling,
        num_workers=args.num_workers,
        device_name=args.device,
        allow_record_level=args.allow_record_level_split,
    )
    print("\nCompleted experiments:")
    print(results.to_string(index=False))
    print(f"Results: {result_dir.resolve()}")

if __name__ == "__main__":
    main()
