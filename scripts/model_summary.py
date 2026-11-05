"""Print model architecture and trainable parameter count."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from mpsio_eco.models import build_model, count_trainable_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="alpha", choices=[
        "alpha", "alpha_ce", "beta", "gamma", "delta", "epsilon"
    ])
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--signal-length", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = build_model(args.model, args.num_classes).to(args.device)
    parameters = count_trainable_parameters(model)
    print(model)
    print(f"\nModel: {args.model}")
    print(f"Classes: {args.num_classes}")
    print(f"Trainable parameters: {parameters:,}")
    model.eval()
    with torch.no_grad():
        sample = torch.zeros(2, 1, args.signal_length, device=args.device)
        logits, embedding = model(sample, return_embedding=True)
    print(f"Input shape: {(2, 1, args.signal_length)}")
    print(f"Logit shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")


if __name__ == "__main__":
    main()
