"""Reproduce the supplied auxiliary feature-ranking encoder training run.

Important: this auxiliary notebook variant uses CE + 5 * normalized,
 squared-centroid-distance SI and a record-level stratified 65/15/20 split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.data import ArrayECGDataset, load_manifest
from mpsio_eco.losses import legacy_normalized_squared_separation_index
from mpsio_eco.models import AlphaECGNet
from mpsio_eco.seed import get_device, set_seed


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Sample a small balanced batch from every class for stable SI estimates."""
    def __init__(self, labels: np.ndarray, samples_per_class: int = 4, seed: int = 456):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = np.unique(self.labels)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.indices = {int(c): np.flatnonzero(self.labels == c) for c in self.classes}
        too_small = {c: len(idx) for c, idx in self.indices.items() if len(idx) < 2}
        if too_small:
            raise ValueError(f"Each class needs at least two training examples: {too_small}")
        if len(self.classes) < 2:
            raise ValueError("At least two classes are required")
        effective_batch = sum(min(self.samples_per_class, len(idx)) for idx in self.indices.values())
        self.num_batches = max(1, math.ceil(len(self.labels) / effective_batch))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch: list[int] = []
            for class_id in self.classes:
                class_indices = self.indices[int(class_id)]
                n = min(self.samples_per_class, len(class_indices))
                batch.extend(rng.choice(class_indices, size=n, replace=False).tolist())
            rng.shuffle(batch)
            yield batch


def collect(model, loader, device):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, embedding = model(x.to(device), return_embedding=True)
            latents.append(embedding.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(latents), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="outputs/feature_ranking_encoder")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--si-weight", type=float, default=5.0)
    parser.add_argument("--lead-index", type=int, default=0)
    parser.add_argument("--expected-length", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    selected_classes = [x.strip() for x in args.classes.split(",")] if args.classes else None
    data = load_manifest(
        args.manifest, data_root=args.data_root, classes=selected_classes,
        lead_index=args.lead_index, expected_length=args.expected_length,
    )
    indices = np.arange(len(data.labels))
    trainval, test = train_test_split(
        indices, test_size=0.20, random_state=args.seed, stratify=data.labels,
    )
    train, validation = train_test_split(
        trainval, test_size=0.15 / 0.80, random_state=args.seed + 1,
        stratify=data.labels[trainval],
    )

    dataset = ArrayECGDataset(data.signals, data.labels)
    train_subset = Subset(dataset, train.tolist())
    train_labels = data.labels[train]
    sampler = BalancedClassBatchSampler(train_labels, args.samples_per_class, args.seed)
    train_loader = DataLoader(train_subset, batch_sampler=sampler)
    train_eval_loader = DataLoader(Subset(dataset, train.tolist()), batch_size=64, shuffle=False)
    val_loader = DataLoader(Subset(dataset, validation.tolist()), batch_size=64, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test.tolist()), batch_size=64, shuffle=False)

    model = AlphaECGNet(num_classes=len(data.class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_values = []
        ce_values = []
        si_values = []
        latent_stds = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, embedding = model(x, return_embedding=True)
            ce = criterion(logits, y)
            si = legacy_normalized_squared_separation_index(embedding, y)
            loss = ce + args.si_weight * si
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite auxiliary feature-ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_values.append(float(loss.detach().cpu()))
            ce_values.append(float(ce.detach().cpu()))
            si_values.append(float(si.detach().cpu()))
            latent_stds.append(float(embedding.detach().std(dim=0, unbiased=False).mean().cpu()))
        history.append({
            "epoch": epoch,
            "total_loss": float(np.mean(total_values)),
            "cross_entropy_loss": float(np.mean(ce_values)),
            "legacy_separation_index": float(np.mean(si_values)),
            "latent_std": float(np.mean(latent_stds)),
        })
        print(f"epoch={epoch:02d}/{args.epochs} loss={history[-1]['total_loss']:.5f} "
              f"CE={history[-1]['cross_entropy_loss']:.5f} legacy-SI={history[-1]['legacy_separation_index']:.5f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    train_latents, train_y = collect(model, train_eval_loader, device)
    val_latents, val_y = collect(model, val_loader, device)
    test_latents, test_y = collect(model, test_loader, device)
    np.savez_compressed(
        out / "latent_arrays.npz",
        train_latents=train_latents, train_labels=train_y,
        validation_latents=val_latents, validation_labels=val_y,
        test_latents=test_latents, test_labels=test_y,
        class_names=np.asarray(data.class_names, dtype=str),
    )
    torch.save({
        "model_name": "alpha",
        "num_classes": len(data.class_names),
        "class_names": data.class_names,
        "model_state_dict": model.state_dict(),
        "seed": args.seed,
        "loss": "cross_entropy + si_weight * legacy_normalized_squared_SI",
        "si_weight": args.si_weight,
    }, out / "feature_ranking_encoder.pt")
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "class_names": data.class_names,
            "seed": args.seed,
            "epochs": args.epochs,
            "si_weight": args.si_weight,
            "split": "single stratified record-level 65/15/20 split",
            "warning": "Auxiliary feature-ranking training variant; not the primary-paper SI formula.",
        }, f, indent=2)
    print(f"Saved auxiliary encoder and latent arrays under {out.resolve()}")

if __name__ == "__main__":
    main()
