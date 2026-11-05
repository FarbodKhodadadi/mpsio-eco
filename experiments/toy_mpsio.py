"""Reconstructed toy-data experiments described in the MPSIO-ECO manuscript.

This is a clean implementation from the method and hyperparameters in the
manuscript, not a recovered copy of the original experimental notebook.
It uses six 2-D synthetic datasets, a reconstruction/classification/separation
objective, and the manuscript's Armijo-type backtracking update.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mpsio_eco.losses import SeparationIndexLoss
from mpsio_eco.seed import set_seed


class ToyAutoencoderClassifier(nn.Module):
    def __init__(self, depth: int, hidden_width: int = 128, latent_dim: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 2
        for _ in range(depth):
            layers.extend([nn.Linear(in_features, hidden_width), nn.ReLU()])
            in_features = hidden_width
        layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 2))
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), self.classifier(z), z


def _spiral(n: int = 800, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_per = n // 2
    theta = np.sqrt(rng.random(n_per)) * 3.5 * np.pi
    r = theta / (3.5 * np.pi)
    x0 = np.c_[r * np.cos(theta), r * np.sin(theta)]
    x1 = np.c_[r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)]
    x = np.vstack([x0, x1]) + rng.normal(0, noise, size=(2 * n_per, 2))
    y = np.r_[np.zeros(n_per), np.ones(n_per)].astype(int)
    return x.astype(np.float32), y


def _convex_blobs(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    inner = rng.normal(0, 0.25, size=(n // 2, 2))
    outer = rng.normal(0, 1.0, size=(n - n // 2, 2))
    x = np.vstack([inner, outer])
    y = np.r_[np.zeros(len(inner)), np.ones(len(outer))].astype(int)
    return x.astype(np.float32), y


def _crossing_ellipses(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    base = np.diag([1.0, 0.16])
    angles = [np.deg2rad(35), np.deg2rad(-35)]
    samples = []
    for angle in angles:
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cov = rot @ base @ rot.T
        samples.append(rng.multivariate_normal([0, 0], cov, size=n0))
    x = np.vstack(samples)
    y = np.r_[np.zeros(n0), np.ones(n0)].astype(int)
    return x.astype(np.float32), y


def _convex_disks(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    angle0 = rng.uniform(0, 2 * np.pi, n0)
    radius0 = 0.6 * np.sqrt(rng.random(n0))
    angle1 = rng.uniform(0, 2 * np.pi, n - n0)
    # Outer class occupies the annulus between radii 0.6 and 1.8.
    radius1 = np.sqrt(rng.uniform(0.6**2, 1.8**2, n - n0))
    x0 = np.c_[radius0 * np.cos(angle0), radius0 * np.sin(angle0)]
    x1 = np.c_[radius1 * np.cos(angle1), radius1 * np.sin(angle1)]
    x = np.vstack([x0, x1])
    y = np.r_[np.zeros(n0), np.ones(n - n0)].astype(int)
    return x.astype(np.float32), y


def make_toy_datasets(seed: int = 0):
    x_c, y_c = make_circles(n_samples=800, factor=0.45, noise=0.08, random_state=seed)
    x_m, y_m = make_moons(n_samples=800, noise=0.10, random_state=seed)
    return {
        "circles": (x_c.astype(np.float32), y_c.astype(int)),
        "moons": (x_m.astype(np.float32), y_m.astype(int)),
        "spiral": _spiral(seed=seed),
        "convex_blobs": _convex_blobs(seed=seed),
        "crossing_ellipses": _crossing_ellipses(seed=seed),
        "convex_disks": _convex_disks(seed=seed),
    }


def _composite_loss(model, x, y, si_loss):
    reconstruction, logits, z = model(x)
    rec = F.mse_loss(reconstruction, x)
    cls = F.binary_cross_entropy_with_logits(logits.squeeze(1), y.to(dtype=x.dtype))
    sep = si_loss(z, y)
    total = 0.4 * rec + 0.3 * cls + 0.5 * sep
    return total, rec, cls, sep, z


def armijo_subgradient_descent(model, x, y, iterations: int = 300,
                               initial_step: float = 1.0, rho: float = 0.5,
                               c: float = 1e-4, max_backtracking: int = 30,
                               tolerance: float = 1e-8):
    """Full-batch gradient descent with Armijo-type backtracking."""
    si_loss = SeparationIndexLoss()
    history = []
    params = [p for p in model.parameters() if p.requires_grad]
    for iteration in range(iterations):
        model.zero_grad(set_to_none=True)
        loss, rec, cls, sep, _ = _composite_loss(model, x, y, si_loss)
        loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        grad_norm_sq = sum(float(g.pow(2).sum()) for g in grads)
        current = float(loss.detach())
        if np.sqrt(grad_norm_sq) <= tolerance:
            break
        original = [p.detach().clone() for p in params]
        step = initial_step
        accepted = False
        for _ in range(max_backtracking):
            with torch.no_grad():
                for p, p0, g in zip(params, original, grads):
                    p.copy_(p0 - step * g)
            with torch.no_grad():
                candidate = float(_composite_loss(model, x, y, si_loss)[0])
            if np.isfinite(candidate) and candidate <= current - c * step * grad_norm_sq:
                accepted = True
                break
            step *= rho
        if not accepted:
            with torch.no_grad():
                for p, p0 in zip(params, original):
                    p.copy_(p0)
            break
        history.append({
            "iteration": iteration + 1,
            "total_loss": current,
            "reconstruction_loss": float(rec.detach()),
            "classification_loss": float(cls.detach()),
            "separation_loss": float(sep.detach()),
            "accepted_step": step,
        })
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/toy_mpsio")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = make_toy_datasets(args.seed)
    rows = []
    preferred_depth = {
        "circles": 5, "moons": 5, "spiral": 4,
        "convex_blobs": 4, "crossing_ellipses": 3, "convex_disks": 2,
    }
    for name, (features, labels) in datasets.items():
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.20, stratify=labels, random_state=0,
        )
        tx = torch.as_tensor(x_train, device=device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=device)
        for depth in range(1, args.max_depth + 1):
            set_seed(args.seed + depth)
            model = ToyAutoencoderClassifier(depth=depth).to(device)
            history = armijo_subgradient_descent(model, tx, ty, iterations=args.iterations)
            model.eval()
            with torch.no_grad():
                x_test_tensor = torch.as_tensor(x_test, device=device)
                _, logits, latent = model(x_test_tensor)
                predictions = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy()
                latent_np = latent.cpu().numpy()
            accuracy = float(np.mean(predictions == y_test))
            rows.append({"dataset": name, "depth": depth, "test_accuracy": accuracy,
                         "n_train": len(x_train), "n_test": len(x_test),
                         "preferred_depth_in_paper": preferred_depth[name]})
            depth_dir = output / name / f"depth_{depth}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(depth_dir / "loss_history.csv", index=False)
            if depth == preferred_depth[name]:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=12, cmap="coolwarm")
                axes[0].set_title("Input space")
                axes[1].scatter(latent_np[:, 0], latent_np[:, 1], c=y_test, s=12, cmap="coolwarm")
                axes[1].set_title("Learned latent space")
                for ax in axes:
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                fig.suptitle(f"{name}: depth {depth}")
                fig.tight_layout()
                fig.savefig(depth_dir / "input_vs_latent.png", dpi=250)
                plt.close(fig)
    result = pd.DataFrame(rows)
    result.to_csv(output / "toy_depth_results.csv", index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
