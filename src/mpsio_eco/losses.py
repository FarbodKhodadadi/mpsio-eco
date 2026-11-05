from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationIndexLoss(nn.Module):
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, representations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = representations
        if z.ndim == 3:
            z = z.mean(dim=1)
        if z.ndim != 2:
            raise ValueError(
                "Expected representations shaped [B,D] or [B,T,D], "
                f"received {tuple(representations.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
            raise ValueError(
                f"labels must have shape [B]={z.shape[0]}, got {tuple(labels.shape)}"
            )

        labels = labels.to(device=z.device, dtype=torch.long)
        centers: list[torch.Tensor] = []
        within_terms: list[torch.Tensor] = []

        for class_id in torch.unique(labels, sorted=True):
            class_z = z[labels == class_id]
            if class_z.shape[0] < 2:
                continue
            center = class_z.mean(dim=0)
            centers.append(center)
            within_terms.append((class_z - center).pow(2).sum(dim=1).mean())

        if len(centers) < 2:
            return z.sum() * 0.0

        center_tensor = torch.stack(centers, dim=0)
        intra = torch.stack(within_terms).mean()
        inter = torch.pdist(center_tensor, p=2).mean()
        return intra / (inter + self.eps)


def legacy_normalized_squared_separation_index(
    representations: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    
    z = representations
    if z.ndim == 3:
        z = z.mean(dim=1)
    if z.ndim != 2:
        raise ValueError("Expected [B,D] or [B,T,D] representations")
    labels = labels.to(device=z.device, dtype=torch.long)
    z = F.normalize(z, p=2, dim=1, eps=eps)
    centers: list[torch.Tensor] = []
    intra: list[torch.Tensor] = []
    for class_id in torch.unique(labels, sorted=True):
        group = z[labels == class_id]
        if group.shape[0] < 2:
            continue
        center = group.mean(dim=0)
        centers.append(center)
        intra.append((group - center).pow(2).sum(dim=1).mean())
    if len(centers) < 2:
        return z.sum() * 0.0
    center_tensor = torch.stack(centers)
    inter_squared = torch.pdist(center_tensor, p=2).pow(2).mean()
    return torch.stack(intra).mean() / (inter_squared + eps)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("SupervisedContrastiveLoss expects [B,D] features")
        if features.shape[0] < 2:
            return features.sum() * 0.0
        z = F.normalize(features, p=2, dim=1)
        labels = labels.reshape(-1, 1).to(z.device)
        if labels.shape[0] != z.shape[0]:
            raise ValueError("features and labels have different batch sizes")
        same = (labels == labels.T).to(z.dtype)
        non_self = 1.0 - torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        positives = same * non_self
        positive_count = positives.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return z.sum() * 0.0
        logits = (z @ z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * non_self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        mean_positive_log_prob = (positives * log_prob).sum(dim=1) / (positive_count + self.eps)
        return -mean_positive_log_prob[valid].mean()


class CenterMarginLoss(nn.Module):

    def __init__(self, margin: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = float(margin)
        self.normalize = bool(normalize)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("CenterMarginLoss expects [B,D] embeddings")
        z = F.normalize(embeddings, p=2, dim=1) if self.normalize else embeddings
        labels = labels.to(device=z.device, dtype=torch.long)
        centers: list[torch.Tensor] = []
        intra: list[torch.Tensor] = []
        for class_id in torch.unique(labels, sorted=True):
            group = z[labels == class_id]
            if group.shape[0] == 0:
                continue
            center = group.mean(dim=0)
            centers.append(center)
            intra.append((group - center).pow(2).sum(dim=1).mean())
        if len(centers) < 2:
            return z.sum() * 0.0
        center_tensor = torch.stack(centers)
        distances = torch.pdist(center_tensor, p=2)
        inter_hinge = F.relu(self.margin - distances).pow(2).mean()
        return torch.stack(intra).mean() + inter_hinge
