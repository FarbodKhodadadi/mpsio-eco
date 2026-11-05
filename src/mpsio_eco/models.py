from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ConvBackbone1D(nn.Module):
    """Three 1-D CNN stages; output shape [B,64,T/8]."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _prepare_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 3 and x.shape[1] != 1 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected ECG input [B,1,T], received {tuple(x.shape)}")
    if x.shape[-1] < 8:
        raise ValueError("ECG input must contain at least 8 samples for three pooling stages")
    return x


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AlphaECGNet(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        num_heads: int = 4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = int(num_classes)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.num_heads = int(num_heads)
        self.bidirectional = bool(bidirectional)

        self.conv = ConvBackbone1D()
        lstm_dim = lstm_hidden * (2 if bidirectional else 1)
        if lstm_dim % num_heads:
            raise ValueError("BiLSTM output dimension must be divisible by num_heads")
        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_dim)
        self.lstm2 = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = ClassifierHead(lstm_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = _prepare_input(x)
        x = self.conv(x).transpose(1, 2).contiguous()  # [B,T',64]
        x, _ = self.lstm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attention_features = self.norm(attn_out + x)  # [B,T',64]
        embedding = attention_features.mean(dim=1)  # [B,64]; correct time pooling
        temporal_features, _ = self.lstm2(attention_features)
        logits = self.classifier(temporal_features[:, -1, :])
        return (logits, embedding) if return_embedding else logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)[1]


class AlphaCEECGNet(AlphaECGNet):
    """Same complete Alpha architecture, trained with cross-entropy only."""


class BetaECGNet(nn.Module):
    """CNN + four-head self-attention + MLP, without recurrent layers."""

    def __init__(self, num_classes: int, num_heads: int = 4) -> None:
        super().__init__()
        self.conv = ConvBackbone1D()
        self.attn = nn.MultiheadAttention(64, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = _prepare_input(x)
        seq = self.conv(x).transpose(1, 2).contiguous()
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        features = self.norm(attn_out + seq)
        embedding = features.mean(dim=1)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)[1]


class GammaECGNet(nn.Module):
    """CNN-only baseline with global average pooling and an MLP head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = ConvBackbone1D()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = _prepare_input(x)
        features = self.pool(self.conv(x)).squeeze(-1)
        logits = self.classifier(features)
        return (logits, features) if return_embedding else logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)[1]


class DeltaECGNet(nn.Module):
    """CNN + two BiLSTM layers, without the attention module."""

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ConvBackbone1D()
        dim = lstm_hidden * (2 if bidirectional else 1)
        self.lstm1 = nn.LSTM(
            64, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.lstm2 = nn.LSTM(
            dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = ClassifierHead(dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = _prepare_input(x)
        seq = self.conv(x).transpose(1, 2).contiguous()
        seq, _ = self.lstm1(seq)
        seq, _ = self.lstm2(seq)
        embedding = seq[:, -1, :]
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)[1]


class EpsilonECGNet(nn.Module):
    """Raw-signal BiLSTM + attention + BiLSTM model (no CNN front-end)."""

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        num_heads: int = 4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        dim = lstm_hidden * (2 if bidirectional else 1)
        if dim % num_heads:
            raise ValueError("BiLSTM output dimension must be divisible by num_heads")
        self.lstm1 = nn.LSTM(
            1, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.lstm2 = nn.LSTM(
            dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = ClassifierHead(dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = _prepare_input(x).transpose(1, 2).contiguous()  # [B,T,1]
        seq, _ = self.lstm1(x)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(attn_out + seq)
        seq, _ = self.lstm2(seq)
        embedding = seq[:, -1, :]
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)[1]


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "alpha": AlphaECGNet,
    "alpha_ce": AlphaCEECGNet,
    "beta": BetaECGNet,
    "gamma": GammaECGNet,
    "delta": DeltaECGNet,
    "epsilon": EpsilonECGNet,
}


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
