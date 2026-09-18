from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEvidence(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, anchor_dim: int, reliability_dim: int = 1) -> None:
        super().__init__()
        self.delta = MLP(input_dim, hidden_dim, anchor_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(anchor_dim + reliability_dim),
            nn.Linear(anchor_dim + reliability_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, anchor: torch.Tensor, evidence: torch.Tensor, reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if reliability.ndim == 1:
            reliability = reliability[:, None]
        gate = self.gate(torch.cat((anchor, reliability), dim=1))
        return self.delta(evidence) * gate, gate
