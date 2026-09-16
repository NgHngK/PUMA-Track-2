from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


class DenseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class BoundedResidualHead(nn.Module):
    """Bound the actual residual vector, not only a scalar multiplier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        limit: float,
        dropout: float,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.limit = float(limit)
        if self.limit <= 0:
            raise ValueError("residual limit must be > 0")
        self.norm = nn.LayerNorm(input_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)
        if zero_init:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.hidden(self.norm(x.float())))
        h = self.dropout(h)
        return self.limit * torch.tanh(self.output(h))


class ReliabilityGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, initial_probability: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _logit(initial_probability))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features.float()))


class ContextGate(nn.Module):
    def __init__(self, identity_dim: int, evidence_dim: int, reliability_dim: int, hidden_dim: int, initial_probability: float) -> None:
        super().__init__()
        input_dim = identity_dim + evidence_dim + reliability_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _logit(initial_probability))

    def forward(
        self,
        identity: torch.Tensor,
        evidence: torch.Tensor,
        reliability: torch.Tensor,
        detached_entropy: torch.Tensor,
        detached_margin: torch.Tensor,
    ) -> torch.Tensor:
        stats = torch.stack((detached_entropy.float(), detached_margin.float()), dim=1)
        values = torch.cat((identity.float(), evidence.float(), reliability.float(), stats), dim=1)
        return torch.sigmoid(self.net(values))


