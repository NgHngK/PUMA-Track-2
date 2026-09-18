from __future__ import annotations

import torch
import torch.nn as nn


class GraphResidualRefiner(nn.Module):
    def __init__(self, hidden_dim: int, classes: int = 10) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.LayerNorm(hidden_dim + classes + 2),
            nn.Linear(hidden_dim + classes + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, classes)
        )
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim + 2), nn.Linear(hidden_dim + 2, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(
        self,
        embedding: torch.Tensor,
        initial_probabilities: torch.Tensor,
        coordinates: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_distance: torch.Tensor,
        node_validity: torch.Tensor | None = None,
        disagreement: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, k = neighbor_index.shape
        if n == 0:
            return embedding.new_zeros((0, initial_probabilities.shape[1])), embedding.new_zeros((0,))
        neighbor_embedding = embedding[neighbor_index]
        neighbor_prob = initial_probabilities.detach()[neighbor_index]
        delta_xy = coordinates[neighbor_index] - coordinates[:, None]
        scale = neighbor_distance[..., None].clamp_min(1.0)
        edge = (delta_xy / scale).clamp(-2, 2)
        msg = self.message(torch.cat((neighbor_embedding, neighbor_prob, edge), dim=-1))
        self_index = torch.arange(n, device=neighbor_index.device)[:, None]
        valid = (neighbor_index != self_index).float()[..., None]
        if node_validity is None:
            neighbor_reliability = torch.ones((n, k, 1), device=embedding.device, dtype=embedding.dtype)
        else:
            neighbor_reliability = node_validity.detach().reshape(-1).clamp(0, 1)[neighbor_index][..., None]
        weight = valid * neighbor_reliability
        aggregate = (msg * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-3)
        correction = self.update(torch.cat((embedding, aggregate), dim=1))
        if disagreement is None:
            local = initial_probabilities.detach()
            neighbor_mean = (neighbor_prob * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-3)
            disagreement = (local - neighbor_mean).abs().mean(dim=1)
        density = (neighbor_distance < 100.0).float().mean(dim=1)
        gate = self.gate(torch.cat((embedding, disagreement[:, None], density[:, None]), dim=1)).squeeze(1)
        has_neighbors = (valid.sum(dim=1).squeeze(-1) > 0).to(gate.dtype)
        mean_neighbor_reliability = (weight.sum(dim=1).squeeze(-1) / valid.sum(dim=1).squeeze(-1).clamp_min(1.0)).clamp(0, 1)
        gate = gate * has_neighbors * mean_neighbor_reliability
        return correction * gate[:, None], gate
