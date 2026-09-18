from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PumaConfig
from ..constants import BIOLOGY_DIM, DETECTION_DIM, NUCLEUS_CLASSES, SPATIAL_DIM, STAGE1_PRIOR_DIM, TISSUE_CONTEXT_DIM, UNI2_POOLED_DIM
from .biomask import BioMaskHead
from .biology import extract_biology_features
from .fusion import MLP, ResidualEvidence
from .graph_refine import GraphResidualRefiner


class AppearanceAnchor(nn.Module):
    def __init__(self, hidden_dim: int, views: int) -> None:
        super().__init__()
        self.projection = nn.ModuleList([MLP(UNI2_POOLED_DIM, hidden_dim, hidden_dim, 0.05) for _ in range(views)])
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([layer(features[:, i].float()) for i, layer in enumerate(self.projection)], dim=1)
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return self.norm((tokens * weights[..., None]).sum(dim=1)), weights


class BioContextRefine(nn.Module):
    def __init__(self, config: PumaConfig) -> None:
        super().__init__()
        h = config.stage2_hidden_dim
        self.config = config
        self.biomask = BioMaskHead(config.biomask_base_channels)
        self.appearance = AppearanceAnchor(h, len(config.stage2_views))
        self.bio = ResidualEvidence(BIOLOGY_DIM, config.stage2_biology_hidden_dim, h, 2)
        self.spatial = ResidualEvidence(SPATIAL_DIM, config.stage2_spatial_hidden_dim, h, 1)
        self.tissue = ResidualEvidence(TISSUE_CONTEXT_DIM, config.stage2_tissue_hidden_dim, h, 1)
        self.prior = ResidualEvidence(STAGE1_PRIOR_DIM, config.stage2_prior_hidden_dim, h, 1)
        self.final_norm = nn.LayerNorm(h)
        self.local_head = nn.Linear(h, len(NUCLEUS_CLASSES))
        self.validity = nn.Sequential(
            nn.LayerNorm(h + DETECTION_DIM + 1), nn.Linear(h + DETECTION_DIM + 1, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.graph = GraphResidualRefiner(h, len(NUCLEUS_CLASSES))

    def forward_local(
        self,
        appearance_features: torch.Tensor,
        biomask_rgb: torch.Tensor,
        roi_relative_h: torch.Tensor,
        spatial: torch.Tensor,
        tissue: torch.Tensor,
        stage1_prior: torch.Tensor,
        detection: torch.Tensor,
        *,
        context_dropout: bool = False,
    ) -> dict[str, torch.Tensor]:
        anchor, view_weights = self.appearance(appearance_features)
        mask_out = self.biomask(biomask_rgb)
        mask_prob = mask_out["mask_logits"].sigmoid()
        mask_conf = mask_out["mask_confidence_logit"].sigmoid()
        biology = extract_biology_features(mask_prob, mask_out["hematoxylin"], mask_out["hematoxylin_gradient"], mask_conf, roi_relative_h)
        bio_input = biology
        tissue_input = tissue.float()
        tissue_entropy = tissue_input[:, -1:].clamp(0, 1)
        prior_input = stage1_prior.float()
        if self.training and context_dropout:
            bio_input = F.dropout(bio_input, p=self.config.biology_context_dropout, training=True)
            tissue_input = F.dropout(tissue_input, p=self.config.tissue_context_dropout, training=True)
            prior_input = F.dropout(prior_input, p=self.config.stage1_prior_dropout, training=True)
        bio_delta, bio_gate = self.bio(anchor, bio_input, torch.stack((mask_conf, detection[:, 0].clamp(0, 1)), dim=1))
        spatial_delta, spatial_gate = self.spatial(anchor, spatial.float(), torch.ones_like(mask_conf))
        tissue_delta, tissue_gate = self.tissue(anchor, tissue_input, 1.0 - tissue_entropy)
        prior_delta, prior_gate = self.prior(anchor, prior_input, detection[:, 0].clamp(0, 1))
        fused = self.final_norm(anchor + bio_delta + spatial_delta + tissue_delta + prior_delta)
        logits = self.local_head(fused)
        validity_logit = self.validity(torch.cat((fused, detection.float(), mask_conf[:, None]), dim=1)).squeeze(1)
        return {
            "embedding": fused, "local_logits": logits, "validity_logit": validity_logit,
            "mask_logits": mask_out["mask_logits"], "mask_confidence_logit": mask_out["mask_confidence_logit"],
            "biology": biology, "view_weights": view_weights,
            "bio_gate": bio_gate.squeeze(1), "spatial_gate": spatial_gate.squeeze(1),
            "tissue_gate": tissue_gate.squeeze(1), "prior_gate": prior_gate.squeeze(1),
        }

    def refine(
        self,
        local: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_distance: torch.Tensor,
        *,
        probability_temperature: float | None = None,
        probability_dropout: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        logits = local["local_logits"]
        temp = float(probability_temperature or self.config.graph_probability_temperature)
        prob = torch.softmax(logits.detach() / temp, dim=1)
        if self.training and probability_dropout > 0:
            prob = F.dropout(prob, p=probability_dropout, training=True)
            prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        node_validity = local["validity_logit"].sigmoid().detach()
        correction, gate = self.graph(local["embedding"], prob, coordinates.float(), neighbor_index.long(), neighbor_distance.float(), node_validity=node_validity)
        return {**local, "graph_delta": correction, "graph_gate": gate, "final_logits": logits + correction}
