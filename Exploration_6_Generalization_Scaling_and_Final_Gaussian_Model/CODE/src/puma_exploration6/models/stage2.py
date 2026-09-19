from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ..constants import EMBED_DIM
from ..config import ModelConfig
from .a5 import A5Head
from .pooling import TokenPooler


class Stage2FromTokens(nn.Module):
    def __init__(self,model_cfg: ModelConfig):
        super().__init__();model_cfg.validate();self.pooler=TokenPooler(model_cfg.representation,model_cfg.gaussian_sigma,model_cfg.neighborhood_radius,model_cfg.scalar_mix_init)
        self.head=A5Head(model_cfg.interaction_rank,model_cfg.representation_dropout,model_cfg.interaction_dropout,model_cfg.tier_a_dropout,model_cfg.use_tier_a)
    def forward(self,tokens: torch.Tensor,bio: torch.Tensor,u: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
        h=self.pooler(tokens,u,v);return self.head(h,bio)


class Stage2FromCachedCLS(nn.Module):
    def __init__(self,model_cfg: ModelConfig):
        super().__init__();model_cfg.validate()
        if model_cfg.representation!="cls":raise ValueError("cached CLS model requires representation=cls")
        self.head=A5Head(model_cfg.interaction_rank,model_cfg.representation_dropout,model_cfg.interaction_dropout,model_cfg.tier_a_dropout,model_cfg.use_tier_a)
    def forward(self,features: torch.Tensor,bio: torch.Tensor) -> torch.Tensor:
        if features.ndim!=2 or features.shape[1]!=EMBED_DIM:raise ValueError("features must be Bx1536")
        h=F.layer_norm(features.float(),(EMBED_DIM,));return self.head(h,bio)
