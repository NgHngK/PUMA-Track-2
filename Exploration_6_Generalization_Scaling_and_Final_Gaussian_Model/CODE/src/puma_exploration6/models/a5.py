from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ..constants import EMBED_DIM, NUM_CLASSES, TIER_A_DIM


class A5Head(nn.Module):
    """Retained Exploration-3 A5 with optional Exploration-6 regularization.

    Baseline rank=8/dropouts=0/use_tier_a=True has exactly 27,866 trainable params.
    Input h must already be non-affine LayerNorm-normalized.
    """
    def __init__(self,interaction_rank: int=8,representation_dropout: float=0.0,interaction_dropout: float=0.0,
                 tier_a_dropout: float=0.0,use_tier_a: bool=True):
        super().__init__();self.rank=int(interaction_rank);self.use_tier_a=bool(use_tier_a)
        if self.rank<1:raise ValueError("interaction_rank must be positive")
        for v in (representation_dropout,interaction_dropout,tier_a_dropout):
            if not 0<=v<1:raise ValueError("dropout outside [0,1)")
        self.rep_dropout=nn.Dropout(float(representation_dropout))
        self.inter_dropout=nn.Dropout(float(interaction_dropout))
        self.bio_dropout=nn.Dropout(float(tier_a_dropout))
        self.head=nn.Linear(EMBED_DIM,NUM_CLASSES)
        if self.use_tier_a:
            self.ph=nn.Linear(EMBED_DIM,self.rank,bias=False)
            self.pb=nn.Linear(TIER_A_DIM,self.rank,bias=False)
            self.branch=nn.Linear(self.rank,NUM_CLASSES,bias=False)
            nn.init.zeros_(self.branch.weight)
        else:
            self.ph=None;self.pb=None;self.branch=None
        self.diagnostics: dict[str,torch.Tensor] = {}

    def forward(self,h: torch.Tensor,bio: torch.Tensor | None=None) -> torch.Tensor:
        if h.ndim!=2 or h.shape[1]!=EMBED_DIM:raise ValueError("h must be Bx1536")
        if not torch.isfinite(h).all():raise FloatingPointError("nonfinite representation")
        h_d=self.rep_dropout(h);app=self.head(h_d);self.diagnostics={"appearance_logit_norm":app.detach().norm(dim=1)}
        if not self.use_tier_a:return app
        if bio is None or bio.shape!=(len(h),TIER_A_DIM):raise ValueError("Tier-A must be Bx16")
        if not torch.isfinite(bio).all():raise FloatingPointError("nonfinite Tier-A")
        bio_d=self.bio_dropout(bio);u=self.ph(h_d);v=self.pb(bio_d);inter=self.inter_dropout(u*v);delta=self.branch(inter)
        self.diagnostics.update(
            appearance_projection_norm=u.detach().norm(dim=1),biology_projection_norm=v.detach().norm(dim=1),
            interaction_norm=inter.detach().norm(dim=1),correction_norm=delta.detach().norm(dim=1),
        )
        return app+delta

    def reset_classifier_outputs(self) -> None:
        """For cRT-style stage: retain Ph/Pb feature transforms; reset W and R."""
        self.head.reset_parameters()
        if self.branch is not None:nn.init.zeros_(self.branch.weight)

    def freeze_feature_transform(self) -> None:
        if self.ph is not None:self.ph.requires_grad_(False)
        if self.pb is not None:self.pb.requires_grad_(False)

    def classifier_parameters(self):
        yield from self.head.parameters()
        if self.branch is not None:yield from self.branch.parameters()
