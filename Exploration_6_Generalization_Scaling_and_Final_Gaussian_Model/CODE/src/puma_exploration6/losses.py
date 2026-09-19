from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .constants import NUM_CLASSES
from .config import LossConfig


class PriorAdjustedCrossEntropy(nn.Module):
    """CE with explicit training/sampling-prior logit adjustment.

    tau=0 is ordinary CE. tau=1 with natural class-count prior is the Balanced
    Softmax formulation used in the historical report. Label smoothing is kept
    explicit and defaults to zero.
    """
    def __init__(self, prior: torch.Tensor, cfg: LossConfig):
        super().__init__();cfg.validate();p=torch.as_tensor(prior,dtype=torch.float32)
        if p.shape!=(NUM_CLASSES,) or not torch.isfinite(p).all() or (p<=0).any(): raise ValueError("all ten priors must be finite and positive")
        p=p/p.sum();self.register_buffer("log_prior",p.log());self.cfg=cfg
        self.tau=0.0 if cfg.kind=="ce" else (1.0 if cfg.kind=="balanced_softmax" else float(cfg.tau))

    def forward(self,logits: torch.Tensor,labels: torch.Tensor,reduction: str="mean") -> torch.Tensor:
        if logits.ndim!=2 or logits.shape[1]!=NUM_CLASSES: raise ValueError("logits must be Bx10")
        # Defensive device alignment: registered buffers normally follow module.to(),
        # but keep the loss safe if a caller forgets to move the criterion first.
        prior=self.log_prior
        if prior.device!=logits.device:
            prior=prior.to(logits.device,non_blocking=logits.is_cuda)
        return F.cross_entropy(logits.float()+self.tau*prior,labels.to(logits.device,non_blocking=logits.is_cuda).long(),reduction=reduction,label_smoothing=self.cfg.label_smoothing)
