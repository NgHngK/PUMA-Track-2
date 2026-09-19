from __future__ import annotations

import math
import torch
from torch.utils.data import WeightedRandomSampler
from .constants import NUM_CLASSES
from .config import SamplerConfig


def class_counts(labels: torch.Tensor) -> torch.Tensor:
    labels=torch.as_tensor(labels,dtype=torch.long)
    if labels.ndim!=1 or (labels<0).any() or (labels>=NUM_CLASSES).any(): raise ValueError("labels must be canonical 0..9")
    counts=torch.bincount(labels,minlength=NUM_CLASSES).double()
    if (counts==0).any(): raise ValueError(f"training split lacks classes: {torch.where(counts==0)[0].tolist()}")
    return counts


def build_sampler(labels: torch.Tensor, cfg: SamplerConfig, seed: int) -> tuple[WeightedRandomSampler | None,torch.Tensor,torch.Tensor]:
    cfg.validate();labels=torch.as_tensor(labels,dtype=torch.long);counts=class_counts(labels)
    if cfg.mode=="natural":
        q=counts/counts.sum();weights=torch.ones(len(labels),dtype=torch.double)
        return None,q,weights
    alpha=1.0 if cfg.mode=="inverse" else cfg.alpha
    weights=counts[labels].pow(-alpha)
    num_samples=max(1,int(round(len(labels)*cfg.num_samples_multiplier)))
    sampler=WeightedRandomSampler(weights,num_samples=num_samples,replacement=True,generator=torch.Generator().manual_seed(seed))
    class_mass=torch.zeros(NUM_CLASSES,dtype=torch.double).scatter_add_(0,labels,weights)
    q=class_mass/class_mass.sum()
    return sampler,q,weights


def expected_unique_draws(weights: torch.Tensor, num_samples: int) -> float:
    """Expected number of unique indices under iid replacement draws."""
    w=torch.as_tensor(weights,dtype=torch.double);p=w/w.sum()
    return float((1-(1-p).pow(int(num_samples))).sum())
