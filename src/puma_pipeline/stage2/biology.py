from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..constants import BIOLOGY_DIM


def _weighted_stats(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = weights.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    mean = (values * weights).sum(dim=(1, 2, 3)) / denom
    variance = ((values - mean[:, None, None, None]).square() * weights).sum(dim=(1, 2, 3)) / denom
    return mean, variance.clamp_min(0).sqrt()


def extract_biology_features(
    mask_probability: torch.Tensor,
    hematoxylin: torch.Tensor,
    hematoxylin_gradient: torch.Tensor,
    mask_confidence: torch.Tensor,
    roi_relative: torch.Tensor,
) -> torch.Tensor:
    m = mask_probability.float().clamp(0, 1)
    h = hematoxylin.float()
    g = hematoxylin_gradient.float()
    b, _, height, width = m.shape
    eps = 1e-6
    area = m.sum(dim=(1, 2, 3)).clamp_min(eps)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=m.device, dtype=m.dtype),
        torch.arange(width, device=m.device, dtype=m.dtype),
        indexing="ij",
    )
    xx = xx[None, None]
    yy = yy[None, None]
    cx = (m * xx).sum((1, 2, 3)) / area
    cy = (m * yy).sum((1, 2, 3)) / area
    dx = xx - cx[:, None, None, None]
    dy = yy - cy[:, None, None, None]
    cxx = (m * dx.square()).sum((1, 2, 3)) / area
    cyy = (m * dy.square()).sum((1, 2, 3)) / area
    cxy = (m * dx * dy).sum((1, 2, 3)) / area
    trace = cxx + cyy
    root = torch.sqrt((cxx - cyy).square() + 4 * cxy.square() + eps)
    l1 = ((trace + root) / 2).clamp_min(eps)
    l2 = ((trace - root) / 2).clamp_min(eps)
    major = 4.0 * torch.sqrt(l1)
    minor = 4.0 * torch.sqrt(l2)
    axis_ratio = (major / minor.clamp_min(0.5)).clamp(max=12.0)
    eccentricity = torch.sqrt((1.0 - l2 / l1).clamp(0, 1))
    tvx = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1]).sum((1, 2, 3))
    tvy = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :]).sum((1, 2, 3))
    perimeter = (tvx + tvy).clamp_min(1.0)
    circularity = (4.0 * math.pi * area / perimeter.square()).clamp(0, 1.5)
    compactness = (perimeter.square() / (4.0 * math.pi * area)).clamp(0, 20)
    h_mean, h_std = _weighted_stats(h, m)
    g_mean, g_std = _weighted_stats(g, m)
    lap = F.conv2d(h, h.new_tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]), padding=1).abs()
    lap_mean, _ = _weighted_stats(lap, m)
    bins = torch.linspace(0, 2, 9, device=h.device, dtype=h.dtype)
    histogram = []
    for low, high in zip(bins[:-1], bins[1:]):
        membership = ((h >= low) & (h < high)).float() * m
        histogram.append(membership.sum((1, 2, 3)))
    hist = torch.stack(histogram, dim=1)
    hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(hist * torch.log(hist.clamp_min(eps))).sum(dim=1) / math.log(8.0)
    dilated = F.max_pool2d(m, kernel_size=11, stride=1, padding=5)
    ring = (dilated - m).clamp_min(0)
    ring_mean, _ = _weighted_stats(h, ring)
    nucleus_ring = (h_mean - ring_mean).clamp(-2, 2)
    peak = m.amax((1, 2, 3))
    mask_mean = m.mean((1, 2, 3))
    conf = mask_confidence.float().clamp(0, 1)
    rr = roi_relative.float().reshape(b, 2)
    features = torch.stack(
        (
            torch.log1p(area), major / 32.0, minor / 32.0, axis_ratio, eccentricity,
            perimeter / 64.0, circularity, compactness,
            h_mean, h_std, g_mean, g_std, lap_mean, entropy, nucleus_ring,
            peak, mask_mean, conf, rr[:, 0], rr[:, 1],
        ), dim=1,
    )
    features = torch.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
    if features.shape[1] != BIOLOGY_DIM:
        raise RuntimeError(f"Biology feature contract is {BIOLOGY_DIM}, got {features.shape[1]}.")
    return features
