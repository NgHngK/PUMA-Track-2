from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..constants import BIOLOGY_DIM


def _weighted_moments(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=mask.device, dtype=mask.dtype),
        torch.arange(width, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    weight = mask[:, 0].clamp_min(0)
    total = weight.sum(dim=(1, 2)).clamp_min(1e-6)
    mx = (weight * xx).sum(dim=(1, 2)) / total
    my = (weight * yy).sum(dim=(1, 2)) / total
    dx = xx[None] - mx[:, None, None]
    dy = yy[None] - my[:, None, None]
    cxx = (weight * dx * dx).sum(dim=(1, 2)) / total
    cyy = (weight * dy * dy).sum(dim=(1, 2)) / total
    cxy = (weight * dx * dy).sum(dim=(1, 2)) / total
    trace = cxx + cyy
    determinant = (cxx * cyy - cxy.square()).clamp_min(0)
    root = torch.sqrt((trace.square() - 4 * determinant).clamp_min(0))
    eig1 = ((trace + root) / 2).clamp_min(1e-6)
    eig2 = ((trace - root) / 2).clamp_min(1e-6)
    major = 4 * torch.sqrt(eig1)
    minor = 4 * torch.sqrt(eig2)
    eccentricity = torch.sqrt((1 - eig2 / eig1).clamp(0, 1))
    return total, major, minor, eccentricity


def extract_biology_features(
    mask_probability: torch.Tensor,
    hematoxylin: torch.Tensor,
    gradient: torch.Tensor,
    quality: torch.Tensor,
    roi_median: torch.Tensor,
    roi_iqr: torch.Tensor,
) -> torch.Tensor:
    mask = mask_probability.float().clamp(0, 1)
    h = hematoxylin.float()
    grad = gradient.float()
    total, major, minor, eccentricity = _weighted_moments(mask)
    axis_ratio = minor / major.clamp_min(1e-6)
    gx = F.pad(mask[:, :, :, 1:] - mask[:, :, :, :-1], (0, 1, 0, 0)).abs()
    gy = F.pad(mask[:, :, 1:, :] - mask[:, :, :-1, :], (0, 0, 0, 1)).abs()
    perimeter = (gx + gy).sum(dim=(1, 2, 3)).clamp_min(1e-6)
    circularity = (4 * math.pi * total / perimeter.square()).clamp(0, 1.5)
    compactness = perimeter.square() / total.clamp_min(1e-6)
    w = mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    h_mean = (w * h).sum(dim=(1, 2, 3))
    h_var = (w * (h - h_mean[:, None, None, None]).square()).sum(dim=(1, 2, 3))
    g_mean = (w * grad).sum(dim=(1, 2, 3))
    g_var = (w * (grad - g_mean[:, None, None, None]).square()).sum(dim=(1, 2, 3))
    lap = (
        -4 * h
        + F.pad(h[:, :, 1:, :], (0, 0, 0, 1))
        + F.pad(h[:, :, :-1, :], (0, 0, 1, 0))
        + F.pad(h[:, :, :, 1:], (0, 1, 0, 0))
        + F.pad(h[:, :, :, :-1], (1, 0, 0, 0))
    ).abs()
    lap_mean = (w * lap).sum(dim=(1, 2, 3))
    entropy_proxy = torch.log1p(h_var * 100.0)
    dilated = F.max_pool2d(mask, 9, stride=1, padding=4)
    ring = (dilated - mask).clamp_min(0)
    ring_w = ring / ring.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    ring_mean = (ring_w * h).sum(dim=(1, 2, 3))
    robust_z = (h_mean - roi_median.float()) / roi_iqr.float().clamp_min(1e-4)
    percentile_proxy = torch.sigmoid(robust_z * 1.25)
    features = torch.stack(
        (
            torch.log1p(total), major, minor, axis_ratio, eccentricity, perimeter,
            circularity, compactness, h_mean, torch.sqrt(h_var.clamp_min(0)),
            g_mean, torch.sqrt(g_var.clamp_min(0)), lap_mean, entropy_proxy,
            h_mean - ring_mean, mask.amax(dim=(1, 2, 3)), mask.mean(dim=(1, 2, 3)),
            quality.float(), robust_z, percentile_proxy,
        ),
        dim=1,
    )
    if features.shape[1] != BIOLOGY_DIM:
        raise RuntimeError("Biology feature schema mismatch")
    return features

