from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PumaConfig


def _draw_gaussian_max(heatmap: np.ndarray, x: float, y: float, sigma: float) -> None:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    anchor_x, anchor_y = int(np.floor(x)), int(np.floor(y))
    left = max(0, anchor_x - radius)
    right = min(heatmap.shape[1], anchor_x + radius + 1)
    top = max(0, anchor_y - radius)
    bottom = min(heatmap.shape[0], anchor_y + radius + 1)
    if left >= right or top >= bottom:
        return
    dx = (np.arange(left, right, dtype=np.float32) + 0.5) - np.float32(x)
    dy = (np.arange(top, bottom, dtype=np.float32) + 0.5) - np.float32(y)
    kernel = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2.0 * sigma * sigma))
    np.maximum(
        heatmap[top:bottom, left:right],
        kernel.astype(np.float32),
        out=heatmap[top:bottom, left:right],
    )
    if 0 <= anchor_y < heatmap.shape[0] and 0 <= anchor_x < heatmap.shape[1]:
        heatmap[anchor_y, anchor_x] = 1.0


def build_stage1_targets(
    coordinates: np.ndarray,
    class_ids: np.ndarray,
    config: PumaConfig,
) -> dict[str, np.ndarray]:
    size = int(config.image_size)
    heatmap = np.zeros((1, size, size), dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if len(coordinates) != len(class_ids):
        raise ValueError("Stage-1 coordinates and class IDs have different lengths.")

    center_indices: list[tuple[int, int]] = []
    center_offsets: list[tuple[float, float]] = []
    center_classes: list[int] = []
    offset_linear: list[np.ndarray] = []
    offset_distance: list[np.ndarray] = []
    offset_values: list[np.ndarray] = []
    radius = float(config.stage1_offset_radius_pixels)

    for (raw_x, raw_y), class_id in zip(coordinates, class_ids, strict=True):
        x, y = float(raw_x), float(raw_y)
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if not (0 <= xi < size and 0 <= yi < size):
            continue
        _draw_gaussian_max(heatmap[0], x, y, float(config.stage1_heatmap_sigma_pixels))
        center_indices.append((yi, xi))
        center_offsets.append((x - (xi + 0.5), y - (yi + 0.5)))
        center_classes.append(int(class_id))

        left = max(0, int(np.floor(x - radius)))
        right = min(size, int(np.ceil(x + radius)) + 1)
        top = max(0, int(np.floor(y - radius)))
        bottom = min(size, int(np.ceil(y + radius)) + 1)
        grid_x = np.arange(left, right, dtype=np.float32) + 0.5
        grid_y = np.arange(top, bottom, dtype=np.float32) + 0.5
        dx = np.float32(x) - grid_x
        dy = np.float32(y) - grid_y
        distance = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
        valid_y, valid_x = np.nonzero(distance <= radius)
        if not len(valid_y):
            continue
        yy = valid_y.astype(np.int64) + top
        xx = valid_x.astype(np.int64) + left
        offset_linear.append(yy * size + xx)
        offset_distance.append(distance[valid_y, valid_x].astype(np.float32))
        offset_values.append(
            np.column_stack((dx[valid_x], dy[valid_y])).astype(np.float32, copy=False)
        )

    if offset_linear:
        linear = np.concatenate(offset_linear)
        distance = np.concatenate(offset_distance)
        values = np.concatenate(offset_values)
        order = np.lexsort((distance, linear))
        linear_sorted = linear[order]
        first = np.empty(len(order), dtype=bool)
        first[0] = True
        first[1:] = linear_sorted[1:] != linear_sorted[:-1]
        chosen = order[first]
        chosen_linear = linear[chosen]
        offset_index = np.column_stack((chosen_linear // size, chosen_linear % size)).astype(
            np.int64, copy=False
        )
        offset_target = values[chosen].astype(np.float32, copy=False)
    else:
        offset_index = np.empty((0, 2), dtype=np.int64)
        offset_target = np.empty((0, 2), dtype=np.float32)

    return {
        "heatmap": heatmap,
        "offset_index": offset_index,
        "offset_target": offset_target,
        "center_index": np.asarray(center_indices, dtype=np.int64).reshape(-1, 2),
        "center_offset": np.asarray(center_offsets, dtype=np.float32).reshape(-1, 2),
        "coarse_type": np.asarray(center_classes, dtype=np.int64),
    }


def center_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    probability = logits.sigmoid().clamp(1.0e-5, 1.0 - 1.0e-5)
    positive = targets.eq(1).to(logits.dtype)
    negative = targets.lt(1).to(logits.dtype)
    positive_loss = -(1.0 - probability).pow(alpha) * probability.log() * positive
    negative_loss = (
        -probability.pow(alpha) * (1.0 - probability).log() * (1.0 - targets).pow(beta) * negative
    )
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)


def stage1_loss(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: PumaConfig,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap = center_focal_loss(outputs["heatmap_logits"], targets["heatmap"])
    native_feature = outputs["native_feature"]
    zero = native_feature.sum() * 0.0

    offset_index = targets["offset_index"]
    if offset_index.numel():
        offset_prediction = model.predict_points(native_feature, offset_index)["offset"]
        offset = F.smooth_l1_loss(offset_prediction, targets["offset_target"], beta=1.0)
    else:
        offset = zero

    center_index = targets["center_index"]
    if center_index.numel():
        center_prediction = model.predict_points(native_feature, center_index)
        residual = (center_prediction["offset"] - targets["center_offset"]).square().mean(dim=1)
        normalized_error = residual.detach().sqrt() / float(config.match_radius_pixels)
        quality_target = torch.exp(-normalized_error).clamp(0.0, 1.0)
        quality = F.binary_cross_entropy_with_logits(
            center_prediction["quality_logits"], quality_target
        )
        residual_normalized = residual / float(config.match_radius_pixels**2)
        log_variance = center_prediction["log_variance"]
        uncertainty = (
            0.5 * torch.exp(-log_variance) * residual_normalized + 0.5 * log_variance
        ).mean()
        coarse_type = F.cross_entropy(
            center_prediction["coarse_type_logits"],
            targets["coarse_type"],
            weight=class_weights,
        )
    else:
        quality = zero
        uncertainty = zero
        coarse_type = zero

    total = (
        config.stage1_heatmap_loss_weight * heatmap
        + config.stage1_offset_loss_weight * offset
        + config.stage1_quality_loss_weight * quality
        + config.stage1_uncertainty_loss_weight * uncertainty
        + config.stage1_coarse_type_loss_weight * coarse_type
    )
    return total, {
        "heatmap": float(heatmap.detach()),
        "offset": float(offset.detach()),
        "quality": float(quality.detach()),
        "uncertainty": float(uncertainty.detach()),
        "coarse_type": float(coarse_type.detach()),
    }
