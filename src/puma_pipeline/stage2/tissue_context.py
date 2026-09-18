from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from ..constants import TISSUE_CONTEXT_DIM


def tissue_context_from_probability_map(
    probabilities: np.ndarray,
    coordinates: np.ndarray,
    *,
    source_image_size: int = 1024,
    pool_radius: int = 32,
) -> np.ndarray:
    """Build 18 tissue-context features for each candidate."""
    prob = np.asarray(probabilities, dtype=np.float32)
    if prob.ndim != 3 or prob.shape[0] != 6:
        raise ValueError(f"Expected tissue probabilities [6,H,W], got {prob.shape}.")
    coords = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if not len(coords):
        return np.empty((0, TISSUE_CONTEXT_DIM), dtype=np.float32)
    _, height, width = prob.shape
    scale_x = width / float(source_image_size)
    scale_y = height / float(source_image_size)
    xx = np.clip(np.floor(coords[:, 0] * scale_x).astype(int), 0, width - 1)
    yy = np.clip(np.floor(coords[:, 1] * scale_y).astype(int), 0, height - 1)
    center = prob[:, yy, xx].T
    radius_feature = max(1, int(round(pool_radius * (width / float(source_image_size)))))
    kernel = 2 * radius_feature + 1
    local_map = np.stack(
        [uniform_filter(channel, size=kernel, mode="nearest") for channel in prob], axis=0
    )
    local = local_map[:, yy, xx].T
    hard = prob.argmax(axis=0)
    distance_features: list[np.ndarray] = []
    for class_id in (3, 1, 2, 4, 5):
        mask = hard == class_id
        if np.any(mask):
            distance = distance_transform_edt(~mask)
            sampled = distance[yy, xx] / max(width, height)
        else:
            sampled = np.ones(len(coords), dtype=np.float32)
        distance_features.append(np.clip(sampled, 0, 1).astype(np.float32))
    entropy = -(center * np.log(np.clip(center, 1e-7, 1.0))).sum(axis=1) / math.log(6.0)
    output = np.column_stack((center, local, *distance_features, entropy)).astype(np.float32)
    if output.shape[1] != TISSUE_CONTEXT_DIM:
        raise RuntimeError(f"Tissue context dimension mismatch: {output.shape}.")
    if not np.isfinite(output).all():
        raise ValueError("Tissue context contains non-finite values.")
    return output
