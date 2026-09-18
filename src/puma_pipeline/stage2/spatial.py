from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..constants import SPATIAL_DIM


def build_knn_graph(
    coordinates: np.ndarray,
    k: int,
    radius_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    n = len(coordinates)
    if k < 1:
        raise ValueError("k must be positive.")
    if radius_cap <= 0:
        raise ValueError("radius_cap must be positive.")
    if n == 0:
        return np.empty((0, k), np.int64), np.empty((0, k), np.float32)
    out_index = np.repeat(np.arange(n, dtype=np.int64)[:, None], k, axis=1)
    out_distance = np.full((n, k), float(radius_cap), dtype=np.float32)
    if n == 1:
        return out_index, out_distance

    effective = min(n, k + 1)
    distance, index = cKDTree(coordinates).query(coordinates, k=effective, workers=1)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    for row in range(n):
        # Remove only this node and keep real duplicate coordinates.
        keep = index[row] != row
        row_index = np.asarray(index[row][keep], dtype=np.int64)
        row_distance = np.asarray(distance[row][keep], dtype=np.float32)
        valid = row_distance <= float(radius_cap)
        row_index = row_index[valid][:k]
        row_distance = row_distance[valid][:k]
        take = len(row_index)
        if take:
            out_index[row, :take] = row_index
            out_distance[row, :take] = row_distance
    return out_index, out_distance


def spatial_features_from_reference(
    query_xy: np.ndarray,
    reference_xy: np.ndarray,
    image_size: int = 1024,
    *,
    exclude_self: bool | None = None,
) -> np.ndarray:
    query = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    reference = np.asarray(reference_xy, dtype=np.float32).reshape(-1, 2)
    output = np.zeros((len(query), SPATIAL_DIM), dtype=np.float32)
    if len(query) == 0:
        return output
    if len(reference) == 0:
        output[:, :3] = np.log1p(float(image_size))
        return output
    tree = cKDTree(reference)
    same_reference = len(query) == len(reference) and np.array_equal(query, reference)
    if exclude_self is None:
        exclude_self = same_reference
    if exclude_self and not same_reference:
        raise ValueError("exclude_self=True requires query/reference arrays with identical coordinates and order.")
    offset = 1 if exclude_self else 0
    for column, k in enumerate((3, 5, 10)):
        effective = min(len(reference), k + offset)
        distances = tree.query(query, k=max(1, effective), workers=1)[0]
        if distances.ndim == 1:
            distances = distances[:, None]
        values = distances[:, offset:] if distances.shape[1] > offset else np.empty((len(query), 0))
        mean = values.mean(axis=1) if values.shape[1] else np.full(len(query), image_size)
        output[:, column] = np.log1p(mean)
    for column, radius in enumerate((25.0, 50.0, 100.0), start=3):
        counts = np.asarray([len(v) for v in tree.query_ball_point(query, radius)], dtype=np.float32)
        if exclude_self:
            counts = np.maximum(counts - 1, 0)
        output[:, column] = np.log1p(counts / (math.pi * radius * radius) * 10_000.0)
    return output
