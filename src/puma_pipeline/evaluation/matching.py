from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class MatchResult:
    pred_indices: np.ndarray
    gt_indices: np.ndarray
    distances: np.ndarray
    unmatched_pred: np.ndarray
    unmatched_gt: np.ndarray


def match_centroids(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    radius: float,
    scores: np.ndarray | None = None,
) -> MatchResult:
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
    ground_truth = np.asarray(ground_truth, dtype=np.float32).reshape(-1, 2)
    if not len(predictions) or not len(ground_truth):
        return MatchResult(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.arange(len(predictions), dtype=np.int64),
            np.arange(len(ground_truth), dtype=np.int64),
        )
    confidence = (
        np.ones(len(predictions), dtype=np.float32)
        if scores is None
        else np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-np.inf)
    )
    neighbours = cKDTree(predictions).query_ball_point(ground_truth, float(radius), workers=1)
    available = np.ones(len(predictions), dtype=bool)
    matched_gt = np.zeros(len(ground_truth), dtype=bool)
    pred_ids: list[int] = []
    gt_ids: list[int] = []
    distances: list[float] = []
    for gt_index, raw in enumerate(neighbours):
        eligible = np.asarray(sorted(raw), dtype=np.int64)
        eligible = eligible[available[eligible]]
        if not len(eligible):
            continue
        distance = np.linalg.norm(predictions[eligible] - ground_truth[gt_index], axis=1)
        inside = distance < float(radius)
        eligible, distance = eligible[inside], distance[inside]
        if not len(eligible):
            continue
        selected = int(np.lexsort((distance, -confidence[eligible]))[0])
        pred_index = int(eligible[selected])
        available[pred_index] = False
        matched_gt[gt_index] = True
        pred_ids.append(pred_index)
        gt_ids.append(gt_index)
        distances.append(float(distance[selected]))
    return MatchResult(
        np.asarray(pred_ids, dtype=np.int64),
        np.asarray(gt_ids, dtype=np.int64),
        np.asarray(distances, dtype=np.float32),
        np.flatnonzero(available),
        np.flatnonzero(~matched_gt),
    )
