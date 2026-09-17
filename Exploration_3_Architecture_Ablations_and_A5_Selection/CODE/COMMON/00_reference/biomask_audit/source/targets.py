from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_polygon

from ..constants import (
    BIOLOGICAL_SUPERVISION_RADIUS_PIXELS,
    INSTANCE_MODE_AMBIGUOUS,
    INSTANCE_MODE_CLEAR,
    INSTANCE_MODE_NONE,
    MASK_SUPERVISION_AMBIGUOUS,
    MASK_SUPERVISION_EMPTY,
    MASK_SUPERVISION_INSTANCE,
    NEGATIVE_BACKGROUND,
    NEGATIVE_HARD,
    NEGATIVE_NEAR_MISS,
    NUMBER_OF_CLASSES,
    POSITIVE_REGION,
    PUBLIC_MATCH_RADIUS_PIXELS,
    ROI_SIZE,
    SEMANTIC_MODE_HARD,
    SEMANTIC_MODE_RESOLVED,
    SEMANTIC_MODE_SAME_CLASS,
)
from .crops import CropTransform


BIOMASK_TARGET_DTYPE = np.dtype(
    [
        ("proposal_index_meta", "i8"),
        ("nearest_gt_index_meta", "i4"),
        ("nearest_distance_diagnostic", "f4"),
        ("second_distance_diagnostic", "f4"),
        ("nucleus_presence_target", "u1"),
        ("geometry_viability_target", "u1"),
        ("instance_gt_index_target", "i4"),
        ("instance_mode_target", "i1"),
        ("instance_weight_target", "f4"),
        ("presence_loss_weight_target", "f4"),
        ("mask_supervision_mode_target", "i1"),
        ("instance_mask_loss_weight_target", "f4"),
        ("empty_mask_loss_weight_target", "f4"),
        ("offset_loss_weight_target", "f4"),
        ("quality_loss_weight_target", "f4"),
        ("center_offset_x_target", "f4"),
        ("center_offset_y_target", "f4"),
        ("negative_stratum_target", "i1"),
    ]
)

SEMANTIC_TARGET_DTYPE = np.dtype(
    [
        ("semantic_class_target", "i2"),
        ("semantic_weight_target", "f4"),
        ("semantic_mode_target", "i1"),
        ("resolved_gt_index_meta", "i4"),
        ("resolved_group_meta", "i4"),
    ]
)

RESOLVER_DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("candidate_count_diagnostic", "i2"),
        ("candidate_class_count_diagnostic", "i1"),
        ("resolver_score_gt_diagnostic", "f4"),
        ("resolver_margin_gt_diagnostic", "f4"),
        ("resolver_selected_gt_meta", "i4"),
        ("resolved_target_in_public_radius_diagnostic", "u1"),
    ]
)


@dataclass(frozen=True, slots=True)
class TargetBuildSummary:
    biological_positive: int
    geometry_viable: int
    clear_instances: int
    ambiguous_instances: int
    backgrounds: int
    near_misses: int
    hard_negatives: int


@dataclass(frozen=True, slots=True)
class ResolverOperatingPoint:
    quality_threshold: float
    margin_threshold: float
    diagnostic_precision: float
    diagnostic_coverage: float
    diagnostic_correct: int
    diagnostic_resolved: int
    diagnostic_total: int


@dataclass(frozen=True, slots=True)
class ResolverSummary:
    hard: int
    same_class: int
    resolved: int
    dropped_cross_class: int
    no_candidate: int
    fold_operating_points: tuple[dict[str, Any], ...]
    gt_group_admission_by_class: tuple[float, ...]
    gt_group_admission_overall: float
    gt_group_admission_by_crowding: dict[str, float]
    gt_group_admission_by_size: dict[str, float]
    gt_group_admission_by_roi: dict[str, float]


def _polygon_from_flat(nuclei: np.ndarray, polygon_points: np.ndarray, gt_index: int) -> np.ndarray:
    row = nuclei[int(gt_index)]
    start = int(row["polygon_start"])
    length = int(row["polygon_length"])
    points = np.asarray(polygon_points[start : start + length], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"invalid polygon for nucleus {gt_index}")
    return points


def _point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    points = np.asarray(polygon, dtype=np.float64)
    px = points[:, 0]
    py = points[:, 1]
    x0, y0 = px[-1], py[-1]
    inside = False
    for x1, y1 in zip(px, py, strict=True):
        crosses = (y1 > y) != (y0 > y)
        if crosses:
            x_at_y = (x0 - x1) * (y - y1) / ((y0 - y1) + 1.0e-12) + x1
            if x < x_at_y:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _point_polygon_distance(x: float, y: float, polygon: np.ndarray) -> float:
    if _point_inside_polygon(x, y, polygon):
        return 0.0
    point = np.asarray((x, y), dtype=np.float64)
    points = np.asarray(polygon, dtype=np.float64)
    distances = [
        _point_segment_distance(point, points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
    ]
    return float(min(distances)) if distances else float("inf")


def _polygon_mask_in_transform(points: np.ndarray, transform: CropTransform) -> np.ndarray:
    local = transform.roi_to_local(points)
    mask = np.zeros((transform.crop_size, transform.crop_size), dtype=np.float32)
    rr, cc = draw_polygon(local[:, 1], local[:, 0], shape=mask.shape)
    mask[rr, cc] = 1.0
    mask *= transform.valid_region_mask(dtype=np.float32)
    return mask


def _candidate_indices_for_roi(
    proposal_xy: np.ndarray,
    gt_indices: np.ndarray,
    nuclei: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    if len(gt_indices) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(len(proposal_xy))]
    gt_xy = np.column_stack((nuclei["x"][gt_indices], nuclei["y"][gt_indices])).astype(np.float64)
    tree = cKDTree(gt_xy)
    proposal_xy = proposal_xy.astype(np.float64)
    local_lists = tree.query_ball_point(proposal_xy, r=float(radius))
    output: list[np.ndarray] = []
    for row, local in enumerate(local_lists):
        local = np.asarray(local, dtype=np.int64)
        if len(local):
            delta = gt_xy[local] - proposal_xy[row]
            local = local[np.hypot(delta[:, 0], delta[:, 1]) < float(radius)]
        output.append(gt_indices[local])
    return output


def presence_boundary_weight(
    distance: float,
    *,
    biological_radius: float,
    ramp_pixels: float,
    minimum_weight: float,
) -> float:
    if ramp_pixels <= 0 or not (0.0 < minimum_weight <= 1.0):
        raise ValueError("invalid presence boundary-ramp settings")
    normalized = min(1.0, abs(float(distance) - float(biological_radius)) / float(ramp_pixels))
    return float(minimum_weight + (1.0 - minimum_weight) * normalized)


def build_biomask_targets(
    proposals: np.ndarray,
    nuclei: np.ndarray,
    polygon_points: np.ndarray,
    *,
    number_of_rois: int,
    semantic_radius: float = PUBLIC_MATCH_RADIUS_PIXELS,
    biological_radius: float = BIOLOGICAL_SUPERVISION_RADIUS_PIXELS,
    instance_margin: float = 3.0,
    boundary_ramp_pixels: float = 2.0,
    boundary_min_weight: float = 0.25,
    hard_score_quantile: float = 0.80,
) -> tuple[np.ndarray, TargetBuildSummary]:
    if not (0.0 < float(semantic_radius) <= float(biological_radius)):
        raise ValueError("require 0 < semantic_radius <= biological_radius")
    targets = np.zeros(len(proposals), dtype=BIOMASK_TARGET_DTYPE)
    targets["proposal_index_meta"] = np.arange(len(proposals), dtype=np.int64)
    targets["nearest_gt_index_meta"] = -1
    targets["nearest_distance_diagnostic"] = np.inf
    targets["second_distance_diagnostic"] = np.inf
    targets["instance_gt_index_target"] = -1
    targets["instance_mode_target"] = INSTANCE_MODE_NONE
    targets["presence_loss_weight_target"] = 1.0
    targets["mask_supervision_mode_target"] = MASK_SUPERVISION_EMPTY
    targets["negative_stratum_target"] = NEGATIVE_BACKGROUND

    hard_cutoff = float(np.quantile(proposals["heatmap_score"], hard_score_quantile)) if len(proposals) else float("inf")
    clear = ambiguous = 0
    for roi in range(int(number_of_rois)):
        p_idx = np.flatnonzero(proposals["roi_index"] == roi)
        if not len(p_idx):
            continue
        g_idx = np.flatnonzero(nuclei["roi_index"] == roi)
        if not len(g_idx):
            high = proposals["heatmap_score"][p_idx] >= hard_cutoff
            targets["negative_stratum_target"][p_idx[high]] = NEGATIVE_HARD
            targets["mask_supervision_mode_target"][p_idx] = MASK_SUPERVISION_EMPTY
            targets["empty_mask_loss_weight_target"][p_idx] = 1.0
            targets["presence_loss_weight_target"][p_idx] = 1.0
            continue
        gt_xy = np.column_stack((nuclei["x"][g_idx], nuclei["y"][g_idx])).astype(np.float64)
        p_xy = np.column_stack((proposals["x"][p_idx], proposals["y"][p_idx])).astype(np.float64)
        tree = cKDTree(gt_xy)
        k = min(2, len(g_idx))
        distances, neighbors = tree.query(p_xy, k=k)
        if k == 1:
            distances = np.asarray(distances)[:, None]
            neighbors = np.asarray(neighbors)[:, None]
        biological_candidates = _candidate_indices_for_roi(p_xy, g_idx, nuclei, biological_radius)
        semantic_candidates = _candidate_indices_for_roi(p_xy, g_idx, nuclei, semantic_radius)

        for local_row, proposal_index in enumerate(p_idx):
            nearest = float(distances[local_row, 0])
            second = float(distances[local_row, 1]) if distances.shape[1] > 1 else float("inf")
            nearest_gt = int(g_idx[int(neighbors[local_row, 0])])
            targets["nearest_gt_index_meta"][proposal_index] = nearest_gt
            targets["nearest_distance_diagnostic"][proposal_index] = nearest
            targets["second_distance_diagnostic"][proposal_index] = second
            has_biology = bool(len(biological_candidates[local_row]))
            has_semantic = bool(len(semantic_candidates[local_row]))
            targets["nucleus_presence_target"][proposal_index] = int(has_biology)
            targets["geometry_viability_target"][proposal_index] = int(has_semantic)
            targets["presence_loss_weight_target"][proposal_index] = presence_boundary_weight(
                nearest, biological_radius=biological_radius, ramp_pixels=boundary_ramp_pixels,
                minimum_weight=boundary_min_weight,
            )
            if has_biology:
                targets["negative_stratum_target"][proposal_index] = POSITIVE_REGION
                targets["mask_supervision_mode_target"][proposal_index] = MASK_SUPERVISION_AMBIGUOUS
            else:
                targets["mask_supervision_mode_target"][proposal_index] = MASK_SUPERVISION_EMPTY
                targets["empty_mask_loss_weight_target"][proposal_index] = targets["presence_loss_weight_target"][proposal_index]
                if nearest < biological_radius + 5.0:
                    targets["negative_stratum_target"][proposal_index] = NEGATIVE_NEAR_MISS
                elif proposals["heatmap_score"][proposal_index] >= hard_cutoff:
                    targets["negative_stratum_target"][proposal_index] = NEGATIVE_HARD

            if not has_biology:
                continue
            candidates = biological_candidates[local_row]
            x = float(proposals["x"][proposal_index])
            y = float(proposals["y"][proposal_index])
            containing = [
                int(gt) for gt in candidates
                if _point_inside_polygon(x, y, _polygon_from_flat(nuclei, polygon_points, int(gt)))
            ]
            chosen = -1
            mode = INSTANCE_MODE_AMBIGUOUS
            weight = 0.0
            if len(containing) == 1:
                chosen = containing[0]
                mode = INSTANCE_MODE_CLEAR
                weight = 1.0
            elif len(candidates) == 1:
                chosen = int(candidates[0])
                mode = INSTANCE_MODE_CLEAR
                weight = 1.0
            elif second - nearest >= float(instance_margin):
                chosen = nearest_gt
                mode = INSTANCE_MODE_CLEAR
                weight = 1.0
            if chosen >= 0:
                clear += 1
                targets["instance_gt_index_target"][proposal_index] = chosen
                targets["instance_mode_target"][proposal_index] = mode
                targets["instance_weight_target"][proposal_index] = weight
                targets["mask_supervision_mode_target"][proposal_index] = MASK_SUPERVISION_INSTANCE
                targets["instance_mask_loss_weight_target"][proposal_index] = weight
                targets["offset_loss_weight_target"][proposal_index] = weight
                targets["quality_loss_weight_target"][proposal_index] = weight
                # The center head predicts target displacement from the actual prompt
                # coordinate, not from the rounded crop center. Keeping this float-
                # relative makes the supervision consistent with shifted prompt crops.
                targets["center_offset_x_target"][proposal_index] = float(nuclei["x"][chosen] - x)
                targets["center_offset_y_target"][proposal_index] = float(nuclei["y"][chosen] - y)
            else:
                ambiguous += 1
                targets["instance_mode_target"][proposal_index] = INSTANCE_MODE_AMBIGUOUS
                targets["mask_supervision_mode_target"][proposal_index] = MASK_SUPERVISION_AMBIGUOUS

    strata = targets["negative_stratum_target"]
    summary = TargetBuildSummary(
        biological_positive=int(np.sum(targets["nucleus_presence_target"] == 1)),
        geometry_viable=int(np.sum(targets["geometry_viability_target"] == 1)),
        clear_instances=int(clear),
        ambiguous_instances=int(ambiguous),
        backgrounds=int(np.sum(strata == NEGATIVE_BACKGROUND)),
        near_misses=int(np.sum(strata == NEGATIVE_NEAR_MISS)),
        hard_negatives=int(np.sum(strata == NEGATIVE_HARD)),
    )
    return targets, summary


def _soft_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=np.float32).clip(0.0, 1.0)
    t = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
    inter = float(np.minimum(p, t).sum())
    union = float(np.maximum(p, t).sum())
    return inter / union if union > 1.0e-6 else 0.0


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    total = float(values.sum())
    if total <= 1.0e-6:
        return None
    yy, xx = np.indices(values.shape, dtype=np.float32)
    return float((values * xx).sum() / total), float((values * yy).sum() / total)


def _resolver_candidates(
    proposal: np.void,
    nuclei: np.ndarray,
    semantic_gt_indices: np.ndarray,
    polygon_points: np.ndarray,
    predicted_mask: np.ndarray,
    transform: CropTransform,
    geometry_weight: float,
) -> list[tuple[int, float]]:
    predicted = np.asarray(predicted_mask, dtype=np.float32)
    if predicted.size and float(np.nanmax(predicted)) > 1.5:
        predicted = predicted / 255.0
    predicted = np.nan_to_num(predicted, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    predicted *= transform.valid_region_mask(dtype=np.float32)
    centroid = _mask_centroid(predicted)
    output: list[tuple[int, float]] = []
    for gt_index in semantic_gt_indices:
        gt_index = int(gt_index)
        polygon = _polygon_from_flat(nuclei, polygon_points, gt_index)
        gt_mask = _polygon_mask_in_transform(polygon, transform)
        iou = _soft_iou(predicted, gt_mask)
        if centroid is None:
            centroid_consistency = 0.0
        else:
            gt_local = transform.roi_to_local(np.asarray([[nuclei["x"][gt_index], nuclei["y"][gt_index]]], dtype=np.float32))[0]
            distance = float(np.linalg.norm(np.asarray(centroid) - gt_local))
            centroid_consistency = float(np.exp(-distance / 8.0))
        x, y = float(proposal["x"]), float(proposal["y"])
        inside = 1.0 if _point_inside_polygon(x, y, polygon) else 0.0
        polygon_distance = _point_polygon_distance(x, y, polygon)
        geometry = 0.5 * inside + 0.5 * float(np.exp(-polygon_distance / 5.0))
        mask_evidence = 0.7 * iou + 0.3 * centroid_consistency
        score = float(mask_evidence + float(geometry_weight) * geometry)
        output.append((gt_index, score))
    output.sort(key=lambda item: (-item[1], item[0]))
    return output


def _diagnose_fixed_operating_point(
    proposals: np.ndarray,
    nuclei: np.ndarray,
    polygon_points: np.ndarray,
    biomask_targets: np.ndarray,
    predicted_masks: np.ndarray,
    indices: np.ndarray,
    *,
    crop_size: int,
    semantic_radius: float,
    geometry_weight: float,
    quality_threshold: float,
    margin_threshold: float,
) -> ResolverOperatingPoint:
    """Evaluate a fixed resolver policy on clear OOF examples without selecting it.

    This is diagnostic only. The thresholds are fixed in configuration so labels for
    a held fold never depend on metrics from models that may have trained on it.
    """
    correct = resolved = total = 0
    for proposal_index in np.asarray(indices, dtype=np.int64):
        target_gt = int(biomask_targets["instance_gt_index_target"][proposal_index])
        if target_gt < 0 or float(biomask_targets["instance_weight_target"][proposal_index]) <= 0:
            continue
        roi = int(proposals["roi_index"][proposal_index])
        g_idx = np.flatnonzero(nuclei["roi_index"] == roi)
        if not len(g_idx):
            continue
        dx = nuclei["x"][g_idx].astype(np.float64) - float(proposals["x"][proposal_index])
        dy = nuclei["y"][g_idx].astype(np.float64) - float(proposals["y"][proposal_index])
        semantic = g_idx[np.hypot(dx, dy) < float(semantic_radius)]
        if target_gt not in set(int(value) for value in semantic):
            continue
        total += 1
        transform = CropTransform.from_center(
            float(proposals["x"][proposal_index]), float(proposals["y"][proposal_index]),
            crop_size, (ROI_SIZE, ROI_SIZE),
        )
        candidates = _resolver_candidates(
            proposals[proposal_index], nuclei, semantic, polygon_points,
            predicted_masks[proposal_index], transform, geometry_weight,
        )
        if not candidates:
            continue
        best_gt, best_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = float(best_score - second_score)
        if best_score >= float(quality_threshold) and margin >= float(margin_threshold):
            resolved += 1
            correct += int(int(best_gt) == target_gt)
    precision = float(correct / resolved) if resolved else 1.0
    coverage = float(resolved / total) if total else 0.0
    return ResolverOperatingPoint(
        float(quality_threshold), float(margin_threshold), precision, coverage,
        int(correct), int(resolved), int(total),
    )


def _gt_admission_audit(
    nuclei: np.ndarray,
    polygon_points: np.ndarray,
    admitted_gt: set[int],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
    valid = [int(i) for i, cls in enumerate(nuclei["class_id"].astype(int)) if 0 <= cls < NUMBER_OF_CLASSES]
    if not valid:
        return 0.0, {}, {}, {}
    overall = float(sum(i in admitted_gt for i in valid) / len(valid))

    # Crowding is measured by GT-centroid nearest-neighbor distance within each ROI.
    crowding: dict[str, list[bool]] = {"lt12": [], "12to20": [], "20to32": [], "ge32": []}
    size_values: list[tuple[int, float]] = []
    roi_values: dict[str, list[bool]] = {}
    for roi in np.unique(nuclei["roi_index"]).astype(int):
        idx = np.flatnonzero(nuclei["roi_index"] == roi)
        idx = idx[np.isin(idx, valid)]
        if not len(idx):
            continue
        xy = np.column_stack((nuclei["x"][idx], nuclei["y"][idx])).astype(np.float64)
        nearest = np.full(len(idx), np.inf, dtype=np.float64)
        if len(idx) > 1:
            distances, _ = cKDTree(xy).query(xy, k=2)
            nearest = distances[:, 1]
        for local, gt_index in enumerate(idx):
            d = float(nearest[local])
            key = "lt12" if d < 12.0 else "12to20" if d < 20.0 else "20to32" if d < 32.0 else "ge32"
            admitted = int(gt_index) in admitted_gt
            crowding[key].append(admitted)
            roi_values.setdefault(str(roi), []).append(admitted)
            try:
                polygon = _polygon_from_flat(nuclei, polygon_points, int(gt_index))
                x, y = polygon[:, 0], polygon[:, 1]
                area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            except Exception:
                area = 0.0
            size_values.append((int(gt_index), area))

    positive_areas = np.asarray([area for _, area in size_values if area > 0], dtype=np.float64)
    if len(positive_areas) >= 3:
        q1, q2 = np.quantile(positive_areas, (1 / 3, 2 / 3))
    elif len(positive_areas):
        q1 = q2 = float(np.median(positive_areas))
    else:
        q1 = q2 = 0.0
    size: dict[str, list[bool]] = {"small": [], "medium": [], "large": []}
    for gt_index, area in size_values:
        key = "small" if area <= q1 else "medium" if area <= q2 else "large"
        size[key].append(gt_index in admitted_gt)

    def rates(values: dict[str, list[bool]]) -> dict[str, float]:
        return {key: float(np.mean(items)) if items else 0.0 for key, items in values.items()}
    return overall, rates(crowding), rates(size), rates(roi_values)

def resolve_semantic_targets(
    proposals: np.ndarray,
    nuclei: np.ndarray,
    polygon_points: np.ndarray,
    biomask_targets: np.ndarray,
    predicted_masks: np.ndarray,
    *,
    number_of_folds: int,
    crop_size: int,
    semantic_radius: float = PUBLIC_MATCH_RADIUS_PIXELS,
    geometry_weight: float = 0.25,
    quality_threshold: float = 0.65,
    margin_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, ResolverSummary]:
    semantic = np.zeros(len(proposals), dtype=SEMANTIC_TARGET_DTYPE)
    semantic["semantic_class_target"] = -1
    semantic["resolved_gt_index_meta"] = -1
    semantic["resolved_group_meta"] = -1
    diagnostics = np.zeros(len(proposals), dtype=RESOLVER_DIAGNOSTIC_DTYPE)
    diagnostics["resolver_selected_gt_meta"] = -1

    operating_points: dict[int, ResolverOperatingPoint] = {}
    for held_fold in range(int(number_of_folds)):
        diagnostic_indices = np.flatnonzero(proposals["fold"] != held_fold)
        operating_points[held_fold] = _diagnose_fixed_operating_point(
            proposals, nuclei, polygon_points, biomask_targets, predicted_masks, diagnostic_indices,
            crop_size=crop_size, semantic_radius=semantic_radius, geometry_weight=geometry_weight,
            quality_threshold=quality_threshold, margin_threshold=margin_threshold,
        )

    hard = same_class = resolved = dropped = no_candidate = 0
    admitted_gt: list[set[int]] = [set() for _ in range(NUMBER_OF_CLASSES)]
    all_gt: list[set[int]] = [set() for _ in range(NUMBER_OF_CLASSES)]
    for gt_index, cls in enumerate(nuclei["class_id"].astype(int)):
        if 0 <= cls < NUMBER_OF_CLASSES:
            all_gt[cls].add(int(gt_index))

    for roi in np.unique(proposals["roi_index"]).astype(int):
        p_idx = np.flatnonzero(proposals["roi_index"] == roi)
        g_idx = np.flatnonzero(nuclei["roi_index"] == roi)
        if not len(p_idx) or not len(g_idx):
            continue
        p_xy = np.column_stack((proposals["x"][p_idx], proposals["y"][p_idx])).astype(np.float64)
        candidates_by_row = _candidate_indices_for_roi(p_xy, g_idx, nuclei, semantic_radius)
        for local_row, proposal_index in enumerate(p_idx):
            candidates = candidates_by_row[local_row]
            diagnostics["candidate_count_diagnostic"][proposal_index] = len(candidates)
            if not len(candidates):
                no_candidate += 1
                continue
            classes = nuclei["class_id"][candidates].astype(np.int64)
            unique_classes = np.unique(classes)
            diagnostics["candidate_class_count_diagnostic"][proposal_index] = len(unique_classes)
            if len(candidates) == 1:
                gt = int(candidates[0])
                cls = int(nuclei["class_id"][gt])
                semantic[proposal_index] = (cls, 1.0, SEMANTIC_MODE_HARD, gt, gt)
                diagnostics["resolver_selected_gt_meta"][proposal_index] = gt
                diagnostics["resolved_target_in_public_radius_diagnostic"][proposal_index] = 1
                admitted_gt[cls].add(gt)
                hard += 1
                continue
            if len(unique_classes) == 1:
                cls = int(unique_classes[0])
                # Semantic identity is certain even if the physical instance is not.
                # Use the nearest eligible GT only as a sampling-group metadata key;
                # do not pretend that the instance itself was resolved.
                px = float(proposals["x"][proposal_index])
                py = float(proposals["y"][proposal_index])
                distances = np.hypot(
                    nuclei["x"][candidates].astype(np.float64) - px,
                    nuclei["y"][candidates].astype(np.float64) - py,
                )
                group = int(candidates[int(np.argmin(distances))])
                semantic[proposal_index] = (cls, 1.0, SEMANTIC_MODE_SAME_CLASS, -1, group)
                admitted_gt[cls].add(group)
                same_class += 1
                continue

            fold = int(proposals["fold"][proposal_index])
            op = operating_points[fold]
            transform = CropTransform.from_center(
                float(proposals["x"][proposal_index]), float(proposals["y"][proposal_index]), crop_size, (ROI_SIZE, ROI_SIZE)
            )
            scored = _resolver_candidates(
                proposals[proposal_index], nuclei, candidates, polygon_points,
                predicted_masks[proposal_index], transform, geometry_weight,
            )
            if not scored:
                dropped += 1
                continue
            gt, best_score = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else 0.0
            margin = float(best_score - second_score)
            diagnostics["resolver_score_gt_diagnostic"][proposal_index] = best_score
            diagnostics["resolver_margin_gt_diagnostic"][proposal_index] = margin
            diagnostics["resolver_selected_gt_meta"][proposal_index] = int(gt)
            if best_score >= op.quality_threshold and margin >= op.margin_threshold:
                cls = int(nuclei["class_id"][gt])
                semantic[proposal_index] = (cls, 1.0, SEMANTIC_MODE_RESOLVED, int(gt), int(gt))
                diagnostics["resolved_target_in_public_radius_diagnostic"][proposal_index] = 1
                admitted_gt[cls].add(int(gt))
                resolved += 1
            else:
                dropped += 1

    admission = tuple(
        float(len(admitted_gt[c]) / len(all_gt[c])) if len(all_gt[c]) else 0.0
        for c in range(NUMBER_OF_CLASSES)
    )
    admitted_union = set().union(*admitted_gt) if admitted_gt else set()
    admission_overall, admission_crowding, admission_size, admission_roi = _gt_admission_audit(
        nuclei, polygon_points, admitted_union
    )
    summary = ResolverSummary(
        hard=int(hard),
        same_class=int(same_class),
        resolved=int(resolved),
        dropped_cross_class=int(dropped),
        no_candidate=int(no_candidate),
        fold_operating_points=tuple(asdict(operating_points[f]) | {"held_fold": int(f)} for f in sorted(operating_points)),
        gt_group_admission_by_class=admission,
        gt_group_admission_overall=admission_overall,
        gt_group_admission_by_crowding=admission_crowding,
        gt_group_admission_by_size=admission_size,
        gt_group_admission_by_roi=admission_roi,
    )
    return semantic, diagnostics, summary


def save_array(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)
