from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Sampler

from ..constants import (
    BIOMASK_NEGATIVE_CROWDED,
    BIOMASK_NEGATIVE_EASY,
    BIOMASK_NEGATIVE_HIGH_SCORE,
    BIOMASK_NEGATIVE_NEAR,
    BIOMASK_NEGATIVE_STRATUM_COUNT,
    BIOMASK_POSITIVE,
)


@dataclass(frozen=True, slots=True)
class BioMaskSamplingReport:
    negative_count: int
    positive_count: int
    high_score_cutoff: float
    natural_counts: tuple[int, ...]
    natural_probabilities: tuple[float, ...]
    planned_counts: tuple[int, ...]
    planned_probabilities: tuple[float, ...]
    importance_weights: tuple[float, ...]
    oversampling_strength: float
    maximum_importance_weight: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_negative_strata(
    proposals: np.ndarray,
    targets: np.ndarray,
    nuclei: np.ndarray,
    indices: np.ndarray,
    *,
    crop_size: int,
    near_outer_radius: float,
    crowded_min_nuclei: int,
    high_score_quantile: float,
    high_score_cutoff: float | None = None,
) -> tuple[np.ndarray, float]:
    indices = np.asarray(indices, dtype=np.int64)
    strata = np.full(len(proposals), BIOMASK_POSITIVE, dtype=np.int8)
    negative_indices = indices[targets["nucleus_presence_target"][indices] == 0]
    if not len(negative_indices):
        return strata, float("inf")
    high_cutoff = (
        float(np.quantile(proposals["heatmap_score"][negative_indices], float(high_score_quantile)))
        if high_score_cutoff is None
        else float(high_score_cutoff)
    )
    crowd_radius = float(crop_size) / 2.0

    for roi in np.unique(proposals["roi_index"][negative_indices]).astype(int):
        local = negative_indices[proposals["roi_index"][negative_indices] == roi]
        gt_indices = np.flatnonzero(nuclei["roi_index"] == roi)
        if len(gt_indices):
            gt_xy = np.column_stack((nuclei["x"][gt_indices], nuclei["y"][gt_indices])).astype(np.float64)
            tree = cKDTree(gt_xy)
            p_xy = np.column_stack((proposals["x"][local], proposals["y"][local])).astype(np.float64)
            crowd_count = np.asarray([len(v) for v in tree.query_ball_point(p_xy, r=crowd_radius)], dtype=np.int32)
        else:
            crowd_count = np.zeros(len(local), dtype=np.int32)
        for proposal_index, count in zip(local, crowd_count, strict=True):
            distance = float(targets["nearest_distance_diagnostic"][proposal_index])
            if distance < float(near_outer_radius):
                value = BIOMASK_NEGATIVE_NEAR
            elif int(count) >= int(crowded_min_nuclei):
                value = BIOMASK_NEGATIVE_CROWDED
            elif float(proposals["heatmap_score"][proposal_index]) >= high_cutoff:
                value = BIOMASK_NEGATIVE_HIGH_SCORE
            else:
                value = BIOMASK_NEGATIVE_EASY
            strata[proposal_index] = value
    return strata, high_cutoff


def _integer_plan(total: int, probabilities: np.ndarray, available: np.ndarray) -> np.ndarray:
    if total <= 0:
        return np.zeros_like(available, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64).copy()
    probability[available <= 0] = 0.0
    if probability.sum() <= 0:
        raise ValueError("negative sampling plan has no available strata")
    probability /= probability.sum()
    raw = probability * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[int(idx)] += 1
    nonempty = np.flatnonzero(available > 0)
    if total >= len(nonempty):
        for idx in nonempty:
            if counts[idx] > 0:
                continue
            donor_order = np.argsort(-counts)
            donor = next((int(d) for d in donor_order if counts[d] > 1), None)
            if donor is not None:
                counts[donor] -= 1
                counts[idx] += 1
    if counts.sum() != total:
        raise RuntimeError("negative sampling integer plan does not preserve total count")
    return counts


def build_sampling_plan(
    proposals: np.ndarray,
    targets: np.ndarray,
    nuclei: np.ndarray,
    indices: np.ndarray,
    *,
    crop_size: int,
    near_outer_radius: float,
    crowded_min_nuclei: int,
    high_score_quantile: float,
    target_fractions: tuple[float, float, float, float],
    max_importance_weight: float,
) -> tuple[np.ndarray, np.ndarray, BioMaskSamplingReport]:
    indices = np.asarray(indices, dtype=np.int64)
    strata, high_cutoff = classify_negative_strata(
        proposals, targets, nuclei, indices,
        crop_size=crop_size,
        near_outer_radius=near_outer_radius,
        crowded_min_nuclei=crowded_min_nuclei,
        high_score_quantile=high_score_quantile,
    )
    positive = indices[targets["nucleus_presence_target"][indices] == 1]
    negative = indices[targets["nucleus_presence_target"][indices] == 0]
    natural_counts = np.asarray([(strata[negative] == s).sum() for s in range(BIOMASK_NEGATIVE_STRATUM_COUNT)], dtype=np.int64)
    negative_total = int(len(negative))
    if negative_total == 0:
        weights = np.ones(len(proposals), dtype=np.float32)
        report = BioMaskSamplingReport(
            0, int(len(positive)), high_cutoff,
            tuple(int(v) for v in natural_counts), tuple(0.0 for _ in natural_counts),
            tuple(int(v) for v in natural_counts), tuple(0.0 for _ in natural_counts),
            tuple(1.0 for _ in natural_counts), 0.0, 1.0,
        )
        return strata, weights, report

    p = natural_counts.astype(np.float64) / float(negative_total)
    desired = np.asarray(target_fractions, dtype=np.float64)
    if desired.shape != (BIOMASK_NEGATIVE_STRATUM_COUNT,):
        raise ValueError("target_fractions must have four entries")
    desired[natural_counts == 0] = 0.0
    desired = desired / desired.sum() if desired.sum() > 0 else p.copy()

    strength = 1.0
    planned_counts = natural_counts.copy()
    actual_q = p.copy()
    importance = np.ones_like(p)
    while True:
        q_proposed = (1.0 - strength) * p + strength * desired
        planned_counts = _integer_plan(negative_total, q_proposed, natural_counts)
        actual_q = planned_counts.astype(np.float64) / float(negative_total)
        importance = np.divide(p, actual_q, out=np.ones_like(p), where=actual_q > 0)
        active_max = float(importance[natural_counts > 0].max(initial=1.0))
        if active_max <= float(max_importance_weight) + 1.0e-12 or strength <= 0.0:
            break
        strength = max(0.0, strength - 0.1)

    presence_weights = np.ones(len(proposals), dtype=np.float32)
    for stratum in range(BIOMASK_NEGATIVE_STRATUM_COUNT):
        presence_weights[strata == stratum] = float(importance[stratum])
    report = BioMaskSamplingReport(
        negative_count=negative_total,
        positive_count=int(len(positive)),
        high_score_cutoff=high_cutoff,
        natural_counts=tuple(int(v) for v in natural_counts),
        natural_probabilities=tuple(float(v) for v in p),
        planned_counts=tuple(int(v) for v in planned_counts),
        planned_probabilities=tuple(float(v) for v in actual_q),
        importance_weights=tuple(float(v) for v in importance),
        oversampling_strength=float(strength),
        maximum_importance_weight=float(importance[natural_counts > 0].max(initial=1.0)),
    )
    # Exact conditional correction should integrate to one under the actual negative plan.
    if not np.isclose(float(np.sum(actual_q * importance)), 1.0, atol=1.0e-6):
        raise RuntimeError("BioMask importance correction does not integrate to one")
    return strata, presence_weights, report


class BioMaskStratifiedSampler(Sampler[int]):
    """Epoch sampler that preserves natural positive/negative prevalence.

    Positives are each exposed once. Negative subtype counts follow the planned
    distribution; sampling with replacement is allowed inside a scarce hard stratum.
    """

    def __init__(
        self,
        dataset_indices: np.ndarray,
        targets: np.ndarray,
        strata: np.ndarray,
        planned_negative_counts: tuple[int, ...],
        *,
        seed: int,
    ) -> None:
        self.dataset_indices = np.asarray(dataset_indices, dtype=np.int64)
        self.targets = targets
        self.strata = np.asarray(strata, dtype=np.int8)
        self.planned_negative_counts = tuple(int(v) for v in planned_negative_counts)
        self.seed = int(seed)
        self.epoch = 0
        self.global_to_local = {int(g): i for i, g in enumerate(self.dataset_indices.tolist())}
        self.positive_local = np.asarray(
            [self.global_to_local[int(g)] for g in self.dataset_indices if int(targets["nucleus_presence_target"][g]) == 1],
            dtype=np.int64,
        )
        self.negative_local: list[np.ndarray] = []
        for stratum in range(BIOMASK_NEGATIVE_STRATUM_COUNT):
            globals_in_stratum = self.dataset_indices[
                (targets["nucleus_presence_target"][self.dataset_indices] == 0)
                & (self.strata[self.dataset_indices] == stratum)
            ]
            self.negative_local.append(
                np.asarray([self.global_to_local[int(g)] for g in globals_in_stratum], dtype=np.int64)
            )
        if sum(self.planned_negative_counts) + len(self.positive_local) != len(self.dataset_indices):
            raise ValueError("BioMask sampler plan must preserve epoch length and pos/neg prevalence")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        pieces: list[np.ndarray] = []
        if len(self.positive_local):
            pieces.append(rng.permutation(self.positive_local))
        for available, count in zip(self.negative_local, self.planned_negative_counts, strict=True):
            if count <= 0:
                continue
            if not len(available):
                raise RuntimeError("planned BioMask negative stratum has no available samples")
            chosen = rng.choice(available, size=int(count), replace=bool(count > len(available)))
            pieces.append(np.asarray(chosen, dtype=np.int64))
        if not pieces:
            return iter(())
        merged = np.concatenate(pieces)
        if len(merged) != len(self.dataset_indices):
            raise RuntimeError("BioMask sampler emitted unexpected epoch length")
        return iter(rng.permutation(merged).tolist())

