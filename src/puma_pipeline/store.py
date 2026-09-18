from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import STAGE1_CANDIDATE_DTYPE, STAGE1_GT_FEATURE_DTYPE


@dataclass(slots=True)
class PumaArtifactStore:
    images: np.ndarray
    manifest: np.ndarray
    centroids: np.ndarray
    offsets: np.ndarray
    folds: np.ndarray

    @classmethod
    def open(cls, artifact_dir: Path) -> "PumaArtifactStore":
        names = {
            "images": "puma_rgb_images.npy",
            "manifest": "puma_roi_manifest.npy",
            "centroids": "puma_nuclei_centroids.npy",
            "offsets": "puma_roi_centroid_offsets.npy",
            "folds": "puma_fold_assignments.npy",
        }
        missing = [name for name, filename in names.items() if not (artifact_dir / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing preprocessing artifacts {missing} in {artifact_dir}.")
        arrays = {
            name: np.load(artifact_dir / filename, mmap_mode="r", allow_pickle=False)
            for name, filename in names.items()
        }
        if arrays["images"].ndim != 4 or arrays["images"].shape[1:] != (1024, 1024, 3):
            raise ValueError(f"Expected [N,1024,1024,3] RGB images, got {arrays['images'].shape}.")
        if len(arrays["manifest"]) != len(arrays["images"]):
            raise ValueError("Manifest and image counts disagree.")
        if len(arrays["folds"]) != len(arrays["images"]):
            raise ValueError("Fold assignments and image counts disagree.")
        if len(arrays["offsets"]) != len(arrays["images"]) + 1:
            raise ValueError("Centroid offsets must include one terminal entry.")
        if arrays["images"].dtype != np.uint8:
            raise ValueError(f"Preprocessed RGB images must be uint8, got {arrays['images'].dtype}.")
        offsets = np.asarray(arrays["offsets"])
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("Centroid offsets must use an integer dtype.")
        if offsets[0] != 0 or offsets[-1] != len(arrays["centroids"]) or np.any(np.diff(offsets) < 0):
            raise ValueError("Centroid offsets are not monotonic [0..N] boundaries for the centroid array.")
        folds = np.asarray(arrays["folds"])
        if not np.issubdtype(folds.dtype, np.integer) or np.any(folds < 0):
            raise ValueError("Fold assignments must be non-negative integers.")
        centroid_names = set(arrays["centroids"].dtype.names or ())
        required_centroid_fields = {"roi_index", "x", "y", "class_id"}
        if not required_centroid_fields <= centroid_names:
            raise ValueError(f"Centroid dtype is missing fields: {sorted(required_centroid_fields - centroid_names)}")
        centroids = arrays["centroids"]
        if len(centroids):
            roi_index = np.asarray(centroids["roi_index"], dtype=np.int64)
            class_id = np.asarray(centroids["class_id"], dtype=np.int64)
            x = np.asarray(centroids["x"], dtype=np.float64)
            y = np.asarray(centroids["y"], dtype=np.float64)
            if np.any((roi_index < 0) | (roi_index >= len(arrays["images"]))):
                raise ValueError("Centroid roi_index contains out-of-range values.")
            if np.any((class_id < 0) | (class_id >= 10)):
                raise ValueError("Centroid class_id must be in 0..9.")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Centroid coordinates contain non-finite values.")
            if np.any((x < 0.0) | (x >= 1024.0) | (y < 0.0) | (y >= 1024.0)):
                raise ValueError("Centroid coordinates must lie inside the 1024x1024 ROI.")
            for roi in range(len(arrays["images"])):
                start, stop = int(offsets[roi]), int(offsets[roi + 1])
                if stop > start and np.any(roi_index[start:stop] != roi):
                    raise ValueError(f"Centroid offset slice for ROI {roi} contains rows from another ROI.")
        return cls(**arrays)

    def roi_centroids(self, roi_index: int) -> np.ndarray:
        start = int(self.offsets[roi_index])
        stop = int(self.offsets[roi_index + 1])
        return self.centroids[start:stop]

    def global_gt_index(self, roi_index: int, local_gt_index: int) -> int:
        if local_gt_index < 0:
            return -1
        start = int(self.offsets[roi_index])
        stop = int(self.offsets[roi_index + 1])
        index = start + int(local_gt_index)
        if index >= stop:
            raise IndexError(f"ROI {roi_index} has no GT index {local_gt_index}.")
        return index


def _validate_stage1_coordinates(rows: np.ndarray, label: str) -> None:
    if len(rows) == 0:
        raise ValueError(f"{label} artifact is empty.")
    x = np.asarray(rows["x"], dtype=np.float64)
    y = np.asarray(rows["y"], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{label} contains non-finite coordinates.")
    if np.any((x < 0.0) | (x >= 1024.0) | (y < 0.0) | (y >= 1024.0)):
        raise ValueError(f"{label} coordinates must lie inside the 1024x1024 ROI.")
    if np.any(np.asarray(rows["roi_index"], dtype=np.int64) < 0):
        raise ValueError(f"{label} contains a negative roi_index.")


def _validate_stage1_numeric_features(rows: np.ndarray, label: str) -> None:
    for field in ("heatmap_score", "quality", "uncertainty", "peak_sharpness"):
        values = np.asarray(rows[field], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite {field} values.")
    embedding = np.asarray(rows["stage1_embedding"], dtype=np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError(f"{label} contains non-finite Stage-1 embeddings.")


def _validate_stage1_folds(rows: np.ndarray, expected_folds: int) -> None:
    observed = set(np.unique(rows["fold"]).astype(int).tolist())
    if observed != set(range(expected_folds)):
        raise ValueError(f"Expected folds 0..{expected_folds - 1}, got {sorted(observed)}.")


def load_stage1_oof(path: Path, expected_folds: int = 5) -> np.ndarray:
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if rows.dtype != STAGE1_CANDIDATE_DTYPE:
        raise ValueError("Stage-1 OOF dtype does not match the frozen stage contract.")
    _validate_stage1_coordinates(rows, "Stage-1 OOF")
    _validate_stage1_numeric_features(rows, "Stage-1 OOF")
    _validate_stage1_folds(rows, expected_folds)
    class_id = np.asarray(rows["class_id"], dtype=np.int64)
    is_reject = np.asarray(rows["is_reject"], dtype=np.int64)
    matched = np.asarray(rows["matched_gt_index"], dtype=np.int64)
    if np.any((class_id < 0) | (class_id > 10)):
        raise ValueError("Stage-1 OOF class_id must be in 0..10 (10=reject).")
    if np.any((is_reject != 0) & (is_reject != 1)):
        raise ValueError("Stage-1 OOF is_reject must contain only 0/1.")
    if np.any((is_reject == 1) != (class_id == 10)) or np.any((is_reject == 1) != (matched < 0)):
        raise ValueError("Stage-1 OOF reject/class/matched_gt_index fields are inconsistent.")
    positive = is_reject == 0
    distance = np.asarray(rows["match_distance"], dtype=np.float32)
    if np.any(~np.isfinite(distance[positive])) or np.any(distance[positive] < 0.0):
        raise ValueError("Matched Stage-1 OOF rows require finite non-negative match_distance.")
    nearest = np.asarray(rows["nearest_distance"], dtype=np.float32)
    if not np.isfinite(nearest).all() or np.any(nearest < 0.0):
        raise ValueError("Stage-1 OOF nearest_distance must be finite and non-negative.")
    return rows


def load_stage1_gt_features(path: Path, expected_folds: int = 5) -> np.ndarray:
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if rows.dtype != STAGE1_GT_FEATURE_DTYPE:
        raise ValueError("Stage-1 GT-feature dtype does not match the frozen stage contract.")
    _validate_stage1_coordinates(rows, "Stage-1 GT-feature")
    _validate_stage1_numeric_features(rows, "Stage-1 GT-feature")
    _validate_stage1_folds(rows, expected_folds)
    class_id = np.asarray(rows["class_id"], dtype=np.int64)
    if np.any((class_id < 0) | (class_id >= 10)):
        raise ValueError("Stage-1 GT-feature class_id must be in 0..9.")
    return rows
