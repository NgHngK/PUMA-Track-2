from __future__ import annotations

import numpy as np

NUCLEUS_CLASSES: tuple[str, ...] = (
    "nuclei_tumor",
    "nuclei_lymphocyte",
    "nuclei_plasma_cell",
    "nuclei_histiocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_stroma",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_apoptosis",
)
CLASS_TO_ID = {name: index for index, name in enumerate(NUCLEUS_CLASSES)}
REJECT_CLASS_ID = len(NUCLEUS_CLASSES)
TISSUE_OUTPUT_LABELS: dict[str, int] = {
    "background": 0,
    "stroma": 1,
    "blood_vessel": 2,
    "tumor": 3,
    "epidermis": 4,
    "necrosis": 5,
}
TISSUE_CLASSES = tuple(TISSUE_OUTPUT_LABELS)

ROI_SIZE = 1024
MATCH_RADIUS_PIXELS = 15.0
VIEW_SIZES: dict[str, int] = {"V2": 64, "V3": 128, "V4": 256}
UNI2_EMBED_DIM = 1536
UNI2_POOLED_DIM = UNI2_EMBED_DIM * 3
BIOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "log_soft_area",
    "major_axis",
    "minor_axis",
    "axis_ratio",
    "eccentricity",
    "soft_perimeter",
    "circularity",
    "compactness",
    "h_mean",
    "h_std",
    "h_gradient_mean",
    "h_gradient_std",
    "h_laplacian_abs_mean",
    "h_entropy_proxy",
    "nucleus_minus_ring_h",
    "mask_peak_probability",
    "mask_mean_probability",
    "mask_confidence",
    "roi_h_robust_z",
    "roi_h_percentile",
)
BIOLOGY_DIM = len(BIOLOGY_FEATURE_NAMES)
SPATIAL_FEATURE_NAMES: tuple[str, ...] = (
    "log_knn3_mean_distance",
    "log_knn5_mean_distance",
    "log_knn10_mean_distance",
    "log_density_r25",
    "log_density_r50",
    "log_density_r100",
)
SPATIAL_DIM = len(SPATIAL_FEATURE_NAMES)
DETECTION_FEATURE_NAMES: tuple[str, ...] = (
    "heatmap_score",
    "quality",
    "log_uncertainty",
    "peak_sharpness",
    "border_reliability",
    "log_nearest_distance",
    "duplicate_count_r6",
    "local_score_margin",
)
DETECTION_DIM = len(DETECTION_FEATURE_NAMES)
TISSUE_CONTEXT_DIM = 18
STAGE1_PRIOR_DIM = len(NUCLEUS_CLASSES)

STAGE1_CANDIDATE_DTYPE = np.dtype(
    [
        ("oof_row_id", "i8"),
        ("roi_index", "i4"),
        ("candidate_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
        ("nearest_distance", "f4"),
        ("stage1_embedding", "f2", (STAGE1_PRIOR_DIM,)),
        ("matched_gt_index", "i4"),
        ("match_distance", "f4"),
        ("class_id", "i2"),
        ("is_reject", "u1"),
        ("fold", "i1"),
    ]
)

STAGE1_GT_FEATURE_DTYPE = np.dtype(
    [
        ("source_id", "i8"),
        ("roi_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("class_id", "i2"),
        ("fold", "i1"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
        ("stage1_embedding", "f2", (STAGE1_PRIOR_DIM,)),
    ]
)

CANDIDATE_DTYPE = np.dtype(
    [
        ("source_id", "i8"),
        ("roi_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("heatmap_score", "f4"),
        ("quality", "f4"),
        ("uncertainty", "f4"),
        ("peak_sharpness", "f4"),
        ("stage1_prior", "f2", (STAGE1_PRIOR_DIM,)),
        ("class_id", "i2"),
        ("kind", "u1"),
        ("fold", "i1"),
        ("gt_global_index", "i8"),
    ]
)
