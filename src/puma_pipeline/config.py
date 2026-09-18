from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import ROI_SIZE, UNI2_POOLED_DIM, VIEW_SIZES


@dataclass(slots=True)
class PumaConfig:
    project_root: str
    artifact_dir: str = "PUMA_outputs"
    stage1_output_dir: str = "PUMA_stage1_outputs"
    tissue_output_dir: str = "PUMA_tissue_outputs"
    stage2_output_dir: str = "PUMA_stage2_outputs"
    cache_dir: str = "PUMA_stage2_cache"
    uni2_checkpoint: str = "PUMA_pretrained_checkpoints/uni2_h_model.bin"
    image_dir: str = "Dataset/01_training_dataset_tif_ROIs"
    nuclei_geojson_dir: str = "Dataset/01_training_dataset_geojson_nuclei"
    tissue_geojson_dir: str = "Dataset/01_training_dataset_geojson_tissue"

    seed: int = 2026
    number_of_folds: int = 5
    fold_assignment_method: str = "capacity_global_delta"
    image_size: int = ROI_SIZE
    match_radius_pixels: float = 15.0
    expected_public_rois: int = 205
    expected_public_nuclei: int = 97378
    strict_expected_dataset: bool = False

    use_bfloat16: bool = True
    use_tf32: bool = True
    deterministic: bool = False
    number_of_workers: int = 0
    prefetch_factor: int = 2

    stage1_architecture: str = "native1024_sparse"
    stage1_epochs: int = 30
    stage1_base_channels: int = 64
    stage1_micro_batch_size: int = 8
    stage1_effective_batch_size: int = 8
    stage1_inference_batch_size: int = 8
    stage1_repeats_per_roi_per_epoch: int = 4
    stage1_learning_rate: float = 3.0e-4
    stage1_minimum_learning_rate: float = 2.0e-6
    stage1_weight_decay: float = 5.0e-5
    stage1_warmup_epochs: int = 3
    stage1_gradient_clip_norm: float = 1.0
    stage1_validation_interval: int = 5
    stage1_checkpoint_every_steps: int = 100
    stage1_ema_decay: float = 0.999
    stage1_early_stopping_patience: int = 15
    stage1_early_stopping_min_delta: float = 0.0
    stage1_compile: bool = True
    stage1_heatmap_prior_probability: float = 0.05
    stage1_heatmap_sigma_pixels: float = 2.5
    stage1_offset_radius_pixels: float = 6.0
    stage1_heatmap_loss_weight: float = 0.55
    stage1_offset_loss_weight: float = 0.25
    stage1_quality_loss_weight: float = 0.08
    stage1_uncertainty_loss_weight: float = 0.02
    stage1_coarse_type_loss_weight: float = 0.08
    stage1_deployment_threshold: float = 0.06
    stage1_deployment_local_max_radius: int = 2
    stage1_deployment_suppression_radius: float = 3.0

    stage2_feature_contract: str = "self_exclusion_pixel_sampling"
    stage2_epochs: int = 100
    stage2_local_epochs: int = 70
    stage2_views: tuple[str, ...] = ("V2", "V3", "V4")
    stage2_cache_variants: int = 2
    stage2_pooled_feature_dim: int = UNI2_POOLED_DIM
    stage2_hidden_dim: int = 256
    stage2_biology_hidden_dim: int = 96
    stage2_spatial_hidden_dim: int = 48
    stage2_tissue_hidden_dim: int = 96
    stage2_prior_hidden_dim: int = 48
    biomask_crop_size: int = 96
    biomask_base_channels: int = 32
    stage2_cache_batch_size: int = 512
    stage2_uni2_micro_batch_size: int = 96
    stage2_train_batch_size: int = 1024
    stage2_roi_batch_size: int = 4
    stage2_inference_batch_size: int = 2048
    deployment_uni2_micro_batch_size: int = 8
    deployment_stage2_inference_batch_size: int = 512

    local_learning_rate: float = 2.5e-4
    biomask_learning_rate: float = 2.5e-4
    fusion_learning_rate: float = 1.5e-4
    graph_learning_rate: float = 1.0e-4
    validity_learning_rate: float = 2.0e-4
    context_local_lr_multiplier: float = 0.15
    stage2_minimum_learning_rate: float = 2.0e-6
    stage2_weight_decay: float = 1.0e-4
    stage2_warmup_epochs: int = 5
    stage2_gradient_clip_norm: float = 1.0
    stage2_ema_decay: float = 0.997
    label_smoothing: float = 0.02
    mask_loss_weight: float = 0.35
    graph_loss_weight: float = 0.65
    validity_loss_weight: float = 0.65
    roi_class_balanced_fraction: float = 0.65

    stage1_prior_dropout: float = 0.35
    tissue_context_dropout: float = 0.25
    biology_context_dropout: float = 0.10
    graph_neighbor_k: int = 12
    graph_radius_cap: float = 160.0
    graph_probability_dropout: float = 0.15
    graph_probability_temperature: float = 1.5

    tissue_epochs: int = 100
    tissue_base_channels: int = 32
    tissue_batch_size: int = 16
    tissue_learning_rate: float = 3.0e-4
    tissue_minimum_learning_rate: float = 2.0e-6
    tissue_weight_decay: float = 1.0e-4
    tissue_warmup_epochs: int = 5
    tissue_ema_decay: float = 0.997
    tissue_dice_weight: float = 0.6
    tissue_ce_weight: float = 0.4
    tissue_context_pool_radius: int = 32

    risk_fit_steps: int = 600
    risk_learning_rate: float = 0.03
    risk_weight_decay: float = 1.0e-3
    global_threshold_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(round(0.02 + i * 0.01, 3) for i in range(97))
    )

    def __post_init__(self) -> None:
        self.stage2_views = tuple(self.stage2_views)
        self.global_threshold_grid = tuple(float(v) for v in self.global_threshold_grid)
        if self.fold_assignment_method != "capacity_global_delta":
            raise ValueError("Unsupported fold assignment method.")
        if self.number_of_folds < 2:
            raise ValueError("number_of_folds must be >= 2.")
        if self.match_radius_pixels <= 0.0:
            raise ValueError("match_radius_pixels must be positive.")
        if self.expected_public_rois < 0 or self.expected_public_nuclei < 0:
            raise ValueError("Expected dataset counts cannot be negative.")
        if self.image_size != ROI_SIZE:
            raise ValueError("PUMA is locked to 1024x1024 ROI inference.")
        if self.stage1_architecture != "native1024_sparse":
            raise ValueError("Unsupported Stage-1 architecture.")
        if self.stage1_epochs < 1:
            raise ValueError("stage1_epochs must be positive.")
        if not 0.0 < self.stage1_heatmap_prior_probability < 1.0:
            raise ValueError("stage1_heatmap_prior_probability must be in (0,1).")
        if self.stage1_heatmap_sigma_pixels <= 0.0:
            raise ValueError("stage1_heatmap_sigma_pixels must be positive.")
        if self.stage1_offset_radius_pixels <= 0.0:
            raise ValueError("stage1_offset_radius_pixels must be positive.")
        if self.stage1_deployment_local_max_radius < 1:
            raise ValueError("stage1_deployment_local_max_radius must be positive.")
        if self.stage1_micro_batch_size < 1 or self.stage1_inference_batch_size < 1:
            raise ValueError("Stage-1 batch sizes must be positive.")
        if self.stage1_effective_batch_size < self.stage1_micro_batch_size:
            raise ValueError("Stage-1 effective batch cannot be smaller than its micro batch.")
        if self.stage1_effective_batch_size % self.stage1_micro_batch_size:
            raise ValueError("Stage-1 effective batch must be divisible by micro batch.")
        if self.stage1_repeats_per_roi_per_epoch < 1:
            raise ValueError("stage1_repeats_per_roi_per_epoch must be positive.")
        if self.stage1_learning_rate <= 0.0 or self.stage1_minimum_learning_rate <= 0.0:
            raise ValueError("Stage-1 learning rates must be positive.")
        if self.stage1_minimum_learning_rate > self.stage1_learning_rate:
            raise ValueError("Stage-1 minimum LR cannot exceed its peak LR.")
        if self.stage1_weight_decay < 0.0:
            raise ValueError("stage1_weight_decay cannot be negative.")
        if not 0.0 < self.stage1_deployment_threshold < 1.0:
            raise ValueError("stage1_deployment_threshold must be in (0,1).")
        if self.stage1_deployment_suppression_radius <= 0.0:
            raise ValueError("stage1_deployment_suppression_radius must be positive.")
        if self.stage1_checkpoint_every_steps < 1 or self.stage1_validation_interval < 1:
            raise ValueError("Stage-1 checkpoint/validation intervals must be positive.")
        if self.stage1_early_stopping_patience < 1:
            raise ValueError("stage1_early_stopping_patience must be positive.")
        if self.stage1_early_stopping_min_delta < 0.0:
            raise ValueError("stage1_early_stopping_min_delta cannot be negative.")
        if self.stage1_base_channels < 1:
            raise ValueError("stage1_base_channels must be positive.")
        if self.stage1_warmup_epochs < 0 or self.stage1_warmup_epochs >= self.stage1_epochs:
            raise ValueError("stage1_warmup_epochs must be in [0, stage1_epochs).")
        if self.stage1_gradient_clip_norm <= 0.0 or self.stage2_gradient_clip_norm <= 0.0:
            raise ValueError("Stage-1/Stage-2 gradient clip norms must be positive.")
        for name in (
            "stage1_heatmap_loss_weight", "stage1_offset_loss_weight", "stage1_quality_loss_weight",
            "stage1_uncertainty_loss_weight", "stage1_coarse_type_loss_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if sum(float(getattr(self, name)) for name in (
            "stage1_heatmap_loss_weight", "stage1_offset_loss_weight", "stage1_quality_loss_weight",
            "stage1_uncertainty_loss_weight", "stage1_coarse_type_loss_weight",
        )) <= 0.0:
            raise ValueError("At least one Stage-1 loss weight must be positive.")
        if self.stage2_feature_contract != "self_exclusion_pixel_sampling":
            raise ValueError("Unsupported Stage-2 feature contract.")
        if self.stage2_epochs < 1:
            raise ValueError("stage2_epochs must be positive.")
        if not 1 <= self.stage2_local_epochs < self.stage2_epochs:
            raise ValueError("stage2_local_epochs must be smaller than stage2_epochs.")
        if self.tissue_epochs < 1:
            raise ValueError("tissue_epochs must be positive.")
        if self.stage2_pooled_feature_dim != UNI2_POOLED_DIM:
            raise ValueError("UNI2 cache contract is CLS+center+ring = 4608 dimensions.")
        if self.biomask_crop_size < 16 or self.biomask_crop_size % 16:
            raise ValueError("BioMask crop size must be positive and divisible by 16.")
        if self.deployment_uni2_micro_batch_size < 1:
            raise ValueError("deployment_uni2_micro_batch_size must be positive.")
        if self.deployment_stage2_inference_batch_size < 1:
            raise ValueError("deployment_stage2_inference_batch_size must be positive.")
        unknown = sorted(set(self.stage2_views) - set(VIEW_SIZES))
        if unknown or not self.stage2_views:
            raise ValueError(f"Unsupported Stage-2 views: {unknown}.")
        if len(set(self.stage2_views)) != len(self.stage2_views):
            raise ValueError("stage2_views must not contain duplicates.")
        for name in (
            "stage2_hidden_dim", "stage2_biology_hidden_dim", "stage2_spatial_hidden_dim",
            "stage2_tissue_hidden_dim", "stage2_prior_hidden_dim",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.number_of_workers < 0:
            raise ValueError("number_of_workers cannot be negative; use 0 for automatic worker selection.")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        if self.graph_neighbor_k < 1:
            raise ValueError("graph_neighbor_k must be positive.")
        if self.graph_radius_cap <= 0.0:
            raise ValueError("graph_radius_cap must be positive.")
        if self.graph_probability_temperature <= 0.0:
            raise ValueError("graph_probability_temperature must be positive.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0,1).")
        if self.stage2_cache_variants < 1:
            raise ValueError("stage2_cache_variants must be positive.")
        for name in (
            "stage2_cache_batch_size", "stage2_uni2_micro_batch_size", "stage2_train_batch_size",
            "stage2_roi_batch_size", "stage2_inference_batch_size", "tissue_batch_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "local_learning_rate", "biomask_learning_rate", "fusion_learning_rate",
            "graph_learning_rate", "validity_learning_rate", "stage2_minimum_learning_rate",
            "tissue_learning_rate", "tissue_minimum_learning_rate", "risk_learning_rate",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.stage2_minimum_learning_rate > min(self.local_learning_rate, self.biomask_learning_rate, self.fusion_learning_rate, self.graph_learning_rate, self.validity_learning_rate):
            raise ValueError("stage2_minimum_learning_rate cannot exceed a Stage-2 peak learning rate.")
        if self.tissue_minimum_learning_rate > self.tissue_learning_rate:
            raise ValueError("tissue_minimum_learning_rate cannot exceed tissue_learning_rate.")
        for name in ("stage2_weight_decay", "tissue_weight_decay", "risk_weight_decay", "mask_loss_weight", "graph_loss_weight", "validity_loss_weight", "tissue_dice_weight", "tissue_ce_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if self.tissue_dice_weight + self.tissue_ce_weight <= 0.0:
            raise ValueError("At least one tissue loss weight must be positive.")
        for name in ("stage1_ema_decay", "stage2_ema_decay", "tissue_ema_decay"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0,1).")
        if self.stage2_warmup_epochs < 0 or self.stage2_warmup_epochs >= self.stage2_epochs:
            raise ValueError("stage2_warmup_epochs must be in [0, stage2_epochs).")
        if self.tissue_warmup_epochs < 0 or self.tissue_warmup_epochs >= self.tissue_epochs:
            raise ValueError("tissue_warmup_epochs must be in [0, tissue_epochs).")
        if self.risk_fit_steps < 1:
            raise ValueError("risk_fit_steps must be positive.")
        if not self.global_threshold_grid:
            raise ValueError("global_threshold_grid cannot be empty.")
        if any((not 0.0 <= value <= 1.0) for value in self.global_threshold_grid):
            raise ValueError("global_threshold_grid values must be in [0,1].")
        if tuple(sorted(set(self.global_threshold_grid))) != self.global_threshold_grid:
            raise ValueError("global_threshold_grid must be sorted and contain unique values.")
        if not 0.0 <= self.context_local_lr_multiplier <= 1.0:
            raise ValueError("context_local_lr_multiplier must be in [0,1].")
        if self.tissue_context_pool_radius < 1:
            raise ValueError("tissue_context_pool_radius must be positive.")
        if self.biomask_base_channels < 1 or self.tissue_base_channels < 1:
            raise ValueError("BioMask/tissue base channel counts must be positive.")
        for name in (
            "stage1_prior_dropout",
            "tissue_context_dropout",
            "biology_context_dropout",
            "graph_probability_dropout",
            "roi_class_balanced_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1).")

    @property
    def root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def path(self, field_name: str) -> Path:
        value = Path(str(getattr(self, field_name))).expanduser()
        return value.resolve() if value.is_absolute() else (self.root / value).resolve()

    @property
    def workers(self) -> int:
        if self.number_of_workers > 0:
            return self.number_of_workers
        count = os.cpu_count() or 4
        return max(2, min(16, count - 2 if count > 4 else count))

    @property
    def stage1_accumulation_steps(self) -> int:
        if self.stage1_effective_batch_size % self.stage1_micro_batch_size:
            raise ValueError("Stage-1 effective batch must be divisible by micro batch.")
        return self.stage1_effective_batch_size // self.stage1_micro_batch_size

    @staticmethod
    def _fingerprint_values(values: dict[str, Any]) -> str:
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def stage1_fingerprint(self) -> str:
        values = asdict(self)
        path_fields = {"stage1_output_dir"}
        selected = {
            key: value
            for key, value in values.items()
            if (key.startswith("stage1_") and key not in path_fields)
            or key in {"seed", "number_of_folds", "fold_assignment_method", "image_size", "match_radius_pixels", "use_bfloat16", "use_tf32", "deterministic"}
        }
        return self._fingerprint_values(selected)

    @property
    def tissue_fingerprint(self) -> str:
        values = asdict(self)
        selected = {
            key: value
            for key, value in values.items()
            if key.startswith("tissue_") and key not in {"tissue_output_dir", "tissue_geojson_dir"}
        }
        for key in ("seed", "number_of_folds", "fold_assignment_method", "image_size", "use_bfloat16", "use_tf32", "deterministic", "stage1_base_channels"):
            selected[key] = values[key]
        return self._fingerprint_values(selected)

    @property
    def fingerprint(self) -> str:
        values = asdict(self)
        for key in (
            "project_root",
            "artifact_dir",
            "number_of_workers",
            "stage1_output_dir",
            "tissue_output_dir",
            "stage2_output_dir",
            "cache_dir",
            "uni2_checkpoint",
            "image_dir",
            "nuclei_geojson_dir",
            "tissue_geojson_dir",
            "deployment_uni2_micro_batch_size",
            "deployment_stage2_inference_batch_size",
        ):
            values.pop(key, None)
        return self._fingerprint_values(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PumaConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
