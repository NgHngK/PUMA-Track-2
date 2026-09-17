from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..constants import (
    BIOLOGY_DIM,
    GEOMETRY_FEATURE_DIM,
    RELIABILITY_DIM,
    ROI_SIZE,
    SPATIAL_DIM,
    STAGE2_ROW_DTYPE,
)
from ..data.preprocess import load_shared_artifacts
from ..stage1.inference import OOF_CHECKPOINT_POLICY, generate_stage1_oof_proposals
from ..utils.provenance import assert_fold_independent, canonical_fingerprint, file_stat_identity, sha256_file
from ..utils.runtime import configure_cpu_parallelism, configure_runtime, dataloader_worker_count, shutdown_dataloader
from .biology_features import extract_biology_features
from .biomask import ProposalBiologyNetwork
from .biomask_data import ProposalBiologyDataset
from .biomask_trainer import _biomask_data_signature, _biomask_training_signature, train_biology_fold
from .targets import build_biomask_targets, resolve_semantic_targets, save_array


BIOMASK_DIAGNOSTIC_DTYPE = np.dtype([
    ("border_touch_pred", "f4"),
    ("prompt_inside_mask_pred", "f4"),
    ("valid_mask_fraction_pred", "f4"),
    ("mask_area_pred", "f4"),
    ("center_agreement_pred", "f4"),
    ("presence_local_support_pred", "f4"),
    ("mask_usability_pred", "f4"),
])


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    temporary.replace(path)


def _feature_dtype(config: PipelineConfig) -> np.dtype:
    return np.dtype(config.stage2_preprocess.feature_dtype)


def _feature_bank_root(config: PipelineConfig) -> Path:
    # Exactly one active Stage-2 feature bank exists. The UNI2 mode is bound into
    # its contract/provenance; switching frozen<->LoRA rebuilds UNI2 arrays in
    # place instead of keeping two parallel banks that complicate Docker export.
    return config.paths.stage2_preprocessed / "feature_bank"








def _selected_preprocess_folds(config: PipelineConfig) -> tuple[int, ...]:
    """Return the validated Stage-2 preprocessing fold subset in notebook order."""
    return tuple(int(fold) for fold in config.stage2_preprocess.selected_folds)


def _select_preprocess_proposals(proposals: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Materialize only proposals belonging to the requested preprocessing folds."""
    selected = _selected_preprocess_folds(config)
    proposal_folds = np.asarray(proposals["fold"], dtype=np.int64)
    keep = np.isin(proposal_folds, np.asarray(selected, dtype=np.int64))
    if not bool(np.any(keep)):
        raise RuntimeError(f"Stage-2 preprocessing found no proposals for selected_folds={list(selected)}")
    if bool(np.all(keep)):
        return proposals
    return np.asarray(proposals[keep]).copy()





def _feature_bank_contract_signature(config: PipelineConfig) -> str:
    """Fingerprint the code/config contract that gives cached Stage-2 arrays their meaning."""
    here = Path(__file__).resolve().parent
    sources = (
        Path(__file__),
        here / "targets.py",
        here / "crops.py",
        here / "biomask.py",
        here / "biomask_data.py",
        here / "biomask_sampling.py",
        here / "biology_features.py",
        Path(__file__).resolve().parents[1] / "constants.py",
    )
    payload = {
        "stage2_preprocess": asdict(config.stage2_preprocess),
        "data_folds": int(config.data.number_of_folds),
        "data_seed": int(config.data.seed),
    }
    return canonical_fingerprint(payload, sources)

def _feature_bank_signature(config: PipelineConfig, stage1_summary: dict[str, Any], checkpoint: Path | None = None) -> str:
    """v16.4 preprocessing signature. UNI2 is trained online, not cached here."""
    here = Path(__file__).resolve().parent
    sources = (
        Path(__file__),
        here / "targets.py",
        here / "crops.py",
        here / "biomask.py",
        here / "biomask_data.py",
        here / "biomask_sampling.py",
        here / "biology_features.py",
        Path(__file__).resolve().parents[1] / "constants.py",
    )
    payload = {
        "stage2_preprocess": asdict(config.stage2_preprocess),
        "data_folds": int(config.data.number_of_folds),
        "data_seed": int(config.data.seed),
        "shared_preprocessing_summary_sha256": sha256_file(
            config.paths.preprocessed / "preprocessing_summary.json"
        ),
        "stage1_sources": stage1_summary.get("fold_operating_points", []),
        "stage1_proposal_sha256": stage1_summary.get("proposal_sha256"),
        "stage1_roi_offsets_sha256": stage1_summary.get("roi_offsets_sha256"),
        "uni2_representation": "online_integrated_stage2_v16.4",
    }
    return canonical_fingerprint(payload, sources)



def _hematoxylin_numpy(rgb_uint8: np.ndarray) -> np.ndarray:
    x = np.clip(rgb_uint8.astype(np.float32) / 255.0, 1.0 / 255.0, 1.0)
    od = -np.log(x)
    return (0.650 * od[..., 0] + 0.704 * od[..., 1] + 0.286 * od[..., 2]) / 2.0


def _roi_stain_stats(images: np.ndarray) -> np.ndarray:
    stats = np.empty((len(images), 2), dtype=np.float32)
    for roi in range(len(images)):
        h = _hematoxylin_numpy(np.asarray(images[roi])[::8, ::8])
        q25, median, q75 = np.quantile(h, (0.25, 0.50, 0.75))
        stats[roi] = (median, max(float(q75 - q25), 1.0e-4))
    return stats


def _biomask_dataset(
    shared: dict[str, np.ndarray],
    proposals: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    config: PipelineConfig,
) -> ProposalBiologyDataset:
    p = config.stage2_preprocess
    return ProposalBiologyDataset(
        shared["images"], proposals, targets, shared["nuclei"], shared["polygon_points"], indices,
        crop_size=p.crop_size,
        prompt_sigma=p.biomask_prompt_sigma_pixels,
        crop_shift_pixels=p.biomask_crop_shift_pixels,
        crop_shift_attempts=p.biomask_crop_shift_attempts,
        positive_visibility_retained_min=p.biomask_positive_visibility_retained_min,
        ambiguous_visibility_retained_min=p.biomask_ambiguous_visibility_retained_min,
        ambiguous_shift_scale=p.biomask_ambiguous_shift_scale,
        d4_probability=p.biomask_d4_probability,
        augment=False,
        seed=config.data.seed,
    )


def _load_biomask_checkpoint(
    path: Path,
    base_channels: int,
    device: torch.device,
    *,
    held_fold: int,
    presence_sigma_pixels: float,
    presence_support_radius_pixels: float,
) -> tuple[ProposalBiologyNetwork, dict[str, Any]]:
    model = ProposalBiologyNetwork(
        base_channels,
        presence_sigma_pixels=presence_sigma_pixels,
        presence_support_radius_pixels=presence_support_radius_pixels,
    ).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    extra = dict(payload.get("extra") or {})
    training_folds = [int(value) for value in extra.get("training_folds", [])]
    if not bool(extra.get("oof_safe", False)) or int(extra.get("fold", -1)) != int(held_fold):
        raise RuntimeError(f"BioMask checkpoint is not OOF-safe for held fold {held_fold}: {path}")
    if int(held_fold) in training_folds:
        raise RuntimeError(f"BioMask provenance leak: held fold {held_fold} appears in checkpoint training folds")
    model.load_state_dict(payload["model"], strict=True)
    stat = path.stat()
    source = {
        "training_folds": training_folds,
        "fold": int(held_fold),
        "checkpoint": str(path),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "training_signature": extra.get("training_signature"),
        "data_signature": extra.get("data_signature"),
    }
    return model.eval(), source


def _legacy_biomask_signature_is_reusable(
    config: PipelineConfig,
    *,
    held_fold: int,
    training_folds: list[int],
    saved_signature: Any,
) -> bool:
    """Return True when an older BioMask checkpoint is safe for inference/cache reuse.

    Completed BioMask predictions must not be invalidated by training-only or
    throughput settings (epochs, batch size, worker count) or by a source-code
    fingerprint changing elsewhere in the pipeline.  For legacy reuse we bind
    only the durable OOF/model/input semantics; proposal/target identity, cache
    shapes/dtypes, and fold provenance are validated separately by the cache
    contract and provenance checks.
    """
    if not isinstance(saved_signature, dict):
        return False
    saved_training_folds = [int(v) for v in saved_signature.get("training_folds", [])]
    if saved_training_folds != list(training_folds):
        return False

    expected_training_folds = [
        fold for fold in range(int(config.data.number_of_folds)) if fold != int(held_fold)
    ]
    if sorted(training_folds) != expected_training_folds:
        return False

    saved_preprocess = saved_signature.get("preprocess")
    if not isinstance(saved_preprocess, dict):
        return False
    current = config.stage2_preprocess
    durable_fields = {
        "crop_size": int(current.crop_size),
        "biomask_base_channels": int(current.biomask_base_channels),
        "biomask_prompt_sigma_pixels": float(current.biomask_prompt_sigma_pixels),
        "biomask_presence_sigma_pixels": float(current.biomask_presence_sigma_pixels),
        "biomask_presence_support_radius_pixels": float(current.biomask_presence_support_radius_pixels),
    }
    for key, expected in durable_fields.items():
        if key not in saved_preprocess or saved_preprocess[key] != expected:
            return False
    return True


def _biomask_checkpoint_signature_is_reusable(
    config: PipelineConfig,
    *,
    held_fold: int,
    training_folds: list[int],
    saved_signature: Any,
) -> bool:
    """Accept the exact current contract or a safely compatible legacy one."""
    if saved_signature == _biomask_training_signature(config, training_folds):
        return True
    return _legacy_biomask_signature_is_reusable(
        config, held_fold=held_fold, training_folds=training_folds, saved_signature=saved_signature
    )


def _biomask_source_from_checkpoint(path: Path, config: PipelineConfig, *, held_fold: int) -> dict[str, Any]:
    """Read/validate BioMask OOF provenance without materializing the CUDA model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra") or {})
    training_folds = [int(value) for value in extra.get("training_folds", [])]
    if not bool(extra.get("oof_safe", False)) or int(extra.get("fold", -1)) != int(held_fold):
        raise RuntimeError(f"BioMask checkpoint is not OOF-safe for held fold {held_fold}: {path}")
    if int(held_fold) in training_folds:
        raise RuntimeError(f"BioMask provenance leak: held fold {held_fold} appears in checkpoint training folds")
    if not _biomask_checkpoint_signature_is_reusable(
        config, held_fold=held_fold, training_folds=training_folds,
        saved_signature=extra.get("training_signature"),
    ):
        raise RuntimeError(f"BioMask checkpoint inference contract is incompatible for held fold {held_fold}: {path}")
    stat = path.stat()
    return {
        "training_folds": training_folds,
        "fold": int(held_fold),
        "checkpoint": str(path),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "training_signature": extra.get("training_signature"),
        "data_signature": extra.get("data_signature"),
    }


def _mask_centroid(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # mask [B,1,H,W]
    batch, _, height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=mask.device, dtype=mask.dtype),
        torch.arange(width, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    weight = mask[:, 0].clamp_min(0.0)
    total = weight.sum(dim=(1, 2)).clamp_min(1.0e-6)
    x = (weight * xx).sum(dim=(1, 2)) / total
    y = (weight * yy).sum(dim=(1, 2)) / total
    return x, y, total


def _predicted_reliability_features(
    raw_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    prompt: torch.Tensor,
    presence: torch.Tensor,
    quality: torch.Tensor,
    center_offset: torch.Tensor,
    presence_local_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    raw = raw_mask.float().clamp(0.0, 1.0)
    valid = valid_mask.float().clamp(0.0, 1.0)
    masked = raw * valid
    raw_mass = raw.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)
    valid_mass = masked.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)
    valid_fraction = (valid_mass / raw_mass).clamp(0.0, 1.0)
    prompt_inside = ((masked * prompt.float()).sum(dim=(1, 2, 3)) / prompt.float().sum(dim=(1, 2, 3)).clamp_min(1.0e-6)).clamp(0.0, 1.0)

    border = torch.zeros_like(masked)
    width = min(3, masked.shape[-1] // 2, masked.shape[-2] // 2)
    if width > 0:
        border[:, :, :width] = 1
        border[:, :, -width:] = 1
        border[:, :, :, :width] = 1
        border[:, :, :, -width:] = 1
    border_touch = ((masked * border).sum(dim=(1, 2, 3)) / valid_mass).clamp(0.0, 1.0)
    mask_x, mask_y, area = _mask_centroid(masked)
    prompt_x, prompt_y, _ = _mask_centroid(prompt.float())
    mask_offset = torch.stack((mask_x - prompt_x, mask_y - prompt_y), dim=1)
    center_disagreement = torch.linalg.vector_norm(center_offset.float() - mask_offset.float(), dim=1)
    center_agreement = torch.exp(-center_disagreement / 5.0).clamp(0.0, 1.0)
    offset_norm = (torch.linalg.vector_norm(center_offset.float(), dim=1) / 15.0).clamp(0.0, 3.0)
    area_plausibility = (
        torch.sigmoid((area - 15.0) / 10.0) * torch.sigmoid((1200.0 - area) / 200.0)
    ).clamp(0.0, 1.0)
    area_log = (torch.log1p(area) / np.log1p(1500.0)).clamp(0.0, 1.5)
    # The last column is filled later with target-token validity after UNI2 weight construction.
    reliability = torch.stack(
        (
            presence.float().clamp(0.0, 1.0),
            quality.float().clamp(0.0, 1.0),
            (presence.float() * quality.float()).clamp(0.0, 1.0),
            prompt_inside,
            offset_norm,
            border_touch,
            valid_fraction,
            area_plausibility,
            center_agreement,
            torch.zeros_like(area_log),
        ),
        dim=1,
    )
    if reliability.shape[1] != RELIABILITY_DIM:
        raise RuntimeError("reliability feature schema mismatch")
    diagnostics = {
        "border_touch": border_touch,
        "prompt_inside": prompt_inside,
        "valid_fraction": valid_fraction,
        "area": area,
        "center_agreement": center_agreement,
        "presence_local_support": presence_local_support.clamp(0.0, 1.0),
        "mask_usability": (presence.float() * quality.float()).clamp(0.0, 1.0),
    }
    return masked, reliability, diagnostics


def _extract_biomask_oof(
    shared: dict[str, np.ndarray],
    proposals: np.ndarray,
    targets: np.ndarray,
    *,
    config: PipelineConfig,
    output_directory: Path,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dtype = _feature_dtype(config)
    count = len(proposals)
    crop_size = int(config.stage2_preprocess.crop_size)
    mask_path = output_directory / "biomask_probability.npy"
    biology_path = output_directory / "biology.npy"
    presence_path = output_directory / "presence_probability.npy"
    quality_path = output_directory / "mask_quality.npy"
    offset_path = output_directory / "center_offset.npy"
    reliability_path = output_directory / "reliability.npy"
    diagnostics_path = output_directory / "biomask_diagnostics.npy"
    completed_path = output_directory / "biomask_complete.npy"
    contract_path = output_directory / "biomask_cache_contract.json"
    contract = _biomask_cache_contract(config, proposals, targets)

    cache_paths = (
        mask_path, biology_path, presence_path, quality_path, offset_path,
        reliability_path, diagnostics_path, completed_path,
    )
    expected_shapes = {
        mask_path: (count, crop_size, crop_size),
        biology_path: (count, BIOLOGY_DIM),
        presence_path: (count,),
        quality_path: (count,),
        offset_path: (count, 2),
        reliability_path: (count, RELIABILITY_DIM),
        diagnostics_path: (count,),
    }
    provenance_path = output_directory / "biomask_provenance.json"
    existing_count = sum(path.is_file() for path in cache_paths)
    rebuild = bool(force)
    resumed_partial = False

    if not rebuild and existing_count == len(cache_paths):
        completed = np.load(completed_path, allow_pickle=False).astype(bool, copy=True)
        if len(completed) != count:
            raise RuntimeError("BioMask partial cache completion vector has the wrong length")
        arrays = {
            mask_path: np.load(mask_path, mmap_mode="r+", allow_pickle=False),
            biology_path: np.load(biology_path, mmap_mode="r+", allow_pickle=False),
            presence_path: np.load(presence_path, mmap_mode="r+", allow_pickle=False),
            quality_path: np.load(quality_path, mmap_mode="r+", allow_pickle=False),
            offset_path: np.load(offset_path, mmap_mode="r+", allow_pickle=False),
            reliability_path: np.load(reliability_path, mmap_mode="r+", allow_pickle=False),
            diagnostics_path: np.load(diagnostics_path, mmap_mode="r+", allow_pickle=False),
        }
        expected_dtypes = {
            mask_path: np.dtype(np.uint8),
            biology_path: dtype,
            presence_path: dtype,
            quality_path: dtype,
            offset_path: dtype,
            reliability_path: dtype,
            diagnostics_path: BIOMASK_DIAGNOSTIC_DTYPE,
        }
        for path, values in arrays.items():
            if values.shape != expected_shapes[path]:
                raise RuntimeError(
                    f"BioMask cache has wrong shape for {path.name}: {values.shape} != {expected_shapes[path]}"
                )
            if values.dtype != expected_dtypes[path]:
                raise RuntimeError(
                    f"BioMask cache has wrong dtype for {path.name}: {values.dtype} != {expected_dtypes[path]}"
                )

        cached_contract = (
            json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else None
        )
        contract_ok = cached_contract == contract
        if not contract_ok and isinstance(cached_contract, dict) and _legacy_biomask_contract_matches(cached_contract, contract):
            _validate_existing_biomask_model_contracts(config, output_directory, np.unique(proposals["fold"]))
            _atomic_write_json(contract_path, contract)
            contract_ok = True
            print("Validated and upgraded legacy BioMask v1 cache contract")
        elif cached_contract is None and np.all(completed):
            # Very old complete caches predate any cache contract. Bind them only
            # after exact target equality and OOF provenance/model validation.
            legacy_targets_path = output_directory / "biomask_targets.npy"
            if not legacy_targets_path.is_file() or not provenance_path.is_file():
                raise RuntimeError("Legacy complete BioMask cache lacks targets/provenance and cannot be adopted safely")
            legacy_targets = np.load(legacy_targets_path, mmap_mode="r", allow_pickle=False)
            if legacy_targets.shape != targets.shape or legacy_targets.dtype != targets.dtype:
                raise RuntimeError("Legacy BioMask targets do not match the current target schema")
            for start in range(0, len(targets), 8192):
                if not np.array_equal(
                    np.asarray(legacy_targets[start : start + 8192]),
                    np.asarray(targets[start : start + 8192]),
                ):
                    raise RuntimeError("Legacy BioMask cache targets differ from current proposals")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            _validate_completed_biomask_cache(
                expected_shapes, expected_feature_dtype=dtype, provenance=provenance,
                required_folds=np.unique(proposals["fold"]),
            )
            _validate_existing_biomask_model_contracts(config, output_directory, np.unique(proposals["fold"]))
            _atomic_write_json(contract_path, contract)
            contract_ok = True
            print("Validated and adopted legacy complete BioMask cache")

        if not contract_ok:
            if np.all(completed):
                if config.stage2_preprocess.require_existing_complete_biomask:
                    raise RuntimeError(
                        "Completed BioMask cache belongs to different proposals/configuration; "
                        "refusing to retrain because require_existing_complete_biomask=True"
                    )
                print("BioMask cache contract changed; rebuilding learned mask features")
                rebuild = True
                del arrays
            else:
                raise RuntimeError(
                    "Partial BioMask cache contract mismatch; use force=True only if a rebuild is intended"
                )
        elif np.all(completed):
            if not provenance_path.is_file():
                raise RuntimeError("Completed BioMask cache is missing biomask_provenance.json")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            _validate_completed_biomask_cache(
                expected_shapes, expected_feature_dtype=dtype, provenance=provenance,
                required_folds=np.unique(proposals["fold"]),
            )
            print(f"Reusing complete OOF BioMask features ({count:,}/{count:,}); training is skipped")
            return (
                arrays[mask_path], arrays[biology_path], arrays[presence_path], arrays[quality_path],
                arrays[offset_path], arrays[reliability_path], provenance,
            )
        else:
            masks = arrays[mask_path]
            biology = arrays[biology_path]
            presence = arrays[presence_path]
            quality = arrays[quality_path]
            center_offset = arrays[offset_path]
            reliability = arrays[reliability_path]
            diagnostics = arrays[diagnostics_path]
            resumed_partial = True
            print(f"Resuming BioMask OOF extraction: {int(completed.sum()):,}/{count:,} completed")
    elif not rebuild and existing_count:
        raise RuntimeError("Incomplete set of BioMask cache files; use force=True to rebuild safely")
    else:
        rebuild = True

    if config.stage2_preprocess.require_existing_complete_biomask:
        completed_count = int(np.sum(completed)) if "completed" in locals() else 0
        raise RuntimeError(
            "require_existing_complete_biomask=True, but the BioMask cache is missing, stale, or incomplete "
            f"({completed_count:,}/{count:,}). Refusing to retrain BioMask."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("BioMask training/extraction requires CUDA")
    if rebuild:
        masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(count, crop_size, crop_size))
        biology = np.lib.format.open_memmap(biology_path, mode="w+", dtype=dtype, shape=(count, BIOLOGY_DIM))
        presence = np.lib.format.open_memmap(presence_path, mode="w+", dtype=dtype, shape=(count,))
        quality = np.lib.format.open_memmap(quality_path, mode="w+", dtype=dtype, shape=(count,))
        center_offset = np.lib.format.open_memmap(offset_path, mode="w+", dtype=dtype, shape=(count, 2))
        reliability = np.lib.format.open_memmap(reliability_path, mode="w+", dtype=dtype, shape=(count, RELIABILITY_DIM))
        diagnostics = np.lib.format.open_memmap(
            diagnostics_path, mode="w+", dtype=BIOMASK_DIAGNOSTIC_DTYPE, shape=(count,)
        )
        completed = np.zeros(count, dtype=bool)
        _atomic_save(completed_path, completed)
        _atomic_write_json(contract_path, contract)

    stain_path = output_directory / "roi_stain_stats.npy"
    stain_contract_path = output_directory / "roi_stain_stats_contract.json"
    shared_summary_path = config.paths.preprocessed / "preprocessing_summary.json"
    stain_contract = {
        "schema": "roi-stain-stats-v2",
        "image_identity": file_stat_identity(config.paths.preprocessed / "images.npy"),
        "shared_preprocessing_summary_sha256": (
            sha256_file(shared_summary_path) if shared_summary_path.is_file() else None
        ),
        "roi_count": int(len(shared["images"])),
    }
    reuse_stain = False
    if not force and stain_path.is_file() and stain_contract_path.is_file():
        try:
            previous_stain_contract = json.loads(stain_contract_path.read_text(encoding="utf-8"))
            candidate = np.load(stain_path, allow_pickle=False)
            reuse_stain = (
                previous_stain_contract == stain_contract
                and candidate.shape == (len(shared["images"]), 2, 3)
                and np.isfinite(candidate).all()
            )
            if reuse_stain:
                stain_stats = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            reuse_stain = False
    if not reuse_stain:
        stain_stats = _roi_stain_stats(shared["images"])
        _atomic_save(stain_path, stain_stats)
        _atomic_write_json(stain_contract_path, stain_contract)

    # Hash the BioMask training rows once for all OOF folds. Without this, each
    # fold restart rereads and hashes the same potentially large proposals/targets.
    # The resulting checkpoint provenance is identical; this is I/O-only speedup.
    biomask_data_signature = _biomask_data_signature(config, shared, proposals, targets)

    runtime = configure_runtime(
        config.data.seed + 422,
        use_bfloat16=config.runtime.use_bfloat16,
        allow_tf32=config.runtime.allow_tf32,
        deterministic=config.runtime.deterministic,
    )
    fold_sources: list[dict[str, Any]] = []
    stain_med_gpu = torch.as_tensor(stain_stats[:, 0], device=runtime.device)
    stain_iqr_gpu = torch.as_tensor(stain_stats[:, 1], device=runtime.device)
    diagnostic_source = {
        "border_touch_pred": "border_touch",
        "prompt_inside_mask_pred": "prompt_inside",
        "valid_mask_fraction_pred": "valid_fraction",
        "mask_area_pred": "area",
        "center_agreement_pred": "center_agreement",
        "presence_local_support_pred": "presence_local_support",
        "mask_usability_pred": "mask_usability",
    }
    checkpoint_every = max(2048, int(config.stage2_preprocess.biomask_batch_size) * 32)
    processed_since_commit = 0

    def commit_progress() -> None:
        for memmap in (masks, biology, presence, quality, center_offset, reliability, diagnostics):
            memmap.flush()
        _atomic_save(completed_path, completed)

    for fold in range(config.data.number_of_folds):
        held_indices = np.flatnonzero(proposals["fold"] == fold)
        if not len(held_indices):
            continue
        pending_indices = held_indices[~completed[held_indices]]
        fold_checkpoint = output_directory / "biomask_models" / f"fold_{fold}" / "checkpoints" / "final.pt"

        if not len(pending_indices):
            if not fold_checkpoint.is_file():
                raise RuntimeError(
                    f"BioMask rows for fold {fold} are marked complete but final checkpoint is missing: {fold_checkpoint}"
                )
            source = _biomask_source_from_checkpoint(fold_checkpoint, config, held_fold=fold)
            source["checkpoint_sha256"] = sha256_file(fold_checkpoint)
            fold_sources.append(source)
            continue

        checkpoint = train_biology_fold(
            shared, proposals, targets, fold=fold, config=config,
            output_directory=output_directory / "biomask_models", force=force,
            data_signature=biomask_data_signature,
        )
        model, source = _load_biomask_checkpoint(
            checkpoint, config.stage2_preprocess.biomask_base_channels, runtime.device, held_fold=fold,
            presence_sigma_pixels=config.stage2_preprocess.biomask_presence_sigma_pixels,
            presence_support_radius_pixels=config.stage2_preprocess.biomask_presence_support_radius_pixels,
        )
        source["checkpoint_sha256"] = sha256_file(checkpoint)
        fold_sources.append(source)
        _atomic_write_json(
            provenance_path,
            {"fold_sources": fold_sources, "external_encoder": "UNI2-h frozen", "partial": True},
        )

        dataset = _biomask_dataset(shared, proposals, targets, pending_indices, config)
        workers = min(8, dataloader_worker_count(config.stage2_preprocess.cpu_crop_workers))
        loader_kwargs: dict[str, Any] = {
            "batch_size": config.stage2_preprocess.biomask_batch_size * 2,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": bool(config.runtime.pin_memory and torch.cuda.is_available()),
        }
        if workers > 0:
            loader_kwargs.update(persistent_workers=config.runtime.persistent_workers, prefetch_factor=2)
        loader = DataLoader(dataset, **loader_kwargs)
        try:
            with torch.inference_mode():
                for batch in loader:
                    x = batch["input"].to(runtime.device, non_blocking=True)
                    valid = batch["valid_mask"].to(runtime.device, non_blocking=True)
                    proposal_indices = batch["proposal_index"].detach().cpu().numpy().astype(np.int64)
                    with torch.autocast(
                        device_type=runtime.device.type, dtype=runtime.amp_dtype, enabled=runtime.amp_enabled
                    ):
                        outputs = model(x, valid_mask=valid)
                    raw_mask = outputs["mask_logits"].sigmoid().float()
                    p_presence = outputs["presence_logits"].sigmoid().float()
                    p_quality = outputs["quality_logits"].sigmoid().float()
                    masked, rel, diag = _predicted_reliability_features(
                        raw_mask, valid, x[:, 3:4], p_presence, p_quality, outputs["center_offset"].float(),
                        outputs["presence_local_support"].float(),
                    )
                    roi_indices = torch.as_tensor(
                        proposals["roi_index"][proposal_indices].astype(np.int64), device=runtime.device
                    )
                    features = extract_biology_features(
                        masked, outputs["hematoxylin"], outputs["hematoxylin_gradient"], p_quality,
                        stain_med_gpu[roi_indices], stain_iqr_gpu[roi_indices],
                    )
                    masks[proposal_indices] = (
                        torch.round(masked[:, 0].clamp(0, 1) * 255.0).byte().cpu().numpy()
                    )
                    # Transfer all small floating outputs to host in one CUDA sync.
                    # Previously every field called .cpu().numpy() independently.
                    diagnostic_tensor = torch.stack(
                        [diag[source_key].float() for source_key in diagnostic_source.values()], dim=1
                    )
                    host_values = torch.cat(
                        (
                            features.float(),
                            p_presence[:, None],
                            p_quality[:, None],
                            outputs["center_offset"].float(),
                            rel.float(),
                            diagnostic_tensor,
                        ),
                        dim=1,
                    ).cpu().numpy()
                    cursor = 0
                    biology[proposal_indices] = host_values[:, cursor : cursor + BIOLOGY_DIM].astype(dtype)
                    cursor += BIOLOGY_DIM
                    presence[proposal_indices] = host_values[:, cursor].astype(dtype)
                    cursor += 1
                    quality[proposal_indices] = host_values[:, cursor].astype(dtype)
                    cursor += 1
                    center_offset[proposal_indices] = host_values[:, cursor : cursor + 2].astype(dtype)
                    cursor += 2
                    reliability[proposal_indices] = host_values[:, cursor : cursor + RELIABILITY_DIM].astype(dtype)
                    cursor += RELIABILITY_DIM
                    for diagnostic_column, key in enumerate(diagnostic_source):
                        diagnostics[key][proposal_indices] = host_values[:, cursor + diagnostic_column].astype(np.float32)
                    completed[proposal_indices] = True
                    processed_since_commit += len(proposal_indices)
                    if processed_since_commit >= checkpoint_every:
                        commit_progress()
                        processed_since_commit = 0
        except BaseException:
            commit_progress()
            raise
        finally:
            shutdown_dataloader(loader)
            del loader, model
            torch.cuda.empty_cache()

        commit_progress()
        processed_since_commit = 0
        _atomic_write_json(
            provenance_path,
            {"fold_sources": fold_sources, "external_encoder": "UNI2-h frozen", "partial": not bool(np.all(completed))},
        )

    if not np.all(completed):
        raise RuntimeError("BioMask OOF extraction did not cover all proposals")
    provenance = {"fold_sources": fold_sources, "external_encoder": "UNI2-h frozen"}
    _atomic_write_json(output_directory / "biomask_provenance.json", provenance)
    _atomic_write_json(contract_path, contract)
    return masks, biology, presence, quality, center_offset, reliability, provenance












def _identity_token_maps_vectorized(
    masks: np.ndarray,
    images: np.ndarray,
    proposals: np.ndarray,
    indices: np.ndarray,
    *,
    biomask_crop_size: int,
    identity_size: int,
    resize_batch_size: int,
) -> np.ndarray:
    if biomask_crop_size < identity_size or (biomask_crop_size - identity_size) % 2:
        raise ValueError("BioMask and identity crops must be nested and center-aligned")
    offset = (biomask_crop_size - identity_size) // 2
    mask64 = np.asarray(
        masks[indices, offset : offset + identity_size, offset : offset + identity_size],
        dtype=np.float32,
    ) / 255.0

    heights = np.full(len(indices), int(images.shape[1]), dtype=np.int64)
    widths = np.full(len(indices), int(images.shape[2]), dtype=np.int64)
    centers_x = np.rint(proposals["x"][indices]).astype(np.int64)
    centers_y = np.rint(proposals["y"][indices]).astype(np.int64)
    coordinates = np.arange(identity_size, dtype=np.int64)
    absolute_x = centers_x[:, None] - identity_size // 2 + coordinates[None]
    absolute_y = centers_y[:, None] - identity_size // 2 + coordinates[None]
    valid = (
        (absolute_y[:, :, None] >= 0)
        & (absolute_y[:, :, None] < heights[:, None, None])
        & (absolute_x[:, None, :] >= 0)
        & (absolute_x[:, None, :] < widths[:, None, None])
    ).astype(np.float32)

    target = np.clip(mask64, 0.0, 1.0) * valid
    binary = target >= 0.5
    structure = np.zeros((1, 3, 3), dtype=bool)
    structure[0, 1, :] = True
    structure[0, :, 1] = True
    dilated1 = binary_dilation(binary, structure=structure, iterations=1)
    eroded1 = binary_erosion(binary, structure=structure, iterations=1)
    boundary = (dilated1 ^ eroded1).astype(np.float32) * valid
    outer = binary_dilation(binary, structure=structure, iterations=6)
    inner = binary_dilation(binary, structure=structure, iterations=2)
    ring = (outer & ~inner).astype(np.float32) * valid

    yy, xx = np.indices((identity_size, identity_size), dtype=np.float32)
    totals = target.sum(axis=(1, 2))
    safe_totals = np.maximum(totals, 1.0e-6)
    centroid_x = (target * xx[None]).sum(axis=(1, 2)) / safe_totals
    centroid_y = (target * yy[None]).sum(axis=(1, 2)) / safe_totals
    centroid = np.exp(
        -(
            (xx[None] - centroid_x[:, None, None]) ** 2
            + (yy[None] - centroid_y[:, None, None]) ** 2
        )
        / (2.0 * 2.5**2)
    ).astype(np.float32)
    centroid *= (totals > 1.0e-6)[:, None, None]
    centroid *= valid

    native = np.stack((target, boundary, ring, centroid), axis=1)
    output = np.empty((len(indices), 4, 16, 16), dtype=np.float32)
    resize_batch_size = max(1, int(resize_batch_size))
    for start in range(0, len(native), resize_batch_size):
        stop = min(len(native), start + resize_batch_size)
        tensor = torch.from_numpy(native[start:stop])
        resized = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        output[start:stop] = F.avg_pool2d(resized.clamp_(0.0, 1.0), kernel_size=14, stride=14).numpy()
    return output


def _proposal_digest(proposals: np.ndarray) -> str:
    values = np.ascontiguousarray(proposals)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _biomask_cache_contract(
    config: PipelineConfig,
    proposals: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    p = config.stage2_preprocess
    biomask_settings = {
        key: value
        for key, value in asdict(p).items()
        if key.startswith("biomask_")
    }
    payload = {
        "schema": "oof-biomask-v2",
        "proposal_count": int(len(proposals)),
        "proposal_sha256": _proposal_digest(proposals),
        "target_sha256": _proposal_digest(targets),
        "number_of_folds": int(config.data.number_of_folds),
        "crop_size": int(p.crop_size),
        "feature_dtype": str(p.feature_dtype),
        "biology_dim": int(BIOLOGY_DIM),
        "reliability_dim": int(RELIABILITY_DIM),
        "base_channels": int(p.biomask_base_channels),
        "presence_sigma_pixels": float(p.biomask_presence_sigma_pixels),
        "presence_support_radius_pixels": float(p.biomask_presence_support_radius_pixels),
        "biomask_settings": biomask_settings,
        "runtime": {
            "use_bfloat16": bool(config.runtime.use_bfloat16),
            "allow_tf32": bool(config.runtime.allow_tf32),
            "deterministic": bool(config.runtime.deterministic),
        },
    }
    here = Path(__file__).resolve().parent
    payload["contract_fingerprint"] = canonical_fingerprint(
        payload,
        (
            here / "biomask.py",
            here / "biomask_data.py",
            here / "biomask_sampling.py",
            here / "biomask_trainer.py",
            here / "biology_features.py",
            here / "targets.py",
            here / "crops.py",
        ),
    )
    return payload


def _legacy_biomask_contract_matches(cached: dict[str, Any], current: dict[str, Any]) -> bool:
    """Permit safe one-time adoption of a completed/partial v1 cache.

    v1 already bound the exact proposal and target bytes. The missing durable
    OOF/model/inference settings are verified against each fold checkpoint
    before migration; training-only throughput settings are intentionally ignored.
    """
    if cached.get("schema") != "oof-biomask-v1":
        return False
    keys = (
        "proposal_count",
        "proposal_sha256",
        "target_sha256",
        "number_of_folds",
        "crop_size",
        "feature_dtype",
        "biology_dim",
        "reliability_dim",
        "base_channels",
        "presence_sigma_pixels",
        "presence_support_radius_pixels",
    )
    return all(cached.get(key) == current.get(key) for key in keys)


def _validate_existing_biomask_model_contracts(
    config: PipelineConfig,
    output_directory: Path,
    required_folds: np.ndarray,
) -> None:
    """Validate old BioMask checkpoints for safe completed-cache reuse.

    This deliberately does not require equality of the full historical training
    signature.  Epoch count, batch size, CPU workers and code fingerprints do
    not change the meaning of already-generated predictions.
    """
    for fold_value in np.asarray(required_folds, dtype=np.int64):
        fold = int(fold_value)
        final_path = output_directory / "biomask_models" / f"fold_{fold}" / "checkpoints" / "final.pt"
        latest_path = output_directory / "biomask_models" / f"fold_{fold}" / "checkpoints" / "latest.pt"
        checkpoint = final_path if final_path.is_file() else latest_path
        if not checkpoint.is_file():
            raise RuntimeError(f"Legacy BioMask cache is missing checkpoint for fold {fold}: {final_path}")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extra = dict(payload.get("extra") or {})
        training_folds = [int(v) for v in extra.get("training_folds", [])]
        if not bool(extra.get("oof_safe", False)) or int(extra.get("fold", -1)) != fold:
            raise RuntimeError(f"Legacy BioMask checkpoint is not OOF-safe for fold {fold}: {checkpoint}")
        if fold in training_folds:
            raise RuntimeError(f"Legacy BioMask checkpoint leaks held fold {fold}: {checkpoint}")
        if not _biomask_checkpoint_signature_is_reusable(
            config, held_fold=fold, training_folds=training_folds,
            saved_signature=extra.get("training_signature"),
        ):
            raise RuntimeError(
                f"Legacy BioMask cache cannot be adopted because fold {fold} inference/model settings differ"
            )

        # State-dict compatibility is the strongest architecture check and also
        # catches accidental base-channel/model-head changes. Presence geometry
        # is checked above because those values are not stored in the state dict.
        probe = ProposalBiologyNetwork(
            int(config.stage2_preprocess.biomask_base_channels),
            presence_sigma_pixels=float(config.stage2_preprocess.biomask_presence_sigma_pixels),
            presence_support_radius_pixels=float(config.stage2_preprocess.biomask_presence_support_radius_pixels),
        )
        try:
            probe.load_state_dict(payload["model"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"Legacy BioMask cache cannot be adopted because fold {fold} checkpoint architecture differs"
            ) from exc
        del probe, payload


def _validate_completed_biomask_cache(
    expected_shapes: dict[Path, tuple[int, ...]],
    *,
    expected_feature_dtype: np.dtype,
    provenance: dict[str, Any],
    required_folds: np.ndarray,
) -> None:
    expected_dtypes: dict[str, np.dtype] = {
        "biomask_probability.npy": np.dtype(np.uint8),
        "biology.npy": expected_feature_dtype,
        "presence_probability.npy": expected_feature_dtype,
        "mask_quality.npy": expected_feature_dtype,
        "center_offset.npy": expected_feature_dtype,
        "reliability.npy": expected_feature_dtype,
        "biomask_diagnostics.npy": BIOMASK_DIAGNOSTIC_DTYPE,
    }
    for path, expected_shape in expected_shapes.items():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != expected_shape:
            raise RuntimeError(f"BioMask cache shape mismatch for {path.name}")
        expected_dtype = expected_dtypes.get(path.name)
        if expected_dtype is not None and values.dtype != expected_dtype:
            raise RuntimeError(
                f"BioMask cache dtype mismatch for {path.name}: {values.dtype} != {expected_dtype}"
            )
        if values.dtype.fields is None and values.dtype.kind == "f":
            # These arrays are small enough to check in bounded row blocks. A
            # full finite scan prevents silent reuse of a truncated/corrupt run.
            for start in range(0, len(values), 8192):
                if not np.isfinite(np.asarray(values[start : start + 8192])).all():
                    raise RuntimeError(f"BioMask cache contains NaN/Inf in {path.name}")

    fold_sources = provenance.get("fold_sources", [])
    by_fold = {int(item.get("fold", -1)): item for item in fold_sources}
    for fold in np.asarray(required_folds, dtype=np.int64):
        fold = int(fold)
        if fold not in by_fold:
            raise RuntimeError(f"BioMask provenance is missing fold {fold}")
        assert_fold_independent(sample_fold=fold, source=by_fold[fold], source_name="BioMask")













def _build_spatial_features(proposals: np.ndarray, roi_count: int, dtype: np.dtype) -> np.ndarray:
    output = np.zeros((len(proposals), SPATIAL_DIM), dtype=np.float32)
    for roi in range(roi_count):
        idx = np.flatnonzero(proposals["roi_index"] == roi)
        if not len(idx):
            continue
        points = np.column_stack((proposals["x"][idx], proposals["y"][idx])).astype(np.float64)
        scores = proposals["heatmap_score"][idx].astype(np.float64)
        if len(idx) == 1:
            output[idx, 0:3] = np.log1p(ROI_SIZE)
            continue
        tree = cKDTree(points)
        k = min(6, len(idx))
        distances, neighbors = tree.query(points, k=k)
        if k == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        other_d = distances[:, 1:]
        output[idx, 0] = np.log1p(other_d[:, 0] if other_d.shape[1] else ROI_SIZE)
        output[idx, 1] = np.log1p(np.mean(other_d[:, : min(3, other_d.shape[1])], axis=1))
        output[idx, 2] = np.log1p(np.mean(other_d[:, : min(5, other_d.shape[1])], axis=1))
        for offset, radius in enumerate((16.0, 32.0, 64.0)):
            lists = tree.query_ball_point(points, radius)
            counts = np.asarray([max(0, len(values) - 1) for values in lists], dtype=np.float32)
            weighted = np.asarray([
                max(0.0, float(scores[np.asarray(values, dtype=np.int64)].sum() - scores[row]))
                for row, values in enumerate(lists)
            ], dtype=np.float32)
            output[idx, 3 + offset] = np.log1p(counts)
            output[idx, 6 + offset] = np.log1p(weighted)
        nearest_idx = neighbors[:, 1] if neighbors.shape[1] > 1 else np.zeros(len(idx), dtype=np.int64)
        output[idx, 9] = (scores - scores[nearest_idx]).astype(np.float32)
        output[idx, 10] = np.log1p(np.std(other_d[:, : min(5, other_d.shape[1])], axis=1))
        anisotropy = np.zeros(len(idx), dtype=np.float32)
        for row, point in enumerate(points):
            local = np.asarray(tree.query_ball_point(point, 64.0), dtype=np.int64)
            local = local[local != row]
            if len(local) >= 3:
                centered = points[local] - point
                covariance = centered.T @ centered / max(len(centered), 1)
                eigenvalues = np.linalg.eigvalsh(covariance)
                anisotropy[row] = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues[-1] + eigenvalues[0], 1.0e-6))
        output[idx, 11] = anisotropy
    return output.astype(dtype)


def _build_geometry_features(
    proposals: np.ndarray,
    presence: np.ndarray,
    quality: np.ndarray,
    center_offset: np.ndarray,
    diagnostics: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    output = np.zeros((len(proposals), GEOMETRY_FEATURE_DIM), dtype=np.float32)
    x = proposals["x"].astype(np.float64)
    y = proposals["y"].astype(np.float64)
    border = np.minimum.reduce((x, y, float(ROI_SIZE - 1) - x, float(ROI_SIZE - 1) - y))
    output[:, 0] = proposals["heatmap_score"]
    output[:, 1] = proposals["quality"]
    output[:, 2] = np.log1p(np.maximum(proposals["uncertainty"], 0))
    output[:, 3] = proposals["peak_sharpness"]
    output[:, 4] = np.clip(border / 32.0, 0.0, 1.0)
    output[:, 5] = np.asarray(presence, dtype=np.float32)
    # Quality is conditional on a target existing; empties do not supervise the
    # quality head, so geometry viability uses presence-conditioned usability.
    output[:, 6] = np.asarray(presence, dtype=np.float32) * np.asarray(quality, dtype=np.float32)
    output[:, 7] = np.linalg.norm(np.asarray(center_offset, dtype=np.float32), axis=1) / 15.0
    output[:, 8] = diagnostics["border_touch_pred"]
    output[:, 9] = diagnostics["valid_mask_fraction_pred"]
    output[:, 10] = diagnostics["prompt_inside_mask_pred"]
    return output.astype(dtype)


def _build_rows(
    proposals: np.ndarray,
    biomask_targets: np.ndarray,
    semantic_targets: np.ndarray,
    presence: np.ndarray,
    quality: np.ndarray,
    center_offset: np.ndarray,
    biomask_diagnostics: np.ndarray,
) -> np.ndarray:
    rows = np.empty(len(proposals), dtype=STAGE2_ROW_DTYPE)
    for i, proposal in enumerate(proposals):
        semantic = semantic_targets[i]
        target = biomask_targets[i]
        diag = biomask_diagnostics[i]
        rows[i] = (
            str(proposal["proposal_uid"]), int(proposal["roi_index"]), float(proposal["x"]), float(proposal["y"]), int(proposal["fold"]),
            int(semantic["semantic_class_target"]), float(semantic["semantic_weight_target"]), int(semantic["semantic_mode_target"]),
            int(target["nucleus_presence_target"]), int(target["geometry_viability_target"]),
            int(semantic["resolved_gt_index_meta"]), int(semantic["resolved_group_meta"]), int(target["negative_stratum_target"]),
            float(proposal["heatmap_score"]), float(proposal["quality"]), float(proposal["uncertainty"]), float(proposal["peak_sharpness"]),
            float(presence[i]), float(quality[i]), float(center_offset[i, 0]), float(center_offset[i, 1]),
            float(diag["border_touch_pred"]), float(diag["prompt_inside_mask_pred"]), float(diag["valid_mask_fraction_pred"]),
            float(diag["mask_area_pred"]), float(diag["center_agreement_pred"]),
        )
    return rows


def _assert_provenance(rows: np.ndarray, stage1_summary: dict[str, Any], biomask_provenance: dict[str, Any], number_of_folds: int) -> None:
    stage1_sources = {int(item["fold"]): item for item in stage1_summary.get("fold_operating_points", [])}
    biomask_sources = {int(item["fold"]): item for item in biomask_provenance.get("fold_sources", [])}
    for fold in range(int(number_of_folds)):
        if not np.any(rows["fold"] == fold):
            continue
        if fold not in stage1_sources or fold not in biomask_sources:
            raise RuntimeError(f"missing learned-feature provenance for fold {fold}")
        assert_fold_independent(sample_fold=fold, source=stage1_sources[fold], source_name="Stage-1")
        assert_fold_independent(sample_fold=fold, source=biomask_sources[fold], source_name="BioMask")
        if stage1_sources[fold].get("decoder_label_dependency_folds") not in ([], None):
            raise RuntimeError(f"Stage-1 OOF decoder has held-label dependencies for fold {fold}")


def _build_identity_token_map_cache(
    images: np.ndarray,
    proposals: np.ndarray,
    masks: np.ndarray,
    reliability: np.ndarray,
    *,
    config: PipelineConfig,
    output_directory: Path,
) -> Path:
    """Precompute only BioMask->UNI2 patch weights; never run UNI2 here.

    The last reliability column is exactly the target-pool validity that v16.3.2
    filled during offline UNI2 extraction.  Moving this calculation here keeps
    the EvidenceFusion input contract unchanged while allowing gradients through
    live UNI2 tokens in v16.4.
    """
    path = output_directory / "identity_token_maps.npy"
    count = len(proposals)
    dtype = np.float16 if config.stage2_preprocess.feature_dtype == "float16" else np.float32
    token_maps = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count, 4, 16, 16))
    chunk = max(256, int(config.stage2_preprocess.uni2_chunk_size))
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        indices = np.arange(start, stop, dtype=np.int64)
        values = _identity_token_maps_vectorized(
            masks,
            images,
            proposals,
            indices,
            biomask_crop_size=config.stage2_preprocess.crop_size,
            identity_size=config.stage2_preprocess.identity_view_size,
            resize_batch_size=config.stage2_preprocess.uni2_token_map_batch_size,
        )
        if not np.isfinite(values).all():
            raise RuntimeError("identity_token_maps contain NaN/Inf")
        token_maps[start:stop] = values.astype(dtype, copy=False)
        target_support = values[:, 0].reshape(len(values), -1).sum(axis=1)
        reliability[start:stop, -1] = (target_support >= 1.0e-4).astype(reliability.dtype, copy=False)
    token_maps.flush()
    if hasattr(reliability, "flush"):
        reliability.flush()
    return path


def preprocess_stage2_features(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    """Build the v16.4 Stage-2 bank without any offline UNI2 inference.

    Learned UNI2 representations are intentionally absent.  The bank contains
    only fixed/OOF proposal evidence and BioMask token maps so Stage-2 semantic
    loss can backpropagate through UNI2 LoRA at training time.
    """
    config.validate()
    stage1_summary = generate_stage1_oof_proposals(config, force=force)
    stage1_root = config.paths.stage2_preprocessed / "stage1_oof"
    proposals = np.load(stage1_root / "proposals.npy", mmap_mode="r", allow_pickle=False)
    if len(proposals) == 0:
        raise RuntimeError("Stage-1 produced zero OOF proposals")
    proposals = _select_preprocess_proposals(proposals, config)
    selected_folds = _selected_preprocess_folds(config)
    print(
        f"Stage-2 preprocessing selected folds: {list(selected_folds)} "
        f"({len(proposals):,} OOF proposals)"
    )

    output = _feature_bank_root(config)
    output.mkdir(parents=True, exist_ok=True)
    # v16.4 no longer consumes cached UNI2 embeddings. Remove stale learned
    # representations so an old feature bank can never be mixed into training.
    for name in (
        "appearance.npy", "identity_pools.npy", "appearance_complete.npy",
        "uni2_cache_contract.json", "uni2_runtime_summary.json",
    ):
        (output / name).unlink(missing_ok=True)
    summary_name = "summary.json"
    provenance_name = "provenance.json"
    summary_path = output / summary_name
    required = (
        "rows.npy", "proposals.npy", "identity_token_maps.npy", "biology.npy",
        "reliability.npy", "spatial.npy", "geometry.npy", "biomask_targets.npy",
        "semantic_targets.npy", "resolver_diagnostics.npy", "biomask_probability.npy", provenance_name,
    )
    bank_signature = _feature_bank_signature(config, stage1_summary)
    if not force and summary_path.is_file() and all((output / name).is_file() for name in required):
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_identities = cached_summary.get("artifact_identity")
        artifacts_current = (
            isinstance(cached_identities, dict)
            and bool(cached_identities)
            and all(
                (output / str(name)).is_file()
                and cached_identities.get(name) == file_stat_identity(output / str(name))
                for name in cached_identities
            )
        )
        if (
            cached_summary.get("feature_bank_signature") == bank_signature
            and cached_summary.get("stage1_fold_operating_points") == stage1_summary.get("fold_operating_points")
            and artifacts_current
        ):
            print(f"Reusing Stage-2 v16.4 preprocessing bank: {output}")
            return cached_summary
        print("Stage-2 feature-bank contract changed; rebuilding v16.4 fixed inputs.")

    shared = load_shared_artifacts(config)
    biomask_targets, target_summary = build_biomask_targets(
        np.asarray(proposals), np.asarray(shared["nuclei"]), np.asarray(shared["polygon_points"]),
        number_of_rois=len(shared["manifest"]),
        semantic_radius=config.stage2_preprocess.association_radius_pixels,
        biological_radius=config.stage2_preprocess.biological_supervision_radius_pixels,
        instance_margin=config.stage2_preprocess.biomask_instance_margin_pixels,
        boundary_ramp_pixels=config.stage2_preprocess.biomask_boundary_ramp_pixels,
        boundary_min_weight=config.stage2_preprocess.biomask_boundary_min_weight,
    )
    masks, biology, presence, quality, center_offset, reliability, biomask_provenance = _extract_biomask_oof(
        shared, np.asarray(proposals), biomask_targets, config=config, output_directory=output, force=force
    )
    save_array(output / "biomask_targets.npy", biomask_targets)
    _atomic_write_json(output / "biomask_target_summary.json", asdict(target_summary))

    semantic_targets, resolver_diagnostics, resolver_summary = resolve_semantic_targets(
        np.asarray(proposals), np.asarray(shared["nuclei"]), np.asarray(shared["polygon_points"]),
        biomask_targets, masks,
        number_of_folds=config.data.number_of_folds,
        crop_size=config.stage2_preprocess.crop_size,
        semantic_radius=config.stage2_preprocess.association_radius_pixels,
        geometry_weight=config.stage2_preprocess.resolver_geometry_weight,
        quality_threshold=config.stage2_preprocess.resolver_quality_threshold,
        margin_threshold=config.stage2_preprocess.resolver_margin_threshold,
    )
    save_array(output / "semantic_targets.npy", semantic_targets)
    save_array(output / "resolver_diagnostics.npy", resolver_diagnostics)
    _atomic_write_json(output / "resolver_summary.json", asdict(resolver_summary))

    identity_token_maps_path = _build_identity_token_map_cache(
        shared["images"], np.asarray(proposals), masks, reliability,
        config=config, output_directory=output,
    )
    dtype = _feature_dtype(config)
    spatial = _build_spatial_features(np.asarray(proposals), len(shared["manifest"]), dtype)
    biomask_diagnostics = np.load(output / "biomask_diagnostics.npy", mmap_mode="r", allow_pickle=False)
    geometry = _build_geometry_features(np.asarray(proposals), presence, quality, center_offset, biomask_diagnostics, dtype)
    rows = _build_rows(
        np.asarray(proposals), biomask_targets, semantic_targets, presence, quality, center_offset, biomask_diagnostics
    )
    _assert_provenance(rows, stage1_summary, biomask_provenance, config.data.number_of_folds)

    _atomic_save(output / "rows.npy", rows)
    _atomic_save(output / "proposals.npy", np.asarray(proposals))
    _atomic_save(output / "spatial.npy", spatial)
    _atomic_save(output / "geometry.npy", geometry)

    provenance = {
        "stage1": {
            "checkpoint_policy": stage1_summary.get("stage1_checkpoint_policy"),
            "fold_sources": stage1_summary.get("fold_operating_points", []),
        },
        "biomask": biomask_provenance,
        "uni2": {
            "source": "online_integrated_stage2_v16.4",
            "cached_embeddings": False,
            "token_maps": identity_token_maps_path.name,
            "adapted_view_sizes": [64, 128, 256],
        },
    }
    _atomic_write_json(output / provenance_name, provenance)

    semantic_mask = rows["semantic_weight_target"] > 0
    artifact_names = (
        "rows.npy", "proposals.npy", identity_token_maps_path.name,
        "biology.npy", "reliability.npy", "spatial.npy", "geometry.npy",
        "biomask_targets.npy", "semantic_targets.npy", "resolver_diagnostics.npy",
        "biomask_probability.npy", provenance_name,
    )
    token_maps = np.load(identity_token_maps_path, mmap_mode="r", allow_pickle=False)
    summary = {
        "feature_bank_directory": str(output),
        "uni2_mode": "online_integrated_stage2_v16.4",
        "selected_folds": [int(fold) for fold in selected_folds],
        "feature_bank_signature": bank_signature,
        "feature_bank_contract_signature": _feature_bank_contract_signature(config),
        "shared_preprocessing_summary_sha256": sha256_file(config.paths.preprocessed / "preprocessing_summary.json"),
        "artifact_identity": {name: file_stat_identity(output / name) for name in artifact_names},
        "proposals": int(len(proposals)),
        "semantic_training_rows": int(np.sum(semantic_mask)),
        "geometry_viable": int(np.sum(rows["geometry_viability_target"] == 1)),
        "geometry_nonviable": int(np.sum(rows["geometry_viability_target"] == 0)),
        "identity_token_maps_shape": list(token_maps.shape),
        "biology_shape": list(biology.shape),
        "spatial_shape": list(spatial.shape),
        "geometry_shape": list(geometry.shape),
        "resolver": asdict(resolver_summary),
        "stage1_checkpoint_policy": stage1_summary.get("stage1_checkpoint_policy"),
        "stage1_leakage_contract": stage1_summary.get("leakage_contract"),
        "stage1_fold_operating_points": stage1_summary.get("fold_operating_points", []),
    }
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=float))
    return summary


def _validate_bank_provenance(config: PipelineConfig, root: Path, rows: np.ndarray) -> None:
    provenance_name = "provenance.json"
    provenance_path = root / provenance_name
    if not provenance_path.is_file():
        raise RuntimeError(f"Stage-2 feature bank is missing {provenance_name}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    stage1 = {int(item["fold"]): item for item in provenance.get("stage1", {}).get("fold_sources", [])}
    biomask = {int(item["fold"]): item for item in provenance.get("biomask", {}).get("fold_sources", [])}

    def validate_checkpoint_stat(source: dict[str, Any], source_name: str) -> None:
        raw_path = source.get("checkpoint")
        if not raw_path:
            return
        path = Path(str(raw_path))
        if not path.is_file():
            raise RuntimeError(f"{source_name} source checkpoint is missing: {path}")
        stat = path.stat()
        expected_size = source.get("checkpoint_size_bytes")
        expected_mtime = source.get("checkpoint_mtime_ns")
        expected_sha = source.get("checkpoint_sha256")
        if expected_size is not None and int(expected_size) != int(stat.st_size):
            raise RuntimeError(f"{source_name} source checkpoint size changed: {path}")
        if expected_mtime is not None and int(expected_mtime) != int(stat.st_mtime_ns):
            if expected_sha is None or sha256_file(path) != str(expected_sha):
                raise RuntimeError(f"{source_name} source checkpoint content changed: {path}")

    for fold in np.unique(rows["fold"]).astype(int):
        for name, sources in (("stage1", stage1), ("biomask", biomask)):
            if fold not in sources:
                raise RuntimeError(f"feature bank is missing {name} provenance for fold {fold}")
            assert_fold_independent(sample_fold=fold, source=sources[fold], source_name=name)
            validate_checkpoint_stat(sources[fold], name)

    uni2 = provenance.get("uni2", {})
    if uni2.get("source") != "online_integrated_stage2_v16.4":
        raise RuntimeError("v16.4 feature bank must not contain offline UNI2 embeddings")
    if bool(uni2.get("cached_embeddings", True)):
        raise RuntimeError("v16.4 feature bank incorrectly declares cached UNI2 embeddings")
    if uni2.get("adapted_view_sizes") != [64, 128, 256]:
        raise RuntimeError("v16.4 online UNI2 contract requires views 64/128/256")


def load_stage2_feature_bank(config: PipelineConfig, mmap_mode: str = "r") -> dict[str, Any]:
    root = _feature_bank_root(config)
    summary_name = "summary.json"
    summary_path = root / summary_name
    if not summary_path.is_file():
        raise FileNotFoundError("Stage-2 feature bank not found. Run notebook 02_Preprocess_Stage2.ipynb first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_contract = _feature_bank_contract_signature(config)
    if summary.get("feature_bank_contract_signature") != expected_contract:
        raise RuntimeError(
            "Stage-2 feature bank code/config contract is stale. Re-run notebook 02_Preprocess_Stage2.ipynb."
        )
    if summary.get("stage1_checkpoint_policy") != OOF_CHECKPOINT_POLICY:
        raise RuntimeError("Stage-2 feature bank does not match the current Stage-1 OOF checkpoint/decoder contract")
    preprocessing_summary = config.paths.preprocessed / "preprocessing_summary.json"
    if not preprocessing_summary.is_file():
        raise RuntimeError("Shared preprocessing summary is missing")
    if summary.get("shared_preprocessing_summary_sha256") != sha256_file(preprocessing_summary):
        raise RuntimeError("Stage-2 feature bank was built from a different shared preprocessing cache")
    identities = summary.get("artifact_identity")
    if not isinstance(identities, dict) or not identities:
        raise RuntimeError("Stage-2 feature bank is missing artifact identity metadata; rebuild preprocessing")
    for name, expected in identities.items():
        path = root / str(name)
        if not path.is_file():
            raise RuntimeError(f"Stage-2 feature-bank artifact is missing: {path}")
        if file_stat_identity(path) != expected:
            raise RuntimeError(f"Stage-2 feature-bank artifact changed after preprocessing: {path.name}")
    rows = np.load(root / "rows.npy", mmap_mode=mmap_mode, allow_pickle=False)
    _validate_bank_provenance(config, root, rows)
    return {
        "root": root,
        "summary": summary,
        "rows": rows,
        "proposals": np.load(root / "proposals.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "identity_token_maps": np.load(root / "identity_token_maps.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "biology": np.load(root / "biology.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "reliability": np.load(root / "reliability.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "spatial": np.load(root / "spatial.npy", mmap_mode=mmap_mode, allow_pickle=False),
        "geometry": np.load(root / "geometry.npy", mmap_mode=mmap_mode, allow_pickle=False),
    }
