from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import PumaConfig
from ..constants import (
    REJECT_CLASS_ID,
    STAGE1_CANDIDATE_DTYPE,
    STAGE1_GT_FEATURE_DTYPE,
)
from ..evaluation.matching import match_centroids
from .data import FullRoiDataset, full_roi_collate, prepare_full_roi_batch
from .decode import decode_stage1_batch, prepare_stage1_maps, sample_stage1_outputs, suppress_stage1
from .model import build_stage1_model
from ..store import PumaArtifactStore, load_stage1_gt_features, load_stage1_oof
from ..utils.runtime import configure_runtime, resolve_cuda_amp_dtype, runtime_batch_size


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loader(
    store: PumaArtifactStore,
    rois: np.ndarray,
    config: PumaConfig,
    fold: int,
) -> DataLoader:
    dataset = FullRoiDataset(store, rois, config, augment=False, seed=config.seed + fold)
    kwargs: dict[str, Any] = {
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    if config.workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(
        dataset,
        batch_size=runtime_batch_size(config.stage1_inference_batch_size, training=False),
        shuffle=False,
        collate_fn=full_roi_collate,
        **kwargs,
    )


def _nearest(coordinates: np.ndarray, fallback: float) -> np.ndarray:
    if len(coordinates) > 1:
        return cKDTree(coordinates).query(coordinates, k=2, workers=1)[0][:, 1].astype(np.float32)
    return np.full(len(coordinates), fallback, dtype=np.float32)


def _candidate_rows(
    roi_index: int,
    fold: int,
    prediction,
    ground_truth: np.ndarray,
    radius: float,
) -> list[tuple[Any, ...]]:
    gt_xy = np.column_stack([ground_truth["x"], ground_truth["y"]]).astype(np.float32)
    match = match_centroids(
        prediction.coordinates,
        gt_xy,
        float(radius),
        None,
    )
    matched = {
        int(pred): (int(gt), float(distance))
        for pred, gt, distance in zip(
            match.pred_indices, match.gt_indices, match.distances, strict=True
        )
    }
    nearest = _nearest(prediction.coordinates, 1024.0)
    rows: list[tuple[Any, ...]] = []
    for candidate_index in range(len(prediction.coordinates)):
        gt_index, distance = matched.get(candidate_index, (-1, float("nan")))
        class_id = int(ground_truth[gt_index]["class_id"]) if gt_index >= 0 else REJECT_CLASS_ID
        rows.append(
            (
                -1,
                roi_index,
                candidate_index,
                float(prediction.coordinates[candidate_index, 0]),
                float(prediction.coordinates[candidate_index, 1]),
                float(prediction.heatmap_score[candidate_index]),
                float(prediction.quality[candidate_index]),
                float(prediction.uncertainty[candidate_index]),
                float(prediction.peak_sharpness[candidate_index]),
                float(nearest[candidate_index]),
                prediction.embedding[candidate_index],
                gt_index,
                distance,
                class_id,
                int(gt_index < 0),
                fold,
            )
        )
    return rows


@torch.inference_mode()
def generate_stage1_oof(config: PumaConfig, *, force: bool = False) -> dict[str, Any]:
    """Create out-of-fold detector candidates and GT-aligned features."""
    if not torch.cuda.is_available():
        raise RuntimeError("PUMA Stage-1 OOF generation requires CUDA.")
    configure_runtime(config.seed + 40_009, config.use_tf32, config.deterministic)

    output_dir = config.path("stage1_output_dir")
    candidates_path = output_dir / "stage1_oof_candidates.npy"
    gt_features_path = output_dir / "stage1_oof_gt_features.npy"
    metadata_path = output_dir / "stage1_oof_metadata.json"
    checkpoints = [
        output_dir / f"fold_{fold}" / "best_ema.pt"
        for fold in range(config.number_of_folds)
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-1 final EMA checkpoints: {missing}")
    checkpoint_sha256 = {str(fold): _sha256(path) for fold, path in enumerate(checkpoints)}

    if not force and candidates_path.is_file() and gt_features_path.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatible = (
            cached.get("config_fingerprint") == config.stage1_fingerprint
            and cached.get("checkpoint_sha256") == checkpoint_sha256
        )
        if compatible:
            candidates = load_stage1_oof(candidates_path, config.number_of_folds)
            gt_features = load_stage1_gt_features(gt_features_path, config.number_of_folds)
            if (
                len(candidates) == int(cached.get("number_of_candidates", -1))
                and len(gt_features) == int(cached.get("number_of_gt_features", -1))
            ):
                return cached
        raise RuntimeError("Existing Stage-1 OOF artifacts are stale/corrupt or use different checkpoints; rebuild with --force.")

    store = PumaArtifactStore.open(config.path("artifact_dir"))
    device = torch.device("cuda")
    amp_dtype = resolve_cuda_amp_dtype(config.use_bfloat16)
    if config.use_bfloat16 and amp_dtype == torch.float16:
        print("Stage-1 OOF: BF16 is unsupported on this GPU; using FP16.")
    candidate_rows: list[tuple[Any, ...]] = []
    gt_rows: list[tuple[Any, ...]] = []
    diagnostics: dict[str, Any] = {}
    for fold, checkpoint in enumerate(checkpoints):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config.stage1_fingerprint:
            raise RuntimeError(f"Stage-1 fold {fold} checkpoint/config mismatch.")
        model = build_stage1_model(config).to(device, memory_format=torch.channels_last)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        rois = np.flatnonzero(np.asarray(store.folds).astype(int) == fold)
        fold_candidates = fold_positive = fold_reject = 0
        loader = _loader(store, rois, config, fold)
        for batch in tqdm(loader, desc=f"Stage1 OOF fold {fold}"):
            images = prepare_full_roi_batch(batch["image"], device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
            maps = prepare_stage1_maps(outputs)
            decoded = decode_stage1_batch(
                model,
                outputs,
                minimum_threshold=config.stage1_deployment_threshold,
                local_max_radius=config.stage1_deployment_local_max_radius,
                maps=maps,
            )
            for position, raw_prediction in enumerate(decoded):
                roi_index = int(batch["roi_index"][position])
                prediction = suppress_stage1(
                    raw_prediction,
                    config.stage1_deployment_threshold,
                    config.stage1_deployment_suppression_radius,
                    config.image_size,
                )
                nuclei = store.roi_centroids(roi_index)
                rows = _candidate_rows(
                    roi_index,
                    fold,
                    prediction,
                    nuclei,
                    config.match_radius_pixels,
                )
                candidate_rows.extend(rows)
                fold_candidates += len(rows)
                fold_positive += sum(int(row[-2] == 0) for row in rows)
                fold_reject += sum(int(row[-2] == 1) for row in rows)
                coordinates = np.column_stack([nuclei["x"], nuclei["y"]]).astype(np.float32)
                sampled = sample_stage1_outputs(model, outputs, position, coordinates, maps=maps)
                global_start = int(store.offsets[roi_index])
                for local_index, nucleus in enumerate(nuclei):
                    global_index = global_start + local_index
                    gt_rows.append(
                        (
                            -(global_index + 1),
                            roi_index,
                            float(nucleus["x"]),
                            float(nucleus["y"]),
                            int(nucleus["class_id"]),
                            fold,
                            float(sampled["heatmap_score"][local_index]),
                            float(sampled["quality"][local_index]),
                            float(sampled["uncertainty"][local_index]),
                            float(sampled["peak_sharpness"][local_index]),
                            sampled["embedding"][local_index],
                        )
                    )
        diagnostics[str(fold)] = {
            "rois": int(len(rois)),
            "candidates": fold_candidates,
            "matched": fold_positive,
            "rejects": fold_reject,
        }
        del model
        torch.cuda.empty_cache()

    candidates = np.asarray(candidate_rows, dtype=STAGE1_CANDIDATE_DTYPE)
    candidates["oof_row_id"] = np.arange(len(candidates), dtype=np.int64)
    gt_features = np.asarray(gt_rows, dtype=STAGE1_GT_FEATURE_DTYPE)
    if len(gt_features) != int(store.offsets[-1]):
        raise RuntimeError("Did not create exactly one OOF-model feature row per GT nucleus.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, array in (
        (candidates_path, candidates),
        (gt_features_path, gt_features),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    metadata = {
        "artifact_schema": "puma",
        "config_fingerprint": config.stage1_fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "number_of_candidates": int(len(candidates)),
        "number_of_matched_candidates": int(np.sum(candidates["is_reject"] == 0)),
        "number_of_reject_candidates": int(np.sum(candidates["is_reject"] == 1)),
        "number_of_gt_features": int(len(gt_features)),
        "folds": diagnostics,
        "postprocessing_locked_before_stage2": {
            "threshold": config.stage1_deployment_threshold,
            "local_max_radius": config.stage1_deployment_local_max_radius,
            "suppression_radius": config.stage1_deployment_suppression_radius,
        },
    }
    _atomic_json(metadata_path, metadata)
    return metadata
