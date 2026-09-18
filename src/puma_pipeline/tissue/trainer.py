from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PumaConfig
from ..data.annotations import rasterize_tissue_map, resolve_annotation_path
from ..stage1.model import build_stage1_model
from ..stage1.data import prepare_full_roi_batch
from ..stage1 import stage1_final_checkpoint, stage1_fold_checkpoint
from ..store import PumaArtifactStore
from ..utils.checkpoint import atomic_torch_save
from ..utils.ema import ModelEMA
from ..utils.provenance import directory_metadata_signature, file_signature
from ..utils.runtime import (
    configure_runtime,
    make_cuda_grad_scaler,
    resolve_cuda_amp_dtype,
    runtime_batch_size,
)
from . import TISSUE_TRAINING_CONTRACT
from .model import TissueHead


# Remember devices that need the non-cuDNN FPN fallback.
_FPN_CUDNN_DISABLED_DEVICES: set[int] = set()



def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _tissue_run_seed(config: PumaConfig, fold: int | None) -> int:
    return int(config.seed + (30_013 if fold is None else 2_003 * (fold + 1)))


def _require_training_amp(config: PumaConfig) -> torch.dtype:
    amp = resolve_cuda_amp_dtype(config.use_bfloat16)
    if config.use_bfloat16 and amp == torch.float16:
        print("Tissue: BF16 is unsupported on this GPU; using FP16 with GradScaler.")
    return amp

def build_tissue_targets(config: PumaConfig, force: bool = False) -> Path:
    out_dir = config.path("tissue_output_dir")
    out = out_dir / "tissue_targets.npy"
    manifest_path = out_dir / "tissue_targets_manifest.json"
    source_signature = {
        "tissue_annotations": directory_metadata_signature(config.path("tissue_geojson_dir"), ("*.json", "*.geojson")),
        "roi_manifest": file_signature(config.path("artifact_dir") / "puma_roi_manifest.npy"),
    }
    if out.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_schema") == "puma" and manifest.get("source_signature") == source_signature:
            existing = np.load(out, mmap_mode="r", allow_pickle=False)
            if existing.dtype == np.uint8 and existing.ndim == 3 and existing.shape[1:] == (1024, 1024):
                return out
        print("Rebuilding stale tissue target cache automatically."); force=True
    store = PumaArtifactStore.open(config.path("artifact_dir"))
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(store.images), 1024, 1024))
    directory = config.path("tissue_geojson_dir")
    for roi in range(len(store.images)):
        path = resolve_annotation_path(store.manifest[roi], directory, tissue=True)
        targets[roi] = rasterize_tissue_map(path)
    targets.flush()
    manifest_path.write_text(
        json.dumps({"artifact_schema": "puma", "rows": len(store.images), "source_signature": source_signature}, indent=2),
        encoding="utf-8",
    )
    return out


def _resolve_detector_path(config: PumaConfig, fold: int | None) -> Path:
    return stage1_final_checkpoint(config) if fold is None else stage1_fold_checkpoint(config, fold)


def _load_detector(config: PumaConfig, fold: int | None, device: torch.device) -> torch.nn.Module:
    path = _resolve_detector_path(config, fold)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_fingerprint") != config.stage1_fingerprint or payload.get("architecture") != "full_roi_1024_native_stride1_point_detector":
        raise RuntimeError(f"Incompatible Stage-1 detector checkpoint: {path}")
    model = build_stage1_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Keep the frozen detector in the same CUDA memory format as Stage 1.
    return model.to(device=device, memory_format=torch.channels_last).eval()


def _prepare(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Use the same RGB preprocessing as Stage 1.
    return prepare_full_roi_batch(image, device)


def _stage1_fpn_micro_batch_size(config: PumaConfig) -> int:
    """Choose a safe Stage-1 micro-batch for tissue feature extraction."""
    requested = max(1, int(config.stage1_inference_batch_size))
    try:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 1
    if total_gib <= 18:
        cap = 1
    elif total_gib <= 28:
        cap = 2
    elif total_gib <= 50:
        cap = 4
    elif total_gib <= 80:
        cap = 6
    else:
        cap = requested
    return max(1, min(requested, cap))


def _recoverable_fpn_cuda_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "out of memory" in message
        or "unable to find an engine" in message
        or "cudnn_status_not_supported" in message
        or "cudnn_status_execution_failed" in message
    )


@torch.no_grad()
def _extract_fpn_microbatched(
    detector: torch.nn.Module,
    image: torch.Tensor,
    amp_dtype: torch.dtype,
    micro_batch_size: int,
) -> torch.Tensor:
    """Extract frozen Stage-1 FPN features and retry with smaller batches if needed."""
    total = int(image.shape[0])
    if total <= 0:
        raise ValueError("Cannot extract FPN features from an empty image batch.")
    micro = max(1, min(int(micro_batch_size), total))
    output: torch.Tensor | None = None
    start = 0
    device_index = image.device.index
    if device_index is None and image.is_cuda:
        device_index = torch.cuda.current_device()
    cudnn_disabled = (
        image.is_cuda
        and device_index is not None
        and int(device_index) in _FPN_CUDNN_DISABLED_DEVICES
    )
    while start < total:
        stop = min(start + micro, total)
        chunk = image[start:stop]
        try:
            if cudnn_disabled:
                with torch.backends.cudnn.flags(enabled=False):
                    with torch.autocast("cuda", dtype=amp_dtype):
                        feature = detector.extract_fpn(chunk).detach()
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    feature = detector.extract_fpn(chunk).detach()
        except RuntimeError as exc:
            if not _recoverable_fpn_cuda_error(exc):
                raise
            torch.cuda.empty_cache()
            if micro > 1:
                new_micro = max(1, micro // 2)
                print(
                    f"Tissue: Stage-1 FPN CUDA fallback {micro} -> {new_micro} "
                    f"after: {str(exc).splitlines()[0]}"
                )
                micro = new_micro
                continue
            message = str(exc).lower()
            if "unable to find an engine" not in message and "cudnn" not in message:
                raise
            if image.is_cuda and device_index is not None:
                _FPN_CUDNN_DISABLED_DEVICES.add(int(device_index))
            if not cudnn_disabled:
                print(
                    "Tissue: cuDNN cannot execute the frozen Stage-1 FPN at "
                    "batch=1 on this GPU; switching tissue FPN extraction to "
                    "the non-cuDNN convolution fallback for the rest of this "
                    "Python process."
                )
            cudnn_disabled = True
            with torch.backends.cudnn.flags(enabled=False):
                with torch.autocast("cuda", dtype=amp_dtype):
                    feature = detector.extract_fpn(chunk).detach()

        if output is None:
            output = torch.empty(
                (total, *feature.shape[1:]),
                device=feature.device,
                dtype=feature.dtype,
                memory_format=torch.channels_last,
            )
        output[start:stop].copy_(feature)
        start = stop

    if output is None:
        raise RuntimeError("Stage-1 FPN extraction produced no output.")
    return output


def _loss(logits: torch.Tensor, target: torch.Tensor, config: PumaConfig) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    prob = logits.softmax(1)
    onehot = F.one_hot(target, num_classes=6).permute(0, 3, 1, 2).float()
    intersection = (prob * onehot).sum((0, 2, 3))
    denominator = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    class_dice = (2 * intersection + 1.0) / (denominator + 1.0)
    foreground_dice = 1.0 - class_dice[1:].mean()
    return config.tissue_ce_weight * ce + config.tissue_dice_weight * foreground_dice


def _lr(config: PumaConfig, epoch: int) -> float:
    if epoch <= config.tissue_warmup_epochs:
        return config.tissue_learning_rate * epoch / max(config.tissue_warmup_epochs, 1)
    progress = (epoch - config.tissue_warmup_epochs) / max(config.tissue_epochs - config.tissue_warmup_epochs, 1)
    return config.tissue_minimum_learning_rate + 0.5 * (config.tissue_learning_rate - config.tissue_minimum_learning_rate) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _cache_fpn_on_gpu(
    detector: torch.nn.Module,
    store: PumaArtifactStore,
    indices: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    indices = np.asarray(indices, dtype=np.int64)
    cache: torch.Tensor | None = None
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        roi = indices[start:stop]
        image = torch.from_numpy(np.asarray(store.images[roi]).copy()).permute(0, 3, 1, 2)
        image = _prepare(image, device)
        fpn = _extract_fpn_microbatched(
            detector, image, amp_dtype, max(1, int(batch_size))
        )
        if cache is None:
            cache = torch.empty((len(indices), *fpn.shape[1:]), dtype=amp_dtype, device=device)
        cache[start:stop].copy_(fpn.to(dtype=amp_dtype))
    if cache is None:
        raise RuntimeError("Cannot build a tissue FPN cache from an empty ROI set.")
    return cache


def _fpn_cache_fits_gpu(config: PumaConfig, number_of_rois: int) -> bool:
    """Check whether the full FPN cache can fit on the GPU."""
    try:
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return False
    # Stage-1 FPN has 128 channels at half resolution.
    side = config.image_size // 2
    bytes_per_value = 2  # FP16 or BF16
    estimated = int(number_of_rois) * 128 * side * side * bytes_per_value
    return estimated <= int(0.45 * total_memory)


def train_tissue(config: PumaConfig, fold: int | None, *, resume: bool = True) -> Path:
    if fold is not None and fold not in range(config.number_of_folds):
        raise ValueError(f"fold must be 0..{config.number_of_folds - 1} or None.")
    if not torch.cuda.is_available():
        raise RuntimeError("Tissue training requires CUDA.")
    run_seed = _tissue_run_seed(config, fold)
    configure_runtime(run_seed, config.use_tf32, config.deterministic)
    amp_dtype = _require_training_amp(config)

    target_path = build_tissue_targets(config)
    store = PumaArtifactStore.open(config.path("artifact_dir"))
    targets = np.load(target_path, mmap_mode="r", allow_pickle=False)
    folds = np.asarray(store.folds, dtype=np.int64)
    train_indices = np.arange(len(store.images)) if fold is None else np.flatnonzero(folds != fold)
    if not len(train_indices):
        raise RuntimeError("Tissue training split is empty.")
    device = torch.device("cuda")
    detector = _load_detector(config, fold, device)
    fpn_micro_batch = _stage1_fpn_micro_batch_size(config)
    if fpn_micro_batch < int(config.stage1_inference_batch_size):
        print(
            f"Tissue: capping frozen Stage-1 FPN micro-batch "
            f"{config.stage1_inference_batch_size} -> {fpn_micro_batch} for "
            f"{torch.cuda.get_device_name(0)}."
        )
    fpn_cache: torch.Tensor | None = None
    if _fpn_cache_fits_gpu(config, len(train_indices)):
        try:
            fpn_cache = _cache_fpn_on_gpu(
                detector,
                store,
                train_indices,
                device,
                amp_dtype,
                fpn_micro_batch,
            )
        except RuntimeError as exc:
            if not _recoverable_fpn_cuda_error(exc):
                raise
            fpn_cache = None
            torch.cuda.empty_cache()
            print(
                "Tissue: GPU FPN cache/extraction could not be allocated/executed; "
                "using on-demand micro-batched FPN extraction."
            )
    if fpn_cache is not None:
        del detector
        detector = None
        torch.cuda.empty_cache()
    else:
        print("Tissue: using low-memory on-demand Stage-1 FPN extraction instead of a full GPU cache.")

    head = TissueHead(base=config.tissue_base_channels).to(device)
    ema = ModelEMA(head, config.tissue_ema_decay)
    try:
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config.tissue_learning_rate,
            weight_decay=config.tissue_weight_decay, fused=True
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config.tissue_learning_rate,
            weight_decay=config.tissue_weight_decay
        )
    scaler = make_cuda_grad_scaler(amp_dtype)
    tissue_batch_size = runtime_batch_size(config.tissue_batch_size, training=True)
    run_dir = config.path("tissue_output_dir") / ("full" if fold is None else f"fold_{fold}")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    start_epoch = 1
    if resume and latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config.tissue_fingerprint:
            raise RuntimeError("Tissue resume checkpoint was produced by another config.")
        if payload.get("training_contract") != TISSUE_TRAINING_CONTRACT:
            print("Ignoring stale tissue resume checkpoint: preprocessing contract changed; restarting this tissue run from epoch 1.")
        else:
            head.load_state_dict(payload["model_state"], strict=True)
            ema.load_state_dict(payload["ema_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            _move_optimizer(optimizer, device)
            if "scaler_state" in payload:
                scaler.load_state_dict(payload["scaler_state"])
            start_epoch = int(payload["epoch"]) + 1

    for epoch in range(start_epoch, config.tissue_epochs + 1):
        epoch_seed = run_seed + epoch * 1_000_003
        configure_runtime(epoch_seed, config.use_tf32, config.deterministic)
        rng = np.random.default_rng(epoch_seed)
        head.train()
        for group in optimizer.param_groups:
            group["lr"] = _lr(config, epoch)
        order = rng.permutation(len(train_indices))
        for start in range(0, len(order), tissue_batch_size):
            local = order[start:start + tissue_batch_size]
            roi = train_indices[local]
            if fpn_cache is not None:
                feature = fpn_cache[torch.as_tensor(local, device=device, dtype=torch.long)]
            else:
                if detector is None:
                    raise RuntimeError("Tissue low-memory FPN extractor is unavailable.")
                image = torch.from_numpy(np.asarray(store.images[roi]).copy()).permute(0, 3, 1, 2)
                image = _prepare(image, device)
                feature = _extract_fpn_microbatched(
                    detector, image, amp_dtype, fpn_micro_batch
                )
            target = torch.from_numpy(np.asarray(targets[roi]).copy()).to(
                device=device, dtype=torch.long, non_blocking=True
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = head(feature, target.shape[-2:])
                loss = _loss(logits, target, config)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(head)
        atomic_torch_save(
            {
                "epoch": epoch,
                "config_fingerprint": config.tissue_fingerprint,
                "training_contract": TISSUE_TRAINING_CONTRACT,
                "model_state": head.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
            },
            latest,
        )
    final = run_dir / "tissue_final_ema.pt"
    atomic_torch_save(
        {
            "artifact_schema": "puma",
            "config_fingerprint": config.tissue_fingerprint,
            "training_contract": TISSUE_TRAINING_CONTRACT,
            "epoch": config.tissue_epochs,
            "model_state": ema.state_dict(),
            "architecture": "stage1_fpn_tissue_head",
            "runtime_amp_dtype": str(amp_dtype).replace("torch.", ""),
        },
        final,
    )
    if fpn_cache is not None:
        del fpn_cache
    if detector is not None:
        del detector
    torch.cuda.empty_cache()
    return final


def generate_tissue_oof(config: PumaConfig, force: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Tissue OOF generation requires CUDA.")
    amp_dtype = _require_training_amp(config)
    out = config.path("tissue_output_dir") / "tissue_oof_probabilities.npy"
    metadata_path = config.path("tissue_output_dir") / "tissue_oof_metadata.json"
    target_path = build_tissue_targets(config)
    detector_signatures = {
        str(fold): file_signature(_resolve_detector_path(config, fold))
        for fold in range(config.number_of_folds)
    }
    head_paths = {
        str(fold): config.path("tissue_output_dir") / f"fold_{fold}" / "tissue_final_ema.pt"
        for fold in range(config.number_of_folds)
    }
    base_expected = {
        "artifact_schema": "puma",
        "config_fingerprint": config.tissue_fingerprint,
        "training_contract": TISSUE_TRAINING_CONTRACT,
        "detector_signatures": detector_signatures,
        "target_signature": file_signature(target_path),
    }
    existing_head_signatures = (
        {fold: file_signature(path) for fold, path in head_paths.items()}
        if all(path.is_file() for path in head_paths.values())
        else None
    )
    if out.is_file() and metadata_path.is_file() and not force:
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {**base_expected, "head_signatures": existing_head_signatures}
        if existing_head_signatures is not None and all(current.get(key) == value for key, value in expected.items()):
            existing = np.load(out, mmap_mode="r", allow_pickle=False)
            if (
                existing.dtype == np.float16
                and existing.ndim == 4
                and existing.shape == (int(current.get("rows", -1)), 6, 1024, 1024)
            ):
                return out
        print("Rebuilding stale tissue OOF predictions automatically.")
        force = True

    store = PumaArtifactStore.open(config.path("artifact_dir"))
    out.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float16, shape=(len(store.images), 6, 1024, 1024)
    )
    device = torch.device("cuda")
    fpn_micro_batch = _stage1_fpn_micro_batch_size(config)
    if fpn_micro_batch < int(config.stage1_inference_batch_size):
        print(
            f"Tissue OOF: capping frozen Stage-1 FPN micro-batch "
            f"{config.stage1_inference_batch_size} -> {fpn_micro_batch} for "
            f"{torch.cuda.get_device_name(0)}."
        )
    final_head_signatures: dict[str, dict] = {}
    for fold in range(config.number_of_folds):
        head_path = train_tissue(config, fold, resume=True)
        final_head_signatures[str(fold)] = file_signature(head_path)
        detector = _load_detector(config, fold, device)
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config.tissue_fingerprint or payload.get("training_contract") != TISSUE_TRAINING_CONTRACT:
            raise RuntimeError(f"Incompatible tissue fold checkpoint: {head_path}")
        head = TissueHead(base=config.tissue_base_channels).to(device)
        head.load_state_dict(payload["model_state"], strict=True)
        head.eval()
        indices = np.flatnonzero(np.asarray(store.folds, dtype=np.int64) == fold)
        inference_batch_size = runtime_batch_size(config.tissue_batch_size, training=False)
        for batch_start in range(0, len(indices), inference_batch_size):
            roi = indices[batch_start:batch_start + inference_batch_size]
            image = torch.from_numpy(np.asarray(store.images[roi]).copy()).permute(0, 3, 1, 2)
            image = _prepare(image, device)
            fpn = _extract_fpn_microbatched(
                detector, image, amp_dtype, fpn_micro_batch
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                prob = head(fpn, (1024, 1024)).softmax(1)
            probabilities[roi] = prob.float().cpu().numpy().astype(np.float16)
        del detector, head
        torch.cuda.empty_cache()
    probabilities.flush()
    metadata_path.write_text(
        json.dumps(
            {
                **base_expected,
                "head_signatures": final_head_signatures,
                "rows": len(store.images),
                "shape": [len(store.images), 6, 1024, 1024],
                "dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out

