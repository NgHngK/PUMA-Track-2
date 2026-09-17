from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..utils.checkpointing import CheckpointManager
from ..utils.history import TrainingHistory
from ..utils.provenance import canonical_fingerprint, numpy_array_sha256, sha256_file
from ..utils.runtime import configure_runtime, dataloader_worker_count, dataloader_worker_init, make_grad_scaler
from .biomask import ProposalBiologyNetwork, biology_loss
from .biomask_data import ProposalBiologyDataset
from .biomask_sampling import (
    BioMaskStratifiedSampler,
    build_sampling_plan,
    classify_negative_strata,
)


def _biomask_training_signature(config: PipelineConfig, training_folds: list[int]) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    payload = {
        "training_folds": list(training_folds),
        "preprocess": asdict(config.stage2_preprocess),
        "runtime": {
            "use_bfloat16": bool(config.runtime.use_bfloat16),
            "allow_tf32": bool(config.runtime.allow_tf32),
            "deterministic": bool(config.runtime.deterministic),
        },
    }
    payload["code_fingerprint"] = canonical_fingerprint(
        {},
        (
            here / "biomask.py",
            here / "biomask_data.py",
            here / "biomask_sampling.py",
            here / "targets.py",
            here / "crops.py",
            Path(__file__).resolve().parents[1] / "constants.py",
        ),
    )
    return payload


def _biomask_data_signature(
    config: PipelineConfig,
    shared: dict[str, np.ndarray],
    proposals: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Bind resumable BioMask optimizer state to the exact rows/targets/data cache.

    This is intentionally stored separately from the historical
    ``training_signature`` so existing v16.3.2 checkpoints remain resumable.
    New checkpoints carry the stronger identity automatically.
    """
    summary_path = config.paths.preprocessed / "preprocessing_summary.json"
    shared_identity: dict[str, Any] = {}
    if summary_path.is_file():
        shared_identity["preprocessing_summary_sha256"] = sha256_file(summary_path)
    for name in ("images", "nuclei", "polygon_points", "folds"):
        array = shared.get(name)
        filename = getattr(array, "filename", None)
        if filename is not None and Path(filename).is_file():
            stat = Path(filename).stat()
            shared_identity[name] = {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    return {
        "proposal_sha256": numpy_array_sha256(np.asarray(proposals)),
        "target_sha256": numpy_array_sha256(np.asarray(targets)),
        "shared": shared_identity,
    }


def _loader(
    dataset,
    batch_size: int,
    config: PipelineConfig,
    shuffle: bool,
    *,
    sampler=None,
) -> DataLoader:
    if len(dataset) == 0 and shuffle:
        raise RuntimeError("BioMask training dataset is empty for this fold")
    if sampler is not None and shuffle:
        raise ValueError("BioMask DataLoader cannot use shuffle=True together with a sampler")
    workers = min(8, dataloader_worker_count(config.stage2_preprocess.cpu_crop_workers))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(config.runtime.pin_memory and torch.cuda.is_available()),
        # A stratified sampler encodes an exact epoch plan; dropping the tail would
        # change its actual q_s and invalidate the importance correction.
        "drop_last": bool(shuffle and sampler is None and len(dataset) >= batch_size),
        "worker_init_fn": dataloader_worker_init,
    }
    if workers > 0:
        persistent = bool(config.runtime.persistent_workers and not getattr(dataset, "augment", False))
        kwargs.update(persistent_workers=persistent, prefetch_factor=2)
    loader = DataLoader(dataset, **kwargs)
    if (shuffle or sampler is not None) and len(loader) == 0:
        raise RuntimeError("BioMask training loader has zero batches; reduce biomask_batch_size")
    return loader


def _cosine_lr(base: float, minimum: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return base
    progress = epoch / max(epochs - 1, 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _dataset(
    shared: dict[str, np.ndarray],
    proposals: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    config: PipelineConfig,
    *,
    augment: bool,
    seed: int,
    presence_sampling_weights: np.ndarray | None = None,
    negative_strata: np.ndarray | None = None,
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
        augment=augment,
        seed=seed,
        presence_sampling_weights=presence_sampling_weights,
        negative_strata=negative_strata,
    )


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positive = target == 1
    negative = target == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank
        i = j
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int((target == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = target[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y == 1].sum() / n_pos)


def _presence_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=np.float64).clip(0.0, 1.0)
    y = np.asarray(targets, dtype=np.int8)
    d = np.asarray(distances, dtype=np.float64)
    s = np.asarray(strata, dtype=np.int8)
    positive = y == 1
    negative = y == 0
    result: dict[str, Any] = {
        "auroc": _binary_auc(y, p),
        "auprc": _average_precision(y, p),
        "brier": float(np.mean(np.square(p - y))) if len(y) else float("nan"),
        "positive_mean": float(p[positive].mean()) if positive.any() else float("nan"),
        "negative_mean": float(p[negative].mean()) if negative.any() else float("nan"),
    }
    result["gap"] = float(result["positive_mean"] - result["negative_mean"])
    distance_bins = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 22), (22, 28), (28, float("inf")))
    result["distance_curve"] = {
        f"{lo:g}-{hi:g}": {
            "count": int(((d >= lo) & (d < hi)).sum()),
            "mean": float(p[(d >= lo) & (d < hi)].mean()) if ((d >= lo) & (d < hi)).any() else float("nan"),
        }
        for lo, hi in distance_bins
    }
    result["negative_strata"] = {
        str(value): {
            "count": int(((s == value) & negative).sum()),
            "mean": float(p[(s == value) & negative].mean()) if ((s == value) & negative).any() else float("nan"),
        }
        for value in range(4)
    }
    return result


def _negative_mask_metrics(mask_probability: np.ndarray, target: np.ndarray, strata: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(mask_probability, dtype=np.float32)
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(strata, dtype=np.int8)
    negative = y == 0
    if not negative.any():
        return {}
    flat = probability[negative].reshape(int(negative.sum()), -1)
    output: dict[str, Any] = {
        "mean_probability": float(flat.mean()),
        "mean_soft_area": float(flat.sum(axis=1).mean()),
        "mean_max_probability": float(flat.max(axis=1).mean()),
    }
    by_stratum: dict[str, Any] = {}
    neg_strata = s[negative]
    for value in range(4):
        mask = neg_strata == value
        if not mask.any():
            continue
        values = flat[mask]
        by_stratum[str(value)] = {
            "count": int(mask.sum()),
            "mean_probability": float(values.mean()),
            "mean_soft_area": float(values.sum(axis=1).mean()),
            "mean_max_probability": float(values.max(axis=1).mean()),
        }
    output["by_stratum"] = by_stratum
    return output


def _batch_target(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mask": batch["mask"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "presence_target": batch["presence_target"].to(device, non_blocking=True),
        "presence_loss_weight": batch["presence_loss_weight"].to(device, non_blocking=True),
        "presence_sampling_weight": batch["presence_sampling_weight"].to(device, non_blocking=True),
        "instance_mask_loss_weight": batch["instance_mask_loss_weight"].to(device, non_blocking=True),
        "empty_mask_loss_weight": batch["empty_mask_loss_weight"].to(device, non_blocking=True),
        "offset_loss_weight": batch["offset_loss_weight"].to(device, non_blocking=True),
        "quality_loss_weight": batch["quality_loss_weight"].to(device, non_blocking=True),
        "center_offset_target": batch["center_offset_target"].to(device, non_blocking=True),
    }


def train_biology_fold(
    shared: dict[str, np.ndarray],
    proposals: np.ndarray,
    targets: np.ndarray,
    *,
    fold: int,
    config: PipelineConfig,
    output_directory: Path,
    force: bool = False,
    data_signature: dict[str, Any] | None = None,
) -> Path:
    runtime = configure_runtime(
        config.data.seed + 811 * fold,
        use_bfloat16=config.runtime.use_bfloat16,
        allow_tf32=config.runtime.allow_tf32,
        deterministic=config.runtime.deterministic,
    )
    train_indices = np.flatnonzero(proposals["fold"] != int(fold))
    held_indices = np.flatnonzero(proposals["fold"] == int(fold))
    if not len(train_indices) or not len(held_indices):
        raise RuntimeError(f"BioMask fold {fold} has empty train or held split")
    training_folds = sorted(set(proposals["fold"][train_indices].astype(int).tolist()))
    if int(fold) in training_folds:
        raise RuntimeError("BioMask OOF provenance failure: held fold appears in training folds")

    p = config.stage2_preprocess
    train_strata, presence_weights, sampling_report = build_sampling_plan(
        proposals, targets, shared["nuclei"], train_indices,
        crop_size=p.crop_size,
        near_outer_radius=p.biomask_negative_near_outer_radius_pixels,
        crowded_min_nuclei=p.biomask_negative_crowded_min_nuclei,
        high_score_quantile=p.biomask_negative_high_score_quantile,
        target_fractions=p.biomask_negative_sampling_fractions,
        max_importance_weight=p.biomask_max_importance_weight,
    )
    held_strata, _ = classify_negative_strata(
        proposals, targets, shared["nuclei"], held_indices,
        crop_size=p.crop_size,
        near_outer_radius=p.biomask_negative_near_outer_radius_pixels,
        crowded_min_nuclei=p.biomask_negative_crowded_min_nuclei,
        high_score_quantile=p.biomask_negative_high_score_quantile,
        high_score_cutoff=sampling_report.high_score_cutoff,
    )
    train_dataset = _dataset(
        shared, proposals, targets, train_indices, config,
        augment=True, seed=config.data.seed + fold,
        presence_sampling_weights=presence_weights,
        negative_strata=train_strata,
    )
    held_dataset = _dataset(
        shared, proposals, targets, held_indices, config,
        augment=False, seed=config.data.seed + fold,
        negative_strata=held_strata,
    )
    train_sampler = BioMaskStratifiedSampler(
        train_indices, targets, train_strata, sampling_report.planned_counts,
        seed=config.data.seed + 1709 * fold,
    )
    train_loader = _loader(train_dataset, p.biomask_batch_size, config, False, sampler=train_sampler)
    held_loader = _loader(held_dataset, p.biomask_batch_size * 2, config, False)

    fold_directory = Path(output_directory) / f"fold_{fold}"
    checkpoint = CheckpointManager(fold_directory / "checkpoints")
    if force:
        checkpoint.latest_path.unlink(missing_ok=True)
        checkpoint.final_path.unlink(missing_ok=True)
        (fold_directory / "history.jsonl").unlink(missing_ok=True)
        (fold_directory / "history.csv").unlink(missing_ok=True)
    history = TrainingHistory(fold_directory)
    model = ProposalBiologyNetwork(
        p.biomask_base_channels,
        presence_sigma_pixels=p.biomask_presence_sigma_pixels,
        presence_support_radius_pixels=p.biomask_presence_support_radius_pixels,
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=p.biomask_learning_rate,
        weight_decay=p.biomask_weight_decay,
    )
    scaler = make_grad_scaler(enabled=runtime.amp_enabled and runtime.amp_dtype == torch.float16)
    signature = _biomask_training_signature(config, training_folds)
    if data_signature is None:
        data_signature = _biomask_data_signature(config, shared, proposals, targets)
    else:
        data_signature = dict(data_signature)
    if checkpoint.latest_path.is_file():
        peek = torch.load(checkpoint.latest_path, map_location="cpu", weights_only=False)
        extra = dict(peek.get("extra") or {})
        if extra.get("training_signature") != signature:
            raise RuntimeError(
                f"BioMask resume settings do not match {checkpoint.latest_path}; "
                "remove or archive the incompatible checkpoint, or run with force=True."
            )
        previous_data_signature = extra.get("data_signature")
        if previous_data_signature is not None and previous_data_signature != data_signature:
            raise RuntimeError(
                f"BioMask resume data identity does not match {checkpoint.latest_path}; "
                "run with force=True rather than mixing optimizer state across feature rows."
            )
    start_epoch, resume_payload = checkpoint.resume(model, optimizer=optimizer, scaler=scaler, map_location="cpu")
    if start_epoch:
        # CheckpointManager.resume already returns the checkpoint's ``extra``
        # dictionary. The previous code incorrectly looked for another nested
        # ``extra`` key, which made every genuine resume fail provenance checks.
        extra = dict(resume_payload or {})
        if int(extra.get("fold", -1)) != int(fold) or extra.get("training_folds") != training_folds:
            raise RuntimeError("BioMask resume checkpoint provenance does not match the requested fold split")
        if start_epoch > p.biomask_epochs:
            raise RuntimeError(f"BioMask resume checkpoint epoch exceeds configured epochs: {checkpoint.latest_path}")
    history.truncate_from("epoch", start_epoch)

    for epoch in range(start_epoch, p.biomask_epochs):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        lr = _cosine_lr(p.biomask_learning_rate, p.biomask_minimum_learning_rate, epoch, p.biomask_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        train_acc: dict[str, float] = {}
        train_n = 0
        for batch in train_loader:
            x = batch["input"].to(runtime.device, non_blocking=True)
            target = _batch_target(batch, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=runtime.device.type, dtype=runtime.amp_dtype, enabled=runtime.amp_enabled):
                outputs = model(x, valid_mask=target["valid_mask"])
                loss, terms = biology_loss(outputs, target, empty_mask_coefficient=p.biomask_empty_mask_loss_weight)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite BioMask loss at fold={fold}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("Non-finite BioMask gradient norm")
            scaler.step(optimizer)
            scaler.update()
            n = len(x)
            train_n += n
            for key, value in terms.items():
                train_acc[key] = train_acc.get(key, 0.0) + float(value) * n

        model.eval()
        held_acc: dict[str, float] = {}
        held_n = 0
        held_presence: list[np.ndarray] = []
        held_target: list[np.ndarray] = []
        held_distance: list[np.ndarray] = []
        held_stratum_values: list[np.ndarray] = []
        held_masks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in held_loader:
                x = batch["input"].to(runtime.device, non_blocking=True)
                target = _batch_target(batch, runtime.device)
                with torch.autocast(device_type=runtime.device.type, dtype=runtime.amp_dtype, enabled=runtime.amp_enabled):
                    outputs = model(x, valid_mask=target["valid_mask"])
                    _, terms = biology_loss(outputs, target, empty_mask_coefficient=p.biomask_empty_mask_loss_weight)
                n = len(x)
                held_n += n
                for key, value in terms.items():
                    held_acc[key] = held_acc.get(key, 0.0) + float(value) * n
                held_presence.append(outputs["presence_logits"].sigmoid().float().cpu().numpy())
                held_target.append(batch["presence_target"].numpy())
                held_distance.append(batch["nearest_distance"].numpy())
                held_stratum_values.append(batch["negative_stratum"].numpy())
                held_masks.append((outputs["mask_logits"].sigmoid().float() * target["valid_mask"]).cpu().numpy()[:, 0])

        presence_probability = np.concatenate(held_presence) if held_presence else np.empty(0, dtype=np.float32)
        presence_target = np.concatenate(held_target).astype(np.int8) if held_target else np.empty(0, dtype=np.int8)
        presence_distance = np.concatenate(held_distance) if held_distance else np.empty(0, dtype=np.float32)
        presence_strata = np.concatenate(held_stratum_values).astype(np.int8) if held_stratum_values else np.empty(0, dtype=np.int8)
        mask_probability = np.concatenate(held_masks) if held_masks else np.empty((0, p.crop_size, p.crop_size), dtype=np.float32)
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train": {k: v / max(train_n, 1) for k, v in train_acc.items()},
            "held": {k: v / max(held_n, 1) for k, v in held_acc.items()},
            "presence": _presence_metrics(presence_probability, presence_target, presence_distance, presence_strata),
            "negative_mask": _negative_mask_metrics(mask_probability, presence_target, presence_strata),
            "sampling": sampling_report.as_dict(),
        }
        history.append(record)
        print(
            f"BioMask fold={fold} epoch={epoch+1}/{p.biomask_epochs} "
            f"train={record['train'].get('total', float('nan')):.5f} held={record['held'].get('total', float('nan')):.5f}"
        )
        checkpoint.save_latest(
            epoch=epoch, model=model, optimizer=optimizer, scaler=scaler,
            extra={
                "fold": int(fold),
                "training_folds": training_folds,
                "oof_safe": True,
                "training_signature": signature,
                "data_signature": data_signature,
                "sampling": sampling_report.as_dict(),
            },
        )
    checkpoint.save_final(
        model=model,
        extra={
            "epoch": p.biomask_epochs - 1,
            "fold": int(fold),
            "training_folds": training_folds,
            "oof_safe": True,
            "training_signature": signature,
            "data_signature": data_signature,
            "sampling": sampling_report.as_dict(),
        },
    )
    return checkpoint.final_path

