from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from ..utils.checkpoint import atomic_torch_save, capture_rng_state, restore_rng_state
from ..config import PumaConfig
from ..utils.ema import ModelEMA
from ..utils.runtime import (
    configure_runtime,
    cuda_compile_recommended,
    make_cuda_grad_scaler,
    resolve_cuda_amp_dtype,
    runtime_batch_size,
)
from .data import FullRoiDataset, full_roi_collate, prepare_full_roi_batch
from .decode import decode_stage1_batch, suppress_stage1
from .model import build_stage1_model
from .targets import stage1_loss
from ..store import PumaArtifactStore


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _decay_groups(model: torch.nn.Module, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def _learning_rate(
    config: PumaConfig, optimizer_step: int, total_steps: int, steps_per_epoch: int
) -> float:
    warmup = max(1, config.stage1_warmup_epochs * steps_per_epoch)
    if optimizer_step < warmup:
        return config.stage1_learning_rate * float(optimizer_step + 1) / warmup
    progress = min(1.0, (optimizer_step - warmup) / max(total_steps - warmup - 1, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.stage1_minimum_learning_rate + cosine * (
        config.stage1_learning_rate - config.stage1_minimum_learning_rate
    )


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


class EpochRandomSampler(Sampler[int]):
    def __init__(self, dataset: FullRoiDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def _loader(
    dataset: FullRoiDataset,
    config: PumaConfig,
    *,
    batch_size: int,
    sampler: Sampler[int] | None = None,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    if config.workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=full_roi_collate,
        drop_last=drop_last,
        **kwargs,
    )


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    store: PumaArtifactStore,
    config: PumaConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    distance_sum = 0.0
    match_count = 0
    from ..evaluation.matching import match_centroids

    amp_dtype = resolve_cuda_amp_dtype(config.use_bfloat16)
    for batch in loader:
        images = prepare_full_roi_batch(batch["image"], device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(images)
        predictions = decode_stage1_batch(
            model,
            outputs,
            minimum_threshold=config.stage1_deployment_threshold,
            local_max_radius=config.stage1_deployment_local_max_radius,
        )
        for position, prediction in enumerate(predictions):
            prediction = suppress_stage1(
                prediction,
                config.stage1_deployment_threshold,
                config.stage1_deployment_suppression_radius,
                config.image_size,
            )
            nuclei = store.roi_centroids(int(batch["roi_index"][position]))
            gt = np.column_stack([nuclei["x"], nuclei["y"]]).astype(np.float32)
            match = match_centroids(
                prediction.coordinates,
                gt,
                config.match_radius_pixels,
                None,
            )
            tp += len(match.pred_indices)
            fp += len(match.unmatched_pred)
            fn += len(match.unmatched_gt)
            distance_sum += float(match.distances.sum())
            match_count += len(match.distances)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "mean_localization_error": distance_sum / max(match_count, 1),
    }


def _save_resume(
    path: Path,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: PumaConfig,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
) -> None:
    atomic_torch_save(
        {
            "artifact_schema": "puma",
            "config_fingerprint": config.stage1_fingerprint,
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "global_step": int(global_step),
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def _early_stopping_due(
    epoch: int,
    best_epoch: int,
    patience: int,
    validation_ran: bool,
) -> bool:
    return bool(
        validation_ran
        and best_epoch > 0
        and epoch - best_epoch >= patience
    )


def train_stage1_fold(
    config: PumaConfig,
    fold: int | None,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Train the Stage-1 detector on full 1024x1024 ROIs."""
    if fold is not None and fold not in range(config.number_of_folds):
        raise ValueError(f"fold must be 0..{config.number_of_folds - 1} or None.")
    if not torch.cuda.is_available():
        raise RuntimeError("Full-ROI Stage 1 requires CUDA.")
    run_seed = config.seed + (10_007 if fold is None else fold * 101)
    configure_runtime(run_seed, config.use_tf32, config.deterministic)
    device = torch.device("cuda")
    amp_dtype = resolve_cuda_amp_dtype(config.use_bfloat16)
    if config.use_bfloat16 and amp_dtype == torch.float16:
        print("Stage 1: BF16 is unsupported on this GPU; using FP16 with GradScaler.")

    store = PumaArtifactStore.open(config.path("artifact_dir"))
    folds = np.asarray(store.folds).astype(int)
    if fold is None:
        train_rois = np.arange(len(store.images), dtype=np.int64)
        validation_rois = np.empty(0, dtype=np.int64)
        run_name = "final_all_data"
    else:
        train_rois = np.flatnonzero(folds != fold)
        validation_rois = np.flatnonzero(folds == fold)
        run_name = f"fold_{fold}"
    train_data = FullRoiDataset(store, train_rois, config, augment=True, seed=run_seed)
    validation_data = FullRoiDataset(
        store, validation_rois, config, augment=False, seed=run_seed + 1
    )
    train_sampler = EpochRandomSampler(train_data, run_seed)
    train_batch_size = runtime_batch_size(config.stage1_micro_batch_size, training=True)
    while train_batch_size > 1 and config.stage1_effective_batch_size % train_batch_size:
        train_batch_size -= 1
    validation_batch_size = runtime_batch_size(
        config.stage1_inference_batch_size, training=False
    )
    if train_batch_size != config.stage1_micro_batch_size:
        print(
            f"Stage 1: reducing runtime micro-batch {config.stage1_micro_batch_size} -> "
            f"{train_batch_size} for available GPU memory; gradient accumulation preserves "
            "the configured effective batch size."
        )
    train_loader = _loader(
        train_data,
        config,
        batch_size=train_batch_size,
        sampler=train_sampler,
        drop_last=True,
    )
    validation_loader = (
        _loader(
            validation_data,
            config,
            batch_size=validation_batch_size,
        )
        if fold is not None
        else None
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            "Stage-1 training loader is empty. Reduce the micro-batch size or check the split."
        )
    train_centroid_mask = np.isin(store.centroids["roi_index"], train_rois)
    class_counts = np.bincount(
        store.centroids["class_id"][train_centroid_mask].astype(int), minlength=10
    )[:10]
    if np.any(class_counts == 0):
        raise RuntimeError("A Stage-1 training split is missing at least one nuclei class.")
    class_weights_np = np.sqrt(class_counts.max() / class_counts)
    class_weights_np = np.minimum(class_weights_np, 4.0)
    class_weights_np /= class_weights_np.mean()
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    run_dir = config.path("stage1_output_dir") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "resolved_config.json")

    final_path = run_dir / "stage1_final_ema.pt"
    summary_path = run_dir / "summary.json"
    if resume and final_path.is_file() and summary_path.is_file():
        final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        complete = bool(summary.get("early_stopped")) or int(
            summary.get("completed_epoch", 0)
        ) >= config.stage1_epochs
        if final_payload.get("config_fingerprint") == config.stage1_fingerprint and complete:
            print(f"COMPLETE Stage 1 {run_name}: {final_path}")
            return summary

    model = build_stage1_model(config).to(device, memory_format=torch.channels_last)
    ema = ModelEMA(model, config.stage1_ema_decay)
    groups = _decay_groups(model, config.stage1_learning_rate, config.stage1_weight_decay)
    try:
        optimizer = torch.optim.AdamW(groups, fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(groups)
    scaler = make_cuda_grad_scaler(amp_dtype)
    runtime_model: torch.nn.Module = model
    if config.stage1_compile and hasattr(torch, "compile"):
        if not cuda_compile_recommended():
            print("Stage-1 torch.compile disabled on this pre-Ampere/unsupported CUDA device.")
        else:
            try:
                runtime_model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            except Exception as exception:
                print(f"Stage-1 torch.compile disabled: {exception}")

    latest = run_dir / "latest.pt"
    best_path = run_dir / "best_ema.pt"
    start_epoch, resume_batch, global_step = 1, 0, 0
    if resume and latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config.stage1_fingerprint:
            raise RuntimeError("Stage-1 resume checkpoint was produced by another config.")
        model.load_state_dict(payload["model_state"], strict=True)
        ema.load_state_dict(payload["ema_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer(optimizer, device)
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"])
        resume_batch = int(payload["batch_in_epoch"])
        global_step = int(payload["global_step"])
        if resume_batch >= len(train_loader):
            start_epoch += 1
            resume_batch = 0
        print(f"RESUME Stage 1 {run_name}: epoch={start_epoch}, batch={resume_batch}")

    batches_per_epoch = len(train_loader)
    runtime_accumulation_steps = config.stage1_effective_batch_size // train_batch_size
    steps_per_epoch = math.ceil(batches_per_epoch / runtime_accumulation_steps)
    total_steps = config.stage1_epochs * steps_per_epoch
    history_path = run_dir / "history.json"
    history: list[dict[str, Any]] = []
    if resume and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])

    best_f1 = float("-inf")
    best_epoch = 0
    for row in history:
        value = row.get("validation_binary_f1")
        if value is not None and float(value) > best_f1:
            best_f1 = float(value)
            best_epoch = int(row["epoch"])

    stopped_early = False
    completed_epoch = max((int(row.get("epoch", 0)) for row in history), default=0)
    for epoch in range(start_epoch, config.stage1_epochs + 1):
        train_data.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        loader = train_loader
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {
            name: 0.0
            for name in (
                "total",
                "heatmap",
                "offset",
                "quality",
                "uncertainty",
                "coarse_type",
            )
        }
        seen = 0
        progress = tqdm(loader, desc=f"Stage1 {run_name} {epoch:03d}/{config.stage1_epochs}")
        for batch_index, batch in enumerate(progress):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            images = prepare_full_roi_batch(batch["image"], device, batch["photometric"])
            index_targets = {"offset_index", "center_index", "coarse_type"}
            targets = {
                key: value.to(
                    device=device,
                    dtype=torch.long if key in index_targets else torch.float32,
                    non_blocking=True,
                )
                for key, value in batch["targets"].items()
            }
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                try:
                    outputs = runtime_model(images)
                except Exception as exception:
                    if runtime_model is model:
                        raise
                    print(
                        "Stage-1 compiled forward failed; disabling torch.compile and "
                        f"retrying eagerly. Original error: {exception}"
                    )
                    runtime_model = model
                    dynamo = getattr(torch, "_dynamo", None)
                    if dynamo is not None and hasattr(dynamo, "reset"):
                        dynamo.reset()
                    outputs = model(images)
                loss, components = stage1_loss(
                    model, outputs, targets, config, class_weights=class_weights
                )
                group_start = (
                    batch_index // runtime_accumulation_steps
                ) * runtime_accumulation_steps
                group_size = min(runtime_accumulation_steps, len(loader) - group_start)
                scaled = loss / group_size
            scaler.scale(scaled).backward()
            final_batch = batch_index + 1 == len(loader)
            update = (batch_index + 1) % runtime_accumulation_steps == 0 or final_batch
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage1_gradient_clip_norm)
                _set_lr(
                    optimizer, _learning_rate(config, global_step, total_steps, steps_per_epoch)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
            count = len(images)
            sums["total"] += float(loss.detach()) * count
            for name, value in components.items():
                sums[name] += float(value) * count
            seen += count
            progress.set_postfix(loss=f"{sums['total'] / max(seen, 1):.4f}")
            checkpoint_interval = max(1, int(getattr(config, "stage1_checkpoint_every_steps", 100)))
            if update and global_step % checkpoint_interval == 0:
                _save_resume(
                    latest, model, ema, optimizer, scaler, config, epoch, batch_index + 1, global_step
                )
        resume_batch = 0
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value / max(seen, 1) for key, value in sums.items()},
        }
        validation_ran = False
        if fold is not None and (
            epoch % config.stage1_validation_interval == 0 or epoch == config.stage1_epochs
        ):
            validation_ran = True
            validation_metrics = _validate(
                ema.module, validation_loader, store, config, device
            )
            record.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            current_f1 = float(validation_metrics["binary_f1"])
            improved = current_f1 > best_f1 + config.stage1_early_stopping_min_delta
            record["validation_improved"] = bool(improved)
            if improved:
                best_f1 = current_f1
                best_epoch = epoch
                atomic_torch_save(
                    {
                        "artifact_schema": "puma",
                        "config_fingerprint": config.stage1_fingerprint,
                        "epoch": epoch,
                        "model_state": ema.state_dict(),
                        "postprocessing": {
                            "threshold": config.stage1_deployment_threshold,
                            "local_max_radius": config.stage1_deployment_local_max_radius,
                            "suppression_radius": config.stage1_deployment_suppression_radius,
                        },
                        "architecture": "full_roi_1024_native_stride1_point_detector",
                    },
                    best_path,
                )

        stopped_early = bool(
            fold is not None
            and epoch < config.stage1_epochs
            and _early_stopping_due(
                epoch,
                best_epoch,
                config.stage1_early_stopping_patience,
                validation_ran,
            )
        )
        record["early_stopping_triggered"] = stopped_early
        record["best_validation_epoch"] = best_epoch if best_epoch > 0 else None
        record["best_validation_binary_f1"] = best_f1 if best_epoch > 0 else None

        history = [row for row in history if int(row.get("epoch", -1)) != epoch]
        history.append(record)
        history.sort(key=lambda row: int(row["epoch"]))
        _atomic_json(history_path, {"epochs": history})
        _save_resume(latest, model, ema, optimizer, scaler, config, epoch, len(loader), global_step)
        completed_epoch = epoch
        if stopped_early:
            print(
                f"EARLY STOP Stage 1 {run_name}: best epoch={best_epoch}, "
                f"best validation F1={best_f1:.6f}, patience={config.stage1_early_stopping_patience}"
            )
            break

    selected_epoch = completed_epoch
    selected_state = ema.state_dict()
    selection = "last_ema"
    if fold is not None and best_path.is_file():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if best_payload.get("config_fingerprint") != config.stage1_fingerprint:
            raise RuntimeError("Stage-1 best checkpoint was produced by another config.")
        selected_epoch = int(best_payload["epoch"])
        selected_state = best_payload["model_state"]
        selection = "best_validation_ema"

    atomic_torch_save(
        {
            "artifact_schema": "puma",
            "config_fingerprint": config.stage1_fingerprint,
            "epoch": selected_epoch,
            "model_state": selected_state,
            "postprocessing": {
                "threshold": config.stage1_deployment_threshold,
                "local_max_radius": config.stage1_deployment_local_max_radius,
                "suppression_radius": config.stage1_deployment_suppression_radius,
            },
            "architecture": "full_roi_1024_native_stride1_point_detector",
            "runtime_amp_dtype": str(amp_dtype).replace("torch.", ""),
        },
        final_path,
    )
    summary = {
        "run": run_name,
        "maximum_epochs": config.stage1_epochs,
        "completed_epoch": completed_epoch,
        "selected_epoch": selected_epoch,
        "selection": selection,
        "early_stopping_enabled": fold is not None,
        "early_stopping_patience": config.stage1_early_stopping_patience if fold is not None else None,
        "early_stopped": stopped_early,
        "best_validation_binary_f1": best_f1 if best_epoch > 0 else None,
        "final_checkpoint": str(final_path),
        "runtime_amp_dtype": str(amp_dtype).replace("torch.", ""),
        "runtime_micro_batch_size": train_batch_size,
        "runtime_accumulation_steps": runtime_accumulation_steps,
        "last_epoch": history[-1] if history else {},
    }
    _atomic_json(summary_path, summary)
    return summary
