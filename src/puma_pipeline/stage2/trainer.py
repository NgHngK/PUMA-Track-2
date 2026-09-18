from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import PumaConfig
from ..constants import REJECT_CLASS_ID
from ..utils.checkpoint import atomic_torch_save
from ..utils.ema import ModelEMA
from ..utils.provenance import file_signature
from ..utils.runtime import configure_runtime, make_cuda_grad_scaler, resolve_cuda_amp_dtype
from .biomask import biomask_loss
from .cache import build_stage2_cache
from .dataset import CandidateDataset, Stage2Cache, roi_class_sampler
from .model import BioContextRefine
from .spatial import build_knn_graph

OOF_PRED_DTYPE=np.dtype([
    ("source_id","i8"),("roi_index","i4"),("x","f4"),("y","f4"),("class_id","i2"),("true_class_id","i2"),("is_reject","u1"),("fold","i1"),
    ("probabilities","f4",(10,)),("local_probabilities","f4",(10,)),("validity","f4"),("heatmap_score","f4"),("mask_confidence","f4"),("graph_gate","f4")
])



def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _run_seed(config: PumaConfig, fold: int | None) -> int:
    return int(config.seed + (20_011 if fold is None else 1_009 * (fold + 1)))


def _require_training_amp(config: PumaConfig) -> torch.dtype:
    amp = resolve_cuda_amp_dtype(config.use_bfloat16)
    if config.use_bfloat16 and amp == torch.float16:
        print("Stage 2: BF16 is unsupported on this GPU; using FP16 with GradScaler.")
    return amp

def _device_batch(batch:dict[str,torch.Tensor],device:torch.device)->dict[str,torch.Tensor]:
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) and k!="ids" else v) for k,v in batch.items()}


def _focal_binary(logits:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    target=target.float(); prob=logits.sigmoid(); pt=torch.where(target>0.5,prob,1-prob); alpha=torch.where(target>0.5,torch.full_like(target,.65),torch.full_like(target,.35))
    return (-alpha*(1-pt).square()*torch.log(pt.clamp_min(1e-6))).mean()


def _phase_lr(config:PumaConfig,epoch:int,kind:str)->float:
    if epoch<=config.stage2_local_epochs:
        base={"local":config.local_learning_rate,"biomask":config.biomask_learning_rate,"fusion":config.fusion_learning_rate,"graph":0.0,"validity":config.validity_learning_rate}[kind]
        total=config.stage2_local_epochs; local_epoch=epoch
    else:
        base={"local":config.local_learning_rate*config.context_local_lr_multiplier,"biomask":config.biomask_learning_rate*config.context_local_lr_multiplier,"fusion":config.fusion_learning_rate*config.context_local_lr_multiplier,"graph":config.graph_learning_rate,"validity":config.validity_learning_rate}[kind]
        total=config.stage2_epochs-config.stage2_local_epochs; local_epoch=epoch-config.stage2_local_epochs
    if base==0:return 0.0
    warm=min(config.stage2_warmup_epochs,total//3)
    if local_epoch<=warm:return base*local_epoch/max(warm,1)
    progress=(local_epoch-warm)/max(total-warm,1)
    return config.stage2_minimum_learning_rate+0.5*(base-config.stage2_minimum_learning_rate)*(1+math.cos(math.pi*progress))


def _optimizer(model:BioContextRefine,config:PumaConfig)->torch.optim.Optimizer:
    graph=list(model.graph.parameters())
    biomask=list(model.biomask.parameters())
    validity=list(model.validity.parameters())
    fusion=list(model.bio.parameters())+list(model.spatial.parameters())+list(model.tissue.parameters())+list(model.prior.parameters())
    fusion_ids={id(p) for p in fusion}
    special={id(p) for p in graph+biomask+validity}|fusion_ids
    local=[p for p in model.parameters() if id(p) not in special]
    groups=[
        {"params":local,"name":"local","lr":config.local_learning_rate},
        {"params":fusion,"name":"fusion","lr":config.fusion_learning_rate},
        {"params":biomask,"name":"biomask","lr":config.biomask_learning_rate},
        {"params":graph,"name":"graph","lr":0.0},
        {"params":validity,"name":"validity","lr":config.validity_learning_rate},
    ]
    try:
        return torch.optim.AdamW(groups,weight_decay=config.stage2_weight_decay,fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups,weight_decay=config.stage2_weight_decay)


def _set_lrs(opt:torch.optim.Optimizer,config:PumaConfig,epoch:int)->None:
    for group in opt.param_groups: group["lr"]=_phase_lr(config,epoch,str(group["name"]))



def _collate_ids(items:list[int])->np.ndarray:
    return np.asarray(items,dtype=np.int64)


def _loader_kwargs(config:PumaConfig)->dict[str,Any]:
    kwargs:dict[str,Any]={"num_workers":config.workers,"pin_memory":True,"persistent_workers":config.workers>0}
    if config.workers>0:
        kwargs["prefetch_factor"]=config.prefetch_factor
    return kwargs


def _forward(model:BioContextRefine,b:dict[str,torch.Tensor],config:PumaConfig,context_dropout:bool)->dict[str,torch.Tensor]:
    return model.forward_local(b["appearance"],b["rgb"],b["relative"],b["spatial"],b["tissue"],b["prior"],b["detection"],context_dropout=context_dropout)


def _local_loss(model:BioContextRefine,b:dict[str,torch.Tensor],config:PumaConfig)->tuple[torch.Tensor,dict[str,float]]:
    out=_forward(model,b,config,True); labels=b["class_id"].long(); positive=labels<REJECT_CLASS_ID
    semantic=F.cross_entropy(out["local_logits"][positive],labels[positive],label_smoothing=config.label_smoothing) if positive.any() else out["local_logits"].sum()*0
    mask,parts=biomask_loss(out["mask_logits"][positive],out["mask_confidence_logit"][positive],b["mask_target"][positive]) if positive.any() else (out["mask_logits"].sum()*0,{})
    validity=_focal_binary(out["validity_logit"],positive.float())
    total=semantic+config.mask_loss_weight*mask+config.validity_loss_weight*validity
    return total,{"semantic":float(semantic.detach()),"mask":float(mask.detach()),"validity":float(validity.detach())}


def _graph_loss(model:BioContextRefine,b:dict[str,torch.Tensor],config:PumaConfig)->tuple[torch.Tensor,dict[str,float]]:
    local=_forward(model,b,config,True); idx,dist=build_knn_graph(b["xy"].detach().cpu().numpy(),config.graph_neighbor_k,config.graph_radius_cap); idx=torch.from_numpy(idx).to(b["xy"].device); dist=torch.from_numpy(dist).to(b["xy"].device)
    out=model.refine(local,b["xy"],idx,dist,probability_dropout=config.graph_probability_dropout)
    labels=b["class_id"].long(); positive=labels<REJECT_CLASS_ID
    final_ce=F.cross_entropy(out["final_logits"][positive],labels[positive],label_smoothing=config.label_smoothing) if positive.any() else out["final_logits"].sum()*0
    preserve=F.cross_entropy(out["local_logits"][positive],labels[positive],label_smoothing=config.label_smoothing) if positive.any() else final_ce*0
    mask,_=biomask_loss(out["mask_logits"][positive],out["mask_confidence_logit"][positive],b["mask_target"][positive]) if positive.any() else (final_ce*0,{})
    validity=_focal_binary(out["validity_logit"],positive.float())
    total=config.graph_loss_weight*final_ce+0.25*preserve+config.mask_loss_weight*0.25*mask+config.validity_loss_weight*validity
    return total,{"graph":float(final_ce.detach()),"preserve":float(preserve.detach()),"validity":float(validity.detach())}


def train_stage2_fold(config: PumaConfig, fold: int | None, *, resume: bool = True) -> Path:
    if fold is not None and fold not in range(config.number_of_folds):
        raise ValueError(f"fold must be 0..{config.number_of_folds - 1} or None.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 training requires CUDA.")
    run_seed = _run_seed(config, fold)
    configure_runtime(run_seed, config.use_tf32, config.deterministic)
    amp = _require_training_amp(config)

    build_stage2_cache(config)
    cache = Stage2Cache(config.path("cache_dir"), expected_fingerprint=config.fingerprint)
    c = cache.candidates
    train_ids = np.flatnonzero(
        np.ones(len(c), bool) if fold is None else np.asarray(c["fold"], int) != fold
    )
    if not len(train_ids):
        raise RuntimeError("Stage-2 training split is empty.")

    device = torch.device("cuda")
    model = BioContextRefine(config).to(device)
    ema = ModelEMA(model, config.stage2_ema_decay)
    opt = _optimizer(model, config)
    scaler = make_cuda_grad_scaler(amp)
    run_dir = config.path("stage2_output_dir") / ("full" if fold is None else f"fold_{fold}")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    start = 1
    history: list[dict[str, Any]] = []
    if resume and latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config.fingerprint:
            raise RuntimeError("Stage-2 resume checkpoint was produced by another config/feature contract.")
        model.load_state_dict(payload["model_state"], strict=True)
        ema.load_state_dict(payload["ema_state"])
        opt.load_state_dict(payload["optimizer_state"])
        _move_optimizer(opt, device)
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        start = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))

    for epoch in range(start, config.stage2_epochs + 1):
        # Use a fixed seed for each epoch so resume stays reproducible.
        epoch_seed = run_seed + epoch * 1_000_003
        configure_runtime(epoch_seed, config.use_tf32, config.deterministic)
        rng = np.random.default_rng(epoch_seed)
        model.train()
        _set_lrs(opt, config, epoch)
        sums: dict[str, float] = {}
        steps = 0

        if epoch <= config.stage2_local_epochs:
            sampler = roi_class_sampler(
                cache, train_ids, config.roi_class_balanced_fraction, len(train_ids), epoch_seed + 17
            )
            loader = DataLoader(
                CandidateDataset(cache, train_ids),
                batch_size=config.stage2_train_batch_size,
                sampler=sampler,
                collate_fn=_collate_ids,
                **_loader_kwargs(config),
            )
            for ids in loader:
                variants = rng.integers(0, cache.variants, size=len(ids), dtype=np.int64)
                batch = _device_batch(cache.batch(ids, variants), device)
                with torch.autocast("cuda", dtype=amp):
                    loss, parts = _local_loss(model, batch, config)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage2_gradient_clip_norm)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                for key, value in parts.items():
                    sums[key] = sums.get(key, 0.0) + value
                steps += 1
        else:
            real = train_ids[np.asarray(c["kind"][train_ids]) == 0]
            if not len(real):
                raise RuntimeError("Stage-2 graph phase has no detector-derived training candidates.")
            real_roi = np.asarray(c["roi_index"][real], int)
            rois = np.unique(real_roi)
            rng.shuffle(rois)
            roi_batch = int(config.stage2_roi_batch_size)
            for group_start in range(0, len(rois), roi_batch):
                group = rois[group_start:group_start + roi_batch]
                opt.zero_grad(set_to_none=True)
                for roi in group:
                    ids = real[real_roi == int(roi)]
                    variants = np.full(len(ids), int(rng.integers(0, cache.variants)), np.int64)
                    batch = _device_batch(cache.batch(ids, variants), device)
                    with torch.autocast("cuda", dtype=amp):
                        loss, parts = _graph_loss(model, batch, config)
                        scaled = loss / max(len(group), 1)
                    scaler.scale(scaled).backward()
                    for key, value in parts.items():
                        sums[key] = sums.get(key, 0.0) + value
                    steps += 1
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage2_gradient_clip_norm)
                scaler.step(opt)
                scaler.update()
                ema.update(model)

        record = {
            "epoch": epoch,
            "phase": "local" if epoch <= config.stage2_local_epochs else "graph",
            **{key: value / max(steps, 1) for key, value in sums.items()},
        }
        history = [row for row in history if int(row.get("epoch", -1)) != epoch]
        history.append(record)
        history.sort(key=lambda row: int(row["epoch"]))
        atomic_torch_save(
            {
                "epoch": epoch,
                "config_fingerprint": config.fingerprint,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
            },
            latest,
        )

    final = run_dir / "stage2_final_ema.pt"
    atomic_torch_save(
        {
            "artifact_schema": "puma",
            "epoch": config.stage2_epochs,
            "config_fingerprint": config.fingerprint,
            "model_state": ema.state_dict(),
            "architecture": "BioContextRefine",
            "checkpoint_policy": "fixed_epoch_100_ema_no_held_fold_selection",
            "runtime_amp_dtype": str(amp).replace("torch.", ""),
        },
        final,
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return final


def _load_model(config: PumaConfig, path: Path, device: torch.device) -> BioContextRefine:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_fingerprint") != config.fingerprint or payload.get("architecture") != "BioContextRefine":
        raise RuntimeError(f"Incompatible Stage-2 checkpoint: {path}")
    model = BioContextRefine(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_cached_rois(config:PumaConfig,model:BioContextRefine,roi_indices:np.ndarray,cache:Stage2Cache)->np.ndarray:
    c=cache.candidates; rows=[]; device=next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Stage-2 cached prediction requires a CUDA model.")
    amp = _require_training_amp(config)
    for roi in map(int,roi_indices):
        ids=np.flatnonzero((np.asarray(c["roi_index"],int)==roi)&(np.asarray(c["kind"],int)==0))
        if not len(ids):continue
        b=_device_batch(cache.batch(ids,0),device)
        with torch.autocast("cuda",dtype=amp): local=_forward(model,b,config,False); ni,nd=build_knn_graph(b["xy"].cpu().numpy(),config.graph_neighbor_k,config.graph_radius_cap); out=model.refine(local,b["xy"],torch.from_numpy(ni).to(device),torch.from_numpy(nd).to(device),probability_temperature=config.graph_probability_temperature)
        prob=out["final_logits"].softmax(1).float().cpu().numpy(); local_prob=out["local_logits"].softmax(1).float().cpu().numpy(); pred=prob.argmax(1); validity=out["validity_logit"].sigmoid().float().cpu().numpy(); mask_conf=out["mask_confidence_logit"].sigmoid().float().cpu().numpy(); graph=out["graph_gate"].float().cpu().numpy()
        for j,row_id in enumerate(ids):
            row=c[row_id]; rows.append((int(row["source_id"]),roi,float(row["x"]),float(row["y"]),int(pred[j]),int(row["class_id"]),int(row["class_id"]==REJECT_CLASS_ID),int(row["fold"]),prob[j],local_prob[j],float(validity[j]),float(row["heatmap_score"]),float(mask_conf[j]),float(graph[j])))
    return np.asarray(rows,dtype=OOF_PRED_DTYPE)


def _stage2_fold_checkpoint(config: PumaConfig, fold: int) -> Path:
    return config.path("stage2_output_dir") / f"fold_{fold}" / "stage2_final_ema.pt"


def _stage2_oof_metadata(config: PumaConfig) -> dict[str, Any]:
    checkpoint_signatures: dict[str, dict[str, Any]] = {}
    for fold in range(config.number_of_folds):
        checkpoint = _stage2_fold_checkpoint(config, fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_signatures[str(fold)] = file_signature(checkpoint)
    cache_manifest = config.path("cache_dir") / "cache_manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(cache_manifest)
    return {
        "artifact_schema": "puma",
        "config_fingerprint": config.fingerprint,
        "cache_manifest_signature": file_signature(cache_manifest),
        "checkpoint_signatures": checkpoint_signatures,
        "dtype": repr(OOF_PRED_DTYPE.descr),
    }


def generate_stage2_oof(config: PumaConfig, *, force: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 OOF generation requires CUDA.")
    _require_training_amp(config)
    output_dir = config.path("stage2_output_dir")
    out = output_dir / "stage2_oof_predictions.npy"
    metadata_path = output_dir / "stage2_oof_predictions.json"
    if out.is_file() and metadata_path.is_file() and not force:
        try:
            expected = _stage2_oof_metadata(config)
        except FileNotFoundError:
            expected = None
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        if expected is not None and all(current.get(key) == value for key, value in expected.items()):
            existing = np.load(out, mmap_mode="r", allow_pickle=False)
            if existing.dtype == OOF_PRED_DTYPE and existing.ndim == 1:
                return out
        raise RuntimeError("Existing Stage-2 OOF predictions are stale or incompatible; rebuild with --force.")

    build_stage2_cache(config)
    cache = Stage2Cache(config.path("cache_dir"), expected_fingerprint=config.fingerprint)
    device = torch.device("cuda")
    all_rows = []
    for fold in range(config.number_of_folds):
        checkpoint = train_stage2_fold(config, fold, resume=True)
        model = _load_model(config, checkpoint, device)
        rois = np.unique(
            np.asarray(
                cache.candidates["roi_index"][np.asarray(cache.candidates["fold"], int) == fold],
                int,
            )
        )
        all_rows.append(predict_cached_rois(config, model, rois, cache))
        del model
        torch.cuda.empty_cache()
    rows = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=OOF_PRED_DTYPE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, rows, allow_pickle=False)
    metadata = _stage2_oof_metadata(config)
    metadata.update({"rows": int(len(rows))})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out
