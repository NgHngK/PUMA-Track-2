from __future__ import annotations

import csv
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .constants import CLASSES
from .datasets import RawNucleiDataset,CachedNucleiDataset
from .losses import PriorAdjustedCrossEntropy
from .manifest import read_manifest,split_rows,row_identity_sha256
from .models.stage2 import Stage2FromCachedCLS,Stage2FromTokens
from .models.uni2 import UNI2Backbone,checkpoint_sha256
from .sampling import build_sampler
from .tier_a import TierANormalizer
from .training import make_optimizer,make_scheduler,train_one_epoch,evaluate,model_parameter_norms
from .utils import atomic_json_dump,count_trainable_parameters,seed_all,sha256_file,stable_json_hash


def _verify_cache_sidecar(path: str | Path,manifest_hash: str,kind: str,rows: list[dict] | None=None) -> dict:
    meta_path=Path(path).with_suffix(".json")
    if not meta_path.is_file():raise FileNotFoundError(f"{kind} cache metadata missing: {meta_path}")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("row_identity_sha256") is not None and rows is not None:
        if meta.get("row_identity_sha256")!=row_identity_sha256(rows):raise ValueError(f"{kind} cache/row-identity hash mismatch")
    elif meta.get("manifest_sha256")!=manifest_hash:
        raise ValueError(f"{kind} cache/manifest hash mismatch")
    return meta

def _load_tier_a(path: str | None,n: int,manifest_hash: str,rows: list[dict] | None=None) -> np.ndarray | None:
    if path is None:return None
    _verify_cache_sidecar(path,manifest_hash,"Tier-A",rows)
    a=np.load(path,mmap_mode="r")
    if a.shape!=(n,16):raise ValueError(f"Tier-A cache shape {a.shape}, expected {(n,16)}")
    return np.asarray(a,dtype=np.float32)


def _split_indices(rows: list[dict],name: str) -> np.ndarray:
    name="dev" if name=="val" else name
    return np.array([i for i,r in enumerate(rows) if r["split"]==name],dtype=int)


def _dataset_for_indices(cfg: ExperimentConfig,rows: list[dict],ix: np.ndarray,tier: np.ndarray | None,train: bool,
                         tier_normalizer: tuple[np.ndarray,np.ndarray] | None=None,augmentation_seed: int | None=None):
    rs=[rows[i] for i in ix];bio=None if tier is None else tier[ix]
    if cfg.uni2_weights is not None:
        aug=cfg.augmentation if train else type(cfg.augmentation)()
        return RawNucleiDataset(rs,bio,cfg.fov,aug,augmentation_seed=cfg.seed if augmentation_seed is None else augmentation_seed,
                                tier_a_normalizer=tier_normalizer if train else None)
    path=cfg.cached_features if cfg.cached_features is not None else cfg.cached_tokens
    meta=_verify_cache_sidecar(path,sha256_file(cfg.manifest),"representation",rows)
    rep=np.load(path,mmap_mode="r")
    if len(rep)!=len(rows):raise ValueError("cached representation length does not match manifest")
    if meta.get("shape") is not None and list(rep.shape)!=list(meta["shape"]):raise ValueError("representation cache shape disagrees with sidecar")
    expected_kind="cls" if cfg.cached_features is not None else "tokens"
    if meta.get("kind") is not None and meta["kind"]!=expected_kind:raise ValueError("representation cache kind disagrees with config")
    if meta.get("dtype") is not None and str(rep.dtype)!=str(meta["dtype"]):raise ValueError("representation cache dtype disagrees with sidecar")
    return CachedNucleiDataset(rs,rep,bio,cfg.fov,representation_indices=ix,validate_finite=not bool(meta.get("finite_verified",False)))


def _build_model(cfg: ExperimentConfig,device: torch.device):
    encoder=None
    if cfg.cached_features is not None:model=Stage2FromCachedCLS(cfg.model)
    else:model=Stage2FromTokens(cfg.model)
    if cfg.uni2_weights is not None:encoder=UNI2Backbone(cfg.uni2_weights,cfg.encoder.adaptation,cfg.encoder.lora_rank,cfg.encoder.lora_alpha,cfg.encoder.last_blocks)
    return model.to(device),None if encoder is None else encoder.to(device)


def _selection_value(metrics: dict,selection_metric: str) -> float:
    if selection_metric=="macro_f1":return float(metrics["macro_f1"])
    try:return float(metrics["puma"]["fixed10"]["macro_f1"])
    except Exception as e:raise KeyError("roi_macro_f1 requested but V17 fixed10 macro_f1 is unavailable") from e


def _make_loader(ds,labels,cfg: ExperimentConfig,train: bool,seed: int,sampler_cfg=None,batch_size: int | None=None):
    sampler=None;q=None;weights=None
    if train:
        sampler,q,weights=build_sampler(torch.as_tensor(labels,dtype=torch.long),sampler_cfg or cfg.sampler,seed)
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds,batch_size=batch_size or cfg.optimizer.batch_size,shuffle=train and sampler is None,sampler=sampler,
                      num_workers=cfg.num_workers,pin_memory=cfg.device.startswith("cuda"),generator=g,persistent_workers=cfg.num_workers>0)
    return loader,q,weights


def _save_checkpoint(path: Path,model,encoder,cfg,normalizer,epoch,score,artifact_hashes: dict):
    state={"model":model.state_dict(),"config":cfg.to_dict(),"tier_a_normalizer":normalizer,"epoch":epoch,"score":score,"classes":list(CLASSES),
           "artifact_hashes":{k:artifact_hashes.get(k) for k in ("manifest_sha256","tier_a_sha256","representation_sha256","uni2_sha256")}}
    if encoder is not None:
        trainable={n:p.detach().cpu() for n,p in encoder.state_dict().items() if n in {k for k,v in encoder.named_parameters() if v.requires_grad}}
        if trainable:state["encoder_trainable"]=trainable
    torch.save(state,path)


def run_experiment(cfg: ExperimentConfig) -> dict:
    cfg.validate();seed_all(cfg.seed);out=Path(cfg.output_dir)/cfg.experiment_id
    if (out/"summary.json").exists():
        stored_cfg_path=out/"config.json"
        if not stored_cfg_path.is_file():raise RuntimeError(f"completed run is missing config.json: {out}")
        stored_cfg=json.loads(stored_cfg_path.read_text(encoding="utf-8"))
        if stable_json_hash(stored_cfg)!=stable_json_hash(cfg.to_dict()):
            raise ValueError(f"experiment_id already exists with a different config; use a new immutable experiment_id: {cfg.experiment_id}")
        return json.loads((out/"summary.json").read_text(encoding="utf-8"))
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"partial run preserved: {out}")
    out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir();(out/"predictions").mkdir()
    rows=read_manifest(cfg.manifest,require_files=cfg.uni2_weights is not None);train_ix=_split_indices(rows,"train");dev_ix=_split_indices(rows,"dev")
    if len(train_ix)==0 or len(dev_ix)==0:raise ValueError("manifest must contain non-empty train and dev splits")
    if cfg.model.use_tier_a:
        manifest_hash=sha256_file(cfg.manifest);tier_raw=_load_tier_a(cfg.tier_a,len(rows),manifest_hash,rows);norm=TierANormalizer().fit(tier_raw[train_ix]);tier=norm.transform(tier_raw);norm_state=norm.state_dict();norm_tuple=(norm.mean.copy(),norm.std.copy())
    else:tier=None;norm_state={"disabled":True};norm_tuple=None
    train_ds=_dataset_for_indices(cfg,rows,train_ix,tier,True,norm_tuple,cfg.seed+100_000)
    train_eval_ds=_dataset_for_indices(cfg,rows,train_ix,tier,False,None,cfg.seed+200_000)
    dev_ds=_dataset_for_indices(cfg,rows,dev_ix,tier,False,None,cfg.seed+300_000)
    train_labels=[int(rows[i]["label"]) for i in train_ix];dev_rows=[rows[i] for i in dev_ix];train_rows=[rows[i] for i in train_ix]
    train_loader,q,_=_make_loader(train_ds,train_labels,cfg,True,cfg.seed);train_eval_loader,_,_=_make_loader(train_eval_ds,train_labels,cfg,False,cfg.seed);dev_loader,_,_=_make_loader(dev_ds,[int(r["label"]) for r in dev_rows],cfg,False,cfg.seed)
    if q is None:
        counts=torch.bincount(torch.tensor(train_labels),minlength=10).float();q=counts/counts.sum()
    criterion=PriorAdjustedCrossEntropy(q,cfg.loss)
    device=torch.device(cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    model,encoder=_build_model(cfg,device);initial_trainable=count_trainable_parameters(model)+(0 if encoder is None else count_trainable_parameters(encoder));optimizer=make_optimizer(model,cfg,encoder);optimizer_steps_per_epoch=max(1,int(np.ceil(len(train_loader)/cfg.optimizer.accumulation_steps)))
    total_steps=cfg.optimizer.max_epochs*optimizer_steps_per_epoch;scheduler=make_scheduler(optimizer,total_steps,cfg)
    atomic_json_dump(out/"config.json",cfg.to_dict());atomic_json_dump(out/"tier_a_normalizer.json",norm_state)
    provenance={"manifest_sha256":sha256_file(cfg.manifest),"tier_a_sha256":sha256_file(cfg.tier_a) if cfg.tier_a else None,
                "representation_sha256":sha256_file(cfg.cached_features or cfg.cached_tokens) if (cfg.cached_features or cfg.cached_tokens) else None,
                "uni2_sha256":checkpoint_sha256(cfg.uni2_weights) if cfg.uni2_weights else None,"config_hash":stable_json_hash(cfg.to_dict()),
                "python":platform.python_version(),"torch":torch.__version__,"device":str(device)};atomic_json_dump(out/"provenance.json",provenance)
    best=-float("inf");stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,cfg.optimizer.max_epochs+1):
        er=train_one_epoch(model,train_loader,optimizer,criterion,device,cfg.optimizer.grad_clip,scheduler,encoder,cfg.optimizer.amp,cfg.optimizer.accumulation_steps)
        tm,_,_,_=evaluate(model,train_eval_loader,train_rows,device,encoder);dm,y,z,ix=evaluate(model,dev_loader,dev_rows,device,encoder)
        score=_selection_value(dm,cfg.selection_metric)
        if not np.isfinite(score):raise FloatingPointError("nonfinite development selection score")
        rec={"epoch":epoch,"objective":er.loss,"train":tm,"dev":dm,"selection_score":score,
            "exposure":er.exposure,"unique_examples":er.unique_examples,"repeat_fraction":er.repeat_fraction,
            "unique_examples_by_class":er.unique_examples_by_class,"repeat_fraction_by_class":er.repeat_fraction_by_class,"unique_groups":er.unique_groups,"grad_norm_mean":er.grad_norm_mean,
            "generalization_gap_macro_f1":tm["macro_f1"]-dm["macro_f1"],"generalization_gap_loss":dm["nll"]-tm["nll"],
            "parameter_norms":model_parameter_norms(model),
            "lr":[float(g["lr"]) for g in optimizer.param_groups]};history.append(rec)
        with (out/"history.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(rec)+"\n")
        if score>best+cfg.optimizer.min_delta:
            best=score;stale=0;_save_checkpoint(out/"checkpoints/best.pt",model,encoder,cfg,norm_state,epoch,score,provenance)
            np.savez_compressed(out/"predictions/best_dev.npz",labels=y,logits=z,indices=ix,uids=np.array([dev_rows[int(i)]["uid"] for i in ix]))
        else:stale+=1
        if cfg.optimizer.selection_mode=="early_stop" and stale>=cfg.optimizer.patience:break
    # Preserve final endpoint independently from the development-best checkpoint.
    final_rec=history[-1];_save_checkpoint(out/"checkpoints/final.pt",model,encoder,cfg,norm_state,final_rec["epoch"],final_rec["selection_score"],provenance)
    np.savez_compressed(out/"predictions/final_dev.npz",labels=y,logits=z,indices=ix,uids=np.array([dev_rows[int(i)]["uid"] for i in ix]))
    selected_path=out/("checkpoints/final.pt" if cfg.optimizer.selection_mode=="fixed_final" else "checkpoints/best.pt")
    ck=torch.load(selected_path,map_location=device,weights_only=True);model.load_state_dict(ck["model"],strict=True)
    if encoder is not None and ck.get("encoder_trainable"):
        current=encoder.state_dict();current.update(ck["encoder_trainable"]);encoder.load_state_dict(current,strict=True)
    decoupled_summary=None
    if cfg.decoupled.enabled:
        if not hasattr(model,"head"):raise RuntimeError("unexpected model wrapper")
        if hasattr(model,"pooler"):model.pooler.requires_grad_(False)
        model.head.freeze_feature_transform();
        if cfg.decoupled.reset_classifier:model.head.reset_classifier_outputs()
        classifier_params=set(model.head.classifier_parameters())
        for p in model.parameters():
            if p.requires_grad and p not in classifier_params:p.requires_grad_(False)
        if encoder is not None:encoder.requires_grad_(False)
        # cRT isolates classifier rebalancing. Use the deterministic TRAIN-eval
        # view rather than the stochastic augmented Stage-A dataset, so Stage B
        # changes classifier sampling/loss only and cannot hide an augmentation
        # change inside the decoupled ablation.
        dloader,dq,_=_make_loader(train_eval_ds,train_labels,cfg,True,cfg.seed+7000,cfg.decoupled.sampler)
        if dq is None:
            counts=torch.bincount(torch.tensor(train_labels),minlength=10).float();dq=counts/counts.sum()
        dcrit=PriorAdjustedCrossEntropy(dq,cfg.decoupled.loss)
        decay=[];no_decay=[]
        for p in model.head.classifier_parameters():
            if cfg.optimizer.decay_bias_and_vectors or p.ndim>1:decay.append(p)
            else:no_decay.append(p)
        groups=[]
        if decay:groups.append({"params":decay,"weight_decay":cfg.optimizer.weight_decay})
        if no_decay:groups.append({"params":no_decay,"weight_decay":0.0})
        dopt=torch.optim.AdamW(groups,lr=cfg.decoupled.retrain_lr,betas=(.9,.999),eps=1e-8)
        drows=[]
        for de in range(1,cfg.decoupled.retrain_epochs+1):
            er=train_one_epoch(model,dloader,dopt,dcrit,device,cfg.optimizer.grad_clip,None,encoder,cfg.optimizer.amp,cfg.optimizer.accumulation_steps)
            dtm,_,_,_=evaluate(model,train_eval_loader,train_rows,device,encoder)
            dm,y,z,ix=evaluate(model,dev_loader,dev_rows,device,encoder)
            drows.append({
                "epoch":de,"train":dtm,"dev":dm,"objective":er.loss,
                "exposure":er.exposure,"unique_examples":er.unique_examples,"repeat_fraction":er.repeat_fraction,
                "unique_examples_by_class":er.unique_examples_by_class,"repeat_fraction_by_class":er.repeat_fraction_by_class,"unique_groups":er.unique_groups,
                "grad_norm_mean":er.grad_norm_mean,
                "generalization_gap_macro_f1":dtm["macro_f1"]-dm["macro_f1"],
                "generalization_gap_loss":dm["nll"]-dtm["nll"],
                "parameter_norms":model_parameter_norms(model),
            })
        dscore=_selection_value(drows[-1]["dev"],cfg.selection_metric);decoupled_summary={"epochs":drows,"final_score":dscore,"selected_checkpoint":"checkpoints/decoupled_final.pt"}
        _save_checkpoint(out/"checkpoints/decoupled_final.pt",model,encoder,cfg,norm_state,cfg.decoupled.retrain_epochs,dscore,provenance)
    final_checkpoint="checkpoints/decoupled_final.pt" if cfg.decoupled.enabled else str(selected_path.relative_to(out))
    final_score=float(decoupled_summary["final_score"]) if decoupled_summary is not None else float(ck["score"])
    summary={"experiment_id":cfg.experiment_id,"best_score":best,"best_diagnostic_epoch":int(max(history,key=lambda r:r["selection_score"])["epoch"]),
             "selected_score":float(ck["score"]),"selected_epoch":int(ck["epoch"]),"selection_mode":cfg.optimizer.selection_mode,"epochs_run":len(history),
             "trainable_parameters":initial_trainable,"final_stage_trainable_parameters":count_trainable_parameters(model)+(0 if encoder is None else count_trainable_parameters(encoder)),
             "runtime_seconds":time.perf_counter()-start,"selected":history[int(ck["epoch"])-1],"decoupled":decoupled_summary,
             "final_model_checkpoint":final_checkpoint,"final_model_score":final_score}
    atomic_json_dump(out/"summary.json",summary)
    print(json.dumps({"run":cfg.experiment_id,"status":"complete","score":round(final_score,6),"epochs":len(history),"summary":str(out/"summary.json")},separators=(",",":")),flush=True)
    return summary


def evaluate_checkpoint(cfg: ExperimentConfig,checkpoint_path: str | Path,split: str="locked") -> dict:
    """Evaluate a frozen selected checkpoint without refitting normalization or selection.

    This function is intended for the one-shot Exploration-6 locked confirmation. Tier-A
    normalization is loaded from the checkpoint, never fitted on the target split.
    """
    cfg.validate();rows=read_manifest(cfg.manifest,require_files=cfg.uni2_weights is not None);ix=_split_indices(rows,split)
    if len(ix)==0:raise ValueError(f"empty evaluation split: {split}")
    ck=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if tuple(ck.get("classes",()))!=CLASSES:raise ValueError("checkpoint class ontology mismatch")
    if stable_json_hash(ck["config"])!=stable_json_hash(cfg.to_dict()):raise ValueError("checkpoint/config mismatch")
    saved_hashes=ck.get("artifact_hashes")
    if not isinstance(saved_hashes,dict):raise ValueError("checkpoint lacks bound artifact hashes")
    current_hashes={
        "manifest_sha256":sha256_file(cfg.manifest),
        "tier_a_sha256":sha256_file(cfg.tier_a) if cfg.tier_a else None,
        "representation_sha256":sha256_file(cfg.cached_features or cfg.cached_tokens) if (cfg.cached_features or cfg.cached_tokens) else None,
        "uni2_sha256":checkpoint_sha256(cfg.uni2_weights) if cfg.uni2_weights else None,
    }
    if any(saved_hashes.get(k)!=v for k,v in current_hashes.items()):
        raise ValueError("checkpoint artifact hash mismatch; manifest/cache/Tier-A/UNI2 changed after training")
    if cfg.model.use_tier_a:
        manifest_hash=sha256_file(cfg.manifest);raw=_load_tier_a(cfg.tier_a,len(rows),manifest_hash,rows);state=ck["tier_a_normalizer"]
        norm=TierANormalizer(np.asarray(state["mean"],dtype=np.float32),np.asarray(state["std"],dtype=np.float32));tier=norm.transform(raw)
    else:tier=None
    ds=_dataset_for_indices(cfg,rows,ix,tier,False);target_rows=[rows[i] for i in ix]
    loader,_,_=_make_loader(ds,[int(r["label"]) for r in target_rows],cfg,False,cfg.seed)
    device=torch.device(cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    model,encoder=_build_model(cfg,device);model.load_state_dict(ck["model"],strict=True)
    if encoder is not None and ck.get("encoder_trainable"):
        current=encoder.state_dict();current.update(ck["encoder_trainable"]);encoder.load_state_dict(current,strict=True)
    metrics,y,z,local_ix=evaluate(model,loader,target_rows,device,encoder)
    return {"split":split,"metrics":metrics,"labels":y,"logits":z,"indices":local_ix,"uids":[target_rows[int(i)]["uid"] for i in local_ix]}
