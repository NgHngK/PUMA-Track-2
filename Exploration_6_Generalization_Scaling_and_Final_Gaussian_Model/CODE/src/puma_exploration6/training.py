from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset

from .config import ExperimentConfig, SamplerConfig, LossConfig
from .constants import NUM_CLASSES
from .losses import PriorAdjustedCrossEntropy
from .metrics import combined_metrics
from .sampling import build_sampler


@dataclass
class EpochResult:
    loss: float
    exposure: list[int]
    unique_examples: int
    repeat_fraction: float
    unique_examples_by_class: list[int]
    repeat_fraction_by_class: list[float]
    unique_groups: int
    grad_norm_mean: float


def make_optimizer(model: nn.Module, cfg: ExperimentConfig, encoder: nn.Module | None = None) -> torch.optim.Optimizer:
    groups=[]
    def append_module(module: nn.Module, lr: float):
        decay=[];no_decay=[]
        for _,p in module.named_parameters():
            if not p.requires_grad:continue
            if cfg.optimizer.decay_bias_and_vectors or p.ndim>1:decay.append(p)
            else:no_decay.append(p)
        if decay:groups.append({"params":decay,"weight_decay":cfg.optimizer.weight_decay,"lr":lr})
        if no_decay:groups.append({"params":no_decay,"weight_decay":0.0,"lr":lr})
    append_module(model,cfg.optimizer.lr)
    if encoder is not None:append_module(encoder,cfg.encoder.adapter_lr)
    if not groups:raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)


def make_scheduler(optimizer,total_steps: int,cfg: ExperimentConfig):
    kind=cfg.optimizer.scheduler
    if kind=="constant":return None
    warmup=round(cfg.optimizer.warmup_fraction*total_steps) if kind=="warmup_cosine" else 0
    min_ratio=cfg.optimizer.min_lr_ratio
    def factor(step: int):
        if warmup and step<warmup:return max(1e-8,(step+1)/warmup)
        progress=(step-warmup)/max(1,total_steps-warmup);progress=min(max(progress,0),1)
        return min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def _forward(model,batch,device,encoder=None):
    bio=batch["bio"].to(device,non_blocking=True)
    if "features" in batch:return model(batch["features"].to(device,non_blocking=True),bio)
    if "tokens" in batch:return model(batch["tokens"].to(device,non_blocking=True),bio,batch["u"].to(device,non_blocking=True),batch["v"].to(device,non_blocking=True))
    if encoder is None:raise ValueError("raw-image batch requires encoder")
    tokens=encoder.forward_tokens(batch["image"].to(device,non_blocking=True))
    return model(tokens,bio,batch["u"].to(device,non_blocking=True),batch["v"].to(device,non_blocking=True))


def train_one_epoch(model,loader,optimizer,criterion,device,grad_clip: float,scheduler=None,encoder=None,amp: bool=False,
                    accumulation_steps: int=1) -> EpochResult:
    """Train one epoch with sample-normalized gradient accumulation.

    Losses are backpropagated as *sums* over each physical microbatch. Immediately
    before every optimizer step, accumulated gradients are divided by the exact
    number of samples in that accumulation window. Therefore a partial final window
    is normalized correctly and matches a single effective batch up to floating-point
    operation ordering. Scheduler steps occur once per successful optimizer update.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >=1")
    model.train()
    if encoder is not None:
        # A truly frozen backbone must remain in eval mode so stochastic training
        # layers cannot silently alter the frozen-feature contract. Q/V LoRA itself
        # has no dropout; UNI2Backbone.train keeps the pretrained encoder in eval.
        if any(p.requires_grad for p in encoder.parameters()): encoder.train()
        else: encoder.eval()
    total=0.0;seen=0;exposure=torch.zeros(NUM_CLASSES,dtype=torch.long);seen_indices=set();seen_by_class=[set() for _ in range(NUM_CLASSES)];seen_groups=set();grad_norms=[]
    use_amp=bool(amp and device.type=="cuda");dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler=torch.amp.GradScaler("cuda",enabled=use_amp and dtype==torch.float16)
    else:  # PyTorch 2.1/2.2 compatibility
        scaler=torch.cuda.amp.GradScaler(enabled=use_amp and dtype==torch.float16)
    trainable=[p for p in list(model.parameters())+(list(encoder.parameters()) if encoder is not None else []) if p.requires_grad]
    if not trainable: raise ValueError("no trainable parameters")
    optimizer.zero_grad(set_to_none=True)
    window_samples=0
    micro_in_window=0
    loader_len=len(loader)
    for batch_i,batch in enumerate(loader):
        y=batch["label"].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
            z=_forward(model,batch,device,encoder);loss_sum=criterion(z,y,reduction="sum")
        if not torch.isfinite(loss_sum):raise FloatingPointError("nonfinite training loss")
        scaler.scale(loss_sum).backward()
        n=len(y);window_samples+=n;micro_in_window+=1
        total+=float(loss_sum.detach());seen+=n;exposure+=torch.bincount(y.detach().cpu(),minlength=NUM_CLASSES)
        local_indices=[int(i) for i in batch["index"].detach().cpu().tolist()];local_labels=[int(v) for v in y.detach().cpu().tolist()]
        seen_indices.update(local_indices)
        for ii,cc in zip(local_indices,local_labels):seen_by_class[cc].add(ii)
        if "group" in batch:seen_groups.update(str(g) for g in batch["group"] if str(g))

        boundary = micro_in_window >= accumulation_steps or batch_i == loader_len-1
        if boundary:
            if window_samples <= 0: raise RuntimeError("empty accumulation window")
            scaler.unscale_(optimizer)
            # Convert gradients of summed per-example losses to the exact mean over
            # this optimizer window, including a partial final accumulation window.
            inv=1.0/float(window_samples)
            for p in trainable:
                if p.grad is not None: p.grad.mul_(inv)
            norm=torch.nn.utils.clip_grad_norm_(trainable,grad_clip,error_if_nonfinite=True);grad_norms.append(float(norm))
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scheduler is not None and scaler.get_scale()>=old:scheduler.step()
            optimizer.zero_grad(set_to_none=True);window_samples=0;micro_in_window=0
    if seen==0:raise ValueError("empty training loader")
    if window_samples != 0 or micro_in_window != 0: raise RuntimeError("unflushed accumulation window")
    exp=exposure.tolist();unique_by=[len(x) for x in seen_by_class];repeat_by=[0.0 if exp[c]==0 else 1-unique_by[c]/exp[c] for c in range(NUM_CLASSES)]
    return EpochResult(total/seen,exp,len(seen_indices),1-len(seen_indices)/seen,unique_by,repeat_by,len(seen_groups),float(np.mean(grad_norms)) if grad_norms else 0.0)


@torch.inference_mode()
def evaluate(model,loader,rows,device,encoder=None) -> tuple[dict,np.ndarray,np.ndarray,np.ndarray]:
    model.eval();
    if encoder is not None:encoder.eval()
    ys=[];zs=[];ids=[];diag_sum={};diag_count={}
    for batch in loader:
        z=_forward(model,batch,device,encoder);ys.append(batch["label"].cpu());zs.append(z.float().cpu());ids.append(batch["index"].cpu())
        head=getattr(model,"head",None);diag=getattr(head,"diagnostics",{}) if head is not None else {}
        for key,value in diag.items():
            vv=torch.as_tensor(value).detach().float()
            if vv.numel()==0:continue
            diag_sum[key]=diag_sum.get(key,0.0)+float(vv.sum().cpu())
            diag_count[key]=diag_count.get(key,0)+int(vv.numel())
    if not ys:raise ValueError("empty evaluation loader")
    y=torch.cat(ys).numpy();z=torch.cat(zs).numpy();ix=torch.cat(ids).numpy()
    ordered=[rows[int(i)] for i in ix];metrics=combined_metrics(ordered,y,z)
    if diag_sum:metrics["model_diagnostics"]={k:diag_sum[k]/diag_count[k] for k in sorted(diag_sum)}
    return metrics,y,z,ix


def model_parameter_norms(model: nn.Module) -> dict[str,float]:
    """Compact A5/pooler parameter diagnostics without changing model state."""
    head=getattr(model,"head",None)
    if head is None:return {}
    out={
        "W_weight_norm":float(head.head.weight.detach().float().norm().cpu()),
        "W_bias_norm":float(head.head.bias.detach().float().norm().cpu()),
    }
    if getattr(head,"ph",None) is not None:
        out.update(
            Ph_weight_norm=float(head.ph.weight.detach().float().norm().cpu()),
            Pb_weight_norm=float(head.pb.weight.detach().float().norm().cpu()),
            R_weight_norm=float(head.branch.weight.detach().float().norm().cpu()),
        )
    pooler=getattr(model,"pooler",None)
    mix=getattr(pooler,"mix_logit",None) if pooler is not None else None
    if mix is not None:out["global_local_alpha"]=float(mix.detach().sigmoid().cpu())
    return out
