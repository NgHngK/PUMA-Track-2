from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RISK_BASE_DIM=8
RISK_DIM=RISK_BASE_DIM+10


def risk_features(prob: np.ndarray, validity: np.ndarray, heatmap_score: np.ndarray, mask_conf: np.ndarray, graph_gate: np.ndarray, local_prob: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(prob, np.float32)
    if p.ndim != 2 or p.shape[1] != 10:
        raise ValueError(f"Expected probabilities [N,10], got {p.shape}.")
    n = len(p)
    vectors = {
        "validity": np.asarray(validity, np.float32).reshape(-1),
        "heatmap_score": np.asarray(heatmap_score, np.float32).reshape(-1),
        "mask_conf": np.asarray(mask_conf, np.float32).reshape(-1),
        "graph_gate": np.asarray(graph_gate, np.float32).reshape(-1),
    }
    if any(len(v) != n for v in vectors.values()):
        raise ValueError("Risk feature vectors must all have the same number of rows as probabilities.")
    if not np.isfinite(p).all() or any(not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Risk features contain non-finite values.")
    local = None if local_prob is None else np.asarray(local_prob, np.float32)
    if local is not None and (local.shape != p.shape or not np.isfinite(local).all()):
        raise ValueError(f"local_prob must match probability shape {p.shape} and be finite.")
    top=np.partition(p,-2,axis=1)[:,-2:]; maxp=p.max(1); margin=top[:,1]-top[:,0]
    entropy=-(p*np.log(np.clip(p,1e-7,1))).sum(1)/math.log(p.shape[1]); predicted=p.argmax(1); onehot=np.eye(10,dtype=np.float32)[predicted]
    disagreement=(np.abs(p-local).mean(1) if local is not None else 1.0-maxp)
    base=np.column_stack((maxp,margin,1-entropy,vectors["validity"],vectors["heatmap_score"],vectors["mask_conf"],vectors["graph_gate"],disagreement))
    return np.concatenate((base,onehot),axis=1).astype(np.float32)



class RiskCalibrator(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.linear=nn.Linear(RISK_DIM,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.linear(x).squeeze(1)


def fit_risk(x:np.ndarray,y:np.ndarray,steps:int=600,lr:float=.03,weight_decay:float=1e-3,seed:int=2026)->dict[str,np.ndarray|float]:
    torch.manual_seed(seed); model=RiskCalibrator(); xx=torch.from_numpy(np.asarray(x,np.float32)); yy=torch.from_numpy(np.asarray(y,np.float32))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    for _ in range(steps):
        logits=model(xx); loss=F.binary_cross_entropy_with_logits(logits,yy); opt.zero_grad(); loss.backward(); opt.step()
    state=model.state_dict(); return {"weight":state["linear.weight"].detach().numpy(),"bias":state["linear.bias"].detach().numpy()}


def apply_risk(state:dict[str,np.ndarray|float],x:np.ndarray)->np.ndarray:
    w=np.asarray(state["weight"],np.float32).reshape(1,-1); b=float(np.asarray(state["bias"]).reshape(-1)[0]); z=np.asarray(x,np.float32)@w.T+b
    return (1.0/(1.0+np.exp(-np.clip(z[:,0],-30,30)))).astype(np.float32)
