from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..constants import EMBED_DIM, UNI2_GRID_SIZE, UNI2_SPATIAL_START, UNI2_TOKEN_COUNT


def _check_tokens(tokens: torch.Tensor) -> None:
    if tokens.ndim != 3 or tokens.shape[1:] != (UNI2_TOKEN_COUNT, EMBED_DIM):
        raise ValueError(f"tokens must be Bx{UNI2_TOKEN_COUNT}x{EMBED_DIM}")
    if not torch.isfinite(tokens).all(): raise FloatingPointError("nonfinite UNI2 tokens")


def _check_uv(u: torch.Tensor,v: torch.Tensor,batch: int) -> tuple[torch.Tensor,torch.Tensor]:
    u=torch.as_tensor(u,dtype=torch.float32,device=u.device if torch.is_tensor(u) else None).reshape(-1)
    v=torch.as_tensor(v,dtype=torch.float32,device=v.device if torch.is_tensor(v) else None).reshape(-1)
    if len(u)!=batch or len(v)!=batch or not torch.isfinite(u).all() or not torch.isfinite(v).all(): raise ValueError("invalid u/v")
    eps=torch.finfo(u.dtype).eps
    return u.clamp(0,UNI2_GRID_SIZE-eps),v.clamp(0,UNI2_GRID_SIZE-eps)


def gaussian_pool(tokens: torch.Tensor,u: torch.Tensor,v: torch.Tensor,sigma: float=1.5) -> torch.Tensor:
    _check_tokens(tokens)
    if sigma<=0:raise ValueError("sigma must be positive")
    b=tokens.shape[0];u,v=_check_uv(u,v,b);spatial=tokens[:,UNI2_SPATIAL_START:,:]
    yy,xx=torch.meshgrid(torch.arange(UNI2_GRID_SIZE,device=tokens.device,dtype=tokens.dtype),
                         torch.arange(UNI2_GRID_SIZE,device=tokens.device,dtype=tokens.dtype),indexing="ij")
    gridx=xx.reshape(1,-1);gridy=yy.reshape(1,-1)
    d2=(gridx-u[:,None].to(tokens.dtype)).square()+(gridy-v[:,None].to(tokens.dtype)).square()
    w=torch.exp(-d2/(2*sigma*sigma));w=w/w.sum(1,keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    return torch.einsum("bn,bnd->bd",w,spatial)


def target_patch_pool(tokens: torch.Tensor,u: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
    _check_tokens(tokens);b=tokens.shape[0];u,v=_check_uv(u,v,b)
    x=torch.floor(u).long().clamp(0,UNI2_GRID_SIZE-1);y=torch.floor(v).long().clamp(0,UNI2_GRID_SIZE-1)
    idx=UNI2_SPATIAL_START+y*UNI2_GRID_SIZE+x
    return tokens[torch.arange(b,device=tokens.device),idx]


def neighborhood_pool(tokens: torch.Tensor,u: torch.Tensor,v: torch.Tensor,radius: int=1) -> torch.Tensor:
    """Mean of the clipped square token neighborhood, vectorized over the batch.

    ``count_include_pad=False`` reproduces the historical edge-clipped mean exactly
    while avoiding a Python loop over nuclei.
    """
    _check_tokens(tokens)
    if radius<0 or radius>7:raise ValueError("radius outside [0,7]")
    b=tokens.shape[0];u,v=_check_uv(u,v,b);x=torch.floor(u).long();y=torch.floor(v).long()
    spatial=tokens[:,UNI2_SPATIAL_START:,:].reshape(b,UNI2_GRID_SIZE,UNI2_GRID_SIZE,EMBED_DIM).permute(0,3,1,2)
    k=2*radius+1
    pooled=F.avg_pool2d(spatial,kernel_size=k,stride=1,padding=radius,count_include_pad=False)
    return pooled.permute(0,2,3,1)[torch.arange(b,device=tokens.device),y,x]


class TokenPooler(nn.Module):
    """Exploration-4-compatible token representations.

    Every output receives exactly one final non-affine LayerNorm. For global_local,
    the historical A4 equation is reproduced exactly:
      LN((LN(CLS) + LN(Gaussian))/2).
    """
    def __init__(self,kind: str="cls",gaussian_sigma: float=1.5,neighborhood_radius: int=1,scalar_mix_init: float=0.5):
        super().__init__();self.kind=kind;self.gaussian_sigma=float(gaussian_sigma);self.radius=int(neighborhood_radius)
        if kind not in {"cls","target_patch","gaussian","neighborhood","global_local","scalar_global_local"}:raise ValueError(kind)
        if not 0<=scalar_mix_init<=1:raise ValueError("scalar_mix_init")
        if kind=="scalar_global_local":
            p=min(max(float(scalar_mix_init),1e-6),1-1e-6);self.mix_logit=nn.Parameter(torch.tensor(math.log(p/(1-p)),dtype=torch.float32))
        else:self.register_parameter("mix_logit",None)

    def forward(self,tokens: torch.Tensor,u: torch.Tensor,v: torch.Tensor) -> torch.Tensor:
        _check_tokens(tokens)
        cls=tokens[:,0,:]
        if self.kind=="cls": raw=cls
        elif self.kind=="target_patch":raw=target_patch_pool(tokens,u,v)
        elif self.kind=="gaussian":raw=gaussian_pool(tokens,u,v,self.gaussian_sigma)
        elif self.kind=="neighborhood":raw=neighborhood_pool(tokens,u,v,self.radius)
        else:
            local=gaussian_pool(tokens,u,v,self.gaussian_sigma)
            cls_n=F.layer_norm(cls,(EMBED_DIM,));local_n=F.layer_norm(local,(EMBED_DIM,))
            if self.kind=="global_local":raw=(cls_n+local_n)/2
            else:
                alpha=self.mix_logit.sigmoid().to(tokens.dtype);raw=alpha*cls_n+(1-alpha)*local_n
        return F.layer_norm(raw.float(),(EMBED_DIM,)).to(tokens.dtype)
