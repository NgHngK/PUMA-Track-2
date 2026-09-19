from __future__ import annotations

import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..constants import EMBED_DIM, UNI2_TOKEN_COUNT


def checkpoint_sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def load_uni2_h(path: str | Path) -> nn.Module:
    """Strict local UNI2-h loader. Never downloads weights."""
    try: import timm
    except ImportError as e: raise RuntimeError("UNI2 raw-image mode requires timm==1.0.20") from e
    if getattr(timm,"__version__",None)!="1.0.20":raise RuntimeError(f"Expected timm==1.0.20, got {getattr(timm,'__version__',None)}")
    state=torch.load(Path(path),map_location="cpu",weights_only=True,mmap=True)
    if isinstance(state,dict) and "state_dict" in state:state=state["state_dict"]
    expected={"cls_token":(1,1,EMBED_DIM),"reg_token":(1,8,EMBED_DIM),"patch_embed.proj.weight":(EMBED_DIM,3,14,14)}
    for key,shape in expected.items():
        if key not in state or tuple(state[key].shape)!=shape:raise ValueError(f"not UNI2-h: expected {key} {shape}")
    if not hasattr(timm.layers,"SwiGLUPacked"):raise RuntimeError("timm build lacks SwiGLUPacked")
    kw=dict(img_size=224,patch_size=14,depth=24,num_heads=24,init_values=1e-5,embed_dim=EMBED_DIM,mlp_ratio=2.66667*2,
            num_classes=0,no_embed_class=True,mlp_layer=timm.layers.SwiGLUPacked,act_layer=nn.SiLU,reg_tokens=8,dynamic_img_size=True)
    encoder=timm.create_model("vit_giant_patch14_224",pretrained=False,weight_init="skip",**kw)
    encoder.load_state_dict(state,strict=True,assign=True);return encoder


class QVLoRA(nn.Module):
    def __init__(self,base: nn.Linear,rank: int=8,alpha: float=16.0):
        super().__init__();d=base.in_features
        if base.out_features!=3*d or rank<1:raise ValueError("expected fused 3D x D QKV")
        self.base=base;self.base.requires_grad_(False);self.scale=float(alpha)/rank
        self.aq=nn.Parameter(torch.empty(rank,d));self.av=nn.Parameter(torch.empty(rank,d));self.bq=nn.Parameter(torch.zeros(d,rank));self.bv=nn.Parameter(torch.zeros(d,rank))
        nn.init.kaiming_uniform_(self.aq,a=math.sqrt(5));nn.init.kaiming_uniform_(self.av,a=math.sqrt(5))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        q,k,v=self.base(x).chunk(3,dim=-1);q=q+self.scale*F.linear(F.linear(x,self.aq),self.bq);v=v+self.scale*F.linear(F.linear(x,self.av),self.bv)
        return torch.cat((q,k,v),dim=-1)


class UNI2Backbone(nn.Module):
    def __init__(self,weights: str | Path,adaptation: str="frozen",lora_rank: int=8,lora_alpha: float=16,last_blocks: int=4):
        super().__init__();self.encoder=load_uni2_h(weights);self.adaptation=adaptation;self.encoder.requires_grad_(False)
        if adaptation not in {"frozen","lora"}:raise ValueError("adaptation must be frozen or lora")
        if adaptation=="lora":
            if not 1<=last_blocks<=24:raise ValueError("last_blocks outside [1,24]")
            for b in self.encoder.blocks[-last_blocks:]:b.attn.qkv=QVLoRA(b.attn.qkv,lora_rank,lora_alpha)
        self.encoder.eval()

    def train(self,mode: bool=True):
        # Keep the pretrained backbone in eval mode for both frozen and PEFT runs.
        # Eval mode does not disable autograd; it only prevents stochastic/dropout-state drift.
        super().train(mode);self.encoder.eval();return self

    def forward_tokens(self,x: torch.Tensor) -> torch.Tensor:
        if self.adaptation=="frozen":
            with torch.no_grad():tokens=self.encoder.forward_features(x)
        else:tokens=self.encoder.forward_features(x)
        if tokens.ndim!=3 or tokens.shape[1:]!=(UNI2_TOKEN_COUNT,EMBED_DIM):raise RuntimeError(f"unexpected UNI2 token shape {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():raise FloatingPointError("nonfinite UNI2 tokens")
        return tokens.float()
