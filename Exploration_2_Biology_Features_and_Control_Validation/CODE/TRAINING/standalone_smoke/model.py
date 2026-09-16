import torch
from torch import nn

class CachedClassifier(nn.Module):
    """Minimal linear appearance head with optional zero-initialized correction."""
    def __init__(self,biology_dim=0,biology_only=False):
        super().__init__();self.biology_only=biology_only
        self.head=nn.Linear(biology_dim if biology_only else 1536,10)
        self.correction=nn.Linear(biology_dim,10,bias=False) if biology_dim and not biology_only else None
        if self.correction is not None:nn.init.zeros_(self.correction.weight)
    def forward(self,x):
        if self.biology_only:return self.head(x)
        z=self.head(x[:,:1536])
        return z+self.correction(x[:,1536:]) if self.correction is not None else z


class FrozenStage2(nn.Module):
    """Optional direct image wrapper around the same frozen encoder and classifier."""
    def __init__(self,weights,biology_dim=0):
        super().__init__()
        from uni2 import load_uni2
        self.encoder=load_uni2(weights).eval().requires_grad_(False)
        self.classifier=CachedClassifier(biology_dim)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,images,normalized_biology=None):
        with torch.no_grad():h=nn.functional.layer_norm(self.encoder(images).float(),(1536,))
        if normalized_biology is not None:h=torch.cat((h,normalized_biology),1)
        return self.classifier(h)
