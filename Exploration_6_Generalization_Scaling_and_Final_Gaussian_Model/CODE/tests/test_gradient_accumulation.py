from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from puma_exploration6.config import ExperimentConfig, LossConfig, ModelConfig
from puma_exploration6.losses import PriorAdjustedCrossEntropy
from puma_exploration6.models.stage2 import Stage2FromCachedCLS
from puma_exploration6.training import train_one_epoch


class _DS(Dataset):
    def __init__(self, x, bio, y): self.x=x; self.bio=bio; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"features":self.x[i],"bio":self.bio[i],"label":self.y[i],"index":torch.tensor(i),"group":f"g{i}"}


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, betas=(.9,.999), eps=1e-8)


def test_partial_window_gradient_accumulation_matches_one_full_effective_batch():
    torch.manual_seed(3)
    n=10
    x=torch.randn(n,1536);bio=torch.randn(n,16);y=torch.arange(10,dtype=torch.long)
    ds=_DS(x,bio,y)
    base=Stage2FromCachedCLS(ModelConfig())
    full=copy.deepcopy(base);micro=copy.deepcopy(base)
    criterion=PriorAdjustedCrossEntropy(torch.ones(10)/10,LossConfig())
    r_full=train_one_epoch(full,DataLoader(ds,batch_size=10,shuffle=False),_optimizer(full),criterion,torch.device('cpu'),1.0,accumulation_steps=1)
    r_micro=train_one_epoch(micro,DataLoader(ds,batch_size=3,shuffle=False),_optimizer(micro),criterion,torch.device('cpu'),1.0,accumulation_steps=4)
    assert np.isclose(r_full.loss,r_micro.loss,rtol=1e-6,atol=1e-6)
    for (n1,p1),(n2,p2) in zip(full.named_parameters(),micro.named_parameters()):
        assert n1==n2
        assert torch.allclose(p1,p2,rtol=2e-6,atol=2e-7),n1


def test_raw_hpo_preserves_microbatch_and_varies_effective_batch():
    from puma_exploration6.hpo import generate_hpo_configs
    base={
        "experiment_id":"raw","manifest":"m.csv","output_dir":"runs","uni2_weights":"uni.pt","tier_a":"tier.npy",
        "optimizer":{"batch_size":4,"accumulation_steps":16,"max_epochs":20},
    }
    trials=generate_hpo_configs(base,20,"optimizer",seed=5)
    for c in trials:
        assert c["optimizer"]["batch_size"]==4
        assert c["optimizer"]["batch_size"]*c["optimizer"]["accumulation_steps"] in {32,64,128}
