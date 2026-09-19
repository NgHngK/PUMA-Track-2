import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from puma_exploration6.training import train_one_epoch
from puma_exploration6.losses import PriorAdjustedCrossEntropy
from puma_exploration6.config import LossConfig

class DS(Dataset):
    def __len__(self): return 2
    def __getitem__(self,i):
        return {"image":torch.zeros(3,224,224),"bio":torch.zeros(16),"label":torch.tensor(i),"index":torch.tensor(i),"group":f"g{i}","u":torch.tensor(8.),"v":torch.tensor(8.)}

class Enc(nn.Module):
    def __init__(self, trainable=False):
        super().__init__(); self.p=nn.Parameter(torch.tensor(0.),requires_grad=trainable); self.was_training=[]
    def forward_tokens(self,x):
        self.was_training.append(self.training)
        t=torch.zeros(len(x),265,1536,device=x.device); t[:,0,0]=self.p
        return t

class Model(nn.Module):
    def __init__(self): super().__init__(); self.w=nn.Linear(1536,10)
    def forward(self,t,b,u,v): return self.w(t[:,0])

def _run(trainable):
    m=Model(); e=Enc(trainable); opt=torch.optim.AdamW(list(m.parameters())+([e.p] if trainable else []),lr=1e-3)
    crit=PriorAdjustedCrossEntropy(torch.ones(10)/10,LossConfig())
    train_one_epoch(m,DataLoader(DS(),batch_size=2),opt,crit,torch.device('cpu'),1.0,encoder=e)
    return e

def test_frozen_encoder_remains_eval_during_training():
    e=_run(False); assert e.was_training and set(e.was_training)=={False}

def test_trainable_encoder_uses_train_mode():
    e=_run(True); assert e.was_training and set(e.was_training)=={True}
