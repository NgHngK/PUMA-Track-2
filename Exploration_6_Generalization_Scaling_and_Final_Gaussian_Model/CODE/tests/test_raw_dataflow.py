import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from puma_exploration6.config import AugmentationConfig,ModelConfig,ExperimentConfig,SamplerConfig,LossConfig
from puma_exploration6.datasets import RawNucleiDataset
from puma_exploration6.models.stage2 import Stage2FromTokens
from puma_exploration6.losses import PriorAdjustedCrossEntropy
from puma_exploration6.training import train_one_epoch

class FakeEncoder(torch.nn.Module):
    def forward_tokens(self,x):
        b=len(x);base=x.mean((1,2,3),keepdim=False)[:,None,None]
        return base.expand(b,265,1536).contiguous()


def test_raw_image_to_tokens_to_a5_train_step(tmp_path):
    rows=[];tier=[]
    for c in range(10):
        p=tmp_path/f'{c}.png';Image.fromarray(np.full((128,128,3),20+c*20,dtype=np.uint8)).save(p)
        rows.append({'uid':str(c),'image':str(p),'x':64.1,'y':63.9,'label':c});tier.append(np.linspace(-1,1,16)+c*.01)
    ds=RawNucleiDataset(rows,np.asarray(tier,dtype=np.float32),96,AugmentationConfig(geometric=True,stain_mode='fixed_hed',stain_strength=.05));dl=DataLoader(ds,batch_size=5,shuffle=False,num_workers=0)
    m=Stage2FromTokens(ModelConfig(representation='global_local'));enc=FakeEncoder();opt=torch.optim.AdamW(m.parameters(),lr=1e-3);crit=PriorAdjustedCrossEntropy(torch.ones(10)/10,LossConfig(kind='ce',tau=0))
    result=train_one_epoch(m,dl,opt,crit,torch.device('cpu'),1.0,None,enc,False);assert np.isfinite(result.loss);assert sum(result.exposure)==10
