import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from puma_exploration6.tier_a import roi_channels,point_features,TierANormalizer,NAMES
from puma_exploration6.sampling import build_sampler,expected_unique_draws
from puma_exploration6.config import SamplerConfig,LossConfig
from puma_exploration6.losses import PriorAdjustedCrossEntropy


def test_tiera_exact_16_and_train_only_standardization():
    rng=np.random.default_rng(3);im=Image.fromarray(rng.integers(1,255,(128,128,3),dtype=np.uint8));f=point_features(roi_channels(im),64.2,63.8)
    assert f.shape==(16,) and len(NAMES)==16 and np.isfinite(f).all()
    x=np.stack([f+i*.001 for i in range(30)]);n=TierANormalizer().fit(x[:20]);z=n.transform(x);assert np.abs(z[:20].mean(0)).max()<1e-3;assert np.abs(z[:20].std(0)-1).max()<1e-3


def test_inverse_sampler_induced_prior_uniform():
    labels=torch.tensor(sum(([c]*(c+1) for c in range(10)),[]));_,q,w=build_sampler(labels,SamplerConfig(mode='inverse',alpha=1),17)
    torch.testing.assert_close(q,torch.full((10,),.1,dtype=torch.double),rtol=1e-12,atol=1e-12)
    assert expected_unique_draws(w,len(labels))<len(labels)


def test_tempered_prior_between_natural_and_uniform():
    labels=torch.tensor(sum(([c]*(c+1) for c in range(10)),[]));counts=torch.bincount(labels,minlength=10).double();natural=counts/counts.sum();_,q,_=build_sampler(labels,SamplerConfig(mode='tempered',alpha=.5),17)
    assert torch.linalg.vector_norm(q-.1)<torch.linalg.vector_norm(natural-.1)


def test_ce_tau0_exact_pytorch_ce():
    torch.manual_seed(2);z=torch.randn(17,10);y=torch.randint(0,10,(17,));prior=torch.arange(1,11,dtype=torch.float)
    loss=PriorAdjustedCrossEntropy(prior,LossConfig(kind='ce',tau=0))(z,y);torch.testing.assert_close(loss,F.cross_entropy(z,y),rtol=0,atol=0)


def test_balanced_softmax_formula():
    torch.manual_seed(2);z=torch.randn(17,10);y=torch.randint(0,10,(17,));prior=torch.arange(1,11,dtype=torch.float);p=prior/prior.sum()
    loss=PriorAdjustedCrossEntropy(prior,LossConfig(kind='balanced_softmax',tau=1))(z,y)
    torch.testing.assert_close(loss,F.cross_entropy(z+p.log(),y),rtol=1e-7,atol=1e-7)


def test_epoch_repeat_diagnostics_are_classwise_and_group_aware(tmp_path):
    from torch.utils.data import DataLoader
    from puma_exploration6.datasets import CachedNucleiDataset
    from puma_exploration6.models.stage2 import Stage2FromCachedCLS
    from puma_exploration6.config import ModelConfig,ExperimentConfig
    from puma_exploration6.training import train_one_epoch
    rows=[];labels=[]
    classes=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
    for c in range(10):
        rows.append({'uid':str(c),'roi':f'r{c}','group':f'g{c}','image':'x','x':8,'y':8,'label':c,'class_name':classes[c],'split':'train','coordinate_source':'gt'});labels.append(c)
    rep=np.random.default_rng(1).normal(size=(10,1536)).astype('float32');ds=CachedNucleiDataset(rows,rep,None);dl=DataLoader(ds,batch_size=10,shuffle=False)
    m=Stage2FromCachedCLS(ModelConfig(use_tier_a=False));opt=torch.optim.AdamW(m.parameters(),lr=1e-3);crit=PriorAdjustedCrossEntropy(torch.ones(10)/10,LossConfig(kind='ce',tau=0))
    er=train_one_epoch(m,dl,opt,crit,torch.device('cpu'),1.0)
    assert er.unique_groups==10 and er.unique_examples_by_class==[1]*10 and er.repeat_fraction_by_class==[0.0]*10
