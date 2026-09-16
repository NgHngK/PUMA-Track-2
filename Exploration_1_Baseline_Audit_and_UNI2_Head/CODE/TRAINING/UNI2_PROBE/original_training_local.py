"""Head-only ten-epoch probes on verified frozen UNI2-h features."""
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).parent

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from exploration1_dataset import read_manifest,CLASSES
from prompt1_loss import LogitAdjustedCE
from exploration1_train_eval import seed_all,metrics,train_epoch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=17);a=parser.parse_args()
    torch.set_num_threads(2)
    if not np.load(ROOT/'uni2_complete.npy').all():raise RuntimeError('Feature extraction incomplete')
    x=torch.from_numpy(np.load(ROOT/'uni2_features.npy').copy());x=nn.functional.layer_norm(x,(1536,))
    rows=read_manifest(ROOT/'uni2_sample.csv');y=torch.tensor([r['label'] for r in rows]);tr=torch.tensor([r['split']=='train' for r in rows])
    xt,yt=x[tr],y[tr];xv,yv=x[~tr],y[~tr];counts=torch.bincount(yt,minlength=10).float();q=counts/counts.sum()
    out=OUTPUT_DIR;out.mkdir(exist_ok=True)
    for recipe,tau,balanced in [v for v in [('ce',0,False),('la',1,False),('balanced_ce',0,True)] if v[0]==RECIPE]:
        recipe=recipe if a.seed==17 else recipe+'_seed'+str(a.seed)
        if (out/(recipe+'.json')).exists():continue
        seed_all(a.seed);m=nn.Linear(1536,10);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
        sampler=WeightedRandomSampler(1/counts[yt],len(yt),replacement=True,generator=torch.Generator().manual_seed(a.seed)) if balanced else None
        dl=DataLoader(TensorDataset(xt,yt,torch.arange(len(yt))),batch_size=64,shuffle=sampler is None,sampler=sampler)
        loss=LogitAdjustedCE(torch.ones(10)/10 if balanced else q,tau);hist=[];best=-1
        for e in range(1,11):
            obj,exposure=train_epoch(m,dl,opt,loss,torch.device('cpu'))
            m.eval()
            with torch.no_grad():zt=m(xt);zv=m(xv);mt=metrics(yt.numpy(),zt);mv=metrics(yv.numpy(),zv)
            grads=[]
            for c in range(10):
                ix=torch.where(yt==c)[0]
                l=loss(m(xt[ix]),yt[ix]);g=torch.autograd.grad(l,m.weight)[0][c]
                grads.append(float(g.norm()))
            rec={'epoch':e,'objective':obj,'train':mt,'val':mv,'exposure':exposure,'positive_head_grad':grads};hist.append(rec)
            print(recipe,e,round(obj,4),round(mt['macro_f1'],4),round(mv['macro_f1'],4),flush=True)
            if mv['macro_f1']>best:
                best=mv['macro_f1'];torch.save(m.state_dict(),out/(recipe+'_best.pt'))
                np.savez_compressed(out/(recipe+'_best.npz'),logits=zv.numpy(),labels=yv.numpy(),rois=np.array([r['roi'] for r,t in zip(rows,tr) if not t]))
        (out/(recipe+'.json')).write_text(json.dumps(hist,indent=2))
    # Geometry is exploratory; no decoder/feature update or data fitting on validation.
    h=xt.numpy();hc=h-h.mean(0);s=np.linalg.svd(hc,compute_uv=False);v=s*s;v/=v.sum()
    means=np.stack([h[yt.numpy()==c].mean(0) for c in range(10)])
    n=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-8)
    geometry={'train_covariance_effective_rank':float(np.exp(-(v[v>0]*np.log(v[v>0])).sum())),
              'train_max_rank':len(xt)-1,'centroid_cosine':(n@n.T).tolist(),'train_counts':counts.tolist()}
    (out/'geometry.json').write_text(json.dumps(geometry,indent=2))

if __name__=='__main__':main()
