import sys,time,json,gc,copy
from pathlib import Path
ROOT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'outputs')]
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from dataset import read_manifest,NucleiDataset
from model import load_uni2,QVLoRA
from loss import LogitAdjustedCE
from train_eval import metrics,seed_all,train_epoch

class Tail(nn.Module):
    def __init__(self,blocks,norm,head):super().__init__();self.blocks=blocks;self.norm=norm;self.head=head
    def forward(self,x):
        x=self.blocks(x);h=self.norm(x)[:,0]
        return self.head(nn.functional.layer_norm(h.float(),(1536,)))

@torch.no_grad()
def evaluate(m,x,y):
    m.eval();zs=[]
    for chunk in x.split(5):zs.append(m(chunk))
    return metrics(y.numpy(),torch.cat(zs))

def main():
    torch.set_num_threads(4);seed_all(17)
    rows=read_manifest(ROOT/'uni2_sample.csv');selected=[]
    for split,n in [('train',3),('val',2)]:
        for c in range(10):selected.extend([r for r in rows if r['split']==split and r['label']==c][:n])
    out=ROOT/'lora_results_v2';out.mkdir(exist_ok=True)
    (out/'sample.json').write_text(json.dumps(selected,indent=2))
    y=torch.tensor([r['label'] for r in selected]);tr=torch.tensor([r['split']=='train' for r in selected])
    enc=load_uni2(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin').eval();enc.requires_grad_(False)
    path=ROOT/'lora_results/prefix.pt'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        parts=[]
        with torch.no_grad():
            for i,(x,_,_) in enumerate(DataLoader(NucleiDataset(selected,96),batch_size=2)):
                z=enc.patch_embed(x);z=enc._pos_embed(z);z=enc.patch_drop(z);z=enc.norm_pre(z)
                for b in enc.blocks[:20]:z=b(z)
                parts.append(z.detach());print('prefix',2*(i+1),'/',len(selected),flush=True)
        prefix=torch.cat(parts);torch.save(prefix,path)
    else:prefix=torch.load(path,weights_only=True)
    tailblocks=nn.Sequential(*list(enc.blocks[20:]));norm=enc.norm
    del enc;gc.collect()
    seed_all(17);inithead=nn.Linear(1536,10);headstate=copy.deepcopy(inithead.state_dict())
    with torch.no_grad():
        final_features=torch.cat([nn.functional.layer_norm(norm(tailblocks(chunk))[:,0].float(),(1536,)) for chunk in prefix.split(5)])
    for mode in ['frozen','lora']:
        if (out/(mode+'.json')).exists():continue
        seed_all(17)
        if mode=='lora':
            for b in tailblocks:b.attn.qkv=QVLoRA(b.attn.qkv,8,16)
        head=nn.Linear(1536,10);head.load_state_dict(headstate)
        m=Tail(tailblocks,norm,head) if mode=='lora' else head
        inputs=prefix if mode=='lora' else final_features
        params=[p for p in tailblocks.parameters() if p.requires_grad] if mode=='lora' else []
        groups=[{'params':head.parameters(),'lr':.001}]
        if params:groups.append({'params':params,'lr':.00001})
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        dl=DataLoader(TensorDataset(inputs[tr],y[tr],torch.arange(int(tr.sum()))),batch_size=5,shuffle=True,generator=torch.Generator().manual_seed(17))
        loss=LogitAdjustedCE(torch.ones(10),0);hist=[]
        for e in range(1,6):
            objective,exposure=train_epoch(m,dl,opt,loss,torch.device('cpu'))
            mt=evaluate(m,inputs[tr],y[tr]);mv=evaluate(m,inputs[~tr],y[~tr]);gradient=[]
            if params:
                for c in range(10):
                    ix=torch.where(tr & (y==c))[0][:1]
                    l=loss(m(prefix[ix]),y[ix]);grads=torch.autograd.grad(l,params)
                    gradient.append(float(torch.sqrt(sum(g.square().sum() for g in grads))))
            r={'epoch':e,'objective':objective,'train':mt,'val':mv,'positive_adapter_gradient':gradient,'exposure':exposure,
               'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)};hist.append(r)
            (out/(mode+'.partial.json')).write_text(json.dumps(hist,indent=2))
            print(mode,e,round(objective,4),round(mt['macro_f1'],4),round(mv['macro_f1'],4),flush=True)
        (out/(mode+'.json')).write_text(json.dumps(hist,indent=2))
        torch.save({k:v for k,v in m.state_dict().items() if k in dict(m.named_parameters()) and dict(m.named_parameters())[k].requires_grad},out/(mode+'_trainable.pt'))

if __name__=='__main__':main()
