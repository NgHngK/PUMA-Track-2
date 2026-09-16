"""Real image small-CNN mechanisms screen; never a surrogate for UNI2 accuracy."""
import sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).parent

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from PIL import Image,ImageDraw
from exploration1_dataset import read_manifest,centered_crop,CLASSES,MEAN,STD
from exploration1_train_eval import metrics,seed_all,train_epoch
from prompt1_loss import LogitAdjustedCE

class Pixels(Dataset):
    def __init__(self,x,y,augment=False):self.x=x;self.y=y;self.augment=augment
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        t=torch.from_numpy(self.x[i].copy()).permute(2,0,1).float()/255
        if self.augment:
            t=torch.rot90(t,int(torch.randint(4,()).item()),[1,2])
            if torch.rand(())<.5:t=t.flip(-1)
        return (t-MEAN)/STD,int(self.y[i]),i

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__();layers=[];n=3
        for d in [16,32,64]:layers.extend([nn.Conv2d(n,d,3,padding=1),nn.GELU(),nn.MaxPool2d(2)]);n=d
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((3,3)),nn.Flatten())
        self.head=nn.Linear(64*9,10)
    def forward(self,x):return self.head(self.encoder(x))

@torch.no_grad()
def evaluate(m,ds):
    m.eval();zs=[]
    for x,y,_ in DataLoader(ds,batch_size=128):zs.append(m(x))
    z=torch.cat(zs);return metrics(ds.y,z),z.numpy()

def positive_gradients(m,ds):
    m.eval();result=[]
    for c in range(10):
        ix=np.flatnonzero(ds.y==c)[:8]
        if not len(ix):result.append(None);continue
        x=torch.stack([ds[int(i)][0] for i in ix]);y=torch.full((len(ix),),c,dtype=torch.long)
        loss=nn.functional.cross_entropy(m(x),y)
        grad=torch.autograd.grad(loss,m.head.weight)[0]
        result.append(float(grad[c].norm()))
    return result

def prepare():
    cache=ROOT/'cnn_pixels.npz'
    if cache.exists():return np.load(cache)
    rows=read_manifest(ROOT/'manifest.csv');rng=np.random.default_rng(17);by={}
    for r in rows:by.setdefault(r['roi'],[]).append(r)
    selected=[];pixels=[]
    for roi,rr in by.items():
        ix=sorted(rng.choice(len(rr),min(100,len(rr)),replace=False))
        with Image.open(rr[0]['image']) as im:
            image=im.convert('RGB')
            for i in ix:
                r=rr[i];pixels.append(np.asarray(centered_crop(image,r['x'],r['y'],96).resize((48,48),Image.Resampling.BICUBIC)));selected.append(r)
    x=np.stack(pixels);y=np.array([r['label'] for r in selected]);train=np.array([r['split']=='train' for r in selected])
    np.savez_compressed(cache,x=x,y=y,train=train,uids=np.array([r['uid'] for r in selected]),rois=np.array([r['roi'] for r in selected]))
    with (ROOT/'cnn_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    # Show all ten classes with the target coordinate always at the middle of the crop.
    canvas=Image.new('RGB',(500,10*80),'white');draw=ImageDraw.Draw(canvas)
    for c,name in enumerate(CLASSES):
        draw.text((2,c*80+4),name,fill='black')
        for j,i in enumerate(np.flatnonzero(y==c)[:4]):
            im=Image.fromarray(x[i]).resize((64,64));canvas.paste(im,(140+j*80,c*80))
            draw.line((140+j*80+30,c*80+32,140+j*80+34,c*80+32),fill='red')
            draw.line((140+j*80+32,c*80+30,140+j*80+32,c*80+34),fill='red')
    canvas.save(ROOT/'crop_contact_sheet.png')
    return np.load(cache)

def main():
    torch.set_num_threads(2);d=prepare();out=OUTPUT_DIR;out.mkdir(exist_ok=True)
    train=d['train'];tr=Pixels(d['x'][train],d['y'][train]);va=Pixels(d['x'][~train],d['y'][~train])
    counts=np.bincount(tr.y,minlength=10);q=counts/counts.sum()
    (out/'supports.json').write_text(json.dumps({'train':counts.tolist(),'val':np.bincount(va.y,minlength=10).tolist(),'n':len(d['y'])}))
    for recipe,tau,balanced in [v for v in [('ce',0,False),('la',1,False),('balanced_ce',0,True)] if v[0]==RECIPE]:
        if (out/(recipe+'.json')).exists():continue
        seed_all(17);model=TinyCNN();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        sampler=WeightedRandomSampler(torch.as_tensor(1/counts[tr.y]),len(tr),replacement=True,generator=torch.Generator().manual_seed(17)) if balanced else None
        dl=DataLoader(Pixels(tr.x,tr.y,True),batch_size=64,shuffle=sampler is None,sampler=sampler)
        loss=LogitAdjustedCE(np.ones(10)/10 if balanced else q,tau);records=[];best=-1;t=time.time()
        for epoch in range(1,11):
            objective,exposure=train_epoch(model,dl,opt,loss,torch.device('cpu'))
            mt,_=evaluate(model,tr);mv,z=evaluate(model,va)
            record={'epoch':epoch,'objective':objective,'train':mt,'val':mv,'exposure':exposure,'positive_head_grad':positive_gradients(model,tr),'seconds':time.time()-t}
            records.append(record)
            with (out/(recipe+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
            print(recipe,epoch,'train',round(mt['macro_f1'],4),'val',round(mv['macro_f1'],4),'CE',round(mv['nll'],4),flush=True)
            if mv['macro_f1']>best:
                best=mv['macro_f1'];np.savez_compressed(out/(recipe+'_best.npz'),logits=z,labels=va.y,rois=d['rois'][~train]);torch.save(model.state_dict(),out/(recipe+'_best.pt'))
        (out/(recipe+'.json')).write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
