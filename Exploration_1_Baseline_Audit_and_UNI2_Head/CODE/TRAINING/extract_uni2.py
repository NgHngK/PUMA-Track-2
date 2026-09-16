import sys,time,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'outputs')]
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import read_manifest,NucleiDataset,CLASSES
from model import load_uni2

def main():
    torch.set_num_threads(4)
    if (ROOT/'uni2_features.npy').exists():
        raise FileExistsError('Preserve existing feature cache; use a fresh workspace for re-extraction')
    rows=read_manifest(ROOT/'manifest.csv');rng=np.random.default_rng(17);selected=[]
    for split,cap,extra in [('train',12,100),('val',6,50)]:
        rr=[r for r in rows if r['split']==split];chosen=set()
        for c in range(10):
            ix=[i for i,r in enumerate(rr) if r['label']==c]
            rng.shuffle(ix);unique=[];seen=set()
            for i in ix:
                if rr[i]['roi'] not in seen:unique.append(i);seen.add(rr[i]['roi'])
            chosen.update(unique[:cap])
        rest=sorted(set(range(len(rr)))-chosen)
        chosen.update(rng.choice(rest,min(extra,len(rest)),replace=False).tolist())
        selected.extend(rr[i] for i in sorted(chosen))
    p=ROOT/'uni2_sample.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    counts={s:np.bincount([r['label'] for r in selected if r['split']==s],minlength=10).tolist() for s in ['train','val']}
    p.with_suffix('.json').write_text(json.dumps({'classes':CLASSES,'counts':counts},indent=2))
    print('sample',len(selected),counts,flush=True)
    weights=Path(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin')
    m=load_uni2(weights).eval();m.requires_grad_(False)
    ds=NucleiDataset(selected,fov=96);dl=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0)
    dest=ROOT/'uni2_features.npy';complete=ROOT/'uni2_complete.npy'
    features=np.lib.format.open_memmap(dest,mode='r+' if dest.exists() else 'w+',dtype='float32',shape=(len(ds),1536))
    done=np.load(complete) if complete.exists() else np.zeros(len(ds),dtype=bool)
    t=time.time()
    with torch.inference_mode():
        for x,y,ix in dl:
            ii=ix.numpy()
            if done[ii].all():continue
            z=m(x).float()
            if not torch.isfinite(z).all():raise ValueError('Nonfinite embedding')
            features[ii]=z.numpy();done[ii]=True
            if int(done.sum())%20==0:
                features.flush();np.save(complete,done)
                print('extracted',int(done.sum()),'/',len(ds),'elapsed',round(time.time()-t,1),flush=True)
    features.flush();np.save(complete,done)
    h=hashlib.sha256()
    with weights.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    meta={'checkpoint':str(weights),'sha256':h.hexdigest(),'classes':CLASSES,'counts':counts,'fov':96,'input_size':224,
          'sampling':'12/class train +100 random; 6/class val +50 random; spread class picks over ROIs; enriched validation','seconds':time.time()-t,'complete':bool(done.all()),'n':len(ds)}
    (ROOT/'uni2_features.json').write_text(json.dumps(meta,indent=2))
    print('DONE',meta,flush=True)

if __name__=='__main__':main()
