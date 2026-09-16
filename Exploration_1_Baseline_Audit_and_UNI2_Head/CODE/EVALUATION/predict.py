"""Classify every frozen Stage-1 proposal; preserve coordinates and original score."""
import argparse,csv,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import CLASSES,NucleiDataset
from model import Stage2,checkpoint_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--proposals',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.temperature<=0:raise ValueError('Temperature must be positive')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config']
    if cfg['classes']!=CLASSES:raise ValueError('Class-map mismatch')
    if cfg['weights_sha256']!=checkpoint_sha256(a.weights):raise ValueError('Wrong foundation checkpoint hash')
    model=Stage2(a.weights,cfg['mode'],cfg['rank'],cfg['alpha'],cfg['last_blocks']).to(a.device)
    state=model.state_dict();state.update(ck['trainable']);model.load_state_dict(state,strict=True);model.eval()
    with open(a.proposals,newline='',encoding='utf-8') as f:source=list(csv.DictReader(f))
    if not source or not {'uid','roi','image','x','y','score'}.issubset(source[0]):raise ValueError('Need uid,roi,image,x,y,score')
    if len({r['uid'] for r in source})!=len(source):raise ValueError('Duplicate proposal UID')
    rows=[]
    for r in source:
        rr=r.copy();rr.update(x=float(r['x']),y=float(r['y']),label=-1)
        image=Path(r['image']);rr['image']=str(image if image.is_absolute() else Path(a.proposals).parent/image)
        rows.append(rr)
    ds=NucleiDataset(rows,cfg['fov']);dl=DataLoader(ds,batch_size=a.batch_size)
    output=[]
    with torch.inference_mode():
        for x,_,ix in dl:
            prob=(model(x.to(a.device))/a.temperature).softmax(-1).cpu()
            for i,pv in zip(ix.tolist(),prob):
                r=source[i].copy();c=int(pv.argmax());r.update(class_id=c,class_name=CLASSES[c],semantic_confidence=float(pv[c]))
                r.update({f'p_{name}':float(pv[k]) for k,name in enumerate(CLASSES)});output.append(r)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    # These are classifier scores under its trained prior. No detection-score substitution.

if __name__=='__main__':main()
