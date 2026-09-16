"""Frozen-feature training; exact V17 full-ROI validation from all proposal rows."""
import argparse,json,hashlib,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from dataset import read_manifest,CLASSES
from model import CachedClassifier
from loss import LogitAdjustedCE
from numerical import metrics,train_epoch,seed_all
from evaluation import evaluate_proposals
from features import input_contract

def prepare(cache,manifest,config,normalizer=None):
    cache=Path(cache);contract=json.loads((cache/'contract.json').read_text())
    if not (cache/'complete.json').exists():raise ValueError('Incomplete feature cache')
    complete=json.loads((cache/'complete.json').read_text())
    if complete['feature_sha256']!=hashlib.sha256((cache/'features.npy').read_bytes()).hexdigest():raise ValueError('Feature cache checksum mismatch')
    if contract['manifest_sha256']!=hashlib.sha256(Path(manifest).read_bytes()).hexdigest():raise ValueError('Manifest/cache mismatch')
    if contract['input_contract']!=input_contract(config):raise ValueError('Selected architecture/cache mismatch')
    rows=read_manifest(manifest);y=torch.tensor([r['label'] for r in rows]);tr=torch.tensor([r['split']=='train' and r['label']>=0 for r in rows])
    x=nn.functional.layer_norm(torch.from_numpy(np.load(cache/'features.npy').copy()),(1536,));bdim=0
    if config.get('biology_families'):
        if complete.get('biology_sha256')!=hashlib.sha256((cache/'biology.npy').read_bytes()).hexdigest():raise ValueError('Biology cache checksum mismatch')
        from biology import SCHEMA,NAMES
        names=sum([SCHEMA[f] for f in config['biology_families']],[]);columns=[i for i,n in enumerate(NAMES) if n in names];b=np.load(cache/'biology.npy')[:,columns]
        if normalizer is None:
            if not tr.any():raise ValueError('Cannot fit biology normalization without training rows')
            normalizer={'columns':columns,'mean':b[tr.numpy()].mean(0).tolist(),'std':np.maximum(b[tr.numpy()].std(0),1e-6).tolist()}
        if normalizer['columns']!=columns:raise ValueError('Biology schema mismatch')
        b=(b-np.asarray(normalizer['mean']))/np.asarray(normalizer['std']);bdim=b.shape[1];x=torch.cat((x,torch.from_numpy(b.astype(np.float32))),1)
    return rows,y,tr,x,bdim,normalizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--cache',required=True);p.add_argument('--annotations-root',required=True);p.add_argument('--roi-splits',required=True);p.add_argument('--config',default='config.json');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--eval-checkpoint');p.add_argument('--eval-split',choices=['val','calibration','test'],default='val');p.add_argument('--temperature',type=float,default=1.);p.add_argument('--allow-gt-control',action='store_true');p.add_argument('--stage1-provenance');a=p.parse_args()
    torch.set_num_threads(2);seed_all(a.seed);config=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.eval_checkpoint,map_location='cpu',weights_only=True) if a.eval_checkpoint else None
    if ck and ck['config']!=config:raise ValueError('Checkpoint architecture mismatch')
    rows,y,tr,x,bdim,normalizer=prepare(a.cache,a.manifest,config,None if ck is None else ck['normalizer']);model=CachedClassifier(bdim)
    if any(r['coordinate_source']=='gt' for r in rows) and not a.allow_gt_control:raise ValueError('GT-centred controls require --allow-gt-control; never label them deployment')
    if not a.allow_gt_control:
        if not a.stage1_provenance:raise ValueError('Supply documented frozen Stage1/outer-group provenance')
        provenance=json.loads(Path(a.stage1_provenance).read_text())
        held_groups={r['group'] for r in rows if r['split']!='train'}
        if not provenance.get('training_centers_are_oof') or not held_groups.issubset(set(provenance.get('excluded_outer_groups',[]))):raise ValueError('Incomplete outer-group exclusion assertions')
        if len(provenance.get('checkpoint_sha256',''))!=64:raise ValueError('Missing Stage1 checkpoint hash')
        # Caller assertions require documentary verification; this does not inspect Stage1 training.
    splits=json.loads(Path(a.roi_splits).read_text());all_rois=[r for rs in splits.values() for r in rs]
    if len(set(all_rois))!=len(all_rois):raise ValueError('ROI leakage across declared splits')
    for r in rows:
        if r['roi'] not in splits.get(r['split'],[]):raise ValueError('Manifest and ROI split disagree')
    def evaluate(split,temperature=1.):
        ix=torch.tensor([r['split']==split for r in rows]);rr=[r for r in rows if r['split']==split]
        with torch.no_grad():z=model(x[ix])/temperature;prob=z.softmax(-1).numpy()
        yy=y[ix];known=yy>=0
        semantic=metrics(yy[known].numpy(),z[known]) if known.any() else None
        return {'v17':evaluate_proposals(rr,prob,a.annotations_root,splits[split]),'conditional_semantic':semantic,'semantic_labeled_denominator':int(known.sum()),'all_proposals':len(rr)},z,yy,rr
    if ck:
        if a.temperature<=0:raise ValueError('Positive temperature required')
        model.load_state_dict(ck['model'],strict=True);m,z,yy,rr=evaluate(a.eval_split,a.temperature);(out/'evaluation.json').write_text(json.dumps(m,indent=2));known=yy>=0
        np.savez_compressed(out/'predictions.npz',logits=z.numpy()[known],labels=yy.numpy()[known],uids=np.array([r['uid'] for r,k in zip(rr,known) if k]));return
    if (out/'history.jsonl').exists():raise FileExistsError('Research run already exists')
    counts=torch.bincount(y[tr],minlength=10).float()
    if (counts==0).any():raise ValueError('Training must contain all ten classes')
    balanced=config['sampler']=='balanced';tau=config['tau']
    if balanced and tau!=0:raise ValueError('Do not stack imbalance corrections')
    q=torch.ones(10)/10 if balanced else counts/counts.sum();loss=LogitAdjustedCE(q,tau)
    sample=WeightedRandomSampler(1/counts[y[tr]],int(tr.sum()),generator=torch.Generator().manual_seed(a.seed)) if balanced else None
    loader=DataLoader(TensorDataset(x[tr],y[tr],torch.arange(int(tr.sum()))),batch_size=64,shuffle=sample is None,sampler=sample,generator=torch.Generator().manual_seed(a.seed))
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8);best=-1.;stale=0
    for epoch in range(1,config['epochs']+1):
        objective,exposure=train_epoch(model,loader,opt,loss,torch.device('cpu'));m,z,yy,rr=evaluate('val')
        with torch.no_grad():mt=metrics(y[tr].numpy(),model(x[tr]))
        rec={'epoch':epoch,'objective':objective,'train':mt,'validation':m,'exposure':exposure,'positive_head_gradient':model.positive_target_row_gradient_mean}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        score=m['v17']['fixed10']['macro_f1'];print(epoch,objective,score,flush=True)
        if score>best+1e-4:
            best=score;stale=0;torch.save({'model':model.state_dict(),'config':config,'normalizer':normalizer,'classes':CLASSES,'epoch':epoch,'train_prior':q.tolist(),'seed':a.seed,'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()},out/'best.pt')
        else:stale+=1
        if stale>=config['patience']:break
if __name__=='__main__':main()
