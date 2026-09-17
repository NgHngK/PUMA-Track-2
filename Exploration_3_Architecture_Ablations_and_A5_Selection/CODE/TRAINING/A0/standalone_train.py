"""Shared finite-family CV runner. Reuses original numerical/evaluation core."""
import sys,csv,json,time,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from torch import nn
from prompt3_models import Architecture
from train import summary_metrics,sha,dump
from train_eval import seed_all,train_epoch
from loss import LogitAdjustedCE
from dataset import read_manifest
C=Path(__file__).parent;R=C.parent
def train_step(model,loader,opt,criterion):
    model.train();total=0.;seen=0;exposure=torch.zeros(10,dtype=torch.long);gs=torch.zeros(10,dtype=torch.float64);captured={}
    head=model.output[-1] if model.kind=='A6' else model.head
    handle=head.register_forward_pre_hook(lambda mod,args:captured.update(h=args[0].detach()))
    for x,y,_ in loader:
        opt.zero_grad(set_to_none=True);z=model(x);loss=criterion(z,y)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        with torch.no_grad():
            gn=(1-z.softmax(-1).gather(1,y[:,None]).squeeze(1))*captured['h'].norm(dim=1);gs.scatter_add_(0,y,gn.double());exposure+=torch.bincount(y,minlength=10)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(y);seen+=len(y)
    handle.remove();model.positive_target_row_gradient_mean=(gs/exposure.clamp_min(1)).tolist();return total/seen,exposure.tolist()
def run(cfg):
    out=Path(cfg['output_root'])/cfg['id']
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    if out.exists():raise FileExistsError('Partial run preserved: '+str(out))
    torch.set_num_threads(2);rows=read_manifest(R/'01_sample_definition/sample_manifest.csv');y=torch.tensor([r['label'] for r in rows]);spl=list(csv.DictReader((R/'01_sample_definition/prompt3_cv.csv').open()));cv=np.array([int(r['cv_fold']) for r in spl]);fold=cfg['fold'];seed=cfg['seed']
    ti=np.flatnonzero((cv>=0)&(cv!=fold)) if fold>=0 else np.flatnonzero(cv>=0);vi=np.flatnonzero(cv==fold) if fold>=0 else np.flatnonzero(cv<0)
    h=nn.functional.layer_norm(torch.from_numpy(np.load(R/'02_cache/uni2_features/gt_fov96/features.npy')),(1536,));a=np.load(R/'02_cache/biology/tierA.npy');kind=cfg['architecture'];control=cfg.get('control','real');normalizer={}
    b=a.copy()
    if kind.startswith('B'):
        bb=np.load(R/'02_cache/biology/tierB.npy');cols=list(range(8)) if kind=='B1' else (list(range(8,14)) if kind=='B2' else list(range(20)));bb=bb[:,cols]
        if control=='placebo':bb=np.tanh(h.numpy()@np.random.default_rng(1701).normal(0,1/np.sqrt(1536),(1536,bb.shape[1])).astype('float32'))
        if control=='shuffle':bb=bb.copy();bb[ti]=bb[np.random.default_rng(seed+10000).permutation(ti)]
        b=np.concatenate((a,bb),1) if kind=='B4' else bb
    else:
        if control=='placebo':b=np.tanh(h.numpy()@np.random.default_rng(1701).normal(0,1/np.sqrt(1536),(1536,16)).astype('float32'))
        if control=='shuffle':b[ti]=b[np.random.default_rng(seed+10000).permutation(ti)]
    mean=b[ti].mean(0);std=np.maximum(b[ti].std(0),1e-6);b=torch.from_numpy(((b-mean)/std).astype('float32'));x=b if kind=='B0' else (h if kind=='A0' else torch.cat((h,b),1))
    seed_all(seed);m=Architecture(kind,b.shape[1]);counts=torch.bincount(y[ti],minlength=10).float();assert(counts>0).all();criterion=LogitAdjustedCE(torch.ones(10)/10,0)
    sampler=WeightedRandomSampler(1/counts[y[ti]],len(ti),generator=torch.Generator().manual_seed(seed));loader=DataLoader(TensorDataset(x[ti],y[ti],torch.arange(len(ti))),batch_size=64,sampler=sampler,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);out.mkdir(parents=True);dump(out/'config.json',cfg)
    sources={str(p.relative_to(C)):sha(p) for p in C.rglob('*.py')};dump(out/'provenance.json',{'sources':sources,'manifest':sha(R/'01_sample_definition/sample_manifest.csv'),'cv':sha(R/'01_sample_definition/prompt3_cv.csv'),'features':sha(R/'02_cache/uni2_features/gt_fov96/features.npy'),'tierA':sha(R/'02_cache/biology/tierA.npy'),'tierB':sha(R/'02_cache/biology/tierB.npy') if kind.startswith('B') else None,'protocol':sha(R/'00_reference/EXPLORATION3_PROTOCOL.md'),'upstream_contaminated':kind.startswith('B')})
    normalizer={'mean':mean.tolist(),'std':std.tolist()};dump(out/'normalizer.json',normalizer);(out/'command.txt').write_text('python 01_shared_core/exploration3_train.py --config experiments/'+cfg['id']+'/config.json\n');(out/'hypothesis.md').write_text('Predeclared '+kind+' '+control+'; see EXPLORATION3_PROTOCOL.md.\n')
    start=time.perf_counter();cpu=time.process_time();history=[]
    for epoch in range(1,11):
        objective,exposure=train_step(m,loader,opt,criterion)
        with torch.no_grad():
            m.eval();zt=m(x[ti]);zv=m(x[vi]);diag={k:v.cpu().tolist() for k,v in m.diagnostics.items()};mt=summary_metrics([rows[i] for i in ti],y[ti].numpy(),zt);mv=summary_metrics([rows[i] for i in vi],y[vi].numpy(),zv)
        gradient=m.positive_target_row_gradient_mean
        rec={'epoch':epoch,'objective':objective,'train_CE':mt['nll'],'val_CE':mv['nll'],'train':mt,'val':mv,'lr':.001,'exposure':exposure,'positive_target_head_gradient':gradient,'gradient_scope':'ten-class output head positive target contribution','gap':mt['macro_f1']-mv['macro_f1'],'diagnostics':diag};history.append(rec)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    np.savez_compressed(out/'predictions.npz',indices=vi,labels=y[vi].numpy(),logits=zv.numpy(),uids=np.array([rows[i]['uid'] for i in vi]));torch.save({'model':m.state_dict(),'config':cfg,'normalizer':normalizer},out/'model.pt')
    summary={'id':cfg['id'],'config':cfg,'selected':rec,'runtime_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'trainable_parameters':sum(p.numel() for p in m.parameters()),'selection':'fixed epoch10; no per-fold best selection'};dump(out/'summary.json',summary)
    print(cfg['id'],round(mv['puma']['fixed10']['macro_f1'],4),flush=True);return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
