"""One immutable cached-head trainer; experiments differ through JSON configuration."""
import argparse,csv,datetime,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np,torch,psutil
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from dataset import read_manifest,CLASSES
from fusion import CachedClassifier
from loss import LogitAdjustedCE
from train_eval import seed_all,metrics as semantic_metrics,train_epoch
from metrics import puma_metrics
CORE=Path(__file__).parent;R=CORE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def summary_metrics(rows,y,z):
    s=semantic_metrics(y,z);s['macro_precision']=float(np.mean(s['precision']));s['macro_recall']=float(np.mean(s['recall']))
    s['true_prevalence']=(np.asarray(s['support'])/len(y)).tolist();s['predicted_prevalence']=(np.asarray(s['predicted'])/len(y)).tolist()
    s['puma']=puma_metrics(rows,y,z.softmax(-1).numpy());return s
def run(config):
    torch.set_num_threads(2);seed=int(config.get('seed',17));seed_all(seed)
    manifest=R/'01_sample_definition/sample_manifest.csv';rows=read_manifest(manifest)
    y=torch.tensor([r['label'] for r in rows]);tr=torch.tensor([r['split']=='train' for r in rows]);trrows=[r for r in rows if r['split']=='train'];vr=[r for r in rows if r['split']=='val']
    fov=int(config.get('fov',96));cache=R/'02_cache/uni2_features'/('gt_fov'+str(fov));bio=config.get('biology','none');only=config.get('biology_only',False)
    hashes={str(p.relative_to(CORE)):sha(p) for p in CORE.rglob('*.py')};module_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    extra=None;normalizer={};schema_hash=None;cache_hash=None
    if not only:
        contract=json.loads((cache/'contract.json').read_text());assert contract['manifest_sha256']==sha(manifest)
        assert np.load(cache/'done.npy').all();x=torch.from_numpy(np.load(cache/'features.npy').copy());x=nn.functional.layer_norm(x,(1536,));cache_hash=sha(cache/'features.npy')
    if bio!='none':
        file='oracle' if bio=='oracle' else 'tierA';schema_path=R/'02_cache/biology'/(file+'_schema.json');schema=json.loads(schema_path.read_text());assert schema['manifest_sha256']==sha(manifest);schema_hash=sha(schema_path)
        b=np.load(R/'02_cache/biology'/(file+'.npy')).copy();names=schema['names'];keep=list(range(b.shape[1]))
        if config.get('families'):
            valid=sum([schema['families'][f] for f in config['families']],[]);keep=[i for i,n in enumerate(names) if n in valid];b=b[:,keep]
        if bio=='placebo':
            rng=np.random.default_rng(1701);projection=rng.normal(0,1/np.sqrt(1536),(1536,b.shape[1])).astype(np.float32)
            b=np.tanh(x.numpy()@projection)
        mean=b[tr.numpy()].mean(0);std=np.maximum(b[tr.numpy()].std(0),1e-6);b=(b-mean)/std
        normalizer={'mean':mean.tolist(),'std':std.tolist(),'columns':keep,'names':[names[i] for i in keep]}
        if bio=='shuffle':
            ix=np.flatnonzero(tr.numpy());b[ix]=b[np.random.default_rng(seed+10000).permutation(ix)]
        extra=torch.from_numpy(b.astype(np.float32));x=extra if only else torch.cat((x,extra),1)
    seed_all(seed);m=CachedClassifier(0 if extra is None else extra.shape[1],only)
    counts=torch.bincount(y[tr],minlength=10).float();balanced=config.get('sampler','natural')=='balanced';tau=float(config.get('tau',1))
    if balanced and tau!=0:raise ValueError('No stacked prior correction')
    prior=torch.ones(10)/10 if balanced else counts/counts.sum();criterion=LogitAdjustedCE(prior,tau)
    sampler=WeightedRandomSampler(1/counts[y[tr]],int(tr.sum()),replacement=True,generator=torch.Generator().manual_seed(seed)) if balanced else None
    loader=DataLoader(TensorDataset(x[tr],y[tr],torch.arange(int(tr.sum()))),batch_size=64,shuffle=sampler is None,sampler=sampler,generator=torch.Generator().manual_seed(seed))
    out=Path(config['output_root'])/config['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics/summary.json').exists():return json.loads((out/'metrics/summary.json').read_text())
    if (out/'logs/history.jsonl').exists():raise FileExistsError('Incomplete run preserved; use explicit new experiment ID or supported resume')
    for d in ['logs','checkpoints','predictions','metrics']: (out/d).mkdir(exist_ok=True)
    dump(out/'config.json',config);command=f'python 01_shared_core/train.py --config experiments/{config["id"]}/config.json'
    (out/'command.txt').write_text(command+'\n');(out/'hypothesis.md').write_text(config.get('hypothesis','Controlled cached-feature comparison')+'\n')
    (out/'README.md').write_text('Executed fixed-view development probe. Shared implementation in 01_shared_core; GT-centred subset V17 evaluation, not end-to-end deployment.\n')
    provenance={'experiment_id':config['id'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shared_core_hash':module_hash,'modules':hashes,
        'evaluator_hash':sha(CORE/'puma_v17_evaluator/evaluation/public_matcher.py'),'manifest_hash':sha(manifest),'UNI2_checkpoint_hash':json.loads((R/'00_reference/RESEARCH_STATE.json').read_text())['checkpoint_sha256'],
        'biology_schema_hash':schema_hash,'feature_cache_hash':cache_hash,'config_hash':sha(out/'config.json'),'seed':seed,'command':command,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'device':'CPU; two head-training threads','coordinate_source':'GT area-centroid crop; V17 exterior-component GT','FOV':fov,'split':'preserved prior ROI development split; patient separation unverified'}
    dump(out/'provenance.json',provenance);dump(out/'normalizer.json',normalizer)
    lr=float(config.get('lr',.001));wd=float(config.get('weight_decay',.01));opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    hist=[];best=-1.;start=time.perf_counter();cpu=time.process_time();peak=0;proc=psutil.Process()
    for epoch in range(1,int(config.get('epochs',10))+1):
        t=time.perf_counter();objective,exposure=train_epoch(m,loader,opt,criterion,torch.device('cpu'));train_secs=time.perf_counter()-t;t=time.perf_counter()
        with torch.no_grad():
            m.eval();zt=m(x[tr]);zv=m(x[~tr]);mt=summary_metrics(trrows,y[tr].numpy(),zt);mv=summary_metrics(vr,y[~tr].numpy(),zv)
            train_loss=float(criterion(zt,y[tr]));val_loss=float(criterion(zv,y[~tr]));correction_norm=float(m.correction(x[:,1536:]).norm(dim=1).mean()) if m.correction is not None else 0.
        rec={'epoch':epoch,'objective':objective,'train_loss':train_loss,'validation_loss':val_loss,'train':mt,'val':mv,'generalization_gap':mt['macro_f1']-mv['macro_f1'],'exposure':exposure,'positive_target_head_gradient':m.positive_target_row_gradient_mean,'lr':lr,'train_seconds':train_secs,'evaluation_seconds':time.perf_counter()-t,'correction_norm':correction_norm,'biology_feature_norm':float(extra.norm(dim=1).mean()) if extra is not None else 0.}
        hist.append(rec)
        with (out/'logs/history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        score=mv['puma']['fixed10']['macro_f1'];peak=max(peak,proc.memory_info().rss)
        if score>best+1e-12:
            best=score;dump(out/'metrics/best.json',rec);torch.save({'model':m.state_dict(),'config':config,'normalizer':normalizer,'provenance':provenance},out/'checkpoints/best.pt')
            np.savez_compressed(out/'predictions/best.npz',labels=y[~tr].numpy(),logits=zv.numpy(),uids=np.array([r['uid'] for r in vr]),train_logits=zt.numpy())
        if epoch==int(config.get('epochs',10)):
            dump(out/'metrics/final.json',rec);np.savez_compressed(out/'predictions/final.npz',labels=y[~tr].numpy(),logits=zv.numpy(),uids=np.array([r['uid'] for r in vr]))
    selected=json.loads((out/'metrics/best.json').read_text());s={'id':config['id'],'config':config,'selected':selected,'runtime_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_sampled_rss_gb':peak/1e9,'trainable_parameters':sum(p.numel() for p in m.parameters()),'best_semantic_epoch':max(hist,key=lambda r:r['val']['macro_f1'])['epoch'],'best_pooled_epoch':max(hist,key=lambda r:r['val']['puma']['summed']['macro_f1'])['epoch'],'best_roi_epoch':selected['epoch']}
    dump(out/'metrics/summary.json',s)
    print(config['id'],'roi',round(best,4),'pooled',round(selected['val']['puma']['summed']['macro_f1'],4),'semantic',round(selected['val']['macro_f1'],4),'seconds',round(s['runtime_seconds'],1),flush=True)
    return s
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
