"""Train/evaluate Stage 2. No Stage-1 modifications; no implicit test-set selection."""
import argparse, csv, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from exploration1_dataset import CLASSES,NucleiDataset,read_manifest,sampling
from prompt1_loss import LogitAdjustedCE
from prompt1_model import Stage2,checkpoint_sha256

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def metrics(y, logits):
    y=np.asarray(y);logits=torch.as_tensor(logits,dtype=torch.float32)
    p=logits.softmax(-1).numpy();pred=p.argmax(-1)
    cm=np.bincount(10*y+pred,minlength=100).reshape(10,10)
    tp=cm.diagonal();support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(10,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(10,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(10,dtype=float),where=(support+predicted)>0)
    conf=p.max(1);correct=pred==y;ece=0.
    bins=np.minimum((conf*15).astype(int),14)
    for b in range(15):
        ix=bins==b
        if ix.any():ece+=ix.mean()*abs(conf[ix].mean()-correct[ix].mean())
    result={'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[support>0].mean()),
            'balanced_accuracy_fixed10':float(recall.mean()),'accuracy':float(correct.mean()),'ece15':float(ece),
            'nll':float(F.cross_entropy(logits,torch.as_tensor(y).long())),
            'support':support.tolist(),'predicted':predicted.tolist(),'precision':precision.tolist(),
            'recall':recall.tolist(),'f1':f1.tolist(),'confusion':cm.tolist(),
            'missing_classes':[CLASSES[i] for i in range(10) if support[i]==0]}
    return result

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];zs=[];ids=[]
    for x,y,i in loader:
        z=model(x.to(device));ys.append(y);zs.append(z.float().cpu());ids.append(i)
    if not ys:raise ValueError('Empty evaluation split')
    y=torch.cat(ys);z=torch.cat(zs);ix=torch.cat(ids)
    return metrics(y.numpy(),z),y,z,ix

def train_epoch(model,loader,optimizer,criterion,device,accum=1,amp=False,scheduler=None):
    """Sum gradients then divide by actual window sample count, including remainder."""
    model.train();optimizer.zero_grad(set_to_none=True)
    total=0.;seen=0;window=0;exposure=torch.zeros(10,dtype=torch.long)
    use_amp=amp and device.type=='cuda'
    dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp and dtype==torch.float16)
    trainable=[p for p in model.parameters() if p.requires_grad]
    positive_grad_sum=torch.zeros(10,dtype=torch.float64)
    captured={}
    head=getattr(model,'head',model if isinstance(model,nn.Linear) else None)
    handle=head.register_forward_pre_hook(lambda module,args:captured.update(h=args[0].detach())) if head is not None else None
    for step,(x,y,_) in enumerate(loader):
        x=x.to(device);y=y.to(device);n=len(y)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
            logits=model(x)
            loss=criterion(logits,y,reduction='sum')
        if 'h' in captured:
            with torch.no_grad():
                ptrue=(logits.float()+criterion.tau*criterion.log_prior).softmax(-1).gather(1,y[:,None]).squeeze(1)
                # Norm of each positive example's target-row head gradient before reduction.
                gn=(1-ptrue)*captured.pop('h').float().norm(dim=-1)
                positive_grad_sum.scatter_add_(0,y.cpu(),gn.double().cpu())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward();window+=n;total+=loss.item();seen+=n
        exposure+=torch.bincount(y.cpu(),minlength=10)
        if (step+1)%accum==0 or step+1==len(loader):
            scaler.unscale_(optimizer)
            for p in trainable:
                if p.grad is not None:p.grad.div_(window)
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scheduler is not None and scaler.get_scale()>=old_scale:scheduler.step()
            optimizer.zero_grad(set_to_none=True);window=0
    if handle is not None:handle.remove()
    model.positive_target_row_gradient_mean=(positive_grad_sum/exposure.clamp_min(1)).tolist()
    return total/seen,exposure.tolist()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['linear','lora'],default='linear');p.add_argument('--sampler',choices=['natural','balanced'],default='natural')
    p.add_argument('--tau',type=float,default=1.);p.add_argument('--fov',type=int,default=96)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--head-lr',type=float,default=1e-3);p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=float,default=16.);p.add_argument('--last-blocks',type=int,default=4)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--amp',action='store_true')
    p.add_argument('--checkpointing',action='store_true');p.add_argument('--stain-strength',type=float,default=0.)
    p.add_argument('--eval-checkpoint');p.add_argument('--eval-split',choices=['val','calibration','test'],default='val')
    a=p.parse_args()
    if a.accum<1 or a.epochs<1 or a.batch_size<1:raise ValueError('Positive batch/epoch/accum required')
    if a.sampler=='balanced' and a.tau!=0:raise ValueError('Use balanced sampler with tau=0; natural sampler with tau=0 or 1')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seed_all(a.seed);device=torch.device(a.device)
    rows=read_manifest(a.manifest)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    sampler,q=sampling(train,a.sampler,a.seed)
    criterion=LogitAdjustedCE(q,a.tau).to(device)
    model=Stage2(a.weights,a.mode,a.rank,a.alpha,a.last_blocks,a.checkpointing).to(device)
    seed_all(a.seed)  # Augmentation stream must not depend on adapter initialization.
    def loader(rs,augment=False,sample=None):
        ds=NucleiDataset(rs,a.fov,augment=augment,stain_strength=a.stain_strength)
        return DataLoader(ds,batch_size=a.batch_size,shuffle=augment and sample is None,sampler=sample,num_workers=a.workers,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(a.seed))
    config=vars(a).copy();config['classes']=CLASSES;config['sampling_prior']=q.tolist()
    config['manifest_sha256']=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    config['weights_sha256']=checkpoint_sha256(a.weights)
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if a.eval_checkpoint:
        ck=torch.load(a.eval_checkpoint,map_location=device,weights_only=True)
        if ck['config']['classes']!=CLASSES:raise ValueError('Checkpoint class map mismatch')
        if ck['config']['weights_sha256']!=config['weights_sha256']:raise ValueError('Wrong foundation checkpoint hash')
        for key in ['mode','rank','alpha','last_blocks','fov']:
            if ck['config'][key]!=config[key]:raise ValueError(f'Checkpoint configuration mismatch: {key}')
        current=model.state_dict();current.update(ck['trainable']);model.load_state_dict(current,strict=True)
        rs=[r for r in rows if r['split']==a.eval_split]
        m,y,z,ix=evaluate(model,loader(rs),device)
        (out/'evaluation.json').write_text(json.dumps(m,indent=2))
        np.savez_compressed(out/'predictions.npz',labels=y.numpy(),logits=z.numpy(),uids=np.array([rs[i]['uid'] for i in ix]))
        return
    if (out/'history.jsonl').exists():raise FileExistsError('Use a fresh output directory; never overwrite a research run')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    groups=[]
    for role,lr in [('encoder',a.backbone_lr),('head',a.head_lr)]:
        for decay in [0.,.01]:
            params=[param for name,param in model.named_parameters() if param.requires_grad and name.startswith(role) and ((param.ndim>1)==(decay>0))]
            if params:groups.append(dict(params=params,lr=lr,weight_decay=decay))
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    tl=loader(train,True,sampler);vl=loader(val)
    total_steps=a.epochs*math.ceil(len(tl)/a.accum);warmup=max(1,round(.05*total_steps))
    def lr_factor(step):
        if step<warmup:return (step+1)/warmup
        progress=min(1.,(step-warmup)/max(1,total_steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    best=-1.;stale=0
    for epoch in range(1,a.epochs+1):
        loss,exposure=train_epoch(model,tl,optimizer,criterion,device,a.accum,a.amp,scheduler)
        m,y,z,ix=evaluate(model,vl,device)
        rec=dict(epoch=epoch,objective=loss,class_exposure=exposure,validation=m,
                 positive_example_target_row_gradient=model.positive_target_row_gradient_mean)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps({'epoch':epoch,'loss':loss,'val_macro_f1':m['macro_f1'],'zero_recall':[CLASSES[k] for k in range(10) if m['recall'][k]==0]}),flush=True)
        if m['macro_f1']>best+1e-4:
            best=m['macro_f1'];stale=0
            names={name for name,param in model.named_parameters() if param.requires_grad}
            ck={'config':config,'epoch':epoch,'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}}
            torch.save(ck,out/'best.pt')
            np.savez_compressed(out/'best_validation.npz',labels=y.numpy(),logits=z.numpy(),uids=np.array([val[i]['uid'] for i in ix]))
        else:stale+=1
        if stale>=a.patience:break

if __name__=='__main__':main()
