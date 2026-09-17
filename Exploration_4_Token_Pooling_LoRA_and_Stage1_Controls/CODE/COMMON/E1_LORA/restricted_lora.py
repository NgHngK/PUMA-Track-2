"""Restricted UNI2 Q/V LoRA using immutable cached blocks0â€“19 outputs."""
import sys,time,json,gc,collections,argparse
sys.dont_write_bytecode=True
from bootstrap import *
import os
WEIGHTS=os.environ.get('PUMA_UNI2_WEIGHTS','D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin')
OUTPUT=Path(os.environ.get('PUMA_RUN_OUTPUT','NEW_RUNS'))
sys.path.insert(0,str(Path(__file__).parent))
import torch,numpy as np,psutil
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from model import load_uni2,QVLoRA
from engine import Model,load_data,inputs,extra_metrics,P,norm
from train_eval import seed_all
class Adapted(nn.Module):
 def __init__(self,seed,representation='A3'):
  super().__init__();seed_all(seed);self.head=Model();self.representation=representation
  enc=load_uni2(WEIGHTS).eval().requires_grad_(False);self.blocks=nn.ModuleList(list(enc.blocks[20:]));self.norm=enc.norm;del enc;gc.collect()
  for b in self.blocks:b.attn.qkv=QVLoRA(b.attn.qkv,8,16)
  self.prefix=np.load(P/'TOKEN_AUDIT/CACHE/prefix20.npy',mmap_mode='r');self.geometry=np.load(P/'TARGET_POOLING/REPRESENTATIONS/pooling_geometry.npz')['coordinates']
 def forward(self,ids,b):
  z=torch.from_numpy(np.array(self.prefix[ids],copy=True))
  for block in self.blocks:z=block(z)
  z=self.norm(z);patch=z[:,9:];out=[]
  for j,i in enumerate(ids):
   cx,cy=np.floor(self.geometry[i]).astype(int);ix=[yy*16+xx for yy in range(max(cy-1,0),min(cy+2,16)) for xx in range(max(cx-1,0),min(cx+2,16))];out.append(patch[j,ix].mean(0))
  h=nn.functional.layer_norm(torch.stack(out),(1536,));return self.head(torch.cat((h,b),1))
def main(profile=False):
 torch.set_num_threads(8);torch.set_num_interop_threads(1);rows,cv,y=load_data();ti=np.flatnonzero(cv>=0);vi=np.flatnonzero(cv<0);x,normalizer=inputs({'representation':'A3'},ti);bio=x[:,1536:];proc=psutil.Process()
 for seed in ([17] if profile else [17,29,43]):
  out=OUTPUT/('profile' if profile else f'E1_s{seed}');out.mkdir(parents=True,exist_ok=True)
  if (out/'complete.json').exists():continue
  if (out/'history.jsonl').exists():raise RuntimeError('Partial run must resume explicitly')
  m=Adapted(seed);params=[p for p in m.parameters() if p.requires_grad];adapters=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('blocks')];hp=list(m.head.parameters());cfg={'seed':seed,'epochs':5,'representation':'A3','rank':8,'alpha':16,'blocks':[20,21,22,23],'targets':['Q','V'],'head_lr':.001,'adapter_lr':1e-5,'weight_decay':.01,'microbatch':4,'effective_batch':64,'prefix_cache':'blocks0â€“19 FP32','params':sum(p.numel() for p in params),'adapter_params':sum(p.numel() for p in adapters),'coordinate_source':'GT','train':len(ti),'val':len(vi),'selection':'fixed epoch5; all epochs diagnostic logging'};dump(out/'config.json',cfg)
  if profile:
   ids=ti[:4];t=time.perf_counter();z=m(ids,bio[ids]);forward=time.perf_counter()-t;loss=nn.functional.cross_entropy(z,y[ids]);t=time.perf_counter();loss.backward();backward=time.perf_counter()-t
   with torch.no_grad():expected=m.head(x[ids]);err=float((z-expected).abs().max())
   assert err<2e-5,err
   dump(out/'complete.json',{'forward4_seconds':forward,'backward4_seconds':backward,'initial_frozen_parity_max_error':err,'RSS':proc.memory_info().rss,'config':cfg});print('LORA PROFILE',forward,backward,err,flush=True);return
  counts=torch.bincount(y[ti],minlength=10).float();sampler=WeightedRandomSampler(1/counts[y[ti]],len(ti),generator=torch.Generator().manual_seed(seed));loader=DataLoader(TensorDataset(torch.tensor(ti)),batch_size=64,sampler=sampler,generator=torch.Generator().manual_seed(seed));opt=torch.optim.AdamW([{'params':hp,'lr':.001},{'params':adapters,'lr':1e-5}],weight_decay=.01)
  dump(out/'provenance.json',{'code':sha(Path(__file__)),'prefix_contract':sha(P/'TOKEN_AUDIT/CACHE/contract.json'),'protocol':sha(P/'PROVENANCE/PROTOCOL.md'),'normalizer':normalizer,'frozen_source_hash':'32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'})
  start=time.perf_counter();cpu=time.process_time();peak=proc.memory_info().rss
  for epoch in range(1,6):
   ep=time.perf_counter();epcpu=time.process_time();loss_sum=0;idsall=[];grads=collections.defaultdict(list);steps=0
   m.eval() # disables stochastic encoder behavior; gradients remain enabled
   for (ids_tensor,) in loader:
    ids=ids_tensor.numpy();opt.zero_grad(set_to_none=True);bt=time.perf_counter();batch_loss=0
    for k in range(0,len(ids),4):
     ii=ids[k:k+4];z=m(ii,bio[ii]);loss=nn.functional.cross_entropy(z,y[ii]);assert torch.isfinite(loss);(loss*len(ii)/len(ids)).backward();batch_loss+=float(loss.detach())*len(ii)
    gn=float(nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));g={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None};opt.step();steps+=1;idsall.extend(ids.tolist());loss_sum+=batch_loss
    for n,v in g.items():grads[n].append(v)
    with (out/'batch_history.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'step':steps,'indices':ids.tolist(),'loss':batch_loss/len(ids),'gradient_norm_before_clip':gn,'gradients':g,'seconds':time.perf_counter()-bt,'LRs':[.001,1e-5]})+'\n')
    peak=max(peak,proc.memory_info().rss)
   trainseconds=time.perf_counter()-ep;t=time.perf_counter()
   zz=[]
   with torch.no_grad():
    for k in range(0,len(rows),4):
     ii=np.arange(k,min(k+4,len(rows)));zz.append(m(ii,bio[ii]))
   z=torch.cat(zz);mt=extra_metrics([rows[i] for i in ti],y[ti].numpy(),z[ti]);mv=extra_metrics([rows[i] for i in vi],y[vi].numpy(),z[vi]);expo=np.bincount(y[idsall].numpy(),minlength=10);rois=collections.Counter(rows[i]['roi'] for i in idsall);buckets=collections.Counter(rows[i]['roi']+'|'+str(int(y[i])) for i in idsall)
   rec={'epoch':epoch,'objective':loss_sum/len(idsall),'train':mt,'val':mv,'train_CE':mt['nll'],'val_CE':mv['nll'],'gap':mt['macro_f1']-mv['macro_f1'],'exposure':expo.tolist(),'ROI_exposure':dict(rois),'bucket_exposure':dict(buckets),'repeat_rate':1-len(set(idsall))/len(idsall),'effective_prevalence':(expo/len(idsall)).tolist(),'optimizer_steps':steps,'batches':steps,'microbatch':4,'effective_batch_size':64,'gradient_accumulation':'16 microbatches; final batch normalized by actual size','LRs':[.001,1e-5],'gradients':{k:float(np.mean(v)) for k,v in grads.items()},'adapter_norm':float(torch.sqrt(sum(p.detach().square().sum() for p in adapters))),'parameter_norm':float(torch.sqrt(sum(p.detach().square().sum() for p in params))),'correction_norm':float(m.head.base.diagnostics['correction_norm'].mean()),'train_seconds':trainseconds,'evaluation_seconds':time.perf_counter()-t,'epoch_seconds':time.perf_counter()-ep,'cpu_seconds':time.process_time()-epcpu,'peak_RSS':peak,'checkpoint_status':'epoch saved; final5 selected','early_stopping_counter':'disabled','evaluation_calls':2}
   with (out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   np.savez_compressed(out/f'epoch_{epoch:02}_predictions.npz',indices=vi,logits=z[vi].numpy(),labels=y[vi].numpy(),train_indices=ti,train_logits=z[ti].numpy())
   torch.save({'trainable_state':{n:p.detach() for n,p in m.named_parameters() if p.requires_grad},'optimizer':opt.state_dict(),'epoch':epoch,'config':cfg,'normalizer':normalizer},out/f'epoch_{epoch:02}.pt')
   print('LoRA',seed,epoch,'ROI',mv['puma']['fixed10']['macro_f1'],'seconds',round(time.perf_counter()-ep),flush=True)
  dump(out/'complete.json',{'config':cfg,'selected':rec,'runtime_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu});del m,opt;gc.collect()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');a=ap.parse_args();main(a.profile)
