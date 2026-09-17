import sys,importlib.util,collections,json
sys.dont_write_bytecode=True
from bootstrap import *
import numpy as np,torch
from engine import load_data as original_data,norm,P,extra_metrics,Model
def prepare():
 rows,cv,y=original_data();pairs=json.loads((P/'STAGE1_PROPOSALS/paired450.json').read_text());selected=[v['index'] for v in pairs if v['status']=='matched' and v['historical_fold']==0];rr=[rows[i].copy() for i in selected];yy=y[selected];cc=cv[selected].copy();cc[cc>=0]=0
 cache=P/'STAGE1_PROPOSALS/CACHE';srows=json.loads((cache/'manifest.json').read_text());mapping={int(k):i for i,k in enumerate(np.load(cache/'original_indices.npy'))};si=np.array([mapping[i] for i in selected]);stage=[]
 for i,j in zip(selected,si):
  r=rows[i].copy();r.update(x=srows[j]['x'],y=srows[j]['y'],coordinate_source='stage1_oof');stage.append(r)
 hgt=norm(np.load(P/'TARGET_POOLING/REPRESENTATIONS/A3.npy')[selected]);bg=np.load(OLD/'02_cache/biology/tierA.npy')[selected];bs=np.load(cache/'tierA.npy')[si];patch=np.load(cache/'patch.npy',mmap_mode='r');hs=[]
 for j,r in zip(si,stage):
  cx,cy=np.floor([(r[k]-np.floor(r[k]-48+.5))/6 for k in ['x','y']]).astype(int);near=[yy*16+xx for yy in range(cy-1,cy+2) for xx in range(cx-1,cx+2)];hs.append(patch[j,near].mean(0))
 hs=norm(np.array(hs));dump(P/'STAGE1_PROPOSALS/clean_cohort.json',{'original_indices':selected,'stage1_cache_indices':si.tolist(),'train':int((cc>=0).sum()),'val':int((cc<0).sum()),'train_counts':np.bincount(yy[cc>=0],minlength=10).tolist(),'val_counts':np.bincount(yy[cc<0],minlength=10).tolist(),'excluded_detector_training_fold':0,'source_checkpoint_hash':sha(W/'stage1_final_fold0.pt')});return rr,stage,cc,yy,hgt,hs,bg,bs
if __name__=='__main__':
 src=(W/'engine.py').read_text();src=src.replace("  np.savez_compressed(out/f'epoch_", "  additional_evaluation(m,ti,vi,epoch,out)\n  np.savez_compressed(out/f'epoch_")
 target=W/'alignment_engine.py'
 if target.exists():assert target.read_text()==src
 else:target.write_text(src)
 spec=importlib.util.spec_from_file_location('alignment_engine',target);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
 gt,st,cv,y,hg,hs,bg,bs=prepare();ti=np.flatnonzero(cv>=0);vi=np.flatnonzero(cv<0)
 for kind in ['C_GTTRAIN','C_STAGE1TRAIN']:
  train_stage=kind=='C_STAGE1TRAIN';rows=st if train_stage else gt;bb=bs if train_stage else bg;mean=bb[ti].mean(0);std=np.maximum(bb[ti].std(0),1e-6);xgt=torch.cat((hg,torch.from_numpy(((bg-mean)/std).astype('float32'))),1);xst=torch.cat((hs,torch.from_numpy(((bs-mean)/std).astype('float32'))),1)
  e.load_data=lambda:(rows,cv,y)
  e.inputs=lambda cfg,ti:((xst if train_stage else xgt),{'mean':mean.tolist(),'std':std.tolist()})
  def extra(m,ti,vi,epoch,out):
   with torch.no_grad():zgt=m(xgt[vi]);zs=m(xst[vi]);mg=extra_metrics([gt[i] for i in vi],y[vi].numpy(),zgt);ms=extra_metrics([st[i] for i in vi],y[vi].numpy(),zs)
   rec={'epoch':epoch,'GT_centered':mg,'Stage1_centered':ms,'population':'matched conditional, historicalfold0; not fullproposal end-to-end','paired_semantic_delta':ms['macro_f1']-mg['macro_f1'],'paired_ROI_delta':ms['puma']['fixed10']['macro_f1']-mg['puma']['fixed10']['macro_f1']}
   with (out/'paired_evaluation.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   np.savez_compressed(out/f'paired_epoch_{epoch:02}.npz',indices=vi,labels=y[vi].numpy(),gt_logits=zgt.numpy(),stage1_logits=zs.numpy())
  e.additional_evaluation=extra
  for seed in [17,29,43]:e.run({'id':f'{kind}_s{seed}','family':kind,'seed':seed,'fold':-1,'epochs':10,'sampler':'balanced','population':'matched historicalfold0 originalsplit','train_coordinates':'Stage1' if train_stage else 'GT','representation':'A3'})
 for f in ['stage1_final_fold0.pt','stage1_checkpoint_metadata.json']:cp(W/f,P/'STAGE1_PROPOSALS'/f)
 print('C_ALIGNMENT_COMPLETE',flush=True)
