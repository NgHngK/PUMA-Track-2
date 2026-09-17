"""New GT-prompt-only mask model, excluding historicalfold0 at every train dependency."""
import sys,time,json,importlib.util,collections
sys.dont_write_bytecode=True
from bootstrap import *
import torch,numpy as np,psutil
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from engine import load_data,P
A=OLD/'00_reference/biomask_audit';D=P/'BIOMASK_GUIDED';O=D/'CLEAN_GT_ONLY'
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def prepare():
 rows,cv,y=load_data();rm=np.load(A/'roi_manifest.npy');folds=np.load(A/'folds.npy');mp=dict(zip(rm['roi_id'],folds));f=np.array([mp[r['roi']] for r in rows]);ix=np.flatnonzero(f!=0);rgb=np.load(A/'rgb_crops_qc.npy');gt=np.load(A/'gt_masks_diagnostic.npy');crop=load('clean_crop',A/'source/crops.py');prompts=[];valid=[];trans=[];sizes={}
 for r in rows:
  if r['image'] not in sizes:
   with Image.open(r['image']) as im:sizes[r['image']]=im.size
  width,height=sizes[r['image']];tr=crop.CropTransform.from_center(r['x'],r['y'],96,(height,width,3));yy,xx=np.indices((96,96),dtype='float32');prompts.append(np.exp(-((xx-(r['x']-tr.crop_left))**2+(yy-(r['y']-tr.crop_top))**2)/(2*2.5**2)));valid.append(tr.valid_region_mask());trans.append([tr.crop_left,tr.crop_top])
 x=torch.from_numpy(np.concatenate([rgb.transpose(0,3,1,2).astype('float32')/255,np.array(prompts)[:,None]],1));g=torch.from_numpy(gt[:,None].astype('float32'));v=torch.from_numpy(np.array(valid)[:,None].astype('float32'));return rows,ix,np.flatnonzero(f==0),x,g,v,np.array(trans)
def main():
 torch.set_num_threads(4);torch.manual_seed(17);O.mkdir(exist_ok=True)
 if (O/'complete.json').exists():print('Clean mask already complete');return
 if (O/'history.jsonl').exists():raise RuntimeError('Partial mask run requires explicit resume')
 rows,tr,va,x,g,valid,trans=prepare();bm=load('clean_mask_model',A/'source/biomask.py');m=bm.ProposalBiologyNetwork(32)
 for name,p in m.named_parameters():
  if any(name.startswith(prefix) for prefix in ['presence_','quality_head','center_head']):p.requires_grad_(False)
 params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4);loader=DataLoader(TensorDataset(torch.tensor(tr)),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(17))
 config={'seed':17,'epochs':10,'batch':16,'LR':3e-4,'WD':1e-4,'loss':'.6 validpixelBCE+.4 softDice','train_n':len(tr),'held_n':len(va),'train_indices':tr.tolist(),'excluded_fold':0,'parameter_count_total':sum(p.numel() for p in m.parameters()),'trainable_mask_parameters':sum(p.numel() for p in params),'auxiliary_heads':'frozen and excluded from objective','Stage1_training_input':False,'pretraining':False,'augmentation':'none','validation_selection':'none; fixed epoch10'};dump(O/'config.json',config);dump(O/'provenance.json',{'source':sha(A/'source/biomask.py'),'GT_masks':sha(A/'gt_masks_diagnostic.npy'),'RGB_crops':sha(A/'rgb_crops_qc.npy'),'source_foldmap':sha(A/'folds.npy'),'training_ROIs':sorted({rows[i]['roi'] for i in tr}),'excluded_ROIs':sorted({rows[i]['roi'] for i in va}),'protocol':sha(P/'PROVENANCE/ALIGNMENT_MASK_PROTOCOL.md')})
 proc=psutil.Process();start=time.perf_counter();cpu=time.process_time()
 for epoch in range(1,11):
  t=time.perf_counter();losses=[];dice=[];steps=0;grads=[];m.train()
  for (ids,) in loader:
   bt=time.perf_counter();opt.zero_grad(set_to_none=True);z=m(x[ids],valid_mask=valid[ids])['mask_logits'];prob=z.sigmoid()*valid[ids];gt=g[ids]*valid[ids];bce=(nn.functional.binary_cross_entropy_with_logits(z,gt,reduction='none')*valid[ids]).sum((1,2,3))/valid[ids].sum((1,2,3)).clamp_min(1);dd=(2*(prob*gt).sum((1,2,3))+1)/(prob.sum((1,2,3))+gt.sum((1,2,3))+1);loss=(.6*bce+.4*(1-dd)).mean();assert torch.isfinite(loss);loss.backward();gn=float(nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));opt.step();steps+=1;losses.extend([float(loss.detach())]*len(ids));dice.extend(dd.detach().tolist());grads.append(gn)
   with (O/'batch_history.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'step':steps,'indices':ids.tolist(),'loss':float(loss.detach()),'gradient_norm':gn,'seconds':time.perf_counter()-bt})+'\n')
  rec={'epoch':epoch,'objective':float(np.mean(losses)),'train_soft_dice':float(np.mean(dice)),'seconds':time.perf_counter()-t,'steps':steps,'samples_seen':len(tr),'unique_ROIs':len({rows[i]['roi'] for i in tr}),'gradient_norm':float(np.mean(grads)),'LR':3e-4,'RSS':proc.memory_info().rss,'validation_metrics':'not evaluated until fixed final model','checkpoint':'fixed10only'}
  with (O/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  print('clean BioMask',epoch,rec['objective'],flush=True)
 torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'config':config},O/'final.pt');m.eval();out=[]
 with torch.no_grad():
  for k in range(0,len(rows),16):out.append((m(x[k:k+16],valid_mask=valid[k:k+16])['mask_logits'].sigmoid()*valid[k:k+16]).numpy())
 masks=np.concatenate(out)[:,0];np.save(O/'masks.npy',masks);np.save(O/'transforms.npy',trans);np.save(O/'valid.npy',valid.numpy());q=[]
 for i in va:
  p=masks[i]>.5;gt=g[i,0].numpy()>0;inter=(p&gt).sum();q.append({'index':int(i),'uid':rows[i]['uid'],'class':rows[i]['class_name'],'Dice':float(2*inter/max(p.sum()+gt.sum(),1)),'IoU':float(inter/max((p|gt).sum(),1)),'predicted_area':int(p.sum()),'GT_area':int(gt.sum())})
 csvout(O/'held_fold0_mask_quality.csv',q);dump(O/'complete.json',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'held_Dice_mean':float(np.mean([r['Dice'] for r in q])),'held_IoU_mean':float(np.mean([r['IoU'] for r in q])),'checkpoint_sha256':sha(O/'final.pt'),'mask_sha256':sha(O/'masks.npy'),'config':config});print('CLEANMASK_COMPLETE',flush=True)
if __name__=='__main__':main()
