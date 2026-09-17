import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from clean_mask import prepare,load
p=argparse.ArgumentParser(description="Evaluate the clean mask checkpoint on the excluded fold0 nuclei")
p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--reference-masks');a=p.parse_args();torch.set_num_threads(4)
rows,tr,va,x,g,valid,trans=prepare();source=Path(__file__).parent/'mask_source/biomask.py';bm=load('mask_eval_model',source);m=bm.ProposalBiologyNetwork(32);ck=torch.load(a.checkpoint,weights_only=True,map_location='cpu');m.load_state_dict(ck['model'],strict=True);m.eval();pred=[]
with torch.inference_mode():
 for k in range(0,len(va),16):
  ids=va[k:k+16];pred.append((m(x[ids],valid_mask=valid[ids])['mask_logits'].sigmoid()*valid[ids]).numpy()[:,0])
prob=np.concatenate(pred);quality=[]
for j,i in enumerate(va):
 mask=prob[j]>.5;gt=g[i,0].numpy()>0;inter=(mask&gt).sum();quality.append({'index':int(i),'uid':rows[i]['uid'],'Dice':float(2*inter/max(mask.sum()+gt.sum(),1)),'IoU':float(inter/max((mask|gt).sum(),1))})
result={'held_n':len(va),'Dice_mean':float(np.mean([v['Dice'] for v in quality])),'IoU_mean':float(np.mean([v['IoU'] for v in quality])),'per_nucleus':quality,'checkpoint_strict_load':True}
if a.reference_masks:
 ref=np.load(a.reference_masks)[va];err=float(np.abs(ref-prob).max());result['reference_max_error']=err;result['reference_pass']=bool(np.allclose(ref,prob,atol=2e-5,rtol=1e-5));assert result['reference_pass'],err
Path(a.output).write_text(json.dumps(result,indent=2));print(result['Dice_mean'],result.get('reference_max_error'))
