"""Run in a fresh interpreter per package; no training or result mutation."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
package=Path(sys.argv[1]).resolve();sys.path.insert(0,str(package/'CODE'))
import numpy as np,torch
torch.set_num_threads(1)
import engine
from torch import nn
from prompt3_models import Architecture
rows,cv,y=engine.load_data();results=[]
for run in sorted((package/'RESULTS/RUNS').iterdir()):
 if not (run/'final.pt').exists():continue
 ck=torch.load(run/'final.pt',weights_only=True,map_location='cpu');cfg=ck['config'];fold=cfg.get('fold',0)
 if package.name.startswith('F_'):
  ix=np.array(json.loads((engine.P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text())['original_indices']);bio=np.load(engine.OLD/'02_cache/biology/tierA.npy')[ix];mean=np.array(ck['normalizer']['mean'],dtype='float32');std=np.array(ck['normalizer']['std'],dtype='float32');h=torch.from_numpy(np.load(engine.P/'BIOMASK_GUIDED/REPRESENTATIONS'/(cfg['family']+'.npy')));x=torch.cat([h,torch.from_numpy(((bio-mean)/std).astype('float32'))],1)
 elif package.name=='C_ALIGNMENT':
  from alignment_data import prepare
  gt,st,cc,yy,hg,hs,bg,bs=prepare();is_stage=cfg['family']=='C_STAGE1TRAIN';b=bs if is_stage else bg;h=hs if is_stage else hg;mean=np.array(ck['normalizer']['mean'],dtype='float32');std=np.array(ck['normalizer']['std'],dtype='float32');x=torch.cat([h,torch.from_numpy(((b-mean)/std).astype('float32'))],1)
 else:
  ti=np.flatnonzero((cv>=0)&(cv!=fold)) if fold>=0 else np.flatnonzero(cv>=0);x,norm=engine.inputs(cfg,ti)
 if package.name=='DIAG_A3LINEAR':
  class Plain(nn.Module):
   def __init__(self):super().__init__();self.base=Architecture('A0')
   def forward(self,x):return self.base(x[:,:1536])
  model=Plain()
 else:model=engine.Model(len(cfg.get('scales',[])))
 model.load_state_dict(ck['model'],strict=True);model.eval();saved=np.load(run/f"epoch_{cfg.get('epochs',10):02}_predictions.npz")
 with torch.no_grad():z=model(x[saved['indices']]).numpy()
 err=float(np.max(np.abs(z-saved['logits'])));ok=bool(np.allclose(z,saved['logits'],rtol=1e-5,atol=2e-5));results.append({'run':run.name,'max_logit_error':err,'pass':ok})
 assert ok,(run,err)
dest=package/'PROVENANCE/checkpoint_forward_verification.json';dest.write_text(json.dumps({'checks':results,'pass':all(r['pass'] for r in results),'scope':'strict checkpoint load and independent local-package forward versus archived final predictions; no optimization'},indent=2));print(package.name,len(results),'forward checks passed')
