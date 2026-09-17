import sys,json,shutil,collections
from bootstrap import *
from package_history import local_modules,copytree,write_new
P=R/'EXPLORATION_4'
def build():
 groups=collections.defaultdict(list)
 for ed in (P/'RUNS').iterdir():
  if not (ed/'summary.json').exists():continue
  cfg=json.loads((ed/'config.json').read_text());family=cfg['family'];arch=family.replace('_broken','')
  if family.startswith('C_'):arch='C_ALIGNMENT'
  groups[arch].append((ed,cfg))
 for arch,experiments in groups.items():
  d=P/'ARCHITECTURES'/arch
  if (d/'PROVENANCE/package_complete.json').exists():continue
  for folder in ['CODE','CONFIGS/original','RESULTS/RUNS','PROVENANCE','INPUT_MANIFESTS','CACHE_REFERENCES']:(d/folder).mkdir(parents=True,exist_ok=True)
  local_modules(d);idx=[]
  for ed,cfg in experiments:
   copytree(ed,d/'RESULTS/RUNS'/cfg['id']);cp(ed/'config.json',d/'CONFIGS/original'/(cfg['id']+'.json'))
   for f in ed.rglob('*'):
    if f.is_file():idx.append({'original_path':str(f),'archival_path':str((d/'RESULTS/RUNS'/cfg['id']/f.relative_to(ed)).relative_to(R)),'sha256':sha(f),'experiment_id':cfg['id'],'seed':cfg['seed'],'fold':cfg.get('fold'), 'type':f.suffix})
  csvout(d/'RESULT_INDEX.csv',idx);csvout(d/'ORIGINAL_PATH_MAP.csv',idx);cp(OLD/'01_sample_definition/sample_manifest.csv',d/'INPUT_MANIFESTS/sample_manifest.csv');cp(OLD/'01_sample_definition/prompt3_cv.csv',d/'INPUT_MANIFESTS/prompt3_cv.csv')
  # Bootstrap resolves only this archive and local code. Shared input snapshots are immutable.
  bootstrap='''from pathlib import Path
import json,hashlib,shutil,csv
W=Path(__file__).resolve().parent
R=W.parents[3]
BASE=R
OLD=R/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf8')
def cp(s,d):
 d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists():assert sha(s)==sha(d)
 else:shutil.copy2(s,d)
def csvout(p,rs):
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
'''
  write_new(d/'CODE/bootstrap.py',bootstrap)
  source=(W/'engine.py').read_text();source=source.replace("sys.path.insert(0,str(OLD/'01_shared_core'))","sys.path.insert(0,str(Path(__file__).parent))").replace("out=P/'RUNS'/cfg['id']","out=Path(cfg['output_root'])/cfg['id']").replace("OLD/'01_shared_core/prompt3_models.py'","Path(__file__).parent/'prompt3_models.py'").replace("OLD/'01_shared_core/puma_v17_evaluator/evaluation/public_matcher.py'","Path(__file__).parent/'puma_v17_evaluator/evaluation/public_matcher.py'")
  # New runs do not depend on old stored predictions for reference checks.
  source=source.replace("if cfg['id'].startswith('A0_') and fold>=0:","if cfg.get('verify_historical_reference',False) and fold>=0:")
  write_new(d/'CODE/engine.py',source)
  if arch=='C_ALIGNMENT':
   cp(W/'alignment.py',d/'CODE/alignment_original.py.txt')
  train='''import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import engine
p=argparse.ArgumentParser(description="Run this frozen representation with local code")
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());cfg['output_root']=str(Path(a.output).resolve())
'''
  if arch=='DIAG_A3LINEAR':train+='''from torch import nn
from prompt3_models import Architecture
class Plain(nn.Module):
 def __init__(self,extra=0):super().__init__();self.base=Architecture('A0');self.corrections=[]
 def forward(self,x):return self.base(x[:,:1536])
engine.Model=Plain
'''
  if arch=='C_ALIGNMENT':
   # Self-contained preparation code copied from executed alignment implementation.
   al=(W/'alignment.py').read_text();al=al[:al.index("if __name__=='__main__':")];al=al.replace("sha(W/'stage1_final_fold0.pt')","sha(P/'STAGE1_PROPOSALS/stage1_final_fold0.pt')").replace("dump(P/'STAGE1_PROPOSALS/clean_cohort.json',", "dump(Path(__file__).resolve().parents[1]/'PROVENANCE/new_clean_cohort.json',")
   write_new(d/'CODE/alignment_data.py',al)
   train+='''from alignment_data import prepare
import numpy as np,torch
gt,st,cv,y,hg,hs,bg,bs=prepare();ti=np.flatnonzero(cv>=0);is_stage=cfg['family']=='C_STAGE1TRAIN';rows=st if is_stage else gt;b=bs if is_stage else bg;h=hs if is_stage else hg;mean=b[ti].mean(0);std=np.maximum(b[ti].std(0),1e-6);x=torch.cat((h,torch.from_numpy(((b-mean)/std).astype('float32'))),1)
engine.load_data=lambda:(rows,cv,y)
engine.inputs=lambda cfg,ti:(x,{'mean':mean.tolist(),'std':std.tolist()})
'''
  train+='engine.run(cfg)\n';write_new(d/'CODE/run_train.py',train)
  ev='''import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from engine import load_data,extra_metrics
p=argparse.ArgumentParser(description="Re-evaluate saved logits under the exact V17 contract")
p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--stage1-centers',action='store_true');a=p.parse_args();rows,cv,y=load_data()
'''
  if arch=='C_ALIGNMENT':ev+='from alignment_data import prepare\ngt,st,cv,y,*_=prepare();rows=st if a.stage1_centers else gt\n'
  ev+="d=np.load(a.predictions);ii=d['indices'];m=extra_metrics([rows[int(i)] for i in ii],d['labels'],torch.from_numpy(d['logits']));Path(a.output).write_text(json.dumps(m,indent=2));print(m['puma']['fixed10']['macro_f1'])\n"
  write_new(d/'CODE/run_eval.py',ev);dump(d/'config.json',experiments[0][1]);dump(d/'CACHE_REFERENCES/inputs.json',{'CLS_and_TierA':'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache','target_representation':'EXPLORATION_4/TARGET_POOLING/REPRESENTATIONS','patch_cache':'EXPLORATION_4/TOKEN_AUDIT/CACHE','proposals':'EXPLORATION_4/STAGE1_PROPOSALS/CACHE','UID_contract':'INPUT_MANIFESTS/sample_manifest.csv; cleanC cohort mapping explicitlystored','source_hashes':{'manifest':sha(OLD/'01_sample_definition/sample_manifest.csv'),'token_contract':sha(P/'TOKEN_AUDIT/CACHE/contract.json')}})
  write_new(d/'README.md',f'''# Exploration 4 — {arch}

Independent local cached-representation package. All runtime modules are in CODE; no module imports another architecture's code. Large immutable inputs are referenced once at Exploration level and identified by their UID/hash contracts.

```powershell
python CODE/run_train.py --config CONFIGS/original/{experiments[0][1]['id']}.json --output NEW_RUNS
python CODE/run_eval.py --predictions RESULTS/RUNS/<experiment>/epoch_10_predictions.npz --output evaluation.json
python CODE/parity_test.py
```

For C_ALIGNMENT, `run_eval.py --stage1-centers` evaluates Stage1-centered predictions; the default is GT-centered. Training reproduces the selected coordinate arm, while original paired evaluations for both coordinate inputs are preserved in RESULTS. New output directories are mandatory. Cached runs need no encoder re-extraction. The original RGB dataset and checkpoint remain external for extracting new populations.

The exact executed sources are preserved in PROVENANCE. Local copies only redirect output paths and input-root discovery; the numerical head, sampler, loss and evaluator are unchanged. ARCHITECTURE.md and FLOWCHART.md explain the tested mechanism. RESULT_INDEX.csv maps every copied result to its source.
''')
  dump(d/'PROVENANCE/code_hashes.json',{str(f.relative_to(d)):sha(f) for f in (d/'CODE').rglob('*.py')});dump(d/'PROVENANCE/package_complete.json',{'arch':arch,'experiments':[c['id'] for _,c in experiments],'numerical_engine_source':sha(W/'engine.py'),'standalone_engine':sha(d/'CODE/engine.py'),'output_refactoring':True});print('P4 packaged',arch,len(experiments),flush=True)
if __name__=='__main__':build()
