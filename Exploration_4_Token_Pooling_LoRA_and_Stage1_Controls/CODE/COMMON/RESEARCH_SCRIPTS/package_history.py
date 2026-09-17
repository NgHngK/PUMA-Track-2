"""Add standalone historical packages without modifying any historical source."""
import sys,shutil,json,csv,re,datetime
sys.dont_write_bytecode=True
from bootstrap import *
COMMON=R/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
def copytree(s,d):
 for p in s.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:cp(p,d/p.relative_to(s))
def write_new(p,text):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text(encoding='utf8')==text,str(p)
 else:p.write_text(text,encoding='utf8')
def local_modules(dst):
 copytree(OLD/'01_shared_core',dst/'CODE')
 copytree(OLD/'01_shared_core/puma_v17_evaluator',dst/'CODE/reference_evaluator')
 parity=(BASE/'work/v17_parity.py').read_text();parity=parity.replace("ROOT=Path(__file__).parent.parent/'outputs/STAGE2_RESEARCH_20260907'","ROOT=Path(__file__).resolve().parents[1]")
 parity=parity.replace("ROOT/'01_shared_core'","ROOT/'CODE'")
 parity=parity.replace("src=Path(r'D:\\Research\\PUMA\\Code\\Version 17\\PUMA_Nuclei_Pipeline\\src\\puma_nuclei')","src=ROOT/'CODE/reference_evaluator'")
 parity=parity.replace("(ROOT/'00_reference/v17_evaluation_contract/regression_test.json').write_text", "(ROOT/'PROVENANCE/parity_result.json').write_text")
 write_new(dst/'CODE/parity_test.py',parity)
 for name,body in {'preprocessing.py':'from dataset import centered_crop,image_tensor\n','losses.py':'from loss import LogitAdjustedCE\n','sampling.py':'from torch.utils.data import WeightedRandomSampler\n','evaluator_v17.py':'from metrics import puma_metrics,CANONICAL_TO_V17\n','calibration.py':'"""No learned calibration in these archived cached-head runs. ECE15/NLL are diagnostics only."""\n'}.items():
  if not (dst/'CODE'/name).exists():write_new(dst/'CODE'/name,body)
 write_new(dst/'requirements.txt','torch==2.10.0\nnumpy\nscipy\nscikit-learn\nPillow\npsutil\ntimm==1.0.20\nscikit-image\nmatplotlib\n')
def cached_package(n,arch,exps):
 dst=R/f'PROMPT_{n}/ARCHITECTURES'/arch
 for d in ['CONFIGS/original','CONFIGS/normalized','INPUT_MANIFESTS','RESULTS/RUNS','CACHE_REFERENCES','PROVENANCE']:(dst/d).mkdir(parents=True,exist_ok=True)
 local_modules(dst);idx=[]
 for ed,cfg in exps:
  copytree(ed,dst/'RESULTS/RUNS'/cfg['id']);cp(ed/'config.json',dst/'CONFIGS/original'/(cfg['id']+'.json'))
  for p in ed.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:idx.append({'original_path':str(p),'new_path':str((dst/'RESULTS/RUNS'/cfg['id']/p.relative_to(ed)).relative_to(R)),'sha256':sha(p),'experiment_id':cfg['id'],'seed':cfg.get('seed',17),'fold':cfg.get('fold','original split'),'result_type':p.suffix})
 csvout(dst/'ORIGINAL_PATH_MAP.csv',idx);csvout(dst/'RESULT_INDEX.csv',idx)
 cp(OLD/'01_sample_definition/sample_manifest.csv',dst/'INPUT_MANIFESTS/sample_manifest.csv');cp(OLD/'01_sample_definition/prompt3_cv.csv',dst/'INPUT_MANIFESTS/prompt3_cv.csv')
 dump(dst/'config.json',{'scientific_prompt':n,'architecture':arch,'default_experiment':exps[0][1]['id'],'inputs':'../../../EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS','external_RGB_and_weights':'Original user paths in manifests, verified by provenance. Not embedded in package.'})
 # Local trainer copy refactoring redirects output away from immutable shared input snapshots.
 source_name='exploration3_train.py' if n==3 else 'train.py';text=(OLD/'01_shared_core'/source_name).read_text()
 if n==3:
  text=text.replace("out=R/'experiments'/cfg['id']","out=Path(cfg['output_root'])/cfg['id']")
  begin=text.index("    registry=R/'experiments/registry.csv'");end=text.index("    print(cfg['id']",begin);text=text[:begin]+text[end:]
 else:text=text.replace("out=R/'experiments'/config['id']","out=Path(config['output_root'])/config['id']")
 write_new(dst/'CODE/standalone_train.py',text)
 runner='''import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import standalone_train as trainer
p=argparse.ArgumentParser(description="Reproduce this architecture with local code and immutable cached inputs")
p.add_argument('--config',required=True);p.add_argument('--inputs');p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());cfg['output_root']=str(Path(a.output).resolve())
trainer.R=Path(a.inputs).resolve() if a.inputs else Path(__file__).resolve().parents[4]/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
trainer.run(cfg)
'''
 write_new(dst/'CODE/run_train.py',runner)
 evaltext='''import argparse,json,sys,csv
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from train import summary_metrics
from dataset import read_manifest
p=argparse.ArgumentParser(description="Re-evaluate preserved logits with this package's exact local V17 evaluator")
p.add_argument('--predictions',required=True);p.add_argument('--manifest');p.add_argument('--output',required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[4]/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS';rows=read_manifest(a.manifest or root/'01_sample_definition/sample_manifest.csv')
d=np.load(a.predictions);mapping={r['uid']:r for r in rows}
rr=[mapping[str(u)] for u in d['uids']] if 'uids' in d else [rows[int(i)] for i in d['indices']]
m=summary_metrics(rr,d['labels'],torch.from_numpy(d['logits']));Path(a.output).write_text(json.dumps(m,indent=2));print(m['puma']['fixed10']['macro_f1'])
'''
 write_new(dst/'CODE/run_eval.py',evaltext)
 dump(dst/'CACHE_REFERENCES/inputs.json',{'shared_archival_root':str(COMMON.relative_to(R)),'cache_manifests':{str(p.relative_to(COMMON)):sha(p) for p in COMMON.rglob('*') if p.is_file() and p.name in ['contract.json','tierA_schema.json','tierB_schema.json','oracle_schema.json','sample_manifest.csv']},'exact_UIDs':'INPUT_MANIFESTS/sample_manifest.csv','row_selection':'original config/fold; Exploration3 CV table included'})
 dump(dst/'PROVENANCE/provenance.json',{'scientific_prompt':n,'architecture':arch,'copy_timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_original_path':str(OLD/'01_shared_core'),'source_hashes':{str(p.relative_to(OLD/'01_shared_core')):sha(p) for p in (OLD/'01_shared_core').rglob('*.py') if '__pycache__' not in p.parts},'refactoring':'standalone_train.py redirects output_root and omits shared registry/state mutation; numerical trainer/model/evaluator unchanged','experiment_ids':[c['id'] for _,c in exps],'new_path':str(dst),'module_scope':'All runtime code local to CODE. Shared immutable feature inputs permitted; no import from another architecture.'})
 dump(dst/'PROVENANCE/code_hashes.json',{str(p.relative_to(dst)):sha(p) for p in (dst/'CODE').rglob('*.py')})
 write_new(dst/'README.md',f'''# Exploration {n}: {arch}

This is an independent local code package reconstructed from the exact historical sources. Results are copies, not reruns. Original sources and missing-data limitations remain in PROVENANCE and ORIGINAL_PATH_MAP.csv.

Run from this directory:

```powershell
python CODE/run_train.py --config CONFIGS/original/{exps[0][1]['id']}.json --output NEW_RUNS
python CODE/parity_test.py
python CODE/run_eval.py --predictions RESULTS/RUNS/<experiment>/<prediction_file>.npz --output reevaluation.json
```

The evaluator reads stored logits and exact UID/coordinate manifests. Training reads one shared immutable input snapshot at EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS; override --inputs after relocating it. Raw RGB paths remain the user-provided dataset paths. Feature extraction requires the separately supplied UNI2 checkpoint. No network download is implicit. New training output is directed explicitly to --output. Original logs/checkpoints are not overwritten. See ARCHITECTURE.md for recovered equations and tensor flow, and RESULT_INDEX.csv for every source result.

Historical selected/best and final endpoints must not be interchanged. Exploration2 selected validation-best; Exploration3 fixed epoch10. BioMask/TierB Exploration3 comparisons remain upstream-contaminated diagnostics. Evaluator parity here establishes supplied-local-V17 behavior, not challenge-server certification. Historical Exploration1 scores used semantic metrics; this does not retroactively relabel them as V17 scores.
''')
 return dst
def main():
 for rel in ['01_sample_definition','02_cache']:copytree(OLD/rel,COMMON/rel)
 for name in ['RESEARCH_STATE.json','EXPLORATION3_PROTOCOL.md','CONTINUATION_PROTOCOL.md']:cp(OLD/'00_reference'/name,COMMON/'00_reference'/name)
 groups={}
 for ed in (OLD/'experiments').iterdir():
  if not (ed/'config.json').exists():continue
  cfg=json.loads((ed/'config.json').read_text());n=3 if 'architecture' in cfg else 2;arch=cfg.get('architecture',('BIOLOGY_ONLY' if cfg.get('biology_only') else 'FROZEN_UNI2')+'_'+cfg.get('biology','none'));groups.setdefault((n,arch),[]).append((ed,cfg))
 made=[]
 for (n,arch),exps in sorted(groups.items()):made.append(str(cached_package(n,arch,exps).relative_to(R)));print('packaged',n,arch,len(exps),flush=True)
 dump(W/'historical_packages.json',made)
if __name__=='__main__':main()
