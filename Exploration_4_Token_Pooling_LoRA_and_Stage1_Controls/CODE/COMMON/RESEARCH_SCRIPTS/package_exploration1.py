import sys,shutil,json,csv
from bootstrap import *
from package_history import write_new,copytree,local_modules
D=R/'EXPLORATION_1';COMMON=D/'SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
if __name__=='__main__':
 for name in ['cnn_pixels.npz','cnn_sample.csv','cnn_sample.json','manifest.csv','manifest.json','uni2_features.npy','uni2_features.json','uni2_complete.npy','uni2_sample.csv','uni2_sample.json']:
  cp(BASE/'work'/name,COMMON/name)
 if (BASE/'work/lora_results/prefix.pt').exists():cp(BASE/'work/lora_results/prefix.pt',COMMON/'lora_results/prefix.pt')
 for arch,source,results in [('CNN','cpu_controls.py','cnn_results'),('UNI2_PROBE','probe_uni2.py','uni2_results'),('LORA_CORRECTED','lora_dynamics.py','lora_results_v2')]:
  dst=D/'ARCHITECTURES'/arch
  for d in ['CODE','CONFIGS/original','RESULTS/ORIGINAL_RUNS','PROVENANCE','CACHE_REFERENCES','INPUT_MANIFESTS']:(dst/d).mkdir(parents=True,exist_ok=True)
  local_modules(dst)
  # Preserve original Exploration1 numerical implementations under local independent module names.
  for name in ['dataset.py','loss.py','model.py','train_eval.py']:
   cp(BASE/'outputs'/name,dst/'CODE'/('prompt1_'+name))
  src=(BASE/'work'/source).read_text();src=src.replace("from dataset import", "from exploration1_dataset import").replace("from train_eval import","from exploration1_train_eval import").replace("from model import","from prompt1_model import").replace("from loss import","from prompt1_loss import")
  src=src.replace("sys.path.insert(0,str(ROOT.parent/'outputs'))",'')
  src=src.replace("sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'outputs')]",'')
  src=src.replace("out=ROOT/'"+results+"'","out=OUTPUT_DIR")
  # Controlled recipe selection, without changing the original mathematics or RNG schedule.
  if arch=='CNN':src=src.replace("for recipe,tau,balanced in [('ce',0,False),('la',1,False),('balanced_ce',0,True)]:","for recipe,tau,balanced in [v for v in [('ce',0,False),('la',1,False),('balanced_ce',0,True)] if v[0]==RECIPE]:")
  if arch=='UNI2_PROBE':src=src.replace("for recipe,tau,balanced in [('ce',0,False),('la',1,False),('balanced_ce',0,True)]:","for recipe,tau,balanced in [v for v in [('ce',0,False),('la',1,False),('balanced_ce',0,True)] if v[0]==RECIPE]:")
  if arch=='LORA_CORRECTED':src=src.replace("for mode in ['frozen','lora']:","for mode in MODES:")
  write_new(dst/'CODE/original_training_local.py',src);copytree(BASE/'work'/results,dst/'RESULTS/ORIGINAL_RUNS')
  cfg={'architecture':arch,'seed':17,'recipe':'ce','modes':['frozen','lora'],'epochs':5 if arch=='LORA_CORRECTED' else 10,'exact_input_root':'EXPLORATION_1/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'};dump(dst/'config.json',cfg)
  runner='''import sys,json,argparse
from pathlib import Path
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--inputs');p.add_argument('--output',required=True);a=p.parse_args();c=json.loads(Path(a.config).read_text())
import original_training_local as t
t.ROOT=Path(a.inputs) if a.inputs else Path(__file__).resolve().parents[4]/'EXPLORATION_1/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
t.OUTPUT_DIR=Path(a.output);t.OUTPUT_DIR.mkdir(parents=True,exist_ok=True);t.RECIPE=c.get('recipe','ce');t.MODES=c.get('modes',['frozen','lora']);sys.argv=[sys.argv[0]]+(['--seed',str(c.get('seed',17))] if c['architecture']=='UNI2_PROBE' else [])
t.main()
'''
  write_new(dst/'CODE/run_train.py',runner)
  write_new(dst/'CODE/run_eval.py','''import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from exploration1_train_eval import metrics
p=argparse.ArgumentParser(description="Historical semantic metrics only; V17 was introduced in Exploration2")
p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=np.load(a.predictions);m=metrics(d['labels'],torch.from_numpy(d['logits']));Path(a.output).write_text(json.dumps(m,indent=2));print(m['macro_f1'])
''')
  # Exploration1 module imports must remain local even after moving the package.
  p=dst/'CODE/exploration1_train_eval.py';original=p.read_text();updated=original.replace('from dataset import','from exploration1_dataset import').replace('from model import','from prompt1_model import').replace('from loss import','from prompt1_loss import')
  if original!=updated:
   cp(p,dst/'PROVENANCE/prompt1_train_eval_before_local_imports.py.txt');p.write_text(updated)
  idx=[{'original_path':str(p),'new_path':str((dst/'RESULTS/ORIGINAL_RUNS'/p.relative_to(BASE/'work'/results)).relative_to(R)),'sha256':sha(p),'family':arch,'attribution':'historical execution directory; recipe/seed in exact log and source'} for p in (BASE/'work'/results).rglob('*') if p.is_file()]
  csvout(dst/'RESULT_INDEX.csv',idx);csvout(dst/'ORIGINAL_PATH_MAP.csv',idx);dump(dst/'PROVENANCE/provenance.json',{'source_original_path':str(BASE/'work'/source),'source_original_hash':sha(BASE/'work'/source),'local_code_hash':sha(dst/'CODE/original_training_local.py'),'change':'local imports/output destination/recipe selection only','scientific_prompt':1,'metrics':'original semantic, not retroactively V17','historical_epoch_fields_missing':'NOT RECORDED IN ORIGINAL RUN'})
  dump(dst/'PROVENANCE/code_hashes.json',{str(p.relative_to(dst)):sha(p) for p in (dst/'CODE').rglob('*.py')})
  write_new(dst/'README.md',f'''# Exploration1 {arch}

Independent local implementation recovered from {source}. The original source is preserved by hash and in the source archives. Run `python CODE/run_train.py --config config.json --output NEW_RUNS`; use `--inputs` to relocate the immutable Exploration1input snapshot. {('Set recipe ce/la/balanced_ce and seed17/29/43 in a new config; historical semantic validation-best selection is preserved.' if arch!='LORA_CORRECTED' else 'The corrected frozen/LoRA comparison has only50nuclei and5epochs; it cannot establish global LoRA failure. Original checkpoint paths still require the supplied UNI2-h weights. Install the locally verified timm1.0.20 dependency.')}

`CODE/run_eval.py --predictions <saved.npz> --output metrics.json` re-evaluates available logits with original semantic metrics. If original logits were not saved, that evaluator cannot invent them; use a new training run to produce new outputs. `CODE/parity_test.py` verifies the included V17 evaluator for future use, without relabeling old semantic numbers as V17. Historical metadata, logs and checkpoints are in RESULTS/ORIGINAL_RUNS; all source files and hashes are indexed. No original file is modified.
''')
  print('packaged Exploration1',arch,flush=True)
