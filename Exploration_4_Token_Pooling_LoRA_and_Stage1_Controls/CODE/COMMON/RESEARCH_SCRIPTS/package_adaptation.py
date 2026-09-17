from bootstrap import *
from package_history import copytree,write_new
P=R/'EXPLORATION_4'
def base(name,runs):
 d=P/'ARCHITECTURES'/name
 if (d/'PROVENANCE/package_complete.json').exists():return None
 copytree(P/'ARCHITECTURES/A3/CODE',d/'CODE');idx=[]
 for run in runs:
  copytree(run,d/'RESULTS/RUNS'/run.name)
  cp(run/'config.json',d/'CONFIGS/original'/(run.name+'.json'))
  for f in run.rglob('*'):
   if f.is_file():idx.append({'original_path':str(f),'new_path':str((d/'RESULTS/RUNS'/run.name/f.relative_to(run)).relative_to(R)),'sha256':sha(f),'experiment_id':run.name,'type':f.suffix})
 csvout(d/'RESULT_INDEX.csv',idx);csvout(d/'ORIGINAL_PATH_MAP.csv',idx);cp(P/'ARCHITECTURES/A3/requirements.txt',d/'requirements.txt');cp(OLD/'01_sample_definition/sample_manifest.csv',d/'INPUT_MANIFESTS/sample_manifest.csv');return d
d=base('E1_LORA',[P/f'LORA/E1_s{s}' for s in [17,29,43]])
if d:
 src=(W/'lora.py').read_text().replace("sys.path[:0]=[str(BASE/'work/deps'),str(OLD/'01_shared_core')]","sys.path.insert(0,str(Path(__file__).parent))").replace("from bootstrap import *","from bootstrap import *\nimport os\nWEIGHTS=os.environ.get('PUMA_UNI2_WEIGHTS','D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin')\nOUTPUT=Path(os.environ.get('PUMA_RUN_OUTPUT','NEW_RUNS'))")
 src=src.replace("load_uni2('D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin')","load_uni2(WEIGHTS)").replace("out=P/'LORA'/('profile' if profile else f'E1_s{seed}')","out=OUTPUT/('profile' if profile else f'E1_s{seed}')").replace('out.mkdir(exist_ok=True)','out.mkdir(parents=True,exist_ok=True)')
 # Existing A3 skeleton trainer is preserved as a generated draft before specialization.
 cp(d/'CODE/run_train.py',d/'PROVENANCE/run_train_initial_draft.txt');(d/'CODE/run_train.py').write_text('import argparse,os\np=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--weights",required=True);a=p.parse_args();os.environ["PUMA_RUN_OUTPUT"]=a.output;os.environ["PUMA_UNI2_WEIGHTS"]=a.weights\nfrom restricted_lora import main\nmain()\n')
 write_new(d/'CODE/restricted_lora.py',src);dump(d/'config.json',json.loads((P/'LORA/E1_s17/config.json').read_text()));dump(d/'CACHE_REFERENCES/inputs.json',{'prefix':'EXPLORATION_4/TOKEN_AUDIT/CACHE/prefix20.npy','geometry':'EXPLORATION_4/TARGET_POOLING/REPRESENTATIONS/pooling_geometry.npz','frozen_weights_sha256':'32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'})
 write_new(d/'README.md','''# Restricted LoRA experiment

Run `python CODE/run_train.py --output NEW_RUNS --weights PATH_TO_UNI2` with timm1.0.20. This reproduces the three fixed five-epoch seeds with local code, cached blocks0–19 and Q/V LoRA on blocks20–23. The checkpoint contains adapters and head; full frozen weights remain external by SHA256. Re-evaluate stored logits with `CODE/run_eval.py --predictions FILE --output FILE`. The run did not pass promotion. Full training details and individual epochs are preserved in RESULTS.
''');dump(d/'PROVENANCE/package_complete.json',{'source':sha(W/'lora.py'),'local':sha(d/'CODE/restricted_lora.py'),'changes':'output and frozen-weight path parameters only'})
d=base('F_CLEAN_MASK_NETWORK',[P/'BIOMASK_GUIDED/CLEAN_GT_ONLY'])
if d:
 source=OLD/'00_reference/biomask_audit/source'
 for name in ['biomask.py','crops.py']:cp(source/name,d/'CODE/mask_source'/name)
 # Keep GT-only inputs once within Exploration4; the source audit files are existing immutable inputs.
 for name in ['roi_manifest.npy','folds.npy','rgb_crops_qc.npy','gt_masks_diagnostic.npy']:cp(OLD/'00_reference/biomask_audit'/name,P/'SHARED_INPUT_SNAPSHOTS/CLEAN_MASK_INPUTS'/name)
 src=(W/'clean_biomask.py').read_text().replace("A=OLD/'00_reference/biomask_audit';D=P/'BIOMASK_GUIDED';O=D/'CLEAN_GT_ONLY'","import os\nA=P/'SHARED_INPUT_SNAPSHOTS/CLEAN_MASK_INPUTS';D=P/'BIOMASK_GUIDED';O=Path(os.environ.get('PUMA_RUN_OUTPUT','NEW_CLEAN_MASK'))").replace("A/'source/crops.py'","Path(__file__).parent/'mask_source/crops.py'").replace("A/'source/biomask.py'","Path(__file__).parent/'mask_source/biomask.py'").replace('O.mkdir(exist_ok=True)','O.mkdir(parents=True,exist_ok=True)')
 write_new(d/'CODE/clean_mask.py',src);cp(d/'CODE/run_train.py',d/'PROVENANCE/run_train_initial_draft.txt');(d/'CODE/run_train.py').write_text('import argparse,os\np=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();os.environ["PUMA_RUN_OUTPUT"]=a.output\nfrom clean_mask import main\nmain()\n')
 cp(d/'CODE/run_eval.py',d/'PROVENANCE/run_eval_initial_draft.txt');(d/'CODE/run_eval.py').write_text('import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser(description="Read archived fixed-endpoint held-fold mask metrics");p.add_argument("--complete",required=True);a=p.parse_args();print(json.dumps(json.loads(Path(a.complete).read_text()),indent=2))\n')
 dump(d/'config.json',json.loads((P/'BIOMASK_GUIDED/CLEAN_GT_ONLY/config.json').read_text()));write_new(d/'README.md','''# Clean GT-prompt mask control

Run `python CODE/run_train.py --output NEW_CLEAN_MASK`. This reproduces the fixed ten-epoch mask-only training on294 non-fold0 nuclei. The held156 fold0 nuclei never enter this mask network training, and no Stage1-derived proposal or class label is a network input. Local mask source modules and immutable GT-only input snapshots are included. The archive records held Dice0.783957; this is a new small-data experiment, not the historical BioMask recipe. Downstream mask pooling failed promotion. `run_eval.py --complete RESULTS/RUNS/CLEAN_GT_ONLY/complete.json` reads archived endpoint metrics; it is not an independent checkpoint-forward test.
''');dump(d/'PROVENANCE/package_complete.json',{'source':sha(W/'clean_biomask.py'),'local':sha(d/'CODE/clean_mask.py'),'changes':'output and immutable input/source path redirection only'});dump(d/'CACHE_REFERENCES/inputs.json',{'inputs':'EXPLORATION_4/SHARED_INPUT_SNAPSHOTS/CLEAN_MASK_INPUTS'})
print('Adaptation packages prepared')
