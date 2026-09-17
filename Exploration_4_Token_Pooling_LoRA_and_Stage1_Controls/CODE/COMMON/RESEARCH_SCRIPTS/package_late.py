from bootstrap import *
from package_history import copytree,write_new
P=R/'EXPLORATION_4'
# Adapt the existing package generator in memory, preserving the original file.
src=(W/'package_exploration4.py').read_text()
anchor="  train+='engine.run(cfg)\\n';write_new(d/'CODE/run_train.py',train)"
addition='''  if arch.startswith('F_'):
   train+="import numpy as np,torch\\ncontract=json.loads((engine.P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text());ix=np.array(contract['original_indices']);rr,cc,yy=engine.load_data();rows=[rr[i] for i in ix];cv=cc[ix].copy();cv[cv>=0]=0;y=yy[ix];bio=np.load(engine.OLD/'02_cache/biology/tierA.npy')[ix];ti=np.flatnonzero(cv>=0);mean=bio[ti].mean(0);std=np.maximum(bio[ti].std(0),1e-6);h=torch.from_numpy(np.load(engine.P/'BIOMASK_GUIDED/REPRESENTATIONS'/(cfg['family']+'.npy')));x=torch.cat((h,torch.from_numpy(((bio-mean)/std).astype('float32'))),1);engine.load_data=lambda:(rows,cv,y);engine.inputs=lambda cfg,ti:(x,{'mean':mean.tolist(),'std':std.tolist()})\\n"
'''
assert anchor in src;src=src.replace(anchor,addition+anchor)
anchor2='  ev+="d=np.load(a.predictions);ii=d[\'indices\'];'
addition2='''  if arch.startswith('F_'):ev+="from engine import P\\nix=json.loads((P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text())['original_indices'];rows=[rows[i] for i in ix]\\n"
'''
assert anchor2 in src;src=src.replace(anchor2,addition2+anchor2)
ns={'__name__':'late_builder'};exec(compile(src,'package_prompt4_late_generated','exec'),ns);ns['build']()
# Final recommended package is the unchanged verified Exploration3 A5 raw-image implementation.
dst=P/'ARCHITECTURES/FINAL_RECOMMENDED_A5';dst.mkdir(exist_ok=True);copytree(OLD/'93_final_architecture',dst/'CODE')
for name in ['config.json','requirements.txt']:cp(OLD/'93_final_architecture'/name,dst/name)
copytree(R/'EXPLORATION_3/ARCHITECTURES/A5/RESULTS',dst/'RESULTS')
for name in ['RESULT_INDEX.csv','ORIGINAL_PATH_MAP.csv']:cp(R/'EXPLORATION_3/ARCHITECTURES/A5'/name,dst/name)
write_new(dst/'CODE/run_train.py','from train_eval import main\nif __name__=="__main__":main()\n')
write_new(dst/'CODE/run_eval.py','"""Use --eval-checkpoint with all original train_eval evaluation arguments."""\nfrom train_eval import main\nif __name__=="__main__":main()\n')
copytree(OLD/'01_shared_core/puma_v17_evaluator',dst/'CODE/reference_evaluator')
parity=(R/'EXPLORATION_3/ARCHITECTURES/A5/CODE/parity_test.py').read_text();write_new(dst/'CODE/parity_test.py',parity)
# Parity helper imports historical metric adapter, whose source is copied locally.
for name in ['metrics.py','dataset.py']:
 if name=='metrics.py':cp(OLD/'01_shared_core'/name,dst/'CODE'/name)
copytree(OLD/'01_sample_definition',dst/'INPUT_MANIFESTS')
dump(dst/'PROVENANCE/selection.json',json.loads((P/'TABLES/final_decision.json').read_text()))
dump(dst/'CACHE_REFERENCES/inputs.json',{'CLS':'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache/uni2_features/gt_fov96','TierA':'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache/biology/tierA.npy','new_full_data':'Generate with CODE/features.py from complete Stage1 proposal manifest. External image/weight paths remain user-supplied.'})
write_new(dst/'README.md','''# Single next full-scale architecture: retained Exploration3 A5

Exploration4 A3 failed the predeclared class-recall confirmation constraint. No runner-up was selected. This package retains the exact frozen CLS1536 + standardized TierA16 rank-eight interaction model, 27,866 trainable parameters. The full-image code is copied from the verified Exploration3 deployment blueprint; the archived experiments are GT-centered development controls, not a full-scale trained system.

Run `python CODE/features.py --help`, then `python CODE/run_train.py --help`. Full-proposal evaluation uses `python CODE/run_eval.py --eval-checkpoint CHECKPOINT` with the manifest, cache, annotation root, ROI splits and documented frozen Stage1 provenance. It includes all proposals and zero-proposal ROIs; unknown semantic labels are excluded only from CE. New run directories are mandatory. Temperature remains1 unless fit on a separate calibration split.

The recommended full-scale recipe is AdamW headLR.001, weight decay.01, batch64, inverse training-class count sampling, ordinary CE, at most20epochs and patience5 on the prespecified validation metric. This is a prospective recipe, not an executed full-scale claim. Preserve patient-level exclusion if patient identifiers become available. Local code never imports another architecture's modules.
''')
print('Late and final packages prepared')
