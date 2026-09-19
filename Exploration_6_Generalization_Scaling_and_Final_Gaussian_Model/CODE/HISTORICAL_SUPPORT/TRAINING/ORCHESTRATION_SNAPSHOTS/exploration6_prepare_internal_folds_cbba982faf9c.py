from pathlib import Path
import sys,json,collections,csv
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';sys.path.insert(0,str(C/'src'))
from puma_exploration6.cv import create_internal_group_cv_manifests
from puma_exploration6.manifest import read_manifest
dest=O/'HPO/FROZEN_INTERNAL_FOLDS_D900'
if (dest/'CV_FOLDS.json').exists():folds=json.loads((dest/'CV_FOLDS.json').read_text())['folds']
else:folds=create_internal_group_cv_manifests(R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION/SELECTION/D900.csv',dest,n_splits=3,seed=1706,require_all_classes=True)
rows=[];held=[]
for fold in folds:
 data=read_manifest(fold['manifest'],require_files=False)
 held.extend(r['uid'] for r in data if r['split']=='dev')
 for split in ['train','dev','predict']:
  for k in range(10):
   rr=[r for r in data if r['split']==split and r['label']==k];rows.append(dict(fold=fold['fold'],split=split,label=k,annotation_rows=len(rr),positive_ROIs=len({r['roi'] for r in rr})))
assert len(held)==len(set(held))==900
with (dest/'CLASS_GROUP_COVERAGE.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(dest/'READINESS.json').write_text(json.dumps(dict(status='FOLDS_FROZEN_ONLY_NO_HPO_RUNS',seed=1706,all_train_UIDs_held_exactly_once=True,all_classes_in_each_train_and_dev=True,external_DEV_and_confirmation_excluded=True,settings='Training recipe and bounded optimizer/regularization configs await B/C resolution; do not tune on external DEV',future_reuse='Use these exact fold manifests when preparing D trials; preserve row-identity-compatible caches'),indent=2),encoding='utf-8')
print(json.dumps([dict(fold=x['fold'],train=x['train_rows'],dev=x['dev_rows'],train_ROIs=x['train_groups'],dev_ROIs=x['dev_groups']) for x in folds]))
