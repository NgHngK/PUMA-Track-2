from pathlib import Path
import sys,json,csv,hashlib
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';sys.path.insert(0,str(C/'src'))
from puma_exploration6.manifest import read_manifest
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
reference=None;rows=[];run_count=0
for root in sorted((O/'EXPERIMENTS').iterdir()):
 sp=root/'summary.json'
 if not sp.exists():continue
 cfg=json.loads((root/'config.json').read_text());manifest=read_manifest(cfg['manifest'],require_files=False);dev=[r for r in manifest if r['split']=='dev'];ident=[(r['uid'],r['roi'],r['label'],r['x'],r['y']) for r in dev]
 if reference is None:reference=ident
 assert ident==reference,f'DEV identity/order differs: {root}'
 names=sorted({r['roi'] for r in dev});d=json.loads(sp.read_text());h=d['decoupled']['epochs'][-1] if d.get('decoupled') else d['selected'];metrics=h['dev']['puma']['roi_metrics'];assert len(names)==len(metrics)==56;run_count+=1
 for name,m in zip(names,metrics):
  val=sum(m.get('nuclei_'+c,{}).get('f1_score',0) for c in CL)/10
  rows.append(dict(run_id=root.name,seed=cfg['seed'],roi=name,roi_fixed10_f1=val,semantic_annotation_rows=sum(r['roi']==name for r in dev),summary_sha256=hashlib.sha256(sp.read_bytes()).hexdigest()))
with (O/'TABLES/ENDPOINT_PAIRED_ROI_SCORES.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'PROVENANCE/PAIRED_ROI_ORDER_AUDIT.json').write_text(json.dumps(dict(runs=run_count,dev_annotation_rows=len(reference),dev_rois=56,ordered_uid_coordinate_label_identity=True,metric_order='metrics.puma_subset_metrics sorts unique row.roi strings; evaluator receives indices0..55 in this order',scope='Completed A/C DEV only; holdouts unopened'),indent=2),encoding='utf-8')
print(json.dumps(dict(runs=run_count,roi_rows=len(rows),identity='PASS')))
