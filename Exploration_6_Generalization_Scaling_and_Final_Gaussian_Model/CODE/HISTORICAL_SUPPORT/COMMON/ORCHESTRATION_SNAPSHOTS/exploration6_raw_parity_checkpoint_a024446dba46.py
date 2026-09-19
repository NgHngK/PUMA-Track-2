from pathlib import Path
import json,hashlib,datetime
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';E=O/'EXPERIMENTS'
ap=E/'p6_A_D900_s17/history.jsonl';bp=E/'p6_B_AUG0_s17/history.jsonl';a=[json.loads(x) for x in ap.read_text().splitlines()];b=[json.loads(x) for x in bp.read_text().splitlines()];rows=[]
for x,y in zip(a,b):
 rows.append(dict(epoch=y['epoch'],roi_f1_delta=y['dev']['puma']['fixed10']['macro_f1']-x['dev']['puma']['fixed10']['macro_f1'],semantic_f1_delta=y['dev']['macro_f1']-x['dev']['macro_f1'],dev_nll_delta=y['dev']['nll']-x['dev']['nll'],same_class_exposure=x['exposure']==y['exposure'],same_unique_counts=x['unique_examples_by_class']==y['unique_examples_by_class']))
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='INTERIM PARITY; not augmentation efficacy',completed_raw_epochs=len(b),rows=rows,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [ap,bp]],note='Active history is a snapshot and will grow; final logit parity awaits completed epoch10. No control runs skipped and no outcome-based endpoint selection.')
(O/'RESEARCH_STATE/RAW_CACHED_PARITY_PROGRESS.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps(dict(epochs=len(b),max_abs_nll_delta=max(abs(r['dev_nll_delta']) for r in rows))))
