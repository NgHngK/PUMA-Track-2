from pathlib import Path
import json,csv,hashlib
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
rows=[]
for root in sorted((O/'EXPERIMENTS').iterdir()):
 cp=root/'config.json';sp=root/'summary.json'
 if not cp.exists():continue
 c=json.loads(cp.read_text());s=json.loads(sp.read_text()) if sp.exists() else {};dec=s.get('decoupled');h=dec['epochs'][-1] if dec else s.get('selected',{});group=c['experiment_id'].split('_')[1]
 rows.append(dict(experiment_group=group,run_id=c['experiment_id'],status='COMPLETE' if s else 'IN_PROGRESS',seed=c['seed'],manifest=c['manifest'],config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),joint_epochs=s.get('epochs_run','IN_PROGRESS'),decoupled_epochs=len(dec['epochs']) if dec else 0,selected_phase='decoupled' if dec else 'joint',selected_epoch=len(dec['epochs']) if dec else s.get('selected_epoch','NOT YET SELECTED'),selected_roi_f1=s.get('final_model_score','NOT YET AVAILABLE'),dev_semantic_f1=h.get('dev',{}).get('macro_f1','NOT YET AVAILABLE'),checkpoint=str(root/s['final_model_checkpoint']) if s else 'NOT YET AVAILABLE',summary=str(sp) if s else 'NOT YET AVAILABLE',summary_sha256=hashlib.sha256(sp.read_bytes()).hexdigest() if s else 'NOT YET AVAILABLE',evidence_status='CONDITIONAL ON B AUGMENTATION' if group=='C' else 'PREDECLARED SCALING CONTROL' if group=='A' else 'RAW AUGMENTATION STUDY'))
with (O/'TABLES/EXPERIMENT_REGISTRY.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(runs=len(rows),complete=sum(r['status']=='COMPLETE' for r in rows))))
