from pathlib import Path
import json, datetime, hashlib, shutil
R=Path(r'C:\Users\Hngk\Documents\Codex')
W=R/'2026-09-07/use-the-autoresearch-skill-x20-https/work/exploration6'
O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=O/'RESEARCH_STATE/WORKING_STATE.json';d=json.loads(p.read_text())
runs=[]
for root in sorted((O/'EXPERIMENTS').iterdir()):
 if not (root/'config.json').exists():continue
 hp=root/'history.jsonl';sp=root/'summary.json'
 h=[json.loads(x) for x in hp.read_text().splitlines() if x.strip()] if hp.exists() else []
 s=json.loads(sp.read_text()) if sp.exists() else None
 runs.append(dict(run_id=root.name,status='COMPLETE' if s else 'IN_PROGRESS',joint_epochs=len(h),decoupled_epochs=len((s.get('decoupled') or {}).get('epochs',[])) if s else 0,summary=str(sp) if s else None))
d.update(status='STUDY_RUNNING',updated_utc=stamp,new_research_training_runs=len(runs),completed_runs=sum(x['status']=='COMPLETE' for x in runs),run_registry=runs,holdouts_opened=False,current_stage='B raw augmentation running; bounded C cached controls conditional on B',scaling_result='D900 minus D300 ROI F1 +0.02027778, CI95 [0.00756803,0.03316475], 3/3 positive; neutrophil recall -0.24074074 fails promotion guard',pending_work=['Complete B augmentation and conditional C interpretation','D HPO, E Tier-A, F representation and conditional G adaptation','H seed replication; I frozen holdouts; J Stage1 if provenance permits','Complete epoch ledgers, standalone packages, final report and byte-preserving master append'])
payload=json.dumps(d,indent=2)
p.write_text(payload,encoding='utf-8');(W/'research-state.json').write_text(payload,encoding='utf-8')
findings=f'''# Exploration6 current findings

Updated {stamp}. Study remains in progress; no final recipe has been selected and neither holdout has been opened.

Canonical artifacts: {O}

Stage A: nine completed controls, three seeds at each nested training size. Mean DEV semantic F1 D300/D600/D900 = 0.390178/0.391375/0.427800; ROI fixed10 F1 = 0.129493/0.133084/0.149770. D900-D300 paired ROI delta +0.020278, 95% ROI bootstrap interval [0.007568,0.033165], three positive seeds. Neutrophil recall delta -0.240741 fails the predeclared class-harm guard. This is mixed evidence, not automatic promotion. Exact values: TABLES/DATA_SCALING_SUMMARY.json.

Stage B raw AUG0 seed17 is active. Stage C cached no-augmentation sampling/cRT controls run under frozen SCHEDULING_AMENDMENT_01 and remain conditional until B resolves. Never restart existing partial or completed run directories. Read run_registry in research-state.json and raw logs for current execution state.

Preselection and caches are frozen. Current package QA: 80 tests passed. Natural holdout has no neutrophil or epithelium GT support. Prospective confirmation was historically exposed. Original Explorations1-5 remain unchanged; copied Exploration4 finite study is already complete and must not be duplicated. Master backup exists; master has not been appended. Final reporting waits for study convergence.
'''
(W/'findings.md').write_text(findings,encoding='utf-8')
(O/'RESEARCH_STATE/FINDINGS_CURRENT.md').write_text(findings,encoding='utf-8')
print(json.dumps(dict(updated_utc=stamp,runs=len(runs),complete=d['completed_runs'])))
