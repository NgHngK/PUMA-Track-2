from pathlib import Path
import json,datetime,hashlib
r=Path('2026-09-07/use-the-autoresearch-skill-x20-https'); o=r/'outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'; w=r/'work/exploration6'; s=o/'RESEARCH_STATE'; stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=s/'AUGMENTATION_PIPELINE_STATUS.json'; prior=json.loads(p.read_text()); (s/'AUGMENTATION_PIPELINE_STATUS_PRE_INTERRUPTION.json').write_text(json.dumps(prior,indent=2))
record=dict(utc=stamp,status='BLOCKED_INTERRUPTED_PARTIAL_RUN',run_id='p6_B_AUG0_s43',completed_epochs=6,completed_runs=23,started_runs=24,evidence='psutil process inventory found no training Python process or finite runner; no summary.json; last recorded epoch6; empty run log.',cause='Unknown; no exception or shutdown cause established.',policy='Partial run preserved. No automatic restart, duplicate, replacement seed or downstream promotion.',required_action='User authorization for a documented recovery plan for interrupted seed43; preserve original partial artifacts.',holdouts_opened=False)
(s/'EXTERNAL_BLOCKER_INTERRUPTED_AUG0_S43.json').write_text(json.dumps(record,indent=2)); p.write_text(json.dumps(record,indent=2))
for q in [s/'WORKING_STATE.json',w/'research-state.json']:
 d=json.loads(q.read_text()); d.update(status=record['status'],updated_utc=stamp,active_runs=[],augmentation_pipeline=record,next_required_input=record['required_action'],current_stage='B blocked: seed43 interrupted after6epochs; no training process active',continuity='Continuation blocked pending authorized recovery; no partial run restart.');
 for run in d['run_registry']:
  if run['run_id']==record['run_id']:run['status']='INTERRUPTED_PARTIAL'
 q.write_text(json.dumps(d,indent=2))
notice=f'\n\n## Execution interruption ({stamp})\n\nAUG0 seed29 completed; 23 runs are complete. AUG0 seed43 stopped after six epochs with no completion summary. Neither training process nor finite runner is present. Cause unknown. Prior RUNNING status was stale. Original partial artifacts are preserved; no restart or replacement was launched. Both holdouts remain unopened. Current ledgers include 266 recorded epochs with sampler replay PASS. Recovery requires user authorization under the explicit no-automatic-restart instruction.\n'
for q in [w/'findings.md',s/'FINDINGS_CURRENT.md']:q.write_text(q.read_text()+notice)
print(json.dumps(record))
