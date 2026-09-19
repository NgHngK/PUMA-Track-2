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
d['active_runs']=[x['run_id'] for x in runs if x['status']=='IN_PROGRESS']
pipeline=O/'RESEARCH_STATE/AUGMENTATION_PIPELINE_STATUS.json'
d['augmentation_pipeline']=json.loads(pipeline.read_text()) if pipeline.exists() else None
active_text=', '.join(f"{x['run_id']} ({x['joint_epochs']} completed epochs)" for x in runs if x['status']=='IN_PROGRESS') or 'No partial run directory; inspect pipeline status before launching anything'
parity_path=O/'RESEARCH_STATE/RAW_CACHED_PARITY_SEED17_FINAL.json'
parity_text='Final raw/cached parity remains pending.'
if parity_path.exists():
 parity=json.loads(parity_path.read_text());d['raw_cached_seed17_final_parity']=parity
 parity_text=f"Seed17 raw AUG0 completed all10 epochs: ROI F1 {parity['raw_roi_f1']}, exactly matching cached ROI F1; {parity['prediction_disagreements']} DEV prediction disagreements; maximum absolute logit error {parity['max_abs_logit_error']}. Runtime {parity['raw_runtime_seconds']/3600:.2f}hours. This validates the seed17 parity comparison, not augmentation efficacy."
payload=json.dumps(d,indent=2)
d['conditional_cached_C']='Complete:12 runs; all four candidates lose mean ROI F1 against reused D900 inverse controls; no promotion'
d['historical_supplement']='REPORTS/HISTORICAL_EXPLORATIONS1_TO5_EPOCH_CONFIG_SUPPLEMENT.md;473 canonical histories plus18 unresolved imported epochs; not yet appended'
d['archival_verification']='17,617 original/archive/master checks passed;96 Exploration5 scientific files copied and verified'
d['standalone_package']='ARCHITECTURES/A5_FROZEN_CLS_RANK8:21 completed runs;79 initial test passes plus repaired packaging-fixture parity test passed'
d['background_sequence']='RESEARCH_STATE/AUGMENTATION_PIPELINE_STATUS.json; finite nine-config sequence; partial runs never auto-restarted'
d['continuity']='User explicitly renewed autoresearch on2026-09-10 local time. Native Codex heartbeat puma-stage-2-autoresearch is ACTIVE every20minutes in this task; existing paused Exploration4 continuation was updated rather than duplicated. Quiet on unchanged state. Finite raw training runner also remains active.'
d['next_stage_preparation']='HPO/FROZEN_INTERNAL_FOLDS_D900 has3 frozen TRAIN-only folds; all10classes in each train/dev fold and every900TRAINUID held exactly once. No D trials launched. HPO_CONFIG_RESOLUTION_01 fixes patience5 per preregistration.'
d['shared_input_archive']='32 files,2,254,709,962bytes verified in SHARED_INPUT_SNAPSHOTS; includes one full token-cache copy'
payload=json.dumps(d,indent=2)
p.write_text(payload,encoding='utf-8');(W/'research-state.json').write_text(payload,encoding='utf-8')
findings=f'''# Exploration6 current findings

Updated {stamp}. Study remains in progress; no final recipe has been selected and neither holdout has been opened.

Canonical artifacts: {O}

Stage A: nine completed controls, three seeds at each nested training size. Mean DEV semantic F1 D300/D600/D900 = 0.390178/0.391375/0.427800; ROI fixed10 F1 = 0.129493/0.133084/0.149770. D900-D300 paired ROI delta +0.020278, 95% ROI bootstrap interval [0.007568,0.033165], three positive seeds. Neutrophil recall delta -0.240741 fails the predeclared class-harm guard. This is mixed evidence, not automatic promotion. Exact values: TABLES/DATA_SCALING_SUMMARY.json.

Active run directories: {active_text}. The finite B runner executes the frozen raw configs sequentially; status is RESEARCH_STATE/AUGMENTATION_PIPELINE_STATUS.json. Stage C cached no-augmentation sampling/cRT controls completed under frozen SCHEDULING_AMENDMENT_01 and remain conditional until B resolves. Mean ROI F1: inverse0.149770, natural0.138005, tempered0.140028, cRT reset0.137769, cRT no-reset0.140762. All fail promotion. Reset lowers gap0.478312 to0.390362 but also lowers ROI F1; gap reduction alone is insufficient. Exact values: TABLES/CACHED_LONG_TAIL_SUMMARY.json. Never restart existing partial or completed run directories. Read run_registry and raw logs for current execution state.

The initial A/C archive covers240 epochs, including30 decoupled epochs, with verified sampler replay; current all-run ledgers also incorporate finished raw epochs. The standalone A5 package preserves21 A/C completed runs. All80 test cases passed across the initial79 passes and the corrected missing-vendor parity fixture test. Historical original/archive/master recheck passed17,617 checks; initial421 missing-path reports were Windows long-path false negatives, not lost files. Exploration5 has96 verified archival copies. The additive historical supplement contains473 canonical histories and all18 unresolved imported epochs; it has not been appended to the master. Figures are under FIGURES with input hashes and source scripts.

{parity_text}

Preselection and caches are frozen. Current package QA: 80 tests passed. Natural holdout has no neutrophil or epithelium GT support. Prospective confirmation was historically exposed. Original Explorations1-5 remain unchanged; copied Exploration4 finite study is already complete and must not be duplicated. Master backup exists; master has not been appended. Final reporting waits for study convergence.

Autoresearch continuity: native Codex heartbeat puma-stage-2-autoresearch is active every20minutes in this task following the user's renewed skill request. It replaces the paused old Exploration4 continuation, without creating a duplicate. Stay quiet on unchanged state and keep the finite study moving. Do not infer completion from this checkpoint. HPO fold manifests are prepared but no D trial has started. Preserve the original protocol's patience5 when resolving generated HPO configs; see HPO_CONFIG_RESOLUTION_01.json.

Interpretation guard: fixed10-epoch D300/D600/D900 controls also differ in optimizer updates (50/100/150 per seed), so do not attribute the entire gain solely to independent data diversity. Annotation UID and V17 component counts differ: D900 TRAIN900/903, DEV225/226, confirmation225/225, natural10712/10725. Details are in DATASET_X3/ANNOTATION_VS_COMPONENT_CENSUS.csv. No holdout model predictions have been opened.
'''
(W/'findings.md').write_text(findings,encoding='utf-8')
(O/'RESEARCH_STATE/FINDINGS_CURRENT.md').write_text(findings,encoding='utf-8')
print(json.dumps(dict(updated_utc=stamp,runs=len(runs),complete=d['completed_runs'])))
