from pathlib import Path
import json,csv,hashlib,shutil,sys,importlib.metadata as md
R=Path(r'C:\Users\Hngk\Documents\Codex'); C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE'
O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
sys.path.insert(0,str(C/'src'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):(O/p).write_text(json.dumps(x,indent=2),encoding='utf-8')
from puma_exploration6.config import ExperimentConfig
cfg=[]
for p in sorted((C/'configs').rglob('*.json')):
    ExperimentConfig.load(p);cfg.append(dict(path=str(p),status='PASS',sha256=sha(p)))
save('PROVENANCE/CONFIG_VALIDATION.json',cfg)
a=json.loads((O/'PROVENANCE/CODEBASE_AUDIT.json').read_text());a.update(configs=cfg,tests_status='FAILED',tests_count=74,tests_pass=71,tests_fail=3,tests_seconds=41.03,failed_tests=['test_internal_cv_preserves_identity_and_hides_external_holdouts','test_prepare_cv_hpo_generates_trial_x_fold_configs','test_cv_hpo_end_to_end_with_shared_identity_bound_caches'],failure='Grouped CV lacks canonical class in validation fold; original exception fold 0 dev missing [8]. Root cause/version dependence not yet established.',environment_packages={x:md.version(x) for x in ['numpy','scipy','scikit-learn','torch','Pillow','shapely']},git_information='No .git directory at certified package root; shipped SHA manifest used as source identity',deviations=['Local full suite does not reproduce 74/74 certificate result. No source patch applied.'])
save('PROVENANCE/CODEBASE_AUDIT.json',a)
shutil.copyfile(R/'prompt6_initial_pytest.log',O/'PROVENANCE/INITIAL_PYTEST.log')
for name in ['exploration6_audit.py','exploration6_save_admission.py']:
    shutil.copyfile(R/name,O/'PROVENANCE'/name)
original=R/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/sample_manifest.csv'
rows=list(csv.DictReader(original.open(encoding='utf-8-sig',newline='')))
derived=list(csv.DictReader((R/'PUMA_STAGE2_PROMPT5/SPLIT_MANIFEST_AUDIT.csv').open(encoding='utf-8-sig',newline='')))
assert {(x['uid'],x['roi']) for x in rows}=={(x['uid'],x['roi']) for x in derived}
save('PROVENANCE/HISTORICAL_MANIFEST_IDENTITY.json',dict(original_path=str(original),sha256=sha(original),rows=len(rows),rois=len({x['roi'] for x in rows}),prompt5_uid_roi_identity=True))
data=json.loads((O/'DATASET_X3/HISTORICAL_OVERLAP_PREFLIGHT.json').read_text())
short=[x for x in data['support'] if x['remaining_nuclei']<x['locked_required']]
state=dict(status='BLOCKED_DATA_ADMISSION',prompt6_complete=False,current_retained_baseline='Exploration3_A5; historical retention only, no new efficacy claim',architecture='z = W h + a + R((P_h h) * (P_b b)); h=non-affine LN(CLS1536), TRAIN-standardized Tier-A16; rank8',trainable_parameters=27866,new_research_training_runs=0,qa_synthetic_tests_are_not_research_trials=True,positive_prior_evidence=['Historical A5 retained','Exploration4 local/global-local internal gains require fresh replication'],negative_evidence=['Exact Exploration4 Q/V LoRA r8 alpha16 last4 five-epoch configuration showed no ROI-F1 gain','Selected local-only representation failed class guards'],unresolved_hypotheses=['Independent data scaling','Deterministic repeat exposure','Augmentation','cRT','Regularization/HPO','Tier-A incremental held-group utility','Global-local replication'],prohibited_reruns=['Unchanged failed Exploration4 restricted LoRA','Old confirmation tuning','LOCKED reuse'],dataset_lineage=dict(historical_rows=450,historical_rois=181,available_rois=205,remaining_rois=24,shortfalls=short),reused_validation='Historical 150-nucleus confirmation is reused development evidence',known_limitations=['PATIENT-LEVEL INDEPENDENCE UNVERIFIED','Fresh ten-class LOCKED225 impossible from remaining ROI pool','Three local CV QA failures unresolved','CPU-only local torch'],next_required_input='Additional historically unused annotated ROIs with adequate all-ten-class support, image paths, and strongest available patient/case/slide mapping; alternatively an explicit protocol amendment accepting development-only evaluation without claiming untouched LOCKED confirmation.',pending_work=['Resolve CV QA failure with mandatory patch protocol if needed','Complete raw historical evidence ingestion/ledgers','Admit independent x3 dataset','Execute stages A through J in order','Generate final report only at convergence','Append reports and verify content identity'],master_report_status='Byte-preserving backup created; master not appended',continuity='Resumable state saved; Claude/OpenClaw wall-clock mechanisms do not apply to this Codex session. No recurring automation requested or created.')
save('RESEARCH_STATE/WORKING_STATE.json',state)
(O/'PROVENANCE/AUTORESEARCH_SKILLS_USED.md').write_text('''# Skills materially applied

NO TRAINING RUNS ARE REPORTED IN THIS DOCUMENT.

- `C:/Users/Hngk/.codex/skills/0-autoresearch-skill/SKILL.md`: establish evaluation and baseline first; test falsifiable prerequisites; record negative evidence; distinguish exploratory diagnostics from confirmatory trials; preserve resumable state. Its generic indefinite search and final-paper workflow yield to Exploration 6 admission gates and convergence rules.
- `C:/Users/Hngk/.codex/skills/0-autoresearch-skill/references/skill-routing.md`: select domain skills only when applicable.
- `C:/Users/Hngk/.codex/skills/ml-training-recipes/SKILL.md`: check data leakage, reproducibility, sampler behavior, training/evaluation diagnostics before HPO. Generic optimizer defaults do not override certified A5.

No PEFT, training or literature experiment was entered; their phase-specific references remain pending. No subagents used. No Claude/OpenClaw continuity tool is available in this Codex session; checkpointed state is used and no unsolicited recurring automation is created.
''',encoding='utf-8')
(O/'PROVENANCE/CODE_PATCH_LOG.md').write_text('''# Code patch log

NO TRAINING RUNS ARE REPORTED IN THIS DOCUMENT.

CERTIFIED CODE USED WITHOUT MODIFICATION

Initial reproduction: `PYTHONPATH=src; python -m pytest -q` with workspace pytest dependency path additionally supplied. Full output: `INITIAL_PYTEST.log`.

74 collected: 71 passed, 3 failed in grouped CV. The direct error is missing validation class 8 in fold 0; the other two failures arise in prepare_cv_hpo subprocesses. Root cause and dependency-version contribution are unresolved. No fold-support guard has been weakened and no source behavior changed. Data admission independently fails; do not launch HPO until QA is repaired and the full suite passes.

Non-source work: installed pytest into the workspace-local prompt6_runtime directory; added independent provenance/census scripts outside the certified package. No historical artifact or reference hash was changed.
''',encoding='utf-8')
table='| Class | Available outside historical ROIs | Required LOCKED |\n|---|---:|---:|\n'+''.join(f"| {x['class_name']} | {x['remaining_nuclei']} | {x['locked_required']} |\n" for x in data['support'])
report='''# Exploration 6 admission audit — incomplete research

NO TRAINING RUNS ARE REPORTED IN THIS DOCUMENT.

Exploration 6 cannot yet execute under its requested independence contract. This is an admission finding, not experimental convergence or a model comparison. No new efficacy metrics, finalist, LOCKED evaluation, or final decision are claimed.

## Evidence

The raw 205-ROI GeoJSON census was compared with the original Exploration-3 450-nucleus manifest. Its UID/ROI pairs exactly match the Exploration-5 derived manifest; 181 distinct ROIs were historically used. Excluding those alone leaves 24 ROIs. This is an optimistic upper bound: other historical exposure can only reduce availability. No predictions or model-selection metrics were inspected on a new LOCKED set.

'''+table+'''
Even LOCKED225 alone cannot be assembled: plasma_cell lacks 10 annotations, melanophage lacks 7, neutrophil lacks 18, epithelium lacks 18. Supplying those numeric deficits is only a necessary lower bound, not sufficient group-diversity or DEV certification. New nuclei inside historically used ROIs do not supply untouched groups. Stronger grouping cannot fix the lack of class support outside those ROIs.

Available GeoJSON feature properties contain classification, isLocked, and objectType; `isLocked` is annotation-editor state, not an untouched research-test designation. No higher grouping map was supplied in the dataset directory. PATIENT-LEVEL INDEPENDENCE UNVERIFIED.

## Code and environment

All 133 shipped SHA records match. All 22 JSON configs validate; package compilation succeeds; all 16 public CLI help commands succeed. Full local pytest: 71 passed, 3 grouped-CV failures. See provenance for exact failure and versions. Source remains unchanged. Python 3.10 uses CPU-only PyTorch 2.11.0; UNI2 weights were located at `D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin` but not loaded or certified in this admission phase.

## Preserved work and resumption

`RESEARCH_STATE/WORKING_STATE.json` records pending stages and prior constraints; historical narrative evidence there is provisional until full raw-ledger ingestion completes. `DATASET_X3/HISTORICAL_OVERLAP_PREFLIGHT.json` and `FRESH_GROUP_CLASS_SUPPORT.csv` record the admission proof. `PROVENANCE` contains source hashes, original-manifest identity, complete test output, config validation, CLI help, and deterministic audit scripts.

The master report has a byte-preserving backup and has not been appended. No final research report has been generated because Section 37 requires experimental convergence first. Historical full-ledger supplements and the authoritative Exploration-5 append remain pending.

Resume with additional historically unused annotated ROI images and class/group metadata. If only the existing dataset may be used, the user must explicitly amend the untouched-evaluation requirement; any such results must be labeled development-only. Independently resolve the grouped-CV QA failures before HPO. Then complete the prescribed historical ingestion, data admission and controlled stages A–J before report generation and append.
'''
(O/'REPORTS/EXPLORATION6_ADMISSION_AUDIT.md').write_text(report,encoding='utf-8')
(O/'README.md').write_text(report,encoding='utf-8')
save('PROVENANCE/ADMISSION_ARTIFACT_HASHES.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='ADMISSION_ARTIFACT_HASHES.json'})
assert all(sha(C/r['path'])==r['expected'] for r in json.loads((O/'PROVENANCE/SOURCE_HASH_VERIFICATION.json').read_text()))
print(json.dumps(dict(status=state['status'],configs_validated=len(cfg),original_manifest_rows=len(rows),shortfall_classes=[x['class_name'] for x in short],report=str(O/'REPORTS/EXPLORATION6_ADMISSION_AUDIT.md'),certified_source_unchanged=True)))
