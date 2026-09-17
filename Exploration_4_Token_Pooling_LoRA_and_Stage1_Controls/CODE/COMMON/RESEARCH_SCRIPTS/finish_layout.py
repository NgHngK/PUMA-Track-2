from bootstrap import *
P=R/'EXPLORATION_4';D=P/'ARCHITECTURES/FINAL_RECOMMENDED_A5'
for name in ['RESULT_INDEX.csv','ORIGINAL_PATH_MAP.csv']:
 path=D/name;cp(path,D/'PROVENANCE'/(name+'.initial_draft.txt'));rows=list(csv.DictReader(path.open(encoding='utf8')))
 for r in rows:r['new_path']=r['new_path'].replace('EXPLORATION_3\\ARCHITECTURES\\A5','EXPLORATION_4\\ARCHITECTURES\\FINAL_RECOMMENDED_A5')
 csvout(path,rows)
for n in range(1,5):
 d=R/f'PROMPT_{n}';path=d/'README.md'
 if not path.exists():path.write_text(f'''# Exploration {n} archive

REPORT_PROMPT_{n}.md is the phase report. EXPLORATION_TEXT preserves the available request text. ARCHITECTURES contains local code, configs, results, flowcharts and lineage for material models. SHARED_INPUT_SNAPSHOTS and CACHE_REFERENCES identify immutable inputs; external RGB and foundation weights remain referenced by original paths and hashes. PROVENANCE preserves source documents and verification evidence. TABLES/COMPLETE_EPOCH_LEDGER.md includes every available unique epoch record.

Read the root MASTER_REPORT_PROMPTS_1_TO_4.md for the chronological integrated decision. Historical recommendations apply to their original phase. The final recommendation is retained Exploration3 A5; Exploration4 A3 failed its class-recall confirmation constraint. Full-scale training and external validation remain unexecuted.
''',encoding='utf8')
path=R/'README.md'
path.write_text('''# PUMA Stage2 research archive — Explorations1–4

Start with **MASTER_REPORT_PROMPTS_1_TO_4.md**. It includes the chronological research narrative, historical source-recovered architecture specifications, exact endpoint tables, retained implementation and all4,437 unique archived epoch records. The complete report is large because no available epoch was filtered away.

**One next full-scale architecture:** EXPLORATION_4/ARCHITECTURES/FINAL_RECOMMENDED_A5. The target-neighborhood candidate improved internal CV but failed the predeclared class-recall confirmation rule, so Exploration3 A5 remains the recommendation. No runner-up shopping or unsupported SOTA claim is made.

Every architecture has independent local modules and explicit cache references. Install its requirements in an isolated environment. Cached-head reproduction does not require downloading encoder weights. New raw-image extraction requires the user's dataset and verified UNI2-h checkpoint. Full-proposal evaluation requires complete ROI annotations, all fixed proposals, ROI splits and documented Stage1 exclusion. Do not relabel matched-only or GT-centered subset results as end-to-end performance.

The initial3040-file historical inventory remains MASTER_ARTIFACT_INDEX.csv. MASTER_ARTIFACT_INDEX_COMPLETE.csv inventories the final archive. ORIGINAL_IMMUTABILITY_VERIFICATION.csv verifies originals against initial hashes. REPORT_DUPLICATE_LINEAGE.csv and EPOCH_RECORD_LINEAGE.csv explain exact duplicate suppression in the monograph while original files remain intact. USER_ASSET_LINEAGE.csv maps supplied documents. ARCHIVE_VERIFICATION_REPORT.json lists checks, limitations and unresolved attribution. Original unresolved files are preserved, not assigned invented provenance.

Historical exclusions and incomplete executable recovery remain labeled. Package CLI/evaluator checks do not imply every historical training run was rerun. Recorded checkpoint-forward tests have their exact scope in each PROVENANCE directory. The final ZIP excludes git internals, bytecode and ephemeral download transports; its member hashes are independently verified.
''',encoding='utf8')
# Readability supplement to historical generic diagrams, with exact branch equations retained.
for n in [1,2,3]:
 for d in (R/f'PROMPT_{n}/ARCHITECTURES').iterdir():
  if not d.is_dir():continue
  path=d/'PROVENANCE/INTERPRETATION_LIMITS.md';path.parent.mkdir(exist_ok=True)
  if not path.exists():path.write_text('''# Historical reconstruction limits

The original source, configuration and measured parameter count take precedence over an explanatory generic flowchart. The diagram's unused biology or encoder paths are absent for appearance-only and biology-only models. Exploration1 CNN uses48-pixel model input and a trainable CNN; it is not a frozen UNI2 model. Exploration1 semantic metrics predate exact localV17 parity. Exploration3 B-family BioMask results are contaminated upstream diagnostics and cannot promote an architecture. The corrected and excluded historical LoRA runs are separate evidence; uncertain executable versions are never reconstructed by guessing.

The full raw history, model, prediction and normalizer artifacts are preserved where they existed. Missing original fields remain NOT RECORDED IN ORIGINAL RUN. New package verification is explicitly limited to the checks recorded in PROVENANCE.
''',encoding='utf8')
print('Layout complete')
