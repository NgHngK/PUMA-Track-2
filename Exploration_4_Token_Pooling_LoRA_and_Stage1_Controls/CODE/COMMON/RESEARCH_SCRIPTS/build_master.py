from bootstrap import *
P=R/'EXPLORATION_4'
def table(path):
 rows=list(csv.DictReader(path.open(encoding='utf8')))
 if not rows:return ''
 fields=list(rows[0]);return '| '+' | '.join(fields)+' |\n|'+'|'.join(['---']*len(fields))+'|\n'+'\n'.join('| '+' | '.join(str(r[k]).replace('|','\\|') for k in fields)+' |' for r in rows)+'\n'
text=(W/'report_prompt4.md').read_text(encoding='utf8')
for old,new in [('one96','one 96'),('from16','from 16'),('with16','with 16'),('contains97','contains 97'),('across205','across 205'),('ratio is156','ratio is 156'),('total97','total 97'),('mean ROI F1=','mean ROI F1 = '),('The original450','The original 450'),('contains300','contains 300'),('and150','and 150'),('with100','with 100'),('uses5000','uses 5000'),('and RNG1701','and RNG 1701'),('the150','the 150'),('all156','all 156'),('on294','on 294'),('fixed epoch10','fixed epoch 10'),('fixed epoch5','fixed epoch 5'),('Exploration3','Exploration 3'),('Exploration4','Exploration 4'),('Exploration2','Exploration 2'),('Exploration1','Exploration 1'),('Stage1','Stage 1'),('Stage2','Stage 2')]:text=text.replace(old,new)
text+='\n## Exact endpoint tables\n\n'
for title,path in [('A/B internal CV',P/'TABLES/A_complete/seed_summary.csv'),('ROI-class sampler and other completed CV controls',P/'TABLES/D_complete/seed_summary.csv'),('Plain-head representation diagnostic',P/'TARGET_POOLING/DIAGNOSTICS/plain_head_endpoints.csv'),('Proposal alignment',P/'STAGE1_PROPOSALS/alignment_endpoints.csv'),('Restricted LoRA',P/'LORA/endpoints.csv'),('Clean mask pooling',P/'BIOMASK_GUIDED/endpoints.csv'),('Final confirmation',P/'TABLES/confirmation_endpoints.csv')]:
 if path.exists():text+=f'### {title}\n\nSource: [{path.name}]({str(path.relative_to(R)).replace(chr(92),"/")}).\n\n'+table(path)+'\n'
text+='\n## Architecture package directory\n\n'
for d in sorted((P/'ARCHITECTURES').iterdir()):
 if (d/'ARCHITECTURE.md').exists():text+=f'- [{d.name}]({str((d/"ARCHITECTURE.md").relative_to(R)).replace(chr(92),"/")}) — local code, configuration, results and provenance.\n'
(P/'REPORT_EXPLORATION_4.md').write_text(text,encoding='utf8')
intro='''# PUMA Stage 2 — integrated Explorations 1–4 research monograph

**Final decision: retain Exploration3 A5 as the single next full-scale architecture.** The frozen UNI2-h CLS96 encoder representation and rank-eight interaction with16 RGB/point features remain the recommendation. Exploration4 neighborhood pooling improved internal CV but failed the predeclared class-recall confirmation constraint. No runner-up was selected. No SOTA, clinical, external-validation or full-scale-training achievement is claimed.

This monograph is chronological. The historical chapters preserve their original claims and decisions as historical evidence; their completion statements and earlier recommendations apply only to their own prompt. The final Exploration4 decision supersedes them. Original artifacts remain unchanged, including unresolved evidence. REPORT_DUPLICATE_LINEAGE.csv records exact repeated content blocks; each unique historical block appears once below. EPOCH_RECORD_LINEAGE.csv similarly maps repeated epoch-record copies.

The repository includes standalone architecture packages, copied evaluator source and parity tests, original result/checkpoint/prediction files, shared immutable cache references, exact configs and artifact lineage. The full epoch appendices are intentionally large and include failures and diagnostics. Missing original measurements remain explicitly missing. The archive is a finite research study and execution blueprint, not a deployed medical system.

## Navigation

- Historical Exploration1: baseline ingestion and verified encoder feasibility.
- Historical Exploration2: exact local V17 parity, FOV/loss and simple biology controls.
- Historical Exploration3: finite head search, rank-eight interaction and BioMask provenance.
- Exploration4: representation, multiscale, clean alignment, sampling, restricted LoRA, clean mask pooling and failed confirmation.
- Architecture specifications and the retained implementation.
- Complete available epoch ledgers and supplied-document evidence.

'''
parts=[intro]
for n in [1,2,3]:parts.append((R/f'PROMPT_{n}/REPORT_PROMPT_{n}.md').read_text(encoding='utf8'))
parts.append(text)
parts.append('\n# Architecture specifications across all prompts\n\nThe following source-recovered specifications accompany each independent package. Exploration4 identifiers A3 and Exploration3 A3 denote different designs; prompt numbers are mandatory when interpreting architecture names.\n\n')
for n in range(1,5):
 for d in sorted((R/f'PROMPT_{n}/ARCHITECTURES').iterdir()):
  if (d/'ARCHITECTURE.md').exists():parts.append(f'\n## Package Exploration{n}/{d.name}\n\n'+(d/'ARCHITECTURE.md').read_text(encoding='utf8'))
parts.append('\n# Self-contained retained full-scale implementation\n\nThese modules are copied from the verified retained architecture. The package contains the remaining local dependencies. Full-scale training has not been executed. Use the standalone README and explicit external data/weights contracts.\n\n')
D=P/'ARCHITECTURES/FINAL_RECOMMENDED_A5/CODE'
for name in ['dataset.py','crop.py','biology.py','model.py','loss.py','features.py','train_eval.py','evaluation.py']:
 parts.append(f'## {name}\n\n```python\n'+(D/name).read_text(encoding='utf8')+'\n```\n\n')
parts.append('\n# Complete available epoch ledgers\n\nNo epoch filtering is applied. Exact raw sources and hashes accompany every record. Per-class precision, recall, F1, support, predicted counts, confusion, calibration and operational fields are included whenever recorded. Full per-ROI arrays and per-batch logs remain in the linked immutable files.\n\n')
for n in range(1,5):parts.append((R/f'PROMPT_{n}/TABLES/COMPLETE_EPOCH_LEDGER.md').read_text(encoding='utf8'))
parts.append('\n# Supplied-document evidence\n\nThese documents are historical inputs, not newly executed results or operational instructions.\n\n'+(R/'EXPLORATION_1/EXTERNAL_DOCUMENT_EVIDENCE.md').read_text(encoding='utf8'))
dest=R/'MASTER_REPORT_PROMPTS_1_TO_4.md';dest.write_text('\n\n'.join(parts),encoding='utf8')
print('Master report bytes',dest.stat().st_size)
