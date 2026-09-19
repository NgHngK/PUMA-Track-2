"""Recover all recorded historical epoch fields without training or rewriting originals."""
from pathlib import Path
import csv,json,hashlib,collections
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';T=O/'TABLES';dest=O/'REPORTS/HISTORICAL_EXPLORATIONS1_TO5_EPOCH_CONFIG_SUPPLEMENT.md'
csv.field_size_limit(10000000)
M='NOT RECORDED IN ORIGINAL RUN'
def esc(v):return str(v).replace('|','\\|').replace('\r','').replace('\n','<br>').replace(M,'NR')
def table(f,rows,fields):
 f.write('| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
 for r in rows:f.write('| '+' | '.join(esc(r.get(k,M)) for k in fields)+' |\n')
 f.write('\n')
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
datasets=['HISTORICAL_PROMPT1','HISTORICAL_PROMPT1_EXCLUDED','HISTORICAL'];lineage=[];n=0
with dest.open('w',encoding='utf-8') as f:
 f.write('# Historical Explorations1–5 epoch and configuration supplement\n\nThis is an additive recovery of original records, prepared during Exploration6. It does not replace the historical reports or rerun historical experiments. Exploration5 executed zero new model-training runs. **NR means NOT RECORDED IN ORIGINAL RUN** throughout this supplement; it never means zero. Original evaluator namespaces and uncertainty labels remain authoritative. Missing metrics are not reconstructed from rounded prose. Every recorded epoch is included. Repeated archive aliases are identified by source/run/hash and retained outside this canonical rendering.\n\n')
 f.write('The separate Exploration1 excluded-LoRA subsection is **EXCLUDED / NOT EXECUTABLE AS ORIGINALLY RUN**. Its numbers are preserved as historical diagnostics, never promoted as valid architecture evidence. The unresolved imported history remains **UNRESOLVED — DO NOT ATTRIBUTE**. Raw CSVs and histories preserve all original precision.\n\n')
 for dataset in datasets:
  folder=T/dataset;configs=read(folder/'RUN_CONFIG_SNAPSHOTS.csv');groups={}
  for name in ['EPOCH_GLOBAL_METRICS.csv','EPOCH_PER_CLASS_METRICS.csv','EPOCH_PUMA_PER_CLASS_METRICS.csv','EPOCH_SAMPLER_EXPOSURE.csv','EPOCH_MODEL_DIAGNOSTICS.csv','CONFUSION_BY_EPOCH.csv']:
   mapping=collections.defaultdict(list)
   for row in read(folder/name):mapping[row['run_id']].append(row)
   groups[name]=mapping
   p=folder/name;lineage.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  f.write(f'## {dataset}\n\n')
  for cfg in configs:
   run=cfg['run_id'];n+=1
   f.write(f'### Exploration{cfg["prompt"]}: {run}\n\n')
   f.write(f'Original history: `{cfg["history_path"]}`. SHA256 `{cfg["history_sha256"]}`. Recorded epochs: {cfg["epochs"]}. Seed/fold: {cfg["seed"]}/{cfg["fold"]}.\n\n')
   f.write('Resolved configuration as recovered from the original artifact (not a claim that undocumented defaults were logged):\n\n```json\n'+cfg['resolved_config']+'\n```\n\n')
   f.write('Global metrics and optimizer/exposure records, every recorded split and epoch:\n\n')
   fields=['epoch','split','objective','loss','macro_precision','macro_recall','macro_f1','balanced_accuracy_fixed10','accuracy','nll','ece15','roi_macro_precision','roi_macro_recall','roi_macro_f1','pooled_macro_f1','lr','gradients','unique_examples','unique_groups','repeat_fraction','checkpoint_status','early_stopping_counter']
   table(f,groups['EPOCH_GLOBAL_METRICS.csv'][run],fields)
   f.write('All ten classes per recorded epoch/split:\n\n')
   table(f,groups['EPOCH_PER_CLASS_METRICS.csv'][run],['epoch','split','class_name','support','predicted','precision','recall','f1','TP','FP','FN','dominant_confusion'])
   diag={}
   for r in groups['EPOCH_PER_CLASS_METRICS.csv'][run]:
    key=(r['epoch'],r['split']);v=r['class_diagnostics']
    if key in diag:assert diag[key]==v
    diag[key]=v
   for key,v in diag.items():
    if v!=M:f.write(f'Epoch{key[0]} {key[1]} recorded class diagnostics:\n\n```json\n'+v+'\n```\n\n')
   f.write('V17 ROI and pooled class metrics are distinct from semantic classification metrics. Counts reflect the original matching contract, not a replacement confusion calculation.\n\n')
   table(f,groups['EPOCH_PUMA_PER_CLASS_METRICS.csv'][run],['epoch','split','class_name','roi_precision','roi_recall','roi_f1','pooled_precision','pooled_recall','pooled_f1','pooled_TP','pooled_FP','pooled_FN'])
   f.write('Confusion rows. Column order: tumor, lymphocyte, plasma_cell, histiocyte, melanophage, neutrophil, stroma, epithelium, endothelium, apoptosis. Every available integer is preserved; absence of rows means NR.\n\n')
   cm=collections.OrderedDict()
   for row in groups['CONFUSION_BY_EPOCH.csv'][run]:
    key=(row['epoch'],row['split'],row['true_class']);cm.setdefault(key,[]).append(row['count'])
   table(f,[dict(epoch=k[0],split=k[1],true_class=k[2],counts='['+','.join(v)+']') for k,v in cm.items()],['epoch','split','true_class','counts'])
   f.write('Sampler exposure:\n\n')
   er=groups['EPOCH_SAMPLER_EXPOSURE.csv'][run]
   table(f,er,['epoch','class_name','draws','unique_nuclei','unique_groups','repeat_fraction'])
   # Bucket maps were repeated once per class in the normalized CSV. Print the
   # identical epoch map once and verify it is identical before deduplicating.
   unique={}
   for r in er:
    key=r['epoch'];v={k:r[k] for k in ['roi_exposure','bucket_exposure']}
    if key in unique:assert unique[key]==v
    unique[key]=v
   for e,v in unique.items():
    if any(x!=M for x in v.values()):f.write(f'Epoch{e} recorded ROI/bucket exposure:\n\n```json\n'+json.dumps(v,separators=(',',':'))+'\n```\n\n')
   f.write('Additional recorded model/gradient diagnostics; no numerical field is dropped from these recovered records:\n\n')
   for r in groups['EPOCH_MODEL_DIAGNOSTICS.csv'][run]:f.write('```json\n'+json.dumps(r,separators=(',',':'))+'\n```\n\n')
 f.write('## Unresolved imported Exploration1 history\n\nThe18-epoch imported Drive history has409 original columns. Its compact seven-column version is a numerically identical projection at tolerance1e−12. Both physical originals remain preserved. These records have unresolved executable/evaluator lineage and cannot be attributed to a new Exploration6 trial. The following preserves every original field without substituting canonical evaluator names.\n\n')
 table(f,read(T/'HISTORICAL_IMPORTED_UNRESOLVED/drive_history_ALL_RECORDED_FIELDS.csv'),['epoch','field','recorded_value'])
 f.write('## Exploration5 training/configuration clarification\n\nThe authoritative Exploration5 final decision reports zero new model-training runs. Its analyses reuse historical epochs already shown above; do not double-count these as Exploration5 trials. The frozen retained baseline is Exploration3 A5, 27,866 trainable parameters. Mean epoch10 historical CV semantic F1: TRAIN0.9095012032899681, held-fold0.4190086273104059, gap0.4904925759795622. The proposed clean full-population baseline was prepared but not executed, with upstream exclusion and data-admission blockers. This is the historical decision, not a claim about the amended Exploration6 study.\n\n')
 for name in ['baseline_config.json','PROMPT5_FINAL_DECISION.json']:
  p=R/'PUMA_STAGE2_PROMPT5'/name;f.write(f'### Original Exploration5 {name}\n\n```json\n'+p.read_text(encoding='utf-8')+'\n```\n\n');lineage.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 f.write('## Source lineage\n\n');table(f,lineage,['path','sha256'])
(O/'PROVENANCE/HISTORICAL_SUPPLEMENT_LINEAGE.json').write_text(json.dumps(dict(report=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,canonical_histories=n,sources=lineage,status='Historical supplement only; master not appended; Exploration6 not complete'),indent=2),encoding='utf-8')
print(json.dumps(dict(histories=n,bytes=dest.stat().st_size)))
