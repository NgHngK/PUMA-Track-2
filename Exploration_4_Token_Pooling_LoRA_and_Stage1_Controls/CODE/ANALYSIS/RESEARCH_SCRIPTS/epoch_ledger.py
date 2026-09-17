"""All available unique archived epoch records, with no selected-epoch filtering."""
from bootstrap import *
import hashlib,collections
R=Path('\\\\?\\'+str(R.resolve()))
seen={};records=collections.defaultdict(list);lineage=[]
for n in range(1,5):
 for arch in sorted((R/f'PROMPT_{n}/ARCHITECTURES').iterdir()):
  if not arch.is_dir():continue
  for path in sorted((arch/'RESULTS').rglob('*')):
   if path.suffix not in ['.json','.jsonl'] or 'batch' in path.name or 'summary' in path.name or 'complete' in path.name:continue
   try:
    raw=path.read_text(encoding='utf8');obj=[json.loads(x) for x in raw.splitlines() if x.strip()] if path.suffix=='.jsonl' else json.loads(raw)
   except (ValueError,OSError):continue
   if not isinstance(obj,list) or not obj or not isinstance(obj[0],dict) or 'epoch' not in obj[0]:continue
   filehash=sha(path)
   for entry in obj:
    if not isinstance(entry,dict) or 'epoch' not in entry:continue
    canonical=json.dumps(entry,sort_keys=True,separators=(',',':'));h=hashlib.sha256(canonical.encode()).hexdigest();source=str(path.relative_to(R)).replace('\\','/');duplicate=h in seen
    lineage.append({'prompt':n,'architecture':arch.name,'path':source,'file_sha256':filehash,'epoch':entry['epoch'],'record_sha256':h,'status':'EXACT_DUPLICATE' if duplicate else 'RETAINED','first_path':seen.get(h,source)})
    if duplicate:continue
    seen[h]=source;records[n].append((arch.name,source,entry,h))
def short(v):
 if v is None:return 'NOT RECORDED IN ORIGINAL RUN'
 if isinstance(v,float):return f'{v:.8g}'
 return str(v)
for n,rr in records.items():
 out=[f'# Exploration {n}: complete available epoch ledger\n\nEvery unique archived epoch record is included below, including failed and excluded runs where preserved. Exact duplicate record copies are mapped in EPOCH_RECORD_LINEAGE.csv. No best-epoch filtering is performed. The raw source link preserves full ROI details and batch files remain in the same run directory. A missing field means NOT RECORDED IN ORIGINAL RUN; it never means zero.\n\n']
 for arch,source,e,h in rr:
  train=e.get('train',{});val=e.get('val',e.get('validation',{}));puma=val.get('puma',val.get('v17',{})) if isinstance(val,dict) else {};roi=puma.get('fixed10',{}).get('macro_f1')
  out.append(f'## {arch} / epoch {e["epoch"]}\n\nSource: [{source}]({source}); record SHA-256 `{h}`.\n\n')
  out.append('| Objective | Train semantic F1 | Validation semantic F1 | V17 ROI F1 | Validation NLL | ECE15 |\n|---|---|---|---|---|---|\n| '+' | '.join(short(x) for x in [e.get('objective'),train.get('macro_f1'),val.get('macro_f1'),roi,val.get('nll'),val.get('ece15')])+' |\n\n')
  # Preserve every non-ROI-list recorded field, including all class metrics and operational diagnostics.
  def compact(x):
   if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ['roi_metrics','ROI_exposure','bucket_exposure']}
   if isinstance(x,list):return [compact(v) for v in x]
   return x
  out.append('<details>\n<summary>All recorded class, confusion, calibration, gradient and training fields</summary>\n\n```json\n'+json.dumps(compact(e),ensure_ascii=False,separators=(',',':'))+'\n```\n\nPer-ROI metrics and ROI/bucket exposure, when recorded, remain in the linked raw epoch file; they are not dropped from the archive.\n\n</details>\n\n')
 dest=R/f'PROMPT_{n}/TABLES/COMPLETE_EPOCH_LEDGER.md';dest.write_text(''.join(out),encoding='utf8');print(n,len(rr),'epochs',dest.stat().st_size,flush=True)
csvout(R/'EPOCH_RECORD_LINEAGE.csv',lineage);dump(R/'EPOCH_LEDGER_AUDIT.json',{'unique_records':{n:len(r) for n,r in records.items()},'duplicate_records':sum(r['status']=='EXACT_DUPLICATE' for r in lineage),'scope':'all list/JSONL epoch records in organized architecture results; raw sources retained','missing_field_policy':'NOT RECORDED IN ORIGINAL RUN'})
