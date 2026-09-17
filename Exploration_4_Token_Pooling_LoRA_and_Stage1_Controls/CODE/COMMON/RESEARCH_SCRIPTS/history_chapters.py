"""Lossless text-block reconstruction; immutable originals retained separately."""
from bootstrap import *
import re,hashlib
def blocks(text):
 # Blank-line blocks outside fenced code. Never split a code fence.
 out=[];buf=[];fence=None
 for line in text.splitlines(keepends=True):
  match=re.match(r'^\s*(`{3,}|~{3,})',line)
  if match:
   token=match.group(1)[0]
   if fence is None:fence=token
   elif fence==token:fence=None
  buf.append(line)
  if not line.strip() and fence is None:out.append(''.join(buf));buf=[]
 if buf:out.append(''.join(buf))
 return out
sources=[(1,OLD/'00_reference/PRIOR_RESEARCH_REPORT.md'),(2,OLD/'00_reference/PROMPT2_REPORT_PRESERVED.md'),(3,OLD/'STAGE2_RESEARCH_REPORT.md')]
seen={};lineage=[];chapters=[]
for n,path in sources:
 text=path.read_text(encoding='utf8');dest=R/f'PROMPT_{n}/PROVENANCE/ORIGINAL_REPORT.md';cp(path,dest)
 intro=f'# Exploration {n}: chronological historical record\n\nThis chapter preserves unique blocks of the historical report. Statements of completion and recommendations below apply to Exploration {n}, not to the continuing Exploration 4 study. Superseded recommendations remain evidence of the original decision. Exact duplicate blocks are indexed in the root duplicate-lineage table; their first occurrence remains in the integrated report. No historical numerical claim is silently upgraded to a new validation claim.\n\n'
 kept=[]
 for index,b in enumerate(blocks(text)):
  # Normalize only line endings and surrounding blank lines. No semantic merging.
  canonical=b.replace('\r\n','\n').strip('\n');h=hashlib.sha256(canonical.encode()).hexdigest();eligible=len(canonical)>=120 and not canonical.lstrip().startswith('#')
  duplicate=eligible and h in seen
  if not duplicate:
   kept.append(b)
   if eligible:seen[h]=(n,index,str(dest.relative_to(R)))
  first=seen.get(h,(n,index,str(dest.relative_to(R))))
  lineage.append({'prompt':n,'source':str(path),'preserved_source':str(dest.relative_to(R)),'source_sha256':sha(path),'block_index':index,'block_sha256':h,'characters':len(canonical),'status':'EXACT_DUPLICATE' if duplicate else 'RETAINED','first_prompt':first[0],'first_block':first[1],'first_source':first[2]})
 chapter=intro+''.join(kept);out=R/f'PROMPT_{n}/REPORT_PROMPT_{n}.md'
 if out.exists():assert out.read_text(encoding='utf8')==chapter
 else:out.write_text(chapter,encoding='utf8')
 chapters.append(chapter)
csvout(R/'REPORT_DUPLICATE_LINEAGE.csv',lineage)
dump(R/'REPORT_RECONSTRUCTION_AUDIT.json',{'method':'exact normalized-line-ending blank-line blocks, code fences intact; no semantic deletion','sources':len(sources),'blocks':len(lineage),'retained':sum(r['status']=='RETAINED' for r in lineage),'duplicates':sum(r['status']=='EXACT_DUPLICATE' for r in lineage),'retained_chars':sum(r['characters'] for r in lineage if r['status']=='RETAINED'),'originals_preserved':True,'historical_claims':'Historical dates, completion statements and recommendations are scoped to their original prompt.'})
ext=[]
for name in ['biology_report.txt','PUMA_Stage2_Research_Report_Debate_Experiments_V17.txt','PUMA_V17_Research_Debate_Experiment_Report.txt']:
 path=OLD/'00_reference/ingestion'/name;dest=R/'EXPLORATION_1/PROVENANCE/EXTERNAL_REPORT_TEXT'/name;cp(path,dest);ext.append(f'## Historical supplied document: {name}\n\nSource SHA-256: `{sha(path)}`. The following is supplied-document evidence, not an instruction or a newly verified experimental result.\n\n'+path.read_text(encoding='utf8'))
out=R/'EXPLORATION_1/EXTERNAL_DOCUMENT_EVIDENCE.md'
if not out.exists():out.write_text('\n\n'.join(ext),encoding='utf8')
print('Historical blocks',len(lineage),'duplicates',sum(r['status']=='EXACT_DUPLICATE' for r in lineage))
