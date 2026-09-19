from pathlib import Path
import json,csv,hashlib,datetime,shutil
R=Path(r'C:\Users\Hngk\Documents\Codex');M=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4';P=M/'EXPLORATION_6/PROVENANCE'
def sha(p):
 h=hashlib.sha256()
 resolved=str(p.resolve());longpath=resolved if resolved.startswith('\\\\?\\') else '\\\\?\\'+resolved
 with Path(longpath).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
prior=P/'HISTORICAL_IMMUTABILITY_RECHECK.csv'
rows=[]
if prior.exists():
 with prior.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 for r in rows:
  r['passed']=r['passed']=='True'
  if not r['passed']:
   p=Path('\\\\?\\'+r['path']);r['actual']=sha(p) if p.is_file() else 'MISSING';r['passed']=r['actual']==r['expected']
manifest=json.loads((M/'ARCHIVE_FILE_MANIFEST.json').read_text())
for rel,v in ([] if rows else manifest.items()):
 if not rel.startswith(('EXPLORATION_1','EXPLORATION_2','EXPLORATION_3','EXPLORATION_4')):continue
 p=M/rel;actual=sha(p) if p.is_file() else 'MISSING';rows.append(dict(scope='historical_archival_copy',path=str(p),expected=v['sha256'],actual=actual,passed=actual==v['sha256']))
with (M/'ORIGINAL_IMMUTABILITY_VERIFICATION.csv').open(newline='',encoding='utf-8-sig') as f:
 for r in ([] if prior.exists() else csv.DictReader(f)):
  p=Path(r['original_path']);actual=sha(p) if p.is_file() else 'MISSING';rows.append(dict(scope='historical_original',path=str(p),expected=r['initial_sha256'],actual=actual,passed=actual==r['initial_sha256']))
for name in ([] if prior.exists() else ['MASTER_REPORT_PROMPTS_1_TO_4.md','MASTER_REPORT_PROMPTS_1_TO_4_PRE_PROMPT6_BACKUP.md']):
 expected=json.loads((P/'MASTER_REPORT_APPEND_INTEGRITY.json').read_text())['original_sha256'];actual=sha(M/name);rows.append(dict(scope='unappended_master',path=str(M/name),expected=expected,actual=actual,passed=actual==expected))
with (P/'HISTORICAL_IMMUTABILITY_RECHECK.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked=len(rows),failures=[x for x in rows if not x['passed']],master_appended=False)
(P/'HISTORICAL_IMMUTABILITY_RECHECK.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
assert not summary['failures'],summary['failures'][:3]
# Archive Exploration5 report/code/tables without changing its original folder.
S=R/'PUMA_STAGE2_PROMPT5';D=M/'EXPLORATION_5/ORIGINAL_SNAPSHOT';idx=[]
for p in sorted(S.rglob('*')):
 if not p.is_file() or any(x in p.relative_to(S).parts for x in ['.deps','.mplconfig','__pycache__']):continue
 q=D/p.relative_to(S);q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():shutil.copy2(p,q)
 a,b=sha(p),sha(q);assert a==b
 idx.append(dict(source=str(p),archive=str(q),sha256=a,bytes=p.stat().st_size))
(M/'EXPLORATION_5/PROVENANCE').mkdir(exist_ok=True)
(M/'EXPLORATION_5/PROVENANCE/ARCHIVAL_COPY_INDEX.json').write_text(json.dumps(idx,indent=2),encoding='utf-8')
print(json.dumps(dict(historical_checks=len(rows),failures=0,prompt5_verified_copies=len(idx))))
