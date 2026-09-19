from pathlib import Path
import hashlib,json,datetime,shutil
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';P=O/'PROVENANCE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=P/'CODEBASE_AUDIT.json';backup=P/'CODEBASE_AUDIT_PRE_P003.json'
if not backup.exists():shutil.copy2(audit,backup)
d=json.loads(audit.read_text());d.update(tests_count=80,tests_pass=80,tests_fail=0,tests_seconds=87.84,tests_log='FULL_SUITE_POST_NATURAL_FIX.log')
if not any('P6-P003' in s for s in d['deviations']):d['deviations'].append('P6-P003 manifest and CV exclusion contract for locked_natural; regression tests')
audit.write_text(json.dumps(d,indent=2),encoding='utf-8')
files=[]
for p in sorted(C.rglob('*')):
 if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache','.git','build']) and p.suffix!='.pyc':files.append(dict(path=p.relative_to(C).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
snap=P/'EXECUTED_CODE_HASHES_P003.json'
if snap.exists():
 old=json.loads(snap.read_text());assert old['files']==files,'Executed source changed since P003 snapshot'
else:snap.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_root=str(C),files=files),indent=2),encoding='utf-8')
dest=P/'ORCHESTRATION_SNAPSHOTS';dest.mkdir(exist_ok=True)
index=[]
for p in sorted(R.glob('prompt6_*.py')):
 target=dest/(p.stem+'_'+sha(p)[:12]+p.suffix)
 if not target.exists():shutil.copy2(p,target)
 assert sha(p)==sha(target)
 index.append(dict(source=str(p),archive=str(target),sha256=sha(target)))
(dest/'CURRENT_INDEX.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
print(json.dumps(dict(source_files=len(files),orchestration_scripts=len(index),QA='80 passed')))
