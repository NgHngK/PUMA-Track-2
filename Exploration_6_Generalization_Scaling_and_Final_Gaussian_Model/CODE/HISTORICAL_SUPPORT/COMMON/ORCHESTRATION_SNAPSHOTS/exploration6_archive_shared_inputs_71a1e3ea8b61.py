from pathlib import Path
import json,csv,hashlib,shutil,datetime
R=Path(r'C:\Users\Hngk\Documents\Codex');S=R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';D=O/'SHARED_INPUT_SNAPSHOTS';D.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
index=[]
for p in sorted(S.rglob('*')):
 if not p.is_file():continue
 q=D/p.relative_to(S);q.parent.mkdir(parents=True,exist_ok=True)
 a=sha(p)
 if not q.exists():shutil.copy2(p,q)
 b=sha(q);assert a==b,str(q)
 index.append(dict(source=str(p),archive=str(q),sha256=a,bytes=p.stat().st_size))
payload=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(S),destination=str(D),files=index,holdout_predictions_created=False,note='Physical copies of frozen input manifests and caches only. No raw holdout predictions or measurements; all source paths preserved.')
(O/'PROVENANCE/SHARED_INPUT_COPY_INDEX.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
A=O/'ARCHITECTURES/A5_FROZEN_CLS_RANK8/CACHE_REFERENCES';refs=json.loads((A/'IMMUTABLE_SHARED_INPUTS.json').read_text());mapping={r['source']:r for r in index}
for r in refs:
 if r['path'] in mapping:r['archival_path']=mapping[r['path']]['archive'];assert r['sha256']==mapping[r['path']]['sha256']
(A/'IMMUTABLE_SHARED_INPUTS.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(index),bytes=sum(x['bytes'] for x in index),verified=True)))
