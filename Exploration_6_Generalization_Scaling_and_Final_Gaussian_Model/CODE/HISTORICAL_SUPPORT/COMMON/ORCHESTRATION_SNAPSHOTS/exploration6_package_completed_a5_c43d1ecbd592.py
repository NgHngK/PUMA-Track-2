from pathlib import Path
import json,hashlib,shutil,csv,datetime
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';A=O/'ARCHITECTURES/A5_FROZEN_CLS_RANK8';A.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
index=[]
def copy(p,q,kind,run=''):
 q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():shutil.copy2(p,q)
 a,b=sha(p),sha(q);assert a==b,f'Archive differs: {q}'
 index.append(dict(source=str(p),archive=str(q),sha256=a,bytes=p.stat().st_size,kind=kind,run_id=run))
for folder in ['src','scripts','tests','configs','docs','vendor']:
 for p in sorted((C/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p,A/'CODE'/p.relative_to(C),'code')
for name in ['requirements.txt','pyproject.toml','README.md','QA_CERTIFICATE.md','SHA256SUMS.txt']:
 if (C/name).exists():copy(C/name,A/'CODE'/name,'code_provenance')
runs=[];references={}
for root in sorted((O/'EXPERIMENTS').iterdir()):
 if not (root/'summary.json').exists():continue
 cfg=json.loads((root/'config.json').read_text());model=cfg['model']
 if model['representation']!='cls' or model['interaction_rank']!=8 or not model['use_tier_a']:continue
 runs.append(root.name)
 for p in sorted(root.rglob('*')):
  if p.is_file():copy(p,A/'RESULTS/RUNS'/root.name/p.relative_to(root),'completed_run',root.name)
 for key in ['manifest','cached_features','cached_tokens','tier_a','uni2_weights']:
  if not cfg.get(key):continue
  p=Path(cfg[key]);references[str(p)]=dict(kind=key,path=str(p),sha256=sha(p),bytes=p.stat().st_size)
  if key=='manifest':copy(p,A/'INPUT_MANIFESTS'/p.name,'manifest')
  if p.suffix=='.npy' and p.with_suffix('.json').exists():copy(p.with_suffix('.json'),A/'CACHE_REFERENCES'/p.with_suffix('.json').name,'cache_sidecar')
(A/'CACHE_REFERENCES').mkdir(exist_ok=True)
(A/'CACHE_REFERENCES/IMMUTABLE_SHARED_INPUTS.json').write_text(json.dumps(list(references.values()),indent=2),encoding='utf-8')
with (A/'ORIGINAL_PATH_MAP.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
(A/'provenance.json').write_text(json.dumps(dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='INTERIM completed A and C artifacts; not final selection',code_source=str(C),approved_patches=['P6-P001','P6-P002','P6-P003'],run_ids=runs,all_copy_hashes_match=True,shared_cache_policy='Verified immutable references; no destructive move; raw UNI2 weights remain user supplied'),indent=2),encoding='utf-8')
print(json.dumps(dict(completed_runs=len(runs),verified_copies=len(index),shared_inputs=len(references))))
