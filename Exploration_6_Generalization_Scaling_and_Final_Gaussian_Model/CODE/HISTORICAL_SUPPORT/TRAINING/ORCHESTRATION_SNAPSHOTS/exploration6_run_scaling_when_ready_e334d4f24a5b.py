from pathlib import Path
import time,json,sys,os,subprocess,hashlib
import numpy as np
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';P=R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
sys.path.insert(0,str(C/'src'))
from puma_exploration6.manifest import read_manifest,row_identity_sha256
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_status(status,**kwargs):
 (O/'RESEARCH_STATE/SCALING_PIPELINE_STATUS.json').write_text(json.dumps(dict(status=status,**kwargs),indent=2),encoding='utf-8')
def call(args,log):
 with (O/'PROVENANCE'/log).open('w',encoding='utf-8') as f:
  subprocess.run([sys.executable,*map(str,args)],cwd=C,stdout=f,stderr=subprocess.STDOUT,check=True)
write_status('WAITING_FOR_FROZEN_CACHES')
needed=[P/'D900_tokens.npy',P/'D900_tokens.json',P/'D900_tierA.npy',P/'D900_tierA.json']
while not all(p.is_file() for p in needed):time.sleep(30)
write_status('VERIFYING_CACHE_IDENTITIES')
rows=read_manifest(P/'SELECTION/D900.csv');index={r['uid']:i for i,r in enumerate(rows)}
tokens=np.load(P/'D900_tokens.npy',mmap_mode='r');tier=np.load(P/'D900_tierA.npy')
tm=json.loads((P/'D900_tokens.json').read_text());bm=json.loads((P/'D900_tierA.json').read_text())
assert tokens.shape==(1350,265,1536) and tier.shape==(1350,16)
for m in [tm,bm]:assert m['row_identity_sha256']==row_identity_sha256(rows) and m['manifest_sha256']==sha(P/'SELECTION/D900.csv') and m['finite_verified']
assert tm['weights_sha256']=='32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'
batch=O/'CONFIGS_RESOLVED/A_SCALING';batch.mkdir(exist_ok=True)
for n in [300,600,900]:
 manifest=P/f'SELECTION/D{n}.csv';subset=read_manifest(manifest);ix=[index[r['uid']] for r in subset]
 arrays=[('cls',np.asarray(tokens[ix,0,:]),'cls')]
 if n!=900:arrays.append(('tierA',tier[ix],None))
 for suffix,array,kind in arrays:
  out=P/f'D{n}_{suffix}.npy'
  if out.exists():raise FileExistsError(out)
  assert np.isfinite(array).all();np.save(out,array)
  meta=dict(shape=list(array.shape),dtype=str(array.dtype),finite_verified=True,manifest_sha256=sha(manifest),row_identity_sha256=row_identity_sha256(subset),source=str(P/'D900_tokens.npy' if kind else P/'D900_tierA.npy'),source_manifest_sha256=sha(P/'SELECTION/D900.csv'),derivation='UID-aligned subset; exact CLS token0' if kind else 'UID-aligned Tier-A subset')
  if kind:meta.update(kind=kind,weights_sha256=tm['weights_sha256'])
  out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 call(['scripts/make_seed_replicates.py','--base',O/f'CONFIGS_RESOLVED/A_D{n}_base.json','--out-dir',batch/f'D{n}'],f'A_D{n}_seed_generation.log')
write_status('RUNNING_NINE_SCALING_CONTROLS')
for n in [300,600,900]:
 call(['scripts/run_batch.py','--config-dir',batch/f'D{n}','--stop-on-error'],f'A_D{n}_batch.log')
write_status('COMPLETE',runs=9,next='Aggregate all epochs and paired size effects before Group B')
print('SCALING COMPLETE: nine controls; holdouts unopened',flush=True)
