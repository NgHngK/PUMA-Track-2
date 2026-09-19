import json, pathlib, numpy as np, pandas as pd
A=pathlib.Path(r'C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4')
def short(v):
 if isinstance(v,dict): return {k:short(x) for k,x in v.items() if k not in ['puma','v17','roi_metrics','source_hashes']}
 if isinstance(v,list): return v if len(v)<17 else str(v[:3])+f' ... [{len(v)}]'
 return v
p=A/'EXPLORATION_3/ARCHITECTURES/A5/RESULTS/RUNS/p3_A5_real_s17_cv0'
h=json.loads((p/'history.jsonl').read_text().splitlines()[0]); print('HISTORY KEYS',h.keys()); print(short({k:v for k,v in h.items() if k not in ['train','val']}))
print('VAL KEYS',h['val'].keys());print('CONFIG',short(json.loads((p/'config.json').read_text())))
for q in [A/'EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/sample_manifest.csv',A/'EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/prompt3_cv.csv']:
 d=pd.read_csv(q);print(q.name,d.shape,d.columns.tolist(),d.head(2).to_dict('records'))
for q in [p/'predictions.npz',A/'EXPLORATION_4/TARGET_POOLING/REPRESENTATIONS/complete.json',A/'EXPLORATION_4/TOKEN_AUDIT/CACHE/complete.json']:
 if q.suffix=='.npz':
  z=np.load(q,allow_pickle=False); print(str(q),[(k,z[k].shape) for k in z.files])
 else: print(q,short(json.loads(q.read_text())))
for root in ['EXPLORATION_2/TABLES','EXPLORATION_3/TABLES','EXPLORATION_4/SHARED_INPUT_SNAPSHOTS']:
 print(root,[str(p.relative_to(A)) for p in (A/root).rglob('*') if p.is_file()][:70])
