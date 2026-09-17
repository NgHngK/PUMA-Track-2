from pathlib import Path
import json,hashlib,shutil,csv
W=Path(__file__).resolve().parent
R=W.parents[3]
BASE=R
OLD=R/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf8')
def cp(s,d):
 d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists():assert sha(s)==sha(d)
 else:shutil.copy2(s,d)
def csvout(p,rs):
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
