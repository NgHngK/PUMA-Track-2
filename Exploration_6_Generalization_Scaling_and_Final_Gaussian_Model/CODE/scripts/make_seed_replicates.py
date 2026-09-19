#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,copy,json
from pathlib import Path
from puma_exploration6.config import ExperimentConfig

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seeds',default='17,29,43');a=p.parse_args()
    seeds=[int(x.strip()) for x in a.seeds.split(',') if x.strip()]
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):raise ValueError('seeds must be unique non-negative integers')
    base=ExperimentConfig.load(a.base).to_dict();out=Path(a.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'seed replicate directory is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True);written=[]
    for seed in seeds:
        d=copy.deepcopy(base);d['seed']=seed;d['experiment_id']=f"{base['experiment_id']}_s{seed}";ExperimentConfig.from_dict(d)
        path=out/f"{d['experiment_id']}.json";path.write_text(json.dumps(d,indent=2),encoding='utf-8');written.append(str(path))
    (out/'SEED_BATCH.json').write_text(json.dumps({'base':str(a.base),'seeds':seeds,'configs':written},indent=2),encoding='utf-8')
    print(json.dumps({'generated':len(written),'out_dir':str(out)},separators=(',',':')))

if __name__=='__main__':main()
