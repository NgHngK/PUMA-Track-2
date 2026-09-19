#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()

import argparse,csv,json
from pathlib import Path
from puma_exploration6.inference import predict_checkpoint_on_manifest


def main() -> None:
    p=argparse.ArgumentParser(description='Inference-only Exploration-6 checkpoint prediction on a new proposal manifest')
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--tier-a');p.add_argument('--representation');p.add_argument('--uni2-weights');p.add_argument('--split',default='predict')
    p.add_argument('--device');p.add_argument('--num-workers',type=int)
    a=p.parse_args();out=Path(a.out)
    if out.suffix.lower()!='.csv':raise ValueError('--out must end in .csv')
    if out.exists():raise FileExistsError(f'prediction output exists: {out}')
    rows=predict_checkpoint_on_manifest(a.checkpoint,a.manifest,target_tier_a=a.tier_a,target_representation=a.representation,
        uni2_weights=a.uni2_weights,split=a.split,device=a.device,num_workers=a.num_workers)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'rows':len(rows),'out':str(out)},separators=(',',':')))

if __name__=='__main__':main()
