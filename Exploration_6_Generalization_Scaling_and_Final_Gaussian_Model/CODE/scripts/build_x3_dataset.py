#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from puma_exploration6.manifest import read_manifest
from puma_exploration6.x3_builder import build_x3

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--full-manifest',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--trials',type=int,default=1000);a=p.parse_args()
    rows=read_manifest(a.full_manifest);print(json.dumps(build_x3(rows,a.out_dir,a.seed,a.trials),indent=2))

if __name__=='__main__':
    main()
