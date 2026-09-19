#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from puma_exploration6.full_manifest import build_full_gt_manifest

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--group-csv');a=p.parse_args()
    print(json.dumps(build_full_gt_manifest(a.root,a.out,a.group_csv),indent=2))

if __name__=='__main__':
    main()
