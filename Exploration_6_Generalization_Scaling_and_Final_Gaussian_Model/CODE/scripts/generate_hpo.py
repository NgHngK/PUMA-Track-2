#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from puma_exploration6.hpo import VALID_HPO_STAGES,write_hpo_configs

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--stage',choices=VALID_HPO_STAGES,required=True);p.add_argument('--trials',type=int,default=24);p.add_argument('--seed',type=int,default=1706);a=p.parse_args()
    configs=write_hpo_configs(a.base,a.out_dir,a.trials,a.stage,a.seed);print(json.dumps({'generated':len(configs),'stage':a.stage,'out_dir':a.out_dir},separators=(',',':')))

if __name__=='__main__':
    main()
