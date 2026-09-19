#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import run_experiment

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=ExperimentConfig.load(a.config);summary=run_experiment(cfg)
    print(json.dumps({'experiment_id':summary['experiment_id'],'best_score':summary['best_score'],'best_epoch':summary['selected_epoch'],'summary':str(cfg.output_dir)},separators=(',',':')))

if __name__=='__main__':
    main()
