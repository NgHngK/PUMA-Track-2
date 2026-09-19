#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from pathlib import Path
import numpy as np
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import evaluate_checkpoint

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',default='locked');p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);pred=out.with_suffix('.npz')
    if out.exists() or pred.exists():raise FileExistsError(f'evaluation output already exists: {out} / {pred}')
    cfg=ExperimentConfig.load(a.config);r=evaluate_checkpoint(cfg,a.checkpoint,a.split);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pred,labels=r.pop('labels'),logits=r.pop('logits'),indices=r.pop('indices'),uids=np.array(r.pop('uids')))
    out.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({'split':a.split,'metrics_file':str(out),'predictions':str(pred)},separators=(',',':')))

if __name__=='__main__':
    main()
