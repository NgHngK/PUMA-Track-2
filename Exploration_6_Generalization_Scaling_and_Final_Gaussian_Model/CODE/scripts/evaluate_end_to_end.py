#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from pathlib import Path
from puma_exploration6.end_to_end import evaluate_prediction_csv

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--gt-dir',required=True);p.add_argument('--predictions',required=True);p.add_argument('--roi-list');p.add_argument('--out',required=True);p.add_argument('--trace',action='store_true');a=p.parse_args()
    rois=None
    if a.roi_list:rois=[x.strip() for x in Path(a.roi_list).read_text(encoding='utf-8').splitlines() if x.strip()]
    out=Path(a.out)
    if out.exists():raise FileExistsError(f'evaluation output exists: {out}')
    r=evaluate_prediction_csv(a.gt_dir,a.predictions,rois,a.trace);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({'fixed10_macro_f1':r['fixed10']['macro_f1'],'rois':len(r['requested_rois'])},separators=(',',':')))

if __name__=='__main__':
    main()
