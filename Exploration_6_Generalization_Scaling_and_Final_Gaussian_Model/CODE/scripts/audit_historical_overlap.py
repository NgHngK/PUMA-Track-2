#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from pathlib import Path
from puma_exploration6.manifest import read_manifest

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--exploration6-manifest',required=True);p.add_argument('--prior-manifest',action='append',default=[]);p.add_argument('--out');a=p.parse_args()
    cur=read_manifest(a.prompt6_manifest,require_files=False);prior=[]
    for x in a.prior_manifest:prior.extend(read_manifest(x,require_files=False))
    pu={r['uid'] for r in prior};pg={r['group'] for r in prior};pr={r['roi'] for r in prior};report={}
    for split in ['train','dev','locked']:
        rs=[r for r in cur if r['split']==split];report[split]={'n':len(rs),'uid_overlap':sum(r['uid'] in pu for r in rs),'group_overlap':len({r['group'] for r in rs}&pg),'roi_overlap':len({r['roi'] for r in rs}&pr),'unique_groups':len({r['group'] for r in rs}),'unique_rois':len({r['roi'] for r in rs})}
    report['locked_group_independence_from_prior']=report.get('locked',{}).get('group_overlap',0)==0
    text=json.dumps(report,indent=2);print(text)
    if a.out:
        out=Path(a.out)
        if out.exists():raise FileExistsError(f'audit output exists: {out}')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')

if __name__=='__main__':
    main()
