#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from collections import Counter,defaultdict
import numpy as np
from puma_exploration6.manifest import read_manifest,assert_disjoint
from puma_exploration6.tier_a import TierANormalizer

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tier-a');a=p.parse_args();rows=read_manifest(a.manifest);assert_disjoint(rows)
    by=defaultdict(Counter)
    for r in rows:by[r['split']][r['class_name']]+=1
    report={'rows':len(rows),'uids_unique':len({r['uid'] for r in rows})==len(rows),'groups_by_split':{s:len({r['group'] for r in rows if r['split']==s}) for s in by},'class_counts':{s:dict(c) for s,c in by.items()}}
    if a.tier_a:
        x=np.load(a.tier_a,mmap_mode='r');assert len(x)==len(rows);tr=np.array([r['split']=='train' for r in rows]);norm=TierANormalizer().fit(np.asarray(x[tr]));z=norm.transform(np.asarray(x));report['tier_a']={'shape':list(x.shape),'finite':bool(np.isfinite(z).all()),'train_standardized_mean_abs_max':float(np.abs(z[tr].mean(0)).max()),'train_standardized_std_error_max':float(np.abs(z[tr].std(0)-1).max())}
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
