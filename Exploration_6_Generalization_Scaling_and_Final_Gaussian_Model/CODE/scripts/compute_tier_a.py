#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json
from pathlib import Path
import numpy as np
from puma_exploration6.manifest import read_manifest,row_identity_sha256
from puma_exploration6.tier_a import compute_tier_a,NAMES
from puma_exploration6.utils import sha256_file

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.suffix.lower()!='.npy':raise ValueError('--out must end in .npy')
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(f'Tier-A output already exists; preserve it and choose a new path: {out}')
    rows=read_manifest(a.manifest);x=compute_tier_a(rows);out.parent.mkdir(parents=True,exist_ok=True);np.save(out,x)
    meta={'shape':list(x.shape),'dtype':'float32','finite_verified':bool(np.isfinite(x).all()),'names':list(NAMES),'manifest_sha256':sha256_file(a.manifest),'row_identity_sha256':row_identity_sha256(rows)};out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,separators=(',',':')))

if __name__=='__main__':
    main()
