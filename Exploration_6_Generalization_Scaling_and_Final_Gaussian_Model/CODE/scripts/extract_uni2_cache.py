#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from puma_exploration6.config import AugmentationConfig
from puma_exploration6.constants import EMBED_DIM,UNI2_TOKEN_COUNT
from puma_exploration6.datasets import RawNucleiDataset
from puma_exploration6.manifest import read_manifest,row_identity_sha256
from puma_exploration6.models.uni2 import UNI2Backbone,checkpoint_sha256
from puma_exploration6.utils import sha256_file

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True);p.add_argument('--tier-a');p.add_argument('--kind',choices=['cls','tokens'],default='tokens');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.batch_size<1:raise ValueError('--batch-size must be positive')
    out=Path(a.out)
    if out.suffix.lower()!='.npy':raise ValueError('--out must end in .npy')
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(f'cache output already exists; preserve it and choose a new path: {out}')
    rows=read_manifest(a.manifest);tier=np.load(a.tier_a,mmap_mode='r') if a.tier_a else None;ds=RawNucleiDataset(rows,tier,96,AugmentationConfig());dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0)
    device=torch.device(a.device if (not a.device.startswith('cuda') or torch.cuda.is_available()) else 'cpu');enc=UNI2Backbone(a.weights,'frozen').to(device)
    shape=(len(rows),EMBED_DIM) if a.kind=='cls' else (len(rows),UNI2_TOKEN_COUNT,EMBED_DIM);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.partial.npy')
    if tmp.exists():raise FileExistsError(f'partial cache exists and is preserved for audit: {tmp}')
    mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape);pos=0
    try:
        with torch.inference_mode():
            for batch in dl:
                t=enc.forward_tokens(batch['image'].to(device));arr=(t[:,0] if a.kind=='cls' else t).float().cpu().numpy()
                if not np.isfinite(arr).all():raise FloatingPointError('nonfinite UNI2 cache batch')
                mm[pos:pos+len(arr)]=arr;pos+=len(arr)
        if pos!=len(rows):raise RuntimeError(f'cache row count mismatch: wrote {pos}, expected {len(rows)}')
        mm.flush();del mm;os.replace(tmp,out)
    except Exception:
        try:
            mm.flush()
        except Exception as flush_error:
            print(json.dumps({'warning':'failed to flush partial cache after extraction error','error':str(flush_error)},separators=(',',':')),file=__import__('sys').stderr)
        raise
    meta={'kind':a.kind,'shape':list(shape),'dtype':'float32','finite_verified':True,'rows_written':pos,'manifest_sha256':sha256_file(a.manifest),'row_identity_sha256':row_identity_sha256(rows),'weights_sha256':checkpoint_sha256(a.weights)};out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,separators=(',',':')))

if __name__=='__main__':
    main()
