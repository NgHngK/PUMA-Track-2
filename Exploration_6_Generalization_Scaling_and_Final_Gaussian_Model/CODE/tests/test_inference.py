from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import run_experiment
from puma_exploration6.inference import predict_checkpoint_on_manifest
from puma_exploration6.manifest import read_manifest, write_manifest, row_identity_sha256
from puma_exploration6.utils import sha256_file
from test_experiment_smoke import make_artifacts, CLASSES


def _target(tmp_path, cfg):
    rows=[]
    for c in range(10):
        rows.append({'uid':f'p:{c}','roi':f'roi{c}','group':f'g{c}','image':f'/missing/p{c}.tif','x':50.0,'y':50.0,
                     'label':-1,'class_name':'','split':'predict','coordinate_source':'stage1_frozen'})
    m=tmp_path/'predict.csv';write_manifest(m,rows)
    rr=read_manifest(m,require_files=False);rng=np.random.default_rng(99)
    f=rng.normal(size=(10,1536)).astype(np.float32);b=rng.normal(size=(10,16)).astype(np.float32)
    fp=tmp_path/'pf.npy';bp=tmp_path/'pb.npy';np.save(fp,f);np.save(bp,b)
    identity=row_identity_sha256(rr);mh=sha256_file(m)
    fp.with_suffix('.json').write_text(json.dumps({'kind':'cls','shape':list(f.shape),'dtype':'float32','finite_verified':True,'manifest_sha256':mh,'row_identity_sha256':identity}))
    bp.with_suffix('.json').write_text(json.dumps({'shape':list(b.shape),'dtype':'float32','finite_verified':True,'manifest_sha256':mh,'row_identity_sha256':identity}))
    return m,fp,bp


def test_predict_new_unlabeled_cached_manifest_uses_checkpoint_normalizer(tmp_path):
    cfg=make_artifacts(tmp_path);run_experiment(cfg);ck=tmp_path/'out/smoke/checkpoints/best.pt'
    m,fp,bp=_target(tmp_path,cfg)
    out=predict_checkpoint_on_manifest(ck,m,target_tier_a=bp,target_representation=fp,device='cpu')
    assert len(out)==10 and len({(r['roi'],r['uid']) for r in out})==10
    assert all(0<=r['class_id']<10 and r['class_name'] in CLASSES for r in out)
    assert all(abs(sum(r[f'p_{c}'] for c in CLASSES)-1)<1e-5 for r in out)


def test_predict_rejects_labeled_predict_rows(tmp_path):
    cfg=make_artifacts(tmp_path);run_experiment(cfg);ck=tmp_path/'out/smoke/checkpoints/best.pt'
    m,fp,bp=_target(tmp_path,cfg)
    rows=read_manifest(m,require_files=False);rows[0]['label']=0;rows[0]['class_name']='tumor';write_manifest(tmp_path/'bad.csv',rows)
    # cache sidecars intentionally absent for bad manifest; label guard occurs first.
    with pytest.raises(ValueError,match='label=-1'):
        predict_checkpoint_on_manifest(ck,tmp_path/'bad.csv',target_tier_a=bp,target_representation=fp,device='cpu')
