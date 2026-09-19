import numpy as np
from pathlib import Path
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import run_experiment,evaluate_checkpoint
from puma_exploration6.manifest import write_manifest
from puma_exploration6.utils import sha256_file
import json

CLASSES=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']

def test_locked_eval_uses_saved_normalizer(tmp_path):
    rows=[]
    for split,n in [('train',3),('dev',1),('locked',1)]:
        for c in range(10):
            for j in range(n):
                x=20+20*(c%5);y=20+20*(c//5)
                rows.append({'uid':f'{split}:{c}:{j}','roi':f'{split}_roi','group':f'{split}_g','image':f'/missing/{split}.tif','x':x,'y':y,'eval_x':x,'eval_y':y,'label':c,'class_name':CLASSES[c],'split':split,'coordinate_source':'gt'})
    # one group per split is valid; all rows in a split share same group/roi.
    manifest=tmp_path/'m.csv';write_manifest(manifest,rows);rng=np.random.default_rng(4);f=rng.normal(size=(len(rows),1536)).astype('float32');b=rng.normal(size=(len(rows),16)).astype('float32');np.save(tmp_path/'f.npy',f);np.save(tmp_path/'b.npy',b);mh=sha256_file(manifest);(tmp_path/'f.json').write_text(json.dumps({'manifest_sha256':mh}));(tmp_path/'b.json').write_text(json.dumps({'manifest_sha256':mh}))
    d={'experiment_id':'lock','manifest':str(manifest),'output_dir':str(tmp_path/'out'),'cached_features':str(tmp_path/'f.npy'),'tier_a':str(tmp_path/'b.npy'),'device':'cpu','selection_metric':'roi_macro_f1','sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'cls'},'optimizer':{'batch_size':10,'max_epochs':1,'patience':1,'scheduler':'constant'}}
    cfg=ExperimentConfig.from_dict(d);run_experiment(cfg);r=evaluate_checkpoint(cfg,tmp_path/'out/lock/checkpoints/best.pt','locked');assert r['metrics']['macro_f1']>=0;assert len(r['uids'])==10


def test_locked_eval_rejects_mutated_representation_artifact(tmp_path):
    import pytest
    rows=[]
    for split,n in [('train',2),('dev',1),('locked',1)]:
        for c in range(10):
            for j in range(n):
                x=20+20*(c%5);y=20+20*(c//5)
                rows.append({'uid':f'{split}:{c}:{j}','roi':f'{split}_roi','group':f'{split}_g','image':f'/missing/{split}.tif','x':x,'y':y,'eval_x':x,'eval_y':y,'label':c,'class_name':CLASSES[c],'split':split,'coordinate_source':'gt'})
    manifest=tmp_path/'m.csv';write_manifest(manifest,rows);rng=np.random.default_rng(44);f=rng.normal(size=(len(rows),1536)).astype('float32');b=rng.normal(size=(len(rows),16)).astype('float32');np.save(tmp_path/'f.npy',f);np.save(tmp_path/'b.npy',b);mh=sha256_file(manifest);(tmp_path/'f.json').write_text(json.dumps({'manifest_sha256':mh}));(tmp_path/'b.json').write_text(json.dumps({'manifest_sha256':mh}))
    d={'experiment_id':'lockmut','manifest':str(manifest),'output_dir':str(tmp_path/'out'),'cached_features':str(tmp_path/'f.npy'),'tier_a':str(tmp_path/'b.npy'),'device':'cpu','selection_metric':'roi_macro_f1','sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'cls'},'optimizer':{'batch_size':10,'max_epochs':1,'patience':1,'scheduler':'constant'}}
    cfg=ExperimentConfig.from_dict(d);run_experiment(cfg)
    f[0,0]+=1;np.save(tmp_path/'f.npy',f)
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        evaluate_checkpoint(cfg,tmp_path/'out/lockmut/checkpoints/best.pt','locked')
