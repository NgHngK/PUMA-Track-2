import json
import numpy as np
import pytest
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import run_experiment
from puma_exploration6.manifest import write_manifest
from puma_exploration6.utils import sha256_file

CLASSES=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']

def test_cache_manifest_mismatch_fails_closed(tmp_path):
    rows=[]
    for split in ['train','dev']:
        for c in range(10):rows.append({'uid':f'{split}{c}','roi':f'{split}r{c}','group':f'{split}g{c}','image':f'/x/{split}{c}.tif','x':20+c*20,'y':20,'eval_x':20+c*20,'eval_y':20,'label':c,'class_name':CLASSES[c],'split':split,'coordinate_source':'gt'})
    m=tmp_path/'m.csv';write_manifest(m,rows);np.save(tmp_path/'f.npy',np.zeros((20,1536),dtype='float32'));np.save(tmp_path/'b.npy',np.zeros((20,16),dtype='float32'))
    (tmp_path/'f.json').write_text(json.dumps({'manifest_sha256':'bad'}));(tmp_path/'b.json').write_text(json.dumps({'manifest_sha256':sha256_file(m)}))
    d={'experiment_id':'bad','manifest':str(m),'output_dir':str(tmp_path/'o'),'cached_features':str(tmp_path/'f.npy'),'tier_a':str(tmp_path/'b.npy'),'device':'cpu','sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'cls'},'optimizer':{'max_epochs':1,'patience':1}}
    with pytest.raises(ValueError,match='cache/manifest hash mismatch'):run_experiment(ExperimentConfig.from_dict(d))

def test_cached_dataset_lazy_index_map_preserves_values_without_subset_copy(tmp_path):
    import numpy as np, torch
    from puma_exploration6.datasets import CachedNucleiDataset
    from puma_exploration6.constants import CLASSES
    path=tmp_path/'rep.npy'
    base=np.arange(6*1536,dtype=np.float32).reshape(6,1536)
    np.save(path,base)
    mm=np.load(path,mmap_mode='r')
    rows=[{'uid':f'u{i}','roi':'r','group':'g','image':'x','x':8.0,'y':8.0,'label':i%10,'class_name':CLASSES[i%10],'split':'train','coordinate_source':'gt'} for i in range(3)]
    ix=np.array([5,2,4])
    ds=CachedNucleiDataset(rows,mm,None,96,representation_indices=ix)
    assert isinstance(ds.rep,np.memmap)
    assert np.array_equal(ds.rep_indices,ix)
    for local,global_i in enumerate(ix):
        assert torch.equal(ds[local]['features'],torch.tensor(base[global_i]))
        assert int(ds[local]['index'])==local


def test_cached_memmap_pickles_by_path_not_payload(tmp_path):
    import pickle
    import numpy as np
    from puma_exploration6.datasets import CachedNucleiDataset
    rows=[{'uid':'u0','roi':'r0','group':'g0','image':'missing.tif','x':48.0,'y':48.0,'label':0,'class_name':'tumor','split':'train','coordinate_source':'gt'}]
    path=tmp_path/'hugeish.npy'
    arr=np.zeros((64,265,1536),dtype=np.float32)
    arr[0,9,0]=3.25
    np.save(path,arr)
    mm=np.load(path,mmap_mode='r')
    ds=CachedNucleiDataset(rows,mm,None,representation_indices=np.array([0]),validate_finite=False)
    payload=pickle.dumps(ds,protocol=5)
    # A correctly path-backed dataset should not serialize the ~100 MB cache.
    assert len(payload) < 100_000
    restored=pickle.loads(payload)
    assert isinstance(restored.rep,np.memmap)
    assert float(restored[0]['tokens'][9,0]) == 3.25
