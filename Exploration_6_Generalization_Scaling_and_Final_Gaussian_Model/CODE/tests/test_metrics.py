import numpy as np
from puma_exploration6.metrics import semantic_metrics,puma_subset_metrics,CANONICAL_TO_V17


def test_semantic_perfect():
    y=np.arange(10);z=np.full((10,10),-10,dtype=np.float32);z[np.arange(10),y]=10;m=semantic_metrics(y,z);assert m['macro_f1']==1.0 and m['accuracy']==1.0


def test_v17_perfect_subset():
    classes=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'];rows=[]
    for c in range(10):
        x=20.0+20.0*(c%5);y=20.0+20.0*(c//5)
        rows.append({'uid':f'u{c}','roi':'r0','group':'g0','image':'x','x':x,'y':y,'eval_x':x,'eval_y':y,'label':c,'class_name':classes[c],'split':'dev','coordinate_source':'gt'})
    p=np.eye(10);m=puma_subset_metrics(rows,np.arange(10),p);assert abs(m['fixed10']['macro_f1']-1)<1e-12


def test_mapping_is_permutation():
    assert sorted(CANONICAL_TO_V17.tolist())==list(range(10))


def test_semantic_metrics_reject_nonfinite_logits_and_bad_labels():
    import pytest
    y=np.arange(10);z=np.eye(10,dtype=np.float32);z[0,0]=np.nan
    with pytest.raises(FloatingPointError): semantic_metrics(y,z)
    with pytest.raises(ValueError): semantic_metrics(np.array([10]),np.zeros((1,10),dtype=np.float32))
