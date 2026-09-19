import csv
from pathlib import Path
import numpy as np
from puma_exploration6.manifest import write_manifest,read_manifest
from puma_exploration6.x3_builder import build_x3


def _rows(groups=60,per_class=30):
    rows=[]
    for g in range(groups):
        for c in range(10):
            for j in range(per_class):
                rows.append({'uid':f'g{g}:c{c}:n{j}','roi':f'roi{g}','group':f'g{g}','image':f'/tmp/fake_{g}.tif','x':50.0,'y':50.0,'eval_x':50.0,'eval_y':50.0,
                             'label':c,'class_name':['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'][c],
                             'split':'train','coordinate_source':'gt'})
    return rows


def test_x3_exact_counts_and_nested(tmp_path):
    result=build_x3(_rows(),tmp_path,17,100)
    assert result['selected_total']==1350
    d300=read_manifest(tmp_path/'D300.csv',require_files=False);d600=read_manifest(tmp_path/'D600.csv',require_files=False);d900=read_manifest(tmp_path/'D900.csv',require_files=False)
    for rows,n in [(d300,300),(d600,600),(d900,900)]:
        tr=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];locked=[r for r in rows if r['split']=='locked']
        assert len(tr)==n and len(dev)==225 and len(locked)==225
        assert not ({r['group'] for r in tr}&{r['group'] for r in dev})
        assert not ({r['group'] for r in tr}&{r['group'] for r in locked})
        assert not ({r['group'] for r in dev}&{r['group'] for r in locked})
    from collections import Counter
    assert [Counter(r['label'] for r in d300 if r['split']=='train')[c] for c in range(10)] == [71,37]+[24]*8
    assert [Counter(r['label'] for r in d600 if r['split']=='train')[c] for c in range(10)] == [142,74]+[48]*8
    assert [Counter(r['label'] for r in d900 if r['split']=='train')[c] for c in range(10)] == [213,111]+[72]*8
    assert [Counter(r['label'] for r in d900 if r['split']=='dev')[c] for c in range(10)] == [58,23]+[18]*8
    assert [Counter(r['label'] for r in d900 if r['split']=='locked')[c] for c in range(10)] == [59,22]+[18]*8
    u300={r['uid'] for r in d300 if r['split']=='train'};u600={r['uid'] for r in d600 if r['split']=='train'};u900={r['uid'] for r in d900 if r['split']=='train'}
    assert u300<u600<u900

def test_x3_output_is_immutable(tmp_path):
    from puma_exploration6.x3_builder import build_x3
    # Reuse helper logic from this module's synthetic dataset if available by
    # constructing a generous class-balanced grouped source.
    rows=[]
    for g in range(60):
        for c in range(10):
            for j in range(3):
                rows.append({'uid':f'{g}:{c}:{j}','roi':f'r{g}','group':f'g{g}','image':f'/tmp/r{g}.tif','x':10.0,'y':10.0,'label':c,'class_name':__import__('puma_exploration6.constants',fromlist=['CLASSES']).CLASSES[c],'split':'train','coordinate_source':'gt'})
    out=tmp_path/'x3';out.mkdir();(out/'existing.txt').write_text('preserve')
    import pytest
    with pytest.raises(FileExistsError):
        build_x3(rows,out,seed=17,trials=5)


def test_x3_rejects_already_partitioned_source(tmp_path):
    import pytest
    from puma_exploration6.x3_builder import build_x3
    # The guard is checked before feasibility, so a minimal contaminated source is sufficient.
    rows=[
        {'uid':'a','roi':'ra','group':'ga','image':'a.tif','x':1.0,'y':1.0,'label':0,'class_name':'tumor','split':'train','coordinate_source':'gt'},
        {'uid':'b','roi':'rb','group':'gb','image':'b.tif','x':1.0,'y':1.0,'label':1,'class_name':'lymphocyte','split':'dev','coordinate_source':'gt'},
    ]
    with pytest.raises(ValueError,match='unpartitioned all-train pool'):
        build_x3(rows,tmp_path/'x3')
