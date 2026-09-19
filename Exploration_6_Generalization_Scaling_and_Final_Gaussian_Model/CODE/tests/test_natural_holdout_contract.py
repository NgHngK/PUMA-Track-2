import pytest
from puma_exploration6.constants import CLASSES
from puma_exploration6.manifest import read_manifest, write_manifest
from puma_exploration6.cv import create_internal_group_cv_manifests


def test_natural_holdout_read_and_hpo_exclusion(tmp_path):
    rows=[dict(uid=f'{c}_{g}',roi=f'{c}_{g}',group=f'{c}_{g}',image=f'{c}_{g}.tif',
               x=1,y=1,label=c,class_name=name,split='train',coordinate_source='gt')
          for c,name in enumerate(CLASSES) for g in range(6)]
    rows.append(dict(rows[0],uid='natural',roi='natural',group='natural',image='natural.tif',split='locked_natural'))
    p=tmp_path/'all.csv';write_manifest(p,rows)
    assert read_manifest(p,require_files=False)[-1]['split']=='locked_natural'
    for f in create_internal_group_cv_manifests(p,tmp_path/'cv'):
        checked=read_manifest(f['manifest'],require_files=False)
        assert checked[-1]['split']=='predict'
        assert f['external_holdout_rows_hidden_as_predict']==1


def test_natural_holdout_still_rejects_group_leakage(tmp_path):
    r=dict(uid='a',roi='r',group='g',image='x.tif',x=1,y=1,label=0,class_name=CLASSES[0],split='train',coordinate_source='gt')
    p=tmp_path/'bad.csv';write_manifest(p,[r,dict(r,uid='b',split='locked_natural')])
    with pytest.raises(ValueError,match='leakage'):
        read_manifest(p,require_files=False)
