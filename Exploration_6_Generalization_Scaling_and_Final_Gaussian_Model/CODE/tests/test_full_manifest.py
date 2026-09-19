from __future__ import annotations
import csv,json
import numpy as np
import pytest
from PIL import Image
from puma_exploration6.full_manifest import build_full_gt_manifest
from puma_exploration6.manifest import read_manifest

def _make_root(tmp_path):
    root=tmp_path/'root';g=root/'01_training_dataset_geojson_nuclei';i=root/'01_training_dataset_tif_ROIs';g.mkdir(parents=True);i.mkdir()
    Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)).save(i/'r1.tif')
    ring=[[10,10],[14,10],[14,14],[10,14],[10,10]]
    feat={'type':'Feature','properties':{'classification':{'name':'nuclei_tumor'}},'geometry':{'type':'Polygon','coordinates':[ring]}}
    (g/'r1_nuclei.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[feat]}))
    return root

def test_full_manifest_build_and_patient_group(tmp_path):
    root=_make_root(tmp_path);groups=tmp_path/'groups.csv'
    with groups.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['roi','patient_id']);w.writeheader();w.writerow({'roi':'r1','patient_id':'p1'})
    out=tmp_path/'full.csv';s=build_full_gt_manifest(root,out,groups);assert s=={'rows':1,'rois':1,'groups':1}
    rows=read_manifest(out);assert rows[0]['group']=='p1' and rows[0]['label']==0

def test_full_manifest_rejects_conflicting_group_mapping(tmp_path):
    root=_make_root(tmp_path);groups=tmp_path/'groups.csv'
    groups.write_text('roi,patient_id\nr1,p1\nr1,p2\n')
    with pytest.raises(ValueError,match='conflicting'):build_full_gt_manifest(root,tmp_path/'x.csv',groups)

def test_full_manifest_is_immutable(tmp_path):
    root=_make_root(tmp_path);out=tmp_path/'full.csv';out.write_text('keep')
    with pytest.raises(FileExistsError):build_full_gt_manifest(root,out)


def test_full_manifest_accepts_case_group_when_patient_unavailable(tmp_path):
    root=_make_root(tmp_path);groups=tmp_path/'groups_case.csv'
    groups.write_text('roi,case_id,slide_id\nr1,c1,s1\n')
    out=tmp_path/'full_case.csv';build_full_gt_manifest(root,out,groups)
    rows=read_manifest(out);assert rows[0]['group']=='c1'
    meta=json.loads(out.with_suffix('.json').read_text())
    assert meta['grouping']=='case_id'


def test_group_map_prefers_complete_patient_over_case(tmp_path):
    root=_make_root(tmp_path);groups=tmp_path/'groups_both.csv'
    groups.write_text('roi,patient_id,case_id\nr1,p1,c1\n')
    out=tmp_path/'full_both.csv';build_full_gt_manifest(root,out,groups)
    assert read_manifest(out)[0]['group']=='p1'
    assert json.loads(out.with_suffix('.json').read_text())['grouping']=='patient_id'
