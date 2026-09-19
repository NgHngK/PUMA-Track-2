from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image
from shapely.geometry import shape
from .constants import CLASSES
from .manifest import write_manifest
from .puma_v17_evaluator.data.annotations import _rings, annotation_centroid


def _load_group_map(path: str | Path | None) -> tuple[dict[str,str], str | None]:
    if path is None:return {},None
    groups: dict[str,str]={}
    priority=("patient_id","case_id","slide_id","group_id")
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or 'roi' not in reader.fieldnames:
            raise ValueError('group CSV requires an roi column')
        available=[k for k in priority if k in reader.fieldnames]
        if not available:
            raise ValueError('group CSV requires one of patient_id, case_id, slide_id, group_id')
        raw_rows=list(reader)
    if not raw_rows:raise ValueError('group CSV is empty')
    # Use the strongest identifier column that is complete for every mapped ROI.
    chosen=None
    for key in available:
        if all((row.get(key) or '').strip() for row in raw_rows):
            chosen=key;break
    if chosen is None:
        raise ValueError('no grouping column is complete for every row')
    for row in raw_rows:
        roi=(row.get('roi') or '').strip();group=(row.get(chosen) or '').strip()
        if not roi or not group:raise ValueError(f'group CSV contains empty roi/{chosen}')
        if roi in groups and groups[roi]!=group:raise ValueError(f'conflicting {chosen} mapping for ROI {roi}')
        groups[roi]=group
    return groups,chosen


def build_full_gt_manifest(root: str | Path,out: str | Path,group_csv: str | Path | None=None) -> dict:
    root=Path(root);out=Path(out)
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(f'manifest output exists; preserve it and choose a new path: {out}')
    groups,group_field=_load_group_map(group_csv);rows=[]
    geo=root/'01_training_dataset_geojson_nuclei';imgs=root/'01_training_dataset_tif_ROIs'
    geo_files=sorted(geo.glob('*.geojson'))
    if not geo_files:raise FileNotFoundError(f'no GeoJSON annotations in {geo}')
    for p in geo_files:
        roi=p.stem.removesuffix('_nuclei');candidates=[imgs/f'{roi}.tif',imgs/f'{roi}.tiff'];existing=[x for x in candidates if x.is_file()]
        if len(existing)!=1:raise FileNotFoundError(f'expected exactly one TIFF for {roi}; found {existing}')
        image=existing[0]
        if group_csv and roi not in groups:raise ValueError(f'missing patient mapping for {roi}')
        with Image.open(image) as im:w,h=im.size
        obj=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(obj,dict) or obj.get('type')!='FeatureCollection' or not isinstance(obj.get('features'),list):raise ValueError(f'unsupported nuclei GeoJSON format: {p}')
        for i,f in enumerate(obj['features']):
            try:name=f['properties']['classification']['name'].removeprefix('nuclei_')
            except Exception as e:raise ValueError(f'missing nucleus classification {roi}/{i}') from e
            if name not in CLASSES:raise ValueError(f'unknown class {name} at {roi}/{i}')
            g=shape(f.get('geometry'))
            if g.is_empty or not g.is_valid or g.area<=0:raise ValueError(f'invalid polygon {roi}/{i}')
            c=g.centroid
            if not (0<=c.x<w and 0<=c.y<h):raise ValueError(f'nucleus centroid outside ROI image {roi}/{i}: {(c.x,c.y)} vs {(w,h)}')
            rings=_rings(f['geometry']);eval_centroids=[annotation_centroid(r).astype(float).tolist() for r in rings]
            if not eval_centroids:raise ValueError(f'no V17 exterior components for {roi}/{i}')
            rows.append({'uid':f'{roi}:{i}','roi':roi,'group':groups.get(roi,roi),'image':str(image.resolve()),'x':c.x,'y':c.y,
                         'eval_x':eval_centroids[0][0],'eval_y':eval_centroids[0][1],'eval_centroids':json.dumps(eval_centroids),
                         'label':CLASSES.index(name),'class_name':name,'split':'train','coordinate_source':'gt'})
    if not rows:raise ValueError('no nuclei found')
    write_manifest(out,rows,{'source':'full GT annotations','grouping':group_field or 'ROI ONLY; patient/case/slide separation unverified'})
    return {'rows':len(rows),'rois':len({r['roi'] for r in rows}),'groups':len({r['group'] for r in rows})}
