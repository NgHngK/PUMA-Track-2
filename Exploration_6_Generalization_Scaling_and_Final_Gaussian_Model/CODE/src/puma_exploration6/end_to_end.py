from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .constants import CLASSES
from .metrics import CANONICAL_TO_V17
from .puma_v17_evaluator.data.annotations import load_nuclei_polygons,annotation_centroid,normalized_stem
from .puma_v17_evaluator.evaluation.dataset_evaluation import evaluate_dataset_predictions


def evaluate_prediction_csv(gt_geojson_dir: str | Path,prediction_csv: str | Path,roi_list: list[str] | None=None,trace: bool=False) -> dict:
    """Exact V17 evaluation for a complete set of frozen Stage-1 proposals.

    Prediction CSV requires: roi,x,y,score and either class_name or canonical class_id.
    All requested GT ROIs are retained even when they contain zero predictions.
    """
    gt_dir=Path(gt_geojson_dir);files={normalized_stem(p):p for p in gt_dir.glob('*.geojson')}
    if not files:raise FileNotFoundError(f'no GT GeoJSON in {gt_dir}')
    requested=sorted(files) if roi_list is None else [normalized_stem(x) for x in roi_list]
    missing=[r for r in requested if r not in files]
    if missing:raise ValueError(f'GT missing requested ROIs: {missing[:5]}')
    with Path(prediction_csv).open(newline='',encoding='utf-8') as f:pred_rows=list(csv.DictReader(f))
    required={'roi','x','y','score'}
    if pred_rows and not required.issubset(pred_rows[0]):raise ValueError(f'prediction CSV missing {sorted(required-set(pred_rows[0]))}')
    pred_by_name={r:[] for r in requested};seen_uid=set()
    for j,row in enumerate(pred_rows):
        roi=normalized_stem(row['roi'])
        if roi not in pred_by_name:continue
        uid=row.get('uid') or row.get('proposal_uid') or f'pred:{j}'
        uid_key=(roi,str(uid))
        if uid_key in seen_uid:raise ValueError(f'duplicate proposal uid within ROI {roi}: {uid}')
        seen_uid.add(uid_key)
        if row.get('class_name'):
            name=row['class_name'].removeprefix('nuclei_')
            if name not in CLASSES:raise ValueError(f'unknown prediction class {name}')
            canonical=CLASSES.index(name)
        elif row.get('class_id') not in (None,''):
            canonical=int(row['class_id'])
            if canonical not in range(10):raise ValueError('prediction class_id must be canonical 0..9')
        else:raise ValueError('prediction requires class_name or canonical class_id')
        x=float(row['x']);y=float(row['y']);score=float(row['score'])
        if not np.isfinite([x,y,score]).all():raise ValueError('nonfinite prediction')
        pred_by_name[roi].append((x,y,int(CANONICAL_TO_V17[canonical]),score,str(uid)))
    gd=np.dtype([('x','f8'),('y','f8'),('class_id','i2'),('nucleus_index','i8')]);pd=np.dtype([('x','f8'),('y','f8'),('class_id','i2'),('score','f8'),('proposal_uid','U160')])
    gt_by={};pred_by={}
    for idx,roi in enumerate(requested):
        nuclei=load_nuclei_polygons(files[roi]);gt=[]
        for n,(class_id,ring) in enumerate(nuclei):
            c=annotation_centroid(ring);gt.append((float(c[0]),float(c[1]),int(class_id),n))
        gt_by[idx]=np.array(gt,dtype=gd);pred_by[idx]=np.array(pred_by_name[roi],dtype=pd)
    result=evaluate_dataset_predictions(gt_by,pred_by,list(range(len(requested))),trace=trace);result['requested_rois']=requested;result['prediction_rows_used']=sum(len(v) for v in pred_by_name.values());return result
