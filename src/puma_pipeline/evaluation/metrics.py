from __future__ import annotations

from typing import Any
import numpy as np
from ..constants import NUCLEUS_CLASSES
from .matching import match_centroids


def evaluate_rois(gt_by_roi: dict[int,np.ndarray], pred_by_roi: dict[int,np.ndarray], roi_indices: list[int] | np.ndarray) -> dict[str,Any]:
    class_scores={name:[] for name in NUCLEUS_CLASSES}; totals=np.zeros((len(NUCLEUS_CLASSES),3),np.int64)
    for roi in map(int,roi_indices):
        gt=gt_by_roi.get(roi,np.empty(0)); pred=pred_by_roi.get(roi,np.empty(0))
        for class_id,name in enumerate(NUCLEUS_CLASSES):
            g=gt[np.asarray(gt["class_id"],int)==class_id] if len(gt) else gt
            p=pred[np.asarray(pred["class_id"],int)==class_id] if len(pred) else pred
            gxy=np.column_stack((g["x"],g["y"])) if len(g) else np.empty((0,2),np.float32)
            pxy=np.column_stack((p["x"],p["y"])) if len(p) else np.empty((0,2),np.float32)
            # Use equal scores here, so distance decides matching ties.
            m=match_centroids(pxy,gxy,radius=15.0,scores=None); tp=len(m.gt_indices); fp=len(p)-tp; fn=len(g)-tp
            f1=0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn)
            class_scores[name].append(float(f1)); totals[class_id]+=np.asarray((tp,fp,fn))
    class_f1={name:float(np.mean(values)) if values else 0.0 for name,values in class_scores.items()}
    macro=float(np.mean(list(class_f1.values())))
    tp,fp,fn=totals.sum(axis=0); micro=0.0 if 2*tp+fp+fn==0 else float(2*tp/(2*tp+fp+fn))
    return {"macro_f1_nuclei":macro,"class_f1":class_f1,"micro_f1_nuclei":micro,"counts":{NUCLEUS_CLASSES[i]:{"TP":int(v[0]),"FP":int(v[1]),"FN":int(v[2])} for i,v in enumerate(totals)}}
