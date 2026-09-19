from __future__ import annotations

import json
import numpy as np
import torch
from torch.nn import functional as F

from .constants import CLASSES, NUM_CLASSES
from .puma_v17_evaluator.constants import NUCLEUS_CLASSES
from .puma_v17_evaluator.evaluation.dataset_evaluation import evaluate_dataset_predictions

CANONICAL_TO_V17=np.array([NUCLEUS_CLASSES.index("nuclei_"+c) for c in CLASSES],dtype=int)


def semantic_metrics(labels,logits) -> dict:
    y=np.asarray(labels,dtype=int);z=torch.as_tensor(logits,dtype=torch.float32)
    if y.ndim!=1 or z.shape!=(len(y),NUM_CLASSES):raise ValueError("semantic metric shape mismatch")
    if len(y)==0:raise ValueError("empty metric input")
    if (y<0).any() or (y>=NUM_CLASSES).any():raise ValueError("semantic labels must be canonical 0..9")
    if not torch.isfinite(z).all():raise FloatingPointError("nonfinite semantic logits")
    p=z.softmax(-1).numpy();pred=p.argmax(-1);cm=np.bincount(NUM_CLASSES*y+pred,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    tp=cm.diagonal();support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(NUM_CLASSES,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(NUM_CLASSES,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(NUM_CLASSES,dtype=float),where=(support+predicted)>0)
    conf=p.max(1);correct=pred==y;bins=np.minimum((conf*15).astype(int),14);ece=0.0
    for b in range(15):
        ix=bins==b
        if ix.any():ece+=float(ix.mean()*abs(conf[ix].mean()-correct[ix].mean()))
    return {
        "macro_f1":float(f1.mean()),"macro_precision":float(precision.mean()),"macro_recall":float(recall.mean()),
        "balanced_accuracy_fixed10":float(recall.mean()),"accuracy":float(correct.mean()),"ece15":float(ece),
        "nll":float(F.cross_entropy(z,torch.as_tensor(y).long())),"support":support.tolist(),"predicted":predicted.tolist(),
        "precision":precision.tolist(),"recall":recall.tolist(),"f1":f1.tolist(),"confusion":cm.tolist(),
        "missing_classes":[CLASSES[i] for i in range(NUM_CLASSES) if support[i]==0],
    }


def _gt_centers(row: dict) -> list[list[float]]:
    if row.get("eval_centroids") not in (None,""):
        centers=json.loads(row["eval_centroids"]) if isinstance(row["eval_centroids"],str) else row["eval_centroids"]
        return [[float(x),float(y)] for x,y in centers]
    if row.get("eval_x") not in (None,"") and row.get("eval_y") not in (None,""):
        return [[float(row["eval_x"]),float(row["eval_y"])]]
    # GT-centred research manifests may use x/y as the evaluator centroid.
    if row["coordinate_source"]=="gt":return [[float(row["x"]),float(row["y"])]]
    raise ValueError("V17 subset metric requires eval_x/eval_y or eval_centroids for non-GT rows")


def puma_subset_metrics(rows: list[dict],labels,probabilities,trace: bool=False) -> dict:
    labels=np.asarray(labels,dtype=int);prob=np.asarray(probabilities,dtype=float)
    if len(rows)!=len(labels) or prob.shape!=(len(rows),NUM_CLASSES):raise ValueError("PUMA metric shape mismatch")
    if not np.isfinite(prob).all():raise ValueError("nonfinite probabilities")
    pred=prob.argmax(1);gt_by={};pred_by={};names=sorted({r["roi"] for r in rows})
    gd=np.dtype([("x","f8"),("y","f8"),("class_id","i2"),("nucleus_index","i8")])
    pd=np.dtype([("x","f8"),("y","f8"),("class_id","i2"),("score","f8"),("proposal_uid","U160")])
    for j,name in enumerate(names):
        ix=[i for i,r in enumerate(rows) if r["roi"]==name];gt=[]
        for i in ix:
            centers=_gt_centers(rows[i]);start=len(gt)
            gt.extend((x,y,int(CANONICAL_TO_V17[labels[i]]),start+k) for k,(x,y) in enumerate(centers))
        gt_by[j]=np.array(gt,dtype=gd)
        pred_by[j]=np.array([(float(rows[i]["x"]),float(rows[i]["y"]),int(CANONICAL_TO_V17[pred[i]]),float(prob[i,pred[i]]),str(rows[i]["uid"])) for i in ix],dtype=pd)
    return evaluate_dataset_predictions(gt_by,pred_by,list(range(len(names))),trace=trace)


def combined_metrics(rows: list[dict],labels,logits) -> dict:
    z=torch.as_tensor(logits,dtype=torch.float32);out=semantic_metrics(labels,z)
    out["puma"]=puma_subset_metrics(rows,labels,z.softmax(-1).numpy())
    return out
