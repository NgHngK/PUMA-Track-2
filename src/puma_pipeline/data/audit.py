from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import PumaConfig
from ..constants import NUCLEUS_CLASSES
from ..stage2.candidates import build_candidates
from ..store import PumaArtifactStore


def audit_dataset(config:PumaConfig)->dict[str,Any]:
    store=PumaArtifactStore.open(config.path("artifact_dir")); n_rois=len(store.images); n_nuclei=int(store.offsets[-1])
    warnings=[]
    if n_rois!=config.expected_public_rois: warnings.append(f"Local ROI count {n_rois} differs from official public count {config.expected_public_rois}.")
    if n_nuclei!=config.expected_public_nuclei: warnings.append(f"Local nuclei count {n_nuclei} differs from official public count {config.expected_public_nuclei}.")
    if config.strict_expected_dataset and warnings: raise RuntimeError(" ".join(warnings))
    folds=np.asarray(store.folds,dtype=np.int64)
    expected=set(range(config.number_of_folds)); observed=set(np.unique(folds).tolist())
    if observed!=expected: raise RuntimeError(f"Fold coverage mismatch: expected {expected}, got {observed}.")
    class_counts=np.bincount(np.asarray(store.centroids["class_id"],dtype=np.int64),minlength=len(NUCLEUS_CLASSES))
    names=store.manifest.dtype.names or (); case_check="unavailable"
    if "case_id" in names:
        cases=np.asarray(store.manifest["case_id"]).astype(str); mapping={}
        for roi,case in enumerate(cases): mapping.setdefault(case,set()).add(int(folds[roi]))
        leaking={case:sorted(v) for case,v in mapping.items() if len(v)>1}
        if leaking: raise RuntimeError(f"Patient/case leakage across folds: {list(leaking.items())[:10]}")
        case_check=f"passed for {len(mapping)} manifest case IDs"
    else: warnings.append("Manifest has no case_id field; ROI-level separation is verified but patient-level leakage cannot be independently proven.")
    report={"roi_count":n_rois,"nuclei_count":n_nuclei,"fold_roi_counts":np.bincount(folds,minlength=config.number_of_folds).tolist(),"class_counts":{name:int(class_counts[i]) for i,name in enumerate(NUCLEUS_CLASSES)},"case_leakage_check":case_check,"warnings":warnings}
    out=config.path("stage2_output_dir")/"dataset_audit.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); return report


def audit_oof(config:PumaConfig)->dict[str,Any]:
    store=PumaArtifactStore.open(config.path("artifact_dir")); path=build_candidates(config); rows=np.load(path,mmap_mode="r",allow_pickle=False); folds=np.asarray(store.folds,dtype=np.int64)
    if np.any(np.asarray(rows["fold"],int)!=folds[np.asarray(rows["roi_index"],int)]): raise RuntimeError("OOF candidate fold does not equal the ROI held fold.")
    clean=np.asarray(rows["kind"],int)==1; real=~clean
    if np.any(clean & (np.asarray(rows["gt_global_index"],int)<0)): raise RuntimeError("Clean candidate missing GT linkage.")
    if np.any((np.asarray(rows["gt_global_index"],int)>=0)&(np.asarray(rows["gt_global_index"],int)>=int(store.offsets[-1]))): raise RuntimeError("Candidate GT linkage out of range.")
    report={"rows":int(len(rows)),"real_oof":int(real.sum()),"clean_gt":int(clean.sum()),"fold_counts":np.bincount(np.asarray(rows["fold"],int),minlength=config.number_of_folds).tolist(),"contract":"Every row inherits the held fold of its ROI; clean GT rows use only OOF Stage-1 sampled features."}
    out=config.path("stage2_output_dir")/"oof_audit.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
