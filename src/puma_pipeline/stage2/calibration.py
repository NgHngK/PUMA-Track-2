from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PumaConfig
from ..evaluation.metrics import evaluate_rois
from ..store import PumaArtifactStore
from .risk import RISK_DIM, apply_risk, fit_risk, risk_features
from .trainer import generate_stage2_oof
from ..utils.provenance import file_signature

FINAL_PRED_DTYPE=np.dtype([("x","f4"),("y","f4"),("class_id","i2"),("confidence","f4")])


def _gt_by_roi(store:PumaArtifactStore)->dict[int,np.ndarray]:
    return {roi:store.roi_centroids(roi) for roi in range(len(store.images))}


def _risk_input(rows:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=risk_features(rows["probabilities"],rows["validity"],rows["heatmap_score"],rows["mask_confidence"],rows["graph_gate"],rows["local_probabilities"])
    correct=(np.asarray(rows["is_reject"],bool)==False)&(np.asarray(rows["class_id"],int)==np.asarray(rows["true_class_id"],int))
    return x,correct.astype(np.float32)


def _predictions(rows:np.ndarray,acceptance:np.ndarray,threshold:float,roi_filter:set[int]|None=None)->dict[int,np.ndarray]:
    result:dict[int,list[tuple[float,float,int,float]]]={}
    keep=acceptance>=float(threshold)
    for row,a,k in zip(rows,acceptance,keep):
        roi=int(row["roi_index"])
        if not k or (roi_filter is not None and roi not in roi_filter): continue
        result.setdefault(roi,[]).append((float(row["x"]),float(row["y"]),int(row["class_id"]),float(a)))
    return {roi:np.asarray(values,dtype=FINAL_PRED_DTYPE) for roi,values in result.items()}


def _best_threshold(config:PumaConfig,store:PumaArtifactStore,rows:np.ndarray,acceptance:np.ndarray,rois:np.ndarray)->tuple[float,float]:
    roi_set=set(map(int,rois)); gt=_gt_by_roi(store); best=(0.5,-1.0)
    for threshold in config.global_threshold_grid:
        metric=evaluate_rois(gt,_predictions(rows,acceptance,threshold,roi_set),rois)["macro_f1_nuclei"]
        if metric>best[1]+1e-12 or (abs(metric-best[1])<=1e-12 and abs(threshold-.5)<abs(best[0]-.5)): best=(float(threshold),float(metric))
    return best


def calibrate_risk(config: PumaConfig, *, force: bool = False) -> Path:
    output = config.path("stage2_output_dir") / "risk_calibration.npz"
    report_path = config.path("stage2_output_dir") / "risk_calibration_report.json"
    oof_path = generate_stage2_oof(config, force=False)
    oof_signature = file_signature(oof_path, content_hash=True)
    if output.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("config_fingerprint") == config.fingerprint and report.get("stage2_oof_signature") == oof_signature:
            try:
                load_calibration(output, expected_fingerprint=config.fingerprint)
                return output
            except (KeyError, RuntimeError, ValueError):
                pass
        print("Rebuilding stale or invalid risk calibration automatically."); force=True

    rows = np.load(oof_path, mmap_mode="r", allow_pickle=False)
    store = PumaArtifactStore.open(config.path("artifact_dir"))
    x, y = _risk_input(rows)
    folds = np.asarray(rows["fold"], int)
    roi_folds = np.asarray(store.folds, int)
    cross_acceptance = np.zeros(len(rows), np.float32)
    thresholds: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    cross_predictions: dict[int, np.ndarray] = {}

    for outer_fold in range(config.number_of_folds):
        outer_train = folds != outer_fold
        outer_held = folds == outer_fold

        inner_acceptance = np.zeros(int(outer_train.sum()), np.float32)
        outer_train_indices = np.flatnonzero(outer_train)
        for inner_fold in range(config.number_of_folds):
            if inner_fold == outer_fold:
                continue
            inner_fit = (folds != outer_fold) & (folds != inner_fold)
            inner_held_global = np.flatnonzero(folds == inner_fold)
            inner_state = fit_risk(
                x[inner_fit], y[inner_fit], config.risk_fit_steps,
                config.risk_learning_rate, config.risk_weight_decay,
                config.seed + 100 * outer_fold + inner_fold,
            )
            positions = np.searchsorted(outer_train_indices, inner_held_global)
            inner_acceptance[positions] = apply_risk(inner_state, x[inner_held_global])

        train_rows = rows[outer_train]
        train_rois = np.flatnonzero(roi_folds != outer_fold)
        threshold, inner_oof_score = _best_threshold(
            config, store, train_rows, inner_acceptance, train_rois
        )
        thresholds.append(threshold)

        outer_state = fit_risk(
            x[outer_train], y[outer_train], config.risk_fit_steps,
            config.risk_learning_rate, config.risk_weight_decay,
            config.seed + outer_fold,
        )
        cross_acceptance[outer_held] = apply_risk(outer_state, x[outer_held])
        held_rois = np.flatnonzero(roi_folds == outer_fold)
        held_predictions = _predictions(
            rows[outer_held], cross_acceptance[outer_held], threshold, set(map(int, held_rois))
        )
        cross_predictions.update(held_predictions)
        held_score = evaluate_rois(
            _gt_by_roi(store), held_predictions, held_rois
        )["macro_f1_nuclei"]
        fold_reports.append(
            {
                "fold": outer_fold,
                "threshold": threshold,
                "inner_cross_fitted_threshold_macro_f1": inner_oof_score,
                "held_macro_f1": held_score,
            }
        )

    deployment_threshold = float(np.median(thresholds))
    final_state = fit_risk(
        x, y, config.risk_fit_steps, config.risk_learning_rate,
        config.risk_weight_decay, config.seed + 99
    )
    all_rois = np.arange(len(store.images))
    nested_cross_score = evaluate_rois(
        _gt_by_roi(store), cross_predictions, all_rois
    )["macro_f1_nuclei"]
    median_threshold_score = evaluate_rois(
        _gt_by_roi(store),
        _predictions(rows, cross_acceptance, deployment_threshold),
        all_rois,
    )["macro_f1_nuclei"]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        weight=np.asarray(final_state["weight"], np.float32),
        bias=np.asarray(final_state["bias"], np.float32),
        threshold=np.float32(deployment_threshold),
        fold_thresholds=np.asarray(thresholds, np.float32),
        config_fingerprint=np.asarray([config.fingerprint]),
    )
    report = {
        "artifact_schema": "puma",
        "config_fingerprint": config.fingerprint,
        "stage2_oof_signature": oof_signature,
        "policy": "class-aware logistic OOF correctness calibrator + nested cross-fitted global acceptance threshold",
        "deployment_threshold": deployment_threshold,
        "nested_cross_fitted_macro_f1": float(nested_cross_score),
        "deployment_median_threshold_cross_fitted_macro_f1": float(median_threshold_score),
        "folds": fold_reports,
        "no_per_class_thresholds": True,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def load_calibration(path: Path, *, expected_fingerprint: str | None = None) -> tuple[dict[str, np.ndarray], float]:
    data = np.load(path, allow_pickle=False)
    required = {"weight", "bias", "threshold"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"Risk calibration is missing arrays: {sorted(missing)}")
    weight = np.asarray(data["weight"], np.float32)
    bias = np.asarray(data["bias"], np.float32)
    threshold_values = np.asarray(data["threshold"], np.float32).reshape(-1)
    if weight.shape != (1, RISK_DIM) or bias.shape != (1,):
        raise RuntimeError(f"Risk calibration shape mismatch: weight={weight.shape}, bias={bias.shape}.")
    if threshold_values.shape != (1,):
        raise RuntimeError(f"Risk calibration threshold must be scalar, got {threshold_values.shape}.")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all() or not np.isfinite(threshold_values).all():
        raise RuntimeError("Risk calibration contains non-finite values.")
    threshold = float(threshold_values[0])
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Risk calibration threshold is outside [0,1]: {threshold}.")
    if expected_fingerprint is not None:
        if "config_fingerprint" not in data.files:
            raise RuntimeError("Risk calibration is missing its config fingerprint.")
        embedded = str(np.asarray(data["config_fingerprint"]).reshape(-1)[0])
        if embedded != expected_fingerprint:
            raise RuntimeError("Risk calibration fingerprint does not match the deployment config.")
    return {"weight": weight, "bias": bias}, threshold
