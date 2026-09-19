from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .constants import NUM_CLASSES
from .manifest import read_manifest, row_identity_sha256, write_manifest
from .utils import sha256_file


EXTERNAL_HOLDOUT_SPLITS = {"dev", "calibration", "locked", "test"}


def _present_classes(rows: list[dict], split: str) -> set[int]:
    return {int(r["label"]) for r in rows if r["split"] == split and int(r["label"]) >= 0}


def create_internal_group_cv_manifests(
    manifest: str | Path,
    out_dir: str | Path,
    *,
    n_splits: int = 3,
    seed: int = 1706,
    require_all_classes: bool = True,
    require_files: bool = False,
) -> list[dict[str, Any]]:
    """Create grouped CV manifests *inside the original TRAIN split only*.

    The returned manifests preserve the exact original row order and sample identity,
    which makes immutable Tier-A/UNI2 caches reusable through ``row_identity_sha256``.
    Original DEV/CALIBRATION/LOCKED/TEST rows are relabelled ``predict`` so broad HPO
    cannot access them through the ordinary train/dev runner.

    Parameters
    ----------
    manifest:
        Exploration-6 master manifest containing at least a non-empty ``train`` split.
    out_dir:
        New/empty directory for fold manifests. Existing non-empty directories fail
        closed to preserve immutable research history.
    n_splits:
        Number of grouped internal folds. Must be >=2 and no greater than the number
        of unique TRAIN groups.
    seed:
        Random state used by StratifiedGroupKFold.
    require_all_classes:
        When true, every internal train and dev fold must contain all canonical
        classes. This is recommended for the ten-class Exploration-6 HPO cohort.
    require_files:
        Whether manifest image paths must exist while constructing folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")
    source_path = Path(manifest)
    rows = read_manifest(source_path, require_files=require_files)
    train_idx = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError("source manifest has no TRAIN rows")

    train_groups = np.asarray([str(rows[int(i)]["group"]) for i in train_idx], dtype=object)
    unique_groups = np.unique(train_groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds unique TRAIN groups={len(unique_groups)}")
    y = np.asarray([int(rows[int(i)]["label"]) for i in train_idx], dtype=np.int64)
    if (y < 0).any() or (y >= NUM_CLASSES).any():
        raise ValueError("TRAIN contains labels outside the canonical ten-class ontology")

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"CV manifest directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    source_identity = row_identity_sha256(rows)
    source_sha = sha256_file(source_path)
    original_external_uids = {r["uid"] for r in rows if r["split"] in EXTERNAL_HOLDOUT_SPLITS}

    # X is intentionally a dummy array; only y/groups matter for this splitter.
    X = np.zeros((len(train_idx), 1), dtype=np.float32)
    for fold, (subtrain_pos, subdev_pos) in enumerate(splitter.split(X, y, groups=train_groups)):
        subtrain_global = set(int(train_idx[int(p)]) for p in subtrain_pos)
        subdev_global = set(int(train_idx[int(p)]) for p in subdev_pos)
        if subtrain_global & subdev_global:
            raise RuntimeError("internal CV row overlap")

        subtrain_groups = {str(rows[i]["group"]) for i in subtrain_global}
        subdev_groups = {str(rows[i]["group"]) for i in subdev_global}
        if subtrain_groups & subdev_groups:
            raise RuntimeError("internal CV group leakage")

        fold_rows: list[dict] = []
        for i, r in enumerate(rows):
            rr = copy.deepcopy(r)
            if i in subtrain_global:
                rr["split"] = "train"
            elif i in subdev_global:
                rr["split"] = "dev"
            else:
                # Original holdouts and any non-TRAIN rows are inaccessible to broad HPO.
                rr["split"] = "predict"
            fold_rows.append(rr)

        if row_identity_sha256(fold_rows) != source_identity:
            raise RuntimeError("internal CV construction changed ordered sample identity")
        if any(r["uid"] in original_external_uids and r["split"] in {"train", "dev"} for r in fold_rows):
            raise RuntimeError("external holdout row leaked into internal HPO")

        if require_all_classes:
            expected = set(range(NUM_CLASSES))
            missing_train = expected - _present_classes(fold_rows, "train")
            missing_dev = expected - _present_classes(fold_rows, "dev")
            if missing_train or missing_dev:
                raise ValueError(
                    f"fold {fold} missing canonical classes; train={sorted(missing_train)}, dev={sorted(missing_dev)}"
                )

        fold_path = out / f"internal_cv_fold{fold}.csv"
        metadata = {
            "purpose": "Exploration-6 broad HPO; grouped CV inside original TRAIN only",
            "fold": fold,
            "n_splits": n_splits,
            "seed": seed,
            "source_manifest": str(source_path),
            "source_manifest_sha256": source_sha,
            "source_row_identity_sha256": source_identity,
            "train_rows": len(subtrain_global),
            "dev_rows": len(subdev_global),
            "train_groups": len(subtrain_groups),
            "dev_groups": len(subdev_groups),
            "external_holdout_rows_hidden_as_predict": len(original_external_uids),
        }
        write_manifest(fold_path, fold_rows, metadata=metadata)
        # Re-read fail-closed validation. require_files follows caller preference.
        checked = read_manifest(fold_path, require_files=require_files)
        if row_identity_sha256(checked) != source_identity:
            raise RuntimeError("written CV manifest changed ordered sample identity")
        folds.append({"fold": fold, "manifest": str(fold_path), **metadata})

    (out / "CV_FOLDS.json").write_text(json.dumps({"folds": folds}, indent=2), encoding="utf-8")
    return folds
