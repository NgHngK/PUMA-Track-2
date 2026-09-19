from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from puma_exploration6.constants import CLASSES
from puma_exploration6.cv import create_internal_group_cv_manifests
from puma_exploration6.manifest import read_manifest, row_identity_sha256, write_manifest


def _rows(n_groups_per_class: int = 3) -> list[dict]:
    rows = []
    uid = 0
    # TRAIN: each class appears in multiple independent groups so 3-fold CV can contain all classes.
    for c, name in enumerate(CLASSES):
        for g in range(n_groups_per_class):
            group = f"train_c{c}_g{g}"
            for j in range(2):
                rows.append({
                    "uid": f"u{uid}", "roi": group, "group": group,
                    "image": f"/not/required/{group}.tif", "x": 20.0 + j, "y": 21.0 + j,
                    "label": c, "class_name": name, "split": "train", "coordinate_source": "gt",
                }); uid += 1
    # External DEV and LOCKED must never enter internal HPO folds.
    for split in ("dev", "locked"):
        for c, name in enumerate(CLASSES):
            group = f"{split}_c{c}"
            rows.append({
                "uid": f"u{uid}", "roi": group, "group": group,
                "image": f"/not/required/{group}.tif", "x": 30.0, "y": 31.0,
                "label": c, "class_name": name, "split": split, "coordinate_source": "gt",
            }); uid += 1
    return rows


def test_internal_cv_preserves_identity_and_hides_external_holdouts(tmp_path: Path):
    src = tmp_path / "master.csv"
    write_manifest(src, _rows())
    original = read_manifest(src, require_files=False)
    original_hash = row_identity_sha256(original)
    external_uids = {r["uid"] for r in original if r["split"] in {"dev", "locked"}}
    folds = create_internal_group_cv_manifests(src, tmp_path / "folds", n_splits=3, seed=1706)
    assert len(folds) == 3
    for f in folds:
        rr = read_manifest(f["manifest"], require_files=False)
        assert row_identity_sha256(rr) == original_hash
        assert external_uids.isdisjoint({r["uid"] for r in rr if r["split"] in {"train", "dev"}})
        train_groups = {r["group"] for r in rr if r["split"] == "train"}
        dev_groups = {r["group"] for r in rr if r["split"] == "dev"}
        assert train_groups.isdisjoint(dev_groups)
        assert {r["label"] for r in rr if r["split"] == "train"} == set(range(10))
        assert {r["label"] for r in rr if r["split"] == "dev"} == set(range(10))


def test_prepare_cv_hpo_generates_trial_x_fold_configs(tmp_path: Path):
    src = tmp_path / "master.csv"; write_manifest(src, _rows())
    base = {
        "experiment_id": "base",
        "manifest": str(src),
        "output_dir": str(tmp_path / "runs"),
        "cached_features": str(tmp_path / "features.npy"),
        "tier_a": str(tmp_path / "tier.npy"),
        "device": "cpu",
        "selection_metric": "macro_f1",
        "optimizer": {"max_epochs": 2, "selection_mode": "early_stop"},
    }
    base_path = tmp_path / "base.json"; base_path.write_text(json.dumps(base), encoding="utf-8")
    out = tmp_path / "cvhpo"
    subprocess.run([
        sys.executable, "scripts/prepare_cv_hpo.py", "--base", str(base_path), "--out-dir", str(out),
        "--stage", "optimizer", "--trials", "2", "--folds", "3", "--seed", "19"
    ], check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    plan = json.loads((out / "CV_HPO_PLAN.json").read_text(encoding="utf-8"))
    assert len(plan["trials"]) == 2
    configs = sorted((out / "configs").glob("*.json"))
    assert len(configs) == 6
    for p in configs:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        assert "_cv" in cfg["experiment_id"]
        assert str(out / "manifests") in cfg["manifest"]


def test_cv_hpo_summary_aggregates_without_raw_histories(tmp_path: Path):
    # Construct a minimal plan and fake compact summaries; the summarizer must not need history.jsonl.
    configs = []
    trials = []
    for t in range(2):
        fold_cfgs = []
        for f in range(3):
            eid = f"trial{t}_cv{f}"
            run_root = tmp_path / "runs"
            cfg = {"experiment_id": eid, "output_dir": str(run_root)}
            cp = tmp_path / f"{eid}.json"; cp.write_text(json.dumps(cfg), encoding="utf-8")
            sd = run_root / eid; sd.mkdir(parents=True)
            score = 0.4 + 0.1 * t + 0.01 * f
            summary = {
                "final_model_score": score,
                "runtime_seconds": 1.0 + f,
                "epochs_run": 5 + f,
                "selected": {"generalization_gap_macro_f1": 0.2 - 0.01 * t},
                "decoupled": None,
            }
            (sd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            fold_cfgs.append(str(cp))
        trials.append({"trial_id": f"trial{t}", "fold_configs": fold_cfgs})
    plan = {"trials": trials}
    pp = tmp_path / "plan.json"; pp.write_text(json.dumps(plan), encoding="utf-8")
    out = tmp_path / "summary.csv"
    subprocess.run([
        sys.executable, "scripts/summarize_cv_hpo.py", "--plan", str(pp), "--out", str(out)
    ], check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["trial_id"] == "trial1"
    assert np.isclose(float(rows[0]["mean_score"]), 0.51)


def test_cv_hpo_end_to_end_with_shared_identity_bound_caches(tmp_path: Path):
    """Broad HPO folds may safely share one master cache without seeing external DEV/LOCKED."""
    from puma_exploration6.utils import sha256_file
    rows = _rows()
    src = tmp_path / "master.csv"; write_manifest(src, rows)
    rr = read_manifest(src, require_files=False)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(len(rr), 1536)).astype(np.float32)
    tier = rng.normal(size=(len(rr), 16)).astype(np.float32)
    fp = tmp_path / "features.npy"; tp = tmp_path / "tier.npy"
    np.save(fp, features); np.save(tp, tier)
    identity = row_identity_sha256(rr)
    fp.with_suffix(".json").write_text(json.dumps({
        "kind": "cls", "shape": list(features.shape), "dtype": "float32",
        "finite_verified": True, "manifest_sha256": sha256_file(src), "row_identity_sha256": identity,
    }), encoding="utf-8")
    tp.with_suffix(".json").write_text(json.dumps({
        "shape": list(tier.shape), "dtype": "float32", "finite_verified": True,
        "manifest_sha256": sha256_file(src), "row_identity_sha256": identity,
    }), encoding="utf-8")
    base = {
        "experiment_id": "smoke",
        "manifest": str(src), "output_dir": str(tmp_path / "runs"),
        "cached_features": str(fp), "tier_a": str(tp), "device": "cpu", "num_workers": 0,
        "selection_metric": "macro_f1",
        "optimizer": {"max_epochs": 1, "patience": 1, "selection_mode": "early_stop", "batch_size": 32},
    }
    bp = tmp_path / "base.json"; bp.write_text(json.dumps(base), encoding="utf-8")
    root = tmp_path / "cvhpo"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/prepare_cv_hpo.py", "--base", str(bp), "--out-dir", str(root), "--stage", "optimizer", "--trials", "1", "--folds", "3", "--seed", "11"], check=True, cwd=repo, capture_output=True, text=True)
    subprocess.run([sys.executable, "scripts/run_batch.py", "--config-dir", str(root / "configs"), "--stop-on-error"], check=True, cwd=repo, capture_output=True, text=True)
    out = root / "CV_HPO_SUMMARY.csv"
    subprocess.run([sys.executable, "scripts/summarize_cv_hpo.py", "--plan", str(root / "CV_HPO_PLAN.json"), "--out", str(out)], check=True, cwd=repo, capture_output=True, text=True)
    with out.open(newline="", encoding="utf-8") as fh:
        agg = list(csv.DictReader(fh))
    assert len(agg) == 1
    assert int(agg[0]["folds"]) == 3
    assert np.isfinite(float(agg[0]["mean_score"]))
