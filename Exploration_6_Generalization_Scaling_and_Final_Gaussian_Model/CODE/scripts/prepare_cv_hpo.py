#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import bootstrap_repo
bootstrap_repo()

import argparse
import copy
import json
from pathlib import Path

from puma_exploration6.config import ExperimentConfig
from puma_exploration6.cv import create_internal_group_cv_manifests
from puma_exploration6.hpo import VALID_HPO_STAGES, generate_hpo_configs


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare token-efficient grouped-CV HPO strictly inside Exploration-6 TRAIN")
    p.add_argument("--base", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stage", choices=VALID_HPO_STAGES, required=True)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=1706)
    a = p.parse_args()

    base_path = Path(a.base)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ExperimentConfig.from_dict(copy.deepcopy(base))
    root = Path(a.out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"CV-HPO output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifests_dir = root / "manifests"
    configs_dir = root / "configs"
    configs_dir.mkdir()

    folds = create_internal_group_cv_manifests(base["manifest"], manifests_dir, n_splits=a.folds, seed=a.seed)
    trials = generate_hpo_configs(base, a.trials, a.stage, a.seed)
    plan_trials = []
    for trial in trials:
        trial_id = trial["experiment_id"]
        fold_cfgs = []
        for f in folds:
            c = copy.deepcopy(trial)
            c["experiment_id"] = f"{trial_id}_cv{f['fold']}"
            c["manifest"] = f["manifest"]
            # Broad HPO must never select on the external Exploration-6 DEV/LOCKED rows.
            ExperimentConfig.from_dict(c)
            path = configs_dir / f"{c['experiment_id']}.json"
            path.write_text(json.dumps(c, indent=2), encoding="utf-8")
            fold_cfgs.append(str(path))
        plan_trials.append({"trial_id": trial_id, "fold_configs": fold_cfgs})

    plan = {
        "purpose": "Broad HPO using group-disjoint internal CV inside original TRAIN only",
        "stage": a.stage,
        "seed": a.seed,
        "n_trials": a.trials,
        "n_folds": a.folds,
        "base_config": str(base_path),
        "source_manifest": base["manifest"],
        "folds": folds,
        "trials": plan_trials,
        "run_command": f"python scripts/run_batch.py --config-dir {configs_dir}",
        "summary_command": f"python scripts/summarize_cv_hpo.py --plan {root / 'CV_HPO_PLAN.json'} --out {root / 'CV_HPO_SUMMARY.csv'}",
    }
    (root / "CV_HPO_PLAN.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"prepared_trials": a.trials, "folds": a.folds, "configs": a.trials * a.folds, "plan": str(root / "CV_HPO_PLAN.json")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
