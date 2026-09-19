from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import numpy as np

from .config import ExperimentConfig

VALID_HPO_STAGES = ("optimizer", "regularization", "joint_refine")


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if not 0 < low <= high:
        raise ValueError("log-uniform bounds must satisfy 0 < low <= high")
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _trial_id(base_id: str, stage: str, i: int) -> str:
    return f"{base_id}_{stage}_hpo_{i:03d}"


def generate_hpo_configs(base: dict, n_trials: int, stage: str, seed: int = 1706) -> list[dict]:
    """Generate one *staged* Exploration-6 HPO batch.

    Stages intentionally separate optimizer/schedule variables from model
    regularization variables. ``joint_refine`` is only for a small final local
    interaction check after each component has independently earned inclusion.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if stage not in VALID_HPO_STAGES:
        raise ValueError(f"stage must be one of {VALID_HPO_STAGES}")
    # Validate the base contract before cloning it. Placeholder paths are allowed;
    # ExperimentConfig validates semantics rather than file existence.
    ExperimentConfig.from_dict(copy.deepcopy(base))
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for i in range(n_trials):
        c = copy.deepcopy(base)
        c["experiment_id"] = _trial_id(base["experiment_id"], stage, i)
        c.setdefault("optimizer", {})
        c.setdefault("model", {})
        c.setdefault("loss", {})
        if stage == "optimizer":
            c["optimizer"]["lr"] = log_uniform(rng, 1e-4, 3e-3)
            c["optimizer"]["weight_decay"] = log_uniform(rng, 1e-4, 1e-1)
            effective_batch = int(rng.choice([32, 64, 128]))
            if c.get("uni2_weights") is not None:
                # Raw UNI2-h microbatch is a memory-safety setting, not the HPO variable.
                # Keep the predeclared physical microbatch and vary optimizer effective batch
                # through exact accumulation whenever divisible.
                micro = int(c["optimizer"].get("batch_size", 4))
                if effective_batch % micro != 0:
                    raise ValueError(f"raw-mode HPO effective batch {effective_batch} is not divisible by microbatch {micro}")
                c["optimizer"]["batch_size"] = micro
                c["optimizer"]["accumulation_steps"] = effective_batch // micro
            else:
                c["optimizer"]["batch_size"] = effective_batch
                c["optimizer"]["accumulation_steps"] = 1
            c["optimizer"]["scheduler"] = str(rng.choice(["constant", "cosine", "warmup_cosine"]))
            c["optimizer"]["patience"] = int(rng.choice([2, 4, 6]))
            c["optimizer"]["selection_mode"] = "early_stop"
            c["optimizer"]["max_epochs"] = max(int(c["optimizer"].get("max_epochs", 25)), 20)
        elif stage == "regularization":
            c["model"]["interaction_rank"] = int(rng.choice([2, 4, 8]))
            c["model"]["representation_dropout"] = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
            c["model"]["interaction_dropout"] = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
            c["model"]["tier_a_dropout"] = float(rng.choice([0.0, 0.1, 0.2]))
            c["loss"]["label_smoothing"] = float(rng.choice([0.0, 0.025, 0.05, 0.1]))
        else:  # joint_refine: intentionally narrow around the already-selected base
            base_lr = float(c["optimizer"].get("lr", 1e-3))
            base_wd = float(c["optimizer"].get("weight_decay", 1e-2))
            c["optimizer"]["lr"] = float(np.clip(base_lr * math.exp(rng.uniform(-0.35, 0.35)), 1e-4, 3e-3))
            c["optimizer"]["weight_decay"] = float(np.clip(base_wd * math.exp(rng.uniform(-0.5, 0.5)), 1e-4, 1e-1))
            # Only nearby regularization values; do not reopen architecture search.
            c["model"]["interaction_dropout"] = float(np.clip(float(c["model"].get("interaction_dropout", 0.0)) + rng.choice([-0.05, 0.0, 0.05]), 0.0, 0.3))
            c["loss"]["label_smoothing"] = float(np.clip(float(c["loss"].get("label_smoothing", 0.0)) + rng.choice([-0.025, 0.0, 0.025]), 0.0, 0.1))
        ExperimentConfig.from_dict(c)
        out.append(c)
    return out


def write_hpo_configs(base_path: str | Path, out_dir: str | Path, n_trials: int, stage: str, seed: int = 1706) -> list[dict]:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"HPO output directory is not empty; preserve prior trials and choose a new directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = generate_hpo_configs(base, n_trials, stage, seed)
    for c in configs:
        (out_dir / f"{c['experiment_id']}.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (out_dir / "HPO_STAGE.json").write_text(json.dumps({"stage": stage, "seed": seed, "trials": n_trials, "base": str(base_path)}, indent=2), encoding="utf-8")
    return configs
