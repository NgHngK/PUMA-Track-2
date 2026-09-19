#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _summary_path(config_path: str | Path) -> tuple[Path, dict]:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return Path(cfg["output_dir"]) / cfg["experiment_id"] / "summary.json", cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate grouped internal-CV HPO without reading raw histories")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    rows = []
    for trial in plan["trials"]:
        vals = []
        gaps = []
        runtimes = []
        epochs = []
        for cfg_path in trial["fold_configs"]:
            sp, cfg = _summary_path(cfg_path)
            if not sp.is_file():
                raise FileNotFoundError(f"missing CV-HPO summary: {sp}")
            s = json.loads(sp.read_text(encoding="utf-8"))
            # final_model_score respects cRT when enabled; otherwise selected checkpoint.
            vals.append(float(s["final_model_score"]))
            selected = s["decoupled"]["epochs"][-1] if s.get("decoupled") else s["selected"]
            gaps.append(float(selected["generalization_gap_macro_f1"]))
            runtimes.append(float(s["runtime_seconds"]))
            epochs.append(int(s["epochs_run"]))
        row = {
            "trial_id": trial["trial_id"],
            "folds": len(vals),
            "mean_score": float(np.mean(vals)),
            "sd_score": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min_score": float(np.min(vals)),
            "max_score": float(np.max(vals)),
            "mean_generalization_gap_macro_f1": float(np.mean(gaps)),
            "mean_runtime_seconds": float(np.mean(runtimes)),
            "mean_epochs_run": float(np.mean(epochs)),
        }
        rows.append(row)
    rows.sort(key=lambda r: (-r["mean_score"], r["mean_generalization_gap_macro_f1"], r["trial_id"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    if not fields:
        raise ValueError("HPO plan contains no trials")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"trials": len(rows), "best_trial": rows[0]["trial_id"], "best_mean_score": rows[0]["mean_score"], "summary": str(out)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
