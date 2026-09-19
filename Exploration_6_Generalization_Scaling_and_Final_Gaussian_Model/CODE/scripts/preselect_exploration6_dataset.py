#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import bootstrap_repo
bootstrap_repo()

import argparse
import json
from pathlib import Path

from puma_exploration6.full_manifest import build_full_gt_manifest
from puma_exploration6.manifest import read_manifest
from puma_exploration6.preselection import preselect_prompt6, read_historical_rois


DEFAULT_DATASET = r"D:\Research\PUMA\Code\TRAINING CODE\Dataset"
DEFAULT_HISTORICAL = (
    r"C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4"
    r"\EXPLORATION_3\ARCHITECTURES\A5\INPUT_MANIFESTS\sample_manifest.csv"
)
DEFAULT_OUTPUT = (
    r"C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_PROMPT6\DATA_PRESELECTION"
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "CPU-only preselection for PUMA Exploration 6. Builds the full GT manifest, "
            "audits historical ROI overlap, preserves all unused ROIs as LOCKED_NATURAL, "
            "and creates group-disjoint D300/D600/D900 + DEV + Exploration6 confirmation."
        )
    )
    p.add_argument("--dataset-root", default=DEFAULT_DATASET)
    p.add_argument(
        "--historical-rois",
        default=DEFAULT_HISTORICAL,
        help="Authoritative CSV/TXT/JSON historical ROI artifact; CSV requires roi column.",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    p.add_argument(
        "--group-csv",
        help="Optional roi→patient/case/slide/group CSV. Without it, grouping is ROI-level.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--min-groups-per-split", type=int, default=5)
    a = p.parse_args()

    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} is not empty. Preserve the prior audit and choose a new --out-dir."
        )
    out.mkdir(parents=True, exist_ok=True)

    full_manifest = out / "FULL_GT_MANIFEST.csv"
    full_summary = build_full_gt_manifest(
        a.dataset_root, full_manifest, a.group_csv
    )
    rows = read_manifest(full_manifest)
    historical_rois = read_historical_rois(a.historical_rois)
    selection = preselect_prompt6(
        rows,
        historical_rois,
        out / "SELECTION",
        seed=a.seed,
        min_groups_per_split=a.min_groups_per_split,
    )
    final = {
        "full_manifest": str(full_manifest),
        "full_manifest_summary": full_summary,
        "selection": selection,
    }
    (out / "RUN_SUMMARY.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
