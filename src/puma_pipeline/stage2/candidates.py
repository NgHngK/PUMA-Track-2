from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ..config import PumaConfig
from ..constants import (
    REJECT_CLASS_ID,
    CANDIDATE_DTYPE,
)
from ..store import PumaArtifactStore, load_stage1_gt_features, load_stage1_oof
from ..stage1 import stage1_oof_paths
from ..utils.provenance import file_signature


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _resolve_stage1_sources(config: PumaConfig) -> tuple[Path, Path]:
    return stage1_oof_paths(config)


def build_candidates(config: PumaConfig, *, force: bool = False) -> Path:
    cache_dir = config.path("cache_dir")
    path = cache_dir / "candidates.npy"
    manifest_path = cache_dir / "candidates_manifest.json"
    oof_path, gt_path = _resolve_stage1_sources(config)
    source_signature = {
        "oof": file_signature(oof_path, content_hash=True),
        "gt": file_signature(gt_path, content_hash=True),
    }
    if path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_dtype = repr(CANDIDATE_DTYPE.descr)
        if (
            manifest.get("artifact_schema") == "puma"
            and manifest.get("stage1_fingerprint") == config.stage1_fingerprint
            and manifest.get("source_signature") == source_signature
            and manifest.get("dtype") == expected_dtype
        ):
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.dtype == CANDIDATE_DTYPE and len(existing) == int(manifest.get("rows", -1)):
                return path
        raise RuntimeError("Existing PUMA candidate cache is stale or incompatible; rebuild with --force.")

    store = PumaArtifactStore.open(config.path("artifact_dir"))
    oof = load_stage1_oof(oof_path, config.number_of_folds)
    gt = load_stage1_gt_features(gt_path, config.number_of_folds)
    output = np.empty(len(oof) + len(gt), dtype=CANDIDATE_DTYPE)

    first = slice(0, len(oof))
    output["source_id"][first] = oof["oof_row_id"]
    for name in (
        "roi_index",
        "x",
        "y",
        "heatmap_score",
        "quality",
        "uncertainty",
        "peak_sharpness",
        "class_id",
        "fold",
    ):
        output[name][first] = oof[name]
    output["stage1_prior"][first] = oof["stage1_embedding"]
    output["kind"][first] = 0
    for i, row in enumerate(oof):
        output["gt_global_index"][i] = store.global_gt_index(
            int(row["roi_index"]), int(row["matched_gt_index"])
        )

    cursor = len(oof)
    for row in gt:
        global_index = -int(row["source_id"]) - 1
        if not 0 <= global_index < int(store.offsets[-1]):
            raise ValueError(f"Invalid GT source_id {int(row['source_id'])}.")
        output["source_id"][cursor] = row["source_id"]
        for name in (
            "roi_index",
            "x",
            "y",
            "heatmap_score",
            "quality",
            "uncertainty",
            "peak_sharpness",
            "class_id",
            "fold",
        ):
            output[name][cursor] = row[name]
        output["stage1_prior"][cursor] = row["stage1_embedding"]
        output["kind"][cursor] = 1
        output["gt_global_index"][cursor] = global_index
        cursor += 1
    if cursor != len(output):
        raise RuntimeError("Candidate conversion wrote the wrong row count.")
    if np.any((output["kind"] == 1) & (output["class_id"] == REJECT_CLASS_ID)):
        raise RuntimeError("Clean GT candidates cannot be reject rows.")
    _atomic_numpy(path, output)
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema": "puma",
                "stage1_fingerprint": config.stage1_fingerprint,
                "stage1_source": str(oof_path),
                "source_signature": source_signature,
                "dtype": repr(CANDIDATE_DTYPE.descr),
                "rows": int(len(output)),
                "oof_rows": int(len(oof)),
                "clean_gt_rows": int(len(gt)),
                "reject_rows": int(np.sum(output["class_id"] == REJECT_CLASS_ID)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
