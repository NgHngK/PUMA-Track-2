from __future__ import annotations

from pathlib import Path

from ..config import PumaConfig


def stage1_oof_paths(config: PumaConfig) -> tuple[Path, Path]:
    output = config.path("stage1_output_dir")
    return (
        output / "stage1_oof_candidates.npy",
        output / "stage1_oof_gt_features.npy",
    )


def stage1_fold_checkpoint(config: PumaConfig, fold: int) -> Path:
    path = config.path("stage1_output_dir") / f"fold_{fold}" / "stage1_final_ema.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stage1_final_checkpoint(config: PumaConfig) -> Path:
    path = config.path("stage1_output_dir") / "final_all_data" / "stage1_final_ema.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
