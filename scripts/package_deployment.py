from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puma_pipeline.config import PumaConfig
from puma_pipeline.stage1.model import build_stage1_model
from puma_pipeline.stage1 import stage1_final_checkpoint
from puma_pipeline.stage2.model import BioContextRefine
from puma_pipeline.stage2.encoder import ensure_uni2_checkpoint
from puma_pipeline.stage2.calibration import load_calibration
from puma_pipeline.tissue import TISSUE_TRAINING_CONTRACT
from puma_pipeline.tissue.model import TissueHead
from puma_pipeline.utils.provenance import sha256_file


def _load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def _validate_stage1(config: PumaConfig, path: Path) -> None:
    payload = _load_checkpoint(path)
    if payload.get("config_fingerprint") != config.stage1_fingerprint:
        raise RuntimeError(f"Stage-1 checkpoint fingerprint does not match deployment config: {path}")
    if payload.get("architecture") != "full_roi_1024_native_stride1_point_detector":
        raise RuntimeError(f"Stage-1 checkpoint is not the native-1024 PUMA detector: {path}")
    model = build_stage1_model(config)
    model.load_state_dict(payload["model_state"], strict=True)


def _validate_stage2(config: PumaConfig, path: Path) -> None:
    payload = _load_checkpoint(path)
    if payload.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError(f"Stage-2 checkpoint fingerprint does not match deployment config: {path}")
    if payload.get("architecture") != "BioContextRefine":
        raise RuntimeError(f"Stage-2 checkpoint has an incompatible architecture: {path}")
    model = BioContextRefine(config)
    model.load_state_dict(payload["model_state"], strict=True)


def _validate_tissue(config: PumaConfig, path: Path) -> None:
    payload = _load_checkpoint(path)
    if payload.get("config_fingerprint") != config.tissue_fingerprint:
        raise RuntimeError(f"Tissue checkpoint fingerprint does not match deployment config: {path}")
    if payload.get("training_contract") != TISSUE_TRAINING_CONTRACT:
        raise RuntimeError(f"Tissue checkpoint has stale preprocessing contract and must be retrained: {path}")
    if payload.get("architecture") != "stage1_fpn_tissue_head":
        raise RuntimeError(f"Tissue checkpoint has an incompatible architecture: {path}")
    model = TissueHead(base=config.tissue_base_channels)
    model.load_state_dict(payload["model_state"], strict=True)


def _validate_risk(config: PumaConfig, path: Path, report_path: Path) -> None:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("config_fingerprint") != config.fingerprint:
        raise RuntimeError("Risk calibration report fingerprint does not match deployment config.")
    # Load the calibration file and check that it matches this config.
    load_calibration(path, expected_fingerprint=config.fingerprint)



def _make_deployment_config(config: PumaConfig) -> PumaConfig:
    payload = config.to_dict()
    payload.update(
        {
            "project_root": "/opt/algorithm",
            "artifact_dir": "unused",
            "stage1_output_dir": "unused",
            "tissue_output_dir": "unused",
            "stage2_output_dir": "unused",
            "cache_dir": "unused",
            "uni2_checkpoint": "models/uni2_h_model.bin",
        }
    )
    return PumaConfig(**payload)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = PumaConfig.load(config_path)
    deployment_config = _make_deployment_config(config)
    deployment_config.save(root / "docker" / "deployment_config.json")

    models = root / "models"
    models.mkdir(exist_ok=True)
    stage1 = stage1_final_checkpoint(config)
    stage2 = config.path("stage2_output_dir") / "full" / "stage2_final_ema.pt"
    tissue = config.path("tissue_output_dir") / "full" / "tissue_final_ema.pt"
    risk = config.path("stage2_output_dir") / "risk_calibration.npz"
    risk_report = config.path("stage2_output_dir") / "risk_calibration_report.json"
    uni2 = ensure_uni2_checkpoint(config.path("uni2_checkpoint"))
    for source in (stage1, stage2, tissue, risk, uni2):
        if not source.is_file():
            raise FileNotFoundError(source)

    _validate_stage1(config, stage1)
    _validate_stage2(config, stage2)
    _validate_tissue(config, tissue)
    _validate_risk(config, risk, risk_report)

    sources = {
        "stage1_final_ema.pt": stage1,
        "stage2_final_ema.pt": stage2,
        "tissue_final_ema.pt": tissue,
        "risk_calibration.npz": risk,
        "risk_calibration_report.json": risk_report,
        "uni2_h_model.bin": uni2,
    }
    manifest = {
        "artifact_schema": "puma",
        "config_fingerprint": config.fingerprint,
        "stage1_fingerprint": config.stage1_fingerprint,
        "tissue_fingerprint": config.tissue_fingerprint,
        "files": {},
    }
    for name, source in sources.items():
        destination = models / name
        shutil.copy2(source, destination)
        manifest["files"][name] = {
            "bytes": int(destination.stat().st_size),
            "sha256": sha256_file(destination),
        }
        print(f"{source} -> {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    (models / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Deployment checkpoint compatibility and hashes verified.")


if __name__ == "__main__":
    main()
