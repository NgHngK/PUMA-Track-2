from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .constants import CLASSES
from .experiment import _build_model, _dataset_for_indices, _load_tier_a
from .manifest import read_manifest
from .models.uni2 import checkpoint_sha256
from .tier_a import TierANormalizer
from .utils import sha256_file


def _safe_inference_batch_size(cfg: ExperimentConfig, device: torch.device) -> int:
    """Execution-only batch sizing. Training/effective batch is never changed."""
    if device.type != "cuda":
        return max(1, int(cfg.optimizer.batch_size))
    # Cached CLS is tiny; cached token tensors are much larger; raw UNI2 is VRAM-heavy.
    if cfg.uni2_weights is None and cfg.cached_features is not None:
        return max(int(cfg.optimizer.batch_size), 128)
    if cfg.uni2_weights is None:
        return max(int(cfg.optimizer.batch_size), 8)
    gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    cap = 1 if gib < 16 else (2 if gib < 24 else 4)
    return max(1, min(cap, int(cfg.optimizer.batch_size)))


def _target_config_from_checkpoint(
    checkpoint: dict[str, Any],
    target_manifest: str | Path,
    *,
    target_tier_a: str | Path | None,
    target_representation: str | Path | None,
    uni2_weights: str | Path | None,
    device: str | None,
    num_workers: int | None,
) -> ExperimentConfig:
    d = copy.deepcopy(checkpoint["config"])
    # The checkpoint is the architecture/training-contract source of truth. Only
    # inference artifact paths and execution-only settings may change here.
    d["manifest"] = str(target_manifest)
    d["tier_a"] = None if target_tier_a is None else str(target_tier_a)
    if device is not None:
        d["device"] = str(device)
    if num_workers is not None:
        d["num_workers"] = int(num_workers)

    source = "raw" if d.get("uni2_weights") is not None else ("cls" if d.get("cached_features") is not None else "tokens")
    if source == "raw":
        if target_representation is not None:
            raise ValueError("target_representation is invalid for a raw-image checkpoint")
        if uni2_weights is not None:
            d["uni2_weights"] = str(uni2_weights)
        if d.get("uni2_weights") is None:
            raise ValueError("raw-image checkpoint requires local UNI2 weights")
        d["cached_features"] = None
        d["cached_tokens"] = None
    else:
        if uni2_weights is not None:
            raise ValueError("uni2_weights is invalid for a cached-representation checkpoint")
        if target_representation is None:
            raise ValueError("cached checkpoint inference requires target_representation")
        d["uni2_weights"] = None
        if source == "cls":
            d["cached_features"] = str(target_representation); d["cached_tokens"] = None
        else:
            d["cached_features"] = None; d["cached_tokens"] = str(target_representation)
    # experiment_id/output_dir are irrelevant to pure inference but retain safe
    # checkpoint values so the normal ExperimentConfig schema remains authoritative.
    return ExperimentConfig.from_dict(d)


@torch.inference_mode()
def predict_checkpoint_on_manifest(
    checkpoint_path: str | Path,
    target_manifest: str | Path,
    *,
    target_tier_a: str | Path | None = None,
    target_representation: str | Path | None = None,
    uni2_weights: str | Path | None = None,
    split: str = "predict",
    device: str | None = None,
    num_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run a frozen selected checkpoint on a new proposal manifest.

    This is inference only: target labels are never used, no normalizer is fitted,
    and no training/selection state changes.  A cached target representation must
    have a sidecar bound to the target manifest/row identity. Raw-image inference
    must use the exact UNI2 checkpoint hash bound into the trained checkpoint.
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(ck.get("classes", ())) != CLASSES:
        raise ValueError("checkpoint class ontology mismatch")
    if not isinstance(ck.get("config"), dict):
        raise ValueError("checkpoint lacks experiment config")
    cfg = _target_config_from_checkpoint(
        ck, target_manifest, target_tier_a=target_tier_a,
        target_representation=target_representation, uni2_weights=uni2_weights,
        device=device, num_workers=num_workers,
    )
    rows = read_manifest(target_manifest, require_files=cfg.uni2_weights is not None)
    split = "dev" if split == "val" else split
    ix = np.asarray([i for i, r in enumerate(rows) if r["split"] == split], dtype=np.int64)
    if len(ix) == 0:
        raise ValueError(f"target manifest has no rows in split={split!r}")
    if any(int(rows[int(i)]["label"]) >= 0 for i in ix) and split == "predict":
        # Labeled rows are legal in predict manifests, but disallow them here to
        # make it impossible to accidentally use inference output as a hidden test
        # during a nominally unlabeled complete-proposal run.
        raise ValueError("predict split must use label=-1 for inference-only proposal rows")

    if cfg.model.use_tier_a:
        if target_tier_a is None:
            raise ValueError("checkpoint uses Tier-A; target_tier_a is required")
        raw = _load_tier_a(str(target_tier_a), len(rows), sha256_file(target_manifest), rows)
        state = ck.get("tier_a_normalizer")
        if not isinstance(state, dict) or "mean" not in state or "std" not in state:
            raise ValueError("checkpoint lacks fitted Tier-A normalizer")
        norm = TierANormalizer(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32))
        tier = norm.transform(raw)
    else:
        if target_tier_a is not None:
            raise ValueError("target_tier_a supplied but checkpoint has Tier-A disabled")
        tier = None

    if cfg.uni2_weights is not None:
        expected = (ck.get("artifact_hashes") or {}).get("uni2_sha256")
        if not expected:
            raise ValueError("raw checkpoint lacks bound UNI2 hash")
        if checkpoint_sha256(cfg.uni2_weights) != expected:
            raise ValueError("target inference UNI2 weights differ from training checkpoint")

    ds = _dataset_for_indices(cfg, rows, ix, tier, False)
    run_device = torch.device(cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    loader_kwargs=dict(
        batch_size=_safe_inference_batch_size(cfg, run_device), shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=run_device.type=="cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers>0: loader_kwargs["prefetch_factor"]=2
    loader = DataLoader(ds, **loader_kwargs)
    model, encoder = _build_model(cfg, run_device)
    model.load_state_dict(ck["model"], strict=True)
    if encoder is not None and ck.get("encoder_trainable"):
        state = encoder.state_dict(); state.update(ck["encoder_trainable"]); encoder.load_state_dict(state, strict=True)
    model.eval()
    if encoder is not None:
        encoder.eval()

    output: list[dict[str, Any]] = []
    offset = 0
    from .training import _forward
    for batch in loader:
        logits = _forward(model, batch, run_device, encoder).float()
        prob = logits.softmax(-1).cpu().numpy()
        local = batch["index"].cpu().numpy()
        for j, local_i in enumerate(local):
            r = rows[int(ix[int(local_i)])]
            pred = int(prob[j].argmax())
            rec: dict[str, Any] = {
                "roi": str(r["roi"]), "uid": str(r["uid"]),
                "x": float(r["x"]), "y": float(r["y"]),
                "class_id": pred, "class_name": CLASSES[pred],
                "score": float(prob[j, pred]),
            }
            for c, name in enumerate(CLASSES):
                rec[f"p_{name}"] = float(prob[j, c])
            output.append(rec)
        offset += len(local)
    if offset != len(ix):
        raise RuntimeError(f"inference row-count mismatch: produced {offset}, expected {len(ix)}")
    if len({(r["roi"], r["uid"]) for r in output}) != len(output):
        raise ValueError("duplicate proposal uid within ROI after inference")
    return output
