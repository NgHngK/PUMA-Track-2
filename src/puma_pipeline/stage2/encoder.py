from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

UNI2_REPO_ID = "MahmoodLab/UNI2-h"
UNI2_HF_FILENAME = "pytorch_model.bin"
_MIN_CHECKPOINT_BYTES = 1_000_000


def _uni2_kwargs() -> dict[str, Any]:
    from timm.layers import SwiGLUPacked

    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def _build_architecture() -> nn.Module:
    import timm

    return timm.create_model("vit_giant_patch14_224", pretrained=False, **_uni2_kwargs())


def _offline_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _validate_checkpoint_file(checkpoint: Path) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size < _MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Existing UNI2-h checkpoint looks incomplete ({checkpoint.stat().st_size} bytes): {checkpoint}. "
            "It will not be overwritten automatically; remove the bad file and rerun to allow a clean download."
        )
    return checkpoint


def ensure_uni2_checkpoint(checkpoint: Path) -> Path:
    """Return the local UNI2-h checkpoint and download it if missing."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.exists():
        return _validate_checkpoint_file(checkpoint)
    if _offline_enabled():
        raise FileNotFoundError(
            f"UNI2-h checkpoint is missing at {checkpoint}, but Hugging Face offline mode is enabled."
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "UNI2-h is missing and automatic download requires `huggingface-hub`. "
            "Install the PUMA project dependencies and rerun."
        ) from exc

    print(f"UNI2-h not found at {checkpoint}. Downloading {UNI2_REPO_ID}/{UNI2_HF_FILENAME} once...")
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=UNI2_REPO_ID,
                filename=UNI2_HF_FILENAME,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download gated UNI2-h weights. Make sure your Hugging Face account has access to "
            f"{UNI2_REPO_ID} and that you are logged in (or set HF_TOKEN). Expected local path: {checkpoint}"
        ) from exc

    part = checkpoint.with_name(checkpoint.name + ".part")
    try:
        shutil.copy2(downloaded, part)
        _validate_checkpoint_file(part)
        part.replace(checkpoint)
    finally:
        part.unlink(missing_ok=True)
    print(f"Saved UNI2-h checkpoint to {checkpoint}")
    return _validate_checkpoint_file(checkpoint)


def load_local_uni2(checkpoint: Path) -> nn.Module:
    checkpoint = ensure_uni2_checkpoint(checkpoint)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("UNI2 checkpoint must be a plain state_dict.")
    try:
        with torch.device("meta"):
            model = _build_architecture()
        model.load_state_dict(state, strict=True, assign=True)
    except (TypeError, RuntimeError, NotImplementedError):
        model = _build_architecture()
        model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def preferred_cuda_dtype(use_bfloat16: bool, bf16_supported: bool) -> torch.dtype:
    return torch.bfloat16 if use_bfloat16 and bf16_supported else torch.float16


def _tokens(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return torch.cat([features["x_norm_clstoken"][:, None], features["x_norm_patchtokens"]], dim=1)
        for key in ("last_hidden_state", "features", "x"):
            if key in features:
                features = features[key]
                break
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("UNI2 forward_features must return [B,N,C] tokens.")
    return features


def pool_cls_center_ring(features: Any, encoder: nn.Module, center_size: int = 4) -> torch.Tensor:
    tokens = _tokens(features)
    prefix = max(1, min(int(getattr(encoder, "num_prefix_tokens", 1)), tokens.shape[1] - 1))
    cls_token = tokens[:, 0]
    patches = tokens[:, prefix:]
    grid = int(round(math.sqrt(patches.shape[1])))
    if grid * grid != patches.shape[1]:
        raise RuntimeError("UNI2 patch-token grid is not square.")
    patch_grid = patches.reshape(len(patches), grid, grid, patches.shape[-1])
    width = min(max(1, center_size), grid)
    start = (grid - width) // 2
    stop = start + width
    center = patch_grid[:, start:stop, start:stop].mean(dim=(1, 2))
    mask = torch.ones((grid, grid), dtype=torch.bool, device=patches.device)
    mask[start:stop, start:stop] = False
    ring = patch_grid[:, mask].mean(dim=1)
    return torch.cat((cls_token, center, ring), dim=-1)


def prepare_uint8_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    images = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False, antialias=True)
    images = images.div_(255.0)
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return images.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)


class Uni2FeatureExtractor:
    def __init__(self, checkpoint: Path, device: torch.device, use_bfloat16: bool = True) -> None:
        self.device = device
        if device.type == "cuda":
            self.dtype = preferred_cuda_dtype(use_bfloat16, torch.cuda.is_bf16_supported())
        else:
            self.dtype = torch.float32
        encoder = load_local_uni2(checkpoint)
        self.encoder = (
            encoder.to(device=device, dtype=self.dtype)
            if device.type == "cuda"
            else encoder.to(device=device)
        )

    @torch.inference_mode()
    def extract(self, images: torch.Tensor, microbatch: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        microbatch = max(1, int(microbatch))
        while start < len(images):
            stop = min(start + microbatch, len(images))
            try:
                prepared = prepare_uint8_batch(images[start:stop], self.device)
                with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
                    forward = getattr(self.encoder, "forward_features", self.encoder)
                    pooled = pool_cls_center_ring(forward(prepared), self.encoder)
                outputs.append(pooled.to(torch.float16).cpu())
                start = stop
            except torch.OutOfMemoryError:
                if microbatch == 1:
                    raise
                microbatch = max(1, microbatch // 2)
                torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0)
