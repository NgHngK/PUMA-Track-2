from __future__ import annotations

import os
import random

import numpy as np
import torch


def cuda_bf16_supported() -> bool:
    """Check whether the current CUDA device supports BF16."""
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker is not None and checker())


def resolve_cuda_amp_dtype(use_bfloat16: bool) -> torch.dtype:
    """Use BF16 when supported, otherwise use FP16."""
    return torch.bfloat16 if use_bfloat16 and cuda_bf16_supported() else torch.float16


def make_cuda_grad_scaler(amp_dtype: torch.dtype):
    """Create a GradScaler when FP16 training needs it."""
    enabled = amp_dtype == torch.float16
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # Compatibility with older PyTorch versions.
            try:
                return amp.GradScaler(device="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_compile_recommended() -> bool:
    """Check whether torch.compile is recommended on this GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return int(major) >= 8


def runtime_batch_size(requested: int, *, training: bool) -> int:
    """Return the configured batch size."""
    return max(1, int(requested))


def configure_runtime(seed: int, use_tf32: bool, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
