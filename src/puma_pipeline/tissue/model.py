from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TissueHead(nn.Module):
    """Small tissue decoder that reads frozen Stage-1 FPN features."""

    def __init__(self, in_channels: int = 128, base: int = 32, classes: int = 6) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 2 * base, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(2 * base), 2 * base),
            nn.GELU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1, groups=2 * base, bias=False),
            nn.Conv2d(2 * base, 2 * base, 1, bias=False),
            nn.GroupNorm(_group_count(2 * base), 2 * base),
            nn.GELU(),
            nn.Conv2d(2 * base, base, 1, bias=False),
            nn.GroupNorm(_group_count(base), base),
            nn.GELU(),
            nn.Conv2d(base, classes, 1),
        )

    def forward(self, fpn_feature: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        logits = self.decoder(fpn_feature)
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, output_size, mode="bilinear", align_corners=False)
        return logits
