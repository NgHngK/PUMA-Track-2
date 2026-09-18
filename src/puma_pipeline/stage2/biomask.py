from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hematoxylin_and_gradient(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = rgb.float()
    if x.max() > 1.5:
        x = x / 255.0
    x = x.clamp(1.0 / 255.0, 1.0)
    od = -torch.log(x)
    h = (0.650 * od[:, 0] + 0.704 * od[:, 1] + 0.286 * od[:, 2]).unsqueeze(1) / 2.0
    gx = F.pad(h[:, :, :, 1:] - h[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(h[:, :, 1:, :] - h[:, :, :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return h.clamp(0, 2), gradient.clamp(0, 2)


def biomask_input(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rgb.float()
    if x.max() > 1.5:
        x = x / 255.0
    h, gradient = hematoxylin_and_gradient(x)
    return torch.cat((x, h, gradient), dim=1), h, gradient


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BioMaskHead(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(5, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 4)
        self.dec2 = ConvBlock(c * 6, c * 2)
        self.dec1 = ConvBlock(c * 3, c)
        self.mask = nn.Conv2d(c, 1, 1)
        self.confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 4, c), nn.SiLU(), nn.Linear(c, 1)
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        x, h, gradient = biomask_input(rgb)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return {
            "mask_logits": self.mask(d1),
            "mask_confidence_logit": self.confidence(b).squeeze(1),
            "hematoxylin": h,
            "hematoxylin_gradient": gradient,
        }


def biomask_loss(
    mask_logits: torch.Tensor,
    confidence_logit: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.float()
    if target.ndim == 3:
        target = target[:, None]
    bce = F.binary_cross_entropy_with_logits(mask_logits, target)
    prob = mask_logits.sigmoid()
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    dice = (2 * intersection + 1.0) / (prob.sum(dims) + target.sum(dims) + 1.0)
    dice_loss = 1.0 - dice.mean()
    with torch.no_grad():
        hard = (prob > 0.5).float()
        inter = (hard * target).sum(dims)
        union = ((hard + target) > 0).float().sum(dims)
        quality = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    confidence_loss = F.binary_cross_entropy_with_logits(confidence_logit, quality.clamp(0, 1))
    total = 0.55 * bce + 0.35 * dice_loss + 0.10 * confidence_loss
    return total, {"mask_bce": bce.detach(), "mask_dice_loss": dice_loss.detach(), "mask_conf_loss": confidence_loss.detach()}
