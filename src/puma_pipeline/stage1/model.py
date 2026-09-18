from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PumaConfig
from ..constants import NUCLEUS_CLASSES


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.permute(0, 2, 3, 1)
        values = F.layer_norm(values, (values.shape[-1],), self.weight, self.bias)
        return values.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.contract = nn.Conv2d(4 * channels, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-6))
        self.drop_path = float(drop_path)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.norm(values)
        values = F.gelu(self.expand(values))
        values = self.contract(values)
        values = values * self.layer_scale[None, :, None, None]
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = values.new_empty((len(values), 1, 1, 1)).bernoulli_(keep)
            values = values * mask / keep
        return residual + values


class EncoderStage(nn.Module):
    def __init__(self, channels: int, depth: int, drop_paths: list[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(channels, drop_path=drop_paths[index]) for index in range(depth)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return values


class GlobalContextMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=12,
                dim_feedforward=2 * channels,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class FullRoiEncoder(nn.Module):
    def __init__(self, base: int) -> None:
        super().__init__()
        self.channels = (base, base * 2, base * 4, base * 8, base * 12)
        depths = (2, 2, 6, 4, 2)
        drop_paths = torch.linspace(0.0, 0.15, sum(depths)).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 4, stride=2, padding=1),
            LayerNorm2d(self.channels[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cursor = 0
        for index, (channels, depth) in enumerate(zip(self.channels, depths, strict=True)):
            self.stages.append(EncoderStage(channels, depth, drop_paths[cursor : cursor + depth]))
            cursor += depth
            if index + 1 < len(self.channels):
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(channels, self.channels[index + 1], 2, stride=2),
                    )
                )
        self.context = GlobalContextMixer(self.channels[-1])

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        feature = self.stem(image)
        outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            feature = stage(feature)
            if index == len(self.stages) - 1:
                feature = self.context(feature)
            outputs.append(feature)
            if index < len(self.downsamples):
                feature = self.downsamples[index](feature)
        return outputs


class FullRoiFpn(nn.Module):
    def __init__(self, channels: tuple[int, ...], output_channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels_in, output_channels, 1) for channels_in in channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels),
                nn.Conv2d(output_channels, output_channels, 1),
                nn.GroupNorm(16, output_channels),
                nn.GELU(),
            )
            for _ in channels
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        current = self.smooth[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            current = F.interpolate(
                current,
                size=features[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            current = self.smooth[index](current + self.lateral[index](features[index]))
        return current


class NativeResolutionFusion(nn.Module):
    """Fuse FPN features with full-resolution RGB features."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.semantic = nn.Sequential(
            nn.Conv2d(128, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, fpn_feature: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic(fpn_feature)
        semantic = F.interpolate(semantic, size=image.shape[-2:], mode="bilinear", align_corners=False)
        fused = semantic + self.spatial(image)
        return fused + self.refine(fused)


class FullRoiPointDetector(nn.Module):
    """Stage-1 point detector with a full-resolution prediction grid."""


    def __init__(self, config: PumaConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FullRoiEncoder(config.stage1_base_channels)
        self.fpn = FullRoiFpn(self.encoder.channels, output_channels=128)
        self.native_fusion = NativeResolutionFusion(channels=32)
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.offset = nn.Linear(32, 2)
        self.quality = nn.Linear(32, 1)
        self.log_variance = nn.Linear(32, 1)
        self.coarse_type = nn.Linear(32, len(NUCLEUS_CLASSES))
        prior = float(config.stage1_heatmap_prior_probability)
        nn.init.constant_(self.heatmap.bias, math.log(prior / (1.0 - prior)))
        nn.init.zeros_(self.quality.bias)
        nn.init.zeros_(self.log_variance.bias)

    def extract_fpn(self, image: torch.Tensor) -> torch.Tensor:
        return self.fpn(self.encoder(image))

    def predict_points(
        self, native_feature: torch.Tensor, indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("Point indices must have shape [N, 3] as batch,y,x.")
        if indices.numel() == 0:
            values = native_feature.new_empty((0, native_feature.shape[1]))
        else:
            batch, yy, xx = indices.unbind(dim=1)
            values = native_feature[batch, :, yy, xx]
        return {
            "offset": self.offset(values),
            "quality_logits": self.quality(values).squeeze(1),
            "log_variance": self.log_variance(values).squeeze(1).clamp(-5.0, 5.0),
            "coarse_type_logits": self.coarse_type(values),
        }

    def forward(self, image: torch.Tensor, *, return_fpn: bool = False) -> dict[str, torch.Tensor]:
        fpn_feature = self.extract_fpn(image)
        native_feature = self.native_fusion(image, fpn_feature)
        outputs = {
            "heatmap_logits": self.heatmap(native_feature),
            "native_feature": native_feature,
        }
        if return_fpn:
            # The tissue branch uses the FPN feature.
            outputs["fpn_feature"] = fpn_feature
        return outputs


def build_stage1_model(config: PumaConfig) -> FullRoiPointDetector:
    return FullRoiPointDetector(config)
