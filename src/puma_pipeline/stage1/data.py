from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PumaConfig
from .targets import build_stage1_targets
from ..store import PumaArtifactStore


def image_to_uint8_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1)


def apply_dihedral(
    image: np.ndarray,
    coordinates: np.ndarray,
    code: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = image
    xy = np.asarray(coordinates, dtype=np.float32).copy()
    width, height = int(image.shape[1]), int(image.shape[0])
    for _ in range(int(code) % 4):
        output = np.rot90(output, k=1)
        if len(xy):
            old_x, old_y = xy[:, 0].copy(), xy[:, 1].copy()
            xy[:, 0], xy[:, 1] = old_y, float(width) - old_x
        width, height = height, width
    if int(code) >= 4:
        output = np.fliplr(output)
        if len(xy):
            xy[:, 0] = float(width) - xy[:, 0]
    if len(xy):
        max_x = np.nextafter(np.float32(width), np.float32(-np.inf))
        max_y = np.nextafter(np.float32(height), np.float32(-np.inf))
        xy[:, 0] = np.clip(xy[:, 0], 0.0, max_x)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, max_y)
    return np.ascontiguousarray(output), xy


class FullRoiDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: PumaArtifactStore,
        roi_indices: np.ndarray,
        config: PumaConfig,
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.store = store
        self.roi_indices = np.asarray(roi_indices, dtype=np.int64)
        self.config = config
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.repeats = config.stage1_repeats_per_roi_per_epoch if augment else 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return len(self.roi_indices) * self.repeats

    def __getitem__(self, item: int) -> dict[str, Any]:
        roi_index = int(self.roi_indices[item // self.repeats])
        epoch = int(self.epoch.item())
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + item * 97)
        image = np.asarray(self.store.images[roi_index])
        nuclei = self.store.roi_centroids(roi_index)
        coordinates = np.column_stack([nuclei["x"], nuclei["y"]]).astype(np.float32)
        photometric = np.empty(0, dtype=np.float32)
        if self.augment:
            image, coordinates = apply_dihedral(image, coordinates, int(rng.integers(0, 8)))
            gain = rng.uniform(0.88, 1.12, size=3).astype(np.float32)
            bias = rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
            gamma = np.float32(rng.uniform(0.88, 1.12))
            noise = np.float32(rng.uniform(0.0, 0.018))
            photometric = np.concatenate([gain, bias, np.asarray([gamma, noise], dtype=np.float32)])
        targets = build_stage1_targets(coordinates, nuclei["class_id"], self.config)
        return {
            "image": image_to_uint8_tensor(image),
            "targets": {key: torch.from_numpy(value) for key, value in targets.items()},
            "photometric": torch.from_numpy(photometric),
            "roi_index": roi_index,
        }


def _batch_sparse_indices(
    batch: list[dict[str, Any]], name: str
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for batch_index, row in enumerate(batch):
        indices = row["targets"][name]
        if not len(indices):
            continue
        prefix = torch.full((len(indices), 1), batch_index, dtype=torch.long)
        parts.append(torch.cat((prefix, indices.long()), dim=1))
    return torch.cat(parts, dim=0) if parts else torch.empty((0, 3), dtype=torch.long)


def _concat_sparse(batch: list[dict[str, Any]], name: str, width: int, dtype: torch.dtype) -> torch.Tensor:
    parts = [row["targets"][name].to(dtype=dtype) for row in batch if len(row["targets"][name])]
    return torch.cat(parts, dim=0) if parts else torch.empty((0, width), dtype=dtype)


def full_roi_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    center_classes = [row["targets"]["coarse_type"].long() for row in batch if len(row["targets"]["coarse_type"])]
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "targets": {
            "heatmap": torch.stack([row["targets"]["heatmap"] for row in batch]),
            "offset_index": _batch_sparse_indices(batch, "offset_index"),
            "offset_target": _concat_sparse(batch, "offset_target", 2, torch.float32),
            "center_index": _batch_sparse_indices(batch, "center_index"),
            "center_offset": _concat_sparse(batch, "center_offset", 2, torch.float32),
            "coarse_type": (
                torch.cat(center_classes, dim=0)
                if center_classes
                else torch.empty(0, dtype=torch.long)
            ),
        },
        "photometric": (
            torch.stack([row["photometric"] for row in batch])
            if batch[0]["photometric"].numel()
            else torch.empty((len(batch), 0), dtype=torch.float32)
        ),
        "roi_index": torch.tensor([row["roi_index"] for row in batch], dtype=torch.long),
    }


def prepare_full_roi_batch(
    images: torch.Tensor,
    device: torch.device,
    photometric: torch.Tensor | None = None,
) -> torch.Tensor:
    images = images.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        memory_format=torch.channels_last if device.type == "cuda" else torch.preserve_format,
    )
    if photometric is not None and photometric.numel():
        parameters = photometric.to(device=device, dtype=torch.float32, non_blocking=True)
        gain = parameters[:, 0:3, None, None]
        bias = parameters[:, 3:6, None, None]
        gamma = parameters[:, 6:7, None, None]
        noise = parameters[:, 7:8, None, None]
        images.mul_(1.0 / 255.0).mul_(gain).add_(bias).clamp_(0.0, 1.0)
        images.pow_(gamma)
        images.add_(torch.randn_like(images) * noise).clamp_(0.0, 1.0)
    else:
        images.mul_(1.0 / 255.0)
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return images.sub_(mean).div_(std)
