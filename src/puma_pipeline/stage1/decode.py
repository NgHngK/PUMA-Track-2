from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..constants import NUCLEUS_CLASSES


@dataclass(slots=True)
class Stage1Prediction:
    coordinates: np.ndarray
    heatmap_score: np.ndarray
    quality: np.ndarray
    uncertainty: np.ndarray
    peak_sharpness: np.ndarray
    embedding: np.ndarray

    def subset(self, indices: np.ndarray) -> "Stage1Prediction":
        return Stage1Prediction(
            coordinates=self.coordinates[indices],
            heatmap_score=self.heatmap_score[indices],
            quality=self.quality[indices],
            uncertainty=self.uncertainty[indices],
            peak_sharpness=self.peak_sharpness[indices],
            embedding=self.embedding[indices],
        )


def empty_prediction() -> Stage1Prediction:
    return Stage1Prediction(
        coordinates=np.empty((0, 2), dtype=np.float32),
        heatmap_score=np.empty(0, dtype=np.float32),
        quality=np.empty(0, dtype=np.float32),
        uncertainty=np.empty(0, dtype=np.float32),
        peak_sharpness=np.empty(0, dtype=np.float32),
        embedding=np.empty((0, len(NUCLEUS_CLASSES)), dtype=np.float16),
    )


def prepare_stage1_maps(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    heatmap = outputs["heatmap_logits"].sigmoid().float()
    local_mean = F.avg_pool2d(F.pad(heatmap, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return {"heatmap": heatmap, "local_mean": local_mean}


def _predict_points(model: torch.nn.Module, feature: torch.Tensor, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    if feature.is_cuda and feature.dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cuda", dtype=feature.dtype):
            return model.predict_points(feature, indices)
    return model.predict_points(feature, indices)


@torch.inference_mode()
def decode_stage1_batch(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    *,
    minimum_threshold: float,
    local_max_radius: int,
    max_candidates: int = 3000,
    maps: dict[str, torch.Tensor] | None = None,
) -> list[Stage1Prediction]:
    maps = prepare_stage1_maps(outputs) if maps is None else maps
    heatmap = maps["heatmap"]
    radius = int(local_max_radius)
    pooled = F.max_pool2d(heatmap, kernel_size=2 * radius + 1, stride=1, padding=radius)
    peaks = (heatmap >= pooled - 1.0e-8) & (heatmap >= float(minimum_threshold))
    local_mean = maps["local_mean"]
    feature = outputs["native_feature"]
    predictions: list[Stage1Prediction] = []

    for batch_index in range(heatmap.shape[0]):
        yy, xx = torch.nonzero(peaks[batch_index, 0], as_tuple=True)
        if xx.numel() == 0:
            predictions.append(empty_prediction())
            continue
        candidate_scores = heatmap[batch_index, 0, yy, xx]
        keep = min(int(max_candidates), int(candidate_scores.numel()))
        _, order = torch.topk(candidate_scores, k=keep, largest=True, sorted=True)
        xx, yy = xx[order], yy[order]
        batch_column = torch.full_like(xx, batch_index)
        point_index = torch.stack((batch_column, yy, xx), dim=1)
        point = _predict_points(model, feature, point_index)
        coordinates = torch.stack((xx, yy), dim=1).float().add_(0.5).add_(point["offset"].float())
        embedding = point["coarse_type_logits"].float().softmax(dim=1)
        selected_heatmap = heatmap[batch_index, 0, yy, xx]
        predictions.append(
            Stage1Prediction(
                coordinates=coordinates.cpu().numpy().astype(np.float32, copy=False),
                heatmap_score=selected_heatmap.cpu().numpy().astype(np.float32, copy=False),
                quality=point["quality_logits"].sigmoid().float().cpu().numpy().astype(np.float32, copy=False),
                uncertainty=point["log_variance"].exp().float().cpu().numpy().astype(np.float32, copy=False),
                peak_sharpness=(selected_heatmap - local_mean[batch_index, 0, yy, xx]).cpu().numpy().astype(np.float32, copy=False),
                embedding=embedding.cpu().numpy().astype(np.float16),
            )
        )
    return predictions


@torch.inference_mode()
def sample_stage1_outputs(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    batch_index: int,
    coordinates: np.ndarray,
    maps: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    height, width = outputs["heatmap_logits"].shape[-2:]
    xx_np = np.clip(np.floor(coordinates[:, 0]).astype(np.int64), 0, width - 1)
    yy_np = np.clip(np.floor(coordinates[:, 1]).astype(np.int64), 0, height - 1)
    device = outputs["heatmap_logits"].device
    xx = torch.as_tensor(xx_np, device=device, dtype=torch.long)
    yy = torch.as_tensor(yy_np, device=device, dtype=torch.long)
    batch_column = torch.full_like(xx, int(batch_index))
    point_index = torch.stack((batch_column, yy, xx), dim=1)
    point = _predict_points(model, outputs["native_feature"], point_index)
    maps = prepare_stage1_maps(outputs) if maps is None else maps
    heatmap = maps["heatmap"]
    local_mean = maps["local_mean"]
    selected_heatmap = heatmap[batch_index, 0, yy, xx]
    return {
        "heatmap_score": selected_heatmap.cpu().numpy().astype(np.float32, copy=False),
        "quality": point["quality_logits"].sigmoid().float().cpu().numpy().astype(np.float32, copy=False),
        "uncertainty": point["log_variance"].exp().float().cpu().numpy().astype(np.float32, copy=False),
        "peak_sharpness": (selected_heatmap - local_mean[batch_index, 0, yy, xx]).cpu().numpy().astype(np.float32, copy=False),
        "embedding": point["coarse_type_logits"].float().softmax(dim=1).cpu().numpy().astype(np.float16),
    }


def suppress_stage1(
    prediction: Stage1Prediction,
    threshold: float,
    radius: float,
    image_size: int = 1024,
) -> Stage1Prediction:
    valid = (
        (prediction.heatmap_score >= float(threshold))
        & (prediction.coordinates[:, 0] >= 0.0)
        & (prediction.coordinates[:, 1] >= 0.0)
        & (prediction.coordinates[:, 0] < image_size)
        & (prediction.coordinates[:, 1] < image_size)
    )
    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) <= 1:
        return prediction.subset(candidate_indices)
    order = candidate_indices[
        np.argsort(-prediction.heatmap_score[candidate_indices], kind="stable")
    ]
    cell_size = max(float(radius), 1.0e-6)
    bins: dict[tuple[int, int], list[int]] = {}
    kept: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        coordinate = prediction.coordinates[index]
        cell = tuple(np.floor(coordinate / cell_size).astype(int).tolist())
        neighbors: list[int] = []
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                neighbors.extend(bins.get((cell[0] + delta_x, cell[1] + delta_y), ()))
        if neighbors:
            nearby = prediction.coordinates[np.asarray(neighbors, dtype=np.int64)]
            if np.any(np.square(nearby - coordinate).sum(axis=1) < float(radius) ** 2):
                continue
        kept.append(index)
        bins.setdefault(cell, []).append(index)
    return prediction.subset(np.asarray(kept, dtype=np.int64))
