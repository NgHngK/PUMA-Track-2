from __future__ import annotations

import numpy as np
import torch
from skimage.draw import polygon as draw_polygon
from torch.utils.data import Dataset

from ..constants import MASK_SUPERVISION_AMBIGUOUS, MASK_SUPERVISION_INSTANCE
from .crops import CropTransform


def _polygon_from_flat(nuclei: np.ndarray, polygon_points: np.ndarray, gt_index: int) -> np.ndarray:
    row = nuclei[int(gt_index)]
    start = int(row["polygon_start"])
    length = int(row["polygon_length"])
    return np.asarray(polygon_points[start : start + length], dtype=np.float32)


def _mask_in_transform(points: np.ndarray, transform: CropTransform) -> np.ndarray:
    local = transform.roi_to_local(points)
    mask = np.zeros((transform.crop_size, transform.crop_size), dtype=np.float32)
    rr, cc = draw_polygon(local[:, 1], local[:, 0], shape=mask.shape)
    mask[rr, cc] = 1.0
    mask *= transform.valid_region_mask(dtype=np.float32)
    return mask


def _gaussian_prompt(transform: CropTransform, proposal_x: float, proposal_y: float, sigma: float) -> np.ndarray:
    point = transform.roi_to_local(np.asarray([[proposal_x, proposal_y]], dtype=np.float32))[0]
    yy, xx = np.indices((transform.crop_size, transform.crop_size), dtype=np.float32)
    prompt = np.exp(-((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / (2.0 * float(sigma) ** 2))
    return prompt.astype(np.float32, copy=False)


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    contrast = float(rng.uniform(0.90, 1.10))
    brightness = float(rng.uniform(-0.04, 0.04))
    gamma = float(rng.uniform(0.90, 1.10))
    channel = rng.uniform(0.96, 1.04, size=(1, 1, 3)).astype(np.float32)
    x = np.clip((x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    x = np.clip(x * channel, 0.0, 1.0)
    x = np.clip(x, 1.0e-4, 1.0) ** gamma
    return (x * 255.0 + 0.5).astype(np.uint8)


def _transform_vector(dx: float, dy: float, rotation: int, reflect: bool) -> tuple[float, float]:
    x, y = float(dx), float(dy)
    for _ in range(int(rotation) % 4):
        x, y = y, -x
    if reflect:
        x = -x
    return x, y


def _d4(
    crop: np.ndarray,
    prompt: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    offset: tuple[float, float],
    rotation: int,
    reflect: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    k = int(rotation) % 4
    crop_t = np.rot90(crop, k=k, axes=(0, 1))
    prompt_t = np.rot90(prompt, k=k, axes=(0, 1))
    mask_t = np.rot90(mask, k=k, axes=(0, 1))
    valid_t = np.rot90(valid, k=k, axes=(0, 1))
    if reflect:
        crop_t = np.flip(crop_t, axis=1)
        prompt_t = np.flip(prompt_t, axis=1)
        mask_t = np.flip(mask_t, axis=1)
        valid_t = np.flip(valid_t, axis=1)
    return (
        np.ascontiguousarray(crop_t),
        np.ascontiguousarray(prompt_t),
        np.ascontiguousarray(mask_t),
        np.ascontiguousarray(valid_t),
        _transform_vector(offset[0], offset[1], k, reflect),
    )


def _center_inside(transform: CropTransform, x: float, y: float) -> bool:
    local = transform.roi_to_local(np.asarray([[x, y]], dtype=np.float32))[0]
    return bool(0.0 <= local[0] < transform.crop_size and 0.0 <= local[1] < transform.crop_size)


class ProposalBiologyDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        proposals: np.ndarray,
        targets: np.ndarray,
        nuclei: np.ndarray,
        polygon_points: np.ndarray,
        indices: np.ndarray,
        *,
        crop_size: int,
        prompt_sigma: float,
        crop_shift_pixels: int,
        crop_shift_attempts: int,
        positive_visibility_retained_min: float,
        ambiguous_visibility_retained_min: float,
        ambiguous_shift_scale: float,
        d4_probability: float,
        augment: bool,
        seed: int,
        presence_sampling_weights: np.ndarray | None = None,
        negative_strata: np.ndarray | None = None,
    ) -> None:
        self.images = images
        self.proposals = proposals
        self.targets = targets
        self.nuclei = nuclei
        self.polygon_points = polygon_points
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_size = int(crop_size)
        self.prompt_sigma = float(prompt_sigma)
        self.crop_shift_pixels = int(crop_shift_pixels)
        self.crop_shift_attempts = int(crop_shift_attempts)
        self.positive_visibility_retained_min = float(positive_visibility_retained_min)
        self.ambiguous_visibility_retained_min = float(ambiguous_visibility_retained_min)
        self.ambiguous_shift_scale = float(ambiguous_shift_scale)
        self.d4_probability = float(d4_probability)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.presence_sampling_weights = (
            np.ones(len(proposals), dtype=np.float32)
            if presence_sampling_weights is None
            else np.asarray(presence_sampling_weights, dtype=np.float32)
        )
        if self.presence_sampling_weights.shape != (len(proposals),):
            raise ValueError("presence_sampling_weights must match proposal count")
        self.negative_strata = (
            np.full(len(proposals), -1, dtype=np.int8)
            if negative_strata is None
            else np.asarray(negative_strata, dtype=np.int8)
        )
        if self.negative_strata.shape != (len(proposals),):
            raise ValueError("negative_strata must match proposal count")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _visibility_reference(self, proposal_index: int, image_shape: tuple[int, ...]) -> tuple[int, np.ndarray | None, float]:
        target = self.targets[proposal_index]
        mode = int(target["mask_supervision_mode_target"])
        if mode == MASK_SUPERVISION_INSTANCE:
            gt_index = int(target["instance_gt_index_target"])
        elif mode == MASK_SUPERVISION_AMBIGUOUS and int(target["nucleus_presence_target"]) == 1:
            gt_index = int(target["nearest_gt_index_meta"])
        else:
            return -1, None, 0.0
        if gt_index < 0:
            return -1, None, 0.0
        proposal = self.proposals[proposal_index]
        points = _polygon_from_flat(self.nuclei, self.polygon_points, gt_index)
        zero = CropTransform.from_center(float(proposal["x"]), float(proposal["y"]), self.crop_size, image_shape)
        native_mass = float(_mask_in_transform(points, zero).sum())
        return gt_index, points, native_mass

    def _sample_transform(
        self,
        proposal_index: int,
        image_shape: tuple[int, ...],
        rng: np.random.Generator,
    ) -> tuple[CropTransform, float]:
        proposal = self.proposals[proposal_index]
        target = self.targets[proposal_index]
        zero = CropTransform.from_center(float(proposal["x"]), float(proposal["y"]), self.crop_size, image_shape)
        if not self.augment or self.crop_shift_pixels <= 0:
            return zero, 1.0

        gt_index, points, native_mass = self._visibility_reference(proposal_index, image_shape)
        mode = int(target["mask_supervision_mode_target"])
        limit = self.crop_shift_pixels
        retained_min = 0.0
        if mode == MASK_SUPERVISION_INSTANCE:
            retained_min = self.positive_visibility_retained_min
        elif mode == MASK_SUPERVISION_AMBIGUOUS and int(target["nucleus_presence_target"]) == 1:
            limit = max(1, int(round(limit * self.ambiguous_shift_scale)))
            retained_min = self.ambiguous_visibility_retained_min

        for _ in range(self.crop_shift_attempts):
            shift_x = int(np.rint(rng.triangular(-limit, 0.0, limit)))
            shift_y = int(np.rint(rng.triangular(-limit, 0.0, limit)))
            transform = CropTransform.from_center(
                float(proposal["x"]) + shift_x,
                float(proposal["y"]) + shift_y,
                self.crop_size,
                image_shape,
            )
            if gt_index < 0 or points is None or retained_min <= 0:
                return transform, 1.0
            gt = self.nuclei[gt_index]
            if not _center_inside(transform, float(gt["x"]), float(gt["y"])):
                continue
            shifted_mass = float(_mask_in_transform(points, transform).sum())
            retained = shifted_mass / max(native_mass, 1.0e-6)
            if retained >= retained_min:
                return transform, float(min(retained, 1.5))
        # Never relabel an impossible artificial positive. Zero shift preserves the native condition.
        return zero, 1.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        proposal_index = int(self.indices[index])
        proposal = self.proposals[proposal_index]
        target = self.targets[proposal_index]
        image = np.asarray(self.images[int(proposal["roi_index"])])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, proposal_index]))
        transform, retained_visibility = self._sample_transform(proposal_index, image.shape, rng)
        crop = transform.extract_uint8(image)
        valid = transform.valid_region_mask(dtype=np.float32)
        prompt = _gaussian_prompt(transform, float(proposal["x"]), float(proposal["y"]), self.prompt_sigma)

        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        mode = int(target["mask_supervision_mode_target"])
        gt_index = int(target["instance_gt_index_target"])
        if mode == MASK_SUPERVISION_INSTANCE and gt_index >= 0:
            points = _polygon_from_flat(self.nuclei, self.polygon_points, gt_index)
            mask = _mask_in_transform(points, transform)
        offset = (
            float(target["center_offset_x_target"]),
            float(target["center_offset_y_target"]),
        )

        if self.augment:
            crop = _photometric(crop, rng)
            if rng.random() < self.d4_probability:
                choices = [(r, f) for r in range(4) for f in (False, True) if not (r == 0 and not f)]
                rotation, reflect = choices[int(rng.integers(0, len(choices)))]
                crop, prompt, mask, valid, offset = _d4(crop, prompt, mask, valid, offset, rotation, reflect)

        rgb = crop.astype(np.float32) / 255.0
        model_input = np.concatenate((rgb.transpose(2, 0, 1), prompt[None]), axis=0)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid[None])),
            "presence_target": torch.tensor(float(target["nucleus_presence_target"]), dtype=torch.float32),
            "presence_loss_weight": torch.tensor(float(target["presence_loss_weight_target"]), dtype=torch.float32),
            "presence_sampling_weight": torch.tensor(float(self.presence_sampling_weights[proposal_index]), dtype=torch.float32),
            "instance_mask_loss_weight": torch.tensor(float(target["instance_mask_loss_weight_target"]), dtype=torch.float32),
            "empty_mask_loss_weight": torch.tensor(float(target["empty_mask_loss_weight_target"]), dtype=torch.float32),
            "offset_loss_weight": torch.tensor(float(target["offset_loss_weight_target"]), dtype=torch.float32),
            "quality_loss_weight": torch.tensor(float(target["quality_loss_weight_target"]), dtype=torch.float32),
            "center_offset_target": torch.tensor(offset, dtype=torch.float32),
            "mask_supervision_mode": torch.tensor(mode, dtype=torch.int64),
            "nearest_distance": torch.tensor(float(target["nearest_distance_diagnostic"]), dtype=torch.float32),
            "negative_stratum": torch.tensor(int(self.negative_strata[proposal_index]), dtype=torch.int64),
            "augmentation_retained_visibility": torch.tensor(float(retained_visibility), dtype=torch.float32),
            "proposal_index": proposal_index,
        }
