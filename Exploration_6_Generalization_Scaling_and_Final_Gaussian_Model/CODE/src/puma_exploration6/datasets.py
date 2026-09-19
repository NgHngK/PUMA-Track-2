from __future__ import annotations

from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from .augmentation import centered_crop_with_token_coordinate, image_to_tensor, sample_coordinate_jitter
from .config import AugmentationConfig
from .constants import EMBED_DIM, TIER_A_DIM, UNI2_TOKEN_COUNT
from .tier_a import point_features, roi_channels


class RawNucleiDataset(Dataset):
    """Raw RGB dataset for stochastic augmentation and UNI2 token extraction.

    ``tier_a`` is normally the immutable, TRAIN-normalized Nx16 matrix aligned to
    ``rows``. Geometry/stain augmentation affects the appearance branch only.

    Coordinate jitter is special: it changes the proposal point itself. When
    jitter is enabled and Tier-A is present, Tier-A is therefore recomputed at
    the jittered point from the *original* ROI and normalized with the supplied
    TRAIN-only mean/std. This keeps point-derived biology aligned with the point
    presented to the model instead of silently mixing GT Tier-A with a shifted
    Stage-1-like proposal.

    Augmentation uses a dataset-local RNG stream, isolated from model/dropout RNG
    so matched architecture trials do not accidentally receive different image
    augmentations merely because their trainable modules consume torch randomness.
    """

    def __init__(
        self,
        rows: list[dict],
        tier_a: np.ndarray | None,
        fov: int = 96,
        augmentation: AugmentationConfig | None = None,
        cache_rois: int = 8,
        augmentation_seed: int = 0,
        tier_a_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.rows = rows
        self.fov = int(fov)
        self.augmentation = augmentation or AugmentationConfig()
        self.augmentation.validate()
        self.cache_rois = int(cache_rois)
        self.augmentation_seed = int(augmentation_seed)
        if self.cache_rois < 1:
            raise ValueError("cache_rois must be >=1")
        if self.augmentation_seed < 0:
            raise ValueError("augmentation_seed must be non-negative")
        if tier_a is not None:
            tier_a = np.asarray(tier_a, dtype=np.float32)
            if tier_a.shape != (len(rows), TIER_A_DIM) or not np.isfinite(tier_a).all():
                raise ValueError(f"tier_a must have shape ({len(rows)},16)")
        self.tier_a = tier_a
        self._tier_mean = self._tier_std = None
        if tier_a_normalizer is not None:
            mean, std = (np.asarray(x, dtype=np.float32) for x in tier_a_normalizer)
            if mean.shape != (TIER_A_DIM,) or std.shape != (TIER_A_DIM,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("tier_a_normalizer must contain finite positive 16D mean/std")
            self._tier_mean, self._tier_std = mean, std
        if self.augmentation.coordinate_jitter_sigma > 0 and self.tier_a is not None and self._tier_mean is None:
            raise ValueError("coordinate jitter with Tier-A requires TRAIN-only Tier-A normalizer for dynamic point alignment")
        self.cache: OrderedDict[str, tuple[Image.Image, dict | None]] = OrderedDict()
        self._aug_generator: torch.Generator | None = None
        self._aug_worker_key: int | None = None

    def __len__(self):
        return len(self.rows)

    def _generator(self) -> torch.Generator:
        info = get_worker_info()
        worker_key = -1 if info is None else int(info.id)
        if self._aug_generator is None or self._aug_worker_key != worker_key:
            # A large odd stride keeps independent worker streams deterministic.
            seed = self.augmentation_seed + (worker_key + 1) * 1_000_003
            self._aug_generator = torch.Generator().manual_seed(seed)
            self._aug_worker_key = worker_key
        return self._aug_generator

    def _image(self, path: str, need_channels: bool = False) -> tuple[Image.Image, dict | None]:
        if path not in self.cache:
            with Image.open(path) as im:
                rgb = im.convert("RGB").copy()
            self.cache[path] = (rgb, roi_channels(rgb) if need_channels else None)
            while len(self.cache) > self.cache_rois:
                self.cache.popitem(last=False)
        rgb, channels = self.cache[path]
        if need_channels and channels is None:
            channels = roi_channels(rgb)
            self.cache[path] = (rgb, channels)
        self.cache.move_to_end(path)
        return rgb, channels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r = self.rows[index]
        need_dynamic_tier = self.tier_a is not None and self.augmentation.coordinate_jitter_sigma > 0
        im, channels = self._image(r["image"], need_channels=need_dynamic_tier)
        w, h = im.size
        g = self._generator()
        dx, dy = sample_coordinate_jitter(self.augmentation, generator=g)
        cx = float(np.clip(float(r["x"]) + dx, 0, max(0, w - 1e-6)))
        cy = float(np.clip(float(r["y"]) + dy, 0, max(0, h - 1e-6)))
        crop, u, v = centered_crop_with_token_coordinate(im, cx, cy, self.fov)
        image, u, v, _ = image_to_tensor(crop, u, v, self.augmentation, generator=g)
        if self.tier_a is None:
            bio = torch.zeros(TIER_A_DIM, dtype=torch.float32)
        elif need_dynamic_tier:
            assert channels is not None and self._tier_mean is not None and self._tier_std is not None
            raw = point_features(channels, cx, cy)
            z = (raw - self._tier_mean) / self._tier_std
            if not np.isfinite(z).all():
                raise FloatingPointError("nonfinite dynamic Tier-A")
            bio = torch.from_numpy(z.astype(np.float32, copy=False))
        else:
            bio = torch.from_numpy(self.tier_a[index])
        return {
            "image": image,
            "bio": bio,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "group": str(r.get("group", "")),
            "u": torch.tensor(u, dtype=torch.float32),
            "v": torch.tensor(v, dtype=torch.float32),
        }


class CachedNucleiDataset(Dataset):
    """Cached CLS or full-token dataset with lazy row indexing.

    A full read-only NumPy memmap can be shared by TRAIN/train-eval/DEV datasets
    without materializing multi-gigabyte fancy-index copies.  ``representation_indices``
    maps local dataset rows to rows in the shared cache; returned ``index`` remains
    local so metric ordering is unchanged.
    """

    def __init__(self, rows: list[dict], representation: np.ndarray, tier_a: np.ndarray | None, fov: int = 96,
                 representation_indices: np.ndarray | list[int] | None = None, validate_finite: bool = True):
        rep = representation if isinstance(representation, np.ndarray) else np.asarray(representation)
        if rep.ndim == 2:
            if rep.shape[1] != EMBED_DIM:
                raise ValueError("cached CLS must have width 1536")
            self.kind = "features"
        elif rep.ndim == 3:
            if rep.shape[1:] != (UNI2_TOKEN_COUNT, EMBED_DIM):
                raise ValueError("cached tokens must be Nx265x1536")
            self.kind = "tokens"
        else:
            raise ValueError("representation must be rank 2 or 3")
        if not np.issubdtype(rep.dtype, np.floating):
            raise ValueError("cached representation must be floating point")
        if representation_indices is None:
            if len(rep) != len(rows):
                raise ValueError("cached representation length must match rows when no index map is supplied")
            idx = np.arange(len(rows), dtype=np.int64)
        else:
            idx = np.asarray(representation_indices, dtype=np.int64)
            if idx.shape != (len(rows),) or (idx < 0).any() or (idx >= len(rep)).any():
                raise ValueError("invalid representation_indices")
        # Legacy/external caches without a sidecar finite-verification flag are
        # scanned lazily row-by-row once. Exploration-6's own streaming extractor marks
        # finite_verified=true, avoiding a second multi-GB read pass.
        if validate_finite:
            for gi in idx:
                if not np.isfinite(np.asarray(rep[int(gi)])).all():
                    raise ValueError("nonfinite cached representation")
        if tier_a is not None:
            tier_a = np.asarray(tier_a, dtype=np.float32)
            if tier_a.shape != (len(rows), TIER_A_DIM) or not np.isfinite(tier_a).all():
                raise ValueError("invalid Tier-A cache")
        self.rows = rows
        self.rep = rep
        # NumPy memmap pickling serializes the mapped bytes. On Windows, where
        # DataLoader workers use spawn, that can copy multi-gigabyte token caches
        # once per worker. Keep the backing .npy path so spawned workers can reopen
        # the read-only mapping instead of receiving the full array payload.
        self._rep_memmap_path = str(rep.filename) if isinstance(rep, np.memmap) else None
        self.rep_indices = idx
        self.tier_a = tier_a
        self.fov = int(fov)
        if self.fov < 16 or self.fov % 2:
            raise ValueError("fov must be even and >=16")


    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_rep_memmap_path") is not None:
            state["rep"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.rep is None:
            path = self._rep_memmap_path
            if not path:
                raise RuntimeError("cached representation memmap path missing during worker restore")
            self.rep = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def _rep_tensor(self, index: int) -> torch.Tensor:
        # torch.tensor copies one selected sample into writable CPU memory; unlike
        # rep[self.rep_indices] this never materializes the whole subset. DataLoader
        # subsequently batches these samples normally.
        return torch.tensor(np.asarray(self.rep[int(self.rep_indices[index])]), dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r = self.rows[index]
        bio = torch.zeros(TIER_A_DIM, dtype=torch.float32) if self.tier_a is None else torch.from_numpy(self.tier_a[index])
        item = {
            "bio": bio,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "group": str(r.get("group", "")),
        }
        if self.kind == "features":
            item["features"] = self._rep_tensor(index)
        else:
            item["tokens"] = self._rep_tensor(index)
            # Historical GT-centred Exploration-4 mapping. For cached tokens the crop
            # center is the manifest point and each token spans fov/16 source px.
            left = int(np.floor(float(r["x"]) - self.fov / 2 + .5))
            top = int(np.floor(float(r["y"]) - self.fov / 2 + .5))
            source_per_patch = self.fov / 16
            item["u"] = torch.tensor((float(r["x"]) - left) / source_per_patch, dtype=torch.float32)
            item["v"] = torch.tensor((float(r["y"]) - top) / source_per_patch, dtype=torch.float32)
        return item
