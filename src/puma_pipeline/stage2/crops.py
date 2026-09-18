from __future__ import annotations

import math

import numpy as np


def reflect_crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    half = int(size) // 2
    left = int(math.floor(float(x))) - half
    top = int(math.floor(float(y))) - half
    x_indices = np.arange(left, left + size, dtype=np.int64)
    y_indices = np.arange(top, top + size, dtype=np.int64)

    def reflect(values: np.ndarray, length: int) -> np.ndarray:
        if length <= 1:
            return np.zeros_like(values)
        period = 2 * length - 2
        folded = np.mod(values, period)
        return np.where(folded < length, folded, period - folded)

    x_indices = reflect(x_indices, image.shape[1])
    y_indices = reflect(y_indices, image.shape[0])
    return np.asarray(image[y_indices[:, None], x_indices[None, :], :], dtype=np.uint8).copy()
