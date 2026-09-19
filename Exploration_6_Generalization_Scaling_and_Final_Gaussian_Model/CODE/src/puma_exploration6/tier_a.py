from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import laplace

SCHEMA = {
    "stain": ["H_mean","H_std","H_p10","H_p50","H_p90"],
    "gradient_texture": ["H_gradient_mean","H_gradient_std","H_abs_laplacian_mean","H_entropy32"],
    "ring": ["H_center_minus_ring","gray_center_minus_ring","H_ring_std","central_valid_fraction","ring_valid_fraction"],
    "roi_relative": ["H_robust_z","H_ROI_percentile"],
}
NAMES = tuple(sum(SCHEMA.values(), []))


def roi_channels(rgb: Image.Image | np.ndarray) -> dict[str,np.ndarray | float]:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("RGB image expected")
    x = x[:,:,:3]
    h = np.clip((-np.log(np.clip(x, 1/255, 1)) @ np.array([.650,.704,.286], dtype=np.float32)) / 2, 0, 2)
    gy,gx = np.gradient(h); grad = np.sqrt(gx*gx + gy*gy + 1e-8)
    return {
        "h": h, "g": grad, "lap": np.abs(laplace(h, mode="reflect")), "gray": x.mean(2),
        "median": float(np.median(h)), "iqr": float(np.percentile(h,75)-np.percentile(h,25)),
        "sorted_h": np.sort(h.ravel()),
    }


def point_features(ch: dict, x: float, y: float) -> np.ndarray:
    yy,xx = np.mgrid[-24:25,-24:25]; d2 = xx*xx + yy*yy
    center = d2 <= 12**2; ring = (d2 >= 16**2) & (d2 <= 24**2)
    ix = xx + int(np.floor(x+.5)); iy = yy + int(np.floor(y+.5))
    height,width = ch["h"].shape
    valid = (ix>=0)&(ix<width)&(iy>=0)&(iy<height)
    c = center & valid; r = ring & valid
    if not c.any() or not r.any():
        raise ValueError("insufficient image support for Tier-A")
    def vals(key: str, mask: np.ndarray) -> np.ndarray:
        return ch[key][iy[mask], ix[mask]]
    h = vals("h",c); hr = vals("h",r); g = vals("g",c); mean = float(h.mean())
    hist = np.histogram(h,bins=32,range=(0,2))[0].astype(float)
    if hist.sum() <= 0: raise ValueError("empty Tier-A histogram")
    hist /= hist.sum(); hist = hist[hist>0]
    v = [mean,h.std(),*np.percentile(h,[10,50,90]),g.mean(),g.std(),vals("lap",c).mean(),-(hist*np.log(hist)).sum(),
         mean-hr.mean(),vals("gray",c).mean()-vals("gray",r).mean(),hr.std(),c.sum()/center.sum(),r.sum()/ring.sum(),
         (mean-ch["median"])/max(ch["iqr"],1e-4),np.searchsorted(ch["sorted_h"],mean,side="right")/len(ch["sorted_h"])]
    out = np.asarray(v,dtype=np.float32)
    if out.shape != (16,) or not np.isfinite(out).all():
        raise ValueError("invalid Tier-A features")
    return out


def compute_tier_a(rows: list[dict], cache_rois: int = 2) -> np.ndarray:
    cache: OrderedDict[str,tuple[Image.Image,dict]] = OrderedDict()
    result = np.empty((len(rows),16), dtype=np.float32)
    for i,row in enumerate(rows):
        path = row["image"]
        if path not in cache:
            with Image.open(path) as im: rgb = im.convert("RGB").copy()
            cache[path] = (rgb, roi_channels(rgb))
            while len(cache) > cache_rois: cache.popitem(last=False)
        cache.move_to_end(path)
        result[i] = point_features(cache[path][1], float(row["x"]), float(row["y"]))
    return result


class TierANormalizer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = None if mean is None else np.asarray(mean,dtype=np.float32)
        self.std = None if std is None else np.asarray(std,dtype=np.float32)

    def fit(self, x: np.ndarray) -> "TierANormalizer":
        x = np.asarray(x,dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 16 or not np.isfinite(x).all():
            raise ValueError("Tier-A fit expects finite Nx16")
        self.mean = x.mean(0); self.std = np.maximum(x.std(0),1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None: raise RuntimeError("TierANormalizer not fitted")
        z = (np.asarray(x,dtype=np.float32)-self.mean)/self.std
        if not np.isfinite(z).all(): raise FloatingPointError("nonfinite normalized Tier-A")
        return z.astype(np.float32,copy=False)

    def state_dict(self) -> dict:
        if self.mean is None or self.std is None: raise RuntimeError("TierANormalizer not fitted")
        return {"mean":self.mean.tolist(),"std":self.std.tolist(),"names":list(NAMES)}
