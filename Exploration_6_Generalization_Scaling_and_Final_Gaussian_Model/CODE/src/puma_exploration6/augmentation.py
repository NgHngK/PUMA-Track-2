from __future__ import annotations

import math
import numpy as np
import torch
from PIL import Image

from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, UNI2_GRID_SIZE
from .config import AugmentationConfig

MEAN = torch.tensor(IMAGENET_MEAN,dtype=torch.float32)[:,None,None]
STD = torch.tensor(IMAGENET_STD,dtype=torch.float32)[:,None,None]
_FIXED_HE_BASIS=np.array([[.650,.704,.286],[.072,.990,.105]],dtype=np.float32)
_FIXED_HE_PINV=np.linalg.pinv(_FIXED_HE_BASIS).astype(np.float32)


def centered_crop_with_token_coordinate(image: Image.Image, center_x: float, center_y: float, fov: int) -> tuple[Image.Image,float,float]:
    """Historical crop convention and exact Exploration-4 token coordinate.

    Crop origin: floor(center - fov/2 + .5). Missing context is padded white.
    Returned u,v are continuous coordinates in the 16x16 spatial-token grid.
    """
    if fov < 16 or fov % 2: raise ValueError("fov must be even and >=16")
    w,h = image.size
    if not (0 <= center_x < w and 0 <= center_y < h):
        raise ValueError("crop center outside ROI")
    left = int(np.floor(center_x - fov/2 + .5)); top = int(np.floor(center_y - fov/2 + .5))
    box = (max(left,0),max(top,0),min(left+fov,w),min(top+fov,h))
    out = Image.new("RGB",(fov,fov),(255,255,255)); out.paste(image.crop(box),(box[0]-left,box[1]-top))
    source_per_patch = fov / UNI2_GRID_SIZE
    u = (center_x-left)/source_per_patch; v = (center_y-top)/source_per_patch
    if not (0 <= u <= UNI2_GRID_SIZE and 0 <= v <= UNI2_GRID_SIZE):
        raise AssertionError("token coordinate outside crop")
    return out,float(u),float(v)


def _transform_token_coordinate(u: float, v: float, k_rot90: int, hflip: bool, vflip: bool, grid: float = 16.0) -> tuple[float,float]:
    k = int(k_rot90)%4
    if k == 1: u,v = v,grid-u
    elif k == 2: u,v = grid-u,grid-v
    elif k == 3: u,v = grid-v,u
    if hflip: u = grid-u
    if vflip: v = grid-v
    eps = np.finfo(np.float32).eps
    return float(np.clip(u,0,grid-eps)),float(np.clip(v,0,grid-eps))


def sample_coordinate_jitter(cfg: AugmentationConfig, generator: torch.Generator | None = None) -> tuple[float,float]:
    if cfg.coordinate_jitter_sigma <= 0: return 0.0,0.0
    g = generator
    delta = torch.randn(2,generator=g,dtype=torch.float32)*cfg.coordinate_jitter_sigma
    delta = delta.clamp(-cfg.coordinate_jitter_clip,cfg.coordinate_jitter_clip)
    return float(delta[0]),float(delta[1])


def image_to_tensor(crop: Image.Image, u: float, v: float, cfg: AugmentationConfig,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor,float,float,dict]:
    crop = crop.resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BICUBIC)
    arr = np.asarray(crop,dtype=np.float32)/255.0
    k=0; hf=False; vf=False
    if cfg.geometric:
        k = int(torch.randint(0,4,(),generator=generator).item())
        hf = bool(torch.rand((),generator=generator).item()<0.5)
        vf = bool(torch.rand((),generator=generator).item()<0.5)
        arr = np.rot90(arr,k)
        if hf: arr = arr[:,::-1,:]
        if vf: arr = arr[::-1,:,:]
        u,v = _transform_token_coordinate(u,v,k,hf,vf)
    stain_scales = []
    if cfg.stain_mode == "rgb_od":
        r = torch.rand(3,generator=generator,dtype=torch.float32).numpy();stain_scales = (1 + cfg.stain_strength*(2*r-1)).astype(np.float32)
        arr = np.exp(np.log(np.clip(arr,1/255,1))*stain_scales);arr = np.clip(arr,0,1)
    elif cfg.stain_mode == "fixed_hed":
        # Fixed H/E optical-density basis; this is not fitted Macenko/Vahadane normalization.
        # It perturbs stain concentrations while preserving the spatial morphology.
        basis=_FIXED_HE_BASIS
        od=-np.log(np.clip(arr,1/255,1));flat=od.reshape(-1,3);conc=flat@_FIXED_HE_PINV;base_recon=conc@basis;residual=flat-base_recon
        r=torch.rand(2,generator=generator,dtype=torch.float32).numpy();stain_scales=(1+cfg.stain_strength*(2*r-1)).astype(np.float32)
        recon=(conc*stain_scales)@basis+residual;arr=np.exp(-recon).reshape(arr.shape);arr=np.clip(arr,0,1)
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2,0,1)
    t = (t-MEAN)/STD
    if t.shape != (3,IMAGE_SIZE,IMAGE_SIZE) or not torch.isfinite(t).all():
        raise FloatingPointError("invalid image tensor")
    return t,float(u),float(v),{"rot90":k,"hflip":hf,"vflip":vf,"stain_scales":np.asarray(stain_scales,dtype=np.float32).tolist()}
