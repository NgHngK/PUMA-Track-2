from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import tifffile

from ..config import PumaConfig
from ..stage2.infer import PumaPipeline
from .format import write_puma_json, write_tissue_tiff
from ..utils.runtime import configure_runtime

INPUT_DIRS=(Path("/input/images/melanoma-whole-slide-image"),Path("/input/images/melanoma-wsi"))
NUCLEI_OUTPUT=Path("/output/melanoma-10-class-nuclei-segmentation.json")
TISSUE_OUTPUT_DIR=Path("/output/images/melanoma-tissue-mask-segmentation")


def find_input_tiff()->Path:
    files=[]
    for directory in INPUT_DIRS:
        if directory.is_dir():files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".tif",".tiff"})
    files=sorted(set(p.resolve() for p in files))
    if len(files)!=1:raise RuntimeError(f"Expected exactly one PUMA TIFF, found {len(files)}: {files}")
    return files[0]


def read_rgb(path:Path)->np.ndarray:
    image=np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim==3 and image.shape[0] in (3,4) and image.shape[-1] not in (3,4):image=np.moveaxis(image,0,-1)
    if image.ndim!=3 or image.shape[-1]<3:raise ValueError(f"Expected RGB TIFF, got {image.shape}")
    image=image[...,:3]
    if image.dtype==np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype,np.integer):
        image=np.rint(image.astype(np.float32)*(255.0/float(np.iinfo(image.dtype).max)))
    else:
        image=np.asarray(image,dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"TIFF contains non-finite pixels: {path}")
        minimum=float(image.min()); maximum=float(image.max())
        if minimum < 0.0:
            raise ValueError(f"Floating-point TIFF contains negative RGB values: {path}")
        if maximum<=1.5:
            image=image*255.0
        elif maximum>255.0:
            raise ValueError(
                f"Floating-point TIFF range [{minimum}, {maximum}] is neither normalized [0,1] nor uint8-like [0,255]: {path}"
            )
    return np.ascontiguousarray(np.clip(image,0.0,255.0).astype(np.uint8))


def main()->None:
    root=Path(os.environ.get("PUMA_ROOT","/opt/algorithm")).resolve(); config=PumaConfig.load(Path(os.environ.get("PUMA_CONFIG",root/"deployment_config.json"))); configure_runtime(config.seed,config.use_tf32,config.deterministic)
    input_path=find_input_tiff(); image=read_rgb(input_path)
    pipeline=PumaPipeline(config,Path(os.environ.get("PUMA_STAGE1",root/"models/stage1_final_ema.pt")),Path(os.environ.get("PUMA_STAGE2",root/"models/stage2_final_ema.pt")),Path(os.environ.get("PUMA_TISSUE",root/"models/tissue_final_ema.pt")),Path(os.environ.get("PUMA_RISK",root/"models/risk_calibration.npz")))
    nuclei,tissue=pipeline.predict(image); write_puma_json(NUCLEI_OUTPUT,nuclei); write_tissue_tiff(TISSUE_OUTPUT_DIR/f"{input_path.stem}.tif",tissue)
    print(f"Processed {input_path.name}: retained {len(nuclei)} nuclei")

if __name__=="__main__":main()
