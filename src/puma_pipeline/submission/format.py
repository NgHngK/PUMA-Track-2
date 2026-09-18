from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import NUCLEUS_CLASSES, TISSUE_OUTPUT_LABELS

FINAL_DTYPE=np.dtype([("x","f4"),("y","f4"),("class_id","i2"),("confidence","f4")])


def predictions_to_puma_polygons(predictions: np.ndarray, half_size: float = 1.0) -> dict[str, Any]:
    rows = np.asarray(predictions)
    required = {"x", "y", "class_id", "confidence"}
    names = set(rows.dtype.names or ())
    if not required <= names:
        raise ValueError(f"Prediction dtype is missing fields: {sorted(required - names)}")
    if not np.isfinite(float(half_size)) or half_size <= 0:
        raise ValueError("half_size must be a positive finite value.")
    polygons = []
    for i, row in enumerate(rows):
        x = float(row["x"]); y = float(row["y"]); class_id = int(row["class_id"]); confidence = float(row["confidence"])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Prediction {i} contains non-finite coordinates.")
        if not 0 <= class_id < len(NUCLEUS_CLASSES):
            raise ValueError(f"Prediction {i} has invalid class_id={class_id}.")
        if not np.isfinite(confidence):
            raise ValueError(f"Prediction {i} has non-finite confidence.")
        half = float(half_size)
        polygons.append({
            "name": NUCLEUS_CLASSES[class_id],
            "seed_point": [x, y, 0.5],
            "path_points": [[x-half,y-half,0.5],[x+half,y-half,0.5],[x+half,y+half,0.5],[x-half,y+half,0.5]],
            "sub_type": "", "groups": [], "probability": float(np.clip(confidence, 0, 1)),
        })
    payload = {"type":"Multiple polygons","polygons":polygons,"version":{"major":1,"minor":0}}
    validate_puma_nuclei_json(payload)
    return payload


def validate_puma_nuclei_json(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != {"type", "polygons", "version"}:
        raise ValueError("Unexpected nuclei JSON top-level fields.")
    if payload["type"] != "Multiple polygons" or payload["version"] != {"major":1,"minor":0}:
        raise ValueError("Invalid PUMA Multiple polygons header.")
    if not isinstance(payload["polygons"], list):
        raise ValueError("Nuclei polygons must be a list.")
    fields = {"name","seed_point","path_points","sub_type","groups","probability"}
    for i,p in enumerate(payload["polygons"]):
        if not isinstance(p, dict) or set(p) != fields:
            raise ValueError(f"Polygon {i} schema mismatch.")
        if p["name"] not in NUCLEUS_CLASSES:
            raise ValueError(f"Polygon {i} unknown class.")
        if not isinstance(p["sub_type"], str) or not isinstance(p["groups"], list):
            raise ValueError(f"Polygon {i} has invalid subtype/groups fields.")
        seed=np.asarray(p["seed_point"],float); points=np.asarray(p["path_points"],float)
        if seed.shape!=(3,) or points.shape!=(4,3) or not np.isfinite(seed).all() or not np.isfinite(points).all():
            raise ValueError(f"Polygon {i} invalid/non-finite geometry.")
        if not np.isclose(seed[2],.5) or not np.allclose(points[:,2],.5):
            raise ValueError(f"Polygon {i} invalid z-coordinate.")
        if not np.allclose(points[:,:2].mean(0),seed[:2],atol=1e-5):
            raise ValueError(f"Polygon {i} seed/centroid mismatch.")
        probability=float(p["probability"])
        if not np.isfinite(probability) or not 0<=probability<=1:
            raise ValueError(f"Polygon {i} invalid probability.")


def write_puma_json(path:Path,predictions:np.ndarray)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(predictions_to_puma_polygons(predictions),allow_nan=False,separators=(",",":")),encoding="utf-8"); tmp.replace(path)


def validate_tissue_mask(mask:np.ndarray)->None:
    arr=np.asarray(mask)
    if arr.shape!=(1024,1024):raise ValueError(f"Tissue mask must be 1024x1024, got {arr.shape}.")
    if arr.dtype!=np.uint8:raise ValueError("Tissue mask must be uint8.")
    allowed=set(TISSUE_OUTPUT_LABELS.values()); observed=set(np.unique(arr).astype(int).tolist())
    if not observed<=allowed:raise ValueError(f"Invalid tissue labels: {sorted(observed-allowed)}")


def _rational_value(value:Any)->float:
    if isinstance(value,tuple) and len(value)==2:
        return float(value[0])/float(value[1])
    numerator=getattr(value,"numerator",None); denominator=getattr(value,"denominator",None)
    if numerator is not None and denominator is not None:
        return float(numerator)/float(denominator)
    return float(value)


def validate_tissue_tiff(path:Path)->np.ndarray:
    """Check the tissue mask and TIFF metadata."""
    import tifffile

    path=Path(path)
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages)!=1:
            raise ValueError(f"Tissue TIFF must contain exactly one page, got {len(tif.pages)}.")
        page=tif.pages[0]
        mask=np.asarray(page.asarray())
        validate_tissue_mask(mask)
        required=("XResolution","YResolution","MinSampleValue","MaxSampleValue")
        missing=[name for name in required if name not in page.tags]
        if missing:
            raise ValueError(f"Tissue TIFF is missing required PUMA tags: {missing}")
        x_resolution=_rational_value(page.tags["XResolution"].value)
        y_resolution=_rational_value(page.tags["YResolution"].value)
        if not np.isclose(x_resolution,300.0) or not np.isclose(y_resolution,300.0):
            raise ValueError(
                f"Tissue TIFF resolution must be 300x300, got {x_resolution}x{y_resolution}."
            )
        minimum=int(page.tags["MinSampleValue"].value)
        maximum=int(page.tags["MaxSampleValue"].value)
        expected_max=int(mask.max(initial=0))
        if minimum!=1:
            raise ValueError(f"Tissue TIFF MinSampleValue must be 1, got {minimum}.")
        if maximum!=expected_max:
            raise ValueError(
                f"Tissue TIFF MaxSampleValue must equal max foreground label {expected_max}, got {maximum}."
            )
    return mask


def write_tissue_tiff(path:Path,mask:np.ndarray)->None:
    import tifffile
    validate_tissue_mask(mask); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    max_value=int(np.asarray(mask).max(initial=0))
    tifffile.imwrite(
        tmp,
        mask,
        photometric="minisblack",
        compression="deflate",
        metadata=None,
        resolution=(300,300),
        resolutionunit="INCH",
        extratags=[("MinSampleValue","I",1,1),("MaxSampleValue","I",1,max_value)],
    )
    tmp.replace(path)
    validate_tissue_tiff(path)
