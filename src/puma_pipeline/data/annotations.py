from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_polygon

from ..constants import NUCLEUS_CLASSES, ROI_SIZE, TISSUE_OUTPUT_LABELS


def _token(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_")


NUCLEUS_ALIASES = {
    "tumor": "nuclei_tumor", "tumour": "nuclei_tumor", "nuclei_tumor": "nuclei_tumor",
    "lymphocyte": "nuclei_lymphocyte", "nuclei_lymphocyte": "nuclei_lymphocyte",
    "plasma": "nuclei_plasma_cell", "plasma_cell": "nuclei_plasma_cell", "nuclei_plasma_cell": "nuclei_plasma_cell",
    "histiocyte": "nuclei_histiocyte", "histocyte": "nuclei_histiocyte", "nuclei_histiocyte": "nuclei_histiocyte",
    "melanophage": "nuclei_melanophage", "melaphonage": "nuclei_melanophage", "nuclei_melanophage": "nuclei_melanophage",
    "neutrophil": "nuclei_neutrophil", "nuclei_neutrophil": "nuclei_neutrophil",
    "stroma": "nuclei_stroma", "stromal": "nuclei_stroma", "nuclei_stroma": "nuclei_stroma",
    "endothelium": "nuclei_endothelium", "endothelial": "nuclei_endothelium", "nuclei_endothelium": "nuclei_endothelium",
    "epithelium": "nuclei_epithelium", "epithelial": "nuclei_epithelium", "nuclei_epithelium": "nuclei_epithelium",
    "apoptosis": "nuclei_apoptosis", "apoptotic": "nuclei_apoptosis", "nuclei_apoptosis": "nuclei_apoptosis",
}
TISSUE_ALIASES = {
    "background": "background", "stroma": "stroma", "tumoral_stroma": "stroma",
    "blood_vessel": "blood_vessel", "bloodvessel": "blood_vessel", "vessel": "blood_vessel",
    "tumor": "tumor", "tumour": "tumor", "epidermis": "epidermis", "necrosis": "necrosis", "necrotic": "necrosis",
}


def _label_from_mapping(mapping: dict[str, Any], aliases: dict[str, str]) -> str | None:
    values: list[Any] = []
    for key in ("name", "label", "class", "classification", "type", "sub_type"):
        value = mapping.get(key)
        if isinstance(value, dict):
            values.extend(value.values())
        else:
            values.append(value)
    for value in values:
        t = _token(value)
        if t in aliases:
            return aliases[t]
        for alias, canonical in aliases.items():
            if t.endswith(alias) or t.startswith(alias):
                return canonical
    return None


def _rings_from_geometry(geometry: dict[str, Any]) -> list[np.ndarray]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon" and coordinates:
        return [np.asarray(coordinates[0], dtype=np.float32)]
    if kind == "MultiPolygon" and coordinates:
        return [np.asarray(poly[0], dtype=np.float32) for poly in coordinates if poly]
    return []


def load_polygons(path: Path, *, tissue: bool = False) -> list[tuple[str, np.ndarray]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = TISSUE_ALIASES if tissue else NUCLEUS_ALIASES
    output: list[tuple[str, np.ndarray]] = []
    if payload.get("type") == "FeatureCollection":
        for feature in payload.get("features", []):
            properties = feature.get("properties") or {}
            label = _label_from_mapping(properties, aliases)
            if label is None:
                label = _label_from_mapping(feature, aliases)
            if label is None:
                continue
            for ring in _rings_from_geometry(feature.get("geometry") or {}):
                if len(ring) >= 3:
                    output.append((label, ring[:, :2]))
        return output
    if payload.get("type") == "Multiple polygons":
        for polygon in payload.get("polygons", []):
            label = _label_from_mapping(polygon, aliases)
            points = np.asarray(polygon.get("path_points", []), dtype=np.float32)
            if label is not None and points.ndim == 2 and points.shape[0] >= 3:
                output.append((label, points[:, :2]))
        return output
    raise ValueError(f"Unsupported annotation JSON format: {path}")


def official_vertex_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points.mean(axis=0)


def match_nucleus_polygons_to_store(polygons: list[tuple[str, np.ndarray]], centroids: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    names = centroids.dtype.names or ()
    class_field = "class_id" if "class_id" in names else "label"
    for class_id, class_name in enumerate(NUCLEUS_CLASSES):
        gt_idx = np.flatnonzero(np.asarray(centroids[class_field], dtype=np.int64) == class_id)
        poly = [ring for label, ring in polygons if label == class_name]
        if not len(gt_idx) or not poly:
            continue
        gt_xy = np.column_stack((centroids["x"][gt_idx], centroids["y"][gt_idx])).astype(np.float32)
        pc = np.asarray([official_vertex_centroid(ring) for ring in poly], dtype=np.float32)
        tree = cKDTree(gt_xy)
        distance, local = tree.query(pc, k=1)
        order = np.argsort(distance)
        used: set[int] = set()
        for p_idx in order:
            local_idx = int(local[p_idx])
            if local_idx in used or float(distance[p_idx]) > 8.0:
                continue
            used.add(local_idx)
            result[int(gt_idx[local_idx])] = poly[p_idx]
    return result


def rasterize_polygon_crop(points: np.ndarray, center_xy: tuple[float, float], crop_size: int) -> np.ndarray:
    half = crop_size / 2.0
    x0 = float(center_xy[0]) - half
    y0 = float(center_xy[1]) - half
    local_x = np.asarray(points[:, 0], dtype=np.float32) - x0
    local_y = np.asarray(points[:, 1], dtype=np.float32) - y0
    rr, cc = draw_polygon(local_y, local_x, shape=(crop_size, crop_size))
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def rasterize_tissue_map(path: Path, image_size: int = ROI_SIZE) -> np.ndarray:
    output = np.zeros((image_size, image_size), dtype=np.uint8)
    for label, points in load_polygons(path, tissue=True):
        value = TISSUE_OUTPUT_LABELS[label]
        rr, cc = draw_polygon(points[:, 1], points[:, 0], shape=output.shape)
        output[rr, cc] = np.uint8(value)
    return output


def manifest_path(manifest_row: np.void, field_candidates: tuple[str, ...], root: Path) -> Path:
    names = manifest_row.dtype.names or ()
    for field in field_candidates:
        if field in names:
            value = str(manifest_row[field])
            path = Path(value)
            return path if path.is_absolute() else root / path
    raise KeyError(f"Manifest does not contain any of {field_candidates}.")


def resolve_annotation_path(manifest_row: np.void, directory: Path, *, tissue: bool) -> Path:
    names = manifest_row.dtype.names or ()
    fields = ("tissue_geojson_file", "tissue_file") if tissue else ("geojson_file", "nuclei_geojson_file", "annotation_file")
    for field in fields:
        if field in names:
            candidate = Path(str(manifest_row[field]))
            if not candidate.is_absolute():
                direct = directory / candidate.name
                if direct.is_file():
                    return direct
                candidate = directory.parent.parent / candidate
            if candidate.is_file():
                return candidate
    image_stem = None
    for field in ("image_file", "image_path", "roi_id", "name"):
        if field in names:
            image_stem = Path(str(manifest_row[field])).stem
            break
    if image_stem is None:
        raise KeyError("Manifest needs an image_file/roi_id-like field for annotation discovery.")
    normalized = re.sub(r"(?i)([_-]?(nuclei|nucleus|tissue|annotations?|annotation))+$", "", image_stem)
    candidates = []
    for path in directory.rglob("*.geojson"):
        stem = re.sub(r"(?i)([_-]?(nuclei|nucleus|tissue|annotations?|annotation))+$", "", path.stem)
        if stem == normalized or stem.endswith(normalized) or normalized.endswith(stem):
            candidates.append(path)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve annotation for {image_stem}: {candidates}")
    return candidates[0]
