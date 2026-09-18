from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from ..config import PumaConfig
from ..constants import CLASS_TO_ID, NUCLEUS_CLASSES, ROI_SIZE
from ..utils.provenance import directory_metadata_signature
from .annotations import load_polygons, official_vertex_centroid


CENTROID_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("class_id", "i2"),
    ]
)

MANIFEST_DTYPE = np.dtype(
    [
        ("roi_id", "U160"),
        ("image_file", "U512"),
        ("nuclei_geojson_file", "U512"),
        ("tissue_geojson_file", "U512"),
        ("sample_type", "U24"),
    ]
)


def _normalized_stem(value: str | Path) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"(?i)([_-]?(nuclei|nucleus|tissue|annotations?|annotation))+$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def _sample_type(stem: str) -> str:
    token = stem.lower()
    if "metastatic" in token or "metastasis" in token:
        return "metastatic"
    if "primary" in token:
        return "primary"
    return "unknown"


def _find_unique_annotation(image_path: Path, directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    target = _normalized_stem(image_path)
    exact: list[Path] = []
    fuzzy: list[Path] = []
    for path in sorted(directory.rglob("*.geojson")):
        stem = _normalized_stem(path)
        if stem == target:
            exact.append(path)
        elif stem.endswith(target) or target.endswith(stem):
            fuzzy.append(path)
    candidates = exact if exact else fuzzy
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not uniquely match {image_path.name} to an annotation in {directory}; "
            f"matches={[str(path) for path in candidates[:10]]}"
        )
    return candidates[0]


def _load_rgb1024(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a color TIFF, got shape {image.shape} for {path}.")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape != (ROI_SIZE, ROI_SIZE, 3):
        raise ValueError(f"Expected exactly 1024x1024 RGB, got {image.shape} for {path}.")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        image = np.rint(np.asarray(image, dtype=np.float32) * (255.0 / maximum))
    else:
        image = np.asarray(image, dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"TIFF contains non-finite pixels: {path}")
        minimum = float(image.min())
        maximum = float(image.max())
        if minimum < 0.0:
            raise ValueError(f"Floating-point TIFF contains negative RGB values: {path}")
        if maximum <= 1.5:
            image = image * 255.0
        elif maximum > 255.0:
            raise ValueError(
                f"Floating-point TIFF range [{minimum}, {maximum}] is neither normalized [0,1] nor uint8-like [0,255]: {path}"
            )
    return np.ascontiguousarray(np.clip(image, 0.0, 255.0).astype(np.uint8))


def _fold_objective(
    assignment: np.ndarray,
    class_counts: np.ndarray,
    sample_codes: np.ndarray,
    folds: int,
) -> float:
    """Score a fold split. Lower is better."""
    total_class = class_counts.sum(axis=0).astype(np.float64)
    present = total_class > 0
    class_weight = np.zeros_like(total_class)
    class_weight[present] = 1.0 / np.sqrt(total_class[present])
    if np.any(present):
        class_weight[present] /= max(float(class_weight[present].mean()), 1e-12)
    target_class = total_class / float(folds)
    sample_totals = np.bincount(sample_codes).astype(np.float64)
    target_sample = sample_totals / float(folds)
    target_size = len(assignment) / float(folds)
    score = 0.0
    presence = class_counts > 0
    for fold in range(folds):
        ids = np.flatnonzero(assignment == fold)
        observed = class_counts[ids].sum(axis=0) if len(ids) else np.zeros_like(total_class)
        score += float(np.sum(class_weight * ((observed - target_class) / np.maximum(target_class, 1.0)) ** 2))
        sobs = np.bincount(sample_codes[ids], minlength=len(sample_totals)).astype(np.float64)
        score += 0.7 * float(np.mean(((sobs - target_sample) / np.maximum(target_sample, 1.0)) ** 2))
    sizes = np.bincount(assignment, minlength=folds).astype(np.float64)
    score += 8.0 * float(np.mean(((sizes - target_size) / max(target_size, 1.0)) ** 2))
    global_presence = presence.sum(axis=0)
    for class_id, count in enumerate(global_presence):
        if int(count) >= folds:
            missing = sum(not np.any(presence[assignment == f, class_id]) for f in range(folds))
            score += 1000.0 * float(missing)
    return score


def _validate_fold_assignment(assignment: np.ndarray, class_counts: np.ndarray, folds: int) -> None:
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.shape != (len(class_counts),):
        raise RuntimeError(f"Fold assignment shape mismatch: {assignment.shape}.")
    if np.any((assignment < 0) | (assignment >= folds)):
        raise RuntimeError(f"Fold assignment contains invalid IDs: {np.unique(assignment).tolist()}.")
    sizes = np.bincount(assignment, minlength=folds)
    if np.any(sizes == 0):
        raise RuntimeError(f"Fold assignment contains an empty fold: {sizes.tolist()}.")
    if int(sizes.max() - sizes.min()) > 1:
        raise RuntimeError(f"Fold ROI counts are imbalanced: {sizes.tolist()}.")
    presence = class_counts > 0
    for class_id, count in enumerate(presence.sum(axis=0)):
        if int(count) >= folds:
            missing = [f for f in range(folds) if not np.any(presence[assignment == f, class_id])]
            if missing:
                raise RuntimeError(f"Class {class_id} occurs in {int(count)} ROIs but is missing from folds {missing}.")


def _balanced_folds(class_counts: np.ndarray, sample_types: list[str], folds: int, seed: int) -> np.ndarray:
    """Create deterministic folds with balanced size and class coverage."""
    class_counts = np.asarray(class_counts, dtype=np.int64)
    n = len(class_counts)
    if n < folds:
        raise ValueError("Need at least one ROI per fold.")
    names = {name: i for i, name in enumerate(sorted(set(sample_types)))}
    sample_codes = np.asarray([names[x] for x in sample_types], dtype=np.int64)
    totals = class_counts.sum(axis=0).astype(np.float64)
    present_total = totals > 0
    class_weight = np.zeros_like(totals)
    class_weight[present_total] = 1.0 / np.sqrt(totals[present_total])
    if np.any(present_total):
        class_weight[present_total] /= max(float(class_weight[present_total].mean()), 1e-12)
    target_class = totals / float(folds)
    sample_totals = np.bincount(sample_codes, minlength=max(1, len(names))).astype(np.float64)
    target_sample = sample_totals / float(folds)
    target_size = n / float(folds)
    presence = class_counts > 0
    global_presence = presence.sum(axis=0)
    rarity = np.zeros_like(totals)
    rarity[present_total] = 1.0 / np.sqrt(totals[present_total])
    priority = presence.astype(np.float64) @ rarity + 0.01 * np.log1p(class_counts.sum(axis=1))

    best = None
    best_score = float("inf")
    for restart in range(64):
        rng = np.random.default_rng(seed + 104729 * restart)
        base, rem = divmod(n, folds)
        capacity = np.full(folds, base, dtype=np.int64)
        if rem:
            capacity[rng.permutation(folds)[:rem]] += 1
        order = np.argsort(-(priority + rng.uniform(0, 1e-6, n)))
        assignment = np.full(n, -1, dtype=np.int64)
        fold_sizes = np.zeros(folds, dtype=np.int64)
        fold_class = np.zeros((folds, class_counts.shape[1]), dtype=np.float64)
        fold_sample = np.zeros((folds, len(sample_totals)), dtype=np.float64)
        fold_presence = np.zeros((folds, class_counts.shape[1]), dtype=np.int64)

        # Start each fold with one ROI.
        for f, roi in enumerate(order[:folds]):
            assignment[roi] = f
            fold_sizes[f] = 1
            fold_class[f] += class_counts[roi]
            fold_sample[f, sample_codes[roi]] += 1
            fold_presence[f] += presence[roi]

        for roi in order[folds:]:
            candidates = np.flatnonzero(fold_sizes < capacity)
            costs = np.full(folds, np.inf, dtype=np.float64)
            for f in candidates:
                oldc = fold_class[f]
                newc = oldc + class_counts[roi]
                denom = np.maximum(target_class, 1.0)
                class_delta = np.sum(class_weight * (((newc-target_class)/denom)**2 - ((oldc-target_class)/denom)**2))
                olds = fold_sample[f]
                news = olds.copy(); news[sample_codes[roi]] += 1
                sden = np.maximum(target_sample, 1.0)
                sample_delta = 0.7 * np.mean(((news-target_sample)/sden)**2 - ((olds-target_sample)/sden)**2)
                old_size = ((fold_sizes[f]-target_size)/max(target_size,1.0))**2
                new_size = ((fold_sizes[f]+1-target_size)/max(target_size,1.0))**2
                size_delta = 8.0 * (new_size-old_size) / folds
                # Prefer folds that do not contain this class yet.
                new_cover = presence[roi] & (fold_presence[f] == 0)
                cover_weight = np.where(global_presence >= folds, 35.0, 4.0)
                coverage_reward = float(np.sum(cover_weight[new_cover]))
                costs[f] = class_delta + sample_delta + size_delta - coverage_reward
            minimum = float(np.min(costs[candidates]))
            choices = candidates[np.isclose(costs[candidates], minimum, rtol=0.0, atol=1e-12)]
            if len(choices) > 1:
                sizes = fold_sizes[choices]
                choices = choices[sizes == sizes.min()]
            chosen = int(rng.choice(choices))
            assignment[roi] = chosen
            fold_sizes[chosen] += 1
            fold_class[chosen] += class_counts[roi]
            fold_sample[chosen, sample_codes[roi]] += 1
            fold_presence[chosen] += presence[roi]

        try:
            _validate_fold_assignment(assignment, class_counts, folds)
        except RuntimeError:
            continue
        score = _fold_objective(assignment, class_counts, sample_codes, folds)
        if score < best_score:
            best_score = score
            best = assignment.copy()

    if best is None:
        raise RuntimeError("Could not construct a valid balanced fold assignment after 64 deterministic restarts.")
    return best.astype(np.int8)


def _class_counts_from_store(centroids: np.ndarray, n_rois: int) -> np.ndarray:
    counts = np.zeros((n_rois, len(NUCLEUS_CLASSES)), dtype=np.int64)
    if len(centroids):
        np.add.at(counts, (np.asarray(centroids["roi_index"], int), np.asarray(centroids["class_id"], int)), 1)
    return counts


def _fold_summary(folds_arr: np.ndarray, class_counts: np.ndarray, sample_types: list[str], k: int) -> dict[str, Any]:
    return {
        "fold_roi_counts": np.bincount(folds_arr.astype(int), minlength=k).astype(int).tolist(),
        "fold_class_counts": {str(f): class_counts[folds_arr == f].sum(axis=0).astype(int).tolist() for f in range(k)},
        "sample_type_fold_counts": {name: [int(np.sum((np.asarray(sample_types) == name) & (folds_arr == f))) for f in range(k)] for name in sorted(set(sample_types))},
    }


def _repair_fold_artifacts(out_dir: Path, current: dict[str, Any], config: PumaConfig, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = np.load(out_dir / "puma_roi_manifest.npy", mmap_mode="r", allow_pickle=False)
    centroids = np.load(out_dir / "puma_nuclei_centroids.npy", mmap_mode="r", allow_pickle=False)
    class_counts = _class_counts_from_store(centroids, len(manifest))
    sample_types = np.asarray(manifest["sample_type"]).astype(str).tolist() if "sample_type" in (manifest.dtype.names or ()) else ["unknown"] * len(manifest)
    folds_arr = _balanced_folds(class_counts, sample_types, config.number_of_folds, config.seed)
    _validate_fold_assignment(folds_arr, class_counts, config.number_of_folds)
    np.save(out_dir / "puma_fold_assignments.npy", folds_arr, allow_pickle=False)
    payload = {**current, **expected, "roi_count": int(len(manifest)), "nuclei_count": int(len(centroids)), "class_counts": class_counts.sum(axis=0).astype(int).tolist(), **_fold_summary(folds_arr, class_counts, sample_types, config.number_of_folds), "fold_policy": "capacity-constrained global balancing v3", "fold_repaired": True}
    manifest_path = out_dir / "preprocessing_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    return payload


def _source_signature(config: PumaConfig) -> dict[str, Any]:
    return {
        "images": directory_metadata_signature(config.path("image_dir"), ("*.tif", "*.tiff")),
        "nuclei": directory_metadata_signature(config.path("nuclei_geojson_dir"), ("*.geojson",)),
        "tissue": directory_metadata_signature(config.path("tissue_geojson_dir"), ("*.geojson",)),
    }


def preprocess_dataset(config: PumaConfig, *, force: bool = False) -> dict[str, Any]:
    """Build the memory-mapped training files from the raw dataset."""
    out_dir = config.path("artifact_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "preprocessing_manifest.json"
    required = [
        out_dir / "puma_rgb_images.npy",
        out_dir / "puma_roi_manifest.npy",
        out_dir / "puma_nuclei_centroids.npy",
        out_dir / "puma_roi_centroid_offsets.npy",
        out_dir / "puma_fold_assignments.npy",
    ]
    source_signature = _source_signature(config)
    expected = {
        "artifact_schema": "puma",
        "preprocessing_revision": config.fold_assignment_method,
        "image_size": config.image_size,
        "number_of_folds": config.number_of_folds,
        "seed": config.seed,
        "source_signature": source_signature,
    }
    if not force and manifest_path.is_file() and all(path.is_file() for path in required):
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable = ("artifact_schema", "image_size", "number_of_folds", "seed", "source_signature")
        if all(current.get(key) == expected.get(key) for key in stable):
            manifest_arr = np.load(out_dir / "puma_roi_manifest.npy", mmap_mode="r", allow_pickle=False)
            centroids_arr = np.load(out_dir / "puma_nuclei_centroids.npy", mmap_mode="r", allow_pickle=False)
            cc = _class_counts_from_store(centroids_arr, len(manifest_arr))
            folds_arr = np.load(out_dir / "puma_fold_assignments.npy", mmap_mode="r", allow_pickle=False)
            valid = True
            try:
                _validate_fold_assignment(folds_arr, cc, config.number_of_folds)
            except RuntimeError:
                valid = False
            if current.get("preprocessing_revision") == config.fold_assignment_method and valid:
                return current
            print("Repairing stale/invalid fold assignment without rebuilding the RGB store.")
            return _repair_fold_artifacts(out_dir, current, config, expected)
        raise RuntimeError("Existing preprocessing artifacts come from different raw data/config; rerun with force=True.")

    image_dir = config.path("image_dir")
    image_paths = sorted({*image_dir.rglob("*.tif"), *image_dir.rglob("*.tiff")})
    if not image_paths:
        raise FileNotFoundError(f"No TIFF ROIs found in {image_dir}.")

    nuclei_dir = config.path("nuclei_geojson_dir")
    tissue_dir = config.path("tissue_geojson_dir")
    records: list[dict[str, Any]] = []
    all_centroids: list[np.ndarray] = []
    class_counts: list[np.ndarray] = []
    sample_types: list[str] = []

    images_tmp = out_dir / "puma_rgb_images.npy.tmp"
    images = np.lib.format.open_memmap(
        images_tmp,
        mode="w+",
        dtype=np.uint8,
        shape=(len(image_paths), ROI_SIZE, ROI_SIZE, 3),
    )
    offsets = np.zeros(len(image_paths) + 1, dtype=np.int64)
    try:
        for roi_index, image_path in enumerate(image_paths):
            nuclei_path = _find_unique_annotation(image_path, nuclei_dir)
            tissue_path = _find_unique_annotation(image_path, tissue_dir)
            images[roi_index] = _load_rgb1024(image_path)
            polygons = load_polygons(nuclei_path, tissue=False)
            rows = np.empty(len(polygons), dtype=CENTROID_DTYPE)
            counts = np.zeros(len(NUCLEUS_CLASSES), dtype=np.int64)
            for local_index, (label, points) in enumerate(polygons):
                if label not in CLASS_TO_ID:
                    raise ValueError(f"Unknown nucleus label {label!r} in {nuclei_path}.")
                centroid = official_vertex_centroid(points)
                x, y = float(centroid[0]), float(centroid[1])
                if not (0.0 <= x < ROI_SIZE and 0.0 <= y < ROI_SIZE):
                    raise ValueError(f"Nucleus centroid {(x, y)} is outside 1024 ROI in {nuclei_path}.")
                class_id = CLASS_TO_ID[label]
                rows[local_index] = (roi_index, x, y, class_id)
                counts[class_id] += 1
            all_centroids.append(rows)
            class_counts.append(counts)
            sample_type = _sample_type(image_path.stem)
            sample_types.append(sample_type)
            records.append(
                {
                    "roi_id": image_path.stem,
                    "image_file": str(image_path.resolve()),
                    "nuclei_geojson_file": str(nuclei_path.resolve()),
                    "tissue_geojson_file": str(tissue_path.resolve()),
                    "sample_type": sample_type,
                }
            )
            offsets[roi_index + 1] = offsets[roi_index] + len(rows)
        images.flush()
    finally:
        del images

    centroids = np.concatenate(all_centroids) if all_centroids else np.empty(0, dtype=CENTROID_DTYPE)
    class_count_array = np.stack(class_counts, axis=0)
    folds = _balanced_folds(class_count_array, sample_types, config.number_of_folds, config.seed)
    _validate_fold_assignment(folds, class_count_array, config.number_of_folds)
    manifest = np.empty(len(records), dtype=MANIFEST_DTYPE)
    for index, record in enumerate(records):
        manifest[index] = tuple(record[name] for name in MANIFEST_DTYPE.names or ())

    np.save(out_dir / "puma_roi_manifest.npy", manifest, allow_pickle=False)
    np.save(out_dir / "puma_nuclei_centroids.npy", centroids, allow_pickle=False)
    np.save(out_dir / "puma_roi_centroid_offsets.npy", offsets, allow_pickle=False)
    np.save(out_dir / "puma_fold_assignments.npy", folds, allow_pickle=False)
    images_final = out_dir / "puma_rgb_images.npy"
    images_tmp.replace(images_final)

    fold_class_counts = {
        str(fold): class_count_array[folds == fold].sum(axis=0).astype(int).tolist()
        for fold in range(config.number_of_folds)
    }
    sample_summary = {
        name: [int(np.sum((np.asarray(sample_types) == name) & (folds == fold))) for fold in range(config.number_of_folds)]
        for name in sorted(set(sample_types))
    }
    payload = {
        **expected,
        "roi_count": len(records),
        "nuclei_count": int(len(centroids)),
        "class_counts": class_count_array.sum(axis=0).astype(int).tolist(),
        "fold_roi_counts": np.bincount(folds.astype(int), minlength=config.number_of_folds).astype(int).tolist(),
        "fold_class_counts": fold_class_counts,
        "sample_type_fold_counts": sample_summary,
        "fold_policy": "capacity-constrained global balancing v3",
        "patient_grouping": "not inferred from filenames; patient-level grouping requires explicit patient/case metadata",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
