from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
import torch

from ..config import PumaConfig
from ..constants import DETECTION_DIM, VIEW_SIZES
from ..data.annotations import load_polygons, match_nucleus_polygons_to_store, rasterize_polygon_crop, resolve_annotation_path
from ..store import PumaArtifactStore
from .candidates import build_candidates
from .crops import reflect_crop
from .encoder import Uni2FeatureExtractor
from .spatial import spatial_features_from_reference
from .tissue_context import tissue_context_from_probability_map
from ..utils.provenance import file_signature


def _open_memmap(path: Path, dtype: np.dtype | type, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _roi_relative(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    if len(reference_values) < 3:
        return np.column_stack((np.zeros(len(values)), np.full(len(values), 0.5))).astype(np.float32)
    median = float(np.median(reference_values)); q1, q3 = np.percentile(reference_values, [25, 75]); scale = max(float(q3 - q1) / 1.349, 1e-3)
    robust_z = np.clip((values - median) / scale, -6.0, 6.0)
    sorted_ref = np.sort(reference_values)
    percentile = np.searchsorted(sorted_ref, values, side="right") / float(len(sorted_ref))
    return np.column_stack((robust_z, percentile)).astype(np.float32)


def _detection_from_reference(
    query: np.ndarray,
    reference: np.ndarray,
    image_size: int,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    out = np.zeros((len(query), DETECTION_DIM), np.float32)
    if not len(query):
        return out
    qxy = np.column_stack((query["x"], query["y"])).astype(np.float32)
    rxy = (
        np.column_stack((reference["x"], reference["y"])).astype(np.float32)
        if len(reference)
        else np.empty((0, 2), np.float32)
    )
    identity_aligned = (
        len(query) == len(reference)
        and np.array_equal(qxy, rxy)
    )
    if exclude_self and not identity_aligned:
        raise ValueError("exclude_self=True requires query/reference rows with identical coordinates and order.")

    if len(reference):
        tree = cKDTree(rxy)
        # Remove the same candidate but keep real duplicates at the same position.
        query_k = min(len(reference), 2 if exclude_self else 1)
        dist, nn_index = tree.query(qxy, k=max(1, query_k), workers=1)
        if query_k == 1:
            dist = dist[:, None]
            nn_index = nn_index[:, None]
        nearest = np.full(len(query), float(image_size), np.float32)
        nearest_score = np.zeros(len(query), np.float32)
        for i in range(len(query)):
            candidates_i = np.arange(dist.shape[1])
            if exclude_self:
                candidates_i = candidates_i[nn_index[i] != i]
            if len(candidates_i):
                pick = int(candidates_i[0])
                nearest[i] = float(dist[i, pick])
                nearest_score[i] = float(reference["heatmap_score"][nn_index[i, pick]])
        counts = np.asarray([len(v) for v in tree.query_ball_point(qxy, 6.0)], np.float32)
        if exclude_self:
            counts = np.maximum(counts - 1.0, 0.0)
        dup = counts
    else:
        nearest = np.full(len(query), image_size, np.float32)
        nearest_score = np.zeros(len(query), np.float32)
        dup = np.zeros(len(query), np.float32)

    border = np.minimum.reduce((qxy[:, 0], qxy[:, 1], image_size - qxy[:, 0], image_size - qxy[:, 1]))
    score = np.asarray(query["heatmap_score"], np.float32)
    out[:] = np.column_stack(
        (
            np.clip(score, 0, 1),
            np.clip(query["quality"], 0, 1),
            np.log1p(np.clip(query["uncertainty"], 0, 100)),
            np.clip(query["peak_sharpness"], -1, 1),
            np.clip(border / 64, 0, 1),
            np.log1p(np.clip(nearest, 0, image_size * 2)),
            np.log1p(dup),
            np.clip(score - nearest_score, -1, 1),
        )
    )
    return out


def _variant_xy(candidates: np.ndarray, variant: int, seed: int, image_size: int = 1024) -> np.ndarray:
    xy=np.column_stack((candidates["x"],candidates["y"])).astype(np.float32)
    if variant == 0: return xy
    rng=np.random.default_rng(seed + 1009*variant)
    jitter=rng.normal(0,1.75,size=xy.shape).astype(np.float32)
    jitter=np.clip(jitter,-4,4)
    positive=np.asarray(candidates["gt_global_index"]>=0)
    jitter[~positive] *= 0.5
    maximum = np.nextafter(np.float32(image_size), np.float32(-np.inf))
    return np.clip(xy + jitter, 0.0, maximum)


def _augment_cached_crop(crop: np.ndarray, row_id: int, variant: int, seed: int, spatial: bool = True) -> np.ndarray:
    if variant == 0:
        return crop
    rng=np.random.default_rng(seed + 7919*variant + 104729*int(row_id))
    out=np.asarray(crop,np.uint8)
    if spatial:
        code=int(rng.integers(0,8)); k=code%4; out=np.rot90(out,k=k)
        if code>=4: out=np.fliplr(out)
    x=out.astype(np.float32)/255.0
    gain=rng.uniform(0.92,1.08,size=(1,1,3)).astype(np.float32); bias=rng.uniform(-0.025,0.025,size=(1,1,3)).astype(np.float32); gamma=float(rng.uniform(0.92,1.08))
    x=np.clip(x*gain+bias,0,1); x=np.power(x,gamma)
    return np.ascontiguousarray(np.clip(np.rint(x*255),0,255).astype(np.uint8))


def _augment_mask(mask: np.ndarray, row_id: int, variant: int, seed: int) -> np.ndarray:
    if variant == 0: return mask
    rng=np.random.default_rng(seed + 7919*variant + 104729*int(row_id)); code=int(rng.integers(0,8)); out=np.rot90(mask,k=code%4)
    if code>=4: out=np.fliplr(out)
    return np.ascontiguousarray(out)


def build_stage2_cache(config: PumaConfig, *, force: bool=False, include_tissue: bool=True) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 cache extraction requires a CUDA GPU.")
    cache_dir=config.path("cache_dir"); manifest_path=cache_dir/"cache_manifest.json"
    required=[cache_dir/"appearance.npy",cache_dir/"biomask_rgb.npy",cache_dir/"biomask_targets_packed.npy",cache_dir/"spatial.npy",cache_dir/"detection.npy",cache_dir/"roi_relative_h.npy",cache_dir/"tissue.npy"]
    candidate_path=build_candidates(config,force=force)
    candidate_manifest_path=cache_dir/"candidates_manifest.json"
    candidate_signature=file_signature(candidate_manifest_path)
    # Hash the candidate data so the cache matches the exact input file.
    candidate_file_signature=file_signature(candidate_path, content_hash=True)
    tissue_prob_path=config.path("tissue_output_dir")/"tissue_oof_probabilities.npy"
    if include_tissue and not tissue_prob_path.is_file(): raise FileNotFoundError(tissue_prob_path)
    tissue_signature=file_signature(tissue_prob_path) if include_tissue else None
    expected_manifest={
        "artifact_schema":"puma",
        "config_fingerprint":config.fingerprint,
        "candidate_manifest_signature":candidate_signature,
        "candidate_file_signature":candidate_file_signature,
        "tissue_probability_signature":tissue_signature,
        "tissue_reference":"oof_predictions" if include_tissue else "disabled",
        "detection_dim": DETECTION_DIM,
    }
    if not force and manifest_path.is_file() and all(p.is_file() for p in required):
        current=json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(current.get(key)==value for key,value in expected_manifest.items()):
            return manifest_path
        print("Rebuilding stale Stage-2 cache automatically."); force=True
    candidates=np.load(candidate_path,mmap_mode="r",allow_pickle=False)
    store=PumaArtifactStore.open(config.path("artifact_dir")); rows=len(candidates); variants=config.stage2_cache_variants; views=len(config.stage2_views)
    cache_dir.mkdir(parents=True,exist_ok=True)
    appearance=_open_memmap(cache_dir/"appearance.npy",np.float16,(variants,rows,views,config.stage2_pooled_feature_dim))
    rgb_cache=_open_memmap(cache_dir/"biomask_rgb.npy",np.uint8,(variants,rows,3,config.biomask_crop_size,config.biomask_crop_size))
    packed=_open_memmap(cache_dir/"biomask_targets_packed.npy",np.uint8,(variants,rows,(config.biomask_crop_size**2+7)//8))
    spatial=_open_memmap(cache_dir/"spatial.npy",np.float32,(rows,6)); detection=_open_memmap(cache_dir/"detection.npy",np.float32,(rows,DETECTION_DIM)); relative=_open_memmap(cache_dir/"roi_relative_h.npy",np.float32,(rows,2)); tissue=_open_memmap(cache_dir/"tissue.npy",np.float32,(rows,18))
    tissue_prob=np.load(tissue_prob_path,mmap_mode="r",allow_pickle=False) if include_tissue else None
    if include_tissue and tissue_prob is None:
        raise FileNotFoundError(tissue_prob_path)
    if tissue_prob is not None:
        expected_tissue_shape = (len(store.images), 6, config.image_size, config.image_size)
        if tissue_prob.shape != expected_tissue_shape:
            raise RuntimeError(
                f"Tissue OOF probability shape is incompatible: {tissue_prob.shape}; "
                f"expected {expected_tissue_shape}."
            )
        if tissue_prob.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise RuntimeError(f"Tissue OOF probabilities must be float16/float32, got {tissue_prob.dtype}.")
    nuclei_dir=config.path("nuclei_geojson_dir")
    for roi in range(len(store.images)):
        idx=np.flatnonzero(np.asarray(candidates["roi_index"])==roi); ref_idx=idx[np.asarray(candidates["kind"][idx])==0]
        if not len(idx): continue
        query=candidates[idx]; ref=candidates[ref_idx]
        qxy=np.column_stack((query["x"],query["y"])).astype(np.float32); rxy=np.column_stack((ref["x"],ref["y"])).astype(np.float32) if len(ref) else np.empty((0,2),np.float32)
        clean_idx=idx[np.asarray(candidates["kind"][idx])!=0]
        if len(ref_idx):
            spatial[ref_idx]=spatial_features_from_reference(rxy,rxy,config.image_size,exclude_self=True)
            detection[ref_idx]=_detection_from_reference(candidates[ref_idx],candidates[ref_idx],config.image_size,exclude_self=True)
        if len(clean_idx):
            clean=candidates[clean_idx]; clean_xy=np.column_stack((clean["x"],clean["y"])).astype(np.float32)
            spatial[clean_idx]=spatial_features_from_reference(clean_xy,rxy,config.image_size,exclude_self=False)
            detection[clean_idx]=_detection_from_reference(clean,ref,config.image_size,exclude_self=False)
        image=np.asarray(store.images[roi]); image_float=image.astype(np.float32)/255.0; od=-np.log(np.clip(image_float,1.0/255.0,1.0)); hmap=0.650*od[...,0]+0.704*od[...,1]+0.286*od[...,2]; hmean=uniform_filter(hmap,size=15,mode="reflect")
        qxx=np.clip(np.floor(qxy[:,0]).astype(int),0,config.image_size-1); qyy=np.clip(np.floor(qxy[:,1]).astype(int),0,config.image_size-1); qh=hmean[qyy,qxx].astype(np.float32)
        if len(rxy):
            rxx=np.clip(np.floor(rxy[:,0]).astype(int),0,config.image_size-1); ryy=np.clip(np.floor(rxy[:,1]).astype(int),0,config.image_size-1); rh=hmean[ryy,rxx].astype(np.float32)
        else: rh=np.empty(0,np.float32)
        relative[idx]=_roi_relative(qh,rh)
        if tissue_prob is not None: tissue[idx]=tissue_context_from_probability_map(np.asarray(tissue_prob[roi],np.float32),qxy,pool_radius=config.tissue_context_pool_radius)
        else: tissue[idx]=0.0
    extractor=Uni2FeatureExtractor(config.path("uni2_checkpoint"),torch.device("cuda"),config.use_bfloat16)
    polygon_cache: dict[int,dict[int,np.ndarray]]={}
    for roi in range(len(store.images)):
        path=resolve_annotation_path(store.manifest[roi],nuclei_dir,tissue=False); polygons=load_polygons(path,tissue=False); polygon_cache[roi]=match_nucleus_polygons_to_store(polygons,store.roi_centroids(roi))
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for variant in range(variants):
            xy_all=_variant_xy(candidates,variant,config.seed,config.image_size)

            def build_one(row_id: int):
                row=candidates[row_id]; roi=int(row["roi_index"]); x,y=map(float,xy_all[row_id]); image=np.asarray(store.images[roi])
                bio_crop=_augment_cached_crop(reflect_crop(image,x,y,config.biomask_crop_size),row_id,variant,config.seed)
                bio=bio_crop.transpose(2,0,1)
                views=[_augment_cached_crop(reflect_crop(image,x,y,int(VIEW_SIZES[name])),row_id,variant,config.seed).transpose(2,0,1) for name in config.stage2_views]
                g=int(row["gt_global_index"])
                if g>=0:
                    local=g-int(store.offsets[roi]); ring=polygon_cache[roi].get(local)
                    if ring is None:
                        raise RuntimeError(f"Missing GT polygon linkage for ROI {roi}, local GT {local}; refusing silent BioMask target corruption.")
                    target=rasterize_polygon_crop(ring,(x,y),config.biomask_crop_size)
                else:
                    target=np.zeros((config.biomask_crop_size,config.biomask_crop_size),np.uint8)
                target=_augment_mask(target,row_id,variant,config.seed)
                return bio, views, np.packbits(target.reshape(-1),bitorder="little")

            for start in range(0,rows,config.stage2_cache_batch_size):
                stop=min(rows,start+config.stage2_cache_batch_size); ids=np.arange(start,stop)
                built=list(pool.map(build_one, map(int,ids)))
                rgb_cache[variant,start:stop]=np.asarray([item[0] for item in built],np.uint8)
                packed[variant,start:stop]=np.asarray([item[2] for item in built],np.uint8)
                for j in range(len(config.stage2_views)):
                    batch=np.asarray([item[1][j] for item in built],np.uint8)
                    appearance[variant,start:stop,j]=extractor.extract(torch.from_numpy(batch),config.stage2_uni2_micro_batch_size).numpy()
    for arr in (appearance,rgb_cache,packed,spatial,detection,relative,tissue): arr.flush()
    payload={**expected_manifest,"rows":rows,"variants":variants,"views":list(config.stage2_views),"biology_features_from":"predicted_biomask_only","spatial_reference":"stage1_oof_detections_only"}
    manifest_path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); return manifest_path
