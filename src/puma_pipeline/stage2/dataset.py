from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..constants import REJECT_CLASS_ID, DETECTION_DIM, UNI2_POOLED_DIM, CANDIDATE_DTYPE


class Stage2Cache:
    def __init__(self, cache_dir: Path, *, expected_fingerprint: str | None = None) -> None:
        manifest_path = cache_dir / "cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_schema") != "puma":
            raise RuntimeError("Stage-2 cache manifest has an unsupported format version.")
        if expected_fingerprint is not None and manifest.get("config_fingerprint") != expected_fingerprint:
            raise RuntimeError("Stage-2 cache fingerprint does not match the requested config.")
        if int(manifest.get("detection_dim", -1)) != DETECTION_DIM:
            raise RuntimeError("Stage-2 cache detection feature contract is incompatible.")

        self.candidates=np.load(cache_dir/"candidates.npy",mmap_mode="r",allow_pickle=False)
        self.appearance=np.load(cache_dir/"appearance.npy",mmap_mode="r",allow_pickle=False)
        self.rgb=np.load(cache_dir/"biomask_rgb.npy",mmap_mode="r",allow_pickle=False)
        self.mask=np.load(cache_dir/"biomask_targets_packed.npy",mmap_mode="r",allow_pickle=False)
        self.spatial=np.load(cache_dir/"spatial.npy",mmap_mode="r",allow_pickle=False)
        self.detection=np.load(cache_dir/"detection.npy",mmap_mode="r",allow_pickle=False)
        self.relative=np.load(cache_dir/"roi_relative_h.npy",mmap_mode="r",allow_pickle=False)
        self.tissue=np.load(cache_dir/"tissue.npy",mmap_mode="r",allow_pickle=False)
        if self.candidates.dtype != CANDIDATE_DTYPE:
            raise RuntimeError("Stage-2 candidate dtype is incompatible with the PUMA contract.")
        if self.appearance.dtype != np.float16:
            raise RuntimeError(f"Stage-2 appearance cache must be float16, got {self.appearance.dtype}.")
        if self.rgb.dtype != np.uint8 or self.mask.dtype != np.uint8:
            raise RuntimeError("Stage-2 BioMask RGB/packed-target caches must be uint8.")
        for name, array in (("spatial",self.spatial),("detection",self.detection),("relative",self.relative),("tissue",self.tissue)):
            if array.dtype != np.float32:
                raise RuntimeError(f"Stage-2 {name} cache must be float32, got {array.dtype}.")

        self.variants=self.appearance.shape[0]
        self.crop_size=self.rgb.shape[-1]
        rows=len(self.candidates)
        expected_rows=int(manifest.get("rows",-1))
        if expected_rows != rows:
            raise RuntimeError(f"Stage-2 cache row mismatch: manifest={expected_rows}, candidates={rows}.")
        manifest_variants=int(manifest.get("variants",-1))
        if manifest_variants != self.variants:
            raise RuntimeError(f"Stage-2 cache variant mismatch: manifest={manifest_variants}, appearance={self.variants}.")
        views=manifest.get("views")
        if not isinstance(views,list) or not views:
            raise RuntimeError("Stage-2 cache manifest is missing its view contract.")
        if self.appearance.ndim != 4 or self.appearance.shape[1:] != (rows,len(views),UNI2_POOLED_DIM):
            raise RuntimeError(f"Stage-2 appearance cache shape is incompatible: {self.appearance.shape}.")
        if self.rgb.ndim != 5 or self.rgb.shape[:2] != (self.variants,rows) or self.rgb.shape[2] != 3 or self.rgb.shape[3] != self.rgb.shape[4]:
            raise RuntimeError("Stage-2 BioMask RGB cache shape is incompatible with candidates.")
        expected_packed=(self.crop_size*self.crop_size+7)//8
        if self.mask.shape != (self.variants,rows,expected_packed):
            raise RuntimeError(f"Stage-2 BioMask target cache shape is incompatible: {self.mask.shape}.")
        for name,array,dim in (("spatial",self.spatial,6),("detection",self.detection,DETECTION_DIM),("relative",self.relative,2),("tissue",self.tissue,18)):
            if array.shape != (rows,dim):
                raise RuntimeError(f"Stage-2 {name} cache has shape {array.shape}, expected {(rows,dim)}.")

    def batch(self, ids: np.ndarray, variant: int | np.ndarray=0) -> dict[str,torch.Tensor]:
        ids=np.asarray(ids,dtype=np.int64)
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= len(self.candidates))):
            raise IndexError("Stage-2 cache ids are invalid.")
        if np.isscalar(variant): variants=np.full(len(ids),int(variant),np.int64)
        else: variants=np.asarray(variant,dtype=np.int64)
        if variants.shape != ids.shape or np.any((variants < 0) | (variants >= self.variants)):
            raise IndexError("Stage-2 cache variant indices are invalid.")
        app=np.asarray([self.appearance[v,i] for v,i in zip(variants,ids)],np.float16)
        rgb=np.asarray([self.rgb[v,i] for v,i in zip(variants,ids)],np.uint8)
        packed=np.asarray([self.mask[v,i] for v,i in zip(variants,ids)],np.uint8)
        bits=np.unpackbits(packed,axis=1,bitorder="little")[:,:self.crop_size*self.crop_size].reshape(-1,self.crop_size,self.crop_size)
        c=self.candidates[ids]
        xy=np.column_stack((np.asarray(c["x"],np.float32),np.asarray(c["y"],np.float32)))
        return {
            "ids":torch.from_numpy(ids),"appearance":torch.from_numpy(app),"rgb":torch.from_numpy(rgb),"mask_target":torch.from_numpy(bits.astype(np.float32)),
            "xy":torch.from_numpy(xy),"spatial":torch.from_numpy(np.asarray(self.spatial[ids]).copy()),"detection":torch.from_numpy(np.asarray(self.detection[ids]).copy()),
            "relative":torch.from_numpy(np.asarray(self.relative[ids]).copy()),"tissue":torch.from_numpy(np.asarray(self.tissue[ids]).copy()),
            "prior":torch.from_numpy(np.asarray(c["stage1_prior"],np.float16).copy()),"class_id":torch.from_numpy(np.asarray(c["class_id"],np.int64).copy()),
            "roi_index":torch.from_numpy(np.asarray(c["roi_index"],np.int64).copy()),"kind":torch.from_numpy(np.asarray(c["kind"],np.int64).copy()),"fold":torch.from_numpy(np.asarray(c["fold"],np.int64).copy()),
        }


class CandidateDataset(Dataset):
    def __init__(self, cache: Stage2Cache, ids: np.ndarray) -> None:
        self.cache=cache; self.ids=np.asarray(ids,dtype=np.int64)
    def __len__(self)->int:return len(self.ids)
    def __getitem__(self,index:int)->int:return int(self.ids[index])


def roi_class_sampler(cache: Stage2Cache, ids: np.ndarray, balanced_fraction: float, samples: int, seed: int) -> WeightedRandomSampler:
    ids=np.asarray(ids,dtype=np.int64); c=cache.candidates[ids]; labels=np.asarray(c["class_id"],np.int64); rois=np.asarray(c["roi_index"],np.int64)
    counts:dict[tuple[int,int],int]={}
    for roi,label in zip(rois,labels): counts[(int(roi),int(label))]=counts.get((int(roi),int(label)),0)+1
    natural=np.ones(len(ids),np.float64); balanced=np.asarray([1.0/max(counts[(int(r),int(y))],1) for r,y in zip(rois,labels)],np.float64)
    class_count=np.bincount(labels[labels<REJECT_CLASS_ID],minlength=REJECT_CLASS_ID).astype(np.float64)
    class_scale=np.ones(REJECT_CLASS_ID+1,np.float64); class_scale[:REJECT_CLASS_ID]=1.0/np.sqrt(np.maximum(class_count,1)); class_scale[:REJECT_CLASS_ID]/=class_scale[:REJECT_CLASS_ID].mean(); class_scale[REJECT_CLASS_ID]=0.6
    balanced*=class_scale[np.clip(labels,0,REJECT_CLASS_ID)]
    balanced/=balanced.mean(); weights=(1-balanced_fraction)*natural+balanced_fraction*balanced
    generator=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=int(samples),replacement=True,generator=generator)
