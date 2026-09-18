from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from ..config import PumaConfig
from ..constants import CANDIDATE_DTYPE
from ..stage1.data import prepare_full_roi_batch
from ..stage1.decode import decode_stage1_batch, suppress_stage1
from ..stage1.model import build_stage1_model
from ..submission.format import FINAL_DTYPE
from ..tissue import TISSUE_TRAINING_CONTRACT
from ..tissue.model import TissueHead
from .cache import _detection_from_reference, _roi_relative
from .calibration import load_calibration
from .crops import reflect_crop
from .encoder import Uni2FeatureExtractor
from .model import BioContextRefine
from .risk import apply_risk, risk_features
from .spatial import build_knn_graph, spatial_features_from_reference
from .tissue_context import tissue_context_from_probability_map


def _load_state(path:Path)->dict:
    return torch.load(path,map_location="cpu",weights_only=False)


class PumaPipeline:
    def __init__(self,config:PumaConfig,stage1_checkpoint:Path,stage2_checkpoint:Path,tissue_checkpoint:Path,risk_path:Path) -> None:
        if not torch.cuda.is_available():raise RuntimeError("PUMA deployment requires CUDA.")
        self.config=config; self.device=torch.device("cuda")
        self.amp=torch.bfloat16 if config.use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16
        stage1_payload = _load_state(stage1_checkpoint)
        stage2_payload = _load_state(stage2_checkpoint)
        tissue_payload = _load_state(tissue_checkpoint)
        if stage1_payload.get("config_fingerprint") != config.stage1_fingerprint:
            raise RuntimeError("Stage-1 checkpoint does not match the deployment config fingerprint.")
        if stage1_payload.get("architecture") != "full_roi_1024_native_stride1_point_detector":
            raise RuntimeError("Stage-1 checkpoint is not the native-1024 PUMA detector.")
        if stage2_payload.get("config_fingerprint") != config.fingerprint:
            raise RuntimeError("Stage-2 checkpoint does not match the deployment config fingerprint.")
        if stage2_payload.get("architecture") != "BioContextRefine":
            raise RuntimeError("Stage-2 checkpoint has an incompatible architecture.")
        if tissue_payload.get("config_fingerprint") != config.tissue_fingerprint:
            raise RuntimeError("Tissue checkpoint does not match the deployment config fingerprint.")
        if tissue_payload.get("training_contract") != TISSUE_TRAINING_CONTRACT:
            raise RuntimeError("Tissue checkpoint used an incompatible Stage-1 preprocessing contract; retrain tissue with the fixed PUMA code.")
        if tissue_payload.get("architecture") != "stage1_fpn_tissue_head":
            raise RuntimeError("Tissue checkpoint has an incompatible architecture.")
        self.stage1=build_stage1_model(config); self.stage1.load_state_dict(stage1_payload["model_state"],strict=True); self.stage1.to(self.device).eval()
        self.stage2=BioContextRefine(config); self.stage2.load_state_dict(stage2_payload["model_state"],strict=True); self.stage2.to(self.device).eval()
        self.tissue=TissueHead(base=config.tissue_base_channels); self.tissue.load_state_dict(tissue_payload["model_state"],strict=True); self.tissue.to(self.device).eval()
        self.uni2=Uni2FeatureExtractor(config.path("uni2_checkpoint"),self.device,self.amp == torch.bfloat16)
        self.risk_state,self.threshold=load_calibration(risk_path,expected_fingerprint=config.fingerprint)

    @torch.inference_mode()
    def _stage1_and_tissue(self,image:np.ndarray):
        tensor=torch.from_numpy(np.asarray(image,np.uint8).copy()).permute(2,0,1)[None]
        prepared=prepare_full_roi_batch(tensor,self.device)
        with torch.autocast("cuda",dtype=self.amp): outputs=self.stage1(prepared,return_fpn=True); tissue_prob=self.tissue(outputs["fpn_feature"],(1024,1024)).softmax(1)[0].float().cpu().numpy()
        prediction=decode_stage1_batch(self.stage1,outputs,minimum_threshold=self.config.stage1_deployment_threshold,local_max_radius=self.config.stage1_deployment_local_max_radius)[0]
        prediction=suppress_stage1(prediction,self.config.stage1_deployment_threshold,self.config.stage1_deployment_suppression_radius,self.config.image_size)
        return prediction,tissue_prob

    def _candidate_array(self,prediction)->np.ndarray:
        n=len(prediction.coordinates); c=np.zeros(n,dtype=CANDIDATE_DTYPE)
        if not n:return c
        xy=prediction.coordinates
        c["source_id"]=np.arange(n); c["roi_index"]=0; c["x"]=xy[:,0]; c["y"]=xy[:,1]; c["heatmap_score"]=prediction.heatmap_score; c["quality"]=prediction.quality; c["uncertainty"]=prediction.uncertainty; c["peak_sharpness"]=prediction.peak_sharpness; c["stage1_prior"]=prediction.embedding; c["class_id"]=-1; c["kind"]=0; c["fold"]=-1; c["gt_global_index"]=-1
        return c

    @torch.inference_mode()
    def predict(self,image:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        image=np.asarray(image,np.uint8)
        if image.shape!=(1024,1024,3):raise ValueError(f"PUMA input must be RGB 1024x1024, got {image.shape}.")
        s1,tissue_prob=self._stage1_and_tissue(image); candidates=self._candidate_array(s1); n=len(candidates); tissue_mask=tissue_prob.argmax(0).astype(np.uint8)
        if n==0:return np.empty(0,dtype=FINAL_DTYPE),tissue_mask
        xy=np.column_stack((candidates["x"],candidates["y"])).astype(np.float32); spatial=spatial_features_from_reference(xy,xy,self.config.image_size,exclude_self=True); detection=_detection_from_reference(candidates,candidates,self.config.image_size,exclude_self=True)
        image_float = image.astype(np.float32) / 255.0
        od = -np.log(np.clip(image_float, 1.0 / 255.0, 1.0))
        hmap = 0.650 * od[..., 0] + 0.704 * od[..., 1] + 0.286 * od[..., 2]
        hmean = uniform_filter(hmap, size=15, mode="reflect")
        xx = np.clip(np.floor(xy[:, 0]).astype(int), 0, self.config.image_size - 1)
        yy = np.clip(np.floor(xy[:, 1]).astype(int), 0, self.config.image_size - 1)
        h = hmean[yy, xx].astype(np.float32)
        relative = _roi_relative(h, h)
        tissue=tissue_context_from_probability_map(tissue_prob,xy,pool_radius=self.config.tissue_context_pool_radius)
        bio=np.asarray([reflect_crop(image,float(x),float(y),self.config.biomask_crop_size).transpose(2,0,1) for x,y in xy],np.uint8)
        appearances=[]
        for name in self.config.stage2_views:
            size={"V2":64,"V3":128,"V4":256}[name]; crops=np.asarray([reflect_crop(image,float(x),float(y),size).transpose(2,0,1) for x,y in xy],np.uint8); appearances.append(self.uni2.extract(torch.from_numpy(crops),self.config.deployment_uni2_micro_batch_size).numpy())
        appearance=np.stack(appearances,axis=1)
        chunks=[]
        for start in range(0,n,self.config.deployment_stage2_inference_batch_size):
            stop=min(n,start+self.config.deployment_stage2_inference_batch_size)
            def t(a,dtype=None):return torch.as_tensor(a[start:stop],device=self.device,dtype=dtype)
            with torch.autocast("cuda",dtype=self.amp): local=self.stage2.forward_local(t(appearance),t(bio),t(relative),t(spatial),t(tissue),t(candidates["stage1_prior"]),t(detection),context_dropout=False)
            chunks.append({k:v for k,v in local.items() if k in ("embedding","local_logits","validity_logit","mask_confidence_logit")})
        local={k:torch.cat([part[k] for part in chunks],0) for k in chunks[0]}; ni,nd=build_knn_graph(xy,self.config.graph_neighbor_k,self.config.graph_radius_cap)
        with torch.autocast("cuda",dtype=self.amp): out=self.stage2.refine(local,torch.from_numpy(xy).to(self.device),torch.from_numpy(ni).to(self.device),torch.from_numpy(nd).to(self.device),probability_temperature=self.config.graph_probability_temperature)
        prob=out["final_logits"].softmax(1).float().cpu().numpy(); local_prob=out["local_logits"].softmax(1).float().cpu().numpy(); predicted=prob.argmax(1); validity=out["validity_logit"].sigmoid().float().cpu().numpy(); mask_conf=out["mask_confidence_logit"].sigmoid().float().cpu().numpy(); graph=out["graph_gate"].float().cpu().numpy()
        rfeat=risk_features(prob,validity,candidates["heatmap_score"],mask_conf,graph,local_prob); acceptance=apply_risk(self.risk_state,rfeat); keep=acceptance>=self.threshold
        final=np.empty(int(keep.sum()),dtype=FINAL_DTYPE); final["x"]=xy[keep,0]; final["y"]=xy[keep,1]; final["class_id"]=predicted[keep]; final["confidence"]=acceptance[keep]
        return final,tissue_mask
