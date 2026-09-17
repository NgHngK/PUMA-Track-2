import json,csv,collections,hashlib,ast
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';A=R/'00_reference/biomask_audit';O=R/'02_cache/biomask/prompt3_gt_diagnostic';Q=A/'qc';Q.mkdir(exist_ok=True)
rs=list(csv.DictReader((A/'quality_per_nucleus.csv').open()));numeric=['dice','iou','area_relative_error','centroid_error','occupancy','quality_confidence','presence_confidence']+[pre+k for pre in ['gt_','pred_'] for k in ['area','major','minor','eccentricity','perimeter','circularity','cx','cy']]
for r in rs:
    for k in numeric:r[k]=float(r[k])
    r['index']=int(r['index']);r['fold']=int(r['fold'])
split=list(csv.DictReader((R/'01_sample_definition/sample_manifest.csv').open()));cut=float(np.median([r['gt_area'] for r in rs if split[r['index']]['split']=='train']))
def corr(x,y):
    x=np.array(x);y=np.array(y);keep=np.isfinite(x)&np.isfinite(y)
    return float(np.corrcoef(x[keep],y[keep])[0,1]) if keep.sum()>2 and np.std(x[keep])>0 and np.std(y[keep])>0 else None
def stats(rr):
    d={'n':len(rr)}
    for k in numeric[:7]:
        x=np.array([r[k] for r in rr]);d[k+'_mean']=float(np.nanmean(x));d[k+'_median']=float(np.nanmedian(x));d[k+'_finite_n']=int(np.isfinite(x).sum())
    for k in ['area','major','minor','eccentricity','perimeter','circularity']:d[k+'_pearson']=corr([r['gt_'+k] for r in rr],[r['pred_'+k] for r in rr])
    d['confidence_vs_iou_pearson']=corr([r['quality_confidence'] for r in rr],[r['iou'] for r in rr]);return d
summary={'status':'GT-point mask diagnostics; historical upstream weights not outer-independent','small_large_threshold_train_GT_area':cut,'overall':stats(rs),'by_class':{},'by_fold':{},'by_border':{},'by_size':{},'by_roi':{}}
for key,fn in [('by_class',lambda r:r['class']),('by_fold',lambda r:str(r['fold'])),('by_border',lambda r:r['border']),('by_size',lambda r:'small' if r['gt_area']<=cut else 'large'),('by_roi',lambda r:r['roi'])]:
    groups=collections.defaultdict(list)
    for r in rs:groups[fn(r)].append(r)
    summary[key]={k:stats(v) for k,v in groups.items()}
(A/'quality_summary.json').write_text(json.dumps(summary,indent=2));flat=[]
for key in ['by_class','by_fold','by_border','by_size','by_roi']:
    for label,v in summary[key].items():flat.append({'group_type':key,'group':label,**v})
with (A/'quality_grouped.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
m=np.load(O/'masks.npy');gt=np.load(A/'gt_masks_diagnostic.npy');rgb=np.load(A/'rgb_crops_qc.npy');order=sorted(rs,key=lambda r:(r['dice'],r['uid']));chosen=[order[0],order[-1]]
for c in ['plasma_cell','histiocyte','melanophage','neutrophil','apoptosis']:
    sub=sorted([r for r in rs if r['class']==c],key=lambda r:(r['dice'],r['uid']));chosen.append(sub[len(sub)//2])
for page in range(2):
    batch=chosen[page*4:(page+1)*4];fig,axs=plt.subplots(len(batch),6,figsize=(15,3*len(batch)),squeeze=False)
    for j,r in enumerate(batch):
        i=r['index'];hard=m[i]>.5;overlay=rgb[i].copy();overlay[hard]=(.5*overlay[hard]+.5*np.array([0,255,255])).astype('uint8');err=np.zeros_like(rgb[i]);err[hard&(gt[i]==0)]=[230,70,50];err[(~hard)&(gt[i]>0)]=[50,100,230];err[hard&(gt[i]>0)]=[80,170,80]
        ims=[rgb[i],gt[i],m[i],overlay,err,np.full((96,96),r['quality_confidence'])];titles=['RGB '+r['class'],'GT raster','Predicted probability','Overlay','FP red / FN blue','Quality '+f"{r['quality_confidence']:.3f}"]
        for k in range(6):axs[j,k].imshow(ims[k],cmap='viridis' if k in [2,5] else 'gray',vmin=0 if k in [1,2,5] else None,vmax=1 if k in [1,2,5] else None);axs[j,k].set_title(titles[k],fontsize=9);axs[j,k].axis('off')
        axs[j,0].set_xlabel(f"{r['uid']}\nDice {r['dice']:.3f}, IoU {r['iou']:.3f}")
    fig.suptitle('Historical BioMask: GT-centered diagnostic, not clean outer validation');fig.tight_layout();fig.savefig(Q/f'quality_examples_{page+1}.png',dpi=130,bbox_inches='tight');plt.close(fig)
(Q/'selection.json').write_text(json.dumps([{'uid':r['uid'],'index':r['index'],'dice':r['dice'],'class':r['class']} for r in chosen],indent=2))
doc='''# BioMask audit — Exploration 3

Previous conclusion: Tier B was unavailable from currently verified artifacts.

New evidence: five historical final checkpoints were recovered through direct Google Drive folder reads and their SHA256 values exactly match the saved BioMask provenance. Search returned empty checkpoint folders while direct folder fetch returned binaries; an empty search was not treated as evidence of absence. Local D:/Research/PUMA/Code and Downloads searches found source and logs but no corresponding local checkpoints. The six-entry Downloads ZIP contained no BioMask weights.

Updated conclusion: the historical model exists and produces measurable masks, but cannot supply a leakage-clean current architecture-selection estimate. Its outputs are restricted to explicitly contaminated diagnostics. Segmentation loss alone neither validates nor invalidates downstream biology.

## Architecture and prediction contract

The copied source implements ProposalBiologyNetwork(base32): GroupNorm/SiLU two-convolution blocks, 6→32 encoder, downsample64/128,128 bottleneck, skip-connected bilinear decoder and one mask-logit channel. External input has four channels RGB [0,1] plus sigma2.5 Gaussian at the proposal; H optical-density projection and forward-difference H gradient create six internal channels. There are sigmoid presence/quality outputs and tanh two-coordinate offsets bounded to ±24 for a96 crop. Presence pools prompt-centered intermediate features (sigma9,radius22) plus global bottleneck context; the mask decoder is not its input. Quality uses the bottleneck. Offsets are relative to prompt coordinates, and are not applied to Stage1 in this investigation.

Historical crop96 is source-pixel scale, Python round-center with reflect padding and a valid-region mask. This differs from the appearance crop's white padding and floor(x+.5) rule at exact half pixels. The source CropTransform was reused for BioMask rather than pretending both pipelines share identical padding. No semantic GT class enters BioMask forward prediction. GT polygons, geometry association and sampling strata enter historical supervision, not inference. Training used proposal-aware OOF Stage1 centers, not merely GT centroids. The new450-point diagnostic is recomputed at existing GT-area centroids, and is not mislabeled as reuse of the historical full proposal bank.

Targets rasterize associated nucleus polygons, distinguish instance/ambiguous/empty supervision and attach masks, presence, quality and offset weights. The loss is .52*(.6 instance BCE+.4 instance soft-Dice + .25 empty BCE)+.23 importance-weighted presence BCE+.17 weighted smooth-L1 center offset+.08 quality BCE. Quality target is current threshold-.5 IoU vs GT. Full source files preserve weight normalization and ambiguous-target details. AdamW LR3e-4 to2e-6 cosine, WD1e-4, batch128,30epochs; shifted-crop and D4/photometric augmentation are training only. The checkpoint manager returns **final.pt epoch29**, not the minimum held-loss epoch. The five supplied best held losses are therefore descriptive, not the checkpoint rule.

## Provenance and leakage

Each saved checkpoint declares held fold k and training folds all others. Actual historical ROI fold sizes are **47,1,86,61,10**. The one-ROI fold is metastatic028. Saved case_id equals ROI ID and is not evidence of patient-level grouping. Fold mappings and hashes are preserved; neither mapping nor checkpoint labels were repaired.

For every current outer held group set, fold_provenance.csv records its historical folds and their overlap with each checkpoint's training folds. There is no clean checkpoint path for these mixed outer groups. Row-wise OOF masks for training row in historical fold j use model_j, which typically trained on the outer held fold k. This is Case B, not nested cross-fitting. Even using one BioMask_k consistently requires upstream Stage1 training-proposal provenance: row-wise OOF detector inputs to BioMask training may themselves use models trained on k. We do not certify indirect outer independence. Nested cross-fitting or an independent fixed upstream training cohort would be needed for clean promotion. No20-model nested retraining was performed.

Direct stage2-train/validation features use RGB and predicted masks only, but their upstream supervised weights carry contamination. Training-only feature standardization cannot remove that leakage. All B-series scores must remain diagnostic, irrespective of magnitude.

## Predicted Tier-B schema

The unchanged historical extract_biology_features returns20 soft-mask measurements: log1p area; covariance-based major/minor axes (4sqrt eigenvalue); minor/major; eccentricity; L1 forward-difference soft perimeter; circularity clipped1.5; perimeter²/area; weighted H mean/std; gradient mean/std; weighted absolute zero-padded Laplacian; log1p(100 H variance) entropy proxy;9-pixel-maxpool ring H contrast; mask maximum/mean; sigmoid predicted quality; ROI H robust z and sigmoid(1.25z) percentile proxy. These are not interchangeable with hard-mask regionprops measurements or the TierA empirical percentile/32-bin entropy. The20D order and family slices are in tierB_schema.json. Historical identity token maps were discovered but not consumed by these minimal models; their larger identity-fusion path is outside the finite architecture set.

## Quality and QC definitions

Binary Dice/IoU use threshold.5 and GT raster from the original annotation feature's polygons (holes removed), masked to valid image pixels. Disconnected predicted regions are treated as one foreground label for aggregate moments. Regionprops axes/eccentricity and skimage digital perimeter are used consistently on binary prediction and GT for diagnostic correlations; they are not the soft-feature definition. Empty predicted foreground has area/axes/perimeter0 and undefined centroid; finite denominators are reported. Area-relative error is absolute area difference / max(GT area,1). Border means crop contains invalid image padding; small/large uses the training-row GT-area median. GT quality never enters classifier inputs.

QC includes the lowest and highest Dice examples plus median-Dice examples for five monitored tail classes. These are diagnostic selections, not an unbiased gallery or pathologist label adjudication. Confidence maps shown are the scalar predicted quality repeated spatially, clearly labeled; they are not pixelwise uncertainty estimates. Raw probability, GT, overlay and FP/FN error map accompany them.
'''
doc+='\n## Measured overall quality\n\n```json\n'+json.dumps(summary['overall'],indent=2)+'\n```\n\nPer-fold/class/ROI/border/size results: quality_grouped.csv and quality_summary.json. Individual results: quality_per_nucleus.csv. Quality/classification complementarity is added after the finite downstream runs.\n'
(A/'BIOMASK_AUDIT.md').write_text(doc,encoding='utf-8');print(json.dumps(summary['overall']))
