# BioMask audit — Exploration 3

Previous conclusion: Tier B was unavailable from currently verified artifacts.

New evidence: five historical final checkpoints were recovered through direct Google Drive folder reads and their SHA256 values exactly match the saved BioMask provenance. Search returned empty checkpoint folders while direct folder fetch returned binaries; an empty search was not treated as evidence of absence. Local source and download searches found code and logs but no corresponding checkpoints. The six-entry downloaded archive contained no BioMask weights.

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

## Measured overall quality

```json
{
  "n": 450,
  "dice_mean": 0.8574086499267342,
  "dice_median": 0.9136087259224879,
  "dice_finite_n": 450,
  "iou_mean": 0.7795691783101596,
  "iou_median": 0.8409573547686838,
  "iou_finite_n": 450,
  "area_relative_error_mean": 0.2272589842070661,
  "area_relative_error_median": 0.1303312285366124,
  "area_relative_error_finite_n": 450,
  "centroid_error_mean": 1.7048852161303698,
  "centroid_error_median": 0.8727653390283873,
  "centroid_error_finite_n": 446,
  "occupancy_mean": 0.05400559413580247,
  "occupancy_median": 0.04676649305555555,
  "occupancy_finite_n": 450,
  "quality_confidence_mean": 0.7987042979399364,
  "quality_confidence_median": 0.8398978114128113,
  "quality_confidence_finite_n": 450,
  "presence_confidence_mean": 0.9355012194294896,
  "presence_confidence_median": 0.9973224699497223,
  "presence_confidence_finite_n": 450,
  "area_pearson": 0.8942640309965603,
  "major_pearson": 0.8413452948284198,
  "minor_pearson": 0.853245689343264,
  "eccentricity_pearson": 0.7023251341825637,
  "perimeter_pearson": 0.8196528676189493,
  "circularity_pearson": 0.553362226507117,
  "confidence_vs_iou_pearson": 0.7229969593664429
}
```

Per-fold/class/ROI/border/size results: quality_grouped.csv and quality_summary.json. Individual results: quality_per_nucleus.csv. Quality/classification complementarity is added after the finite downstream runs.
