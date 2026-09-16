# One next full-scale architecture

**Frozen UNI2-h + a 96-pixel source crop + linear appearance logits + 16-feature linear RGB correction.** This is the architecture selected by the completed CPU controls. It has 15,530 trainable parameters: 1,536×10+10 appearance parameters and 16×10 correction parameters. The encoder is entirely frozen and always in evaluation mode. No LoRA or auxiliary branch is active.

## Inputs and representation

Preserve each Stage 1 proposal's x/y in original ROI pixel coordinates, UID and source score. No coordinate refinement, filtering or tissue segmentation is performed. RGB input is a single 96×96 source-pixel square, centered by the exact `crop.py` implementation, white padded outside the image and bicubically resized to 224×224. Convert to float32 [0,1] and ImageNet channel normalization mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. The local TIFF scale is approximately 0.2263–0.2269 µm/pixel, so 96 source pixels cover about 21.7 µm. This physical scale is a dataset observation; rescale new acquisitions deliberately, not implicitly.

UNI2-h uses ViT giant patch14, 224 input, 1536 embedding dimensions, 24 blocks, 24 attention heads, eight register tokens, `no_embed_class=True`, packed SwiGLU, SiLU, MLP ratio `2.66667*2`, layer-scale initialization 1e-5. Use CLS only, then PyTorch nonaffine layer normalization over 1536 entries, epsilon 1e-5. Strict state loading and the checkpoint SHA256 are mandatory:

`32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe`

No stain fitting, online color/stain augmentation, flips or stochastic crop augmentation is included in the selected deterministic-cache recipe. Augmentation was not separately validated in this continuation. Encoder extraction uses eager FP32, CPU eight threads, batch four, workers zero. GPU extraction is supported by the device argument but was not benchmarked here. Cached features are raw CLS vectors; LayerNorm is applied once when preparing classifier inputs.

## Exact 16-feature schema

Let RGB/255 be X. Define the fixed intensity projection H=clip((−log(clip(X,1/255,1))·[0.650,0.704,0.286])/2,0,2). This is an H-like proxy, not stain unmixing. Grayscale is the mean of the three normalized RGB channels. Point centers are rounded by floor(x+0.5), floor(y+0.5). A 49×49 window contains a central disk radius 12 and an annulus with radii 16 through 24, endpoints inclusive. Exclude out-of-image pixels from both masks; fail if either has no valid support.

| Family | Ordered features | Definition |
|---|---|---|
| stain | H_mean, H_std, H_p10, H_p50, H_p90 | Central-disk population mean/std and NumPy percentiles |
| gradient_texture | H_gradient_mean, H_gradient_std, H_abs_laplacian_mean, H_entropy32 | Central statistics of sqrt(gx²+gy²+1e-8), absolute SciPy Laplacian with reflect boundary, and natural-log entropy of 32 equal bins over [0,2] |
| ring | H_center_minus_ring, gray_center_minus_ring, H_ring_std, central_valid_fraction, ring_valid_fraction | Central-minus-annulus means, annulus H population std, and valid/nominal disk/annulus pixel counts |
| roi_relative | H_robust_z, H_ROI_percentile | (central H mean−whole-ROI median)/max(whole-ROI IQR,1e-4); fraction of whole-ROI H values ≤ central H mean |

`np.gradient` computes image H gradients once, including its default one-sided image-edge rule. Whole-ROI summaries are image-local at inference, not fitted across validation images. All feature means and population standard deviations for standardization are fitted **only on labeled training rows**, std floor 1e-6; do not clip standardized outliers. The exact column order is saved with the checkpoint. No masks, annotations, neighbor labels or tissue segmentation enter this extractor. GT morphology exists only in separate oracle experiments and is absent here.

## Head, initialization and objective

For raw CLS h and standardized biology b, logits are

`z = W_a LayerNorm(h) + a + W_b b`.

W_a and a use the ordinary PyTorch Linear initialization; W_b is exactly zero at initialization. The appearance initialization precedes creation of W_b so seeded baselines match. There is no projection bottleneck, affine normalization, dropout, activation, gate or learned encoder update. Apply a single softmax across ten logits. Canonical classes are tumor, lymphocyte, plasma_cell, histiocyte, melanophage, neutrophil, stroma, epithelium, endothelium, apoptosis. The V17 adapter swaps the integer positions of epithelium/endothelium by name; canonical training IDs never change.

For labeled training class counts n_c, draw N examples with replacement with per-example weight 1/n_y. Expected class mass is uniform; no class-weight multiplier is added. Loss is ordinary CE, `−z_y+logsumexp(z)`. The shared loss implementation can express logit adjustment, but final config τ=0 and the trainer rejects stacking a nonzero τ with balanced sampling. Unmatched proposals labeled −1 are not trained as any of the ten classes and remain in end-to-end evaluation.

AdamW: LR 0.001, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01 on all head parameters including bias. Batch 64, accumulation one, gradient global-norm clip 1.0, FP32, constant LR. No backbone LR exists because all encoder parameters are frozen. Pilot runs used ten epochs; the one prospective full-data schedule is at most 20 epochs with five-epoch patience and minimum development fixed10 ROI-F1 increase 0.0001. Seeds 17/29/43 are repeats of this architecture, not an ensemble. There is no training-resume feature; completed runs are immutable and new runs require new output directories.

## Metrics, calibration and inference

Primary selection is exact local V17 fixed10 ROI macro-F1 from **all proposal rows against full GT for explicitly listed ROIs**. Always report ROI P/R/F1, pooled P/R/F1, per-class TP/FP/FN and zero recalls. The explicit ROI list retains ROIs with zero predictions. Semantic P/R/F1, balanced accuracy, confusion, conditional labeled accuracy, NLL and 15-bin ECE are supplemental; the log records the labeled denominator and all-proposal count. Unmatched proposals are not dropped from PUMA counts.

Inference preserves coordinates and emits class, maximum semantic probability and all ten probabilities; original proposal metadata remains in CSV. The original V17 serializer emits the class mapping and symmetric geometry for JSON. Stage 1 scores are preserved as metadata; semantic confidence is the score used by this selected classification submission, rather than the historical semantic×utility score. No extra rejection threshold is introduced.

If a separate naturally sampled calibration partition exists, fit scalar T using 161 log-spaced values in [0.1,10] minimizing NLL. Keep the classifier fixed. T preserves argmax but can change cross-proposal confidence ranking and greedy matches; evaluate the final serialized calibrated output again. This optional scalar is not fitted on the pilot validation cohort and cannot repair all class-prior mismatch or establish deployment calibration.

## Provenance and validity boundary

The next full-scale run requires verified fixed Stage 1 outputs, out-of-fold training centers, exclusion of all held-out outer groups from corresponding detector training, and disjoint semantic splits. The provenance JSON is an explicit caller assertion requiring documentary verification; code cannot prove how a historical checkpoint was trained. GT-centered controls require an explicit flag and must be reported as controls. Dataset/checkpoint files are read-only inputs. Cache contracts include manifest, image, source and encoder hashes; both appearance and biology cache payloads have checksums.

The completed study demonstrates a small development benefit on 150 GT-centered validation features. It does not establish complete-population Stage 1 performance, patient independence, cross-site generalization or clinical/SOTA readiness. The code is a tested research implementation with explicit input contracts; the complete full-data pipeline has not been run because the exact Stage 1 outer-split artifact provenance has not been verified.
