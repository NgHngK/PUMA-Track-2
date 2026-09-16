# Prospective continuation protocol

Preserve all prior evidence. This is an exploratory development study, not confirmatory external validation. Stage 1 remains immutable. Biology Tier A is reconsidered; Tier B requires verified inference-safe masks. No tissue segmentation or neighbour GT semantics enter deployable models.

## Evaluation before experiments

Vendor byte-identical evaluator modules from the supplied local Version 17 tree. Verify original/new matches, traces, counts and aggregates on strict radius, score priority, tie order, centroid collision, class mismatch, empty and randomized fixtures. Canonical labels are converted by names at the boundary because V17 swaps IDs7/8. No matcher algorithm is replaced. Report this as local V17 parity, not independent certification by a live challenge server.

For GT-centred subset controls: preserve area-centroid crop coordinates for continuity and valid cache reuse. Compute V17 GT evaluation coordinates from float32 serialized exterior vertices, including closure, then float64 mean cast to float32. Predictions retain crop coordinates and maximum semantic probability as confidence. These are subset conditional controls; no claim of full-ROI detection evaluation. Primary selection endpoint: V17 fixed-ten ROI-averaged F1. Also save pooled metrics and semantic metrics. Original model scores are not comparable across changed subsets.

## Sample and compute

Seed17 only for sample selection. Preserve previous 327 nuclei and the original full-manifest train/validation split. Expand to 300 train and 150 validation nuclei, prioritizing underrepresented classes and new positive ROIs before further nuclei per ROI. All10 classes retained; no duplicated UIDs. Patient mapping is unverified. Sample exceeds the prior pilot while limiting CPU cost. Fixed stain/crop preprocessing; no online augmentation in cached probes.

Benchmark CPU threads4/8 and batch1/2/4 on actual forward calls; select fastest stable throughput. Benchmark DataLoader workers0/2 once, including startup. Report RSS and CPU process time where available. FP32 remains reference; do not assume CPU BF16 acceleration. Avoid compilation unless measured workload amortization can justify it. Reuse the prior FOV96 cache only for exact matching UID/image/center/preprocessing/checkpoint records. New FOV64/128 require extraction. Cache metadata must record all relevant hashes. Resume only with exact cache-key match.

## Fixed screens

Each cached-head run: ten epochs, effective batch64, AdamW LR1e-3 WD.01, global clip1, constant LR, fixed seed17 and minibatch order. Head nonaffine LN+Linear10; positive class head gradients and exposures logged. Save all epochs and selected/final predictions. StageA: FOV64/96/128 with natural LA tau1. Choose highest V17 ROI F1; differences below.002 favor FOV96. StageB: chosen FOV, natural CE/tau0 versus LA/tau1 versus balanced sampler/CE. Choose highest primary score, with .002 tie preferring natural CE.

## Biology

Tier A: deterministic RGB optical-density projection (historical H proxy, not stain unmixing), local gradient/texture, central-vs-annular contrast, ROI-relative statistics. Central disk radius12 source pixels; annulus16–24; same descriptor geometry across appearance FOVs. No GT shape enters Tier A. No spatial density using GT coordinate sets: unavailable Stage1 sets make that family ineligible for deployable claims.

Run standardized TierA-only linear head, UNI2 plus standardized TierA linear correction initialized to zero, matched parameter placebo from fixed nonlinear random projections of UNI2, and training-row shuffled biology. Shared UNI2 head initialization identical. Fit mean/std on training only. If real fusion exceeds baseline, inspect family removals and repeat baseline/real/placebo/shuffle seeds29/43. Claim biology incremental value only if advantage over both controls is consistent; otherwise retain simpler appearance model. Oracle polygon morphology (area/perimeter/axes/eccentricity/circularity) gets a separate non-deployable diagnostic and never competes for final deployment selection.

## Adaptation and tuning budget

Do not repeat LoRA merely to fill a table; the previous paired negative result is preserved. Reopen only if simple screens give a concrete representation hypothesis. No learned mask model without checkpoint and group provenance. No historic large fusion replication unless minimal deployable biology earns benefit beyond controls. Surviving architecture: cheap LR screen3e-4/1e-3/3e-3, five-epoch screen and extend best two to ten, preserving state; seed17 development only. Weight decay change only if gap evidence warrants it. No full Cartesian sweep or multiple-comparison significance claim.

Final report integrates old/new evidence, exact evaluator limitations, all-class precision/recall/F1/accuracy, computational costs and one next full-data architecture. Any unexecuted conditional experiment is explicitly labeled with its reason.
