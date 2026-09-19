# Data-flow contracts

## Raw-image arm

`manifest row → ROI RGB → historical white-padded FOV96 crop → stochastic label-preserving transform → ImageNet normalization → strict local UNI2-h → 265 final-normalized tokens → configured token pool → non-affine LN → A5`.

For geometric and stain augmentation, Tier-A is **not** recomputed from the augmented image; it remains the original-ROI point feature standardized with TRAIN-only statistics. For the separately gated coordinate-jitter experiment, the proposal point itself changes, so Tier-A is recomputed at that jittered point from the original unaugmented ROI and transformed with the same TRAIN-fitted normalizer. This prevents a shifted appearance crop from being paired with stale point-derived Tier-A.

## Cached CLS arm

`manifest-aligned raw CLS cache → non-affine LN1536 → A5`.

This reproduces Exploration-3 A5. A manifest SHA-256 sidecar is mandatory to prevent row/cache misalignment.

## Cached token arm

`manifest-aligned B×265×1536 final-token cache → Exploration-4 pooler → non-affine LN → A5`.

## Split safety

`read_manifest` rejects UID duplication and any group/ROI/image crossing split boundaries. Locked evaluation loads the TRAIN-fitted Tier-A normalizer from the checkpoint and never refits it.

## Stage 1

Stage 1 is external and frozen. Coordinate jitter, when authorized, perturbs the Stage-2 proposal/crop point according to a pre-measured error distribution; local token coordinates and point-derived Tier-A follow that same perturbed point. It does not update detector weights or post-processing.
