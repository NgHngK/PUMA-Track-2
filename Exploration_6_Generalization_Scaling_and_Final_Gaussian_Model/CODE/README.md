# PUMA Stage-2 Exploration-6 code package

Generalization-first research implementation for the retained Exploration-3 A5 system and Exploration-6 controlled experiments.

## What is implemented

- exact Exploration-3 A5 cached-CLS forward contract;
- Exploration-4 token representations: CLS, target patch, Gaussian sigma1.5, 3×3 neighborhood, exact A4 global-local, optional one-scalar mix;
- x3 unique enriched research cohort builder: D300 ⊂ D600 ⊂ D900, preserving the historical class-support lineage exactly (1×/2×/3×), with common fresh DEV225 + LOCKED225, group-disjoint at the strongest supplied `group` unit;
- raw-image geometric, fixed-H/E OD concentration, and empirically gated coordinate-jitter augmentation;
- natural, inverse and tempered class sampling;
- CE, logit adjustment and Balanced-Softmax-compatible prior-adjusted CE with anti-stacking validation;
- interaction rank/dropout/Tier-A ablations;
- cRT-style decoupled classifier retraining (freeze Ph/Pb, reset/retrain W/R);
- staged HPO config generator with TRAIN-only grouped-CV preparation;
- prospective early stopping **or** historical fixed-final selection;
- exact sample-normalized gradient accumulation for raw UNI2 runs (safe microbatch, preserved effective batch);
- immutable run directories, cache/manifest SHA validation, saved TRAIN-only Tier-A normalizer;
- one-shot locked checkpoint evaluation;
- inference-only selected-checkpoint prediction on complete Stage-1 proposal manifests (`label=-1`), exporting canonical class + all ten probabilities for exact V17 evaluation;
- Windows-safe cached memmaps: spawned DataLoader workers reopen the `.npy` mapping by path instead of serializing multi-GB token arrays;
- vendored PUMA V17 evaluator core from the supplied A5 package.

## Installation

```bash
python -m pip install -e .
# For real UNI2-h raw-image runs:
python -m pip install 'timm==1.0.20'
```

No network download is performed by the model loader. `uni2_weights` must point to the local verified checkpoint.

## Recommended Exploration-6 sequence

1. Build full GT manifest, preferably with patient/case mapping if available.
2. Build D300/D600/D900 x3 manifests.
3. Audit historical overlap, especially LOCKED group overlap.
4. Compute Tier-A separately for each manifest (cache hash is bound to that manifest).
5. Extract deterministic CLS/token cache for controls.
6. Run data-scaling control first.
7. Run AUG0→AUG1→AUG2; AUG3 only after displacement gate.
8. Prepare broad HPO with `scripts/prepare_cv_hpo.py`; it hides external DEV/LOCKED and uses grouped folds only inside TRAIN. Run limited long-tail + regularization/HPO.
9. Re-audit Tier-A.
10. Replicate fixed global-local only after the training regime stabilizes.
11. Do not run conditional encoder adaptation unless the Exploration-6 gate opens.
12. Freeze all decisions, then use `evaluate_checkpoint.py --split locked` once.

## Important historical control

`configs/00_historical_A5_cached.json` uses `selection_mode=fixed_final`, 10 epochs, constant LR=.001, AdamW weight decay=.01 on **all** A5 parameters including bias, inverse replacement sampling, CE, batch64. This intentionally matches the actual Exploration-3 A5 endpoint recipe rather than silently replacing it with early stopping.

## Cache integrity

Every `.npy` Tier-A / representation cache must have a same-stem `.json` sidecar containing:

```json
{"manifest_sha256": "..."}
```

The provided cache scripts generate this automatically. Training fails closed if the sidecar is missing or mismatched.

## Raw UNI2 batch contract

Raw-image configs use physical microbatch 4 and accumulation 16 by default, preserving effective batch ~64 without attempting a memory-prohibitive UNI2-h batch of 64. Cached historical A5 remains batch64/accumulation1. HPO varies effective batch through accumulation in raw mode.

## Exact x3 class lineage

Exploration-6 interprets “3×” as tripling the actual Exploration-3 research-cohort class support, not silently replacing it with a balanced dataset.

- D300 TRAIN: tumor 71, lymphocyte 37, each other class 24.
- D600 TRAIN: tumor 142, lymphocyte 74, each other class 48.
- D900 TRAIN: tumor 213, lymphocyte 111, each other class 72.
- Fresh DEV225: tumor 58, lymphocyte 23, each other class 18.
- Fresh LOCKED225: tumor 59, lymphocyte 22, each other class 18.

The builder prioritizes group-disjointness and fails if these quotas cannot be satisfied from the supplied unpartitioned source pool. It refuses an already-partitioned source manifest.

## Complete-proposal inference

After the final checkpoint is frozen, prepare a target proposal manifest with `split=predict` and `label=-1`, compute matching Tier-A / representation caches, then run:

```bash
python scripts/predict_checkpoint.py \
  --checkpoint path/to/final.pt \
  --manifest path/to/stage1_proposals.csv \
  --tier-a path/to/proposal_tier_a.npy \
  --representation path/to/proposal_features_or_tokens.npy \
  --out predictions.csv \
  --device cuda

python scripts/evaluate_end_to_end.py \
  --gt-dir path/to/01_training_dataset_geojson_nuclei \
  --predictions predictions.csv \
  --out v17_metrics.json
```

For raw-image checkpoints use `--uni2-weights` instead of `--representation`. The predictor loads the TRAIN-only Tier-A normalizer from the checkpoint and never refits on target proposals.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The package ships tests for baseline numerical parity, parameter counts, token-pooling equations, augmentation coordinates, Tier-A, sampler priors, losses, x3 nested splits, V17 mapping, cached and raw synthetic data flow, cRT, and locked-evaluation normalization.

See `docs/ARCHITECTURE_MATRIX.md` and `docs/DATA_FLOW.md`.
