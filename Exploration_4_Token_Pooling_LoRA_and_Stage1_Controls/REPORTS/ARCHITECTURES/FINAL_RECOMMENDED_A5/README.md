# Single next full-scale architecture: retained Exploration3 A5

Exploration4 A3 failed the predeclared class-recall confirmation constraint. No runner-up was selected. This package retains the exact frozen CLS1536 + standardized TierA16 rank-eight interaction model, 27,866 trainable parameters. The full-image code is copied from the verified Exploration3 deployment blueprint; the archived experiments are GT-centered development controls, not a full-scale trained system.

Run `python features.py --help`, then `python run_train.py --help`. Full-proposal evaluation uses `python run_eval.py --eval-checkpoint CHECKPOINT` with the manifest, cache, annotation root, ROI splits and documented frozen Stage1 provenance. It includes all proposals and zero-proposal ROIs; unknown semantic labels are excluded only from CE. New run directories are mandatory. Temperature remains1 unless fit on a separate calibration split.

The recommended full-scale recipe is AdamW headLR.001, weight decay.01, batch64, inverse training-class count sampling, ordinary CE, at most20epochs and patience5 on the prespecified validation metric. This is a prospective recipe, not an executed full-scale claim. Preserve patient-level exclusion if patient identifiers become available. Local code never imports another architecture's modules.
