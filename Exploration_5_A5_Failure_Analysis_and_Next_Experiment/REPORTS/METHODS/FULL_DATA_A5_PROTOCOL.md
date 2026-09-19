# Exactly one next experiment: clean full-population retained A5

Status: PREPARED, NOT EXECUTED. Admission currently fails upstream outer-exclusion and untouched test availability. This is the next experiment, not a new architecture. Do not bypass admission, regenerate proposals, change Stage 1, or call a conditional subset result end-to-end.

## Question and prediction

Does increasing genuinely independent training coverage with the unchanged A5 reduce held-group error and class instability? The existing 300-cell enriched screen uses only a small fraction of 97,193 annotations and very few independent tail examples. The expected mechanism is increased phenotype/group coverage, not additional head capacity. Failure to improve robust held-group metrics despite increased clean coverage would strengthen the case for a representation or boundary intervention.

## Admission contract

1. Recover highest verified grouping (patient > case > slide > ROI). ROI is the only currently established unit; never invent patient IDs from ROI names. Record the source of the mapping and duplicate-image audit.
2. Training, development, natural calibration and test group sets must be pairwise disjoint. Previously inspected development/confirmation ROIs cannot become locked test. The historical 150 is permanently development.
3. Every upstream detector dependency used to generate Stage2 training/validation inputs must exclude locked outer-held groups. The current 109,637-row mixture has no historical fold excluded by every contributing detector; therefore no four-way complete-population clean split can currently be certified. Row-wise OOF is inadequate for this requested stronger contract.
4. Resolve this through existing independently generated, frozen, exclusion-verified artifacts or an independent cohort with appropriate provenance. If that cannot be recovered, return an admission failure. Any new detector training, weights, proposal generation or operating-point change is outside this study. A fold0-only diagnostic is possible but does not meet complete-population scope and is not silently substituted.
5. Validate all 109,637 proposal UIDs, coordinates, scores, checkpoint hashes, complete GT components and ROI roster, including zero-proposal ROIs (currently zero, but evaluator must support them). Keep 15,292 unmatched proposals in evaluation. Exclude unknown semantic labels only from ten-class CE. Preserve missed GT; distinguish feature census from component-based matching population.
6. Verify cached row identities against proposal UIDs, RGB preprocessing, feature schema, encoder checkpoint hash and representation contract. Existing 450 GT-centered feature caches are not complete-proposal caches.

## Fixed implementation

FOV96 white-padded, coordinate-centered RGB, bicubic resize224, ImageNet normalization. Frozen verified UNI2-h CLS1536; nonaffine LayerNorm. Sixteen Tier-A features from preserved biology.py, training-only standardization (std floor1e-6). Linear1536→10 plus biasless1536→8 and16→8 multiplicative projections and zero-initialized8→10 output. Exactly27,866 trainable head parameters; delta0. No tissue, mask, adapter, multiscale, reject class, threshold tuning or ensemble.

Inverse-count replacement sampler with uniform induced class prior, ordinary CE, AdamW LR0.001, weight decay0.01, batch64, clip1. Seeds17/29/43 independently; shared split and per-seed draw manifests. Maximum20 epochs, inherited prospective patience5, minimum development fixed-ten ROI-F1 gain1e-4. Keep the first epoch on ties. Record the selected epoch before one test evaluation. No post-test epoch changes. Calibration split reserved; temperature remains1 for the primary uncalibrated comparison so calibration cannot change confidence-prioritized matching after test inspection.

## Matched control and inference

Run the same A5 on the original-sized, training-only subset selected from the clean training groups before training (300 nuclei with the historical per-class counts, if feasible). Compare to all eligible matched proposals from those training groups. Identical held development/calibration/test, seeds, initial shared parameters, evaluator, optimizer and fixed maximum optimizer-update budget; replay a predeclared number of balanced draws. The full-data arm changes unique training coverage only; an epoch is defined as the fixed shared draw budget for this comparison. The final operational full-data fit uses the inherited20-epoch schedule, reported separately from the coverage-control result. This is a single data-coverage study, not two architecture candidates.

Exact draw budget and final UID split must be locked after admission and before training. This is a prepared protocol, not a retrospective preregistration of measurements already seen. Estimate cached-head cost on admitted data before launch. Historical CPU extraction rate around0.6 nuclei/s implies roughly50 hours for109,637 crops; this is a planning extrapolation, not a measured runtime. Do not launch this extraction while admission fails.

## Required outputs and decision

For every epoch record sampled objective, full training/development CE and Macro-F1, gap, LR, class counts, observed UID/ROI/group exposures, repeats, parameter and gradient norms, runtime and peak memory. Save configs, command/environment, source and data hashes, RNG/draw manifest, checkpoints and predictions.

Report conditional semantic confusion/precision/recall/F1, accuracy, balanced accuracy, NLL and ECE15 separately from preserved V17 fixed-ten ROI P/R/F1, pooled P/R/F1, and class TP/FP/FN against complete GT. No average fold semantic score may substitute for pooled OOF semantic F1.

Use the highest verified group for paired bootstrap; average seed effects within group before resampling. The coverage intervention qualifies only with ROI-F1 gain>=0.003, >=2/3 positive seeds, 95% lower bound>0, no mean class recall loss>0.10 and gap increase<=0.05. Retain all zero-recall classes. These thresholds cannot overcome tiny independent supports; disclose intervals and group counts. If it fails, retain A5 and record a null result; do not try a second winner on the same test.
