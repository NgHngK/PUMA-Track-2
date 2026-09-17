# Exploration 3: B2

This is an independent local code package reconstructed from the exact historical sources. Results are copies, not reruns. Original sources and missing-data limitations remain in PROVENANCE and ORIGINAL_PATH_MAP.csv.

Run from this directory:

```powershell
python run_train.py --config CONFIGS/original/p3_B2_real_s17_cv0.json --output NEW_RUNS
python parity_test.py
python run_eval.py --predictions RUNS/<experiment>/<prediction_file>.npz --output reevaluation.json
```

The evaluator reads stored logits and exact UID/coordinate manifests. Training reads one shared immutable input snapshot at REPRO_INPUTS; override --inputs after relocating it. Raw RGB paths remain the configured dataset paths. Feature extraction requires the separately supplied UNI2 checkpoint. No network download is implicit. New training output is directed explicitly to --output. Original logs/checkpoints are not overwritten. See ARCHITECTURE.md for recovered equations and tensor flow, and RESULT_INDEX.csv for every source result.

Historical selected/best and final endpoints must not be interchanged. Exploration2 selected validation-best; Exploration3 fixed epoch10. BioMask/TierB Exploration3 comparisons remain upstream-contaminated diagnostics. Evaluator parity here establishes supplied-local-V17 behavior, not challenge-server certification. Historical Exploration1 scores used semantic metrics; this does not retroactively relabel them as V17 scores.
