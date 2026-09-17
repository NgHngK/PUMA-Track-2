# Exploration 4 — E0_FROZEN

Independent local cached-representation package. All runtime modules are in CODE; no module imports another architecture's code. Large immutable inputs are referenced once at Exploration level and identified by their UID/hash contracts.

```powershell
python run_train.py --config CONFIGS/original/E0_FROZEN_s17.json --output NEW_RUNS
python run_eval.py --predictions RUNS/<experiment>/epoch_10_predictions.npz --output evaluation.json
python parity_test.py
```

For C_ALIGNMENT, `run_eval.py --stage1-centers` evaluates Stage1-centered predictions; the default is GT-centered. Training reproduces the selected coordinate arm, while original paired evaluations for both coordinate inputs are preserved in RESULTS. New output directories are mandatory. Cached runs need no encoder re-extraction. The original RGB dataset and checkpoint remain external for extracting new populations.

The exact executed sources are preserved in PROVENANCE. Local copies only redirect output paths and input-root discovery; the numerical head, sampler, loss and evaluator are unchanged. ARCHITECTURE.md and FLOWCHART.md explain the tested mechanism. RESULT_INDEX.csv maps every copied result to its source.
