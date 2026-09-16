# Exploration1 UNI2_PROBE

Independent local implementation recovered from probe_uni2.py. The original source is preserved by hash and in the source archives. Run `python run_train.py --config config.json --output NEW_RUNS`; use `--inputs` to relocate the immutable Exploration1input snapshot. Set recipe ce/la/balanced_ce and seed17/29/43 in a new config; historical semantic validation-best selection is preserved.

`run_eval.py --predictions <saved.npz> --output metrics.json` re-evaluates available logits with original semantic metrics. If original logits were not saved, that evaluator cannot invent them; use a new training run to produce new outputs. `parity_test.py` verifies the included V17 evaluator for future use, without relabeling old semantic numbers as V17. Historical metadata, logs and checkpoints are in ORIGINAL_RUNS; all source files and hashes are indexed. No original file is modified.
