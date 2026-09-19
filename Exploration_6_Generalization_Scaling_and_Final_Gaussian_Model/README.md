# Exploration 6 — generalization, scaling, and final Gaussian model

Exploration 6 completed a finite preregistered model-selection study on an NVIDIA L4: D300/D600/D900 scaling, augmentation, sampling and cRT controls, optimizer/regularization HPO, Tier-A ablation, frozen representation selection, conditional adaptation gating, three-seed replication, and two frozen holdouts. Gaussian token pooling (sigma 1.5) with the rank-8 appearance × Tier-A head was selected. The locked natural population exposed severe rare-class overprediction, so the retained model is a research result rather than a deployment-ready system.

Start with `EXPLORATION6_REPORT.md`. The full joint-training history is `PROMPT6_EPOCH_LEDGER.jsonl`; decoupled classifier epochs are in `PROMPT6_DECOUPLED_EPOCH_LEDGER.jsonl`. Metrics, predictions, and logs are separated under `RESULTS`; code, configurations, tests, and evaluators are under `CODE`; figures are under `FIGURES`.

Use `FILE_INDEX.csv` for the canonical inventory and `RELOCATION_MAP.csv` or `DUPLICATE_REFERENCES.csv` when tracing one of the many old Colab/runtime paths.

The archive intentionally omits all `.pt` checkpoints, UNI2 weights, raw TIFF/GeoJSON inputs, and D300/D600/D900 token/preprocessing arrays. Hashes, configs, ledgers, metrics, and final selection records remain so every executed test and epoch can be audited. Stage J remained infeasible because a certified complete fixed Stage-1 proposal artifact was unavailable.

## Authoritative training code

`src`, `scripts`, `tests`, `configs`, and the files directly under `CODE` are restored byte-for-byte from the latest uploaded Exploration-6 ZIP, `PUMA_STAGE2_COLAB-20260913T092151Z-1-001.zip`. Run the package from `CODE`; `puma_exploration6` is the authoritative implementation.

`EXECUTED_CONFIGS` retains the generated configurations associated with executed runs. `HISTORICAL_SUPPORT` contains older local orchestration, packaging, and superseded source snapshots for provenance only; it is not imported by the authoritative runtime.
