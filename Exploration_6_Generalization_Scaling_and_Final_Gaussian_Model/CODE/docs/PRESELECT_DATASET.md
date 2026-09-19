# Exploration 6 — CPU dataset preselection

This patch is designed to be copied into the certified
`PUMA_STAGE2_PROMPT6_CODE` folder.

It does **not** train a model and does **not** copy image data. It only:

1. scans `01_training_dataset_geojson_nuclei`;
2. validates matching TIFFs in `01_training_dataset_tif_ROIs`;
3. builds one row per annotated nucleus;
4. reads an authoritative historical ROI manifest;
5. audits whether the original historically-untouched LOCKED quota is feasible;
6. reserves every historically unused ROI as `LOCKED_NATURAL`;
7. uses MILP/HiGHS to create leakage-free TRAIN / DEV / Exploration6-confirmation group pools;
8. selects exact D300 ⊂ D600 ⊂ D900 class quotas with round-robin group diversity;
9. writes compact manifests and census tables for Codex.

## Recommended command

From:

`C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_PROMPT6\PUMA_STAGE2_PROMPT6_CODE`

run:

```bat
python scripts\preselect_exploration6_dataset.py ^
  --dataset-root "D:\Research\PUMA\Code\TRAINING CODE\Dataset" ^
  --historical-rois "C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4\EXPLORATION_3\ARCHITECTURES\A5\INPUT_MANIFESTS\sample_manifest.csv" ^
  --out-dir "C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_PROMPT6\DATA_PRESELECTION"
```

If the exact historical manifest used by the blocked Exploration-6 admission is elsewhere,
pass that file instead. **Prefer the exact artifact Codex used for its 181-ROI
historical cohort.**

If patient/case/slide metadata is available:

```bat
  --group-csv "...\roi_groups.csv"
```

The group CSV must contain `roi` plus one complete field from:
`patient_id`, `case_id`, `slide_id`, `group_id`, in that priority order.

## Important output semantics

`SELECTION\D300.csv`, `D600.csv`, `D900.csv`

- `train`: Exploration-6 TRAIN
- `dev`: Exploration-6 DEV
- `locked`: **PROMPT6_CONFIRMATION**, prospectively locked in Exploration 6 but
  historically exposed at the ROI level

`SELECTION\LOCKED_NATURAL.csv`

- all historically-unused ROIs
- no class balancing
- no subsampling
- final unseen-ROI diagnostic cohort

The script will explicitly show why the old untouched 10-class LOCKED protocol is
or is not feasible.

## Outputs

- `FULL_GT_MANIFEST.csv`
- `FULL_GT_MANIFEST.json`
- `SELECTION\D300.csv`
- `SELECTION\D600.csv`
- `SELECTION\D900.csv`
- `SELECTION\LOCKED_NATURAL.csv`
- `SELECTION\CLASS_CENSUS.csv`
- `SELECTION\ROI_CENSUS.csv`
- `SELECTION\HISTORICAL_ROIS.csv`
- `SELECTION\HISTORICALLY_UNUSED_ROIS.csv`
- `SELECTION\RESEARCH_GROUP_ASSIGNMENT.csv`
- `SELECTION\PRESELECTION_SUMMARY.json`
- `SELECTION\PRESELECTION_REPORT.md`
- `RUN_SUMMARY.json`

Codex should consume these outputs rather than rescanning the raw dataset.
