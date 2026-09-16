# Standalone next full-data Stage 2 experiment

Read ARCHITECTURE.md first. This directory is self-contained Python source; raw images, annotations and the 2.7 GB-class UNI2 checkpoint are external read-only inputs. Install requirements into your chosen research environment. The tested runtime was Python 3.10, torch 2.11.0+cpu, timm 1.0.20; GPU throughput is untested. This package has no runtime dependency on the experimental shared-core directory.

## Input contracts

Prepare a CSV with `uid,roi,group,image,x,y,label,class_name,split,coordinate_source`. Image paths may be absolute or relative to the manifest. ROI is the basename before `_nuclei.geojson`. Groups must represent the highest verified independent unit (patient where real metadata establishes it). Splits are train, val, calibration, test or predict and cannot share a group, ROI or image path. IDs/classes follow ARCHITECTURE.md. Unknown semantic targets use label −1 and empty class_name. Training known labels must cover all ten classes. Coordinate sources are `stage1_oof`, `stage1_frozen` or explicitly `gt` for controls. Preserve Stage 1 source score and any extra metadata in extra CSV columns.

Supply `roi_splits.json` such as `{"train":["training_set_primary_roi_001"],"val":["training_set_primary_roi_002"],"calibration":[],"test":[]}` with the **complete actual split** and all zero-proposal ROIs. This illustrative fragment is not an approved data split. The annotation root must contain full `<roi>_nuclei.geojson` files. Do not use the 450-feature control's restricted GT as full-ROI truth.

For full-scale training/evaluation supply a verified `stage1_provenance.json` containing `training_centers_are_oof: true`, `excluded_outer_groups: [actual held-out group IDs]`, and `checkpoint_sha256: "actual 64-character hash"`, plus supporting checkpoint/fold documentation. Multiple detector checkpoints should be fully documented in additional fields; the single hash field identifies the artifact/manifest being asserted. These assertions are checked for presence/coverage, not cryptographically inferred from detector training. Do not populate false assertions to bypass this guard. The currently inspected summary does not establish this provenance for the new outer split.

## Commands (PowerShell, from this directory)

Replace example paths with verified manifests and output directories. The checkpoint path below is the supplied verified UNI2 file. A new full Stage 1 manifest is required; the existing GT sample can only be used with the explicit control flag.

```powershell
python -m pip install -r requirements.txt
python features.py --manifest "full_manifest.csv" --weights "<UNI2_H_WEIGHTS>" --config config.json --out "cache_full" --device cpu --batch 4 --threads 8
python train_eval.py --manifest "full_manifest.csv" --cache "cache_full" --annotations-root "<ANNOTATIONS_ROOT>" --roi-splits roi_splits.json --config config.json --stage1-provenance stage1_provenance.json --seed 17 --out run17
python train_eval.py --manifest "full_manifest.csv" --cache "cache_full" --annotations-root "<ANNOTATIONS_ROOT>" --roi-splits roi_splits.json --config config.json --stage1-provenance stage1_provenance.json --seed 29 --out run29
python train_eval.py --manifest "full_manifest.csv" --cache "cache_full" --annotations-root "<ANNOTATIONS_ROOT>" --roi-splits roi_splits.json --config config.json --stage1-provenance stage1_provenance.json --seed 43 --out run43
```

These are three repetitions of exactly one architecture. Do not compare locked test scores to choose a seed. Cache extraction is a one-time deterministic pass. Existing complete extraction returns only if the input contract matches; training/prediction verify payload checksums. Incomplete caches are preserved and require a new directory. A new prediction manifest requires its own cache extraction, using the same config and checkpoint; the training biology normalizer is loaded from the classifier checkpoint.

Evaluate once on a locked split after all decisions (repeat per predefined seed):

```powershell
python train_eval.py --manifest full_manifest.csv --cache cache_full --annotations-root "<ANNOTATIONS_ROOT>" --roi-splits roi_splits.json --config config.json --stage1-provenance stage1_provenance.json --eval-checkpoint run17/best.pt --eval-split test --out test17
python predict.py --manifest full_manifest.csv --cache cache_full --checkpoint run17/best.pt --out predictions17
```

`predict.py` exports all rows supplied, preserving metadata. To export only test or new inference cases, use a separate test/predict manifest and matching extraction cache. Full-ROI evaluator gets the locked ROI list separately so absent proposal ROIs are retained; the JSON exporter emits only ROIs represented by input rows. Add explicit empty prediction files if an external submission interface requires them; never drop those ROIs from evaluation.

Optional calibration: evaluate `--eval-split calibration` into a new output directory, then run `python calibrate.py --predictions calibration17/predictions.npz --assert-calibration-split --out temperature17.json`. Pass the recorded scalar through `--temperature` to both evaluation and prediction. Do not fit T to locked test labels. Enriched pilot ECE is not calibration evidence for natural prevalence.

## Package map and checks

dataset.py validates ontology, identity and split leakage; crop.py defines image transformation; uni2.py strictly loads the encoder; biology.py implements the exact 16 measurements; features.py builds immutable caches; model.py defines the head and optional direct-image wrapper; loss.py defines the loss; numerical.py implements metrics and training; evaluation.py wraps the byte-identical local V17 evaluator; train_eval.py trains and evaluates full ROIs; predict.py preserves proposal geometry; calibrate.py fits scalar T. Config is fixed in config.json.

VERIFICATION.json records code compilation, all four CLI entrypoints, zero-initialized baseline equivalence, parameter count, strict loading of the locally trained head, finite image-edge features, full-ROI zero-proposal counts, class remapping, original serialization and payload-corruption rejection. The source experiment logs independently cover 30 ten-epoch runs. Integrity fixtures are not performance experiments. The full external-data pipeline is not claimed executed. Caches, labels and metadata are trusted research inputs; this package is not a clinical deployment service.
