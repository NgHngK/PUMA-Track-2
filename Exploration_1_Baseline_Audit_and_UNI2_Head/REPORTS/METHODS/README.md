# PUMA Stage 2 research implementation

Stage 1 remains external and immutable. This package classifies ten nucleus types. The research report distinguishes executed pilots from unvalidated deployment recommendations.

Use Python 3.10+ and install `requirements.txt` in a separate environment. Supply the existing UNI2-h checkpoint; no automatic weight downloads occur. GPU execution requires a CUDA-enabled PyTorch build.

## Build a clean semantic control

```powershell
python dataset.py --root "<DATASET_ROOT>" --out manifests/gt.csv
```

This creates a new **ROI-separated development split**, not the legacy folds and not a patient-verified split. For verified patient groups, add `--group-csv patients.csv` with columns `roi,patient_id`. For a publication experiment, supply a locked manifest retaining existing train/validation/calibration/test group assignments rather than regenerating folds. The generated manifest contains GT centers and must not be called a Stage-1-centered end-to-end result.

Canonical IDs: tumor 0, lymphocyte 1, plasma_cell 2, histiocyte 3, melanophage 4, neutrophil 5, stroma 6, epithelium 7, endothelium 8, apoptosis 9. Never import integer labels without verifying names. A custom manifest needs `uid,roi,group,image,x,y,label,class_name,split,coordinate_source`; coordinates are level-0 ROI pixels. The permitted sources are `gt,stage1_oof,stage1_frozen`.

## Train paired baselines

```powershell
python train_eval.py --manifest manifests/gt.csv --weights "<UNI2_H_WEIGHTS>" --out runs/linear_ce --mode linear --sampler natural --tau 0 --epochs 20
python train_eval.py --manifest manifests/gt.csv --weights "<UNI2_H_WEIGHTS>" --out runs/linear_la --mode linear --sampler natural --tau 1 --epochs 20
python train_eval.py --manifest manifests/gt.csv --weights "<UNI2_H_WEIGHTS>" --out runs/linear_balanced --mode linear --sampler balanced --tau 0 --epochs 20
```

The image-based linear baseline intentionally recomputes frozen features under geometric augmentations. For fast fixed-view head training use the included research feature-extraction/probe scripts and record the change in augmentation contract. Do not train LoRA from final cached embeddings: they disconnect the encoder gradient path.

The candidate LoRA run is an ablation, not a validated improvement:

```powershell
python train_eval.py --manifest manifests/gt.csv --weights "<UNI2_H_WEIGHTS>" --out runs/lora --mode lora --rank 8 --alpha 16 --last-blocks 4 --head-lr 0.001 --backbone-lr 0.00001 --sampler natural --tau 1 --batch-size 8 --accum 8 --epochs 20 --patience 5 --amp --checkpointing
```

Select the sampler/loss using a separate development comparison. The example tau=1 is a candidate. Use the same recipe for matched frozen and LoRA controls. All runs log fixed-10 Macro-F1, per-class precision/recall/F1/support/confusion, ECE15, NLL, sampling exposure and positive-example classifier gradient norms. `best.pt` contains trainable weights plus configuration and foundation-checkpoint SHA256; it is not a standalone foundation model and not an optimizer-resume checkpoint. Training refuses to overwrite an existing history file.

## Use frozen Stage 1 coordinates

For training and conditional semantic evaluation, `attach_stage1.py` performs geometry-only one-to-one association between the GT manifest and supplied OOF proposals:

```powershell
python attach_stage1.py --gt-manifest manifests/gt.csv --proposals stage1_oof.csv --provenance stage1_provenance.json --radius 15 --out manifests/stage1.csv
```

The 15-pixel radius is an explicit example from historical reports; lock the radius required by your evaluator. The provenance JSON must contain `coordinate_units: "roi_level0_pixels"`, `checkpoint_sha256`, `training_centers_are_oof: true`, and `excluded_outer_groups` covering all nontraining groups. These are caller assertions that require documentary verification. Outer-fold isolation is stronger than simple five-fold OOF exclusion of each target ROI. The script reports unmatched counts; excluding those from conditional classification is **not** end-to-end detection evaluation. Do not drop false positives or missed nuclei from the official evaluation.

For inference, classify **every** Stage-1 proposal using CSV columns `uid,roi,image,x,y,score`:

```powershell
python predict.py --proposals stage1_predictions.csv --weights "<UNI2_H_WEIGHTS>" --checkpoint runs/linear_la/best.pt --out predictions.csv
```

The original coordinates and Stage-1 score are preserved. Ten semantic probabilities and the predicted class are appended. No new ranking, suppression, thresholds or tissue segmentation is applied. The final official evaluator is external; this package does not claim to reproduce its matching or aggregation.

## Calibration and held-out evaluation

Set aside a separate naturally sampled calibration group. Run `train_eval.py --eval-checkpoint ... --eval-split calibration` with the exact saved architecture flags, then use `calibrate.py --predictions ... --out temperature.json --assert-calibration-split`. This fits a bounded scalar temperature only. Pass the saved value to `predict.py --temperature ...`. Never fit on validation used for model selection and call the result independently calibrated; never fit on test. Scalar temperature preserves argmax and cannot fix tail recall. Raw logit-adjusted classifiers target a balanced decision rule; their probabilities are not automatically calibrated to natural prevalence. Use `LogitAdjustedCE.population_logits` with a prespecified target prior when that probability interpretation is required, then validate calibration separately.

The package has CPU integrity checks and real-data pilots. CUDA throughput, full-data LoRA, external validation, clinical utility and SOTA performance remain separate validation requirements.
