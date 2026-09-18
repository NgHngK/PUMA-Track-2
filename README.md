# PUMA BioContext-Refine

> **Status:** This project is still under development.

## Project overview

This repository contains my current pipeline for nuclei detection, nuclei classification, and tissue segmentation on the PUMA data. I split the work into two main stages. Stage 1 finds nucleus locations in the full 1024×1024 ROI. A small tissue branch predicts the tissue map from Stage-1 features. Stage 2 then uses image appearance, nucleus shape, spatial context, tissue context, and Stage-1 predictions to classify each detected nucleus. The code also contains calibration and submission packaging for the final pipeline.

The main reason for using two stages is that detection and fine-grained classification have different problems. Stage 1 needs to find almost every nucleus, so it is trained as a high-recall point detector. Stage 2 receives these candidates and has more information for deciding the final nucleus class and whether a candidate should be kept.

## Dataset

The training configuration expects 1024×1024 RGB TIFF ROIs in `Dataset/01_training_dataset_tif_ROIs`, nucleus annotations in `Dataset/01_training_dataset_geojson_nuclei`, and tissue annotations in `Dataset/01_training_dataset_geojson_tissue`. The public-data checks in the config expect 205 ROIs and 97,378 annotated nuclei. The dataset itself is not included in this repository, so these numbers come from the project configuration rather than files stored in the ZIP.

The nucleus task has ten classes: tumor, lymphocyte, plasma cell, histiocyte, melanophage, neutrophil, stroma, endothelium, epithelium, and apoptosis. The tissue branch uses six labels: background, stroma, blood vessel, tumor, epidermis, and necrosis. During preprocessing, the TIFF images and annotations are converted into memory-mapped NumPy files. The code also creates five folds with a deterministic balancing method so that ROI counts and class coverage are kept as even as possible.

The five folds are important because Stage 2 should not learn from Stage-1 predictions made by a detector that already trained on the same ROI. Each ROI is therefore predicted by the Stage-1 fold model that did not use that ROI for training. These out-of-fold predictions are later used to build the Stage-2 candidate set.

## Model architecture

Stage 1 is a native 1024×1024 point detector. Its encoder is a compact ConvNeXt-style hierarchy with five feature levels. With the current base width of 64 channels, the encoder uses 64, 128, 256, 512, and 768 channels with block depths of 2, 2, 6, 4, and 2. The deepest feature map is passed through two transformer encoder layers to add global context. An FPN then brings the multi-scale encoder features into a 128-channel semantic feature map.

The detector does not stop at the lower-resolution FPN output. A native-resolution fusion block upsamples the semantic feature and combines it with a small spatial branch that reads the original RGB image. This produces a 32-channel full-resolution feature map. From this map, Stage 1 predicts a nucleus heatmap and point-level outputs for offsets, localization quality, uncertainty, coarse nucleus type, and the feature vector used as the Stage-1 prior. The current loss combines heatmap, offset, quality, uncertainty, and coarse-type losses with separate weights from the training config.

The tissue branch reuses the frozen 128-channel Stage-1 FPN feature. A small convolutional decoder predicts six tissue classes and upsamples the result to the ROI size. This branch is trained with cross-entropy and Dice loss. Its output is not used as a final nucleus label. Instead, local tissue probabilities, local averages, distances to tissue regions, and tissue entropy are converted into context features for Stage 2.

Stage 2 is implemented as `BioContextRefine`. The main appearance feature comes from a frozen UNI2-h encoder. The current setup uses three crop views, V2, V3, and V4, which correspond to 64, 128, and 256 pixel views. Each view produces a pooled appearance feature, and the model learns a weight for combining the three views into one 256-dimensional anchor feature.

A small BioMask network also works on a 96×96 crop around each candidate. It uses RGB together with hematoxylin and hematoxylin-gradient channels to predict a soft nucleus mask and a mask-confidence value. The soft mask is then used to calculate morphology and stain features such as area, axis lengths, eccentricity, perimeter, circularity, compactness, intensity statistics, gradient statistics, and local nucleus-to-ring contrast.

The appearance anchor is refined by four types of extra evidence: biology features, spatial features, tissue context, and the Stage-1 class prior. Each source has its own small residual network and a learned gate, so weak context does not have to change the appearance feature strongly. The local classifier predicts the ten nucleus classes, while a separate validity head estimates whether a candidate is likely to be a real nucleus.

After local classification, a graph module refines the prediction with neighboring candidates. The current graph uses up to 12 neighbors with a radius cap of 160 pixels. It reads neighboring embeddings, class probabilities, relative positions, and candidate validity. The graph predicts a gated correction that is added to the local class logits. This keeps the local prediction as the main signal while still allowing nearby cells to provide context.

## Training setup

The current Stage-1 config uses five folds, a maximum of 60 epochs, validation every 5 epochs, and early-stopping patience of 15 epochs for the out-of-fold models. The configured micro-batch and effective batch size are both 8. The final all-data Stage-1 model has no held-out fold, so it runs the full configured epoch schedule instead of using validation early stopping. EMA weights are used for the saved model state.

The tissue model is configured for 100 epochs. Stage 2 is also configured for 100 epochs. Its first 70 epochs train the local classifier and context modules, while the remaining epochs train the graph refinement phase. Stage 2 does not use early stopping in the current code. The pipeline also includes OOF risk calibration and a global acceptance threshold for the final candidate filtering step.

## Current Stage-1 results

The result files included in this repository are Stage-1 five-fold training results. They are stored inside `puma_stage1_full_charts.zip`. This ZIP does not contain completed Stage-2 or final end-to-end evaluation results, so the numbers below should be read only as Stage-1 nucleus-detection results.

| Fold | Best epoch | Precision | Recall | F1 | Mean localization error (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 35 | 0.7243 | 0.9871 | 0.8355 | 2.3420 |
| 1 | 60 | 0.7115 | 0.9926 | 0.8288 | 2.0734 |
| 2 | 60 | 0.6748 | 0.9940 | 0.8039 | 2.1221 |
| 3 | 60 | 0.6897 | 0.9901 | 0.8131 | 2.1350 |
| 4 | 60 | 0.6748 | 0.9921 | 0.8032 | 2.2473 |
| **Mean** | **55** | **0.6950** | **0.9912** | **0.8169** | **2.1840** |

The main pattern is very high recall and lower precision. Across the best checkpoints, mean recall is about 99.1%, while mean precision is about 69.5%. The average fold has about 19,304 true positives, 8,483 false positives, and only 172 false negatives. This means Stage 1 is already finding almost all annotated nuclei, but it also keeps many extra candidates. For this pipeline, that behavior is useful only if Stage 2 can later remove false detections without losing many true nuclei. The validity head and risk-calibration code are designed for this part, but their final effect cannot be measured from the result files currently included here.

Training also shows a clear improvement in localization. The mean localization error across the five folds is about 4.14 pixels at epoch 5, about 2.52 pixels at epoch 30, and about 2.18 pixels at the selected best checkpoints. This is small compared with the 15-pixel matching radius used by the project. The detector therefore becomes much better at placing the predicted point near the annotated nucleus center as training continues.

The F1 curve improves quickly at first and then becomes much slower. The mean five-fold F1 is about 0.317 at epoch 5, 0.759 at epoch 20, and 0.806 at epoch 30. After epoch 30, the gains are much smaller. Fold 0 reaches its best F1 at epoch 35 and later triggers early stopping, while folds 1 to 4 continue to improve slowly until the end of the 60-epoch schedule. This suggests that the folds do not converge at exactly the same speed, so keeping validation-based checkpoint selection is useful.

Another important point is that recall stays close to 99% while precision changes more strongly during training. The deployment threshold is currently low at 0.06, so Stage 1 is biased toward keeping possible nuclei instead of rejecting uncertain detections early. The current Stage-1 results are therefore better interpreted as a candidate-generation result than as the final quality of the full pipeline.

## Running the project

The project can be installed from the repository root with the dependencies in `pyproject.toml`. The first notebook, `notebooks/01_Stage1_Preprocess_Train.ipynb`, performs preprocessing, Stage-1 fold training, OOF prediction generation, and final Stage-1 training. The second notebook, `notebooks/02_Stage2_Train.ipynb`, builds the tissue and Stage-2 data, trains the remaining models, fits risk calibration, and packages the deployment files.

Basic code checks can be run with:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
```

The deployment package can be created with:

```bash
python scripts/package_deployment.py --config configs/resolved_config.json
```

## Work in progress

This project is still under development. Stage 1 has five-fold training results in the current repository, but the tissue branch, Stage 2 refinement, risk calibration, and final end-to-end evaluation are still being developed and tested. The architecture, training settings, and result analysis in this README describe the current code version and may change in later experiments.
