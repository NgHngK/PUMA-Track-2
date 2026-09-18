# Architecture

This file is a short note about the model structure used in the current code. The project is still under development.

## Stage 1

Stage 1 receives a full 1024×1024 RGB ROI and predicts nucleus candidates. The encoder has five ConvNeXt-style feature levels. The deepest feature also passes through a small transformer context block. An FPN combines the encoder levels into a 128-channel feature map.

The FPN feature is upsampled and fused with a small RGB spatial branch. This gives a 32-channel full-resolution feature map. The detector predicts a heatmap and point features for offset, quality, uncertainty, coarse nucleus type, and the Stage-1 prior used later by Stage 2.

Five out-of-fold models are trained. Each ROI is predicted by a fold model that did not train on that ROI. This keeps the Stage-2 training features out-of-fold.

## Tissue branch

The tissue model reads the frozen Stage-1 FPN feature and predicts six tissue labels: background, stroma, blood vessel, tumor, epidermis, and necrosis. The predicted tissue probabilities are converted into local context features for Stage 2.

## Stage 2

Stage 2 is `BioContextRefine`. A frozen UNI2-h encoder extracts appearance features from three crop sizes. A small BioMask model predicts a nucleus mask and provides shape and stain features. The model also receives spatial features, tissue context, Stage-1 class priors, and detector-quality features.

Each context source passes through a small residual module with a learned gate. The fused feature is used for the ten-class local prediction and candidate-validity prediction. A graph module then uses nearby candidates to add a final correction to the local class logits.

## Risk calibration

Out-of-fold Stage-2 predictions are used to fit the risk model and one global acceptance threshold. This step is used to reject low-confidence candidates before the final output is written.
