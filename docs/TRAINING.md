# Training

The current config uses five folds. Stage 1 runs for at most 60 epochs with validation every 5 epochs. The OOF folds use validation binary F1 for checkpoint selection and stop when there has been no improvement for 15 epochs. The final all-data Stage-1 run has no validation fold, so it runs the full 60 epochs.

Stage 1 uses a configured micro-batch size of 8 and an effective batch size of 8. EMA weights are saved for the selected checkpoint. The main Stage-1 outputs are the fold checkpoints, out-of-fold detector candidates, and GT-aligned features used to build Stage 2.

The tissue model is configured for 100 epochs. It reads frozen Stage-1 FPN features and trains a six-class tissue decoder with a weighted cross-entropy and Dice loss.

Stage 2 is configured for 100 epochs. The first 70 epochs train the local BioContextRefine model. The remaining epochs train the graph refinement phase. Stage 2 does not use early stopping in the current code. EMA weights are saved at the end of training.

Training settings are stored in `configs/train_config.json`. Changing settings that belong to the Stage-1 or Stage-2 feature contract changes the config fingerprint, so an incompatible checkpoint is not resumed by mistake.

This training setup is still being developed and may change in later experiments.
