# PUMA output format

The current deployment code writes two outputs for each 1024×1024 ROI. The nucleus prediction is written to `/output/melanoma-10-class-nuclei-segmentation.json`. The tissue prediction is written as a TIFF file inside `/output/images/melanoma-tissue-mask-segmentation/`.

The tissue TIFF is a 1024×1024 `uint8` class map. The writer also adds the metadata required by the current submission code. `scripts/validate_submission.py` checks the JSON output, tissue pixels, and TIFF metadata before packaging or submission.

`scripts/package_deployment.py` checks the saved model states and config fingerprints before copying the required model files into `models/`. This helps prevent mixing checkpoints from different training configurations.

This file describes the current project output contract and may change while the project is still under development.
