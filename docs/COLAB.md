# Colab setup

This project can be run from an extracted folder on Google Drive. The notebooks look for the project folder and add its `src` directory to Python before importing `puma_pipeline`.

If the automatic path search does not find the project, set `CODE_ROOT_OVERRIDE` in the notebook to the folder that contains `src`, `configs`, and `notebooks`.

Run the first notebook from top to bottom for preprocessing and Stage 1. Run the second notebook after the Stage-1 outputs have been created. The generated data and checkpoints are saved outside the source folder using the paths in `configs/train_config.json`.

This project is still under development, so notebook cells and training settings may change between versions.
