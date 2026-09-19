from __future__ import annotations

CLASSES = (
    "tumor",
    "lymphocyte",
    "plasma_cell",
    "histiocyte",
    "melanophage",
    "neutrophil",
    "stroma",
    "epithelium",
    "endothelium",
    "apoptosis",
)
NUM_CLASSES = len(CLASSES)
EMBED_DIM = 1536
TIER_A_DIM = 16
UNI2_TOKEN_COUNT = 265
UNI2_REGISTER_TOKENS = 8
UNI2_SPATIAL_START = 9
UNI2_GRID_SIZE = 16
UNI2_PATCH_SIZE = 14
DEFAULT_FOV = 96
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
