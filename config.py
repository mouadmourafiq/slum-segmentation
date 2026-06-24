"""
config.py — Central configuration for the Slum Segmentation pipeline.

All tuneable paths, hyperparameters, and hardware settings live here.
Modify this file to adapt the pipeline to new datasets or machines.
"""

import os
import torch

# ================================================================
#  PATHS
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MULTI_CITY_DATA_DIR = "/Users/oussamalouat/Documents/data set"
MULTI_CITY_FILES = {
    "Colombia": (
        os.path.join(MULTI_CITY_DATA_DIR, "Colombia", "Medellin_40cm.tif"),
        os.path.join(MULTI_CITY_DATA_DIR, "Colombia", "Medellin_ground_truth.tif")
    ),
    "Nigeria Makoko": (
        os.path.join(MULTI_CITY_DATA_DIR, "Nigeria Makoko", "Makoko_50cm_small.tif"),
        os.path.join(MULTI_CITY_DATA_DIR, "Nigeria Makoko", "Makoko_50cm_small_ground_truth.tif")
    ),
    "sudan ElGeneina": (
        os.path.join(MULTI_CITY_DATA_DIR, "sudan ElGeneina", "ElGeneina_40cm.tif"),
        os.path.join(MULTI_CITY_DATA_DIR, "sudan ElGeneina", "ElGeneina_40cm_ground_truth.tif")
    ),
    "Casablanca": (
        os.path.join(DATA_DIR, "image_cropped.tif"),
        os.path.join(DATA_DIR, "mask_cropped.tif")
    )
}

# Used by inference.py and evaluate_prediction.py
INPUT_IMAGE_PATH = os.path.join(DATA_DIR, "image_cropped.tif")
INPUT_MASK_PATH = os.path.join(DATA_DIR, "mask_cropped.tif")

# Generated tiles directory
TILES_DIR = os.path.join(DATA_DIR, "tiles")
TRAIN_DIR = os.path.join(TILES_DIR, "train")
VAL_DIR = os.path.join(TILES_DIR, "val")
TEST_DIR = os.path.join(TILES_DIR, "test")

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
PREDICTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "predicted_casablanca_crop.tif")

# ================================================================
#  DEVICE SELECTION — MPS → CUDA → CPU
# ================================================================
def get_device() -> torch.device:
    """
    Dynamically selects the best available accelerator.
    Priority: Apple MPS (Metal) → NVIDIA CUDA → CPU.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[DEVICE] Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[DEVICE] Using CPU")
    return device


DEVICE = get_device()

# ================================================================
#  DATA PREPROCESSING
# ================================================================
TILE_SIZE = 512              # Patch size in pixels (width = height)
STRIDE = 128                 # Very small stride for high overlap (great for small images like Colombia)
MIN_VALID_RATIO = 0.01       # Minimum fraction of non-black pixels to keep a tile
RANDOM_SEED = 42
MAX_TILES_PER_CITY = 500     # Cap the number of tiles per city to ensure balance


# Mask normalisation
# Set INVERT_MASK = True if, in your mask file, slum = 0 (black)
# and background = 255 (white). The script will flip them so that
# the final binary mask is: 0 = background, 1 = slum.
INVERT_MASK = False

# ================================================================
#  DATA SPLIT STRATEGY (Geographical Holdout)
# ================================================================
TRAIN_CITIES = ["Colombia", "Nigeria Makoko"]
TEST_CITIES = ["Casablanca"]
# Percentage of training city tiles to hold out for validation
VAL_RATIO = 0.15

# ================================================================
#  MODEL ARCHITECTURE
# ================================================================
# 1. ARCHITECTURE & HYPERPARAMETERS
MODEL_ARCH = "moe"           # "unet", "segformer", or "moe"
UNET_ENCODER = "resnet34"    # Only used if MODEL_ARCH="unet"
IN_CHANNELS = 3              # Our data is RGB
NUM_CLASSES = 1              # Binary segmentation (Slum vs Non-Slum)

# ================================================================
#  TRAINING HYPERPARAMETERS
# ================================================================
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2          # Très bien pour contrer l'overfitting
EARLY_STOPPING_PATIENCE = 15

NUM_WORKERS = 0              # Must be 0 for MPS backend compatibility

# Loss weights
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ImageNet normalisation (used by pretrained encoders)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ================================================================
#  INFERENCE
# ================================================================
INFERENCE_TILE_SIZE = 512
INFERENCE_OVERLAP = 0        # No overlap — write directly to disk (zero RAM usage)
CONFIDENCE_THRESHOLD = 0.9   # Seuil très strict pour filtrer les faux positifs
