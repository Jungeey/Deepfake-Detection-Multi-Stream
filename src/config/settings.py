"""Configuration settings for the multi-stream deepfake detection project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
METADATA_DIR = DATA_ROOT / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, 
                 MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "faceforensics": {
        "name": "FaceForensics++_C23",
        "subfolders": [
            "DeepFakeDetection",
            "Deepfakes", 
            "Face2Face",
            "FaceShifter",
            "FaceSwap",
            "NeuralTextures",
            "original"
        ]
    },
    "celeba": {
        "name": "CelebDF",
        "subfolders": [
            "Celeb-real",
            "Celeb-synthesis", 
            "YouTube-real"
        ]
    }
}

# Preprocessing parameters
IMAGE_SIZE = 224
EYE_PATCH_SIZE = 128
MOUTH_PATCH_SIZE = 128
FACE_MESH_CONFIDENCE = 0.5

# Temporal parameters
FRAMES_PER_SEQUENCE = 20
STRIDE = 10  # Overlap between sequences

# MPS settings
USE_MPS = True  # Will be checked at runtime

# Class mappings
CLASS_MAPPING = {
    "real": 0,
    "fake": 1
}

# Generation method mapping
METHOD_MAPPING = {
    "DeepFakeDetection": "DeepFakeDetection",
    "Deepfakes": "Deepfakes",
    "Face2Face": "Face2Face", 
    "FaceShifter": "FaceShifter",
    "FaceSwap": "FaceSwap",
    "NeuralTextures": "NeuralTextures",
    "original": "original",
    "Celeb-real": "Celeb-real",
    "Celeb-synthesis": "Celeb-synthesis",
    "YouTube-real": "YouTube-real"
}