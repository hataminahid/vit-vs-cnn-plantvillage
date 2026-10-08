import random

import numpy as np
import torch

SEED = 42

CFG = dict(
    DATASET_NAME="BrandonFors/Plant-Diseases-PlantVillage-Dataset",
    DATASET_CONFIG=None,  # this dataset has a single default config; no subset name needed
    MAX_TRAIN_PER_CLASS=150,   # samples per class used for training (lower = faster)
    MAX_TEST_PER_CLASS=30,     # samples per class used for eval / disagreement analysis
    EPOCHS=3,                  # matched training budget for both models
    BATCH_SIZE=32,
    LR=3e-5,
    WEIGHT_DECAY=1e-4,
    N_PER_CLASS_OCC=4,         # images per diseased class used in the occlusion experiment
    SEVERITIES=[0.25, 0.5, 0.75],
    MIN_DISEASE_PIXELS=80,     # skip images whose heuristic disease mask is too small
    NUM_WORKERS=2,
    CHECKPOINT_DIR="checkpoints",
    RESULTS_DIR="results",
    DATA_EFFICIENCY_SIZES=[20, 50, 100, 150],  # samples-per-class points for the stretch sweep
    DATA_EFFICIENCY_EPOCHS=2,                  # shorter matched budget per sweep point (keeps runtime sane)
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
