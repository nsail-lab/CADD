"""Utility file for src toolkit."""
import os
import random

import numpy as np
import torch

WHISPER_MODEL_WEIGHTS_PATH = "src/assets/tiny_enc.en.pt"
MEL_FILTERS_PATH = "src/assets/mel_filters.npz"
IN_THE_WILD_TEST_IDS_PATH = "src/assets/ITW_m.npy"

def set_seed(seed: int) -> None:
    """Fix PRNG seed for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
