"""Shared training utilities."""

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Sets seeds for Python's ``random``, NumPy, PyTorch CPU, and (when a CUDA
    device is present) all CUDA devices.  Also enables cuDNN deterministic
    mode and disables benchmark mode so that CUDA kernels are chosen
    deterministically across runs.

    Args:
        seed: Integer seed value to apply to every RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
