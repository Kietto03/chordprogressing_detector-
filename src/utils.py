"""
Utility functions (PR5).

Currently focused on reproducibility (centralized seeding).
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch for reproducible runs.

    When deterministic=True (default for CPU), also forces cuDNN deterministic mode.
    This can slow down GPU training but is required for bit-level reproducibility
    on CUDA (within the limits of floating-point non-determinism).

    Call this early in train.py, dataset loading, and (if needed) app.py.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For MPS (Apple Silicon) we can't fully control determinism yet,
    # but setting the torch seed is still useful.
    if torch.backends.mps.is_available():
        # MPS seed is tied to the torch manual seed above
        pass
