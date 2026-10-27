"""Centralized seed-setting for reproducibility.

Convention: every notebook calls `set_all_seeds()` in its first code cell.
Default seed = 42, fixed across the project so any submission is rebuildable.

For Kaggle notebook submissions (which run in isolation and can't import this file
directly), copy the body of `set_all_seeds` inline into the notebook's first cell.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_all_seeds(seed: int = DEFAULT_SEED) -> int:
    """Set seeds for `random`, `numpy`, and (if installed) `torch`.

    Returns the seed value used so callers can log it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism flags — small perf cost, big reproducibility gain
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed in this env; fine for non-ML scripts
        pass

    return seed


if __name__ == "__main__":
    s = set_all_seeds()
    print(f"All seeds set to {s}")
