"""Reproducible random number generation."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Set all random seeds for reproducibility. Returns a torch Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
