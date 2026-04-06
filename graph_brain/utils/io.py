"""Checkpoint save/load for graph state."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a state dict to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a state dict from disk."""
    return torch.load(path, map_location=device, weights_only=False)
