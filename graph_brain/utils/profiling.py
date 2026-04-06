"""Timing and memory profiling utilities."""

import time
from contextlib import contextmanager
from typing import Optional

import torch


class StepTimer:
    """Accumulates timing for named code sections within a simulation step."""

    def __init__(self):
        self._times: dict[str, list[float]] = {}

    @contextmanager
    def section(self, name: str):
        """Time a named section. Usage: with timer.section('message_passing'): ..."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        self._times.setdefault(name, []).append(elapsed)

    def summary(self, last_n: Optional[int] = None) -> dict[str, float]:
        """Return mean time per section (ms). Optionally over last N entries."""
        result = {}
        for name, times in self._times.items():
            subset = times[-last_n:] if last_n else times
            result[name] = sum(subset) / len(subset) if subset else 0.0
        return result

    def reset(self):
        self._times.clear()


def gpu_memory_mb() -> float:
    """Current GPU memory usage in MB. Returns 0 if no CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def gpu_memory_summary() -> str:
    """Human-readable GPU memory summary."""
    if not torch.cuda.is_available():
        return "No CUDA available"
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return f"GPU: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved"
