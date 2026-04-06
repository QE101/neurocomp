"""Dopamine system: global reward signal for prediction success.

The graph's motivational drive. Without it, the energy-optimal strategy
is silence (the French approach). Dopamine says: "predicting correctly
is rewarding — keep doing it."

Mechanism:
    At each pattern transition (A→B or B→A), measure whether the system
    predicted correctly. If yes → dopamine burst. If no → dopamine dip.

    During a dopamine burst:
    - λ_activity is temporarily reduced graph-wide (nodes can be active)
    - Learning rate is temporarily boosted (lock in what worked)
    - Excitability increases (predictions are generated more freely)

    The burst decays exponentially back to baseline. The system alternates
    between "rewarded, active, learning" and "baseline, efficient, quiet."

    This is event-driven, not continuous. Fires at pattern transitions,
    not every step. The system creates its own rhythm of activity.

Dopamine level dynamics:
    On successful prediction: dopamine += burst_size
    Every step: dopamine *= decay_rate
    Effect: effective_lambda = lambda_base * (1 - dopamine_level)
    At dopamine=0: full sparsity pressure (default)
    At dopamine=1: zero sparsity pressure (fully active)
"""

from __future__ import annotations

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.types import NodeType


class DopamineSystem:
    """Global dopamine signal driven by prediction success."""

    def __init__(
        self,
        n_nodes: int,
        device: str = "cpu",
        burst_size: float = 0.8,
        decay_rate: float = 0.99,
        learning_boost: float = 3.0,
    ):
        """
        Args:
            burst_size: how much dopamine is released on success (0-1)
            decay_rate: per-step decay (0.99 = half-life ~70 steps)
            learning_boost: multiplier on learning rate during burst
        """
        self.burst_size = burst_size
        self.decay_rate = decay_rate
        self.learning_boost = learning_boost

        # Global dopamine level (0 = baseline, 1 = maximum burst)
        self.level = torch.tensor(0.0, device=device)

        # Track prediction quality for triggering bursts
        self._prev_pattern_error = None
        self._pattern_step_count = 0

    @property
    def is_bursting(self) -> bool:
        return self.level.item() > 0.1

    def effective_lambda(self, base_lambda: float) -> float:
        """Compute effective activity penalty, reduced during dopamine burst.

        At dopamine=0: returns base_lambda (full sparsity)
        At dopamine=1: returns 0 (no sparsity penalty — fully free to be active)
        """
        return base_lambda * (1.0 - self.level.item())

    def effective_learning_rate(self, base_lr: float) -> float:
        """Boost learning rate during dopamine burst."""
        return base_lr * (1.0 + self.level.item() * (self.learning_boost - 1.0))

    def step(self) -> None:
        """Decay dopamine level each timestep."""
        self.level *= self.decay_rate

    def on_pattern_transition(
        self,
        graph: NeuromorphicGraph,
        input_nodes: Tensor,
        prev_pattern_output: Tensor = None,
    ) -> float:
        """Called at each pattern transition (A→B or B→A).

        Fixed trigger: requires ACTIVE PREDICTION, not just low error.

        Success = the system was ACTIVE during the PREVIOUS pattern (made a
        prediction) AND output is LOWER during the CURRENT pattern transition
        (the prediction was confirmed, so the system could suppress).

        This can't be faked by silence: a silent system has low output during
        BOTH patterns, so prev_output is low → no burst. Only a system that
        was active (predicting) and then correctly suppressed gets rewarded.

        Returns the dopamine delta (positive = burst, negative = dip).
        """
        ns = graph.node_state

        current_output = ns.output[input_nodes.long()].mean().item()

        if prev_pattern_output is not None:
            prev_output = prev_pattern_output.mean().item()

            # Was the system ACTIVE during the previous pattern?
            # (output > threshold means it was doing something, not just silent)
            was_active = prev_output > 0.5

            # Did output DECREASE at transition? (prediction confirmed → suppress)
            output_dropped = current_output < prev_output * 0.8

            # Success: was active AND output dropped (predicted then suppressed)
            if was_active and output_dropped:
                delta = self.burst_size
                self.level = (self.level + delta).clamp(0.0, 1.0)
            elif was_active and current_output > prev_output * 1.2:
                # Was active but output INCREASED (surprised)
                delta = -0.3
                self.level = (self.level + delta).clamp(0.0, 1.0)
            elif not was_active:
                # Was silent — no reward, slight penalty (complacency tax)
                delta = -0.05
                self.level = (self.level + delta).clamp(0.0, 1.0)
            else:
                delta = 0.0
        else:
            delta = 0.0

        self._prev_pattern_error = current_output
        self._pattern_step_count = 0
        return delta

    def reset(self) -> None:
        """Reset dopamine state."""
        self.level = torch.tensor(0.0, device=self.level.device)
        self._prev_pattern_error = None
        self._pattern_step_count = 0
