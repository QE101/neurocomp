"""Intrinsic plasticity: homeostatic adaptation of node threshold and gain.

Each node adjusts its own threshold and gain to maintain a target firing rate.
This is slower than synaptic plasticity — operates on the intrinsic parameters
rather than connection weights.
"""

from __future__ import annotations

import torch

from graph_brain.config import NodeConfig
from graph_brain.core.graph import NodeState


class IntrinsicPlasticity:
    """Homeostatic intrinsic plasticity: threshold and gain adaptation."""

    def __init__(self, config: NodeConfig):
        self.cfg = config

    def update(self, node_state: NodeState) -> None:
        """Update threshold and gain based on deviation from target rate.

        If activity_ema > target: increase threshold, decrease gain
        If activity_ema < target: decrease threshold, increase gain
        """
        if not self.cfg.ip_enabled:
            return

        lr = self.cfg.ip_learning_rate
        target = self.cfg.ip_target_rate

        # Error signal: positive if too active, negative if too quiet
        error = node_state.activity_ema - target

        # Threshold moves up when too active (harder to fire)
        node_state.threshold += lr * error

        # Gain moves opposite to error (amplify when quiet, suppress when active)
        node_state.gain -= lr * error
        node_state.gain.clamp_(min=0.1, max=10.0)  # prevent degenerate gains
