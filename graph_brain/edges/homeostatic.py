"""Homeostatic synaptic scaling (Turrigiano 2008).

Slow multiplicative renormalization of incoming weights per node.
Nodes that are too active get their incoming weights scaled down.
Nodes that are too quiet get their incoming weights scaled up.

Uses dst_ptr for efficient per-node weight access.
"""

from __future__ import annotations

import torch

from graph_brain.config import HomeostaticConfig
from graph_brain.core.graph import EdgeStore, NodeState


class HomeostaticScaling:
    """Slow homeostatic synaptic scaling to maintain target activity."""

    def __init__(self, config: HomeostaticConfig):
        self.cfg = config

    def update(self, store: EdgeStore, node_state: NodeState, dt: float) -> None:
        """Scale incoming weights per node toward target activity.

        For each post-synaptic node i:
            ratio = target_rate / max(activity_ema[i], eps)
            scale = 1 + lr * (ratio - 1)
            all incoming weights to i *= scale
        """
        if not self.cfg.enabled or store.n_edges == 0:
            return

        N = node_state.n_nodes
        lr = dt / self.cfg.tau

        # Compute per-node scaling factor
        eps = 1e-8
        ratio = self.cfg.target_rate / (node_state.activity_ema + eps)
        scale = 1.0 + lr * (ratio - 1.0)
        scale.clamp_(0.9, 1.1)  # prevent extreme single-step changes

        # Apply scale to each edge based on its destination node
        edge_scale = scale[store.dst.long()]
        store.weight *= edge_scale
