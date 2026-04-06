"""Short-term plasticity: Tsodyks-Markram model.

Per-edge facilitation and depression dynamics.

On each presynaptic spike/activity:
    u (facilitation) increases → more vesicles released
    x (depression) decreases → fewer vesicles available

Between spikes:
    u decays back to U_baseline
    x recovers back to 1.0

release_prob = u * x (effective transmission probability)
"""

from __future__ import annotations

import torch

from graph_brain.config import STPConfig
from graph_brain.core.graph import EdgeStore, NodeState


class ShortTermPlasticity:
    """Tsodyks-Markram short-term facilitation and depression."""

    def __init__(self, config: STPConfig):
        self.cfg = config

    def update(self, store: EdgeStore, node_state: NodeState, dt: float) -> None:
        """Update facilitation, depression, and release probability per edge.

        Args:
            store: EdgeStore (modified in-place)
            node_state: current node states
            dt: timestep in ms
        """
        if not self.cfg.enabled or store.n_edges == 0:
            return

        U = self.cfg.U_baseline
        tau_f = self.cfg.tau_facilitation
        tau_d = self.cfg.tau_depression

        # Presynaptic activity (continuous rate, not binary spike)
        pre_activity = node_state.output[store.src.long()]

        # Facilitation: u decays to 0, jumps by U*(1-u) on activity
        du = dt * (-store.facilitation / tau_f + U * (1.0 - store.facilitation) * pre_activity)
        store.facilitation += du
        store.facilitation.clamp_(0.0, 1.0)

        # Depression: x recovers to 1, decreases by u*x on activity
        dx = dt * ((1.0 - store.depression) / tau_d - store.facilitation * store.depression * pre_activity)
        store.depression += dx
        store.depression.clamp_(0.0, 1.0)

        # Effective release probability
        store.release_prob = (U + store.facilitation) * store.depression
        store.release_prob.clamp_(0.0, 1.0)
