"""Reward system: eligibility traces + three-factor learning for RL.

Eligibility traces maintain a decaying memory of which edges were recently
co-active. When external reward arrives, edges with high eligibility get
strengthened (correct action) or weakened (incorrect action).

Three-factor rule: dw = reward × eligibility × lr
- Factor 1: presynaptic activity (captured in eligibility)
- Factor 2: postsynaptic activity (captured in eligibility)
- Factor 3: reward signal (external, delivered at action outcome)

Reward also temporarily reduces λ_activity (dopamine-like burst),
creating a brief window of increased activity for consolidation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.types import EdgeType


class RewardSystem:
    """Eligibility-trace-based reward learning on a shared graph."""

    def __init__(
        self,
        graph: NeuromorphicGraph,
        reward_lr: float = 0.01,
        eligibility_decay: float = 0.95,
        lambda_modulation: float = 0.3,
        modulation_decay: float = 0.9,
    ):
        self.reward_lr = reward_lr
        self.eligibility_decay = eligibility_decay  # default 0.95, caller can override
        self.lambda_modulation = lambda_modulation
        self.modulation_decay = modulation_decay
        self.device = graph.device

        # Per-edge eligibility traces (separate from graph's EdgeStore)
        self.eligibility: dict[EdgeType, Tensor] = {}
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                n_edges = graph.edge_store(et).n_edges
                self.eligibility[et] = torch.zeros(n_edges, device=self.device)

        # Lambda modulation state (1.0 = no modulation, <1.0 = reduced sparsity)
        self.lambda_mod = 1.0

    def update_eligibility(self, graph: NeuromorphicGraph) -> None:
        """Decay eligibility traces and accumulate from current co-activation.

        Called every step during presentation and decision phases.
        """
        ns = graph.node_state
        for et, trace in self.eligibility.items():
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)

            # Decay
            trace *= self.eligibility_decay

            # Accumulate: pre × post co-activation
            src_out = ns.output[store.src.long()]
            dst_out = ns.output[store.dst.long()]
            trace += src_out * dst_out

    def apply_reward(
        self,
        graph: NeuromorphicGraph,
        reward: float,
    ) -> None:
        """Apply three-factor reward-modulated weight update.

        reward > 0: strengthen eligible edges (correct action)
        reward < 0: weaken eligible edges (incorrect action)

        Called during the reward phase (after action outcome is known).
        """
        for et, trace in self.eligibility.items():
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)

            # Three-factor: reward × eligibility × lr
            dw = self.reward_lr * reward * trace
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

        # Modulate lambda (dopamine-like burst)
        if reward > 0:
            self.lambda_mod = self.lambda_modulation  # reduce sparsity
        else:
            self.lambda_mod = min(1.2, self.lambda_mod + 0.1)  # slight increase

    def effective_lambda(self, base_lambda: float) -> float:
        """Get current effective activity penalty."""
        return base_lambda * self.lambda_mod

    def step_modulation(self) -> None:
        """Decay lambda modulation back toward baseline each step."""
        self.lambda_mod += (1.0 - self.lambda_mod) * (1.0 - self.modulation_decay)

    def reset_eligibility(self) -> None:
        """Clear all eligibility traces (between trials if needed)."""
        for trace in self.eligibility.values():
            trace.zero_()
