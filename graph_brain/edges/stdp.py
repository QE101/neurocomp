"""STDP variants for the neuromorphic graph.

Standard STDP (Bi & Poo 1998):
    Asymmetric exponential window via eligibility traces.
    Pre-before-post → LTP, Post-before-pre → LTD.

Three-factor STDP (Fremaux & Gerstner 2016):
    Same as standard but gated by a third factor — prediction error sign.
    LTP only triggers when prediction error is DECREASING (the edge is helping).
    Prevents STDP from strengthening edges that increase prediction error.
"""

from __future__ import annotations

import torch

from graph_brain.config import STDPConfig
from graph_brain.core.graph import EdgeStore, NodeState


class STDP:
    """Online STDP using eligibility traces."""

    def __init__(self, config: STDPConfig):
        self.cfg = config
        self._pre_decay = None
        self._post_decay = None

    def update(self, store: EdgeStore, node_state: NodeState, dt: float) -> None:
        if not self.cfg.enabled or store.n_edges == 0:
            return

        if self._pre_decay is None or self._pre_decay.device != store.device:
            self._pre_decay = torch.exp(torch.tensor(-dt / self.cfg.tau_plus, device=store.device))
            self._post_decay = torch.exp(torch.tensor(-dt / self.cfg.tau_minus, device=store.device))
        pre_decay = self._pre_decay
        post_decay = self._post_decay
        store.pre_trace *= pre_decay
        store.post_trace *= post_decay

        pre_activity = node_state.output[store.src.long()]
        post_activity = node_state.output[store.dst.long()]

        store.pre_trace += pre_activity
        store.post_trace += post_activity

        dw = (
            self.cfg.a_plus * post_activity * store.pre_trace
            - self.cfg.a_minus * pre_activity * store.post_trace
        )

        store.weight += self.cfg.learning_rate * dw
        store.weight.clamp_(self.cfg.w_min, self.cfg.w_max)


class ThreeFactorSTDP:
    """STDP gated by prediction error — LTP only when error is decreasing.

    The third factor is the sign of prediction error change at the post-synaptic node.
    If error is decreasing (good — the edge is helping reduce mismatch), allow LTP.
    If error is increasing (bad — the edge is making things worse), suppress LTP.

    This aligns STDP with the predictive coding objective.
    """

    def __init__(self, config: STDPConfig):
        self.cfg = config

    def update(self, store: EdgeStore, node_state: NodeState, dt: float) -> None:
        if not self.cfg.enabled or store.n_edges == 0:
            return

        pre_decay = torch.exp(torch.tensor(-dt / self.cfg.tau_plus, device=store.device))
        post_decay = torch.exp(torch.tensor(-dt / self.cfg.tau_minus, device=store.device))
        store.pre_trace *= pre_decay
        store.post_trace *= post_decay

        pre_activity = node_state.output[store.src.long()]
        post_activity = node_state.output[store.dst.long()]

        store.pre_trace += pre_activity
        store.post_trace += post_activity

        # Standard STDP delta
        dw = (
            self.cfg.a_plus * post_activity * store.pre_trace
            - self.cfg.a_minus * pre_activity * store.post_trace
        )

        # Third factor: gate by prediction error at destination node
        # Positive error = evidence > prediction = need more prediction = allow LTP
        # Negative error = prediction > evidence = over-predicting = allow LTD only
        dst_error = node_state.prediction_error[store.dst.long()]

        # Gate: LTP (dw > 0) only when error > 0, LTD (dw < 0) only when error < 0
        # This means: strengthen edges that help, weaken edges that hurt
        gate = torch.where(
            dw > 0,
            torch.clamp(dst_error, min=0.0),       # LTP gated by positive error
            torch.clamp(-dst_error, min=0.0) + 1.0, # LTD always allowed (+ boost when over-predicting)
        )
        # Normalize gate to prevent explosive scaling
        gate = gate / (gate.mean() + 1e-6)

        store.weight += self.cfg.learning_rate * dw * gate
        store.weight.clamp_(self.cfg.w_min, self.cfg.w_max)
