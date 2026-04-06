"""Tests for short-term plasticity (Tsodyks-Markram model)."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import EdgeStore, NodeState, build_dst_ptr
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.types import EdgeType


def _simple_setup(n_edges=10, n_nodes=5):
    cfg = GraphBrainConfig().edges.stp
    stp = ShortTermPlasticity(cfg)

    src = torch.zeros(n_edges, dtype=torch.int32)  # all from node 0
    dst = torch.arange(min(n_edges, n_nodes), dtype=torch.int32)
    if n_edges > n_nodes:
        dst = torch.cat([dst, torch.randint(0, n_nodes, (n_edges - n_nodes,), dtype=torch.int32)])
    dst = dst[:n_edges]
    sort_idx = torch.argsort(dst.to(torch.int64))
    dst_sorted = dst[sort_idx]

    store = EdgeStore(
        edge_type=EdgeType.DRIVING,
        src=src[sort_idx], dst=dst_sorted,
        weight=torch.ones(n_edges),
        delay=torch.zeros(n_edges),
        release_prob=torch.full((n_edges,), cfg.U_baseline),
        facilitation=torch.zeros(n_edges),
        depression=torch.ones(n_edges),
        pre_trace=torch.zeros(n_edges),
        post_trace=torch.zeros(n_edges),
        dst_ptr=build_dst_ptr(dst_sorted, n_nodes),
    )

    ns = NodeState(
        node_type=torch.zeros(n_nodes, dtype=torch.int8),
        position=torch.rand(n_nodes, 3),
        basal=torch.zeros(n_nodes), apical=torch.zeros(n_nodes),
        output=torch.zeros(n_nodes),
        threshold=torch.zeros(n_nodes), gain=torch.ones(n_nodes),
        activity_ema=torch.zeros(n_nodes),
        last_spike_time=torch.zeros(n_nodes),
        node_role=torch.zeros(n_nodes, dtype=torch.int8),
        hierarchy_level=torch.zeros(n_nodes, dtype=torch.int8),
        prediction_error=torch.zeros(n_nodes),
        error_mean_ema=torch.zeros(n_nodes),
        error_var_ema=torch.ones(n_nodes),
        precision=torch.ones(n_nodes),
    )
    return stp, store, ns


class TestShortTermPlasticity:
    def test_facilitation_increases_on_activity(self):
        """Repeated presynaptic activity should increase facilitation."""
        stp, store, ns = _simple_setup()
        ns.output[0] = 1.0  # presynaptic node fires
        initial_f = store.facilitation.clone()
        stp.update(store, ns, dt=1.0)
        assert (store.facilitation >= initial_f).all()

    def test_depression_decreases_on_activity(self):
        """Sustained presynaptic activity should depress (reduce) depression variable."""
        stp, store, ns = _simple_setup()
        ns.output[0] = 1.0
        for _ in range(20):
            stp.update(store, ns, dt=1.0)
        assert store.depression.mean() < 1.0, "Depression should decrease with sustained activity"

    def test_recovery_after_silence(self):
        """After activity stops, STP should recover toward baseline."""
        stp, store, ns = _simple_setup()

        # Drive activity
        ns.output[0] = 1.0
        for _ in range(20):
            stp.update(store, ns, dt=1.0)

        depression_after_drive = store.depression.mean().item()

        # Silence
        ns.output[0] = 0.0
        for _ in range(500):
            stp.update(store, ns, dt=1.0)

        assert store.depression.mean().item() > depression_after_drive, "Depression should recover"

    def test_release_prob_bounded(self):
        """Release probability should stay in [0, 1]."""
        stp, store, ns = _simple_setup()
        ns.output[0] = 5.0  # strong activity
        for _ in range(100):
            stp.update(store, ns, dt=1.0)
        assert (store.release_prob >= 0).all()
        assert (store.release_prob <= 1).all()
