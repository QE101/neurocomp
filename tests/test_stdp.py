"""Tests for STDP learning rule."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import EdgeStore, build_dst_ptr
from graph_brain.core.graph import NodeState
from graph_brain.edges.stdp import STDP
from graph_brain.types import EdgeType, NodeType


def _make_simple_edge_store(n_edges: int, n_nodes: int, device="cpu") -> EdgeStore:
    """Create a simple edge store for testing."""
    src = torch.randint(0, n_nodes, (n_edges,), dtype=torch.int32, device=device)
    dst = torch.randint(0, n_nodes, (n_edges,), dtype=torch.int32, device=device)
    # Ensure no self-connections
    mask = src != dst
    src, dst = src[mask], dst[mask]
    n_edges = src.shape[0]

    sort_idx = torch.argsort(dst.to(torch.int64))
    dst_sorted = dst[sort_idx]
    return EdgeStore(
        edge_type=EdgeType.DRIVING,
        src=src[sort_idx],
        dst=dst_sorted,
        weight=torch.full((n_edges,), 0.5, device=device),
        delay=torch.zeros(n_edges, device=device),
        release_prob=torch.ones(n_edges, device=device),
        facilitation=torch.zeros(n_edges, device=device),
        depression=torch.ones(n_edges, device=device),
        pre_trace=torch.zeros(n_edges, device=device),
        post_trace=torch.zeros(n_edges, device=device),
        dst_ptr=build_dst_ptr(dst_sorted, n_nodes),
    )


def _make_node_state(n_nodes: int, device="cpu") -> NodeState:
    return NodeState(
        node_type=torch.zeros(n_nodes, dtype=torch.int8, device=device),
        position=torch.rand(n_nodes, 3, device=device),
        basal=torch.zeros(n_nodes, device=device),
        apical=torch.zeros(n_nodes, device=device),
        output=torch.zeros(n_nodes, device=device),
        threshold=torch.zeros(n_nodes, device=device),
        gain=torch.ones(n_nodes, device=device),
        activity_ema=torch.zeros(n_nodes, device=device),
        last_spike_time=torch.full((n_nodes,), -1000.0, device=device),
        node_role=torch.zeros(n_nodes, dtype=torch.int8, device=device),
        hierarchy_level=torch.zeros(n_nodes, dtype=torch.int8, device=device),
        prediction_error=torch.zeros(n_nodes, device=device),
        error_mean_ema=torch.zeros(n_nodes, device=device),
        error_var_ema=torch.ones(n_nodes, device=device),
        precision=torch.ones(n_nodes, device=device),
    )


class TestSTDP:
    def test_weight_bounded(self):
        """Weights should stay within [w_min, w_max]."""
        cfg = GraphBrainConfig().edges.stdp
        stdp = STDP(cfg)
        store = _make_simple_edge_store(100, 20)
        ns = _make_node_state(20)
        ns.output = torch.rand(20) * 2.0  # some activity

        for _ in range(100):
            stdp.update(store, ns, dt=1.0)

        assert (store.weight >= cfg.w_min).all()
        assert (store.weight <= cfg.w_max).all()

    def test_pre_post_strengthens(self):
        """Pre-before-post should lead to LTP (weight increase)."""
        cfg = GraphBrainConfig().edges.stdp
        stdp = STDP(cfg)

        # Simple 2-node, 1-edge setup
        store = EdgeStore(
            edge_type=EdgeType.DRIVING,
            src=torch.tensor([0], dtype=torch.int32),
            dst=torch.tensor([1], dtype=torch.int32),
            weight=torch.tensor([0.5]),
            delay=torch.tensor([0.0]),
            release_prob=torch.tensor([1.0]),
            facilitation=torch.tensor([0.0]),
            depression=torch.tensor([1.0]),
            pre_trace=torch.tensor([0.0]),
            post_trace=torch.tensor([0.0]),
            dst_ptr=build_dst_ptr(torch.tensor([1], dtype=torch.int32), 2),
        )
        ns = _make_node_state(2)

        # Step 1: pre fires
        ns.output[0] = 1.0
        ns.output[1] = 0.0
        stdp.update(store, ns, dt=1.0)

        # Step 2: post fires (pre-before-post → LTP)
        ns.output[0] = 0.0
        ns.output[1] = 1.0
        stdp.update(store, ns, dt=1.0)

        assert store.weight[0] > 0.5, "Pre-before-post should increase weight"

    def test_disabled_no_change(self):
        """Disabled STDP should not modify weights."""
        cfg = GraphBrainConfig().edges.stdp.model_copy(update={"enabled": False})
        stdp = STDP(cfg)
        store = _make_simple_edge_store(50, 10)
        ns = _make_node_state(10)
        ns.output.fill_(1.0)
        original = store.weight.clone()
        stdp.update(store, ns, dt=1.0)
        assert torch.equal(store.weight, original)
