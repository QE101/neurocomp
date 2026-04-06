"""Tests for homeostatic synaptic scaling."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import EdgeStore, NodeState, build_dst_ptr
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.types import EdgeType


class TestHomeostaticScaling:
    def _setup(self, activity_level: float):
        """Create a simple setup with uniform activity."""
        N = 10
        E = 30
        cfg = GraphBrainConfig().edges.homeostatic
        scaling = HomeostaticScaling(cfg)

        src = torch.randint(0, N, (E,), dtype=torch.int32)
        dst = torch.randint(0, N, (E,), dtype=torch.int32)
        sort_idx = torch.argsort(dst.to(torch.int64))
        dst_sorted = dst[sort_idx]

        store = EdgeStore(
            edge_type=EdgeType.DRIVING,
            src=src[sort_idx], dst=dst_sorted,
            weight=torch.full((E,), 0.5),
            delay=torch.zeros(E), release_prob=torch.ones(E),
            facilitation=torch.zeros(E), depression=torch.ones(E),
            pre_trace=torch.zeros(E), post_trace=torch.zeros(E),
            dst_ptr=build_dst_ptr(dst_sorted, N),
        )

        ns = NodeState(
            node_type=torch.zeros(N, dtype=torch.int8),
            position=torch.rand(N, 3),
            basal=torch.zeros(N), apical=torch.zeros(N),
            output=torch.zeros(N),
            threshold=torch.zeros(N), gain=torch.ones(N),
            activity_ema=torch.full((N,), activity_level),
            last_spike_time=torch.zeros(N),
            node_role=torch.zeros(N, dtype=torch.int8),
            hierarchy_level=torch.zeros(N, dtype=torch.int8),
            prediction_error=torch.zeros(N),
            error_mean_ema=torch.zeros(N),
            error_var_ema=torch.ones(N),
            precision=torch.ones(N),
        )
        return scaling, store, ns

    def test_high_activity_scales_down(self):
        """Nodes with activity above target should have weights scaled down."""
        scaling, store, ns = self._setup(activity_level=0.5)  # >> target 0.05
        original_mean = store.weight.mean().item()
        scaling.update(store, ns, dt=1.0)
        assert store.weight.mean().item() < original_mean

    def test_low_activity_scales_up(self):
        """Nodes with activity below target should have weights scaled up."""
        scaling, store, ns = self._setup(activity_level=0.001)  # << target 0.05
        original_mean = store.weight.mean().item()
        scaling.update(store, ns, dt=1.0)
        assert store.weight.mean().item() > original_mean

    def test_at_target_no_change(self):
        """Nodes at target activity should have minimal weight change."""
        scaling, store, ns = self._setup(activity_level=0.05)  # == target
        original = store.weight.clone()
        scaling.update(store, ns, dt=1.0)
        diff = (store.weight - original).abs().max().item()
        assert diff < 1e-5, "At target, weights should barely change"
