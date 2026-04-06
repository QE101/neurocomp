"""Tests for the core NeuromorphicGraph data structure."""

import torch
import pytest

from graph_brain.core.graph import NeuromorphicGraph, build_dst_ptr
from graph_brain.types import EdgeType, NodeType


class TestBuildDstPtr:
    def test_simple(self):
        dst = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32)
        ptr = build_dst_ptr(dst, n_nodes=4)
        assert ptr.tolist() == [0, 2, 3, 6, 6]

    def test_empty(self):
        dst = torch.zeros(0, dtype=torch.int32)
        ptr = build_dst_ptr(dst, n_nodes=3)
        assert ptr.tolist() == [0, 0, 0, 0]

    def test_single_node(self):
        dst = torch.tensor([0, 0, 0], dtype=torch.int32)
        ptr = build_dst_ptr(dst, n_nodes=2)
        assert ptr.tolist() == [0, 3, 3]


class TestGraphConstruction:
    def test_node_counts(self, small_graph):
        assert small_graph.n_nodes == 100
        ns = small_graph.node_state
        assert int(ns.type_mask(NodeType.EXCITATORY).sum()) == 80
        assert int(ns.type_mask(NodeType.PV).sum()) == 7
        assert int(ns.type_mask(NodeType.SST).sum()) == 7
        assert int(ns.type_mask(NodeType.VIP).sum()) == 6

    def test_positions_in_bounds(self, small_graph):
        pos = small_graph.node_state.position
        assert pos.shape == (100, 3)
        assert (pos >= 0).all()
        assert (pos <= 1.0).all()

    def test_initial_state_zeros(self, small_graph):
        ns = small_graph.node_state
        assert (ns.basal == 0).all()
        assert (ns.apical == 0).all()
        assert (ns.output == 0).all()

    def test_initial_gain_ones(self, small_graph):
        assert (small_graph.node_state.gain == 1.0).all()

    def test_edges_created(self, small_graph):
        total = small_graph.n_edges()
        assert total > 0, "Graph should have edges after initialization"

    def test_no_self_connections(self, small_graph):
        for et in EdgeType:
            if small_graph.has_edge_type(et):
                store = small_graph.edge_store(et)
                assert not (store.src == store.dst).any(), f"Self-connection in {et.name}"

    def test_edge_type_constraints(self, small_graph):
        """Driving edges should only go from EXC to EXC."""
        ns = small_graph.node_state
        if small_graph.has_edge_type(EdgeType.DRIVING):
            store = small_graph.edge_store(EdgeType.DRIVING)
            src_types = ns.node_type[store.src.long()]
            dst_types = ns.node_type[store.dst.long()]
            assert (src_types == NodeType.EXCITATORY).all()
            assert (dst_types == NodeType.EXCITATORY).all()

    def test_dst_ptr_consistent(self, small_graph):
        """dst_ptr should correctly index into sorted dst array."""
        for et in EdgeType:
            if small_graph.has_edge_type(et):
                store = small_graph.edge_store(et)
                N = small_graph.n_nodes
                assert store.dst_ptr.shape == (N + 1,)
                assert int(store.dst_ptr[0]) == 0
                assert int(store.dst_ptr[-1]) == store.n_edges
                # Check monotonically non-decreasing
                assert (store.dst_ptr[1:] >= store.dst_ptr[:-1]).all()


class TestGraphEdgeModification:
    def test_add_edges(self, small_graph):
        n_before = small_graph.n_edges(EdgeType.DRIVING)
        new_src = torch.tensor([0, 1], dtype=torch.int32)
        new_dst = torch.tensor([2, 3], dtype=torch.int32)
        small_graph.add_edges(EdgeType.DRIVING, new_src, new_dst)
        assert small_graph.n_edges(EdgeType.DRIVING) == n_before + 2

    def test_remove_edges(self, small_graph):
        if not small_graph.has_edge_type(EdgeType.DRIVING):
            pytest.skip("No driving edges")
        store = small_graph.edge_store(EdgeType.DRIVING)
        n_before = store.n_edges
        mask = torch.zeros(n_before, dtype=torch.bool)
        mask[0] = True  # remove first edge
        small_graph.remove_edges(EdgeType.DRIVING, mask)
        assert small_graph.n_edges(EdgeType.DRIVING) == n_before - 1


class TestGraphSerialization:
    def test_state_dict_roundtrip(self, small_graph):
        state = small_graph.state_dict()
        restored = NeuromorphicGraph.from_state_dict(state)
        assert restored.n_nodes == small_graph.n_nodes
        assert restored.n_edges() == small_graph.n_edges()
        assert torch.equal(
            restored.node_state.position.cpu(),
            small_graph.node_state.position.cpu(),
        )

    def test_summary(self, small_graph):
        s = small_graph.summary()
        assert "NeuromorphicGraph" in s
        assert "100" in s
