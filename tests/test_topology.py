"""Tests for spatial indexing and distance-dependent connectivity."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.topology import TopologyBuilder, SpatialIndex
from graph_brain.types import EdgeType


class TestSpatialIndex:
    def test_all_nodes_assigned(self, small_graph):
        builder = TopologyBuilder(small_graph.config, torch.Generator().manual_seed(42))
        idx = builder.build_spatial_index(small_graph.node_state.position)
        # Every node should appear exactly once in cell_nodes
        sorted_nodes = idx.cell_nodes.sort()[0]
        expected = torch.arange(100, dtype=torch.int32)
        assert torch.equal(sorted_nodes, expected)

    def test_cell_ptr_consistent(self, small_graph):
        builder = TopologyBuilder(small_graph.config, torch.Generator().manual_seed(42))
        idx = builder.build_spatial_index(small_graph.node_state.position)
        # Total nodes across all cells should equal N
        total = int(idx.cell_ptr[-1])
        assert total == 100

    def test_cell_ptr_monotonic(self, small_graph):
        builder = TopologyBuilder(small_graph.config, torch.Generator().manual_seed(42))
        idx = builder.build_spatial_index(small_graph.node_state.position)
        assert (idx.cell_ptr[1:] >= idx.cell_ptr[:-1]).all()


class TestConnectivity:
    def test_edges_within_radius(self, small_graph):
        """All edges should connect nodes within max_radius."""
        max_radius = small_graph.config.edges.connectivity.max_radius
        pos = small_graph.node_state.position
        for et in EdgeType:
            if small_graph.has_edge_type(et):
                store = small_graph.edge_store(et)
                if store.n_edges == 0:
                    continue
                dists = torch.norm(
                    pos[store.src.long()] - pos[store.dst.long()], dim=1
                )
                assert (dists <= max_radius * 1.01).all(), (
                    f"{et.name}: edge distance {dists.max():.3f} > max_radius {max_radius}"
                )

    def test_deterministic_with_seed(self, cpu_config):
        """Same seed should produce identical connectivity."""
        g1 = NeuromorphicGraph(cpu_config)
        g1.initialize()
        g2 = NeuromorphicGraph(cpu_config)
        g2.initialize()
        for et in EdgeType:
            n1 = g1.n_edges(et)
            n2 = g2.n_edges(et)
            assert n1 == n2, f"{et.name}: {n1} vs {n2}"

    def test_electrical_bidirectional(self, small_graph):
        """Electrical edges should appear in both directions."""
        if not small_graph.has_edge_type(EdgeType.ELECTRICAL):
            pytest.skip("No electrical edges")
        store = small_graph.edge_store(EdgeType.ELECTRICAL)
        # For each (a, b) edge, (b, a) should also exist
        edges = set(zip(store.src.tolist(), store.dst.tolist()))
        for s, d in list(edges):
            assert (d, s) in edges, f"Missing reverse electrical edge ({d}, {s})"
