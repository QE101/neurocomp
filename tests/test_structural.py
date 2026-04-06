"""Tests for homeostatic structural plasticity."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.dynamics.simulator import Simulator
from graph_brain.types import EdgeType


class TestStructuralPlasticity:
    def _make_graph(self, structural_enabled=True):
        """Create a small graph with structural plasticity configured."""
        config = GraphBrainConfig.from_dict({
            "nodes": {"n_excitatory": 80, "n_pv": 7, "n_sst": 7, "n_vip": 6},
            "edges": {"structural": {
                "enabled": structural_enabled,
                "update_interval": 10,
                "growth_rate": 0.5,
                "prune_threshold": 0.02,
                "edge_cost": 0.0001,
                "max_degree": 500,
            }},
            "simulation": {"device": "cpu", "seed": 42},
            "viz": {"enabled": False},
        })
        graph = NeuromorphicGraph(config)
        graph.initialize()
        return graph, config

    def test_disabled_no_change(self):
        """Disabled structural plasticity should not modify edges."""
        graph, config = self._make_graph(structural_enabled=False)
        sp = StructuralPlasticity(config)
        n_before = graph.n_edges()
        stats = sp.update(graph)
        assert stats["grown"] == 0
        assert stats["pruned"] == 0
        assert graph.n_edges() == n_before

    def test_pruning_removes_weak_edges(self):
        """Edges with weight below threshold should be pruned."""
        graph, config = self._make_graph()
        sp = StructuralPlasticity(config)

        # Set some driving edge weights to zero (below threshold)
        if graph.has_edge_type(EdgeType.DRIVING):
            store = graph.edge_store(EdgeType.DRIVING)
            n_to_kill = min(5, store.n_edges)
            store.weight[:n_to_kill] = 0.0
            n_before = store.n_edges

            stats = sp.update(graph)

            if graph.has_edge_type(EdgeType.DRIVING):
                assert graph.edge_store(EdgeType.DRIVING).n_edges < n_before
            assert stats["pruned"] > 0

    def test_growth_for_starving_nodes(self):
        """Nodes below target activity should grow new connections."""
        graph, config = self._make_graph()
        sp = StructuralPlasticity(config)

        # Set all activity_ema to zero (starving)
        graph.node_state.activity_ema.fill_(0.0)

        n_before = graph.n_edges()
        stats = sp.update(graph)
        n_after = graph.n_edges()

        assert stats["grown"] > 0, "Starving nodes should grow connections"
        assert n_after > n_before

    def test_no_growth_at_target(self):
        """Nodes at target activity should not grow."""
        graph, config = self._make_graph()
        sp = StructuralPlasticity(config)

        # Set activity above target
        graph.node_state.activity_ema.fill_(0.5)  # >> target of 0.05

        stats = sp.update(graph)
        assert stats["grown"] == 0

    def test_no_self_connections_after_growth(self):
        """Growth should never create self-connections."""
        graph, config = self._make_graph()
        sp = StructuralPlasticity(config)
        graph.node_state.activity_ema.fill_(0.0)

        sp.update(graph)

        for et in EdgeType:
            if graph.has_edge_type(et):
                store = graph.edge_store(et)
                assert not (store.src == store.dst).any(), f"Self-connection in {et.name}"

    def test_edge_cost_decays_weights(self):
        """Energy cost should decay individual edge weights."""
        graph, config = self._make_graph()
        # Disable pruning so we can observe pure decay
        config = config.with_overrides(**{"edges.structural.prune_threshold": 0.0})
        sp = StructuralPlasticity(config)

        if graph.has_edge_type(EdgeType.DRIVING):
            store = graph.edge_store(EdgeType.DRIVING)
            # Track a specific edge's weight
            weight_before = store.weight[0].item()
            # Set activity high so no growth happens
            graph.node_state.activity_ema.fill_(0.5)
            sp.update(graph)
            weight_after = graph.edge_store(EdgeType.DRIVING).weight[0].item()
            assert weight_after < weight_before, "Edge cost should decay individual weights"

    def test_integration_connectivity_grows(self):
        """Over many steps with aggressive growth, connectivity should increase."""
        config = GraphBrainConfig.from_dict({
            "nodes": {"n_excitatory": 80, "n_pv": 7, "n_sst": 7, "n_vip": 6,
                       "ip_target_rate": 0.5},  # high target so nodes are always "starving"
            "edges": {"structural": {
                "enabled": True, "update_interval": 10,
                "growth_rate": 1.0, "prune_threshold": 0.0,  # no pruning
                "edge_cost": 0.0, "max_degree": 500,  # no energy cost
            }},
            "simulation": {"device": "cpu", "seed": 42},
            "viz": {"enabled": False},
        })
        graph = NeuromorphicGraph(config)
        graph.initialize()
        # Force activity well below target so nodes are starving from the start
        graph.node_state.activity_ema.fill_(0.0)
        sim = Simulator(graph, config)

        n_initial = graph.n_edges()
        sim.run(n_steps=100, show_progress=False)
        n_final = graph.n_edges()

        assert n_final > n_initial, (
            f"Edges should grow with starving nodes + no pruning, got {n_initial} -> {n_final}"
        )
