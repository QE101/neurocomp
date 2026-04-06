"""Tests for the main Simulator integration."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.dynamics.simulator import Simulator


class TestSimulator:
    def test_single_step_no_crash(self, small_graph):
        sim = Simulator(small_graph)
        metrics = sim.step()
        assert isinstance(metrics, dict)

    def test_100_steps_no_nan(self, small_graph):
        """100 steps should produce no NaN or Inf values."""
        sim = Simulator(small_graph)
        sim.run(n_steps=100, show_progress=False)
        ns = small_graph.node_state
        assert not torch.isnan(ns.output).any(), "NaN in output"
        assert not torch.isinf(ns.output).any(), "Inf in output"
        assert not torch.isnan(ns.basal).any(), "NaN in basal"
        assert not torch.isnan(ns.apical).any(), "NaN in apical"

    def test_output_non_negative(self, small_graph):
        sim = Simulator(small_graph)
        sim.run(n_steps=50, show_progress=False)
        assert (small_graph.node_state.output >= 0).all()

    def test_step_count_increments(self, small_graph):
        sim = Simulator(small_graph)
        assert small_graph.step_count == 0
        sim.step()
        assert small_graph.step_count == 1
        sim.run(n_steps=10, show_progress=False)
        assert small_graph.step_count == 11

    def test_inject_input(self, small_graph):
        sim = Simulator(small_graph)
        # Inject strong input into first 10 nodes
        indices = torch.arange(10)
        values = torch.ones(10) * 10.0
        sim.inject_input(indices, values)
        sim.step()
        # Injected nodes should have higher output than others
        injected_mean = small_graph.node_state.output[:10].mean()
        other_mean = small_graph.node_state.output[10:].mean()
        assert injected_mean > other_mean

    def test_deterministic(self, cpu_config):
        """Same seed should produce identical results."""
        g1 = NeuromorphicGraph(cpu_config)
        g1.initialize()
        sim1 = Simulator(g1)
        sim1.run(n_steps=50, show_progress=False)

        g2 = NeuromorphicGraph(cpu_config)
        g2.initialize()
        sim2 = Simulator(g2)
        sim2.run(n_steps=50, show_progress=False)

        assert torch.allclose(g1.node_state.output, g2.node_state.output, atol=1e-5)

    def test_recorder_captures_metrics(self, small_graph):
        sim = Simulator(small_graph)
        # record_interval=1 for small_test config
        sim.run(n_steps=20, show_progress=False)
        metrics = sim.recorder.get_all_metrics()
        assert "output_mean" in metrics
        assert len(metrics["output_mean"]) > 0

    def test_timing_summary(self, small_graph):
        sim = Simulator(small_graph)
        sim.run(n_steps=10, show_progress=False)
        timing = sim.timing_summary()
        assert "message_passing" in timing
        assert "node_model" in timing

    def test_callback(self, small_graph):
        sim = Simulator(small_graph)
        calls = []
        sim.run(n_steps=5, callback=lambda i, m: calls.append(i), show_progress=False)
        assert len(calls) == 5
