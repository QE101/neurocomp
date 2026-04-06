"""Tests for the two-compartment node model."""

import torch
import pytest

from graph_brain.config import GraphBrainConfig
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import NodeType


def _make_inputs(N: int, device="cpu", **overrides) -> CompartmentInputs:
    """Create CompartmentInputs with zeros, optionally overriding fields."""
    defaults = {
        "basal": torch.zeros(N, device=device),
        "apical": torch.zeros(N, device=device),
        "pv_inhibition": torch.zeros(N, device=device),
        "sst_inhibition": torch.zeros(N, device=device),
        "electrical": torch.zeros(N, device=device),
        "retrograde": torch.zeros(N, device=device),
    }
    defaults.update(overrides)
    return CompartmentInputs(**defaults)


class TestTwoCompartmentModel:
    def test_zero_input_decays(self, small_graph):
        """With zero input, basal and apical should decay toward zero."""
        ns = small_graph.node_state
        ns.basal.fill_(1.0)
        ns.apical.fill_(1.0)

        model = TwoCompartmentModel(small_graph.config.nodes)
        inputs = _make_inputs(small_graph.n_nodes)
        model.step(ns, inputs, current_time=0.0)

        # After one step with zero input, basal should have decayed
        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        # basal started at 1.0, should decay by dt/tau
        assert ns.basal[exc_mask].mean() < 1.0

    def test_driving_input_increases_basal(self, small_graph):
        """Positive driving input should increase basal."""
        ns = small_graph.node_state
        ns.basal.fill_(0.0)
        model = TwoCompartmentModel(small_graph.config.nodes)
        inputs = _make_inputs(small_graph.n_nodes, basal=torch.ones(small_graph.n_nodes))
        model.step(ns, inputs, current_time=0.0)

        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        assert ns.basal[exc_mask].mean() > 0

    def test_pv_inhibition_reduces_output(self, small_graph):
        """PV inhibition should reduce excitatory output."""
        ns = small_graph.node_state
        N = small_graph.n_nodes

        model = TwoCompartmentModel(small_graph.config.nodes)

        # Step 1: with driving input, no inhibition
        ns.basal.fill_(0.5)
        inputs_no_inhib = _make_inputs(N, basal=torch.ones(N))
        model.step(ns, inputs_no_inhib, current_time=0.0)
        output_no_inhib = ns.output.clone()

        # Reset
        ns.basal.fill_(0.5)
        ns.output.fill_(0.0)

        # Step 2: same input, strong PV inhibition
        inputs_with_inhib = _make_inputs(N, basal=torch.ones(N),
                                          pv_inhibition=torch.ones(N) * 0.9)
        model.step(ns, inputs_with_inhib, current_time=1.0)
        output_with_inhib = ns.output.clone()

        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        assert output_with_inhib[exc_mask].mean() < output_no_inhib[exc_mask].mean()

    def test_output_non_negative(self, small_graph):
        """Output should never be negative (clamped)."""
        ns = small_graph.node_state
        model = TwoCompartmentModel(small_graph.config.nodes)
        inputs = _make_inputs(small_graph.n_nodes)
        for _ in range(10):
            model.step(ns, inputs, current_time=0.0)
        assert (ns.output >= 0).all()

    def test_activity_ema_updates(self, small_graph):
        """Activity EMA should track output over time."""
        ns = small_graph.node_state
        model = TwoCompartmentModel(small_graph.config.nodes)
        N = small_graph.n_nodes

        # Drive some nodes
        inputs = _make_inputs(N, basal=torch.ones(N) * 5.0)
        for _ in range(100):
            model.step(ns, inputs, current_time=0.0)

        # Activity EMA should be positive for driven nodes
        assert ns.activity_ema.mean() > 0

    def test_apical_gating_ungated_default(self, small_graph):
        """With no apical input, gating should be ~1.0 (ungated)."""
        model = TwoCompartmentModel(small_graph.config.nodes)
        g = model._g(torch.tensor(0.0))
        # g(0) should be close to 1.0 (the "ungated" state)
        assert abs(g.item() - 1.0) < 0.5  # within reasonable range
