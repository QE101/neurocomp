"""Tests for typed message passing with conduction delays."""

import torch
import pytest

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import DelayBuffer, Channel
from graph_brain.types import EdgeType


class TestDelayBuffer:
    def test_immediate_delivery(self):
        """Messages with delay=0 written via write_immediate arrive same step."""
        buf = DelayBuffer(n_nodes=10, max_delay_steps=5)
        dst = torch.tensor([2, 5], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0])
        buf.write_immediate(Channel.BASAL, dst, vals, current_step=0)

        result = buf.read(current_step=0)
        assert result[2, Channel.BASAL] == 1.0
        assert result[5, Channel.BASAL] == 2.0
        assert result[0, Channel.BASAL] == 0.0  # untouched node

    def test_delayed_delivery(self):
        """Messages with delay>0 arrive at the correct future step."""
        buf = DelayBuffer(n_nodes=5, max_delay_steps=5)
        dst = torch.tensor([1], dtype=torch.int64)
        vals = torch.tensor([3.0])
        delays = torch.tensor([3], dtype=torch.long)  # arrive at step 0+3 = 3

        buf.write(Channel.APICAL, dst, vals, delays, current_step=0)

        # Should NOT arrive at steps 0, 1, 2
        for s in range(3):
            result = buf.read(current_step=s)
            assert result[1, Channel.APICAL] == 0.0, f"Premature arrival at step {s}"

        # Should arrive at step 3
        result = buf.read(current_step=3)
        assert result[1, Channel.APICAL] == 3.0

    def test_multiple_delays_same_target(self):
        """Multiple messages with different delays to the same node accumulate correctly."""
        buf = DelayBuffer(n_nodes=3, max_delay_steps=5)
        dst = torch.tensor([0, 0], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0])
        delays = torch.tensor([2, 2], dtype=torch.long)

        buf.write(Channel.BASAL, dst, vals, delays, current_step=0)

        # Skip step 0, 1
        buf.read(current_step=0)
        buf.read(current_step=1)

        result = buf.read(current_step=2)
        assert result[0, Channel.BASAL] == 3.0  # 1.0 + 2.0

    def test_circular_wrap(self):
        """Buffer wraps around correctly after max_delay steps."""
        buf = DelayBuffer(n_nodes=2, max_delay_steps=3)  # buf_len = 4

        # Write at step 0 with delay 3 → arrives step 3
        dst = torch.tensor([0], dtype=torch.int64)
        vals = torch.tensor([5.0])
        delays = torch.tensor([3], dtype=torch.long)
        buf.write(Channel.BASAL, dst, vals, delays, current_step=0)

        # Read through steps 0-2 (nothing arrives)
        for s in range(3):
            buf.read(current_step=s)

        # Step 3: should arrive
        result = buf.read(current_step=3)
        assert result[0, Channel.BASAL] == 5.0

        # Write again at step 4 with delay 2 → arrives step 6
        buf.write(Channel.BASAL, dst, vals, delays[:1] * 0 + 2, current_step=4)
        buf.read(current_step=4)
        buf.read(current_step=5)
        result = buf.read(current_step=6)
        assert result[0, Channel.BASAL] == 5.0

    def test_read_clears_slot(self):
        """Reading should clear the slot so it doesn't accumulate across cycles."""
        buf = DelayBuffer(n_nodes=2, max_delay_steps=3)
        dst = torch.tensor([0], dtype=torch.int64)
        vals = torch.tensor([1.0])

        buf.write_immediate(Channel.BASAL, dst, vals, current_step=0)
        result1 = buf.read(current_step=0)
        assert result1[0, Channel.BASAL] == 1.0

        # Second read at the same slot (after wrapping) should be zero
        result2 = buf.read(current_step=0)
        assert result2[0, Channel.BASAL] == 0.0


class TestMessagePassingWithDelays:
    def test_messages_delayed(self, small_graph):
        """Messages should NOT arrive on the same step they're sent."""
        small_graph.node_state.output.fill_(1.0)
        passer = TypedMessagePasser(small_graph.config, small_graph.n_nodes, "cpu")

        # Send at step 0
        passer.send_messages(small_graph, current_step=0)
        # Read at step 0 — nothing should have arrived yet (min delay = 1)
        inputs = passer.read_inputs(current_step=0)
        assert inputs.basal.abs().sum() == 0, "No messages should arrive at send step"

    def test_messages_arrive_after_delay(self, small_graph):
        """Messages should arrive after their delay period."""
        if not small_graph.has_edge_type(EdgeType.DRIVING):
            pytest.skip("No driving edges")

        small_graph.node_state.output.fill_(1.0)
        passer = TypedMessagePasser(small_graph.config, small_graph.n_nodes, "cpu")

        # Send at step 0
        passer.send_messages(small_graph, current_step=0)

        # Read steps 0 through max_delay — at some point messages arrive
        total_basal = 0.0
        for step in range(passer.max_delay_steps + 1):
            inputs = passer.read_inputs(current_step=step)
            total_basal += inputs.basal.sum().item()

        assert total_basal > 0, "Messages should arrive within max_delay steps"

    def test_output_shapes(self, small_graph):
        """All compartment inputs should be [N]."""
        N = small_graph.n_nodes
        passer = TypedMessagePasser(small_graph.config, N, "cpu")
        passer.send_messages(small_graph, current_step=0)
        inputs = passer.read_inputs(current_step=0)
        assert inputs.basal.shape == (N,)
        assert inputs.apical.shape == (N,)
        assert inputs.pv_inhibition.shape == (N,)
        assert inputs.sst_inhibition.shape == (N,)
        assert inputs.electrical.shape == (N,)
        assert inputs.retrograde.shape == (N,)

    def test_zero_output_zero_messages(self, small_graph):
        """Zero node output should produce zero messages at all delays."""
        small_graph.node_state.output.fill_(0.0)
        passer = TypedMessagePasser(small_graph.config, small_graph.n_nodes, "cpu")

        passer.send_messages(small_graph, current_step=0)
        for step in range(passer.max_delay_steps + 1):
            inputs = passer.read_inputs(current_step=step)
            assert inputs.basal.abs().sum() == 0
