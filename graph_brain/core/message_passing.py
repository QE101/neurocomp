"""Typed message passing with conduction delays.

Messages don't arrive instantly. Each edge has a distance-dependent delay.
The message passer computes messages and writes them into a circular delay buffer
at the appropriate future timestep. The simulator reads arrived messages each step.

This is the computational hot path. Each edge type:
  1. Gathers source node outputs
  2. Applies STP release probability and weight
  3. Converts delay (ms) to delay_steps (int)
  4. Writes into the delay buffer at current_step + delay_steps
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from graph_brain.config import GraphBrainConfig
from graph_brain.core.delay_buffer import Channel, DelayBuffer
from graph_brain.core.graph import EdgeStore, NeuromorphicGraph
from graph_brain.types import EdgeType


@dataclass
class CompartmentInputs:
    """Aggregated inputs per compartment, ready for node model consumption."""
    basal: Tensor           # [N] — driving input to basal compartment
    apical: Tensor          # [N] — modulatory input to apical compartment
    pv_inhibition: Tensor   # [N] — perisomatic inhibition (scales output)
    sst_inhibition: Tensor  # [N] — dendritic inhibition (gates apical)
    vip_inhibition: Tensor  # [N] — VIP→SST disinhibition (suppresses SST output)
    electrical: Tensor      # [N] — gap junction current (PV↔PV)
    retrograde: Tensor      # [N] — retrograde suppression signal


# Map edge types to delay buffer channels
EDGE_TO_CHANNEL = {
    EdgeType.DRIVING: Channel.BASAL,
    EdgeType.MODULATORY: Channel.APICAL,
    EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
    EdgeType.ELECTRICAL: Channel.ELECTRICAL,
    EdgeType.RETROGRADE: Channel.RETROGRADE,
}


class TypedMessagePasser:
    """Routes messages by edge type through delay buffer to correct compartments."""

    def __init__(self, config: GraphBrainConfig, n_nodes: int, device: str = "cpu"):
        self.config = config
        self.dt = config.nodes.dt

        # Compute max delay in steps from connectivity config
        # max_radius * delay_factor(10) / dt, rounded up + 1 safety margin
        max_delay_ms = config.edges.connectivity.max_radius * 10.0
        self.max_delay_steps = int(max_delay_ms / self.dt) + 2

        self.delay_buffer = DelayBuffer(
            n_nodes=n_nodes,
            max_delay_steps=self.max_delay_steps,
            device=device,
        )

    def send_messages(self, graph: NeuromorphicGraph, current_step: int) -> None:
        """Compute messages for all edge types and write into delay buffer.

        Called once per step BEFORE read_inputs.
        """
        output = graph.node_state.output

        # Chemical synapses: compute msg, write with delay
        for edge_type in (EdgeType.DRIVING, EdgeType.MODULATORY,
                          EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC,
                          EdgeType.RETROGRADE):
            if not graph.has_edge_type(edge_type):
                continue
            store = graph.edge_store(edge_type)
            msg = self._compute_messages(store, output)
            delay_steps = self._delay_to_steps(store.delay)
            channel = EDGE_TO_CHANNEL[edge_type]
            self.delay_buffer.write(channel, store.dst, msg, delay_steps, current_step)

        # Electrical (gap junctions): near-instantaneous, minimal delay
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            src_output = output[store.src.long()]
            dst_output = output[store.dst.long()]
            gap_current = store.weight * (src_output - dst_output)
            # Gap junctions are fast — deliver with 1-step delay (not zero, for causality)
            delay_steps = torch.ones(store.n_edges, dtype=torch.long, device=store.device)
            self.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap_current,
                                    delay_steps, current_step)

    def read_inputs(self, current_step: int) -> CompartmentInputs:
        """Read all messages that have arrived at current_step.

        Called once per step AFTER send_messages.
        Returns CompartmentInputs with one tensor per compartment.
        """
        arrived = self.delay_buffer.read(current_step)  # [N, n_channels]

        return CompartmentInputs(
            basal=arrived[:, Channel.BASAL],
            apical=arrived[:, Channel.APICAL],
            pv_inhibition=arrived[:, Channel.PV_INHIBITION],
            sst_inhibition=arrived[:, Channel.SST_INHIBITION],
            vip_inhibition=arrived[:, Channel.VIP_INHIBITION],
            electrical=arrived[:, Channel.ELECTRICAL],
            retrograde=arrived[:, Channel.RETROGRADE],
        )

    def _compute_messages(self, store: EdgeStore, output: Tensor) -> Tensor:
        """Core message computation for chemical synapses.

        msg = output[src] * release_prob * weight
        """
        src_output = output[store.src.long()]
        return src_output * store.release_prob * store.weight

    def _delay_to_steps(self, delay_ms: Tensor) -> Tensor:
        """Convert delay in ms to delay in timesteps (minimum 1)."""
        steps = (delay_ms / self.dt).ceil().long()
        return steps.clamp(min=1, max=self.max_delay_steps)

    def reset(self) -> None:
        """Clear the delay buffer."""
        self.delay_buffer.reset()

    def to(self, device: str) -> TypedMessagePasser:
        """Move delay buffer to device."""
        self.delay_buffer.to(device)
        return self
