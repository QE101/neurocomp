"""Episodic memory: hippocampal subgraph with fast Hebbian encoding.

A population of excitatory nodes in the middle of the graph (by z-position)
with dense recurrent modulatory connections. Fast Hebbian learning allows
one-shot pattern storage. Pattern completion via cue injection retrieves
stored associations.

No new edge types — uses existing MODULATORY edges for recurrent connections.
No new node types — uses ordinary excitatory nodes.
The "hippocampal" property is emergent from the wiring and learning rate,
not from a type assignment.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.types import EdgeType, NodeType


class EpisodicMemory:
    """Hippocampal-like episodic memory on the shared graph."""

    def __init__(
        self,
        graph: NeuromorphicGraph,
        z_low: float = 0.4,
        z_high: float = 0.6,
        p_connect: float = 0.3,
        fast_lr: float = 0.1,
    ):
        """
        Args:
            graph: the shared graph
            z_low, z_high: z-position range for hippocampal nodes
            p_connect: connection probability for recurrent edges
            fast_lr: learning rate for one-shot encoding (10x normal)
        """
        self.fast_lr = fast_lr
        self.device = graph.device
        ns = graph.node_state

        # Identify hippocampal nodes: excitatory in middle z-range
        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        exc_idx = torch.where(exc_mask)[0]
        exc_z = ns.position[exc_idx, 2]
        hipp_mask = (exc_z > z_low) & (exc_z <= z_high)
        self.hipp_nodes = exc_idx[hipp_mask]
        self.n_hipp = self.hipp_nodes.shape[0]

        # Create cue regions: split hippocampal nodes into 4 quadrants for 4 patterns
        hipp_pos = ns.position[self.hipp_nodes]
        hipp_x = hipp_pos[:, 0]
        hipp_y = hipp_pos[:, 1]
        med_x = hipp_x.median()
        med_y = hipp_y.median()

        self.cue_regions = {
            0: self.hipp_nodes[(hipp_x < med_x) & (hipp_y < med_y)],
            1: self.hipp_nodes[(hipp_x >= med_x) & (hipp_y < med_y)],
            2: self.hipp_nodes[(hipp_x < med_x) & (hipp_y >= med_y)],
            3: self.hipp_nodes[(hipp_x >= med_x) & (hipp_y >= med_y)],
        }

        # Wire dense recurrent modulatory connections within hippocampal population
        self._wire_recurrent(graph, p_connect)

        # Track which edges are hippocampal (for fast learning)
        self._build_hipp_edge_mask(graph)

    def _wire_recurrent(self, graph: NeuromorphicGraph, p_connect: float) -> None:
        """Add dense recurrent MODULATORY connections within hippocampal nodes."""
        device = self.device
        n = self.n_hipp

        if n < 2:
            return

        # All-pairs with probability p_connect
        src_local = []
        dst_local = []

        for i in range(n):
            for j in range(n):
                if i != j and torch.rand(1).item() < p_connect:
                    src_local.append(i)
                    dst_local.append(j)

        if not src_local:
            return

        src_global = self.hipp_nodes[torch.tensor(src_local)].to(torch.int32).to(device)
        dst_global = self.hipp_nodes[torch.tensor(dst_local)].to(torch.int32).to(device)
        n_new = src_global.shape[0]

        # Start with zero weights (no a priori memory content)
        weights = torch.zeros(n_new, device=device)
        graph.add_edges(EdgeType.MODULATORY, src_global, dst_global, weights=weights)

    def _build_hipp_edge_mask(self, graph: NeuromorphicGraph) -> None:
        """Build a boolean mask identifying hippocampal-internal modulatory edges."""
        self.hipp_edge_mask = None
        if not graph.has_edge_type(EdgeType.MODULATORY):
            return

        store = graph.edge_store(EdgeType.MODULATORY)
        hipp_set = set(self.hipp_nodes.tolist())

        # An edge is hippocampal if BOTH src and dst are hippocampal nodes
        src_in = torch.tensor([int(s) in hipp_set for s in store.src.tolist()],
                              dtype=torch.bool, device=self.device)
        dst_in = torch.tensor([int(d) in hipp_set for d in store.dst.tolist()],
                              dtype=torch.bool, device=self.device)
        self.hipp_edge_mask = src_in & dst_in

    def encode(self, graph: NeuromorphicGraph, n_steps: int = 5) -> None:
        """One-shot encoding: fast Hebbian on hippocampal recurrent edges.

        Called after a reward outcome to store the current activity pattern
        as an episodic memory.
        """
        if self.hipp_edge_mask is None or not graph.has_edge_type(EdgeType.MODULATORY):
            return

        store = graph.edge_store(EdgeType.MODULATORY)
        ns = graph.node_state

        for _ in range(n_steps):
            # Fast Hebbian: pre × post on hippocampal edges only
            src_out = ns.output[store.src.long()]
            dst_out = ns.output[store.dst.long()]
            dw = self.fast_lr * src_out * dst_out * self.hipp_edge_mask.float()
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

    def cue(
        self,
        graph: NeuromorphicGraph,
        pattern_id: int,
        cue_strength: float = 1.0,
    ) -> Tensor:
        """Inject a cue to trigger pattern completion.

        Returns the cue node indices for injection by the caller.
        """
        if pattern_id in self.cue_regions:
            return self.cue_regions[pattern_id]
        return torch.tensor([], dtype=torch.long, device=self.device)

    def retrieval_strength(self, graph: NeuromorphicGraph) -> float:
        """Measure hippocampal reactivation magnitude."""
        ns = graph.node_state
        return float(ns.output[self.hipp_nodes].mean())

    @property
    def summary(self) -> str:
        return (f"EpisodicMemory: {self.n_hipp} hippocampal nodes, "
                f"cue regions: {[len(v) for v in self.cue_regions.values()]}")
