"""Spatial graph partitioning for parallel execution.

Splits the graph into spatial partitions that can run semi-independently.
Each partition owns a set of nodes and their local edges. Cross-partition
edges communicate through the delay buffer — the conduction delays provide
temporal slack, allowing partitions to be up to min_delay steps out of sync.

This enables:
  - Multiple CUDA streams processing partitions concurrently
  - Future multi-GPU: one partition per GPU
  - Reduced synchronization: only sync at cross-partition message boundaries

Memory layout: each partition holds contiguous node/edge tensors for cache
locality. Cross-partition edges are stored separately with explicit
source_partition → destination_partition routing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.types import EdgeType


@dataclass
class Partition:
    """A spatial partition of the graph."""
    partition_id: int
    node_indices: Tensor          # [K] global indices of nodes in this partition
    node_mask: Tensor             # [N] bool mask over all nodes
    n_nodes: int

    # Local edges: both src and dst are in this partition
    local_edge_masks: dict[EdgeType, Tensor]   # per-type [E] bool mask

    # Outgoing cross-partition edges: src in this partition, dst in another
    outgoing_edge_masks: dict[EdgeType, Tensor]

    # Incoming cross-partition edges: dst in this partition, src in another
    incoming_edge_masks: dict[EdgeType, Tensor]


class SpatialPartitioner:
    """Splits a graph into spatial partitions along the z-axis.

    Uses the node positions to create balanced partitions. Edges are
    classified as local (both endpoints in same partition) or cross-partition.
    """

    def __init__(self, n_partitions: int = 4):
        self.n_partitions = n_partitions

    def partition(self, graph: NeuromorphicGraph) -> list[Partition]:
        """Create spatial partitions by splitting along z-axis.

        Returns list of Partition objects, one per partition.
        """
        ns = graph.node_state
        N = ns.n_nodes
        device = ns.device
        n_parts = self.n_partitions

        # Sort nodes by z position and split into equal-sized partitions
        z_positions = ns.position[:, 2]
        sorted_indices = torch.argsort(z_positions)
        nodes_per_part = N // n_parts

        # Assign each node to a partition
        node_to_partition = torch.zeros(N, dtype=torch.int32, device=device)
        for p in range(n_parts):
            start = p * nodes_per_part
            end = start + nodes_per_part if p < n_parts - 1 else N
            node_to_partition[sorted_indices[start:end]] = p

        # Build partitions
        partitions = []
        for p in range(n_parts):
            node_mask = node_to_partition == p
            node_indices = torch.where(node_mask)[0]

            local_masks = {}
            outgoing_masks = {}
            incoming_masks = {}

            for et in EdgeType:
                if not graph.has_edge_type(et):
                    continue

                store = graph.edge_store(et)
                src_part = node_to_partition[store.src.long()]
                dst_part = node_to_partition[store.dst.long()]

                # Local: both endpoints in this partition
                local_masks[et] = (src_part == p) & (dst_part == p)

                # Outgoing: src here, dst elsewhere
                outgoing_masks[et] = (src_part == p) & (dst_part != p)

                # Incoming: dst here, src elsewhere
                incoming_masks[et] = (src_part != p) & (dst_part == p)

            partitions.append(Partition(
                partition_id=p,
                node_indices=node_indices,
                node_mask=node_mask,
                n_nodes=int(node_mask.sum()),
                local_edge_masks=local_masks,
                outgoing_edge_masks=outgoing_masks,
                incoming_edge_masks=incoming_masks,
            ))

        return partitions

    def summary(self, graph: NeuromorphicGraph, partitions: list[Partition]) -> str:
        """Print partition statistics."""
        lines = [f"Spatial Partitioning: {len(partitions)} partitions"]
        total_local = 0
        total_cross = 0

        for p in partitions:
            n_local = sum(int(m.sum()) for m in p.local_edge_masks.values())
            n_out = sum(int(m.sum()) for m in p.outgoing_edge_masks.values())
            n_in = sum(int(m.sum()) for m in p.incoming_edge_masks.values())
            total_local += n_local
            total_cross += n_out
            lines.append(
                f"  Partition {p.partition_id}: {p.n_nodes} nodes, "
                f"{n_local} local edges, {n_out} outgoing, {n_in} incoming"
            )

        total_edges = total_local + total_cross
        if total_edges > 0:
            cross_pct = total_cross / total_edges * 100
            lines.append(f"  Cross-partition edges: {cross_pct:.1f}% of total")
        return "\n".join(lines)


class PartitionedMessagePasser:
    """Message passing that processes partitions on separate CUDA streams.

    Each partition's local edges are processed on its own stream.
    Cross-partition messages go through the shared delay buffer.
    Streams synchronize only when cross-partition messages need to be read.

    This gives us parallelism proportional to (1 - cross_partition_ratio).
    With distance-dependent connectivity and spatial partitioning, most
    edges are local, so the parallelism is high.
    """

    def __init__(self, graph: NeuromorphicGraph, partitions: list[Partition]):
        self.graph = graph
        self.partitions = partitions
        self.device = graph.device

        # Create CUDA streams — one per partition
        if str(self.device) != "cpu" and torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in partitions]
        else:
            self.streams = [None] * len(partitions)

    def process_local_edges(
        self,
        edge_type: EdgeType,
        output: Tensor,
        accumulator: Tensor,
    ) -> None:
        """Process local edges for all partitions in parallel on separate streams.

        Each partition's local edges are scattered into the accumulator
        on its own CUDA stream. Since local edges have disjoint destination
        nodes (within partition), there are no write conflicts.
        """
        if not self.graph.has_edge_type(edge_type):
            return

        store = self.graph.edge_store(edge_type)

        for part, stream in zip(self.partitions, self.streams):
            if edge_type not in part.local_edge_masks:
                continue

            mask = part.local_edge_masks[edge_type]
            if not mask.any():
                continue

            if stream is not None:
                with torch.cuda.stream(stream):
                    self._scatter_masked(store, output, accumulator, mask)
            else:
                self._scatter_masked(store, output, accumulator, mask)

        # Synchronize all streams
        if self.streams[0] is not None:
            torch.cuda.synchronize()

    def process_cross_edges(
        self,
        edge_type: EdgeType,
        output: Tensor,
        accumulator: Tensor,
    ) -> None:
        """Process cross-partition edges. These go through the delay buffer
        in the normal message passer, so we just handle them on the default stream.
        """
        if not self.graph.has_edge_type(edge_type):
            return

        store = self.graph.edge_store(edge_type)

        for part in self.partitions:
            if edge_type not in part.outgoing_edge_masks:
                continue
            mask = part.outgoing_edge_masks[edge_type]
            if mask.any():
                self._scatter_masked(store, output, accumulator, mask)

    def _scatter_masked(
        self,
        store: EdgeStore,
        output: Tensor,
        accumulator: Tensor,
        mask: Tensor,
    ) -> None:
        """Scatter messages for edges selected by mask."""
        src_output = output[store.src[mask].long()]
        msg = src_output * store.release_prob[mask] * store.weight[mask]
        accumulator.index_add_(0, store.dst[mask].long(), msg)
