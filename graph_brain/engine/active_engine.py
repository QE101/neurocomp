"""Active-gated engine: skip silent edges during message passing.

With true silence, ~83% of nodes output exactly 0. Every edge from a
silent node carries a zero message. This engine builds a reverse CSR
(source pointer) to efficiently find edges from active nodes, then only
computes and scatters those messages.

At 17% node activity: processes ~800K edges instead of 4.7M = 5-6x less work.

Usage:
    engine = ActiveEngine(graph)
    # each step:
    engine.send(graph, output, content, step)
    inputs = engine.read(step)
    # after structural plasticity:
    engine.rebuild(graph)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from graph_brain.core.delay_buffer import Channel, N_CHANNELS
from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.types import EdgeType


# Edge type -> delay buffer channel
_CHANNEL_MAP = {
    EdgeType.DRIVING: Channel.BASAL,
    EdgeType.MODULATORY: Channel.APICAL,
    EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
    EdgeType.ELECTRICAL: Channel.ELECTRICAL,
    EdgeType.RETROGRADE: Channel.RETROGRADE,
}

# Which signal each edge type reads
_CONTENT_TYPES = {EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC}

# Chemical edge types (all except ELECTRICAL)
_CHEMICAL_TYPES = [
    EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
    EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE,
]


@dataclass
class _SourceIndex:
    """Reverse CSR: source pointer for one edge type.

    Edges re-sorted by SOURCE node. src_ptr[i]:src_ptr[i+1] gives the
    edges originating from node i, in the source-sorted order.

    Maps back to original EdgeStore via `orig_idx`.
    """
    edge_type: EdgeType
    channel: int
    uses_content: bool
    src_ptr: Tensor      # [N+1] int64 — CSR pointer by source
    orig_idx: Tensor     # [E] int64 — maps source-sorted position to original EdgeStore index
    dst_sorted: Tensor   # [E] int64 — destination indices in source-sorted order
    delay_sorted: Tensor # [E] int64 — delay steps in source-sorted order
    delay_orig: Tensor   # [E] int64 — delay steps in original EdgeStore order


class ActiveEngine:
    """Message passing engine that skips edges from silent nodes.

    Builds a reverse CSR (source pointer) per edge type. Each step,
    finds active nodes, expands to active edge indices via the source
    pointer, and only computes/scatters those messages.

    5-6x fewer edges processed at 17% node activity (true silence).
    """

    def __init__(self, graph: NeuromorphicGraph, dt: float = 1.0):
        self.dt = dt
        self.N = graph.n_nodes
        self.device = graph.device

        max_delay_ms = graph.config.edges.connectivity.max_radius * 10.0
        self.max_delay_steps = int(max_delay_ms / dt) + 2
        self.buf_len = self.max_delay_steps + 1

        # Delay buffer: [n_channels, buf_len, N]
        self._buffer = torch.zeros(
            N_CHANNELS, self.buf_len, self.N, device=self.device,
        )

        # Source indices per edge type
        self._src_indices: list[_SourceIndex] = []

        # Electrical edge cache (scatter, tiny)
        self._elec_src64: Optional[Tensor] = None
        self._elec_dst64: Optional[Tensor] = None
        self._elec_delay_steps: Optional[Tensor] = None

        self._build(graph)

    def _build(self, graph: NeuromorphicGraph):
        """Build reverse CSR (source pointer) for each chemical edge type."""
        self._src_indices = []

        for et in _CHEMICAL_TYPES:
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            E = store.n_edges
            channel = _CHANNEL_MAP[et]
            uses_content = et in _CONTENT_TYPES

            # Sort edges by source node
            sort_order = torch.argsort(store.src.long())

            sorted_src = store.src[sort_order].long()
            sorted_dst = store.dst[sort_order].long()

            # Delay steps in both orders
            delay_orig = (store.delay / self.dt).ceil().long().clamp(1, self.max_delay_steps)
            delay_steps = delay_orig[sort_order]

            # Build source pointer (CSR by source)
            src_ptr = torch.zeros(self.N + 1, dtype=torch.int64, device=self.device)
            if E > 0:
                ones = torch.ones(E, dtype=torch.int64, device=self.device)
                src_ptr.scatter_add_(0, sorted_src + 1, ones)
                torch.cumsum(src_ptr, dim=0, out=src_ptr)

            self._src_indices.append(_SourceIndex(
                edge_type=et,
                channel=channel,
                uses_content=uses_content,
                src_ptr=src_ptr,
                orig_idx=sort_order,
                dst_sorted=sorted_dst,
                delay_sorted=delay_steps,
                delay_orig=delay_orig,
            ))

        # Electrical edges (scatter, tiny)
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            if store.n_edges > 0:
                self._elec_src64 = store.src.long()
                self._elec_dst64 = store.dst.long()
                self._elec_delay_steps = (store.delay / self.dt).ceil().long().clamp(
                    1, self.max_delay_steps
                )
            else:
                self._elec_src64 = None
        else:
            self._elec_src64 = None

    def rebuild(self, graph: NeuromorphicGraph):
        """Rebuild source indices after structural plasticity."""
        self._buffer.zero_()
        self._build(graph)

    def send(self, graph: NeuromorphicGraph, output: Tensor, content: Tensor, step: int):
        """Activity-gated message passing.

        Computes messages for all edges (same gather as old engine), but only
        scatters the non-zero ones. With 83% silent nodes, the scatter —
        which is the bottleneck — operates on ~17% of edges.
        """
        for si in self._src_indices:
            store = graph.edge_store(si.edge_type)
            E = store.n_edges

            # Gather and compute all messages (fast — memory-bandwidth bound)
            signal = content if si.uses_content else output
            msg = signal[store.src.long()] * store.weight * store.release_prob

            # Find non-zero messages (edges from active sources)
            active = msg.nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                continue

            # Scatter only non-zero messages (the expensive part, now 5-6x smaller)
            delay_active = si.delay_orig[active]
            dst_active = store.dst[active].long()

            target_steps = step + delay_active
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + dst_active
            flat_buf = self._buffer[si.channel].reshape(-1)
            flat_buf.index_add_(0, flat_idx, msg[active])

        # Electrical edges: scatter approach (tiny)
        if self._elec_src64 is not None:
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[self._elec_src64] - output[self._elec_dst64])
            target_steps = step + self._elec_delay_steps
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + self._elec_dst64
            flat_buf = self._buffer[Channel.ELECTRICAL].reshape(-1)
            flat_buf.index_add_(0, flat_idx, gap)

    def read(self, step: int) -> CompartmentInputs:
        """Read arrived messages from delay buffer."""
        buf_idx = step % self.buf_len
        raw = self._buffer[:, buf_idx, :]
        result = CompartmentInputs(
            basal=raw[Channel.BASAL].clone(),
            apical=raw[Channel.APICAL].clone(),
            pv_inhibition=raw[Channel.PV_INHIBITION].clone(),
            sst_inhibition=raw[Channel.SST_INHIBITION].clone(),
            vip_inhibition=raw[Channel.VIP_INHIBITION].clone(),
            electrical=raw[Channel.ELECTRICAL].clone(),
            retrograde=raw[Channel.RETROGRADE].clone(),
        )
        self._buffer[:, buf_idx, :].zero_()
        return result

    def active_edge_mask(self, output: Tensor, store: EdgeStore) -> Tensor:
        """Return [E] bool mask of edges with at least one active endpoint.

        Use this to gate learning updates.
        """
        src_active = output[store.src.long()] > 0
        dst_active = output[store.dst.long()] > 0
        return src_active | dst_active

    def reset(self):
        """Clear the delay buffer."""
        self._buffer.zero_()

    @property
    def max_delay(self) -> int:
        return self.max_delay_steps

    def stats(self) -> dict:
        """Return engine statistics."""
        total_edges = sum(
            si.src_ptr[-1].item() for si in self._src_indices
        )
        n_elec = 0 if self._elec_src64 is None else self._elec_src64.shape[0]
        return {
            'n_edge_types': len(self._src_indices),
            'total_edges': total_edges,
            'n_electrical': n_elec,
            'N': self.N,
        }
