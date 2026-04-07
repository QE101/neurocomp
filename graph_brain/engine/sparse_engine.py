"""Sparse engine: GPU-optimized message passing using CSR SpMV.

Replaces the scatter/gather pattern in TypedMessagePasser with sparse
matrix-vector multiply via cuSPARSE. Same semantics, 5-10x faster.

Usage:
    engine = SparseEngine(graph)
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


# Which signal each edge type reads from
_OUTPUT_TYPES = {
    EdgeType.DRIVING,
    EdgeType.INHIB_PERISOMATIC,
    EdgeType.DISINHIBITION,
    EdgeType.RETROGRADE,
}
_CONTENT_TYPES = {
    EdgeType.MODULATORY,
    EdgeType.INHIB_DENDRITIC,
}

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

# Chemical edge types that get SpMV treatment
_CHEMICAL_TYPES = list(_OUTPUT_TYPES | _CONTENT_TYPES)


@dataclass
class _SparseGroup:
    """Pre-built CSR structure for one (edge_type, delay) pair."""
    edge_type: EdgeType
    delay: int
    channel: int
    uses_content: bool
    crow: Tensor          # [N+1] int32 — CSR row pointers
    col: Tensor           # [nnz] int32 — CSR column indices
    edge_indices: Tensor  # [nnz_orig] int64 — indices into original EdgeStore
    sort_order: Tensor    # [nnz_orig] int64 — reorder from edge_indices to CSR order
    nnz: int


class SparseEngine:
    """GPU-optimized message passing using CSR sparse matrix-vector multiply.

    Drop-in replacement for the dual_channel_send + mp.read_inputs pattern.
    Pre-builds CSR matrices grouped by (edge_type, delay_step). Each step,
    updates matrix values from current weight * release_prob, runs cuSPARSE
    SpMV, and writes dense results to the delay buffer.

    Electrical edges (gap junctions) use the old scatter approach since
    they're tiny (~36K edges) and have different math (src - dst).
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
            N_CHANNELS, self.buf_len, self.N,
            device=self.device,
        )

        # Electrical edge cache (kept as scatter)
        self._elec_src64: Optional[Tensor] = None
        self._elec_dst64: Optional[Tensor] = None
        self._elec_delay_steps: Optional[Tensor] = None

        # Sparse groups
        self._groups: list[_SparseGroup] = []

        self._build(graph)

    def _build(self, graph: NeuromorphicGraph):
        """Build CSR matrices for all chemical edge types, grouped by delay."""
        self._groups = []

        for et in _CHEMICAL_TYPES:
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            delay_steps = (store.delay / self.dt).ceil().long().clamp(1, self.max_delay_steps)
            channel = _CHANNEL_MAP[et]
            uses_content = et in _CONTENT_TYPES

            for d in delay_steps.unique().tolist():
                d = int(d)
                mask = delay_steps == d
                edge_idx = torch.where(mask)[0]
                nnz_orig = edge_idx.shape[0]

                src_sub = store.src[edge_idx].long()
                dst_sub = store.dst[edge_idx].long()

                # Sort by (dst, src) for CSR — row = dst, col = src
                sort_key = dst_sub * self.N + src_sub
                sort_order = torch.argsort(sort_key)
                sorted_dst = dst_sub[sort_order]
                sorted_src = src_sub[sort_order]

                # Build CSR crow_indices
                crow = torch.zeros(self.N + 1, dtype=torch.int64, device=self.device)
                if sorted_dst.numel() > 0:
                    ones = torch.ones(sorted_dst.shape[0], dtype=torch.int64, device=self.device)
                    crow.scatter_add_(0, sorted_dst + 1, ones)
                    torch.cumsum(crow, dim=0, out=crow)

                self._groups.append(_SparseGroup(
                    edge_type=et,
                    delay=d,
                    channel=channel,
                    uses_content=uses_content,
                    crow=crow,
                    col=sorted_src.long(),
                    edge_indices=edge_idx,
                    sort_order=sort_order,
                    nnz=nnz_orig,
                ))

        # Electrical edges (scatter approach)
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
        """Rebuild all sparse matrices after structural plasticity."""
        self._buffer.zero_()
        self._build(graph)

    def send(self, graph: NeuromorphicGraph, output: Tensor, content: Tensor, step: int):
        """SpMV message passing. Replaces dual_channel_send().

        Args:
            graph: the neuromorphic graph (for reading edge stores)
            output: [N] node output tensor
            content: [N] content signal (softplus(basal), clamped)
            step: current simulation step
        """
        for grp in self._groups:
            store = graph.edge_store(grp.edge_type)

            # Compute CSR values: weight * release_prob, reordered for CSR
            w = store.weight[grp.edge_indices]
            rp = store.release_prob[grp.edge_indices]
            values = (w * rp)[grp.sort_order]

            # Build CSR sparse tensor (wraps existing tensors, no copy)
            sparse = torch.sparse_csr_tensor(
                grp.crow, grp.col, values, (self.N, self.N)
            )

            # SpMV: cuSPARSE kernel
            signal = content if grp.uses_content else output
            messages = torch.mv(sparse, signal)

            # Dense write to delay buffer
            buf_idx = (step + grp.delay) % self.buf_len
            self._buffer[grp.channel, buf_idx] += messages

        # Electrical edges: scatter approach (tiny, ~36K edges)
        if self._elec_src64 is not None:
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[self._elec_src64] - output[self._elec_dst64])

            # Flat scatter into delay buffer
            target_steps = step + self._elec_delay_steps
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + self._elec_dst64
            flat_buf = self._buffer[Channel.ELECTRICAL].reshape(-1)
            flat_buf.index_add_(0, flat_idx, gap)

    def read(self, step: int) -> CompartmentInputs:
        """Read arrived messages from delay buffer. Replaces mp.read_inputs()."""
        buf_idx = step % self.buf_len
        # Read all channels for this timestep
        raw = self._buffer[:, buf_idx, :]  # [n_channels, N]
        result = CompartmentInputs(
            basal=raw[Channel.BASAL].clone(),
            apical=raw[Channel.APICAL].clone(),
            pv_inhibition=raw[Channel.PV_INHIBITION].clone(),
            sst_inhibition=raw[Channel.SST_INHIBITION].clone(),
            vip_inhibition=raw[Channel.VIP_INHIBITION].clone(),
            electrical=raw[Channel.ELECTRICAL].clone(),
            retrograde=raw[Channel.RETROGRADE].clone(),
        )
        # Clear slot for reuse
        self._buffer[:, buf_idx, :].zero_()
        return result

    def active_edge_mask(self, output: Tensor, store: EdgeStore) -> Tensor:
        """Return [E] bool mask of edges with at least one active endpoint.

        Use this to gate learning updates — skip edges where both src and dst
        are silent (output == 0 due to true silence).
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
        total_nnz = sum(g.nnz for g in self._groups)
        n_groups = len(self._groups)
        n_elec = 0 if self._elec_src64 is None else self._elec_src64.shape[0]
        return {
            'n_groups': n_groups,
            'total_nnz': total_nnz,
            'n_electrical': n_elec,
            'buf_len': self.buf_len,
            'N': self.N,
        }
