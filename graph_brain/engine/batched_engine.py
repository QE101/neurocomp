"""Batched engine: pre-concatenate all edge types, minimize kernel launches.

The old engine does 6 Python iterations with ~4 kernels each = ~24 kernel launches.
This engine pre-concatenates all edge data into two groups (output-reading and
content-reading), then does ONE gather + multiply + scatter per group = ~8 launches.

No filtering, no fancy indexing. Just fewer kernel launches.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from graph_brain.core.delay_buffer import Channel, N_CHANNELS
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.types import EdgeType


# Edge type -> delay buffer channel
_CHANNEL_MAP = {
    EdgeType.DRIVING: Channel.BASAL,
    EdgeType.MODULATORY: Channel.APICAL,
    EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
    EdgeType.RETROGRADE: Channel.RETROGRADE,
}

_OUTPUT_TYPES = [EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC,
                 EdgeType.DISINHIBITION, EdgeType.RETROGRADE]
_CONTENT_TYPES = [EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC]


class BatchedEngine:
    """Message passing with pre-concatenated edge data.

    All output-reading edges batched into one tensor set.
    All content-reading edges batched into one tensor set.
    Two scatters per step instead of six.
    """

    def __init__(self, graph: NeuromorphicGraph, dt: float = 1.0):
        self.dt = dt
        self.N = graph.n_nodes
        self.device = graph.device

        max_delay_ms = graph.config.edges.connectivity.max_radius * 10.0
        self.max_delay_steps = int(max_delay_ms / dt) + 2
        self.buf_len = self.max_delay_steps + 1

        self._buffer = torch.zeros(
            N_CHANNELS, self.buf_len, self.N, device=self.device,
        )

        # Electrical cache
        self._elec_src64: Optional[Tensor] = None
        self._elec_dst64: Optional[Tensor] = None
        self._elec_delay_steps: Optional[Tensor] = None

        # Pre-concatenated batches
        self._out_src: Optional[Tensor] = None   # [E_out] int64
        self._out_dst: Optional[Tensor] = None   # [E_out] int64
        self._out_delay: Optional[Tensor] = None  # [E_out] int64
        self._out_channel: Optional[Tensor] = None  # [E_out] int64 — channel per edge
        self._out_weight_idx: Optional[list] = None  # (et, start, end) ranges
        self._out_total: int = 0

        self._con_src: Optional[Tensor] = None   # [E_con] int64
        self._con_dst: Optional[Tensor] = None
        self._con_delay: Optional[Tensor] = None
        self._con_channel: Optional[Tensor] = None
        self._con_weight_idx: Optional[list] = None
        self._con_total: int = 0

        # Pre-allocated message buffers
        self._out_msg: Optional[Tensor] = None
        self._con_msg: Optional[Tensor] = None

        self._build(graph)

    def _build(self, graph: NeuromorphicGraph):
        """Pre-concatenate all edge data into two batches."""
        out_parts = {'src': [], 'dst': [], 'delay': [], 'channel': []}
        con_parts = {'src': [], 'dst': [], 'delay': [], 'channel': []}
        self._out_weight_idx = []
        self._con_weight_idx = []
        out_offset = 0
        con_offset = 0

        for et_list, parts, weight_idx, is_content in [
            (_OUTPUT_TYPES, out_parts, self._out_weight_idx, False),
            (_CONTENT_TYPES, con_parts, self._con_weight_idx, True),
        ]:
            offset = 0
            for et in et_list:
                if not graph.has_edge_type(et):
                    continue
                store = graph.edge_store(et)
                if store.n_edges == 0:
                    continue

                E = store.n_edges
                ch = _CHANNEL_MAP[et]

                parts['src'].append(store.src.long())
                parts['dst'].append(store.dst.long())
                delay = (store.delay / self.dt).ceil().long().clamp(1, self.max_delay_steps)
                parts['delay'].append(delay)
                parts['channel'].append(torch.full((E,), ch, dtype=torch.int64, device=self.device))
                weight_idx.append((et, offset, offset + E))
                offset += E

        # Concatenate output batch
        if out_parts['src']:
            self._out_src = torch.cat(out_parts['src'])
            self._out_dst = torch.cat(out_parts['dst'])
            self._out_delay = torch.cat(out_parts['delay'])
            self._out_channel = torch.cat(out_parts['channel'])
            self._out_total = self._out_src.shape[0]
            self._out_msg = torch.empty(self._out_total, device=self.device)
        else:
            self._out_total = 0

        # Concatenate content batch
        if con_parts['src']:
            self._con_src = torch.cat(con_parts['src'])
            self._con_dst = torch.cat(con_parts['dst'])
            self._con_delay = torch.cat(con_parts['delay'])
            self._con_channel = torch.cat(con_parts['channel'])
            self._con_total = self._con_src.shape[0]
            self._con_msg = torch.empty(self._con_total, device=self.device)
        else:
            self._con_total = 0

        # Electrical
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            if store.n_edges > 0:
                self._elec_src64 = store.src.long()
                self._elec_dst64 = store.dst.long()
                self._elec_delay_steps = (store.delay / self.dt).ceil().long().clamp(
                    1, self.max_delay_steps
                )

    def rebuild(self, graph: NeuromorphicGraph):
        self._buffer.zero_()
        self._build(graph)

    def _gather_weights(self, graph: NeuromorphicGraph, weight_idx_list, total, msg_buf):
        """Gather weight * release_prob from all edge stores into pre-allocated buffer."""
        for et, start, end in weight_idx_list:
            store = graph.edge_store(et)
            msg_buf[start:end] = store.weight * store.release_prob

    def send(self, graph: NeuromorphicGraph, output: Tensor, content: Tensor, step: int):
        """Batched message passing: 2 scatter operations instead of 6."""

        # --- Output-reading edges (DRIVING, PERISOMATIC, DISINHIBITION, RETROGRADE) ---
        if self._out_total > 0:
            # Gather weights from all edge stores into one buffer
            self._gather_weights(graph, self._out_weight_idx, self._out_total, self._out_msg)
            # Multiply by source output
            self._out_msg *= output[self._out_src]

            # Single scatter for ALL output-reading edges across ALL channels
            target_steps = step + self._out_delay
            buf_indices = target_steps % self.buf_len
            # Flat index into the FULL buffer: channel * buf_len * N + buf_idx * N + dst
            flat_idx = self._out_channel * self.buf_len * self.N + buf_indices * self.N + self._out_dst
            flat_buf = self._buffer.reshape(-1)
            flat_buf.index_add_(0, flat_idx, self._out_msg)

        # --- Content-reading edges (MODULATORY, DENDRITIC) ---
        if self._con_total > 0:
            self._gather_weights(graph, self._con_weight_idx, self._con_total, self._con_msg)
            self._con_msg *= content[self._con_src]

            target_steps = step + self._con_delay
            buf_indices = target_steps % self.buf_len
            flat_idx = self._con_channel * self.buf_len * self.N + buf_indices * self.N + self._con_dst
            flat_buf = self._buffer.reshape(-1)
            flat_buf.index_add_(0, flat_idx, self._con_msg)

        # --- Electrical (scatter, tiny) ---
        if self._elec_src64 is not None:
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[self._elec_src64] - output[self._elec_dst64])
            target_steps = step + self._elec_delay_steps
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + self._elec_dst64
            flat_buf = self._buffer[Channel.ELECTRICAL].reshape(-1)
            flat_buf.index_add_(0, flat_idx, gap)

    def read(self, step: int) -> CompartmentInputs:
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

    def reset(self):
        self._buffer.zero_()

    @property
    def max_delay(self) -> int:
        return self.max_delay_steps

    def stats(self) -> dict:
        n_elec = 0 if self._elec_src64 is None else self._elec_src64.shape[0]
        return {
            'output_edges': self._out_total,
            'content_edges': self._con_total,
            'n_electrical': n_elec,
            'N': self.N,
            'kernel_launches_per_step': 2 + (1 if n_elec > 0 else 0),
        }
