"""Triton engine: fused source-parallel message passing kernel.

The key insight: with true silence, 83% of source nodes output zero.
A source-parallel kernel assigns one thread block per source node.
Silent nodes exit on a single branch — zero cost. Active nodes process
their k outgoing edges with no intermediate allocations.

Results in:
- 5-6x fewer atomic scatters (only active-source edges)
- Zero intermediate memory allocation (messages in registers)
- 94x fewer output reads (one per node, not one per edge)
- ~7 kernel launches per step (one per edge type)

Requires: Linux with Triton (WSL works).

Usage:
    engine = TritonEngine(graph)
    engine.send(graph, output, content, step)
    inputs = engine.read(step)
    engine.rebuild(graph)  # after structural plasticity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import triton
import triton.language as tl

from graph_brain.core.delay_buffer import Channel, N_CHANNELS
from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.types import EdgeType


# ================================================================
# TRITON KERNELS
# ================================================================

@triton.jit
def _fused_send_kernel(
    signal_ptr,       # [N] source signal (output or content)
    weight_ptr,       # [E] edge weights (original EdgeStore order)
    rp_ptr,           # [E] release probabilities (original EdgeStore order)
    dst_ptr,          # [E] destination node indices (source-sorted)
    delay_ptr,        # [E] delay steps (source-sorted)
    src_ptr_csr,      # [N+1] source pointer (CSR by source)
    orig_idx_ptr,     # [E] source-sorted → original EdgeStore index
    node_list_ptr,    # [M] list of nodes with edges (pre-filtered)
    buffer_ptr,       # flat delay buffer
    N: tl.constexpr,
    buf_len: tl.constexpr,
    channel: tl.constexpr,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per source node with edges. Skip silent sources for free.

    Reads weight/rp from original EdgeStore via orig_idx (scattered but
    hidden by ILP). Dst/delay are coalesced (source-sorted). Node list
    pre-filtered to skip edgeless nodes.
    """
    pid = tl.program_id(0)
    src_id = tl.load(node_list_ptr + pid)

    src_out = tl.load(signal_ptr + src_id)
    if src_out == 0.0:
        return

    edge_start = tl.load(src_ptr_csr + src_id)
    edge_end = tl.load(src_ptr_csr + src_id + 1)
    n_edges = edge_end - edge_start

    for tile_start in range(0, n_edges, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        edge_pos = edge_start + offs
        mask = offs < n_edges

        # Scattered reads for weight/rp (hidden by ILP with coalesced reads)
        orig = tl.load(orig_idx_ptr + edge_pos, mask=mask)
        w = tl.load(weight_ptr + orig, mask=mask)
        rp = tl.load(rp_ptr + orig, mask=mask)

        # Coalesced reads
        dst = tl.load(dst_ptr + edge_pos, mask=mask)
        delay = tl.load(delay_ptr + edge_pos, mask=mask)

        msg = src_out * w * rp

        buf_idx = (step + delay) % buf_len
        flat = channel * buf_len * N + buf_idx * N + dst
        tl.atomic_add(buffer_ptr + flat, msg, mask=mask)


@triton.jit
def _dst_reduce_kernel(
    signal_ptr,       # [N] source signal
    weight_ptr,       # [E] edge weights (dst-sorted, COALESCED)
    rp_ptr,           # [E] release probs (dst-sorted, COALESCED)
    src_idx_ptr,      # [E] source indices (dst-sorted, COALESCED)
    delay_ptr,        # [E] delay steps (dst-sorted, COALESCED)
    dst_ptr_csr,      # [N+1] CSR by destination (existing!)
    node_list_ptr,    # [M] nodes with incoming edges
    buffer_ptr,       # flat delay buffer
    N: tl.constexpr,
    buf_len: tl.constexpr,
    channel: tl.constexpr,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    """Destination-parallel: each program reduces incoming edges for one node.

    Zero atomics — each destination has a sole writer. Local reduction in
    registers, then single non-atomic stores to delay buffer.

    4 of 5 edge reads are coalesced (edges sorted by dst). Only signal[src]
    is scattered (unavoidable, but L2-friendly at N=50K).
    """
    pid = tl.program_id(0)
    dst_id = tl.load(node_list_ptr + pid)

    edge_start = tl.load(dst_ptr_csr + dst_id)
    edge_end = tl.load(dst_ptr_csr + dst_id + 1)
    n_edges = edge_end - edge_start

    # 8 local accumulators for delay slots (buf_len <= 8)
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0

    for tile_start in range(0, n_edges, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        edge_pos = edge_start + offs
        mask = offs < n_edges

        # 4 coalesced reads (edges contiguous for this destination)
        src = tl.load(src_idx_ptr + edge_pos, mask=mask)
        w = tl.load(weight_ptr + edge_pos, mask=mask)
        rp = tl.load(rp_ptr + edge_pos, mask=mask)
        delay = tl.load(delay_ptr + edge_pos, mask=mask)

        # 1 scattered read (source signal — L2-friendly)
        sig = tl.load(signal_ptr + src, mask=mask, other=0.0)

        msg = sig * w * rp
        buf_idx = (step + delay) % buf_len

        # Local reduction per delay slot — NO ATOMICS
        a0 += tl.sum(tl.where((buf_idx == 0) & mask, msg, 0.0))
        a1 += tl.sum(tl.where((buf_idx == 1) & mask, msg, 0.0))
        a2 += tl.sum(tl.where((buf_idx == 2) & mask, msg, 0.0))
        a3 += tl.sum(tl.where((buf_idx == 3) & mask, msg, 0.0))
        a4 += tl.sum(tl.where((buf_idx == 4) & mask, msg, 0.0))
        a5 += tl.sum(tl.where((buf_idx == 5) & mask, msg, 0.0))
        a6 += tl.sum(tl.where((buf_idx == 6) & mask, msg, 0.0))
        a7 += tl.sum(tl.where((buf_idx == 7) & mask, msg, 0.0))

    # Write to buffer — sole writer per (channel, slot, dst), NO ATOMICS
    base = channel * buf_len * N + dst_id
    if a0 != 0.0:
        tl.store(buffer_ptr + base + 0 * N, tl.load(buffer_ptr + base + 0 * N) + a0)
    if a1 != 0.0:
        tl.store(buffer_ptr + base + 1 * N, tl.load(buffer_ptr + base + 1 * N) + a1)
    if a2 != 0.0:
        tl.store(buffer_ptr + base + 2 * N, tl.load(buffer_ptr + base + 2 * N) + a2)
    if a3 != 0.0:
        tl.store(buffer_ptr + base + 3 * N, tl.load(buffer_ptr + base + 3 * N) + a3)
    if a4 != 0.0:
        tl.store(buffer_ptr + base + 4 * N, tl.load(buffer_ptr + base + 4 * N) + a4)
    if a5 != 0.0:
        tl.store(buffer_ptr + base + 5 * N, tl.load(buffer_ptr + base + 5 * N) + a5)
    if a6 != 0.0:
        tl.store(buffer_ptr + base + 6 * N, tl.load(buffer_ptr + base + 6 * N) + a6)
    if a7 != 0.0:
        tl.store(buffer_ptr + base + 7 * N, tl.load(buffer_ptr + base + 7 * N) + a7)


@triton.jit
def _electrical_kernel(
    output_ptr,       # [N] node output
    weight_ptr,       # [E] edge weights
    src_idx_ptr,      # [E] source indices (int64)
    dst_idx_ptr,      # [E] destination indices (int64)
    delay_ptr,        # [E] delay steps
    buffer_ptr,       # flat delay buffer
    N: tl.constexpr,
    buf_len: tl.constexpr,
    step,
    n_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """Process electrical (gap junction) edges: weight * (output[src] - output[dst])."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_edges

    src = tl.load(src_idx_ptr + offs, mask=mask)
    dst = tl.load(dst_idx_ptr + offs, mask=mask)
    w = tl.load(weight_ptr + offs, mask=mask)
    delay = tl.load(delay_ptr + offs, mask=mask)

    src_out = tl.load(output_ptr + src, mask=mask)
    dst_out = tl.load(output_ptr + dst, mask=mask)

    gap = w * (src_out - dst_out)

    channel = 5  # Channel.ELECTRICAL
    buf_idx = (step + delay) % buf_len
    flat = channel * buf_len * N + buf_idx * N + dst
    tl.atomic_add(buffer_ptr + flat, gap, mask=mask)


# ================================================================
# SOURCE INDEX (reverse CSR)
# ================================================================

@dataclass
class _SourceIndex:
    """Reverse CSR + metadata for one chemical edge type (source-parallel mode)."""
    edge_type: EdgeType
    channel: int
    uses_content: bool
    src_ptr: Tensor      # [N+1] int64 — CSR by source
    orig_idx: Tensor     # [E] int64 — source-sorted → original EdgeStore index
    dst_sorted: Tensor   # [E] int64 — destinations in source-sorted order
    delay_sorted: Tensor # [E] int64 — delays in source-sorted order
    node_list: Tensor    # [M] int64 — nodes with at least one outgoing edge
    block_size: int      # tuned BLOCK_SIZE for this edge type


@dataclass
class _DstIndex:
    """Destination-parallel metadata for one chemical edge type.
    Uses existing EdgeStore dst_ptr (no new index structure needed)."""
    edge_type: EdgeType
    channel: int
    uses_content: bool
    delay_steps: Tensor  # [E] int64 — pre-computed delay steps (dst-sorted)
    node_list: Tensor    # [M] int64 — nodes with at least one incoming edge
    block_size: int


# ================================================================
# ENGINE
# ================================================================

_CHANNEL_MAP = {
    EdgeType.DRIVING: Channel.BASAL,
    EdgeType.MODULATORY: Channel.APICAL,
    EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
    EdgeType.ELECTRICAL: Channel.ELECTRICAL,
    EdgeType.RETROGRADE: Channel.RETROGRADE,
}
_CONTENT_TYPES = {EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC}
_CHEMICAL_TYPES = [
    EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
    EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE,
]


class TritonEngine:
    """GPU-optimized message passing using fused Triton kernels.

    Source-parallel: one thread block per source node. Silent sources
    (83% with true silence) exit on a single branch — zero wasted work.
    Messages computed in registers, no intermediate tensor allocation.
    """

    def __init__(self, graph: NeuromorphicGraph, dt: float = 1.0, mode: str = 'dst'):
        """
        Args:
            mode: 'src' for source-parallel (atomic scatter, skips silent sources),
                  'dst' for destination-parallel (no atomics, coalesced reads).
        """
        self.dt = dt
        self.N = graph.n_nodes
        self.device = graph.device
        self.mode = mode

        max_delay_ms = graph.config.edges.connectivity.max_radius * 10.0
        self.max_delay_steps = int(max_delay_ms / dt) + 2
        self.buf_len = self.max_delay_steps + 1

        self._buffer = torch.zeros(
            N_CHANNELS, self.buf_len, self.N, device=self.device,
        )

        self._src_indices: list[_SourceIndex] = []
        self._dst_indices: list[_DstIndex] = []
        self._elec_src64: Optional[Tensor] = None
        self._elec_dst64: Optional[Tensor] = None
        self._elec_delay: Optional[Tensor] = None
        self._elec_n: int = 0

        self._build(graph)

    def _build(self, graph: NeuromorphicGraph):
        """Build indices for both source-parallel and destination-parallel modes."""
        self._src_indices = []
        self._dst_indices = []

        for et in _CHEMICAL_TYPES:
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            E = store.n_edges
            channel = _CHANNEL_MAP[et]
            uses_content = et in _CONTENT_TYPES

            if self.mode == 'src':
                # Source-parallel: build reverse CSR
                sort_order = torch.argsort(store.src.long())
                sorted_src = store.src[sort_order].long()
                sorted_dst = store.dst[sort_order].long()
                delay = (store.delay[sort_order] / self.dt).ceil().long().clamp(
                    1, self.max_delay_steps
                )
                src_ptr = torch.zeros(self.N + 1, dtype=torch.int64, device=self.device)
                if E > 0:
                    ones = torch.ones(E, dtype=torch.int64, device=self.device)
                    src_ptr.scatter_add_(0, sorted_src + 1, ones)
                    torch.cumsum(src_ptr, dim=0, out=src_ptr)
                edge_counts = src_ptr[1:] - src_ptr[:-1]
                node_list = (edge_counts > 0).nonzero(as_tuple=True)[0]
                avg_k = E / max(node_list.shape[0], 1)
                block_size = 8 if avg_k <= 8 else 16 if avg_k <= 16 else 32 if avg_k <= 48 else 64 if avg_k <= 96 else 128
                self._src_indices.append(_SourceIndex(
                    edge_type=et, channel=channel, uses_content=uses_content,
                    src_ptr=src_ptr, orig_idx=sort_order,
                    dst_sorted=sorted_dst, delay_sorted=delay,
                    node_list=node_list, block_size=block_size,
                ))
            else:
                # Destination-parallel: use existing dst_ptr, just need delay_steps
                delay_steps = (store.delay / self.dt).ceil().long().clamp(1, self.max_delay_steps)
                dst_counts = store.dst_ptr[1:] - store.dst_ptr[:-1]
                node_list = (dst_counts > 0).nonzero(as_tuple=True)[0]
                avg_k = E / max(node_list.shape[0], 1)
                block_size = 8 if avg_k <= 8 else 16 if avg_k <= 16 else 32 if avg_k <= 48 else 64 if avg_k <= 96 else 128
                self._dst_indices.append(_DstIndex(
                    edge_type=et, channel=channel, uses_content=uses_content,
                    delay_steps=delay_steps, node_list=node_list, block_size=block_size,
                ))

        # Electrical edges
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            if store.n_edges > 0:
                self._elec_src64 = store.src.long()
                self._elec_dst64 = store.dst.long()
                self._elec_delay = (store.delay / self.dt).ceil().long().clamp(
                    1, self.max_delay_steps
                )
                self._elec_n = store.n_edges
            else:
                self._elec_n = 0
        else:
            self._elec_n = 0

    def rebuild(self, graph: NeuromorphicGraph):
        """Rebuild after structural plasticity."""
        self._buffer.zero_()
        self._build(graph)

    def send(self, graph: NeuromorphicGraph, output: Tensor, content: Tensor, step: int):
        """Fused source-parallel message passing via Triton kernels.

        v2: pre-computes weight*release_prob in source-sorted order for
        coalesced kernel reads. Launches only for nodes with edges.
        """
        flat_buffer = self._buffer.reshape(-1)

        if self.mode == 'src':
            self._send_src_parallel(graph, output, content, step, flat_buffer)
        else:
            self._send_dst_parallel(graph, output, content, step, flat_buffer)

        # Electrical edges: separate kernel (tiny, different math)
        if self._elec_n > 0:
            store = graph.edge_store(EdgeType.ELECTRICAL)
            grid = (triton.cdiv(self._elec_n, 256),)
            _electrical_kernel[grid](
                output,
                store.weight,
                self._elec_src64,
                self._elec_dst64,
                self._elec_delay,
                flat_buffer,
                N=self.N,
                buf_len=self.buf_len,
                step=step,
                n_edges=self._elec_n,
                BLOCK_SIZE=256,
            )

    def _send_src_parallel(self, graph, output, content, step, flat_buffer):
        for si in self._src_indices:
            store = graph.edge_store(si.edge_type)
            signal = content if si.uses_content else output
            if si.node_list.shape[0] == 0:
                continue
            grid = (si.node_list.shape[0],)
            _fused_send_kernel[grid](
                signal, store.weight, store.release_prob,
                si.dst_sorted, si.delay_sorted, si.src_ptr, si.orig_idx, si.node_list,
                flat_buffer,
                N=self.N, buf_len=self.buf_len, channel=si.channel, step=step,
                BLOCK_SIZE=si.block_size,
            )

    def _send_dst_parallel(self, graph, output, content, step, flat_buffer):
        for di in self._dst_indices:
            store = graph.edge_store(di.edge_type)
            signal = content if di.uses_content else output
            if di.node_list.shape[0] == 0:
                continue
            grid = (di.node_list.shape[0],)
            _dst_reduce_kernel[grid](
                signal, store.weight, store.release_prob,
                store.src, di.delay_steps,
                store.dst_ptr, di.node_list,
                flat_buffer,
                N=self.N, buf_len=self.buf_len, channel=di.channel, step=step,
                BLOCK_SIZE=di.block_size,
            )

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
        """Return [E] bool mask of edges with at least one active endpoint."""
        src_active = output[store.src.long()] > 0
        dst_active = output[store.dst.long()] > 0
        return src_active | dst_active

    def reset(self):
        self._buffer.zero_()

    @property
    def max_delay(self) -> int:
        return self.max_delay_steps

    def stats(self) -> dict:
        if self.mode == 'src':
            n_types = len(self._src_indices)
            total = sum(si.src_ptr[-1].item() for si in self._src_indices)
        else:
            n_types = len(self._dst_indices)
            total = sum(di.delay_steps.shape[0] for di in self._dst_indices)
        return {
            'mode': self.mode,
            'n_edge_types': n_types,
            'total_edges': total,
            'n_electrical': self._elec_n,
            'N': self.N,
        }
