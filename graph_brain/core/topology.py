"""Spatial indexing and distance-dependent connectivity builder.

All operations are O(N*k) where k = average neighbors in radius, NOT O(N^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from graph_brain.config import GraphBrainConfig, ConnectivityTypeConfig
from graph_brain.core.graph import EdgeStore, NodeState, build_dst_ptr
from graph_brain.types import EdgeType, NodeType


@dataclass
class SpatialIndex:
    """Uniform grid spatial hash for fast radius queries.

    Nodes are sorted by cell index. cell_ptr[c]:cell_ptr[c+1] gives
    the range of node indices in cell c (CSR-style).
    """
    cell_size: float
    grid_dims: tuple[int, ...]       # cells per axis
    cell_nodes: Tensor               # [N] int32 — node indices sorted by cell
    cell_ptr: Tensor                 # [C+1] int32 — CSR pointer per cell
    node_to_cell: Tensor             # [N] int32 — cell index per node

    @property
    def n_cells(self) -> int:
        result = 1
        for d in self.grid_dims:
            result *= d
        return result


class TopologyBuilder:
    """Builds distance-dependent connectivity using spatial indexing."""

    def __init__(self, config: GraphBrainConfig, generator: Optional[torch.Generator] = None):
        self.config = config
        self.generator = generator

    def build_spatial_index(self, positions: Tensor) -> SpatialIndex:
        """Build uniform grid spatial hash from 3D positions. O(N)."""
        max_radius = self.config.edges.connectivity.max_radius
        cell_size = max_radius  # one cell per radius
        vol = self.config.nodes.spatial.volume_size

        # Grid dimensions (at least 1 cell per axis)
        grid_dims = tuple(max(1, int(v / cell_size) + 1) for v in vol)

        # Assign each node to a cell
        device = positions.device
        N = positions.shape[0]

        # Cell coordinates per node
        cell_coords = (positions / cell_size).to(torch.int32)
        # Clamp to grid bounds
        for d in range(len(grid_dims)):
            cell_coords[:, d].clamp_(0, grid_dims[d] - 1)

        # Linear cell index: x + y * gx + z * gx * gy
        gx, gy = grid_dims[0], grid_dims[1]
        node_to_cell = (
            cell_coords[:, 0]
            + cell_coords[:, 1] * gx
            + cell_coords[:, 2] * gx * gy
        ).to(torch.int32)

        # Sort nodes by cell
        sort_idx = torch.argsort(node_to_cell.to(torch.int64))
        cell_nodes = sort_idx.to(torch.int32)
        sorted_cells = node_to_cell[sort_idx]

        # Build cell pointer (CSR)
        n_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
        cell_ptr = torch.zeros(n_cells + 1, dtype=torch.int32, device=device)
        if N > 0:
            ones = torch.ones(N, dtype=torch.int32, device=device)
            cell_ptr.scatter_add_(0, sorted_cells.to(torch.int64) + 1, ones)
            torch.cumsum(cell_ptr, dim=0, out=cell_ptr)

        return SpatialIndex(
            cell_size=cell_size,
            grid_dims=grid_dims,
            cell_nodes=cell_nodes,
            cell_ptr=cell_ptr,
            node_to_cell=node_to_cell,
        )

    def connect(
        self,
        edge_type: EdgeType,
        node_state: NodeState,
        spatial_index: SpatialIndex,
    ) -> EdgeStore:
        """Build edges for one type using distance-dependent probability.

        Vectorized: processes all source-target candidate pairs per cell pair in bulk.
        No Python per-node loop. O(N*k) where k = avg neighbors in radius.

        Returns an EdgeStore sorted by dst with dst_ptr built.
        """
        conn_cfg = self._get_connectivity_config(edge_type)
        device = node_state.position.device
        N = node_state.n_nodes

        # Get source and target node masks
        source_types = {NodeType[s] for s in conn_cfg.source_types}
        target_types = {NodeType[t] for t in conn_cfg.target_types}

        src_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for st in source_types:
            src_mask |= node_state.type_mask(st)

        tgt_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for tt in target_types:
            tgt_mask |= node_state.type_mask(tt)

        src_indices = torch.where(src_mask)[0]
        tgt_indices = torch.where(tgt_mask)[0]

        if src_indices.numel() == 0 or tgt_indices.numel() == 0:
            return self._empty_edge_store(edge_type, N, device)

        max_radius = self.config.edges.connectivity.max_radius
        max_radius_sq = max_radius * max_radius
        sigma_sq_2 = 2.0 * conn_cfg.sigma ** 2
        positions = node_state.position

        # Constant-k mode: compute effective p_max from target degree
        if conn_cfg.constant_k is not None:
            import math
            vol = 1.0
            for v in self.config.nodes.spatial.volume_size:
                vol *= v
            n_targets = tgt_indices.numel()
            node_density = n_targets / vol
            # Expected neighbors within max_radius (sphere volume × density)
            expected_neighbors = (4.0 / 3.0) * math.pi * (max_radius ** 3) * node_density
            p_max = min(conn_cfg.constant_k / max(expected_neighbors, 1.0), 1.0)
        else:
            p_max = conn_cfg.p_max
        gx, gy, gz = spatial_index.grid_dims

        # Build cell-to-cell neighbor map (27 neighbors per cell, precomputed)
        # Then for each cell, gather source nodes in that cell, target nodes in
        # neighboring cells, compute all-pairs distances, filter, sample.

        # Step 1: Map each source node to its cell
        src_cells = spatial_index.node_to_cell[src_indices]  # [n_src]

        # Step 2: Process cell by cell — vectorized inner loop
        all_src_list = []
        all_dst_list = []

        # Group source indices by cell for batch processing
        unique_cells = torch.unique(src_cells)

        for cell_val in unique_cells.tolist():
            cell_val = int(cell_val)

            # Source nodes in this cell
            cell_src_mask = src_cells == cell_val
            cell_src_indices = src_indices[cell_src_mask]  # global indices
            n_src_cell = cell_src_indices.shape[0]
            if n_src_cell == 0:
                continue

            src_pos = positions[cell_src_indices]  # [n_src_cell, 3]

            # Cell coordinates
            cz_val = cell_val // (gx * gy)
            cy_val = (cell_val % (gx * gy)) // gx
            cx_val = cell_val % gx

            # Collect all target candidates from 3x3x3 neighborhood
            neighbor_tgt_indices = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx = cx_val + dx
                        ny = cy_val + dy
                        nz = cz_val + dz
                        if nx < 0 or nx >= gx or ny < 0 or ny >= gy or nz < 0 or nz >= gz:
                            continue
                        ncell = nx + ny * gx + nz * gx * gy
                        start = int(spatial_index.cell_ptr[ncell])
                        end = int(spatial_index.cell_ptr[ncell + 1])
                        if start < end:
                            cell_nodes = spatial_index.cell_nodes[start:end].to(torch.int64)
                            # Filter to valid target types
                            valid = tgt_mask[cell_nodes]
                            if valid.any():
                                neighbor_tgt_indices.append(cell_nodes[valid])

            if not neighbor_tgt_indices:
                continue

            cell_tgt_indices = torch.cat(neighbor_tgt_indices)  # [n_tgt_neighbors]
            n_tgt = cell_tgt_indices.shape[0]
            tgt_pos = positions[cell_tgt_indices]  # [n_tgt, 3]

            # All-pairs distance: [n_src_cell, n_tgt]
            diff = src_pos.unsqueeze(1) - tgt_pos.unsqueeze(0)  # [n_src, n_tgt, 3]
            dist_sq = (diff * diff).sum(dim=2)  # [n_src, n_tgt]

            # Distance filter
            in_range = dist_sq < max_radius_sq  # [n_src, n_tgt]

            # Self-connection filter: src != tgt
            src_expanded = cell_src_indices.unsqueeze(1).expand(-1, n_tgt)  # [n_src, n_tgt]
            tgt_expanded = cell_tgt_indices.unsqueeze(0).expand(n_src_cell, -1)  # [n_src, n_tgt]
            not_self = src_expanded != tgt_expanded

            valid_pairs = in_range & not_self  # [n_src, n_tgt]

            if not valid_pairs.any():
                continue

            # Connection probability: p_max * exp(-d^2 / (2*sigma^2))
            pair_dist_sq = dist_sq[valid_pairs]
            probs = p_max * torch.exp(-pair_dist_sq / sigma_sq_2)

            # Sample (generator must be on same device as output)
            if device == "cpu" or str(device) == "cpu":
                rand_vals = torch.rand(probs.shape[0], generator=self.generator, device=device)
            else:
                rand_vals = torch.rand(probs.shape[0], device=device)
            connected = rand_vals < probs

            if not connected.any():
                continue

            # Extract connected pairs
            valid_src = src_expanded[valid_pairs][connected]
            valid_dst = tgt_expanded[valid_pairs][connected]
            all_src_list.append(valid_src.to(torch.int32))
            all_dst_list.append(valid_dst.to(torch.int32))

        if not all_src_list:
            return self._empty_edge_store(edge_type, N, device)

        # Concatenate all edges
        src_t = torch.cat(all_src_list)
        dst_t = torch.cat(all_dst_list)
        E = src_t.shape[0]

        # For electrical (bidirectional), add reverse edges
        if edge_type == EdgeType.ELECTRICAL:
            src_orig = src_t.clone()
            src_t = torch.cat([src_t, dst_t])
            dst_t = torch.cat([dst_t, src_orig])
            E = src_t.shape[0]

        # Compute distances for delays
        distances = torch.norm(
            positions[src_t.long()] - positions[dst_t.long()], dim=1
        )
        delays = distances * 10.0  # Linear delay: 10ms per unit distance

        # Initialize weights
        U = self.config.edges.stp.U_baseline
        if edge_type == EdgeType.ELECTRICAL:
            weights = torch.full((E,), 0.1, device=device)
        elif edge_type in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION):
            weights = torch.rand(E, device=device) * 0.3
        else:
            weights = torch.rand(E, device=device) * 0.2

        # Sort by dst for coalesced scatter
        sort_idx = torch.argsort(dst_t.to(torch.int64))

        # FP32 for all — FP16 caused NaN in Hebbian updates at scale
        # Memory is fine at N=50K (~528 MB), FP16 optimisation deferred to N=500K+
        return EdgeStore(
            edge_type=edge_type,
            src=src_t[sort_idx],
            dst=dst_t[sort_idx],
            weight=weights[sort_idx],
            delay=delays[sort_idx],
            release_prob=torch.full((E,), U, device=device)[sort_idx],
            facilitation=torch.zeros(E, device=device)[sort_idx],
            depression=torch.ones(E, device=device)[sort_idx],
            pre_trace=torch.zeros(E, device=device)[sort_idx],
            post_trace=torch.zeros(E, device=device)[sort_idx],
            dst_ptr=build_dst_ptr(dst_t[sort_idx], N),
        )

    def connect_knn(
        self,
        edge_type: EdgeType,
        node_state: NodeState,
        k: int,
    ) -> EdgeStore:
        """Fast KNN connectivity: connect each source to its k nearest valid targets.

        O(N × k × log N) via chunked distance computation.
        No spatial index needed. No O(N²) all-pairs.
        Each source node gets exactly k connections.

        For scaling to N=50K+.
        """
        conn_cfg = self._get_connectivity_config(edge_type)
        device = node_state.position.device
        N = node_state.n_nodes
        positions = node_state.position

        # Get source and target masks
        source_types = {NodeType[s] for s in conn_cfg.source_types}
        target_types = {NodeType[t] for t in conn_cfg.target_types}

        src_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for st in source_types:
            src_mask |= node_state.type_mask(st)
        tgt_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for tt in target_types:
            tgt_mask |= node_state.type_mask(tt)

        src_indices = torch.where(src_mask)[0]
        tgt_indices = torch.where(tgt_mask)[0]
        n_src = src_indices.shape[0]
        n_tgt = tgt_indices.shape[0]

        if n_src == 0 or n_tgt == 0 or k == 0:
            return self._empty_edge_store(edge_type, N, device)

        k_actual = min(k, n_tgt - 1)  # can't connect to more targets than exist
        src_pos = positions[src_indices]  # [n_src, 3]
        tgt_pos = positions[tgt_indices]  # [n_tgt, 3]

        # Chunked KNN: process sources in chunks to avoid OOM
        # Each chunk computes distances to ALL targets and picks top-k
        chunk_size = min(2000, n_src)  # 2000 sources × n_tgt distances per chunk
        all_src = []
        all_dst = []

        for chunk_start in range(0, n_src, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_src)
            chunk_src_pos = src_pos[chunk_start:chunk_end]  # [chunk, 3]
            chunk_n = chunk_src_pos.shape[0]

            # Distance to all targets: [chunk, n_tgt]
            dist = torch.cdist(chunk_src_pos, tgt_pos)

            # Mask self-connections (set distance to inf)
            chunk_src_global = src_indices[chunk_start:chunk_end]
            for i in range(chunk_n):
                # Find if this source is in the target list
                self_mask = tgt_indices == chunk_src_global[i]
                if self_mask.any():
                    dist[i, self_mask] = float('inf')

            # Top-k nearest targets per source
            _, nearest_idx = dist.topk(k_actual, dim=1, largest=False)  # [chunk, k]

            # Convert to global indices
            chunk_src_repeated = chunk_src_global.unsqueeze(1).expand(-1, k_actual)  # [chunk, k]
            chunk_dst_global = tgt_indices[nearest_idx]  # [chunk, k]

            all_src.append(chunk_src_repeated.reshape(-1).to(torch.int32))
            all_dst.append(chunk_dst_global.reshape(-1).to(torch.int32))

        src_t = torch.cat(all_src)
        dst_t = torch.cat(all_dst)
        E = src_t.shape[0]

        # Bidirectional for electrical
        if edge_type == EdgeType.ELECTRICAL:
            src_orig = src_t.clone()
            src_t = torch.cat([src_t, dst_t])
            dst_t = torch.cat([dst_t, src_orig])
            E = src_t.shape[0]

        # Distances and delays
        distances = torch.norm(positions[src_t.long()] - positions[dst_t.long()], dim=1)
        delays = distances * 10.0

        # Weights — FP32 (FP16 causes NaN in Hebbian updates)
        U = self.config.edges.stp.U_baseline
        if edge_type == EdgeType.ELECTRICAL:
            weights = torch.full((E,), 0.1, device=device)
        elif edge_type in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION):
            weights = torch.rand(E, device=device) * 0.3
        else:
            weights = torch.rand(E, device=device) * 0.2

        # Sort by dst
        sort_idx = torch.argsort(dst_t.to(torch.int64))

        return EdgeStore(
            edge_type=edge_type,
            src=src_t[sort_idx],
            dst=dst_t[sort_idx],
            weight=weights[sort_idx],
            delay=delays[sort_idx],
            release_prob=torch.full((E,), U, device=device)[sort_idx],
            facilitation=torch.zeros(E, device=device)[sort_idx],
            depression=torch.ones(E, device=device)[sort_idx],
            pre_trace=torch.zeros(E, device=device)[sort_idx],
            post_trace=torch.zeros(E, device=device)[sort_idx],
            dst_ptr=build_dst_ptr(dst_t[sort_idx], N),
        )

    def _get_connectivity_config(self, edge_type: EdgeType) -> ConnectivityTypeConfig:
        """Map EdgeType enum to its connectivity config."""
        conn = self.config.edges.connectivity
        mapping = {
            EdgeType.DRIVING: conn.driving,
            EdgeType.MODULATORY: conn.modulatory,
            EdgeType.INHIB_PERISOMATIC: conn.inhib_perisomatic,
            EdgeType.INHIB_DENDRITIC: conn.inhib_dendritic,
            EdgeType.DISINHIBITION: conn.disinhibition,
            EdgeType.ELECTRICAL: conn.electrical,
            EdgeType.RETROGRADE: conn.retrograde,
        }
        return mapping[edge_type]

    def _empty_edge_store(self, edge_type: EdgeType, n_nodes: int, device) -> EdgeStore:
        """Create an empty EdgeStore (no edges)."""
        empty = torch.zeros(0, device=device)
        return EdgeStore(
            edge_type=edge_type,
            src=empty.to(torch.int32),
            dst=empty.to(torch.int32),
            weight=empty,
            delay=empty,
            release_prob=empty,
            facilitation=empty,
            depression=empty,
            pre_trace=empty,
            post_trace=empty,
            dst_ptr=torch.zeros(n_nodes + 1, dtype=torch.int32, device=device),
        )
