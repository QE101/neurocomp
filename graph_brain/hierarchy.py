"""Hierarchy builder for predictive coding with universal error model.

All excitatory nodes use the same dynamics (output = f(|basal - apical|)).
Hierarchy emerges from:
1. Spatial assignment to levels (z-axis quantile split)
2. Inter-level wiring (bottom-up driving errors, top-down modulatory predictions)
3. Time-constant scaling (higher levels integrate over longer timescales)

No error/representation role split needed — the universal error model handles both.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph, build_dst_ptr
from graph_brain.types import EdgeType, HierarchyLevel, NodeType


class HierarchyBuilder:
    """Builds an N-level predictive coding hierarchy on an existing graph."""

    def __init__(self, config: GraphBrainConfig):
        self.config = config
        self.h_cfg = config.hierarchy

    def build(self, graph: NeuromorphicGraph) -> dict:
        """Assign levels, wire inter-level connections, compute tau multipliers.

        Returns stats dict and the tau_multiplier tensor.
        """
        stats = {}
        stats.update(self._assign_levels(graph))
        stats.update(self._wire_inter_level(graph))
        tau_multiplier = self._compute_tau_multipliers(graph)
        stats['tau_min'] = tau_multiplier.min().item()
        stats['tau_max'] = tau_multiplier.max().item()
        return stats, tau_multiplier

    def _assign_levels(self, graph: NeuromorphicGraph) -> dict:
        """Split nodes into n_levels by z-axis quantile."""
        ns = graph.node_state
        device = ns.device
        axis = self.h_cfg.split_axis
        n_levels = self.h_cfg.n_levels

        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        exc_idx = torch.where(exc_mask)[0]
        exc_pos = ns.position[exc_idx, axis]

        # Compute quantile boundaries (custom split or equal)
        if self.h_cfg.level_split is not None:
            cumulative = torch.tensor(self.h_cfg.level_split, device=device).cumsum(0)[:-1]
            boundaries = torch.quantile(exc_pos.float(), cumulative)
        else:
            quantiles = torch.linspace(0, 1, n_levels + 1, device=device)[1:-1]
            boundaries = torch.quantile(exc_pos.float(), quantiles)

        # Assign excitatory nodes to levels
        counts = {}
        for i in range(n_levels):
            level = i + 1  # levels are 1-indexed
            if i == 0:
                mask = exc_pos < boundaries[0]
            elif i == n_levels - 1:
                mask = exc_pos >= boundaries[-1]
            else:
                mask = (exc_pos >= boundaries[i - 1]) & (exc_pos < boundaries[i])

            level_nodes = exc_idx[mask]
            ns.hierarchy_level[level_nodes] = level
            counts[f'level_{level}_exc'] = level_nodes.shape[0]

        # Inhibitory nodes follow their spatial level
        all_pos = ns.position[:, axis]
        for inh_type in (NodeType.PV, NodeType.SST, NodeType.VIP):
            inh_mask = ns.type_mask(inh_type)
            inh_idx = torch.where(inh_mask)[0]
            if inh_idx.numel() == 0:
                continue
            inh_pos = all_pos[inh_idx]
            for i in range(n_levels):
                level = i + 1
                if i == 0:
                    m = inh_pos < boundaries[0]
                elif i == n_levels - 1:
                    m = inh_pos >= boundaries[-1]
                else:
                    m = (inh_pos >= boundaries[i - 1]) & (inh_pos < boundaries[i])
                ns.hierarchy_level[inh_idx[m]] = level

        return counts

    def _wire_inter_level(self, graph: NeuromorphicGraph) -> dict:
        """Wire inter-level connections between adjacent levels.

        For each pair (L, L+1):
        - Bottom-up: L exc → L+1 exc via DRIVING (error signals upward)
        - Top-down: L+1 exc → L exc via MODULATORY (predictions downward)

        Uses KNN-style sampling: each source node connects to k nearest
        targets in the other level. No role split — all excitatory participate.
        """
        ns = graph.node_state
        device = ns.device
        positions = ns.position
        n_levels = self.h_cfg.n_levels
        k = self.h_cfg.inter_level_k
        init_w = self.h_cfg.inter_level_init_weight
        sigma = self.h_cfg.inter_level_sigma

        total_ff = 0
        total_fb = 0

        for level in range(1, n_levels):
            lower_level = level
            upper_level = level + 1

            lower_exc = torch.where(
                ns.type_mask(NodeType.EXCITATORY) & (ns.hierarchy_level == lower_level)
            )[0]
            upper_exc = torch.where(
                ns.type_mask(NodeType.EXCITATORY) & (ns.hierarchy_level == upper_level)
            )[0]

            if lower_exc.numel() == 0 or upper_exc.numel() == 0:
                continue

            # Bottom-up: lower → upper (DRIVING)
            ff_src, ff_dst = self._knn_inter_level(
                positions, lower_exc, upper_exc, k, device
            )
            if ff_src.numel() > 0:
                weights = torch.full((ff_src.shape[0],), init_w, device=device)
                graph.add_edges(EdgeType.DRIVING, ff_src, ff_dst, weights=weights)
                total_ff += ff_src.shape[0]

            # Top-down: upper → lower (MODULATORY)
            fb_src, fb_dst = self._knn_inter_level(
                positions, upper_exc, lower_exc, k, device
            )
            if fb_src.numel() > 0:
                weights = torch.full((fb_src.shape[0],), init_w, device=device)
                graph.add_edges(EdgeType.MODULATORY, fb_src, fb_dst, weights=weights)
                total_fb += fb_src.shape[0]

        return {'ff_edges': total_ff, 'fb_edges': total_fb}

    def _knn_inter_level(
        self, positions: Tensor, src_idx: Tensor, dst_idx: Tensor,
        k: int, device
    ) -> tuple[Tensor, Tensor]:
        """Connect each source to its k nearest targets in the other level.

        Uses chunked cdist for memory efficiency at large N.
        """
        k = min(k, dst_idx.shape[0])
        src_pos = positions[src_idx]  # [n_src, 3]
        dst_pos = positions[dst_idx]  # [n_dst, 3]

        chunk_size = 2000
        all_src = []
        all_dst = []

        for start in range(0, src_idx.shape[0], chunk_size):
            end = min(start + chunk_size, src_idx.shape[0])
            chunk_pos = src_pos[start:end]  # [chunk, 3]

            # Pairwise distances: [chunk, n_dst]
            dists = torch.cdist(chunk_pos, dst_pos)

            # Top-k nearest
            _, topk_local = dists.topk(k, dim=1, largest=False)  # [chunk, k]

            # Expand to edge lists
            n_chunk = end - start
            src_expanded = src_idx[start:end].unsqueeze(1).expand(-1, k).reshape(-1)
            dst_expanded = dst_idx[topk_local.reshape(-1)]

            all_src.append(src_expanded.to(torch.int32))
            all_dst.append(dst_expanded.to(torch.int32))

        if not all_src:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return empty, empty

        return torch.cat(all_src), torch.cat(all_dst)

    def _compute_tau_multipliers(self, graph: NeuromorphicGraph) -> Tensor:
        """Compute per-node time-constant multiplier based on hierarchy level.

        Level 1: multiplier = 1.0 (fastest, sensory)
        Level L: multiplier = time_scale_factor^(L-1) (slower at higher levels)

        Returns [N] tensor.
        """
        ns = graph.node_state
        N = ns.n_nodes
        device = ns.device
        factor = self.h_cfg.time_scale_factor

        tau_mult = torch.ones(N, device=device)
        for level in range(1, self.h_cfg.n_levels + 1):
            mask = ns.hierarchy_level == level
            tau_mult[mask] = factor ** (level - 1)

        # Unassigned nodes (shouldn't exist after build) get 1.0
        return tau_mult

    def summary(self, graph: NeuromorphicGraph) -> str:
        """Human-readable hierarchy summary."""
        ns = graph.node_state
        lines = [f"Predictive Coding Hierarchy ({self.h_cfg.n_levels} levels):"]
        for level in range(1, self.h_cfg.n_levels + 1):
            mask = ns.type_mask(NodeType.EXCITATORY) & (ns.hierarchy_level == level)
            n = int(mask.sum())
            tau = self.h_cfg.time_scale_factor ** (level - 1)
            lines.append(f"  Level {level}: {n} exc nodes, tau_mult={tau:.1f}x")
        lines.append(f"  Total DRIVING edges: {graph.n_edges(EdgeType.DRIVING)}")
        lines.append(f"  Total MODULATORY edges: {graph.n_edges(EdgeType.MODULATORY)}")
        return "\n".join(lines)
