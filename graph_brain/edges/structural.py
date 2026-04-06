"""Homeostatic structural plasticity: activity-driven edge creation and pruning.

The system self-organizes its own connectivity density. No hand-tuned target degree.

Growth rule:
    - Nodes with activity_ema < target_rate are "starving" — they grow new connections
    - New edges are created to random nearby nodes, sampled by distance (Gaussian decay)
    - Growth rate is proportional to how far below target the node is

Pruning rule:
    - Edges with weight below prune_threshold are removed (STDP/homeostatic drove them down)
    - This means the system only keeps edges that "earn their keep" through useful signal flow

Energy constraint:
    - Each edge has a metabolic cost (edge_cost)
    - This penalizes total edge count, preventing runaway growth
    - Applied as a slow weight decay on all edges: dw -= edge_cost per update

The equilibrium connectivity emerges from the balance between:
    "I need more input" (growth) vs "connections are expensive" (energy cost + pruning)
"""

from __future__ import annotations

import torch

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph, EdgeStore, build_dst_ptr
from graph_brain.types import EdgeType, NodeType, EDGE_TYPE_CONSTRAINTS


class StructuralPlasticity:
    """Homeostatic structural plasticity: grow where starving, prune where useless."""

    def __init__(self, config: GraphBrainConfig):
        self.config = config
        self.cfg = config.edges.structural
        self.conn_cfg = config.edges.connectivity

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def update(self, graph: NeuromorphicGraph) -> dict[str, int]:
        """Run one structural plasticity update. Returns stats dict.

        This is called every structural.update_interval steps by the simulator.
        It's a batched operation — all growth and pruning happens at once.
        """
        if not self.cfg.enabled:
            return {"grown": 0, "pruned": 0}

        total_grown = 0
        total_pruned = 0

        # Inhibitory edge types protected from pruning (learn slow, appear weak, but are scaffolding)
        PROTECTED_FROM_PRUNING = {EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION}

        # Process each plastic edge type
        for edge_type in EdgeType:
            if edge_type == EdgeType.ELECTRICAL:
                continue  # gap junctions are non-plastic

            if not graph.has_edge_type(edge_type):
                # Create initial edges if none exist
                grown = self._grow_edges(graph, edge_type)
                total_grown += grown
                continue

            # 1. Apply energy cost (slow weight decay on all edges)
            store = graph.edge_store(edge_type)
            if store.n_edges > 0 and self.cfg.edge_cost > 0:
                store.weight -= self.cfg.edge_cost
                store.weight.clamp_(min=0.0)

            # 2. Prune dead edges (weight below threshold)
            # Skip pruning for inhibitory edges — they learn slow and appear weak
            # but provide essential scaffolding (Peters 2014: inhibitory neurons stable)
            if edge_type not in PROTECTED_FROM_PRUNING:
                pruned = self._prune_edges(graph, edge_type)
                total_pruned += pruned

            # 3. Grow new edges for starving nodes
            grown = self._grow_edges(graph, edge_type)
            total_grown += grown

        return {"grown": total_grown, "pruned": total_pruned}

    def _prune_edges(self, graph: NeuromorphicGraph, edge_type: EdgeType) -> int:
        """Remove edges with weight below threshold."""
        if not graph.has_edge_type(edge_type):
            return 0

        store = graph.edge_store(edge_type)
        if store.n_edges == 0:
            return 0

        dead_mask = store.weight < self.cfg.prune_threshold
        n_dead = int(dead_mask.sum())

        if n_dead > 0:
            graph.remove_edges(edge_type, dead_mask)

        return n_dead

    def _grow_edges(self, graph: NeuromorphicGraph, edge_type: EdgeType) -> int:
        """Grow new edges for nodes that are below target activity.

        Nodes with low activity_ema "reach out" and form new connections
        to nearby valid target nodes, sampled by distance.
        """
        ns = graph.node_state
        device = ns.position.device
        N = ns.n_nodes

        # Get valid source/target types for this edge type
        constraints = EDGE_TYPE_CONSTRAINTS[edge_type]
        src_types = constraints["source_types"]
        tgt_types = constraints["target_types"]

        src_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for st in src_types:
            src_mask |= ns.type_mask(st)

        tgt_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for tt in tgt_types:
            tgt_mask |= ns.type_mask(tt)

        # Find starving SOURCE nodes: activity below target
        target_rate = self.config.nodes.ip_target_rate
        activity = ns.activity_ema

        # Deficit: how far below target (clamped to positive)
        deficit = (target_rate - activity).clamp(min=0.0)
        # Only source nodes can grow
        deficit = deficit * src_mask.float()

        # Nodes with zero deficit don't grow
        growing_mask = deficit > 0
        if not growing_mask.any():
            return 0

        growing_indices = torch.where(growing_mask)[0]

        # How many edges each growing node should attempt to create
        # Proportional to deficit, scaled by growth_rate
        n_to_grow = (deficit[growing_indices] / target_rate * self.cfg.growth_rate * 10).ceil().to(torch.int32)
        n_to_grow.clamp_(min=0, max=50)  # cap per-node growth per update

        # Check degree cap
        if graph.has_edge_type(edge_type):
            store = graph.edge_store(edge_type)
            # Count current incoming edges per source node
            # (we use src here because we're growing FROM source nodes)
            current_out_degree = torch.zeros(N, dtype=torch.int32, device=device)
            if store.n_edges > 0:
                ones = torch.ones(store.n_edges, dtype=torch.int32, device=device)
                current_out_degree.scatter_add_(0, store.src.long(), ones)
            headroom = self.cfg.max_degree - current_out_degree[growing_indices]
            n_to_grow = torch.minimum(n_to_grow, headroom.clamp(min=0))

        total_to_grow = int(n_to_grow.sum())
        if total_to_grow == 0:
            return 0

        # Get target candidates
        tgt_indices = torch.where(tgt_mask)[0]
        if tgt_indices.numel() == 0:
            return 0

        positions = ns.position
        max_radius = self.conn_cfg.max_radius

        # For each growing node, sample targets weighted by distance
        all_new_src = []
        all_new_dst = []

        # Get the connectivity config for this edge type
        conn_type_cfg = self._get_conn_type_cfg(edge_type)
        sigma_sq_2 = 2.0 * conn_type_cfg.sigma ** 2

        for i, src_idx in enumerate(growing_indices.tolist()):
            n_grow = int(n_to_grow[i])
            if n_grow <= 0:
                continue

            src_pos = positions[src_idx]

            # Distance to all target candidates
            dists_sq = ((positions[tgt_indices] - src_pos.unsqueeze(0)) ** 2).sum(dim=1)

            # Filter by radius
            in_range = dists_sq < max_radius * max_radius

            # Remove self
            not_self = tgt_indices != src_idx
            valid = in_range & not_self

            if not valid.any():
                continue

            valid_tgt = tgt_indices[valid]
            valid_dist_sq = dists_sq[valid]

            # Connection probability as sampling weights (unnormalized)
            weights = torch.exp(-valid_dist_sq / sigma_sq_2)

            # Remove already-connected targets
            if graph.has_edge_type(edge_type):
                store = graph.edge_store(edge_type)
                existing_dst = store.dst[store.src == src_idx].long()
                if existing_dst.numel() > 0:
                    existing_set = set(existing_dst.tolist())
                    keep = torch.tensor(
                        [int(t) not in existing_set for t in valid_tgt.tolist()],
                        dtype=torch.bool, device=device,
                    )
                    if not keep.any():
                        continue
                    valid_tgt = valid_tgt[keep]
                    weights = weights[keep]

            if valid_tgt.numel() == 0:
                continue

            # Sample targets (weighted by distance-based probability)
            n_sample = min(n_grow, valid_tgt.numel())
            if weights.sum() <= 0:
                continue

            # Multinomial sampling
            sample_idx = torch.multinomial(weights, n_sample, replacement=False)
            chosen_tgt = valid_tgt[sample_idx]

            all_new_src.append(torch.full((n_sample,), src_idx, dtype=torch.int32, device=device))
            all_new_dst.append(chosen_tgt.to(torch.int32))

        if not all_new_src:
            return 0

        new_src = torch.cat(all_new_src)
        new_dst = torch.cat(all_new_dst)
        n_new = new_src.shape[0]

        # Initialize new edge weights: small but above prune threshold
        init_weight = self.cfg.prune_threshold * 5.0  # start at 5x prune threshold
        new_weights = torch.full((n_new,), init_weight, device=device)

        graph.add_edges(edge_type, new_src, new_dst, weights=new_weights)
        return n_new

    def _get_conn_type_cfg(self, edge_type: EdgeType):
        """Get connectivity config for an edge type."""
        conn = self.conn_cfg
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
