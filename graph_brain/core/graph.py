"""NeuromorphicGraph: the central data structure holding all node and edge state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from graph_brain.config import GraphBrainConfig, resolve_device
from graph_brain.types import (
    DEFAULT_NODE_RATIOS,
    EdgeType,
    HierarchyLevel,
    NodeRole,
    NodeType,
)
from graph_brain.utils.seeding import seed_everything


@dataclass
class NodeState:
    """All node state as contiguous tensors. Index i = same node across all tensors."""

    # Identity (static after init)
    node_type: Tensor       # [N] int8
    position: Tensor        # [N, 3] float32

    # Two-compartment state (dynamic, updated every step)
    basal: Tensor           # [N] float32
    apical: Tensor          # [N] float32
    output: Tensor          # [N] float32

    # Intrinsic parameters (slow-plastic)
    threshold: Tensor       # [N] float32
    gain: Tensor            # [N] float32

    # Activity tracking (for homeostasis + STDP)
    activity_ema: Tensor    # [N] float32
    last_spike_time: Tensor # [N] float32

    # Hierarchy (assigned by HierarchyBuilder, default = unassigned)
    node_role: Tensor        # [N] int8 — NodeRole: NONE, ERROR, REPRESENTATION
    hierarchy_level: Tensor  # [N] int8 — HierarchyLevel: UNASSIGNED, LEVEL_1, LEVEL_2, ...

    # Adaptive precision (for predictive coding — estimated locally per error node)
    prediction_error: Tensor  # [N] float32 — current prediction error (basal - apical)
    error_mean_ema: Tensor    # [N] float32 — EMA of prediction error
    error_var_ema: Tensor     # [N] float32 — EMA of squared error deviation
    precision: Tensor         # [N] float32 — 1/variance, clamped

    @property
    def n_nodes(self) -> int:
        return self.node_type.shape[0]

    @property
    def device(self) -> torch.device:
        return self.node_type.device

    def type_mask(self, node_type: NodeType) -> Tensor:
        """Boolean mask for nodes of a given type. [N] bool."""
        return self.node_type == node_type

    def type_indices(self, node_type: NodeType) -> Tensor:
        """Indices of nodes of a given type. [K] int64."""
        return torch.where(self.node_type == node_type)[0]

    def role_mask(self, role: NodeRole) -> Tensor:
        """Boolean mask for nodes with a given role. [N] bool."""
        return self.node_role == role

    def level_mask(self, level: HierarchyLevel) -> Tensor:
        """Boolean mask for nodes at a given hierarchy level. [N] bool."""
        return self.hierarchy_level == level

    def role_level_mask(self, role: NodeRole, level: HierarchyLevel) -> Tensor:
        """Boolean mask for nodes with given role AND level. [N] bool."""
        return (self.node_role == role) & (self.hierarchy_level == level)

    def to(self, device: str | torch.device) -> NodeState:
        """Move all tensors to device."""
        return NodeState(**{
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in self.__dict__.items()
        })


@dataclass
class EdgeStore:
    """Sparse edge storage for one edge type. Sorted by dst for coalesced scatter.

    The dst_ptr tensor provides CSR-style indexing: edges targeting node i
    are at indices dst_ptr[i]:dst_ptr[i+1] in the sorted arrays.
    """

    edge_type: EdgeType

    # Topology (sorted by dst)
    src: Tensor              # [E] int32 — source node indices
    dst: Tensor              # [E] int32 — destination node indices

    # Edge properties
    weight: Tensor           # [E] float32
    delay: Tensor            # [E] float32 — conduction delay (ms)

    # Short-term plasticity state
    release_prob: Tensor     # [E] float32
    facilitation: Tensor     # [E] float32 — u variable
    depression: Tensor       # [E] float32 — x variable

    # STDP eligibility traces
    pre_trace: Tensor        # [E] float32
    post_trace: Tensor       # [E] float32

    # CSR-style destination pointer for per-node access
    dst_ptr: Tensor          # [N+1] int32

    @property
    def n_edges(self) -> int:
        return self.src.shape[0]

    @property
    def device(self) -> torch.device:
        return self.src.device

    def to(self, device: str | torch.device) -> EdgeStore:
        """Move all tensors to device."""
        return EdgeStore(**{
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in self.__dict__.items()
        })


def build_dst_ptr(dst: Tensor, n_nodes: int) -> Tensor:
    """Build CSR-style row pointer from sorted destination indices.

    Args:
        dst: [E] int32, sorted destination node indices
        n_nodes: total number of nodes

    Returns:
        [N+1] int32 pointer tensor where ptr[i]:ptr[i+1] spans edges targeting node i
    """
    dst_ptr = torch.zeros(n_nodes + 1, dtype=torch.int32, device=dst.device)
    if dst.numel() > 0:
        # Count edges per destination node
        ones = torch.ones(dst.shape[0], dtype=torch.int32, device=dst.device)
        dst_ptr.scatter_add_(0, dst.to(torch.int64) + 1, ones)
        # Cumulative sum gives CSR pointers
        torch.cumsum(dst_ptr, dim=0, out=dst_ptr)
    return dst_ptr


class NeuromorphicGraph:
    """Central data structure owning all node and edge state.

    Construction:
        config = GraphBrainConfig.from_yaml("configs/default.yaml")
        graph = NeuromorphicGraph(config)
        graph.initialize()  # builds nodes + connectivity
    """

    def __init__(self, config: GraphBrainConfig):
        self.config = config
        self._device = resolve_device(config.simulation.device)
        self._generator = seed_everything(config.simulation.seed)
        self._node_state: Optional[NodeState] = None
        self._edge_stores: dict[EdgeType, EdgeStore] = {}
        self._step_count = 0

    @property
    def node_state(self) -> NodeState:
        assert self._node_state is not None, "Graph not initialized. Call initialize() first."
        return self._node_state

    @property
    def n_nodes(self) -> int:
        return self.config.nodes.n_total

    @property
    def device(self) -> str:
        return self._device

    @property
    def step_count(self) -> int:
        return self._step_count

    def edge_store(self, edge_type: EdgeType) -> EdgeStore:
        """Get the EdgeStore for a given edge type."""
        return self._edge_stores[edge_type]

    def has_edge_type(self, edge_type: EdgeType) -> bool:
        return edge_type in self._edge_stores

    def n_edges(self, edge_type: Optional[EdgeType] = None) -> int:
        """Total edge count, optionally filtered by type."""
        if edge_type is not None:
            return self._edge_stores[edge_type].n_edges if edge_type in self._edge_stores else 0
        return sum(s.n_edges for s in self._edge_stores.values())

    def initialize(self) -> None:
        """Build nodes and connectivity from config."""
        self._initialize_nodes()
        self._initialize_connectivity()

    def _initialize_nodes(self) -> None:
        """Create N nodes with random positions and assigned types."""
        cfg = self.config.nodes
        N = cfg.n_total
        device = self._device

        # Assign node types: first n_excitatory are EXC, then PV, SST, VIP
        type_counts = [cfg.n_excitatory, cfg.n_pv, cfg.n_sst, cfg.n_vip]
        node_type = torch.cat([
            torch.full((count,), ntype, dtype=torch.int8)
            for ntype, count in zip(NodeType, type_counts)
        ]).to(device)

        # Random 3D positions in [0, volume_size]
        vol = torch.tensor(cfg.spatial.volume_size, dtype=torch.float32)
        position = torch.rand(N, cfg.spatial.dimensions, generator=self._generator) * vol
        position = position.to(device)

        self._node_state = NodeState(
            node_type=node_type,
            position=position,
            basal=torch.zeros(N, device=device),
            apical=torch.zeros(N, device=device),
            output=torch.zeros(N, device=device),
            threshold=torch.zeros(N, device=device),
            gain=torch.ones(N, device=device),
            activity_ema=torch.full((N,), cfg.ip_target_rate, device=device),
            last_spike_time=torch.full((N,), -1000.0, device=device),
            node_role=torch.zeros(N, dtype=torch.int8, device=device),       # NONE
            hierarchy_level=torch.zeros(N, dtype=torch.int8, device=device), # UNASSIGNED
            prediction_error=torch.zeros(N, device=device),
            error_mean_ema=torch.zeros(N, device=device),
            error_var_ema=torch.ones(N, device=device),   # start with high variance = low precision
            precision=torch.ones(N, device=device),        # initial precision = 1.0
        )

    def _initialize_connectivity(self) -> None:
        """Build edges for all 6 types. Uses KNN when constant_k is set, otherwise distance-based."""
        from graph_brain.core.topology import TopologyBuilder

        builder = TopologyBuilder(self.config, self._generator)

        for edge_type in EdgeType:
            conn_cfg = builder._get_connectivity_config(edge_type)

            # Use fast KNN builder when constant_k is specified
            if conn_cfg.constant_k is not None:
                store = builder.connect_knn(
                    edge_type=edge_type,
                    node_state=self._node_state,
                    k=conn_cfg.constant_k,
                )
            else:
                # Original distance-probability builder
                if not hasattr(builder, '_spatial_index'):
                    builder._spatial_index = builder.build_spatial_index(self._node_state.position)
                store = builder.connect(
                    edge_type=edge_type,
                    node_state=self._node_state,
                    spatial_index=builder._spatial_index,
                )
            if store.n_edges > 0:
                self._edge_stores[edge_type] = store

    def add_edges(
        self,
        edge_type: EdgeType,
        src: Tensor,
        dst: Tensor,
        weights: Optional[Tensor] = None,
    ) -> None:
        """Add edges to an existing EdgeStore. Re-sorts and rebuilds dst_ptr."""
        store = self._edge_stores.get(edge_type)
        device = self._device
        n_new = src.shape[0]

        if weights is None:
            weights = torch.rand(n_new, device=device) * 0.1

        # Compute delays from distances
        positions = self._node_state.position
        distances = torch.norm(positions[src.long()] - positions[dst.long()], dim=1)
        delays = distances * 10.0  # simple linear delay model

        # Create new edge tensors
        new_weight = weights.to(device)
        new_delay = delays.to(device)
        new_src = src.to(torch.int32).to(device)
        new_dst = dst.to(torch.int32).to(device)
        U = self.config.edges.stp.U_baseline
        zeros_n = torch.zeros(n_new, device=device)

        if store is not None:
            # Append to existing
            new_src = torch.cat([store.src, new_src])
            new_dst = torch.cat([store.dst, new_dst])
            new_weight = torch.cat([store.weight, new_weight])
            new_delay = torch.cat([store.delay, new_delay])
            new_release = torch.cat([store.release_prob, torch.full((n_new,), U, device=device)])
            new_facil = torch.cat([store.facilitation, zeros_n])
            new_depr = torch.cat([store.depression, torch.ones(n_new, device=device)])
            new_pre = torch.cat([store.pre_trace, zeros_n])
            new_post = torch.cat([store.post_trace, zeros_n])
        else:
            new_release = torch.full((n_new,), U, device=device)
            new_facil = zeros_n
            new_depr = torch.ones(n_new, device=device)
            new_pre = zeros_n
            new_post = zeros_n.clone()

        # Sort by dst for coalesced scatter
        sort_idx = torch.argsort(new_dst.to(torch.int64))
        self._edge_stores[edge_type] = EdgeStore(
            edge_type=edge_type,
            src=new_src[sort_idx],
            dst=new_dst[sort_idx],
            weight=new_weight[sort_idx],
            delay=new_delay[sort_idx],
            release_prob=new_release[sort_idx],
            facilitation=new_facil[sort_idx],
            depression=new_depr[sort_idx],
            pre_trace=new_pre[sort_idx],
            post_trace=new_post[sort_idx],
            dst_ptr=build_dst_ptr(new_dst[sort_idx], self.n_nodes),
        )

    def remove_edges(self, edge_type: EdgeType, mask: Tensor) -> None:
        """Remove edges where mask is True. Compacts and rebuilds dst_ptr."""
        store = self._edge_stores[edge_type]
        keep = ~mask

        if not keep.any():
            del self._edge_stores[edge_type]
            return

        self._edge_stores[edge_type] = EdgeStore(
            edge_type=edge_type,
            src=store.src[keep],
            dst=store.dst[keep],
            weight=store.weight[keep],
            delay=store.delay[keep],
            release_prob=store.release_prob[keep],
            facilitation=store.facilitation[keep],
            depression=store.depression[keep],
            pre_trace=store.pre_trace[keep],
            post_trace=store.post_trace[keep],
            dst_ptr=build_dst_ptr(store.dst[keep], self.n_nodes),
        )

    def increment_step(self) -> None:
        self._step_count += 1

    def state_dict(self) -> dict:
        """Serialize full graph state for checkpointing."""
        state = {
            "step_count": self._step_count,
            "config": self.config.model_dump(),
            "node_state": {k: v.cpu() for k, v in self._node_state.__dict__.items()
                           if isinstance(v, Tensor)},
            "edge_stores": {},
        }
        for etype, store in self._edge_stores.items():
            state["edge_stores"][etype.value] = {
                k: v.cpu() if isinstance(v, Tensor) else v
                for k, v in store.__dict__.items()
            }
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> NeuromorphicGraph:
        """Restore graph from checkpoint."""
        config = GraphBrainConfig.model_validate(state["config"])
        graph = cls(config)
        graph._step_count = state["step_count"]

        # Restore node state
        device = graph._device
        ns_data = state["node_state"]
        graph._node_state = NodeState(**{k: v.to(device) for k, v in ns_data.items()})

        # Restore edge stores
        for etype_val, es_data in state["edge_stores"].items():
            etype = EdgeType(etype_val)
            tensors = {k: v.to(device) if isinstance(v, Tensor) else v
                       for k, v in es_data.items()}
            graph._edge_stores[etype] = EdgeStore(**tensors)

        return graph

    def to(self, device: str) -> NeuromorphicGraph:
        """Move entire graph to device. Returns self for chaining."""
        self._device = device
        if self._node_state is not None:
            self._node_state = self._node_state.to(device)
        self._edge_stores = {
            k: v.to(device) for k, v in self._edge_stores.items()
        }
        return self

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [
            f"NeuromorphicGraph on {self._device}",
            f"  Nodes: {self.n_nodes} total",
        ]
        if self._node_state is not None:
            for nt in NodeType:
                count = int(self._node_state.type_mask(nt).sum())
                lines.append(f"    {nt.name}: {count}")
        lines.append(f"  Edges: {self.n_edges()} total")
        for et in EdgeType:
            if et in self._edge_stores:
                lines.append(f"    {et.name}: {self._edge_stores[et].n_edges}")
        return "\n".join(lines)
