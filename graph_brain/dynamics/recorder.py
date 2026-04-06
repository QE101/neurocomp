"""Time-series recorder for simulation state.

Uses pre-allocated ring buffers to cap memory at O(max_steps * metrics)
rather than O(total_steps * N).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.types import EdgeType, NodeType


class StateRecorder:
    """Records scalar metrics and optional node-level snapshots."""

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._metrics: dict[str, list[float]] = {}
        self._snapshots: list[dict[str, Tensor]] = []
        self._record_count = 0

    def record_metrics(self, graph: NeuromorphicGraph) -> dict[str, float]:
        """Compute and store scalar metrics for this step. Returns the metrics dict."""
        ns = graph.node_state
        metrics: dict[str, float] = {"step": float(graph.step_count)}

        # Per-type mean output
        for nt in NodeType:
            mask = ns.type_mask(nt)
            if mask.any():
                metrics[f"output_mean_{nt.name}"] = float(ns.output[mask].mean())
                metrics[f"output_std_{nt.name}"] = float(ns.output[mask].std())

        # Overall activity
        metrics["output_mean"] = float(ns.output.mean())
        metrics["output_max"] = float(ns.output.max())
        metrics["activity_ema_mean"] = float(ns.activity_ema.mean())

        # Per-type edge weight stats
        for et in EdgeType:
            if graph.has_edge_type(et):
                store = graph.edge_store(et)
                metrics[f"weight_mean_{et.name}"] = float(store.weight.mean())
                metrics[f"weight_std_{et.name}"] = float(store.weight.std())
                metrics[f"n_edges_{et.name}"] = float(store.n_edges)

        # Intrinsic params
        metrics["threshold_mean"] = float(ns.threshold.mean())
        metrics["gain_mean"] = float(ns.gain.mean())

        # Total edges
        metrics["n_edges_total"] = float(graph.n_edges())

        # Store
        for k, v in metrics.items():
            self._metrics.setdefault(k, []).append(v)

        self._record_count += 1
        return metrics

    def record_snapshot(self, graph: NeuromorphicGraph) -> None:
        """Store a full node-level snapshot (for detailed analysis)."""
        if len(self._snapshots) >= self.max_records:
            return  # cap memory

        ns = graph.node_state
        self._snapshots.append({
            "step": graph.step_count,
            "output": ns.output.cpu().clone(),
            "basal": ns.basal.cpu().clone(),
            "apical": ns.apical.cpu().clone(),
        })

    def get_metric(self, name: str) -> list[float]:
        """Get time series for a named metric."""
        return self._metrics.get(name, [])

    def get_all_metrics(self) -> dict[str, list[float]]:
        """Get all recorded metrics."""
        return dict(self._metrics)

    def get_snapshots(self) -> list[dict[str, Tensor]]:
        return self._snapshots

    @property
    def n_records(self) -> int:
        return self._record_count

    def reset(self) -> None:
        self._metrics.clear()
        self._snapshots.clear()
        self._record_count = 0
