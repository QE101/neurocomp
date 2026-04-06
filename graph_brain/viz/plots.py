"""Static analysis plots for post-simulation analysis.

Uses matplotlib for publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.dynamics.recorder import StateRecorder
from graph_brain.types import EdgeType, NodeType


def plot_metric_timeseries(
    recorder: StateRecorder,
    metrics: list[str],
    title: str = "Metrics Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot one or more scalar metrics over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in metrics:
        data = recorder.get_metric(name)
        if data:
            ax.plot(data, label=name, alpha=0.8)
    ax.set_xlabel("Record Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_output_by_type(
    recorder: StateRecorder,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean output per node type over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"EXCITATORY": "tab:blue", "PV": "tab:red", "SST": "tab:green", "VIP": "tab:orange"}
    for nt in NodeType:
        key = f"output_mean_{nt.name}"
        data = recorder.get_metric(key)
        if data:
            ax.plot(data, label=nt.name, color=colors.get(nt.name, None), alpha=0.8)
    ax.set_xlabel("Record Step")
    ax.set_ylabel("Mean Output")
    ax.set_title("Activity by Node Type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_weight_distributions(
    graph: NeuromorphicGraph,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of edge weights per type."""
    edge_types = [et for et in EdgeType if graph.has_edge_type(et)]
    n_types = len(edge_types)
    if n_types == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edges", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4), squeeze=False)
    for i, et in enumerate(edge_types):
        ax = axes[0, i]
        weights = graph.edge_store(et).weight.cpu().detach().numpy()
        ax.hist(weights, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{et.name}\n({len(weights)} edges)")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
    fig.suptitle("Edge Weight Distributions")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_weight_evolution(
    recorder: StateRecorder,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean weight per edge type over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for et in EdgeType:
        key = f"weight_mean_{et.name}"
        data = recorder.get_metric(key)
        if data:
            ax.plot(data, label=et.name, alpha=0.8)
    ax.set_xlabel("Record Step")
    ax.set_ylabel("Mean Weight")
    ax.set_title("Weight Evolution by Edge Type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_activity_raster(
    snapshots: list[dict],
    max_nodes: int = 500,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Activity raster plot from recorded snapshots."""
    if not snapshots:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No snapshots", ha="center", va="center")
        return fig

    n_steps = len(snapshots)
    n_nodes = min(snapshots[0]["output"].shape[0], max_nodes)
    raster = np.zeros((n_nodes, n_steps))

    for t, snap in enumerate(snapshots):
        raster[:, t] = snap["output"][:n_nodes].numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(raster, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Node Index")
    ax.set_title("Activity Raster")
    plt.colorbar(im, ax=ax, label="Output")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ei_balance(
    recorder: StateRecorder,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot excitatory vs inhibitory activity over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    exc = recorder.get_metric("output_mean_EXCITATORY")
    pv = recorder.get_metric("output_mean_PV")
    sst = recorder.get_metric("output_mean_SST")

    if exc:
        ax.plot(exc, label="Excitatory", color="tab:blue", alpha=0.8)
    if pv:
        ax.plot(pv, label="PV (perisomatic inhib)", color="tab:red", alpha=0.8)
    if sst:
        ax.plot(sst, label="SST (dendritic inhib)", color="tab:green", alpha=0.8)

    ax.set_xlabel("Record Step")
    ax.set_ylabel("Mean Output")
    ax.set_title("E/I Balance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def create_full_report(
    graph: NeuromorphicGraph,
    recorder: StateRecorder,
    output_dir: str = "results",
) -> None:
    """Generate all analysis plots and save to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_output_by_type(recorder, save_path=str(out / "output_by_type.png"))
    plot_weight_distributions(graph, save_path=str(out / "weight_distributions.png"))
    plot_weight_evolution(recorder, save_path=str(out / "weight_evolution.png"))
    plot_ei_balance(recorder, save_path=str(out / "ei_balance.png"))

    snapshots = recorder.get_snapshots()
    if snapshots:
        plot_activity_raster(snapshots, save_path=str(out / "activity_raster.png"))

    plt.close("all")
    print(f"Report saved to {out}/")
