"""Real-time PyQt6 + pyqtgraph dashboard for simulation monitoring.

Four panels:
1. 3D node scatter (colored by output activation)
2. Activity heatmap (time x node rolling window)
3. Scalar metric plots (output, weight, E/I)
4. Weight distribution histogram

Usage:
    dashboard = Dashboard(graph, simulator)
    dashboard.run()  # blocks, runs sim + viz in timer loop
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.dynamics.simulator import Simulator
from graph_brain.types import NodeType, EdgeType


class Dashboard:
    """Real-time simulation dashboard using PyQt6 + pyqtgraph."""

    def __init__(
        self,
        graph: NeuromorphicGraph,
        simulator: Simulator,
        update_interval_ms: int = 50,
        steps_per_update: int = 10,
        history_length: int = 500,
    ):
        self.graph = graph
        self.simulator = simulator
        self.update_interval_ms = update_interval_ms
        self.steps_per_update = steps_per_update
        self.history_length = history_length

        # Activity history buffer
        self.n_display_nodes = min(graph.n_nodes, 500)
        self.activity_history = np.zeros((self.n_display_nodes, history_length))
        self.history_idx = 0

        # Metric history
        self.metric_history: dict[str, list[float]] = {
            "output_exc": [], "output_pv": [], "output_sst": [],
            "weight_driving": [], "weight_modulatory": [],
        }

    def run(self) -> None:
        """Launch the dashboard. Blocks until window is closed."""
        import pyqtgraph as pg
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        import pyqtgraph.opengl as gl
        import sys

        app = QApplication.instance() or QApplication(sys.argv)

        # Main window with grid layout
        win = pg.GraphicsLayoutWidget(title="Graph Brain — Live Dashboard")
        win.resize(1600, 900)

        # --- Panel 1: Activity by type (top-left) ---
        p1 = win.addPlot(title="Activity by Node Type")
        p1.addLegend(offset=(60, 10))
        p1.setLabel("bottom", "Step")
        p1.setLabel("left", "Mean Output")
        self.curve_exc = p1.plot(pen=pg.mkPen("b", width=2), name="EXC")
        self.curve_pv = p1.plot(pen=pg.mkPen("r", width=2), name="PV")
        self.curve_sst = p1.plot(pen=pg.mkPen("g", width=2), name="SST")

        # --- Panel 2: Weight evolution (top-right) ---
        p2 = win.addPlot(title="Mean Weight by Edge Type")
        p2.addLegend(offset=(60, 10))
        p2.setLabel("bottom", "Step")
        p2.setLabel("left", "Mean Weight")
        self.curve_w_drv = p2.plot(pen=pg.mkPen("c", width=2), name="DRIVING")
        self.curve_w_mod = p2.plot(pen=pg.mkPen("m", width=2), name="MODULATORY")

        win.nextRow()

        # --- Panel 3: Activity heatmap (bottom-left) ---
        p3 = win.addPlot(title="Activity Heatmap (node x time)")
        self.heatmap = pg.ImageItem()
        p3.addItem(self.heatmap)
        p3.setLabel("bottom", "Time (steps)")
        p3.setLabel("left", "Node Index")

        # Color map
        colors = [
            (0, 0, 50),      # dark blue (silent)
            (0, 100, 200),   # blue
            (0, 200, 100),   # green
            (255, 255, 0),   # yellow
            (255, 50, 0),    # red (max activity)
        ]
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(colors)), color=colors)
        self.heatmap.setLookupTable(cmap.getLookupTable())

        # --- Panel 4: Intrinsic params (bottom-right) ---
        p4 = win.addPlot(title="Threshold & Gain")
        p4.addLegend(offset=(60, 10))
        p4.setLabel("bottom", "Step")
        self.curve_threshold = p4.plot(pen=pg.mkPen("y", width=2), name="Threshold")
        self.curve_gain = p4.plot(pen=pg.mkPen("w", width=2), name="Gain")
        self.threshold_history: list[float] = []
        self.gain_history: list[float] = []

        # --- 3D scatter in separate window ---
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setWindowTitle("Graph Brain — 3D Nodes")
        self.gl_widget.resize(800, 600)
        self.gl_widget.setCameraPosition(distance=2.0)

        pos = self.graph.node_state.position.cpu().numpy()
        colors_3d = np.ones((pos.shape[0], 4)) * 0.3
        colors_3d[:, 3] = 0.8
        sizes = np.ones(pos.shape[0]) * 3.0

        self.scatter_3d = gl.GLScatterPlotItem(pos=pos, color=colors_3d, size=sizes)
        self.gl_widget.addItem(self.scatter_3d)

        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(1, 1, 1)
        self.gl_widget.addItem(grid)

        # Timer for updates
        timer = QTimer()
        timer.timeout.connect(self._update)
        timer.start(self.update_interval_ms)

        win.show()
        self.gl_widget.show()
        app.exec()

    def _update(self) -> None:
        """Run simulation steps and update all panels."""
        # Run simulation steps
        for _ in range(self.steps_per_update):
            self.simulator.step()

        ns = self.graph.node_state

        # Update activity history
        output = ns.output[:self.n_display_nodes].cpu().detach().numpy()
        col = self.history_idx % self.history_length
        self.activity_history[:, col] = output
        self.history_idx += 1

        # Update metric history
        exc_mask = (ns.node_type == NodeType.EXCITATORY).cpu()
        pv_mask = (ns.node_type == NodeType.PV).cpu()
        sst_mask = (ns.node_type == NodeType.SST).cpu()
        out_cpu = ns.output.cpu().detach()

        self.metric_history["output_exc"].append(float(out_cpu[exc_mask].mean()))
        self.metric_history["output_pv"].append(float(out_cpu[pv_mask].mean()) if pv_mask.any() else 0)
        self.metric_history["output_sst"].append(float(out_cpu[sst_mask].mean()) if sst_mask.any() else 0)

        if self.graph.has_edge_type(EdgeType.DRIVING):
            self.metric_history["weight_driving"].append(
                float(self.graph.edge_store(EdgeType.DRIVING).weight.mean()))
        if self.graph.has_edge_type(EdgeType.MODULATORY):
            self.metric_history["weight_modulatory"].append(
                float(self.graph.edge_store(EdgeType.MODULATORY).weight.mean()))

        self.threshold_history.append(float(ns.threshold.mean()))
        self.gain_history.append(float(ns.gain.mean()))

        # Update curves
        self.curve_exc.setData(self.metric_history["output_exc"][-self.history_length:])
        self.curve_pv.setData(self.metric_history["output_pv"][-self.history_length:])
        self.curve_sst.setData(self.metric_history["output_sst"][-self.history_length:])
        self.curve_w_drv.setData(self.metric_history["weight_driving"][-self.history_length:])
        self.curve_w_mod.setData(self.metric_history["weight_modulatory"][-self.history_length:])
        self.curve_threshold.setData(self.threshold_history[-self.history_length:])
        self.curve_gain.setData(self.gain_history[-self.history_length:])

        # Update heatmap
        # Roll so current time is on the right
        if self.history_idx >= self.history_length:
            start = col + 1
            display = np.roll(self.activity_history, -start, axis=1)
        else:
            display = self.activity_history[:, :self.history_idx]
        self.heatmap.setImage(display.T, autoLevels=True)

        # Update 3D scatter colors based on output
        out_full = ns.output.cpu().detach().numpy()
        out_norm = np.clip(out_full / (out_full.max() + 1e-8), 0, 1)
        colors_3d = np.zeros((len(out_norm), 4))
        colors_3d[:, 0] = out_norm          # R
        colors_3d[:, 1] = 1.0 - out_norm    # G (inverse)
        colors_3d[:, 2] = 0.3               # B
        colors_3d[:, 3] = 0.7               # alpha

        # Node type → size
        types = ns.node_type.cpu().numpy()
        sizes = np.where(types == 0, 3.0, 6.0)  # interneurons larger

        self.scatter_3d.setData(color=colors_3d, size=sizes)
