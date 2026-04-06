"""Main simulation loop orchestrating all components.

One simulation step:
    1. Compute messages (typed scatter-gather)
    2. Update node state (two-compartment model)
    3. Update STP state (fast plasticity)
    4. Update STDP traces + weights (if STDP step)
    5. Update homeostatic scaling (if homeostatic step)
    6. Update intrinsic plasticity (if IP step)
    7. Record metrics (if record step)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.dynamics.recorder import StateRecorder
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.edges.stdp import STDP
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.nodes.predictive_coding import PredictiveCodingModel, PCWeightUpdate
from graph_brain.types import EdgeType
from graph_brain.utils.profiling import StepTimer


class Simulator:
    """Main simulation loop orchestrating all components."""

    def __init__(self, graph: NeuromorphicGraph, config: Optional[GraphBrainConfig] = None):
        self.graph = graph
        self.config = config or graph.config
        cfg = self.config

        # Components
        self.message_passer = TypedMessagePasser(cfg, graph.n_nodes, graph.device)
        # Use PC model if hierarchy is enabled, otherwise standard two-compartment
        if cfg.hierarchy.enabled:
            self.node_model = PredictiveCodingModel(cfg)
            self.pc_weight_update = PCWeightUpdate(cfg)
        else:
            self.node_model = TwoCompartmentModel(cfg.nodes)
            self.pc_weight_update = None
        self.stdp = STDP(cfg.edges.stdp)
        self.homeostatic = HomeostaticScaling(cfg.edges.homeostatic)
        self.stp = ShortTermPlasticity(cfg.edges.stp)
        self.intrinsic = IntrinsicPlasticity(cfg.nodes)
        self.structural = StructuralPlasticity(cfg)

        # Recording
        self.recorder = StateRecorder()
        self.timer = StepTimer()

        # External input buffer (set via inject_input, cleared each step)
        self._external_input: Optional[torch.Tensor] = None
        self._external_indices: Optional[torch.Tensor] = None

    def step(self) -> dict[str, float]:
        """Execute one simulation timestep. Returns metrics dict."""
        graph = self.graph
        dt = self.config.nodes.dt
        step = graph.step_count
        current_time = step * dt

        # 1. Message passing: send messages into delay buffer, then read arrived messages
        with self.timer.section("message_passing"):
            self.message_passer.send_messages(graph, step)
            inputs = self.message_passer.read_inputs(step)

        # Apply external input if set (immediate, no delay)
        if self._external_input is not None and self._external_indices is not None:
            inputs.basal[self._external_indices.long()] += self._external_input
            self._external_input = None
            self._external_indices = None

        # 2. Node model update
        with self.timer.section("node_model"):
            self.node_model.step(graph.node_state, inputs, current_time)

        # 3. Short-term plasticity (every step — fast timescale)
        with self.timer.section("stp"):
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    self.stp.update(graph.edge_store(et), graph.node_state, dt)

        # 4. STDP (configurable interval) — on driving edges only when PC is active
        if step % self.config.edges.stdp.update_interval == 0:
            with self.timer.section("stdp"):
                if self.pc_weight_update is not None:
                    # PC mode: STDP on driving (feedforward) only
                    # Modulatory edges use PC-native update instead
                    if graph.has_edge_type(EdgeType.DRIVING):
                        self.stdp.update(graph.edge_store(EdgeType.DRIVING), graph.node_state, dt)
                else:
                    # Standard mode: STDP on both
                    for et in (EdgeType.DRIVING, EdgeType.MODULATORY):
                        if graph.has_edge_type(et):
                            self.stdp.update(graph.edge_store(et), graph.node_state, dt)

        # 4b. PC-native weight update on modulatory edges (every step when PC active)
        if self.pc_weight_update is not None:
            with self.timer.section("pc_weight"):
                if graph.has_edge_type(EdgeType.MODULATORY):
                    self.pc_weight_update.update(
                        graph.edge_store(EdgeType.MODULATORY), graph.node_state
                    )

        # 5. Homeostatic scaling (slower interval)
        if step % self.config.edges.homeostatic.update_interval == 0:
            with self.timer.section("homeostatic"):
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        self.homeostatic.update(graph.edge_store(et), graph.node_state, dt)

        # 6. Intrinsic plasticity (same interval as homeostatic)
        if step % self.config.edges.homeostatic.update_interval == 0:
            with self.timer.section("intrinsic"):
                self.intrinsic.update(graph.node_state)

        # 7. Structural plasticity (slowest timescale — batched)
        if step > 0 and step % self.config.edges.structural.update_interval == 0:
            with self.timer.section("structural"):
                sp_stats = self.structural.update(graph)

        # 8. Record metrics
        metrics = {}
        if step % self.config.simulation.record_interval == 0:
            with self.timer.section("recording"):
                metrics = self.recorder.record_metrics(graph)

        graph.increment_step()
        return metrics

    def run(
        self,
        n_steps: Optional[int] = None,
        callback: Optional[Callable[[int, dict], None]] = None,
        show_progress: bool = True,
    ) -> None:
        """Run simulation for n_steps.

        Args:
            n_steps: steps to run (defaults to config.simulation.n_steps)
            callback: called every step with (step, metrics_dict)
            show_progress: show rich progress bar
        """
        n_steps = n_steps or self.config.simulation.n_steps

        if show_progress:
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Simulating", total=n_steps)
                for i in range(n_steps):
                    metrics = self.step()
                    if callback:
                        callback(i, metrics)
                    progress.advance(task)
        else:
            for i in range(n_steps):
                metrics = self.step()
                if callback:
                    callback(i, metrics)

    def inject_input(self, node_indices: torch.Tensor, values: torch.Tensor) -> None:
        """Set external input to be applied at the next step's basal compartment.

        Args:
            node_indices: [K] indices of nodes to stimulate
            values: [K] input values to add to basal
        """
        self._external_indices = node_indices.to(self.graph.device)
        self._external_input = values.to(self.graph.device)

    def timing_summary(self, last_n: int = 100) -> dict[str, float]:
        """Get mean timing per section over last N steps (ms)."""
        return self.timer.summary(last_n)
