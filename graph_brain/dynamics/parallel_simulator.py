"""Parallel simulator with partitioned message passing and async weight updates.

Two parallelism levels:
  L1: Spatial partitions — local edges processed on separate CUDA streams
  L3: Async weight updates — learning rules run on a separate stream,
      overlapped with the next step's message passing

The pipeline per step:
    1. sync_weights()     — wait for previous step's weight updates
    2. send_messages()    — partitioned across streams (local edges parallel)
    3. read_inputs()      — read from delay buffer
    4. node_update()      — all nodes (vectorized, single stream)
    5. launch_weights()   — STP + STDP + PC + homeostatic on async stream
    6. record()           — metrics on default stream

Steps 1-4 are on the critical path. Step 5 overlaps with steps 1-3 of
the NEXT iteration.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from graph_brain.config import GraphBrainConfig
from graph_brain.core.async_updates import AsyncWeightUpdater
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.partition import SpatialPartitioner, PartitionedMessagePasser
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


class ParallelSimulator:
    """Simulator with partition parallelism and async weight updates."""

    def __init__(
        self,
        graph: NeuromorphicGraph,
        config: Optional[GraphBrainConfig] = None,
        n_partitions: int = 4,
    ):
        self.graph = graph
        self.config = config or graph.config
        cfg = self.config

        # Standard message passer (still handles delay buffer + cross-partition edges)
        self.message_passer = TypedMessagePasser(cfg, graph.n_nodes, graph.device)

        # Node model
        if cfg.hierarchy.enabled:
            self.node_model = PredictiveCodingModel(cfg)
            self.pc_weight_update = PCWeightUpdate(cfg)
        else:
            self.node_model = TwoCompartmentModel(cfg.nodes)
            self.pc_weight_update = None

        # Learning rules
        self.stdp = STDP(cfg.edges.stdp)
        self.homeostatic = HomeostaticScaling(cfg.edges.homeostatic)
        self.stp = ShortTermPlasticity(cfg.edges.stp)
        self.intrinsic = IntrinsicPlasticity(cfg.nodes)
        self.structural = StructuralPlasticity(cfg)

        # Partition parallelism (L1)
        partitioner = SpatialPartitioner(n_partitions)
        self.partitions = partitioner.partition(graph)
        self.partitioned_mp = PartitionedMessagePasser(graph, self.partitions)

        # Async weight updates (L3)
        self.async_updater = AsyncWeightUpdater(graph.device)

        # Recording
        self.recorder = StateRecorder()
        self.timer = StepTimer()

        # External input
        self._external_input: Optional[torch.Tensor] = None
        self._external_indices: Optional[torch.Tensor] = None

    def step(self) -> dict[str, float]:
        """One simulation step with parallel execution."""
        graph = self.graph
        dt = self.config.nodes.dt
        step = graph.step_count
        current_time = step * dt

        # 1. Sync: wait for previous step's async weight updates
        with self.timer.section("weight_sync"):
            self.async_updater.sync()

        # 2. Message passing (uses delay buffer — handles both local + cross-partition)
        with self.timer.section("message_passing"):
            self.message_passer.send_messages(graph, step)
            inputs = self.message_passer.read_inputs(step)

        # External input
        if self._external_input is not None and self._external_indices is not None:
            inputs.basal[self._external_indices.long()] += self._external_input
            self._external_input = None
            self._external_indices = None

        # 3. Node model update (critical path — must be synchronous)
        with self.timer.section("node_model"):
            self.node_model.step(graph.node_state, inputs, current_time)

        # 4. Launch async weight updates (overlaps with next step's message passing)
        def weight_updates():
            ns = graph.node_state

            # STP (every step)
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    self.stp.update(graph.edge_store(et), ns, dt)

            # PC weight update (every step when PC active)
            if self.pc_weight_update is not None and graph.has_edge_type(EdgeType.MODULATORY):
                self.pc_weight_update.update(graph.edge_store(EdgeType.MODULATORY), ns)

            # STDP (configurable interval)
            if step % self.config.edges.stdp.update_interval == 0:
                if self.pc_weight_update is not None:
                    if graph.has_edge_type(EdgeType.DRIVING):
                        self.stdp.update(graph.edge_store(EdgeType.DRIVING), ns, dt)
                else:
                    for et in (EdgeType.DRIVING, EdgeType.MODULATORY):
                        if graph.has_edge_type(et):
                            self.stdp.update(graph.edge_store(et), ns, dt)

            # Homeostatic + intrinsic (slow interval)
            if step % self.config.edges.homeostatic.update_interval == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        self.homeostatic.update(graph.edge_store(et), ns, dt)
                self.intrinsic.update(ns)

            # Structural (slowest)
            if step > 0 and step % self.config.edges.structural.update_interval == 0:
                self.structural.update(graph)

        with self.timer.section("weight_launch"):
            self.async_updater.launch(weight_updates)

        # 5. Record metrics (on default stream, doesn't read weights)
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
        """Run simulation for n_steps."""
        n_steps = n_steps or self.config.simulation.n_steps

        if show_progress:
            from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
            with Progress(
                SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
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

        # Final sync
        self.async_updater.sync()

    def inject_input(self, node_indices: torch.Tensor, values: torch.Tensor) -> None:
        self._external_indices = node_indices.to(self.graph.device)
        self._external_input = values.to(self.graph.device)

    def timing_summary(self, last_n: int = 100) -> dict[str, float]:
        return self.timer.summary(last_n)
