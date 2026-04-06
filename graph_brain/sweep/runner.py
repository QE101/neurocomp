"""Parameter sweep orchestrator for finding stable regimes.

Runs multiple configs in sequence (or parallel via ProcessPoolExecutor),
classifying each as STABLE, EXPLODED, or DIED.
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from graph_brain.config import GraphBrainConfig


@dataclass
class SweepResult:
    """Result of evaluating one configuration."""
    config_id: int
    params: dict[str, Any]
    status: str  # "STABLE", "EXPLODED", "DIED", "ERROR"
    final_output_mean: float
    final_activity_ema: float
    final_weight_mean: float
    output_std: float
    max_output: float
    min_output_mean: float
    run_time_sec: float
    error_msg: Optional[str] = None


def evaluate_config(
    config_id: int,
    base_config: GraphBrainConfig,
    overrides: dict[str, Any],
    n_steps: int = 2000,
) -> SweepResult:
    """Evaluate a single config. Designed to run in a subprocess."""
    try:
        config = base_config.with_overrides(**overrides)
        # Force CPU for sweep (avoid GPU contention in parallel)
        config = config.with_overrides(**{"simulation.device": "cpu"})

        from graph_brain.core.graph import NeuromorphicGraph
        from graph_brain.dynamics.simulator import Simulator

        t0 = time.perf_counter()
        graph = NeuromorphicGraph(config)
        graph.initialize()
        sim = Simulator(graph, config)

        # Track metrics for classification
        output_means = []
        output_maxes = []

        for step in range(n_steps):
            sim.step()
            if step % 10 == 0:
                ns = graph.node_state
                output_means.append(float(ns.output.mean()))
                output_maxes.append(float(ns.output.max()))

        run_time = time.perf_counter() - t0
        ns = graph.node_state

        # Classification
        output_arr = np.array(output_means)
        status = "STABLE"

        # EXPLODED: output grows > 10x initial within first half
        if len(output_arr) > 10:
            initial_mean = max(output_arr[:5].mean(), 1e-6)
            if output_arr[:len(output_arr)//2].max() > initial_mean * 10:
                status = "EXPLODED"

        # DIED: output stays < 0.001 for last 25% of run
        quarter = max(1, len(output_arr) // 4)
        if output_arr[-quarter:].mean() < 0.001:
            status = "DIED"

        # Check for NaN
        if np.isnan(output_arr).any():
            status = "EXPLODED"

        # Collect weight stats
        weight_means = []
        for et_val in range(6):
            from graph_brain.types import EdgeType
            et = EdgeType(et_val)
            if graph.has_edge_type(et):
                weight_means.append(float(graph.edge_store(et).weight.mean()))

        return SweepResult(
            config_id=config_id,
            params=overrides,
            status=status,
            final_output_mean=float(ns.output.mean()),
            final_activity_ema=float(ns.activity_ema.mean()),
            final_weight_mean=np.mean(weight_means) if weight_means else 0.0,
            output_std=float(output_arr.std()),
            max_output=float(output_arr.max()),
            min_output_mean=float(output_arr.min()),
            run_time_sec=run_time,
        )
    except Exception as e:
        return SweepResult(
            config_id=config_id,
            params=overrides,
            status="ERROR",
            final_output_mean=0, final_activity_ema=0, final_weight_mean=0,
            output_std=0, max_output=0, min_output_mean=0,
            run_time_sec=0, error_msg=str(e),
        )


class SweepRunner:
    """Orchestrates parameter sweeps over the configuration space."""

    def __init__(
        self,
        base_config: GraphBrainConfig,
        sweep_spec: dict[str, list[Any]],
    ):
        """
        Args:
            base_config: starting config
            sweep_spec: maps dotted param paths to value lists.
                e.g. {"edges.stdp.learning_rate": [0.001, 0.01, 0.1]}
        """
        self.base_config = base_config
        self.sweep_spec = sweep_spec

    def grid_configs(self) -> list[dict[str, Any]]:
        """Generate all grid combinations."""
        keys = list(self.sweep_spec.keys())
        values = list(self.sweep_spec.values())
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs

    def random_configs(self, n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
        """Generate random samples from the sweep space."""
        rng = np.random.RandomState(seed)
        keys = list(self.sweep_spec.keys())
        values = list(self.sweep_spec.values())
        configs = []
        for _ in range(n_samples):
            combo = {k: rng.choice(v) for k, v in zip(keys, values)}
            configs.append(combo)
        return configs

    def run(
        self,
        configs: Optional[list[dict[str, Any]]] = None,
        n_samples: int = 50,
        strategy: str = "random",
        n_steps: int = 2000,
        n_workers: int = 1,
    ) -> list[SweepResult]:
        """Run the sweep.

        Args:
            configs: explicit config list (overrides strategy)
            n_samples: number of random samples (if strategy="random")
            strategy: "grid" or "random"
            n_steps: simulation steps per config
            n_workers: parallel workers (1 = sequential)
        """
        if configs is None:
            if strategy == "grid":
                configs = self.grid_configs()
            else:
                configs = self.random_configs(n_samples)

        print(f"Running sweep: {len(configs)} configs, {n_steps} steps each, {n_workers} workers")

        results = []

        if n_workers <= 1:
            for i, overrides in enumerate(configs):
                print(f"  [{i+1}/{len(configs)}] {overrides}")
                result = evaluate_config(i, self.base_config, overrides, n_steps)
                print(f"    → {result.status} (output={result.final_output_mean:.4f}, {result.run_time_sec:.1f}s)")
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(evaluate_config, i, self.base_config, overrides, n_steps): i
                    for i, overrides in enumerate(configs)
                }
                for future in as_completed(futures):
                    result = future.result()
                    print(f"  [{result.config_id+1}/{len(configs)}] {result.status} "
                          f"(output={result.final_output_mean:.4f})")
                    results.append(result)

        results.sort(key=lambda r: r.config_id)
        return results
