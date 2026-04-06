"""CLI entry point: load config, build graph, run simulation."""

import argparse
import sys
import time

sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.dynamics.simulator import Simulator
from graph_brain.utils.profiling import gpu_memory_summary


def main():
    parser = argparse.ArgumentParser(description="Run neuromorphic graph simulation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    console = Console()
    console.print("[bold]Neuromorphic Graph Simulation[/bold]")

    # Load config
    config = GraphBrainConfig.from_yaml(args.config)
    if args.steps:
        config = config.with_overrides(**{"simulation.n_steps": args.steps})
    if args.device:
        config = config.with_overrides(**{"simulation.device": args.device})
    if args.seed is not None:
        config = config.with_overrides(**{"simulation.seed": args.seed})

    # Build graph
    console.print(f"Building graph: {config.nodes.n_total} nodes...")
    t0 = time.perf_counter()
    graph = NeuromorphicGraph(config)
    graph.initialize()
    build_time = time.perf_counter() - t0
    console.print(f"  Built in {build_time:.2f}s")
    console.print(graph.summary())
    console.print(gpu_memory_summary())

    # Run simulation
    n_steps = config.simulation.n_steps
    console.print(f"\nRunning {n_steps} steps on {graph.device}...")

    sim = Simulator(graph, config)
    t0 = time.perf_counter()
    sim.run(n_steps=n_steps)
    run_time = time.perf_counter() - t0

    # Results
    console.print(f"\nCompleted in {run_time:.2f}s ({n_steps / run_time:.0f} steps/sec)")

    # Timing breakdown
    timing = sim.timing_summary(last_n=100)
    if timing:
        table = Table(title="Timing Breakdown (last 100 steps, ms/step)")
        table.add_column("Section")
        table.add_column("Mean (ms)", justify="right")
        total = sum(timing.values())
        for name, ms in sorted(timing.items(), key=lambda x: -x[1]):
            pct = ms / total * 100 if total > 0 else 0
            table.add_row(name, f"{ms:.2f} ({pct:.0f}%)")
        table.add_row("[bold]Total[/bold]", f"[bold]{total:.2f}[/bold]")
        console.print(table)

    # Final metrics
    metrics = sim.recorder.get_all_metrics()
    if "output_mean" in metrics:
        console.print(f"\nFinal output mean: {metrics['output_mean'][-1]:.6f}")
        console.print(f"Final activity EMA: {metrics['activity_ema_mean'][-1]:.6f}")
        console.print(f"Final threshold mean: {metrics['threshold_mean'][-1]:.6f}")
        console.print(f"Final gain mean: {metrics['gain_mean'][-1]:.6f}")


if __name__ == "__main__":
    main()
