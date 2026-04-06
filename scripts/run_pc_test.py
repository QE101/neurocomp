"""Phase 1A Experiment: Temporal sequence prediction with predictive coding.

Test: A-B-A-B-A-B alternating pattern, then violation A-A.
Success criterion: error nodes spike on violation (mismatch negativity).

The system should learn:
    "When I see A, B comes next. When I see B, A comes next."

Violation: present A-A instead of A-B. Error nodes should fire hard
because they predicted B but got A.
"""

import sys
import time

import torch
import numpy as np

sys.path.insert(0, ".")

from rich.console import Console

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.dynamics.simulator import Simulator
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.types import EdgeType, HierarchyLevel, NodeRole, NodeType


def main():
    console = Console()
    console.print("[bold]Phase 1A: Temporal Predictive Coding Test[/bold]")
    console.print("Pattern: A-B-A-B-A-B... then violation A-A\n")

    # Config: moderate graph with hierarchy enabled
    config = GraphBrainConfig.from_dict({
        "nodes": {
            "n_excitatory": 2000,
            "n_pv": 175,
            "n_sst": 175,
            "n_vip": 150,
            "noise_std": 0.005,
        },
        "edges": {
            "structural": {"enabled": False},  # disable for clean PC test
        },
        "simulation": {
            "device": "cuda",
            "seed": 42,
            "record_interval": 1,
        },
        "hierarchy": {
            "enabled": True,
            "error_ratio": 0.4,
            "pc_learning_rate": 0.1,
            "precision_base": 1.0,
            "inter_level_p": 0.3,
            "inter_level_sigma": 0.5,
            "pattern_duration": 50,
            "input_strength": 2.0,
        },
    })

    # Build graph
    console.print("Building graph...")
    graph = NeuromorphicGraph(config)
    graph.initialize()
    console.print(f"  {graph.n_nodes} nodes, {graph.n_edges()} edges")

    # Build hierarchy
    console.print("Building hierarchy...")
    builder = HierarchyBuilder(config)
    stats = builder.build(graph)
    console.print(f"  {stats}")
    console.print(builder.summary(graph))

    # Identify input target nodes
    ns = graph.node_state
    l1_error = torch.where(ns.role_level_mask(NodeRole.ERROR, HierarchyLevel.LEVEL_1))[0]
    l1_repr = torch.where(ns.role_level_mask(NodeRole.REPRESENTATION, HierarchyLevel.LEVEL_1))[0]
    l2_repr = torch.where(ns.role_level_mask(NodeRole.REPRESENTATION, HierarchyLevel.LEVEL_2))[0]

    # Define patterns A and B: different subsets of Level 1 error nodes
    n_l1_error = l1_error.shape[0]
    half = n_l1_error // 2
    pattern_a_nodes = l1_error[:half]    # first half of L1 error nodes
    pattern_b_nodes = l1_error[half:]    # second half

    input_strength = config.hierarchy.input_strength
    pattern_duration = config.hierarchy.pattern_duration

    console.print(f"\n  Pattern A: {pattern_a_nodes.shape[0]} nodes")
    console.print(f"  Pattern B: {pattern_b_nodes.shape[0]} nodes")
    console.print(f"  Duration: {pattern_duration} steps per pattern")

    # Build simulator
    sim = Simulator(graph, config)

    # --- Phase 1: Learning (A-B-A-B repeated) ---
    n_learning_cycles = 20
    n_learning_steps = n_learning_cycles * 2 * pattern_duration  # 20 full A-B cycles

    console.print(f"\n[bold]Phase 1: Learning ({n_learning_cycles} A-B cycles, {n_learning_steps} steps)[/bold]")

    # Track error node activity over time
    l1_error_activity = []
    l2_repr_activity = []
    pattern_labels = []

    t0 = time.perf_counter()

    for cycle in range(n_learning_cycles):
        for pattern_idx, (pattern_name, pattern_nodes) in enumerate([("A", pattern_a_nodes), ("B", pattern_b_nodes)]):
            for step_in_pattern in range(pattern_duration):
                # Inject current pattern into L1
                values = torch.full((pattern_nodes.shape[0],), input_strength, device=graph.device)
                sim.inject_input(pattern_nodes, values)
                sim.step()

                # Record
                l1_err_out = ns.output[l1_error].mean().item()
                l2_rep_out = ns.output[l2_repr].mean().item()
                l1_error_activity.append(l1_err_out)
                l2_repr_activity.append(l2_rep_out)
                pattern_labels.append(pattern_name)

        if (cycle + 1) % 5 == 0:
            recent_err = np.mean(l1_error_activity[-pattern_duration*2:])
            recent_repr = np.mean(l2_repr_activity[-pattern_duration*2:])
            console.print(f"  Cycle {cycle+1}/{n_learning_cycles}: "
                          f"L1 error={recent_err:.4f}, L2 repr={recent_repr:.4f}")

    learn_time = time.perf_counter() - t0
    console.print(f"  Learning done in {learn_time:.1f}s")

    # --- Phase 2: Baseline (one more A-B to measure normal error) ---
    console.print(f"\n[bold]Phase 2: Baseline A-B cycle[/bold]")

    baseline_a_error = []
    baseline_b_error = []

    # Pattern A
    for step in range(pattern_duration):
        values = torch.full((pattern_a_nodes.shape[0],), input_strength, device=graph.device)
        sim.inject_input(pattern_a_nodes, values)
        sim.step()
        baseline_a_error.append(ns.output[l1_error].mean().item())

    # Pattern B (expected after A)
    for step in range(pattern_duration):
        values = torch.full((pattern_b_nodes.shape[0],), input_strength, device=graph.device)
        sim.inject_input(pattern_b_nodes, values)
        sim.step()
        baseline_b_error.append(ns.output[l1_error].mean().item())

    baseline_a_mean = np.mean(baseline_a_error)
    baseline_b_mean = np.mean(baseline_b_error)
    console.print(f"  Baseline L1 error during A: {baseline_a_mean:.4f}")
    console.print(f"  Baseline L1 error during B: {baseline_b_mean:.4f}")

    # --- Phase 3: Violation (A-A instead of A-B) ---
    console.print(f"\n[bold]Phase 3: VIOLATION — presenting A-A (expected B)[/bold]")

    violation_error = []

    # Pattern A (normal)
    for step in range(pattern_duration):
        values = torch.full((pattern_a_nodes.shape[0],), input_strength, device=graph.device)
        sim.inject_input(pattern_a_nodes, values)
        sim.step()

    # Pattern A again (VIOLATION — system expected B)
    for step in range(pattern_duration):
        values = torch.full((pattern_a_nodes.shape[0],), input_strength, device=graph.device)
        sim.inject_input(pattern_a_nodes, values)
        sim.step()
        violation_error.append(ns.output[l1_error].mean().item())

    violation_mean = np.mean(violation_error)
    console.print(f"  Violation L1 error (A instead of B): {violation_mean:.4f}")

    # --- Results ---
    console.print(f"\n{'='*60}")
    console.print("[bold]RESULTS[/bold]")
    console.print(f"  Baseline B error (expected):    {baseline_b_mean:.4f}")
    console.print(f"  Violation A error (unexpected):  {violation_mean:.4f}")

    if violation_mean > baseline_b_mean * 1.1:
        ratio = violation_mean / max(baseline_b_mean, 1e-8)
        console.print(f"  [green]MISMATCH NEGATIVITY DETECTED[/green]: "
                       f"violation/baseline = {ratio:.2f}x")
    elif violation_mean > baseline_b_mean:
        ratio = violation_mean / max(baseline_b_mean, 1e-8)
        console.print(f"  [yellow]WEAK SIGNAL[/yellow]: "
                       f"violation/baseline = {ratio:.2f}x (need >1.1x)")
    else:
        console.print(f"  [red]NO MISMATCH DETECTED[/red]: "
                       f"violation error ≤ baseline error")

    console.print(f"{'='*60}")

    # --- Detailed analysis ---
    console.print(f"\n[bold]Learning curve (L1 error by cycle):[/bold]")
    for cycle in range(n_learning_cycles):
        start = cycle * 2 * pattern_duration
        end = start + 2 * pattern_duration
        cycle_error = np.mean(l1_error_activity[start:end])
        bar = "#" * int(cycle_error * 50)
        console.print(f"  Cycle {cycle+1:2d}: {cycle_error:.4f} {bar}")

    # Save data for analysis
    torch.save({
        "l1_error_activity": l1_error_activity,
        "l2_repr_activity": l2_repr_activity,
        "pattern_labels": pattern_labels,
        "baseline_a_error": baseline_a_error,
        "baseline_b_error": baseline_b_error,
        "violation_error": violation_error,
        "config": config.model_dump(),
    }, "pc_test_results.pt")
    console.print("\nResults saved to pc_test_results.pt")


if __name__ == "__main__":
    main()
