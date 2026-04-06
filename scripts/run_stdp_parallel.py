"""Phase 1A.2: STDP-PC Interaction — 4 conditions in PARALLEL.

Each condition runs in its own process with its own CUDA context.
"""

import sys
import time
import json
from multiprocessing import Process, Queue

sys.path.insert(0, ".")

N_CYCLES = 3000
PD = 50
STRENGTH = 2.0

BASE = {
    "nodes": {"n_excitatory": 2000, "n_pv": 175, "n_sst": 175, "n_vip": 150, "noise_std": 0.005},
    "edges": {"structural": {"enabled": False}},
    "simulation": {"device": "cuda", "seed": 42, "record_interval": 50},
    "hierarchy": {"enabled": True, "error_ratio": 0.4, "pc_learning_rate": 0.1,
                  "inter_level_p": 0.3, "inter_level_sigma": 0.5,
                  "pattern_duration": 50, "input_strength": 2.0},
}


def run_condition(name, driving_rule, result_queue):
    """Run one condition in its own process."""
    import torch
    import numpy as np
    from graph_brain.config import GraphBrainConfig
    from graph_brain.core.graph import NeuromorphicGraph
    from graph_brain.core.message_passing import TypedMessagePasser
    from graph_brain.edges.homeostatic import HomeostaticScaling
    from graph_brain.edges.short_term import ShortTermPlasticity
    from graph_brain.edges.stdp import STDP, ThreeFactorSTDP
    from graph_brain.hierarchy import HierarchyBuilder
    from graph_brain.nodes.intrinsic import IntrinsicPlasticity
    from graph_brain.nodes.predictive_coding import PredictiveCodingModel, PCWeightUpdate
    from graph_brain.types import EdgeType, HierarchyLevel, NodeRole

    config = GraphBrainConfig.from_dict(BASE)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    HierarchyBuilder(config).build(graph)

    ns = graph.node_state
    l1_err = torch.where(ns.role_level_mask(NodeRole.ERROR, HierarchyLevel.LEVEL_1))[0]
    pa = l1_err[:l1_err.shape[0] // 2]
    pb = l1_err[l1_err.shape[0] // 2:]

    # Components
    mp = TypedMessagePasser(config, graph.n_nodes, graph.device)
    nm = PredictiveCodingModel(config)
    pcw_mod = PCWeightUpdate(config)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    # Driving edge rule
    if driving_rule == "stdp":
        drv_updater = STDP(config.edges.stdp)
        drv_mode = "stdp"
    elif driving_rule == "three_factor":
        drv_updater = ThreeFactorSTDP(config.edges.stdp)
        drv_mode = "stdp"
    elif driving_rule == "pc_native":
        drv_updater = PCWeightUpdate(config)
        drv_mode = "pc"
    elif driving_rule == "fixed":
        drv_updater = None
        drv_mode = "none"

    t0 = time.perf_counter()
    cycle_errors = []

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                step = graph.step_count
                mp.send_messages(graph, step)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                nm.step(ns, inputs, step * 1.0)

                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)

                if drv_updater is not None and graph.has_edge_type(EdgeType.DRIVING):
                    if drv_mode == "stdp":
                        drv_updater.update(graph.edge_store(EdgeType.DRIVING), ns, 1.0)
                    else:
                        drv_updater.update(graph.edge_store(EdgeType.DRIVING), ns)

                if graph.has_edge_type(EdgeType.MODULATORY):
                    pcw_mod.update(graph.edge_store(EdgeType.MODULATORY), ns)

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)

                graph.increment_step()
                err_sum += ns.output[l1_err].mean().item()
                n += 1

        cycle_errors.append(err_sum / n)
        if (cycle + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - cycle_errors[-1] / cycle_errors[0]) * 100
            print(f"  [{name}] Cycle {cycle+1}: err={cycle_errors[-1]:.4f} sup={sup:.1f}% ({elapsed:.0f}s)", flush=True)

    # Baseline + violation
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, step * 1.0)
        graph.increment_step()

    baseline = []
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += STRENGTH
        nm.step(ns, inputs, step * 1.0)
        graph.increment_step()
        baseline.append(ns.output[l1_err].mean().item())

    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, step * 1.0)
        graph.increment_step()

    violation = []
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, step * 1.0)
        graph.increment_step()
        violation.append(ns.output[l1_err].mean().item())

    bl = float(np.mean(baseline))
    vl = float(np.mean(violation))
    total = time.perf_counter() - t0

    result = {
        "name": name,
        "driving_rule": driving_rule,
        "suppression": float((1 - cycle_errors[-1] / cycle_errors[0]) * 100),
        "baseline": bl,
        "violation": vl,
        "ratio": float(vl / max(bl, 1e-8)),
        "error_first": float(cycle_errors[0]),
        "error_500": float(cycle_errors[499]),
        "error_1000": float(cycle_errors[999]),
        "error_3000": float(cycle_errors[-1]),
        "time": total,
        "cycle_errors": [float(x) for x in cycle_errors],
    }

    print(f"  [{name}] DONE: sup={result['suppression']:.1f}% mismatch={result['ratio']:.3f}x ({total:.0f}s)", flush=True)
    result_queue.put(result)


def main():
    print("Phase 1A.2: STDP-PC Interaction — PARALLEL", flush=True)
    print(f"{N_CYCLES} cycles per condition, 4 conditions in parallel\n", flush=True)

    conditions = [
        ("STDP+PC", "stdp"),
        ("Fixed+PC", "fixed"),
        ("3Factor+PC", "three_factor"),
        ("PC-only", "pc_native"),
    ]

    queue = Queue()
    procs = []

    for name, rule in conditions:
        p = Process(target=run_condition, args=(name, rule, queue))
        p.start()
        procs.append(p)
        print(f"  Launched: {name} (pid={p.pid})", flush=True)

    # Wait for all
    for p in procs:
        p.join()

    # Collect results
    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r["name"])

    # Summary
    print(f"\n{'='*85}", flush=True)
    print("COMPARISON RESULTS", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"{'Condition':<15} {'Sup%':>6} {'Mismatch':>9} {'Err@500':>8} {'Err@1000':>9} {'Err@3000':>9} {'Time':>6}", flush=True)
    print("-" * 85, flush=True)
    for r in results:
        print(f"{r['name']:<15} {r['suppression']:>5.1f}% {r['ratio']:>8.3f}x {r['error_500']:>8.3f} {r['error_1000']:>8.3f} {r['error_3000']:>8.3f} {r['time']:>5.0f}s", flush=True)

    best_sup = max(results, key=lambda r: r["suppression"])
    best_mm = max(results, key=lambda r: r["ratio"])
    print(f"\nWinner (suppression): {best_sup['name']} at {best_sup['suppression']:.1f}%", flush=True)
    print(f"Winner (mismatch):    {best_mm['name']} at {best_mm['ratio']:.3f}x", flush=True)

    import torch
    torch.save(results, "stdp_comparison_results.pt")
    print("\nSaved to stdp_comparison_results.pt", flush=True)


if __name__ == "__main__":
    main()
