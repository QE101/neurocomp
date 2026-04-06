"""Phase 2: Full oscillatory dynamics on the hand-built PC hierarchy.

Tests whether gamma/theta oscillations help or hurt PC performance.
Uses the Phase 1A hand-built hierarchy as the baseline.

Conditions:
  A. Baseline: hand-built PC, no oscillations (1A replication)
  B. + Gamma: PV gap junction coupling drives gamma oscillation
  C. + Theta: slow sinusoidal modulation of global excitability
  D. + Both: gamma nested in theta (cross-frequency coupling)

Each condition: 3000 A-B cycles, mismatch test at end.
All 4 run in parallel.
"""

import sys
import os
import time
from multiprocessing import Process, Queue

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import numpy as np


def run_condition(name, gamma_on, theta_on, result_queue):
    import torch
    from graph_brain.config import GraphBrainConfig
    from graph_brain.core.graph import NeuromorphicGraph, EdgeStore, build_dst_ptr
    from graph_brain.core.message_passing import TypedMessagePasser
    from graph_brain.dynamics.recorder import StateRecorder
    from graph_brain.edges.homeostatic import HomeostaticScaling
    from graph_brain.edges.short_term import ShortTermPlasticity
    from graph_brain.hierarchy import HierarchyBuilder
    from graph_brain.nodes.intrinsic import IntrinsicPlasticity
    from graph_brain.nodes.predictive_coding import PredictiveCodingModel, PCWeightUpdate
    from graph_brain.types import EdgeType, HierarchyLevel, NodeRole, NodeType

    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 2000, "n_pv": 175, "n_sst": 175, "n_vip": 150, "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": 42, "record_interval": 100},
        "hierarchy": {"enabled": True, "error_ratio": 0.4, "pc_learning_rate": 0.1,
                      "inter_level_p": 0.3, "inter_level_sigma": 0.5,
                      "pattern_duration": 50, "input_strength": 2.0},
    })

    graph = NeuromorphicGraph(config)
    graph.initialize()
    HierarchyBuilder(config).build(graph)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    mp = TypedMessagePasser(config, N, device)
    nm = PredictiveCodingModel(config)
    pcw = PCWeightUpdate(config)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    l1_err = torch.where(ns.role_level_mask(NodeRole.ERROR, HierarchyLevel.LEVEL_1))[0]
    pa = l1_err[:l1_err.shape[0] // 2]
    pb = l1_err[l1_err.shape[0] // 2:]
    pv_idx = torch.where(ns.type_mask(NodeType.PV))[0]

    # Setup gamma: strengthen PV-PV gap junctions
    if gamma_on:
        pv_pos = ns.position[pv_idx]
        diff = pv_pos.unsqueeze(1) - pv_pos.unsqueeze(0)
        dist = (diff * diff).sum(dim=2).sqrt()
        connected = (dist < 0.3) & (dist > 0)
        if not connected.any():
            for i in range(pv_idx.shape[0]):
                _, nearest = dist[i].topk(min(6, pv_idx.shape[0]), largest=False)
                nearest = nearest[nearest != i][:5]
                connected[i, nearest] = True
                connected[nearest, i] = True
        si, di = torch.where(connected)
        sg = pv_idx[si].to(torch.int32)
        dg = pv_idx[di].to(torch.int32)
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            old = graph.edge_store(EdgeType.ELECTRICAL)
            graph.remove_edges(EdgeType.ELECTRICAL, torch.ones(old.n_edges, dtype=torch.bool, device=device))
        weights = torch.full((sg.shape[0],), 0.5, device=device)
        graph.add_edges(EdgeType.ELECTRICAL, sg, dg, weights=weights)

    PD, STRENGTH = 50, 2.0
    N_CYCLES = 3000
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

                # Theta modulation: slow sinusoidal global excitability
                if theta_on:
                    theta_phase = 2.0 * np.pi * 0.008 * step  # ~8Hz at dt=1ms
                    theta_mod = 0.5 * (1.0 + np.sin(theta_phase))  # 0 to 1
                    inputs.basal *= (0.5 + theta_mod)  # modulate between 50% and 150%

                # Gamma drive to PV
                if gamma_on:
                    gamma_phase = 2.0 * np.pi * 0.05 * step  # 50Hz
                    gamma_drive = 0.5 * (1.0 + np.sin(gamma_phase))
                    pv_drive = torch.zeros(N, device=device)
                    pv_drive[pv_idx] = gamma_drive
                    inputs.basal += pv_drive

                nm.step(ns, inputs, float(step))
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)
                if graph.has_edge_type(EdgeType.MODULATORY):
                    pcw.update(graph.edge_store(EdgeType.MODULATORY), ns)
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

    # Mismatch test
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        graph.increment_step()
    bl = []
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        graph.increment_step()
        bl.append(ns.output[l1_err].mean().item())
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        graph.increment_step()
    vl = []
    for s in range(PD):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        graph.increment_step()
        vl.append(ns.output[l1_err].mean().item())

    baseline = float(np.mean(bl))
    violation = float(np.mean(vl))
    ratio = violation / max(baseline, 1e-8)
    sup = (1 - cycle_errors[-1] / cycle_errors[0]) * 100
    elapsed = time.perf_counter() - t0

    print(f"  [{name}] DONE: sup={sup:.1f}% mm={ratio:.3f}x ({elapsed:.0f}s)", flush=True)
    result_queue.put({
        "name": name, "gamma": gamma_on, "theta": theta_on,
        "suppression": sup, "baseline": baseline, "violation": violation,
        "ratio": ratio, "error_start": cycle_errors[0], "error_end": cycle_errors[-1],
        "time": elapsed,
    })


def main():
    print("=" * 70, flush=True)
    print("  PHASE 2: OSCILLATORY DYNAMICS ON HAND-BUILT PC", flush=True)
    print("=" * 70, flush=True)

    conditions = [
        ("Baseline", False, False),
        ("+ Gamma", True, False),
        ("+ Theta", False, True),
        ("+ Both", True, True),
    ]

    queue = Queue()
    procs = []
    for name, gamma, theta in conditions:
        p = Process(target=run_condition, args=(name, gamma, theta, queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r["name"])

    print(f"\n{'='*70}", flush=True)
    print(f"{'Condition':<15} | {'Sup%':>6} | {'MM':>7} | {'Err_end':>8} | {'Time':>5}", flush=True)
    print("-" * 55, flush=True)
    for r in results:
        star = " **" if r["ratio"] > 1.1 else ""
        print(f"{r['name']:<15} | {r['suppression']:5.1f}% | {r['ratio']:.3f}x | "
              f"{r['error_end']:8.4f} | {r['time']:5.0f}s{star}", flush=True)

    best = max(results, key=lambda r: r["ratio"])
    print(f"\nBest mismatch: {best['name']} at {best['ratio']:.3f}x", flush=True)
    best_sup = max(results, key=lambda r: r["suppression"])
    print(f"Best suppression: {best_sup['name']} at {best_sup['suppression']:.1f}%", flush=True)

    import torch
    torch.save(results, "phase2_oscillation_results.pt")
    print("Saved to phase2_oscillation_results.pt", flush=True)


if __name__ == "__main__":
    main()
