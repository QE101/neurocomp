"""Option 3: Sweep lambda_activity to find the longest transient window.

Test lambda_activity = [0.5, 1.0, 1.5, 2.0, 2.5, 3.1, 4.0, 5.0]
Each runs 2000 cycles with mismatch test every 100 cycles.
Find where the transient window is longest.

4 parallel processes on GPU.
"""

import sys
import os
import time
from multiprocessing import Process, Queue

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")


def run_one_lambda(lam, seed, n_cycles, result_queue):
    import torch
    import numpy as np
    from graph_brain.config import GraphBrainConfig
    from graph_brain.core.graph import NeuromorphicGraph
    from graph_brain.core.message_passing import TypedMessagePasser
    from graph_brain.edges.homeostatic import HomeostaticScaling
    from graph_brain.edges.short_term import ShortTermPlasticity
    from graph_brain.nodes.intrinsic import IntrinsicPlasticity
    from graph_brain.nodes.model import TwoCompartmentModel
    from graph_brain.types import EdgeType, NodeType

    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": seed},
        "hierarchy": {"enabled": False},
    })

    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0

    def apply_hebbian(graph, la):
        ns = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0:
                    continue
                src_out = ns.output[store.src.long()]
                dst_out = ns.output[store.dst.long()]
                hebbian = src_out * dst_out
                weight_decay = 0.0065 * 2.0 * store.weight
                activity_penalty = la * (src_out + dst_out) * store.weight
                dw = 0.001 * (hebbian - weight_decay - activity_penalty)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

    def run_step(pat):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian(graph, lam)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()

    def run_mismatch():
        for s in range(PD):
            run_step(pa)
        bl = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            bl.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        vl = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            vl.append(ns.output[input_nodes].mean().item())
        return float(np.mean(bl)), float(np.mean(vl))

    t0 = time.perf_counter()
    mismatch_history = []

    for cycle in range(n_cycles):
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(pat)

        if (cycle + 1) % 100 == 0:
            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            mismatch_history.append((cycle + 1, ratio))

    elapsed = time.perf_counter() - t0

    # Find best mismatch and window duration
    best_ratio = max(r for _, r in mismatch_history) if mismatch_history else 0
    best_cycle = [c for c, r in mismatch_history if r == best_ratio][0] if mismatch_history else 0
    above_1 = [(c, r) for c, r in mismatch_history if r > 1.0]
    window_len = len(above_1) * 100 if above_1 else 0

    print(f"  [lam={lam:.1f}] best={best_ratio:.3f}x @ cycle {best_cycle}, "
          f"window>{1.0}={window_len} cycles ({elapsed:.0f}s)", flush=True)

    result_queue.put({
        "lambda": lam, "best_ratio": best_ratio, "best_cycle": best_cycle,
        "window_above_1": window_len, "history": mismatch_history,
    })


def main():
    print("=" * 70, flush=True)
    print("  LAMBDA SWEEP: Finding the longest transient window", flush=True)
    print("=" * 70, flush=True)

    lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.1, 4.0, 5.0]
    N_CYCLES = 2000
    N_PARALLEL = 4

    queue = Queue()
    results = []

    for batch_start in range(0, len(lambdas), N_PARALLEL):
        batch = lambdas[batch_start:batch_start + N_PARALLEL]
        procs = []
        for lam in batch:
            p = Process(target=run_one_lambda, args=(lam, 42, N_CYCLES, queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r["lambda"])

    print(f"\n{'='*70}", flush=True)
    print(f"{'Lambda':>8} | {'Best MM':>8} | {'@ Cycle':>8} | {'Window>1.0':>10}", flush=True)
    print("-" * 45, flush=True)
    for r in results:
        star = " **" if r["best_ratio"] > 1.1 else ""
        print(f"{r['lambda']:8.1f} | {r['best_ratio']:7.3f}x | {r['best_cycle']:8d} | "
              f"{r['window_above_1']:8d} cyc{star}", flush=True)

    best = max(results, key=lambda r: r["best_ratio"])
    print(f"\nBest: lambda={best['lambda']} with {best['best_ratio']:.3f}x at cycle {best['best_cycle']}", flush=True)

    import torch
    torch.save(results, "lambda_sweep_results.pt")
    print("Saved to lambda_sweep_results.pt", flush=True)


if __name__ == "__main__":
    main()
