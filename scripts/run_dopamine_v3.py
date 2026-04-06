"""Dopamine v3: Two variants in parallel to break the chicken-and-egg.

A: Lower activity threshold (0.1 instead of 0.5)
B: Bootstrap kick — start with dopamine at max for first 500 cycles

Both use the fixed trigger (requires active prediction, not just low error).
Unbuffered output, 5000 cycles each.
"""

import sys
import os
import time
from multiprocessing import Process, Queue

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")


def run_variant(name, activity_threshold, bootstrap_cycles, result_queue):
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

    LAMBDA_BASE = 3.1

    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": 42},
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

    # Dopamine state (inline, not importing to avoid stale code)
    da_level = torch.tensor(0.0, device=device)
    da_burst = 0.8
    da_decay = 0.99
    total_bursts = 0

    if bootstrap_cycles > 0:
        da_level = torch.tensor(0.9, device=device)  # start hot

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0
    N_CYCLES = 5000

    def apply_hebbian(graph, effective_la):
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
                activity_penalty = effective_la * (src_out + dst_out) * store.weight
                dw = 0.001 * (hebbian - weight_decay - activity_penalty)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

    def run_step(pat, cycle):
        nonlocal da_level
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        eff_la = LAMBDA_BASE * (1.0 - da_level.item())
        apply_hebbian(graph, eff_la)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        da_level *= da_decay
        # Bootstrap: keep dopamine high for first N cycles
        if bootstrap_cycles > 0 and cycle < bootstrap_cycles:
            da_level = da_level.clamp(min=0.5)
        graph.increment_step()

    def trigger_dopamine(prev_output):
        nonlocal da_level, total_bursts
        if prev_output is None:
            return 0.0
        prev_mean = prev_output.mean().item()
        curr_mean = ns.output[input_nodes.long()].mean().item()

        was_active = prev_mean > activity_threshold
        output_dropped = curr_mean < prev_mean * 0.8

        if was_active and output_dropped:
            da_level = (da_level + da_burst).clamp(0.0, 1.0)
            total_bursts += 1
            return da_burst
        elif was_active and curr_mean > prev_mean * 1.2:
            da_level = (da_level - 0.3).clamp(0.0, 1.0)
            return -0.3
        elif not was_active:
            da_level = (da_level - 0.05).clamp(0.0, 1.0)
            return -0.05
        return 0.0

    def run_mismatch():
        for s in range(PD):
            run_step(pa, 9999)
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
            run_step(pa, 9999)
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
    all_errors = []
    all_mismatch = []
    prev_pat_output = None

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat_idx, pat in enumerate([pa, pb]):
            current_snap = ns.output[input_nodes.long()].detach().clone()
            trigger_dopamine(prev_pat_output)
            for s in range(PD):
                run_step(pat, cycle)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
            prev_pat_output = ns.output[input_nodes.long()].detach().clone()
        all_errors.append(err_sum / n)

        if (cycle + 1) % 250 == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - all_errors[-1] / all_errors[0]) * 100 if all_errors[0] > 0 else 0
            ap_std = ns.apical[exc_idx].std().item()
            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " **"
            print(f"  [{name}] Cyc {cycle+1:5d}: err={all_errors[-1]:.4f} sup={sup:.1f}% "
                  f"ap={ap_std:.4f} da={da_level.item():.3f} bursts={total_bursts} "
                  f"mm={mm_str} ({elapsed:.0f}s)", flush=True)

    total = time.perf_counter() - t0
    best_mm = max(all_mismatch) if all_mismatch else 0
    print(f"  [{name}] DONE: sup={(1-all_errors[-1]/all_errors[0])*100:.1f}% "
          f"best_mm={best_mm:.3f}x bursts={total_bursts} ({total:.0f}s)", flush=True)

    result_queue.put({
        "name": name, "best_mm": best_mm, "final_mm": all_mismatch[-1] if all_mismatch else 0,
        "suppression": (1 - all_errors[-1] / all_errors[0]) * 100,
        "total_bursts": total_bursts, "errors": all_errors, "mismatch": all_mismatch,
    })


def main():
    print("=" * 70, flush=True)
    print("  DOPAMINE V3: TWO VARIANTS — THRESHOLD vs BOOTSTRAP", flush=True)
    print("=" * 70, flush=True)

    queue = Queue()
    procs = []

    # Variant A: low threshold (0.1 instead of 0.5)
    p = Process(target=run_variant, args=("LowThresh", 0.1, 0, queue))
    p.start()
    procs.append(p)

    # Variant B: bootstrap (dopamine starts hot for 500 cycles)
    p = Process(target=run_variant, args=("Bootstrap", 0.5, 500, queue))
    p.start()
    procs.append(p)

    for p in procs:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())

    print(f"\n{'='*70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    for r in results:
        star = " **" if r["best_mm"] > 1.1 else ""
        print(f"  {r['name']}: sup={r['suppression']:.1f}% best_mm={r['best_mm']:.3f}x "
              f"final_mm={r['final_mm']:.3f}x bursts={r['total_bursts']}{star}", flush=True)

    import torch
    torch.save(results, "dopamine_v3_results.pt")
    print("Saved to dopamine_v3_results.pt", flush=True)


if __name__ == "__main__":
    main()
