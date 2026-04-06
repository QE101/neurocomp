"""Phase 1B + Dopamine: Global reward signal as PC stabiliser.

The system's motivational drive. Without dopamine, the energy-optimal
strategy is silence. With dopamine, correct predictions are rewarded
with a temporary excitability boost that counteracts sparsity pressure.

Protocol:
  - Self-organised graph (no hand-built hierarchy)
  - Simultaneous Hebbian (the code that produces transient PC)
  - Dopamine burst on successful pattern prediction at each transition
  - During burst: reduced lambda_activity, boosted learning rate
  - Between bursts: full sparsity pressure

5000 cycles, comprehensive metrics, unbuffered output.
"""

import sys
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.dopamine import DopamineSystem
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType

LAMBDA_BASE = 3.1


def apply_hebbian_with_dopamine(graph, dopamine, dt=1.0):
    """Simultaneous Hebbian with dopamine-modulated activity penalty."""
    ns = graph.node_state
    effective_la = dopamine.effective_lambda(LAMBDA_BASE)

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

            dw = dt * 0.001 * (hebbian - weight_decay - activity_penalty)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def main():
    print("=" * 70, flush=True)
    print("  PHASE 1B + DOPAMINE: MOTIVATED EMERGENCE", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

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
    dopamine = DopamineSystem(N, device, burst_size=0.8, decay_rate=0.99, learning_boost=3.0)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0
    N_CYCLES = 5000
    CHECKPOINT = 250

    print(f"Graph: {N} nodes, no hierarchy", flush=True)
    print(f"Lambda_base: {LAMBDA_BASE} (reduced during dopamine burst)", flush=True)
    print(f"Dopamine: burst={dopamine.burst_size}, decay={dopamine.decay_rate}, "
          f"lr_boost={dopamine.learning_boost}", flush=True)
    print(f"Half-life: ~{int(-1/np.log(dopamine.decay_rate))} steps\n", flush=True)

    def run_step(pat):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian_with_dopamine(graph, dopamine, 1.0)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        dopamine.step()
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

    print(f"{'Cyc':>5} | {'Err':>8} | {'Sup%':>5} | {'Ap_std':>7} | {'DA_lvl':>7} | "
          f"{'Eff_lam':>7} | {'Bursts':>6} | {'MM':>8} | {'Time':>5}", flush=True)
    print("-" * 80, flush=True)

    t0 = time.perf_counter()
    all_errors = []
    all_mismatch = []
    total_bursts = 0

    prev_pat_output = None

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        cycle_bursts = 0

        for pat_idx, pat in enumerate([pa, pb]):
            # Snapshot output before pattern transition
            current_pat_output = ns.output[input_nodes.long()].detach().clone()

            # Trigger dopamine at pattern transition
            delta = dopamine.on_pattern_transition(graph, input_nodes, prev_pat_output)
            if delta > 0:
                cycle_bursts += 1

            for s in range(PD):
                run_step(pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1

            # Store this pattern's final output for next transition
            prev_pat_output = ns.output[input_nodes.long()].detach().clone()

        all_errors.append(err_sum / n)
        total_bursts += cycle_bursts

        if (cycle + 1) % CHECKPOINT == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - all_errors[-1] / all_errors[0]) * 100 if all_errors[0] > 0 else 0
            ap_std = ns.apical[exc_idx].std().item()
            da_lvl = dopamine.level.item()
            eff_lam = dopamine.effective_lambda(LAMBDA_BASE)

            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " **"

            print(f"{cycle+1:5d} | {all_errors[-1]:8.4f} | {sup:4.1f}% | {ap_std:7.4f} | "
                  f"{da_lvl:7.4f} | {eff_lam:7.3f} | {total_bursts:6d} | {mm_str:>8} | "
                  f"{elapsed:5.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Time: {total:.0f}s ({total/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"Total dopamine bursts: {total_bursts}", flush=True)
    print(f"Best mismatch: {max(all_mismatch):.3f}x", flush=True)
    print(f"Final mismatch: {all_mismatch[-1]:.3f}x", flush=True)

    if max(all_mismatch) > 1.1:
        best_idx = all_mismatch.index(max(all_mismatch))
        print(f"MISMATCH DETECTED at cycle {(best_idx+1)*CHECKPOINT}", flush=True)
    else:
        print("NO MISMATCH DETECTED", flush=True)

    torch.save({
        "errors": all_errors, "mismatch_ratios": all_mismatch,
        "total_bursts": total_bursts,
    }, "dopamine_emergence_results.pt")
    print(f"\nSaved to dopamine_emergence_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
