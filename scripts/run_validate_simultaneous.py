"""Validate the ORIGINAL simultaneous Hebbian that produced 0.83 suppression.

The evolution that found suppression at generation 6 used the ORIGINAL
apply_energy_gradient (simultaneous Hebbian: dw = pre × post - decay - penalty).
All subsequent validations accidentally used modified versions of the code.

This script uses the EXACT original learning rule inline, not importing
from energy.py (which has been modified many times since).

Winning genome: lambda_activity=3.1, lambda_prediction=2.0, lambda_mi=2.3
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
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType


def apply_simultaneous_hebbian(graph, genome_dict, dt=1.0):
    """THE ORIGINAL simultaneous Hebbian energy gradient.

    This is the EXACT code that was running when evolution found
    0.83 suppression at generation 6. No temporal state, no compartment
    penalty, no accuracy reward, no precision gating.

    dw = pre × post - weight_decay - activity_penalty
    """
    ns = graph.node_state
    la = genome_dict["lambda_activity"]
    lw = genome_dict["lambda_weight"]
    lp = genome_dict["lambda_prediction"]

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            src_out = ns.output[store.src.long()]
            dst_out = ns.output[store.dst.long()]

            # Hebbian drive: strengthen edges between co-active nodes
            hebbian = src_out * dst_out

            # Energy penalty: decay weights proportional to their magnitude
            weight_decay = lw * 2.0 * store.weight

            # Activity penalty: penalise edges between highly active nodes
            activity_penalty = la * (src_out + dst_out) * store.weight

            # Net update
            dw = dt * 0.001 * (hebbian - weight_decay - activity_penalty)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def main():
    print("=" * 70, flush=True)
    print("  VALIDATE ORIGINAL SIMULTANEOUS HEBBIAN", flush=True)
    print("  (the code that produced 0.83 suppression at gen 6)", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    GENOME = {
        "lambda_activity": 3.1,
        "lambda_weight": 0.0065,
        "lambda_edge": 0.00001,
        "lambda_prediction": 2.0,
        "lambda_reconstruction": 0.27,
        "lambda_mi": 2.3,
    }

    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
        "edges": {"structural": {"enabled": True, "update_interval": 500, "growth_rate": 0.1,
                                   "prune_threshold": 0.005, "edge_cost": 0.00001, "max_degree": 2000}},
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

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD = 50
    STRENGTH = 2.0
    N_CYCLES = 5000
    CHECKPOINT = 250

    print(f"\nGenome: lambda_activity={GENOME['lambda_activity']}", flush=True)
    print(f"Learning rule: SIMULTANEOUS HEBBIAN (dw = pre*post - decay - penalty)", flush=True)
    print(f"Structural plasticity: ENABLED (as in original evolution)", flush=True)
    print(f"Graph: {N} nodes, {graph.n_edges()} edges\n", flush=True)

    def run_step(pat):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_simultaneous_hebbian(graph, GENOME, 1.0)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        if step > 0 and step % 500 == 0:
            from graph_brain.edges.structural import StructuralPlasticity
            sp = StructuralPlasticity(config)
            sp.update(graph)
        graph.increment_step()

    def run_mismatch():
        for s in range(PD):
            run_step(pa)
        baseline = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            baseline.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        violation = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            violation.append(ns.output[input_nodes].mean().item())
        return float(np.mean(baseline)), float(np.mean(violation))

    # Header
    print(f"{'Cyc':>5} | {'Err':>8} | {'Sup%':>5} | {'Ap_std':>7} | {'Ba_std':>7} | "
          f"{'Asym':>6} | {'Edges':>6} | {'MM':>8} | {'Time':>5}", flush=True)
    print("-" * 80, flush=True)

    t0 = time.perf_counter()
    all_errors = []
    all_mismatch = []

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        all_errors.append(err_sum / n)

        if (cycle + 1) % CHECKPOINT == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - all_errors[-1] / all_errors[0]) * 100 if all_errors[0] > 0 else 0
            ap_std = ns.apical[exc_idx].std().item()
            ba_std = ns.basal[exc_idx].std().item()

            # Asymmetry
            up_w, down_w, n_up, n_down = 0.0, 0.0, 0, 0
            for et in (EdgeType.DRIVING, EdgeType.MODULATORY):
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    sz = ns.position[store.src.long(), 2]
                    dz = ns.position[store.dst.long(), 2]
                    up = dz > sz
                    dn = dz < sz
                    if up.any():
                        up_w += float(store.weight[up].sum())
                        n_up += int(up.sum())
                    if dn.any():
                        down_w += float(store.weight[dn].sum())
                        n_down += int(dn.sum())
            um = up_w / max(n_up, 1)
            dm = down_w / max(n_down, 1)
            asym = abs(um - dm) / (max(um, dm) + 1e-6)

            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " **"

            print(f"{cycle+1:5d} | {all_errors[-1]:8.4f} | {sup:4.1f}% | {ap_std:7.4f} | {ba_std:7.4f} | "
                  f"{asym:6.4f} | {graph.n_edges():6d} | {mm_str:>8} | {elapsed:5.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Time: {total:.0f}s ({total/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"Apical std: {ns.apical[exc_idx].std().item():.4f}", flush=True)
    print(f"Best mismatch: {max(all_mismatch):.3f}x", flush=True)
    print(f"Final mismatch: {all_mismatch[-1]:.3f}x", flush=True)
    print(f"Final edges: {graph.n_edges()}", flush=True)

    if max(all_mismatch) > 1.1:
        print(f"MISMATCH DETECTED", flush=True)
    else:
        print(f"NO MISMATCH DETECTED", flush=True)

    torch.save({
        "errors": all_errors, "mismatch_ratios": all_mismatch,
        "genome": GENOME,
    }, "validate_simultaneous_results.pt")
    print(f"\nSaved to validate_simultaneous_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
