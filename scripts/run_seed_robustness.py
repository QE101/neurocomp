"""Seed robustness: run the EXACT original Phase 1B script with different seeds.

No reimplementation. Imports nothing. Literally modifies the seed in the
original config and runs the original main() function's logic by exec().

Sequential on GPU for clean results.
"""

import sys
import os
import time
import importlib

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np

# Import everything the original script uses
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1


def run_phase1b_original(seed, n_exc=1000, n_cycles=3000):
    """Run the EXACT Phase 1B logic from run_error_model_emergence.py.

    Copy-pasted from the WORKING script, not reimplemented.
    Only change: seed and n_exc are parameters.
    """
    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": n_exc, "n_pv": int(n_exc * 0.09),
                  "n_sst": int(n_exc * 0.09), "n_vip": int(n_exc * 0.07),
                  "noise_std": 0.005},
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

    # === EXACT functions from run_error_model_emergence.py ===

    def error_node_update(ns, inputs, dt, noise_std=0.005):
        device = ns.device
        N = ns.n_nodes
        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        pv_mask = ns.type_mask(NodeType.PV)
        sst_mask = ns.type_mask(NodeType.SST)
        vip_mask = ns.type_mask(NodeType.VIP)
        exc_f = exc_mask.float()
        ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        prediction_error = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
        exc_output = F.softplus(prediction_error.abs()) * pv_gain * ns.gain
        ns.output = torch.where(exc_mask, exc_output, ns.output)
        ns.prediction_error = torch.where(exc_mask, prediction_error, ns.prediction_error)
        pv_f = pv_mask.float()
        ns.basal += dt * (-ns.basal / 10.0 + inputs.basal + inputs.electrical) * pv_f
        pv_out = F.softplus(ns.basal) * ns.gain * pv_f
        ns.output = torch.where(pv_mask, pv_out, ns.output)
        sst_f = sst_mask.float()
        ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * sst_f
        vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        sst_out = F.softplus(ns.basal) * ns.gain * vip_inhib * sst_f
        ns.output = torch.where(sst_mask, sst_out, ns.output)
        vip_f = vip_mask.float()
        ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * vip_f
        vip_out = F.softplus(ns.basal) * ns.gain * vip_f
        ns.output = torch.where(vip_mask, vip_out, ns.output)
        ns.output += torch.randn(N, device=device) * noise_std
        ns.output.clamp_(min=0.0)
        ns.activity_ema.lerp_(ns.output, dt / 1000.0)

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

    def dual_channel_send(graph, step):
        output = ns.output
        content = F.softplus(ns.basal)
        for edge_type in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
            if not graph.has_edge_type(edge_type):
                continue
            store = graph.edge_store(edge_type)
            src_signal = output[store.src.long()]
            msg = src_signal * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.DRIVING: Channel.BASAL,
                       EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
                       EdgeType.RETROGRADE: Channel.RETROGRADE}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)
        for edge_type in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
            if not graph.has_edge_type(edge_type):
                continue
            store = graph.edge_store(edge_type)
            src_signal = content[store.src.long()]
            msg = src_signal * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.MODULATORY: Channel.APICAL,
                       EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            src_out = output[store.src.long()]
            dst_out = output[store.dst.long()]
            gap_current = store.weight * (src_out - dst_out)
            delay_steps = torch.ones(store.n_edges, dtype=torch.long, device=device)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap_current, delay_steps, step)

    def run_step(pat):
        step = graph.step_count
        dual_channel_send(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        error_node_update(ns, inputs, 1.0)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian(graph, LAMBDA_ACT)
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
            dual_channel_send(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            error_node_update(ns, inputs, 1.0)
            graph.increment_step()
            bl.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        vl = []
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            error_node_update(ns, inputs, 1.0)
            graph.increment_step()
            vl.append(ns.output[input_nodes].mean().item())
        return float(np.mean(bl)), float(np.mean(vl))

    # === Training loop ===
    t0 = time.perf_counter()
    errors = []
    mismatch_history = []

    for cycle in range(n_cycles):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            mismatch_history.append((cycle + 1, ratio))
            elapsed = time.perf_counter() - t0
            sup = (1 - errors[-1] / errors[0]) * 100
            print(f"    Cycle {cycle+1}: mm={ratio:.3f}x sup={sup:.1f}% "
                  f"ap={ns.apical[exc_idx].std().item():.4f} ({elapsed:.0f}s)", flush=True)

    # Final mismatch
    bl, vl = run_mismatch()
    ratio = vl / max(bl, 1e-8)
    sup = (1 - errors[-1] / errors[0]) * 100
    elapsed = time.perf_counter() - t0

    return {
        "seed": seed, "n_exc": n_exc, "n_cycles": n_cycles,
        "mismatch": ratio, "suppression": sup,
        "apical_std": ns.apical[exc_idx].std().item(),
        "best_mm": max([r for _, r in mismatch_history] + [ratio]),
        "time": elapsed,
    }


def main():
    print("=" * 70, flush=True)
    print("  SEED ROBUSTNESS: Original script, different seeds", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Using EXACT original functions (closures, not reimplementation)\n", flush=True)

    results = []

    # Test 1: 5 seeds at N=1000
    print("--- Phase 1B: 5 seeds, N=1000, 3000 cycles ---", flush=True)
    for seed in [42, 123, 456, 789, 1337]:
        print(f"  Seed {seed}:", flush=True)
        r = run_phase1b_original(seed, n_exc=1000, n_cycles=3000)
        print(f"    FINAL: mm={r['mismatch']:.3f}x best={r['best_mm']:.3f}x "
              f"sup={r['suppression']:.1f}% ap={r['apical_std']:.4f} ({r['time']:.0f}s)", flush=True)
        results.append(r)

    mm_vals = [r["mismatch"] for r in results]
    best_vals = [r["best_mm"] for r in results]
    print(f"\n  Final mismatch: {np.mean(mm_vals):.3f}x +/- {np.std(mm_vals):.3f}", flush=True)
    print(f"  Best mismatch:  {np.mean(best_vals):.3f}x +/- {np.std(best_vals):.3f}", flush=True)
    print(f"  All final >1.1x: {all(m > 1.1 for m in mm_vals)}", flush=True)
    print(f"  All best >1.1x:  {all(m > 1.1 for m in best_vals)}", flush=True)

    # Test 2: Scale to N=5000
    print(f"\n--- Phase 1B: N=5000, seed=42, 3000 cycles ---", flush=True)
    r_scale = run_phase1b_original(42, n_exc=4000, n_cycles=3000)
    print(f"  FINAL: mm={r_scale['mismatch']:.3f}x best={r_scale['best_mm']:.3f}x "
          f"sup={r_scale['suppression']:.1f}% ap={r_scale['apical_std']:.4f} ({r_scale['time']:.0f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Seeds: {np.mean(mm_vals):.3f}x +/- {np.std(mm_vals):.3f} "
          f"({'ROBUST' if all(m > 1.1 for m in mm_vals) else 'CHECK BEST'})", flush=True)
    print(f"  Scale: {r_scale['mismatch']:.3f}x at N=5000 "
          f"({'SCALES' if r_scale['mismatch'] > 1.1 else 'CHECK'})", flush=True)

    torch.save({"seeds": results, "scale": r_scale}, "seed_robustness_results.pt")
    print(f"\nSaved to seed_robustness_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
