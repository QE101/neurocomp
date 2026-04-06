"""Batch Delta-Oja: symmetric learning prevents oscillation by construction.

Root cause of oscillation: order dependence. A-then-B != B-then-A at the weight level.
Each pattern's step-by-step updates partially undo the other's.

Fix: accumulate eligibility traces during the full A-B cycle, apply ONCE at end.
The averaged update is symmetric w.r.t. pattern ordering -> single fixed point -> no oscillation.

dw_trace accumulated per step: d_post * (d_pre - w * d_post)
Applied once per cycle: w += lr * trace / n_steps

Copy-pasted core functions from run_stability_battery.py (never reimplement).
"""

import sys, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import math
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.types import EdgeType, NodeType

STRENGTH = 2.0
PD = 50
N_CYCLES = 5000
BASELINE = math.log(2)

SMALL_CONFIG = {
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}


# ================================================================
# EXACT copy-paste from run_stability_battery.py (DO NOT MODIFY)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()
    ns.basal += 1.0 * (-ns.basal / 10.0 + inputs.basal * theta_mod) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += 1.0 * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(exc_mask, F.softplus(pred_err.abs()) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
        out = F.softplus(ns.basal) * ns.gain * f
        if inh_type == NodeType.SST:
            out = out * torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        ns.output = torch.where(mask, out, ns.output)
    ns.output += torch.randn(N, device=device) * 0.005
    ns.output.clamp_(min=0.0)
    ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)


def dual_channel_send(ns, graph, mp, device):
    step = graph.step_count
    output = ns.output
    content = F.softplus(ns.basal)
    for et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = output[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
              EdgeType.RETROGRADE: Channel.RETROGRADE}[et]
        mp.delay_buffer.write(ch, store.dst, msg, d, step)
    for et in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = content[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[et]
        mp.delay_buffer.write(ch, store.dst, msg, d, step)
    if graph.has_edge_type(EdgeType.ELECTRICAL):
        store = graph.edge_store(EdgeType.ELECTRICAL)
        gap = store.weight * (output[store.src.long()] - output[store.dst.long()])
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap,
                              torch.ones(store.n_edges, dtype=torch.long, device=device), step)


def run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes):
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
    bl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        bl.append(ns.output[input_nodes].mean().item())
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
    vl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        vl.append(ns.output[input_nodes].mean().item())
    return float(np.mean(bl)), float(np.mean(vl))


# ================================================================
# BATCH DELTA-OJA: accumulate traces, apply once per cycle
# ================================================================
def create_traces(graph):
    """Create zero-initialized eligibility traces for each edge type."""
    traces = {}
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges > 0:
                traces[et] = torch.zeros_like(store.weight)
    return traces


def accumulate_trace(graph, traces):
    """Accumulate delta-Oja update into eligibility trace (don't apply yet)."""
    ns_ = graph.node_state
    for et, trace in traces.items():
        store = graph.edge_store(et)
        pre = ns_.output[store.src.long()]
        post = ns_.output[store.dst.long()]
        d_pre = (pre - BASELINE).clamp(min=0.0)
        d_post = (post - BASELINE).clamp(min=0.0)
        # Accumulate delta-Oja into trace
        trace += d_post * (d_pre - store.weight * d_post)


def apply_traces(graph, traces, n_steps):
    """Apply averaged eligibility traces to weights. One update per cycle."""
    for et, trace in traces.items():
        store = graph.edge_store(et)
        # Average over all steps in the cycle, then apply with learning rate
        store.weight += 0.001 * trace / n_steps
        store.weight.clamp_(0.0, 1.0)
        # Reset trace for next cycle
        trace.zero_()


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  BATCH DELTA-OJA: Symmetric Learning', flush=True)
    print('  Accumulate traces over full A-B cycle, apply once.', flush=True)
    print('  Order-independent -> single fixed point -> no oscillation.', flush=True)
    print('  Theta + A-B, N=1250, 5000 cycles.', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(SMALL_CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    traces = create_traces(graph)
    steps_per_cycle = 2 * PD  # A(50) + B(50) = 100 steps

    print(f'  Patterns: A={pa.shape[0]}, B={pb.shape[0]}', flush=True)
    print(f'  Steps per cycle: {steps_per_cycle}', flush=True)
    print(f'  Traces: {len(traces)} edge types', flush=True)

    t0 = time.perf_counter()
    errors, all_mm = [], []

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)

                # Accumulate trace (DON'T apply weights yet)
                accumulate_trace(graph, traces)

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1

        # END OF CYCLE: apply averaged trace
        apply_traces(graph, traces, steps_per_cycle)
        errors.append(err_sum / n)

        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0
            elapsed = time.perf_counter() - t0

            # Weight stats
            w_stats = []
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    store = graph.edge_store(et)
                    if store.n_edges > 0:
                        w_stats.append(store.weight.mean().item())
            w_mean = np.mean(w_stats) if w_stats else 0

            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} sup={sup:.1f}% w={w_mean:.4f} ({elapsed:.0f}s)', flush=True)

    # Oscillation analysis
    if len(all_mm) >= 4:
        half = len(all_mm) // 2
        fr = max(all_mm[:half]) - min(all_mm[:half])
        sr = max(all_mm[half:]) - min(all_mm[half:])
        if sr < fr * 0.5:
            damping = "DAMPED"
        elif sr < fr * 0.9:
            damping = "PARTIAL"
        else:
            damping = "NO DAMPING"
    else:
        fr = sr = 0
        damping = "N/A"

    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'\n  RESULT: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    print(f'  All mismatch: {[f"{m:.3f}" for m in all_mm]}', flush=True)

    # Check for flat trajectory (the real test)
    if len(all_mm) >= 4:
        diffs = [abs(all_mm[i+1] - all_mm[i]) for i in range(len(all_mm)-1)]
        max_diff = max(diffs)
        avg_diff = np.mean(diffs)
        print(f'  Step-to-step: max_diff={max_diff:.3f} avg_diff={avg_diff:.3f}', flush=True)
        if max_diff < 0.3 and best > 1.1:
            print(f'\n  VERDICT: FLAT TRAJECTORY -- oscillation PREVENTED', flush=True)
        elif best > 1.1 and sr < 0.3:
            print(f'\n  VERDICT: STABLE MISMATCH', flush=True)
        elif best < 1.05:
            print(f'\n  VERDICT: No mismatch (batching too conservative)', flush=True)
        else:
            print(f'\n  VERDICT: Still oscillating (max_diff={max_diff:.3f})', flush=True)
    else:
        print(f'\n  VERDICT: Insufficient data', flush=True)

    torch.save({'all_mm': all_mm, 'errors': errors, 'damping': damping,
                'best': best, 'final': final, 'fr': fr, 'sr': sr},
               'batch_delta_oja_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
