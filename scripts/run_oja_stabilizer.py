"""Oja Stabilizer: replace activity penalty with order-matched stabilization.

Original: dw = lr * (pre*post - 0.013*w - 3.1*(pre+post)*w)
                     O(a^2)     O(1)       O(a)  <-- order mismatch

New:      dw = lr * (pre*post - 0.013*w - post^2*w)
                     O(a^2)     O(1)      O(a^2) <-- order matched

Only change: replace la*(src+dst)*weight with dst^2*weight.
- Same order as Hebbian (both O(a^2))
- Activity-proportional (inactive edges get 0.48w drain vs active getting 4-9w)
- Self-normalizing (Oja's convergence guarantee)
- No la parameter to tune

Everything else identical to the validated Hebbian.
Diagnostic monitoring every 10 cycles to verify the fix.
"""

import sys, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
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
N_CYCLES = 2000

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
# OJA STABILIZER: replace activity penalty with post^2 * weight
# ================================================================
def apply_hebbian_oja_stabilizer(graph):
    """Original Hebbian with Oja stabilizer replacing activity penalty.

    Old: dw = lr * (pre*post - 0.013*w - 3.1*(pre+post)*w)
    New: dw = lr * (pre*post - 0.013*w - post^2*w)
    """
    ns_ = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            hebbian = src * dst                    # O(a^2) potentiation
            weight_decay = 0.0065 * 2.0 * store.weight  # O(1) slow decay
            oja_stabilizer = dst * dst * store.weight    # O(a^2) stabilization

            dw = 0.001 * (hebbian - weight_decay - oja_stabilizer)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  OJA STABILIZER: Activity Penalty Replaced', flush=True)
    print('  dw = lr * (pre*post - 0.013*w - post^2*w)', flush=True)
    print('  O(a^2) stabilization matches O(a^2) potentiation.', flush=True)
    print('  Theta + A-B, N=1250, 2000 cycles.', flush=True)
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
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    # Edge masks for pattern-specific tracking
    pa_set = set(pa.cpu().tolist())
    pb_set = set(pb.cpu().tolist())
    drv_from_a_mask = None
    drv_from_b_mask = None
    if graph.has_edge_type(EdgeType.DRIVING):
        store = graph.edge_store(EdgeType.DRIVING)
        src_np = store.src.cpu().numpy()
        drv_from_a_mask = torch.tensor([s in pa_set for s in src_np], device=device)
        drv_from_b_mask = torch.tensor([s in pb_set for s in src_np], device=device)

    print(f'  Patterns: A={pa.shape[0]}, B={pb.shape[0]}', flush=True)
    print(f'  Total edges: {graph.n_edges():,}', flush=True)

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

                apply_hebbian_oja_stabilizer(graph)

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        # Diagnostic every 100 cycles, mismatch every 200
        if (cycle + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0

            # Key diagnostics
            drv_store = graph.edge_store(EdgeType.DRIVING)
            drv_mean = drv_store.weight.mean().item()
            mod_mean = graph.edge_store(EdgeType.MODULATORY).weight.mean().item() if graph.has_edge_type(EdgeType.MODULATORY) else 0
            drv_a = drv_store.weight[drv_from_a_mask].mean().item() if drv_from_a_mask is not None and drv_from_a_mask.sum() > 0 else 0
            drv_b = drv_store.weight[drv_from_b_mask].mean().item() if drv_from_b_mask is not None and drv_from_b_mask.sum() > 0 else 0

            out_a = ns.output[pa].mean().item()
            out_b = ns.output[pb].mean().item()
            ba_a = ns.basal[pa].mean().item()
            ap_a = ns.apical[pa].mean().item()
            ba_b = ns.basal[pb].mean().item()
            ap_b = ns.apical[pb].mean().item()

            print(f'  Cyc {cycle+1:5d}: '
                  f'drv={drv_mean:.4f} mod={mod_mean:.4f} | '
                  f'drvA={drv_a:.4f} drvB={drv_b:.4f} (diff={drv_b-drv_a:.4f}) | '
                  f'bas_A={ba_a:.3f} ap_A={ap_a:.3f} bas_B={ba_b:.3f} ap_B={ap_b:.3f} | '
                  f'sup={sup:.1f}% ({elapsed:.0f}s)', flush=True)

        if (cycle + 1) % 200 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  *** MM Cyc {cycle+1}: {mm} (bl={bl:.4f} vl={vl:.4f}) ***', flush=True)

    # Analysis
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
    print(f'  All mm: {[f"{m:.3f}" for m in all_mm]}', flush=True)

    # Flatness
    if len(all_mm) >= 3:
        diffs = [abs(all_mm[i+1] - all_mm[i]) for i in range(len(all_mm)-1)]
        print(f'  Step diffs: {[f"{d:.3f}" for d in diffs]}', flush=True)

    torch.save({'all_mm': all_mm, 'errors': errors, 'damping': damping,
                'best': best, 'final': final, 'fr': fr, 'sr': sr},
               'oja_stabilizer_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
