"""Delta-Oja with Hard Dead Zone: mathematically complete prevention.

Delta-Oja alone: eps > 0 due to continuous activity -> alternating targets -> oscillation.
Dead zone: dw = 0 when d_pre * d_post < threshold. Creates exact zero for non-pattern edges.

This mimics the biological calcium threshold for AMPA receptor insertion:
plasticity only occurs when pre-post co-activation exceeds a physical minimum.

Two conditions tested:
A) N=1250, 5000 cycles (where all previous methods oscillated)
B) N=50K, 2000 cycles (scale test)

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
BASELINE = math.log(2)  # 0.6931

SMALL_CONFIG = {
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}

LARGE_CONFIG = {
    'nodes': {'n_excitatory': 40000, 'n_pv': 3500, 'n_sst': 3500, 'n_vip': 3000, 'noise_std': 0.005},
    'edges': {
        'connectivity': {
            'driving': {'p_max': 0.3, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                        'target_types': ['EXCITATORY'], 'constant_k': 30},
            'modulatory': {'p_max': 0.2, 'sigma': 0.25, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 70},
            'inhib_perisomatic': {'p_max': 0.5, 'sigma': 0.10, 'source_types': ['PV'],
                                   'target_types': ['EXCITATORY'], 'constant_k': 5},
            'inhib_dendritic': {'p_max': 0.4, 'sigma': 0.12, 'source_types': ['SST'],
                                'target_types': ['EXCITATORY', 'VIP'], 'constant_k': 5},
            'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'],
                          'target_types': ['PV'], 'constant_k': 5},
            'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 10},
            'max_radius': 0.5,
        },
        'structural': {'enabled': False},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
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


# ================================================================
# DELTA-OJA WITH HARD DEAD ZONE
# ================================================================
# Dead zone threshold: d_pre * d_post must exceed this for ANY update.
# At baseline: d_pre ~ 0.01-0.05 (residual above softplus(0))
# So d_pre * d_post ~ 0.0001-0.0025 at baseline
# Pattern-active: d_pre ~ 0.5-3.0, d_post ~ 0.5-3.0, product ~ 0.25-9.0
# Threshold of 0.01 separates baseline from pattern by 4-100x
DEAD_ZONE = 0.01

def apply_delta_oja_deadzone(graph):
    ns_ = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            pre = ns_.output[store.src.long()]
            post = ns_.output[store.dst.long()]

            d_pre = (pre - BASELINE).clamp(min=0.0)
            d_post = (post - BASELINE).clamp(min=0.0)

            # Hard dead zone: co-activation must exceed threshold
            co_act = d_pre * d_post
            alive = (co_act > DEAD_ZONE).float()

            # Delta-Oja only where alive
            dw = 0.001 * d_post * (d_pre - store.weight * d_post) * alive
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


# ================================================================
# RUN ONE CONDITION
# ================================================================
def run_condition(name, config_dict, n_cycles, measure_interval):
    print(f'\n{"="*60}', flush=True)
    print(f'  {name}', flush=True)
    print(f'  Delta-Oja + dead zone ({DEAD_ZONE}), {n_cycles} cycles', flush=True)
    print(f'{"="*60}', flush=True)

    config = GraphBrainConfig.from_dict(config_dict)
    t_build = time.perf_counter()
    graph = NeuromorphicGraph(config)
    graph.initialize()
    print(f'  Built in {time.perf_counter()-t_build:.1f}s ({graph.n_edges():,} edges)', flush=True)

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
    print(f'  Patterns: A={pa.shape[0]}, B={pb.shape[0]} ({pa.shape[0]/exc_idx.shape[0]*100:.1f}% each)', flush=True)

    def run_mismatch():
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

    t0 = time.perf_counter()
    errors, all_mm = [], []

    for cycle in range(n_cycles):
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

                apply_delta_oja_deadzone(graph)

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        if (cycle + 1) % measure_interval == 0:
            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0
            elapsed = time.perf_counter() - t0

            # Dead zone stats
            total_alive, total_edges = 0, 0
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    store = graph.edge_store(et)
                    if store.n_edges == 0: continue
                    d_pre = (ns.output[store.src.long()] - BASELINE).clamp(min=0.0)
                    d_post = (ns.output[store.dst.long()] - BASELINE).clamp(min=0.0)
                    alive = (d_pre * d_post > DEAD_ZONE).sum().item()
                    total_alive += alive
                    total_edges += store.n_edges
            pct_alive = total_alive / max(total_edges, 1) * 100

            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} sup={sup:.1f}% alive={pct_alive:.1f}% ({elapsed:.0f}s)', flush=True)

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
    print(f'  All mm: {[f"{m:.3f}" for m in all_mm]}', flush=True)

    if best > 1.1 and sr < 0.3:
        print(f'  --> STABLE MISMATCH: oscillation PREVENTED', flush=True)
    elif best > 1.1 and damping == "DAMPED":
        print(f'  --> CONVERGING', flush=True)
    elif best < 1.05:
        print(f'  --> No mismatch (dead zone too aggressive?)', flush=True)
    elif damping == "NO DAMPING":
        print(f'  --> Oscillation PERSISTS', flush=True)
    else:
        print(f'  --> {damping}', flush=True)

    return {'name': name, 'best': best, 'final': final, 'damping': damping,
            'fr': fr, 'sr': sr, 'mm': all_mm}


def main():
    print('=' * 60, flush=True)
    print('  DELTA-OJA + HARD DEAD ZONE', flush=True)
    print(f'  Dead zone: d_pre * d_post > {DEAD_ZONE}', flush=True)
    print(f'  Biological analog: calcium threshold for AMPA insertion', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    r1 = run_condition('N=1250 + Theta + Delta-Oja + Dead Zone', SMALL_CONFIG, 5000, 500)
    r2 = run_condition('N=50K + Theta + Delta-Oja + Dead Zone', LARGE_CONFIG, 2000, 250)

    print(f'\n{"="*60}', flush=True)
    print('  COMPARISON', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Undamped N=1250:    best=2.865x osc=2.33/2.32 NO DAMPING', flush=True)
    print(f'  Delta-Oja N=1250:   best=1.303x osc=0.55/? PARTIAL', flush=True)
    print(f'  + Dead Zone N=1250: best={r1["best"]:.3f}x osc={r1["fr"]:.3f}/{r1["sr"]:.3f} {r1["damping"]}', flush=True)
    print(f'  + Dead Zone N=50K:  best={r2["best"]:.3f}x osc={r2["fr"]:.3f}/{r2["sr"]:.3f} {r2["damping"]}', flush=True)

    torch.save({'small': r1, 'large': r2}, 'delta_oja_deadzone_results.pt')
    print(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
