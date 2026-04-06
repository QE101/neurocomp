"""Stability Battery: 5 biological mechanisms tested sequentially.

All tests: theta + A-B at N=1250, 5000 cycles, universal error model.
Sequential on GPU for clean results. Estimated ~10 hours total.

1. Timing-selective Hebbian (only coincident edges update)
2. BCM sliding threshold (active nodes resist further potentiation)
3. Phase-gated learning (only learn during specific theta phases)
4. Extreme sparsity (PV boost for 1-2% activation, at N=5000)
5. Sleep consolidation (wake/sleep cycling with replay)

Each compared against undamped baseline oscillation.
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

LAMBDA_ACT = 3.1
STRENGTH = 2.0
PD = 50
N_CYCLES = 5000

SMALL_CONFIG = {
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}

# Larger config for extreme sparsity test
MEDIUM_CONFIG = {
    'nodes': {'n_excitatory': 4000, 'n_pv': 350, 'n_sst': 350, 'n_vip': 300, 'noise_std': 0.005},
    'edges': {
        'connectivity': {
            'driving': {'p_max': 0.3, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                        'target_types': ['EXCITATORY'], 'constant_k': 30},
            'modulatory': {'p_max': 0.2, 'sigma': 0.25, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 70},
            'inhib_perisomatic': {'p_max': 0.5, 'sigma': 0.10, 'source_types': ['PV'],
                                   'target_types': ['EXCITATORY'], 'constant_k': 10},
            'inhib_dendritic': {'p_max': 0.4, 'sigma': 0.12, 'source_types': ['SST'],
                                'target_types': ['EXCITATORY', 'VIP'], 'constant_k': 10},
            'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'],
                          'target_types': ['PV'], 'constant_k': 5},
            'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 10},
            'max_radius': 0.5,
        },
        'structural': {'enabled': False},
    },
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}


def setup_small():
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
    return graph, ns, N, device, mp, stp, hom, ip, theta, exc_idx, input_nodes, pa, pb


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


def analyze_oscillation(all_mm):
    if len(all_mm) < 4:
        return 0, 0, "N/A"
    half = len(all_mm) // 2
    fr = max(all_mm[:half]) - min(all_mm[:half])
    sr = max(all_mm[half:]) - min(all_mm[half:])
    if sr < fr * 0.5:
        return fr, sr, "DAMPED"
    elif sr < fr * 0.9:
        return fr, sr, "PARTIAL"
    else:
        return fr, sr, "NO DAMPING"


# ================================================================
# MECHANISM 1: Timing-Selective Hebbian
# ================================================================
def test_timing_selective():
    print(f'\n{"="*60}', flush=True)
    print('  MECHANISM 1: Timing-Selective Hebbian', flush=True)
    print('  Only update edges with coincident pre/post activity', flush=True)
    print(f'{"="*60}', flush=True)

    graph, ns, N, device, mp, stp, hom, ip, theta, exc_idx, input_nodes, pa, pb = setup_small()
    COINCIDENCE_THRESHOLD = 0.3  # both pre_trace and post must exceed this

    def apply_hebbian_timing(graph, la):
        ns_ = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns_.output[store.src.long()]
                dst = ns_.output[store.dst.long()]

                # Timing gate: only update if pre was recently active AND post is active now
                # pre_trace tracks recent pre activity (decays over ~20 steps)
                coincidence = (store.pre_trace > COINCIDENCE_THRESHOLD) & (dst > COINCIDENCE_THRESHOLD)
                gate = coincidence.float()

                hebbian = src * dst
                weight_decay = 0.0065 * 2.0 * store.weight
                activity_penalty = la * (src + dst) * store.weight
                dw = 0.001 * (hebbian - weight_decay - activity_penalty) * gate
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

                # Update pre_trace (fast decay)
                store.pre_trace *= 0.95
                store.pre_trace += src * 0.05

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
                apply_hebbian_timing(graph, LAMBDA_ACT)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)
        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} ({elapsed:.0f}s)', flush=True)

    fr, sr, damping = analyze_oscillation(all_mm)
    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': 'Timing-Selective', 'best': best, 'final': final, 'damping': damping, 'fr': fr, 'sr': sr}


# ================================================================
# MECHANISM 2: BCM Sliding Threshold
# ================================================================
def test_bcm():
    print(f'\n{"="*60}', flush=True)
    print('  MECHANISM 2: BCM Sliding Threshold', flush=True)
    print('  Active nodes raise their LTP threshold', flush=True)
    print(f'{"="*60}', flush=True)

    graph, ns, N, device, mp, stp, hom, ip, theta, exc_idx, input_nodes, pa, pb = setup_small()

    def apply_hebbian_bcm(graph, la):
        ns_ = graph.node_state
        # BCM threshold: quadratic in recent activity
        bcm_threshold = ns_.activity_ema * ns_.activity_ema * 10.0  # scale factor

        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns_.output[store.src.long()]
                dst = ns_.output[store.dst.long()]
                dst_thresh = bcm_threshold[store.dst.long()]

                # BCM: LTP when post > threshold, LTD when post < threshold
                bcm_factor = (dst - dst_thresh)
                hebbian = src * bcm_factor  # can be negative (LTD)

                weight_decay = 0.0065 * 2.0 * store.weight
                activity_penalty = la * (src + dst) * store.weight
                dw = 0.001 * (hebbian - weight_decay - activity_penalty)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

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
                apply_hebbian_bcm(graph, LAMBDA_ACT)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)
        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} ({elapsed:.0f}s)', flush=True)

    fr, sr, damping = analyze_oscillation(all_mm)
    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': 'BCM', 'best': best, 'final': final, 'damping': damping, 'fr': fr, 'sr': sr}


# ================================================================
# MECHANISM 3: Phase-Gated Learning
# ================================================================
def test_phase_gated():
    print(f'\n{"="*60}', flush=True)
    print('  MECHANISM 3: Phase-Gated Learning', flush=True)
    print('  Only learn during rising theta phase', flush=True)
    print(f'{"="*60}', flush=True)

    graph, ns, N, device, mp, stp, hom, ip, theta, exc_idx, input_nodes, pa, pb = setup_small()

    def apply_hebbian_phase_gated(graph, la, step):
        phase = theta.get_phase(step)
        # Only learn during rising phase (pi/4 to 3pi/4)
        in_window = (phase > math.pi * 0.25) and (phase < math.pi * 0.75)
        if not in_window:
            return

        ns_ = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns_.output[store.src.long()]
                dst = ns_.output[store.dst.long()]
                hebbian = src * dst
                weight_decay = 0.0065 * 2.0 * store.weight
                activity_penalty = la * (src + dst) * store.weight
                dw = 0.001 * (hebbian - weight_decay - activity_penalty)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

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
                apply_hebbian_phase_gated(graph, LAMBDA_ACT, step)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)
        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} ({elapsed:.0f}s)', flush=True)

    fr, sr, damping = analyze_oscillation(all_mm)
    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': 'Phase-Gated', 'best': best, 'final': final, 'damping': damping, 'fr': fr, 'sr': sr}


# ================================================================
# MECHANISM 4: Extreme Sparsity (at N=5000)
# ================================================================
def test_extreme_sparsity():
    print(f'\n{"="*60}', flush=True)
    print('  MECHANISM 4: Extreme Sparsity (N=5000, PV boost)', flush=True)
    print('  Target 1-2% activation via strong PV competition', flush=True)
    print(f'{"="*60}', flush=True)

    config = GraphBrainConfig.from_dict(MEDIUM_CONFIG)
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

    # Boost PV inhibition 10x for extreme sparsity
    if graph.has_edge_type(EdgeType.INHIB_PERISOMATIC):
        store = graph.edge_store(EdgeType.INHIB_PERISOMATIC)
        store.weight *= 10.0
        store.weight.clamp_(0.0, 1.0)
        print(f'  PV boost 10x, mean={store.weight.mean():.3f}', flush=True)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    def apply_hebbian_std(graph, la):
        ns_ = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns_.output[store.src.long()]
                dst = ns_.output[store.dst.long()]
                dw = 0.001 * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

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
                apply_hebbian_std(graph, LAMBDA_ACT)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)
        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            # Measure sparsity
            active = (ns.output[exc_idx] > 0.5).float().mean().item()
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} active={active:.1%} ({elapsed:.0f}s)', flush=True)

    fr, sr, damping = analyze_oscillation(all_mm)
    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': 'Extreme Sparsity', 'best': best, 'final': final, 'damping': damping, 'fr': fr, 'sr': sr}


# ================================================================
# MECHANISM 5: Sleep Consolidation
# ================================================================
def test_sleep_consolidation():
    print(f'\n{"="*60}', flush=True)
    print('  MECHANISM 5: Sleep Consolidation', flush=True)
    print('  Wake/sleep cycling: 100 wake cycles then 50 sleep cycles', flush=True)
    print(f'{"="*60}', flush=True)

    graph, ns, N, device, mp, stp, hom, ip, theta, exc_idx, input_nodes, pa, pb = setup_small()
    WAKE_CYCLES = 100
    SLEEP_CYCLES = 50
    SLEEP_LR = 0.0002  # 5x slower than wake (0.001)
    SLEEP_STRENGTH = 1.0  # half strength replay

    def apply_hebbian_std(graph, la, lr=0.001):
        ns_ = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns_.output[store.src.long()]
                dst = ns_.output[store.dst.long()]
                dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

    t0 = time.perf_counter()
    errors, all_mm = [], []
    cycle_count = 0

    while cycle_count < N_CYCLES:
        # WAKE phase: standard learning with theta
        for wake_cyc in range(min(WAKE_CYCLES, N_CYCLES - cycle_count)):
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
                    apply_hebbian_std(graph, LAMBDA_ACT, lr=0.001)
                    if step % 100 == 0:
                        for et in EdgeType:
                            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                                hom.update(graph.edge_store(et), ns, 1.0)
                        ip.update(ns)
                    graph.increment_step()
                    err_sum += ns.output[input_nodes].mean().item()
                    n += 1
            errors.append(err_sum / n)
            cycle_count += 1

            if cycle_count % 500 == 0:
                bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
                ratio = vl / max(bl, 1e-8)
                all_mm.append(ratio)
                elapsed = time.perf_counter() - t0
                mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
                print(f'  Cyc {cycle_count:5d} (wake): mm={mm} ({elapsed:.0f}s)', flush=True)

        # SLEEP phase: slow replay, no theta, interleaved patterns, reduced LR
        if cycle_count < N_CYCLES:
            for sleep_cyc in range(SLEEP_CYCLES):
                # Randomly replay A or B at reduced strength
                pat = pa if np.random.random() < 0.5 else pb
                for s in range(PD):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pat.long()] += SLEEP_STRENGTH  # weaker replay
                    error_node_update(ns, inputs, theta_mod=1.0)  # no theta during sleep
                    apply_hebbian_std(graph, LAMBDA_ACT, lr=SLEEP_LR)
                    graph.increment_step()

    fr, sr, damping = analyze_oscillation(all_mm)
    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': 'Sleep Consolidation', 'best': best, 'final': final, 'damping': damping, 'fr': fr, 'sr': sr}


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  STABILITY BATTERY: 5 Biological Mechanisms', flush=True)
    print('=' * 60, flush=True)
    print(f'All with theta. Sequential. 5000 cycles each.', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    results = []

    results.append(test_timing_selective())
    results.append(test_bcm())
    results.append(test_phase_gated())
    results.append(test_extreme_sparsity())
    results.append(test_sleep_consolidation())

    print(f'\n{"="*60}', flush=True)
    print('  RESULTS SUMMARY', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'{"Mechanism":<20} | {"Best":>7} | {"Final":>7} | {"Osc 1st":>7} | {"Osc 2nd":>7} | {"Status":>12}', flush=True)
    print('-' * 75, flush=True)
    print(f'{"Undamped (ref)":<20} | {"2.865x":>7} | {"0.603x":>7} | {"2.33":>7} | {"2.32":>7} | {"NO DAMPING":>12}', flush=True)
    for r in results:
        star = ' **' if r['damping'] == 'DAMPED' else ''
        print(f'{r["name"]:<20} | {r["best"]:.3f}x | {r["final"]:.3f}x | {r["fr"]:.3f} | {r["sr"]:.3f} | {r["damping"]:>12}{star}', flush=True)

    # Find winners
    damped = [r for r in results if r['damping'] == 'DAMPED']
    partial = [r for r in results if r['damping'] == 'PARTIAL']
    if damped:
        best = max(damped, key=lambda r: r['best'])
        print(f'\n  BEST DAMPED: {best["name"]} (best={best["best"]:.3f}x)', flush=True)
    elif partial:
        best = max(partial, key=lambda r: r['best'])
        print(f'\n  BEST PARTIAL: {best["name"]} (best={best["best"]:.3f}x)', flush=True)
    else:
        print(f'\n  NO MECHANISM ACHIEVED DAMPING', flush=True)

    torch.save(results, 'stability_battery_results.pt')
    print(f'\nSaved to stability_battery_results.pt', flush=True)
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
