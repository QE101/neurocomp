"""Adaptive Consolidation v4: self-calibrating decay rate.

v3 lesson: fixed decay 0.999 was too aggressive (0.999^1500 = 0.22 per epoch).
Stiffness couldn't accumulate past ~0.1, making the entire consolidation spectrum
and un-consolidation mechanism irrelevant. Three coupled rates (decay, build,
un-consolidation) are impossible to tune by hand.

v4 fix: ADAPTIVE DECAY targeting a median stiffness.
Same principle as intrinsic plasticity (adapts threshold to hit target firing rate):
  - Compute median stiffness across all plastic edges
  - If too stiff → decay faster. If too plastic → decay slower.
  - One parameter (target_median = 0.35) replaces three coupled rate constants.

The target 0.35 means:
  - plasticity = 1 - 0.9 * 0.35 = 0.685 → consolidated edges learn at 68%
  - Un-consolidation at stiffness² = 0.12 → meaningful error-gated loosening
  - Sleep can push edges to 0.8+ (consolidated), wake pulls back toward 0.35
  - Genuine spectrum from plastic (0.05) to rigid (0.9)

Everything else identical to v3 for clean comparison.
"""

import sys, os, time, math
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.hippocampus import HippocampalSystem
from graph_brain.types import EdgeType, NodeType
from graph_brain.engine.fused_plasticity import FusedPlasticity, _node_core

BASELINE = math.log(2)
N_EPOCHS = 1000
MEASURE_EVERY = 50
CHECKPOINT_EVERY = 50
SLEEP_EVERY = 20
INPUT_FRACTION = 0.20
SYMBOL_SPARSITY = 0.10
PAUSE = 20
CHECKPOINT_DIR = Path('checkpoints/adaptive_consolidation')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]
WEIGHT_DECAY = 0.013

# Adaptive consolidation: target median stiffness
TARGET_STIFFNESS = 0.35
DECAY_ADAPT_RATE = 0.0001  # how fast the decay rate adjusts
DECAY_MIN = 0.9985         # fastest decay (most forgetting)
DECAY_MAX = 0.99995        # slowest decay (most retention)

OUTPUT_EDGE_CHANNELS = {
    EdgeType.DRIVING: Channel.BASAL,
    EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
    EdgeType.RETROGRADE: Channel.RETROGRADE,
}
CONTENT_EDGE_CHANNELS = {
    EdgeType.MODULATORY: Channel.APICAL,
    EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
}

_edge_cache = {}
_noise_buf = None
_noise_idx = 0
_noise_size = 100
_exc_mask = _pv_mask = _sst_mask = _vip_mask = _exc_f = None
_inh_mask = _inh_f = _pv_f = None
_current_decay = 0.999  # initial decay rate, will adapt
_fused = None  # FusedPlasticity instance
_compiled_node = None  # compiled node update (None on Windows)
_basal_tau = _apical_tau = _input_norm = None  # pre-computed tau tensors


def precompute_edge_data(graph, mp):
    global _edge_cache
    _edge_cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        _edge_cache[et] = {
            'src64': store.src.long(),
            'dst64': store.dst.long(),
            'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps),
        }

def init_noise_buffer(N, device):
    global _noise_buf
    _noise_buf = torch.randn(_noise_size, N, device=device) * 0.005

def get_noise(N):
    global _noise_idx
    noise = _noise_buf[_noise_idx % _noise_size]
    _noise_idx += 1
    return noise

def cache_type_masks(ns):
    global _exc_mask, _pv_mask, _sst_mask, _vip_mask, _exc_f
    global _inh_mask, _inh_f, _pv_f
    _exc_mask = ns.type_mask(NodeType.EXCITATORY)
    _pv_mask = ns.type_mask(NodeType.PV)
    _sst_mask = ns.type_mask(NodeType.SST)
    _vip_mask = ns.type_mask(NodeType.VIP)
    _exc_f = _exc_mask.float()
    _inh_mask = ~_exc_mask
    _inh_f = _inh_mask.float()
    _pv_f = _pv_mask.float()

def get_presentation_steps(epoch):
    if epoch < 300: return 100
    elif epoch < 700: return 50
    else: return 30

def get_strength():
    return 1.0 + np.random.random() * 2.0


CONFIG_50K = {
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
            'disinhibition': {'p_max': 0.4, 'sigma': 0.10, 'source_types': ['VIP'],
                              'target_types': ['SST'], 'constant_k': 10},
            'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'],
                          'target_types': ['PV'], 'constant_k': 5},
            'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 10},
            'max_radius': 0.5,
        },
        'structural': {'enabled': True, 'update_interval': 500, 'growth_rate': 0.2,
                        'prune_threshold': 0.005, 'edge_cost': 1e-6, 'max_degree': 5000},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {
        'enabled': True, 'n_levels': 2, 'split_axis': 2,
        'time_scale_factor': 3.0, 'inter_level_k': 5,
        'inter_level_sigma': 0.5, 'inter_level_init_weight': 0.02,
    },
    'hippocampal': {
        'enabled': True, 'n_dg': 2000, 'n_ca3': 500,
        'dg_sparsity': 0.02, 'dg_fan_in': 2000, 'ca3_sparsity': 0.10,
        'encoding_lr': 0.5, 'replay_strength': 0.5, 'replay_lr_scale': 0.2,
        'max_patterns': 20, 'replay_interleave': 5, 'replay_steps': 50,
    },
}


# ================================================================
# TRUE SILENCE node dynamics
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    N = ns.n_nodes
    if _compiled_node is not None and _basal_tau is not None:
        # Compiled path (WSL) — single fused kernel
        noise = get_noise(N)
        ns.basal, ns.apical, ns.output, ns.prediction_error, ns.activity_ema = _compiled_node(
            ns.basal, ns.apical, ns.output, ns.prediction_error, ns.gain, ns.activity_ema,
            inputs.basal, inputs.apical, inputs.sst_inhibition, inputs.pv_inhibition,
            inputs.electrical, inputs.vip_inhibition,
            _exc_mask, _exc_f, _inh_mask, _inh_f, _sst_mask, _pv_f,
            _basal_tau, _apical_tau, _input_norm, theta_mod, noise)
    else:
        # Eager path (Windows fallback)
        if tau_mult is not None:
            basal_tau = 10.0 * tau_mult
            apical_tau = 20.0 * tau_mult
            input_norm = 1.0 / tau_mult
        else:
            basal_tau = 10.0
            apical_tau = 20.0
            input_norm = 1.0
        ns.basal += 1.0 * (-ns.basal / basal_tau + inputs.basal * theta_mod * input_norm) * _exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += 1.0 * (-ns.apical / apical_tau + inputs.apical * (1.0 - sst_gate) * input_norm) * _exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(_exc_mask, raw.clamp(min=0.0, max=10.0) * pv_gain * ns.gain, ns.output)
        ns.prediction_error = torch.where(_exc_mask, pred_err, ns.prediction_error)
        inh_input = inputs.basal + inputs.electrical * _pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * _inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * _inh_f
        sst_suppress = torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0)
        inh_out = torch.where(_sst_mask, inh_out * sst_suppress, inh_out)
        ns.output = torch.where(_inh_mask, inh_out, ns.output)
        ns.output += get_noise(N)
        ns.output.clamp_(min=0.0, max=10.0)
        ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)


def dual_channel_send(ns, graph, mp, device):
    step = graph.step_count
    output = ns.output
    content = F.softplus(ns.basal).clamp(max=10.0)
    for et, ch in OUTPUT_EDGE_CHANNELS.items():
        if et not in _edge_cache: continue
        cache = _edge_cache[et]
        store = graph.edge_store(et)
        msg = output[cache['src64']] * store.release_prob * store.weight
        mp.delay_buffer.write(ch, store.dst, msg, cache['delay_steps'], step)
    for et, ch in CONTENT_EDGE_CHANNELS.items():
        if et not in _edge_cache: continue
        cache = _edge_cache[et]
        store = graph.edge_store(et)
        msg = content[cache['src64']] * store.release_prob * store.weight
        mp.delay_buffer.write(ch, store.dst, msg, cache['delay_steps'], step)
    if EdgeType.ELECTRICAL in _edge_cache:
        cache = _edge_cache[EdgeType.ELECTRICAL]
        store = graph.edge_store(EdgeType.ELECTRICAL)
        gap = store.weight * (output[cache['src64']] - output[cache['dst64']])
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, cache['delay_steps'], step)




# ================================================================
# ADAPTIVE DECAY: adjust decay rate to hit target median stiffness
# ================================================================
def adapt_decay_rate(graph):
    """Adjust the global decay rate so median stiffness converges to target.

    Like a thermostat: if the room (stiffness) is too hot (high), turn up
    the AC (faster decay). If too cold, turn it down.

    Runs once per epoch (cheap — one median computation).
    """
    global _current_decay
    all_stiff = []
    for et in PLASTIC_EDGE_TYPES:
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            all_stiff.append(store.post_trace)
    if not all_stiff:
        return _current_decay
    combined = torch.cat(all_stiff)
    current_median = combined.median().item()

    # Adjust: too stiff → lower decay (faster forgetting), too plastic → raise decay
    error = current_median - TARGET_STIFFNESS
    _current_decay -= DECAY_ADAPT_RATE * error
    _current_decay = max(DECAY_MIN, min(DECAY_MAX, _current_decay))
    return _current_decay


# ================================================================
# LEARNING WITH ADAPTIVE CONSOLIDATION + UN-CONSOLIDATION v3
# ================================================================
def apply_memory_learning(graph, lr_scale=1.0, is_replay=False):
    """Learning rule with adaptive consolidation decay + un-consolidation v3.

    The decay rate is no longer fixed at 0.999. It adapts each epoch to keep
    median stiffness at TARGET_STIFFNESS (0.35). This means:
    - Early training (lots of activity): decay speeds up to prevent runaway consolidation
    - Late training (settled): decay slows to let important edges consolidate
    - After sleep (stiffness spike): decay temporarily faster to rebalance

    Un-consolidation v3 (stiffness², sigmoid center 2.0) still applies on top.
    At target stiffness 0.35, un-consolidation amount = 0.0001 * 0.12 = meaningful.
    """
    ns_ = graph.node_state
    pred_err = ns_.prediction_error

    global_novelty = pred_err[_exc_mask].abs().mean().clamp(min=0.01)

    for et in PLASTIC_EDGE_TYPES:
        if et not in _edge_cache: continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        cache = _edge_cache[et]
        src = ns_.output[cache['src64']]
        dst = ns_.output[cache['dst64']]

        # Update pre_trace (temporal signal)
        store.pre_trace.lerp_(src, 0.05)

        # Consolidation: adaptive decay + fixed build
        co_act = src * dst
        store.post_trace *= _current_decay  # ADAPTIVE (was fixed 0.999)
        store.post_trace += co_act * 0.0001
        store.post_trace.clamp_(0.0, 1.0)

        stiffness = store.post_trace

        # Error-gated plasticity at destination
        dst_error = pred_err[cache['dst64']].abs()
        error_gate = (dst_error / global_novelty).clamp(0.0, 3.0)

        # Un-consolidation v3: stiffness² + sigmoid center 2.0
        error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
        unconsolidate_amount = 0.0001 * error_signal * stiffness * stiffness
        store.post_trace -= unconsolidate_amount
        store.post_trace.clamp_(0.0, 1.0)
        stiffness = store.post_trace

        # Effective learning rate
        plasticity = error_gate * (1.0 - 0.9 * stiffness)

        if et == EdgeType.DRIVING:
            if is_replay:
                lr = 0.001 * lr_scale
                dw = lr * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
                store.post_trace += co_act * 0.01
                store.post_trace.clamp_(0.0, 1.0)
            else:
                lr = 0.0001 * lr_scale
                dw = lr * plasticity * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.MODULATORY:
            lr = 0.001 * lr_scale
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.DISINHIBITION:
            lr = 0.002 * lr_scale
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC):
            lr = 0.0001 * lr_scale
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        else:
            lr = 0.001 * lr_scale
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)

        store.weight += dw
        store.weight.clamp_(0.0, 1.0)


# ================================================================
# RECURRENT + SMALL-WORLD CONNECTIVITY
# ================================================================
def add_recurrent_edges(graph, k_recurrent=10):
    ns = graph.node_state
    device = ns.position.device
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    positions = ns.position[exc_idx]
    n_exc = exc_idx.shape[0]
    print(f'  Adding recurrent edges (k={k_recurrent})...', flush=True)
    all_src, all_dst = [], []
    chunk_size = 2000
    for start in range(0, n_exc, chunk_size):
        end = min(start + chunk_size, n_exc)
        chunk_pos = positions[start:end]
        dists = torch.cdist(chunk_pos, positions)
        _, topk = dists.topk(k_recurrent + 1, dim=1, largest=False)
        topk = topk[:, 1:k_recurrent + 1]
        src_expanded = exc_idx[torch.arange(start, end, device=device).unsqueeze(1).expand(-1, k_recurrent).reshape(-1)]
        dst_expanded = exc_idx[topk.reshape(-1)]
        all_src.append(src_expanded.to(torch.int32))
        all_dst.append(dst_expanded.to(torch.int32))
    new_src = torch.cat(all_src)
    new_dst = torch.cat(all_dst)
    init_w = torch.full((new_src.shape[0],), 0.02, device=device)
    graph.add_edges(EdgeType.DRIVING, new_src, new_dst, weights=init_w)
    return new_src.shape[0]


def add_small_world_edges(graph, fraction=0.2):
    ns = graph.node_state
    device = ns.position.device
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    n_exc = exc_idx.shape[0]
    n_add = int(graph.n_edges(EdgeType.MODULATORY) * fraction)
    src = torch.randint(0, n_exc, (n_add,), device=device)
    dst = torch.randint(0, n_exc, (n_add,), device=device)
    valid = src != dst
    graph.add_edges(EdgeType.MODULATORY, exc_idx[src[valid]], exc_idx[dst[valid]],
                    weights=torch.full((valid.sum().item(),), 0.05, device=device))
    return valid.sum().item()


# ================================================================
# SENSORY SURFACE
# ================================================================
def build_sensory_symbols(input_region, device):
    n_input = input_region.shape[0]
    n_on = max(10, int(n_input * SYMBOL_SPARSITY))
    torch.manual_seed(777)
    sequences = {
        'short': ['S1', 'S2', 'S3'],
        'digits': ['D1', 'D2', 'D3', 'D4', 'D5'],
        'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    }
    novel_seq = ['N1', 'N2', 'N3', 'N4', 'N5']
    all_names = []
    for seq in sequences.values():
        all_names.extend(seq)
    all_names.extend(novel_seq)
    symbols = {}
    for name in all_names:
        perm = torch.randperm(n_input, device=device)
        symbols[name] = input_region[perm[:n_on]]
    print(f'  Sensory surface: {n_input} nodes, {n_on} ON per symbol', flush=True)
    return symbols, sequences, novel_seq, n_on


# ================================================================
# LEVEL 2 READOUT
# ================================================================
def discover_l2_representations(symbols, graph, ns, mp, device, theta, stp, hom, ip,
                                 tau_mult, l2_exc_idx, steps=100, top_k=200):
    l2_reps = {}
    for name, pattern in symbols.items():
        for s in range(steps):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[pattern.long()] += 2.0
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            _fused.stp(ns.output)
            if step % 100 == 0:
                for et in PLASTIC_EDGE_TYPES:
                    if graph.has_edge_type(et):
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()

        l2_activity = torch.zeros(l2_exc_idx.shape[0], device=device)
        for s in range(10):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[pattern.long()] += 2.0
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()
            l2_activity += ns.output[l2_exc_idx.long()]
        l2_activity /= 10.0

        k = min(top_k, l2_exc_idx.shape[0])
        _, topk_local = l2_activity.topk(k)
        l2_reps[name] = l2_exc_idx[topk_local]

        for s in range(10):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()

    return l2_reps


# ================================================================
# MEASUREMENT
# ================================================================
def measure_apical_l1(pred_name, target_name, symbols, graph, ns, mp, device,
                      theta, stp, hom, ip, tau_mult, steps=50):
    pred_nodes = symbols[pred_name]
    target_nodes = symbols[target_name]
    for s in range(steps):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pred_nodes.long()] += 2.0
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        _fused.stp(ns.output)
        if step % 100 == 0:
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()
    ap_vals = []
    for s in range(5):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        ap_vals.append(ns.apical[target_nodes.long()].mean().item())
    return float(np.mean(ap_vals))


def measure_apical_l2(pred_name, target_name, symbols, l2_reps, graph, ns, mp, device,
                      theta, stp, hom, ip, tau_mult, steps=50):
    pred_nodes = symbols[pred_name]
    target_l2 = l2_reps[target_name]

    for s in range(steps):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pred_nodes.long()] += 2.0
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        _fused.stp(ns.output)
        if step % 100 == 0:
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()

    ap_vals = []
    for s in range(10):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        ap_vals.append(ns.apical[target_l2.long()].mean().item())
    return float(np.mean(ap_vals))


def measure_discrimination_dual(symbols, seq, l2_reps, graph, ns, mp, device,
                                 theta, stp, hom, ip, tau_mult, steps=50):
    l1_correct, l2_correct, n_tested = 0, 0, 0
    l1_disc_total, l2_disc_total = 0, 0

    for i in range(len(seq) - 1):
        pred, target = seq[i], seq[i + 1]
        wrong = seq[(i + 3) % len(seq)]

        ap_correct_l1 = measure_apical_l1(pred, target, symbols, graph, ns, mp, device,
                                           theta, stp, hom, ip, tau_mult, steps=steps)
        ap_wrong_l1 = measure_apical_l1(wrong, target, symbols, graph, ns, mp, device,
                                         theta, stp, hom, ip, tau_mult, steps=steps)
        l1_correct += int(ap_correct_l1 > ap_wrong_l1)
        l1_disc = (ap_correct_l1 - ap_wrong_l1) / max(abs(ap_wrong_l1), 1e-8) * 100
        l1_disc_total += l1_disc

        ap_correct_l2 = measure_apical_l2(pred, target, symbols, l2_reps, graph, ns, mp, device,
                                           theta, stp, hom, ip, tau_mult, steps=steps)
        ap_wrong_l2 = measure_apical_l2(wrong, target, symbols, l2_reps, graph, ns, mp, device,
                                         theta, stp, hom, ip, tau_mult, steps=steps)
        l2_correct += int(ap_correct_l2 > ap_wrong_l2)
        l2_disc = (ap_correct_l2 - ap_wrong_l2) / max(abs(ap_wrong_l2), 1e-8) * 100
        l2_disc_total += l2_disc

        n_tested += 1

    n = max(n_tested, 1)
    return (l1_correct / n * 100, l1_disc_total / n,
            l2_correct / n * 100, l2_disc_total / n)


def measure_echo_persistence(symbols, name, graph, ns, mp, device, theta, tau_mult, steps=50):
    pattern = symbols[name]
    for s in range(steps):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pattern.long()] += 2.0
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
    activity_trace = []
    for s in range(100):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        act = ns.output[pattern.long()].mean().item()
        activity_trace.append(act)
    if activity_trace[0] > 0.01:
        target = activity_trace[0] * 0.5
        half_life = None
        for t, a in enumerate(activity_trace):
            if a < target:
                half_life = t
                break
        if half_life is None:
            half_life = 100
    else:
        half_life = 0
    return half_life, activity_trace


def get_stiffness_stats(graph):
    all_stiff = []
    for et in PLASTIC_EDGE_TYPES:
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            all_stiff.append(store.post_trace)
    if not all_stiff:
        return 0, 0, 0, 0
    combined = torch.cat(all_stiff)
    mean = combined.mean().item()
    p10 = combined.quantile(0.1).item()
    p50 = combined.quantile(0.5).item()
    p90 = combined.quantile(0.9).item()
    return mean, p10, p50, p90


# ================================================================
# CHECKPOINT
# ================================================================
def save_checkpoint(epoch, graph, log, hipp, path):
    state = {
        'epoch': epoch, 'log': log, 'step_count': graph.step_count,
        'n_edges': graph.n_edges(), 'current_decay': _current_decay,
    }
    ns = graph.node_state
    state['node_state'] = {
        'basal': ns.basal.cpu(), 'apical': ns.apical.cpu(),
        'output': ns.output.cpu(), 'activity_ema': ns.activity_ema.cpu(),
        'gain': ns.gain.cpu(), 'prediction_error': ns.prediction_error.cpu(),
    }
    state['edges'] = {}
    for et in EdgeType:
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            state['edges'][et.name] = {
                'weight': store.weight.cpu(), 'pre_trace': store.pre_trace.cpu(),
                'post_trace': store.post_trace.cpu(), 'release_prob': store.release_prob.cpu(),
            }
    torch.save(state, path)


def load_checkpoint(path, graph):
    global _current_decay
    state = torch.load(path, weights_only=False)
    device = graph.device
    ns = graph.node_state
    for key, val in state['node_state'].items():
        getattr(ns, key).copy_(val.to(device))
    for et_name, edge_state in state['edges'].items():
        et = EdgeType[et_name]
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            for key, val in edge_state.items():
                tensor = getattr(store, key)
                if tensor.shape == val.shape:
                    tensor.copy_(val.to(device))
    if 'current_decay' in state:
        _current_decay = state['current_decay']
    return state['epoch'], state['log']


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  ADAPTIVE CONSOLIDATION v4 TEST', flush=True)
    print(f'  Self-calibrating decay -> target median stiffness {TARGET_STIFFNESS}', flush=True)
    print(f'  + Un-consolidation v3 (stiffness^2, sigmoid 2.0)', flush=True)
    print(f'  N=50K, {N_EPOCHS} epochs', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    cache_type_masks(ns)

    n_recurrent = add_recurrent_edges(graph, k_recurrent=10)
    print(f'  Recurrent: +{n_recurrent:,} driving edges (k=10 local)', flush=True)

    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)

    print(f'  Total: {graph.n_edges():,} edges (+{n_sw:,} SW, +{n_recurrent:,} recurrent)', flush=True)
    print(builder.summary(graph), flush=True)

    l2_exc_idx = torch.where(
        ns.type_mask(NodeType.EXCITATORY) & (ns.hierarchy_level == 2)
    )[0]
    print(f'  Level 2 excitatory nodes: {l2_exc_idx.shape[0]:,}', flush=True)

    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    precompute_edge_data(graph, mp)
    init_noise_buffer(N, device)

    exc_idx = torch.where(_exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, sequences, novel_seq, n_on = build_sensory_symbols(input_region, device)
    all_symbol_nodes = input_region

    hipp = HippocampalSystem(config=config.hippocampal, cortical_input_indices=all_symbol_nodes,
                              n_cortical=N, device=device, seed=config.simulation.seed)

    # Check for resume
    latest_ckpt = None
    start_epoch = 0
    log = {'epoch': [],
           'l1_days_acc': [], 'l1_days_disc': [], 'l2_days_acc': [], 'l2_days_disc': [],
           'l1_digits_acc': [], 'l1_digits_disc': [], 'l2_digits_acc': [], 'l2_digits_disc': [],
           'echo_hl': [], 'avg_stiffness': [], 'stiff_p10': [], 'stiff_p50': [], 'stiff_p90': [],
           'decay_rate': [], 'n_edges': []}
    for f in sorted(CHECKPOINT_DIR.glob('epoch_*.pt')):
        latest_ckpt = f

    if latest_ckpt is not None:
        print(f'\n  Resuming from {latest_ckpt}', flush=True)
        start_epoch, log = load_checkpoint(latest_ckpt, graph)
        for key in ['stiff_p10', 'stiff_p50', 'stiff_p90', 'decay_rate']:
            if key not in log:
                log[key] = [0.0] * len(log['epoch'])
        precompute_edge_data(graph, mp)
        print(f'  Resumed at epoch {start_epoch}, decay={_current_decay:.6f}', flush=True)

    # Build fused plasticity (after potential checkpoint restore)
    global _fused
    _fused = FusedPlasticity(graph, config.edges.stp, mp.delay_buffer)
    print(f'  Fused plasticity: {_fused.n_total:,} edges in {len(_fused.active_types)} types', flush=True)

    # torch.compile: ~1.8x on top of fusion (Linux/WSL only)
    _fused.enable_compile()

    # Compiled node update
    global _compiled_node, _basal_tau, _apical_tau, _input_norm
    if tau_mult is not None:
        _basal_tau = 10.0 * tau_mult
        _apical_tau = 20.0 * tau_mult
        _input_norm = 1.0 / tau_mult
    else:
        _basal_tau = torch.tensor(10.0, device=device)
        _apical_tau = torch.tensor(20.0, device=device)
        _input_norm = torch.tensor(1.0, device=device)
    if sys.platform != 'win32':
        try:
            _compiled_node = torch.compile(_node_core, mode='default')
            # Warmup
            _noise = torch.randn(N, device=device) * 0.005
            _compiled_node(
                ns.basal.clone(), ns.apical.clone(), ns.output.clone(),
                ns.prediction_error.clone(), ns.gain, ns.activity_ema.clone(),
                torch.zeros(N, device=device), torch.zeros(N, device=device),
                torch.zeros(N, device=device), torch.zeros(N, device=device),
                torch.zeros(N, device=device), torch.zeros(N, device=device),
                _exc_mask, _exc_f, _inh_mask, _inh_f, _sst_mask, _pv_f,
                _basal_tau, _apical_tau, _input_norm, 1.0, _noise)
            torch.cuda.synchronize()
            print('  [compile] node update compiled', flush=True)
        except Exception as e:
            print(f'  [compile] node update failed: {e}', flush=True)
            _compiled_node = None
    else:
        _compiled_node = None

    day_names = sequences['days']
    digit_names = sequences['digits']
    seq_keys = list(sequences.keys())
    t0 = time.perf_counter()

    # Initial L2 representation discovery
    print(f'\n--- DISCOVERING L2 REPRESENTATIONS ---', flush=True)
    l2_reps = discover_l2_representations(symbols, graph, ns, mp, device, theta, stp, hom, ip,
                                            tau_mult, l2_exc_idx, steps=100, top_k=200)
    for name in list(symbols.keys())[:3]:
        overlap_count = 0
        for other in symbols:
            if other == name: continue
            overlap = len(set(l2_reps[name].cpu().tolist()) & set(l2_reps[other].cpu().tolist()))
            overlap_count += overlap
        print(f'  {name}: L2 rep={l2_reps[name].shape[0]} nodes, avg overlap={overlap_count / (len(symbols)-1):.0f}', flush=True)

    if start_epoch == 0:
        print(f'\n--- BASELINE ---', flush=True)
        l1_da, l1_dd, l2_da, l2_dd = measure_discrimination_dual(
            symbols, day_names, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
        hl, _ = measure_echo_persistence(symbols, 'Mon', graph, ns, mp, device, theta, tau_mult)
        mean_s, p10, p50, p90 = get_stiffness_stats(graph)
        print(f'  Days L1: acc={l1_da:.0f}% disc={l1_dd:+.1f}%', flush=True)
        print(f'  Days L2: acc={l2_da:.0f}% disc={l2_dd:+.1f}%', flush=True)
        print(f'  Echo half-life: {hl} steps', flush=True)
        print(f'  Stiffness: mean={mean_s:.4f} p10={p10:.4f} p50={p50:.4f} p90={p90:.4f}', flush=True)
        print(f'  Decay rate: {_current_decay:.6f}', flush=True)

    print(f'\n--- TRAINING ---', flush=True)
    for epoch in range(start_epoch, N_EPOCHS):
        pd_steps = get_presentation_steps(epoch)
        epoch_order = list(seq_keys)
        np.random.shuffle(epoch_order)

        for seq_key in epoch_order:
            seq = sequences[seq_key]
            for name in seq:
                strength = get_strength()
                pattern = symbols[name]
                for s in range(pd_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pattern.long()] += strength
                    error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                    _fused.stp(ns.output)
                    _fused.learn(ns, _exc_mask, _current_decay, is_replay=False)
                    if step % 100 == 0:
                        for et in PLASTIC_EDGE_TYPES:
                            if graph.has_edge_type(et):
                                hom.update(graph.edge_store(et), ns, 1.0)
                        ip.update(ns)
                    graph.increment_step()
                hipp.encode(ns.output, graph.step_count)

            for s in range(PAUSE):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                graph.increment_step()

        # Adapt decay rate once per epoch (cheap)
        current_decay = adapt_decay_rate(graph)

        # Sleep
        if (epoch + 1) % SLEEP_EVERY == 0 and hipp.n_stored() > 0:
            schedule = hipp.replay_schedule(config.hippocampal.replay_interleave)
            for pidx in schedule:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(config.hippocampal.replay_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    _fused.learn(ns, _exc_mask, _current_decay, is_replay=True)
                    graph.increment_step()

            # Sharp-wave ripples
            schedule2 = hipp.replay_schedule(2)
            for pidx in schedule2:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(10):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay * 2.0
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    _fused.learn(ns, _exc_mask, _current_decay, is_replay=True, lr_scale=3.0)
                    graph.increment_step()

            # Homeostatic downscaling with memory protection
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    protection = 0.95 + 0.05 * store.post_trace
                    store.weight *= protection

        # Structural plasticity
        if (epoch + 1) % 200 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]}', flush=True)
                precompute_edge_data(graph, mp)
                _fused.rebuild(graph, mp.delay_buffer)

        # Measure
        if (epoch + 1) % MEASURE_EVERY == 0:
            elapsed = time.perf_counter() - t0

            l2_reps = discover_l2_representations(symbols, graph, ns, mp, device, theta, stp, hom, ip,
                                                    tau_mult, l2_exc_idx, steps=50, top_k=200)

            l1_da, l1_dd, l2_da, l2_dd = measure_discrimination_dual(
                symbols, day_names, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult, steps=pd_steps)
            l1_dia, l1_did, l2_dia, l2_did = measure_discrimination_dual(
                symbols, digit_names, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult, steps=pd_steps)
            hl, _ = measure_echo_persistence(symbols, 'Mon', graph, ns, mp, device, theta, tau_mult, steps=pd_steps)

            mean_s, p10, p50, p90 = get_stiffness_stats(graph)

            log['epoch'].append(epoch + 1)
            log['l1_days_acc'].append(l1_da); log['l1_days_disc'].append(l1_dd)
            log['l2_days_acc'].append(l2_da); log['l2_days_disc'].append(l2_dd)
            log['l1_digits_acc'].append(l1_dia); log['l1_digits_disc'].append(l1_did)
            log['l2_digits_acc'].append(l2_dia); log['l2_digits_disc'].append(l2_did)
            log['echo_hl'].append(hl)
            log['avg_stiffness'].append(mean_s)
            log['stiff_p10'].append(p10)
            log['stiff_p50'].append(p50)
            log['stiff_p90'].append(p90)
            log['decay_rate'].append(_current_decay)
            log['n_edges'].append(graph.n_edges())

            print(f'  Ep {epoch+1:5d} ({pd_steps}st):', flush=True)
            print(f'    Days  L1={l1_da:.0f}%/{l1_dd:+.1f}%  L2={l2_da:.0f}%/{l2_dd:+.1f}%', flush=True)
            print(f'    Digit L1={l1_dia:.0f}%/{l1_did:+.1f}%  L2={l2_dia:.0f}%/{l2_did:+.1f}%', flush=True)
            print(f'    echo={hl}st stiff={mean_s:.4f} [p10={p10:.4f} p50={p50:.4f} p90={p90:.4f}] decay={_current_decay:.6f} edges={graph.n_edges():,} ({elapsed:.0f}s)', flush=True)

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f'epoch_{epoch+1:05d}.pt'
            save_checkpoint(epoch + 1, graph, log, hipp, ckpt_path)
            ckpts = sorted(CHECKPOINT_DIR.glob('epoch_*.pt'))
            for old in ckpts[:-3]:
                old.unlink()

    # ================================================================
    # TRANSFER TEST
    # ================================================================
    print(f'\n--- TRANSFER TEST ---', flush=True)
    print('  (Does adaptive consolidation allow novel sequences to break through?)', flush=True)

    l2_reps_novel = discover_l2_representations(
        {n: symbols[n] for n in novel_seq}, graph, ns, mp, device,
        theta, stp, hom, ip, tau_mult, l2_exc_idx, steps=100, top_k=200)
    l2_reps.update(l2_reps_novel)

    _, _, l2_novel_before_acc, l2_novel_before_disc = measure_discrimination_dual(
        symbols, novel_seq, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Before: L2 acc={l2_novel_before_acc:.0f}% disc={l2_novel_before_disc:+.1f}%', flush=True)

    for ep in range(200):
        for name in novel_seq:
            pattern = symbols[name]
            for s in range(50):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pattern.long()] += get_strength()
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                _fused.stp(ns.output)
                _fused.learn(ns, _exc_mask, _current_decay, is_replay=False)
                graph.increment_step()

        # Adapt during transfer too
        adapt_decay_rate(graph)

        if (ep + 1) % 50 == 0:
            l2_reps_novel = discover_l2_representations(
                {n: symbols[n] for n in novel_seq}, graph, ns, mp, device,
                theta, stp, hom, ip, tau_mult, l2_exc_idx, steps=50, top_k=200)
            l2_reps.update(l2_reps_novel)

            _, l1_nd, _, l2_nd = measure_discrimination_dual(
                symbols, novel_seq, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            mean_s, p10, p50, p90 = get_stiffness_stats(graph)
            print(f'  Novel ep{ep+1}: L1={l1_nd:+.1f}% L2={l2_nd:+.1f}% stiff={mean_s:.4f} [{p10:.4f}/{p50:.4f}/{p90:.4f}] decay={_current_decay:.6f}', flush=True)

    # Final novel measurement
    l2_reps_novel = discover_l2_representations(
        {n: symbols[n] for n in novel_seq}, graph, ns, mp, device,
        theta, stp, hom, ip, tau_mult, l2_exc_idx, steps=100, top_k=200)
    l2_reps.update(l2_reps_novel)

    _, l1_final, _, l2_final = measure_discrimination_dual(
        symbols, novel_seq, l2_reps, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    l2_transfer = l2_final - l2_novel_before_disc

    # Results
    print(f'\n{"="*60}', flush=True)
    print(f'  RESULTS: Adaptive Consolidation v4', flush=True)
    print(f'  Target stiffness={TARGET_STIFFNESS}, final decay={_current_decay:.6f}', flush=True)
    print(f'{"="*60}', flush=True)

    for key_name, l2_key in [('Days', 'days'), ('Digits', 'digits')]:
        if log[f'l1_{l2_key}_acc']:
            print(f'  {key_name}:', flush=True)
            print(f'    L1: final={log[f"l1_{l2_key}_acc"][-1]:.0f}%/{log[f"l1_{l2_key}_disc"][-1]:+.1f}% '
                  f'peak={max(log[f"l1_{l2_key}_disc"]):+.1f}%', flush=True)
            print(f'    L2: final={log[f"l2_{l2_key}_acc"][-1]:.0f}%/{log[f"l2_{l2_key}_disc"][-1]:+.1f}% '
                  f'peak={max(log[f"l2_{l2_key}_disc"]):+.1f}%', flush=True)

    if log['echo_hl']:
        print(f'  Echo half-life: {log["echo_hl"][0]} -> {log["echo_hl"][-1]} steps', flush=True)
    if log['avg_stiffness']:
        print(f'  Stiffness trajectory:', flush=True)
        print(f'    mean: {log["avg_stiffness"][0]:.4f} -> {log["avg_stiffness"][-1]:.4f}', flush=True)
        print(f'    p50:  {log["stiff_p50"][0]:.4f} -> {log["stiff_p50"][-1]:.4f} (target={TARGET_STIFFNESS})', flush=True)
        print(f'    p90:  {log["stiff_p90"][0]:.4f} -> {log["stiff_p90"][-1]:.4f}', flush=True)
        if len(log['avg_stiffness']) >= 5:
            last5 = log['stiff_p50'][-5:]
            stiff_range = max(last5) - min(last5)
            print(f'    p50 last-5 range: {stiff_range:.4f} (< 0.05 = stable)', flush=True)
    if log['decay_rate']:
        print(f'  Decay rate: {log["decay_rate"][0]:.6f} -> {log["decay_rate"][-1]:.6f} (range [{DECAY_MIN}, {DECAY_MAX}])', flush=True)

    print(f'\n  Transfer (L1): {l1_final:+.1f}%', flush=True)
    print(f'  Transfer (L2): {l2_novel_before_disc:+.1f}% -> {l2_final:+.1f}% ({l2_transfer:+.1f}%)', flush=True)
    print(f'\n  KEY: Does p50 converge to ~{TARGET_STIFFNESS}?', flush=True)
    print(f'  KEY: Does decay rate stabilize (not railing at min/max)?', flush=True)
    print(f'  KEY: Does transfer improve vs v3 (which was inert)?', flush=True)
    print(f'  Compare: v2 transfer was -12.1%, v3 TBD, run_definitive_test was +5.4%', flush=True)

    torch.save(log, 'adaptive_consolidation_results.pt')
    print(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
