"""Memory Substrate Test: recurrent connections + consolidation spectrum.

The hypothesis: oscillation is caused by a disconnect between active computation
and memory. The graph processes patterns but doesn't REMEMBER outcomes in a way
that modulates ongoing learning. Memory should be emergent:

1. RECURRENT EDGES: excitatory self-connections within spatial neighbourhoods.
   Activity sustains itself after input is removed. The echo IS working memory.

2. CONSOLIDATION SPECTRUM: edge stiffness grows with repeated co-activation.
   Not a separate system — a natural property of the weight dynamics.
   Frequently reinforced edges become hard to change. Rarely used edges stay plastic.

3. ACTIVITY-DEPENDENT LEARNING RATE: prediction error modulates plasticity.
   Low error (familiar) → near-zero learning. High error (novel) → full learning.
   This prevents Hebbian and Oja from endlessly wrestling on known patterns.

4. RECURRENT ECHO AS MEMORY QUERY: when a new pattern arrives, the residual
   activity from recurrence represents "what I was just thinking about."
   The interaction between new input and residual activity IS the memory-informed
   prediction. If residual matches input → familiar → low error → stable weights.

Checkpoints saved every 50 epochs for pause/resume.

Based on run_definitive_test.py with true silence + fixed sensory surface.
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

BASELINE = math.log(2)
N_EPOCHS = 1000
MEASURE_EVERY = 50
CHECKPOINT_EVERY = 50
SLEEP_EVERY = 20
INPUT_FRACTION = 0.20
SYMBOL_SPARSITY = 0.10
PAUSE = 20
CHECKPOINT_DIR = Path('checkpoints/memory_substrate')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]
WEIGHT_DECAY = 0.013

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
    _exc_mask = ns.type_mask(NodeType.EXCITATORY)
    _pv_mask = ns.type_mask(NodeType.PV)
    _sst_mask = ns.type_mask(NodeType.SST)
    _vip_mask = ns.type_mask(NodeType.VIP)
    _exc_f = _exc_mask.float()

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
# TRUE SILENCE node dynamics (from definitive test)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    device = ns.device
    N = ns.n_nodes
    exc_mask, pv_mask, sst_mask, vip_mask, exc_f = _exc_mask, _pv_mask, _sst_mask, _vip_mask, _exc_f
    if tau_mult is not None:
        basal_tau = 10.0 * tau_mult
        apical_tau = 20.0 * tau_mult
        input_norm = 1.0 / tau_mult
    else:
        basal_tau = 10.0
        apical_tau = 20.0
        input_norm = 1.0
    ns.basal += 1.0 * (-ns.basal / basal_tau + inputs.basal * theta_mod * input_norm) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += 1.0 * (-ns.apical / apical_tau + inputs.apical * (1.0 - sst_gate) * input_norm) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    raw = F.softplus(pred_err.abs()) - BASELINE
    ns.output = torch.where(exc_mask, raw.clamp(min=0.0, max=10.0) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
        inh_raw = F.softplus(ns.basal) - BASELINE
        out = inh_raw.clamp(min=0.0, max=10.0) * ns.gain * f
        if inh_type == NodeType.SST:
            sst_suppress = torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, min=0.0, max=1.0)
            out = out * sst_suppress
        ns.output = torch.where(mask, out, ns.output)
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
# MEMORY-AWARE LEARNING: consolidation spectrum + error-gated plasticity
# ================================================================
def apply_memory_learning(graph, lr_scale=1.0, is_replay=False):
    """Learning rule with emergent memory properties:

    1. Consolidation spectrum: post_trace tracks lifetime co-activation.
       High post_trace = stiff edge (resists change). Low = plastic.
       This is NOT a tag — it's the natural accumulation of Hebbian co-activation.

    2. Error-gated plasticity: prediction error at the destination modulates lr.
       Low error (familiar) → near-zero lr → weights stable.
       High error (novel/surprising) → full lr → learn fast.

    3. Recurrent edges learn the same way — they'll consolidate patterns that
       persist (working memory) and stay plastic for new patterns.
    """
    ns_ = graph.node_state
    pred_err = ns_.prediction_error

    # Global novelty signal: mean absolute prediction error across excitatory
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

        # Consolidation: post_trace accumulates co-activation over lifetime
        # Very slow accumulation (0.0001), moderate decay (0.999)
        # At 0.0001 build: needs ~10,000 co-activations to reach 0.5 stiffness
        # At 0.999 decay: half-life ~700 steps without reinforcement
        co_act = src * dst
        store.post_trace *= 0.999   # moderate decay — unreinforced memories fade
        store.post_trace += co_act * 0.0001  # very slow build — only repeated patterns consolidate
        store.post_trace.clamp_(0.0, 1.0)

        # Stiffness from consolidation: how resistant is this edge to change?
        # 0 = fully plastic, 1 = nearly frozen
        stiffness = store.post_trace

        # Error-gated plasticity at destination
        dst_error = pred_err[cache['dst64']].abs()
        # Normalize: error / global_novelty. Familiar = low ratio, novel = high ratio
        error_gate = (dst_error / global_novelty).clamp(0.0, 3.0)

        # Effective learning rate: base × error_gate × (1 - stiffness)
        # Familiar + consolidated = near-zero. Novel + plastic = full rate.
        plasticity = error_gate * (1.0 - 0.9 * stiffness)

        if et == EdgeType.DRIVING:
            if is_replay:
                lr = 0.001 * lr_scale
                dw = lr * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
                # Replay boosts consolidation
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
# RECURRENT CONNECTIVITY: add local recurrent driving edges
# ================================================================
def add_recurrent_edges(graph, k_recurrent=10):
    """Add local recurrent driving edges — nodes connect back to their
    spatial neighbours. This lets activity sustain itself (working memory).

    Uses KNN to find the k nearest excitatory neighbours and adds
    driving edges with small initial weights.
    """
    ns = graph.node_state
    device = ns.position.device
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    positions = ns.position[exc_idx]
    n_exc = exc_idx.shape[0]

    print(f'  Adding recurrent edges (k={k_recurrent})...', flush=True)

    all_src = []
    all_dst = []
    chunk_size = 2000

    for start in range(0, n_exc, chunk_size):
        end = min(start + chunk_size, n_exc)
        chunk_pos = positions[start:end]
        dists = torch.cdist(chunk_pos, positions)

        # Top-k nearest (exclude self)
        _, topk = dists.topk(k_recurrent + 1, dim=1, largest=False)
        # Remove self (distance 0)
        topk = topk[:, 1:k_recurrent + 1]

        n_chunk = end - start
        src_expanded = exc_idx[torch.arange(start, end, device=device).unsqueeze(1).expand(-1, k_recurrent).reshape(-1)]
        dst_expanded = exc_idx[topk.reshape(-1)]

        all_src.append(src_expanded.to(torch.int32))
        all_dst.append(dst_expanded.to(torch.int32))

    new_src = torch.cat(all_src)
    new_dst = torch.cat(all_dst)

    # Small initial weights — recurrence should be weak initially
    # and strengthen through Hebbian learning where activity persists
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
# SENSORY + MEASUREMENT (from definitive test)
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


def measure_apical(pred_name, target_name, symbols, graph, ns, mp, device,
                   theta, stp, hom, ip, tau_mult, steps=50):
    pred_nodes = symbols[pred_name]
    target_nodes = symbols[target_name]
    for s in range(steps):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pred_nodes.long()] += 2.0
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        for et in PLASTIC_EDGE_TYPES:
            if graph.has_edge_type(et):
                stp.update(graph.edge_store(et), ns, 1.0)
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


def measure_discrimination(symbols, seq, graph, ns, mp, device,
                            theta, stp, hom, ip, tau_mult, steps=50):
    n_correct, n_tested, total_disc = 0, 0, 0
    for i in range(len(seq) - 1):
        pred, target = seq[i], seq[i + 1]
        wrong = seq[(i + 3) % len(seq)]
        ap_correct = measure_apical(pred, target, symbols, graph, ns, mp, device,
                                     theta, stp, hom, ip, tau_mult, steps=steps)
        ap_wrong = measure_apical(wrong, target, symbols, graph, ns, mp, device,
                                   theta, stp, hom, ip, tau_mult, steps=steps)
        correct = ap_correct > ap_wrong
        n_correct += int(correct)
        n_tested += 1
        disc = (ap_correct - ap_wrong) / max(abs(ap_wrong), 1e-8) * 100
        total_disc += disc
    return n_correct / max(n_tested, 1) * 100, total_disc / max(n_tested, 1)


def measure_echo_persistence(symbols, name, graph, ns, mp, device, theta, tau_mult, steps=50):
    """Measure how long activity persists after input is removed.
    This IS working memory — the recurrent echo."""
    pattern = symbols[name]
    # Present pattern
    for s in range(steps):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pattern.long()] += 2.0
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()

    # Remove input, measure decay
    activity_trace = []
    for s in range(100):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        # NO INPUT — pure echo
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        act = ns.output[pattern.long()].mean().item()
        activity_trace.append(act)

    # Find half-life: when activity drops to 50% of initial
    if activity_trace[0] > 0.01:
        target = activity_trace[0] * 0.5
        half_life = None
        for t, a in enumerate(activity_trace):
            if a < target:
                half_life = t
                break
        if half_life is None:
            half_life = 100  # didn't decay — persistent memory
    else:
        half_life = 0

    return half_life, activity_trace


# ================================================================
# CHECKPOINT: save/load full state
# ================================================================
def save_checkpoint(epoch, graph, log, hipp, path):
    """Save full graph state for pause/resume."""
    state = {
        'epoch': epoch,
        'log': log,
        'step_count': graph.step_count,
        'n_edges': graph.n_edges(),
    }
    # Save node state
    ns = graph.node_state
    state['node_state'] = {
        'basal': ns.basal.cpu(),
        'apical': ns.apical.cpu(),
        'output': ns.output.cpu(),
        'activity_ema': ns.activity_ema.cpu(),
        'gain': ns.gain.cpu(),
        'prediction_error': ns.prediction_error.cpu(),
    }
    # Save edge weights and traces
    state['edges'] = {}
    for et in EdgeType:
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            state['edges'][et.name] = {
                'weight': store.weight.cpu(),
                'pre_trace': store.pre_trace.cpu(),
                'post_trace': store.post_trace.cpu(),
                'release_prob': store.release_prob.cpu(),
            }
    torch.save(state, path)


def load_checkpoint(path, graph):
    """Load full graph state for resume."""
    state = torch.load(path, weights_only=False)
    device = graph.device

    # Restore node state
    ns = graph.node_state
    for key, val in state['node_state'].items():
        getattr(ns, key).copy_(val.to(device))

    # Restore edge state
    for et_name, edge_state in state['edges'].items():
        et = EdgeType[et_name]
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            for key, val in edge_state.items():
                tensor = getattr(store, key)
                if tensor.shape == val.shape:
                    tensor.copy_(val.to(device))

    return state['epoch'], state['log']


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  MEMORY SUBSTRATE TEST', flush=True)
    print('  Recurrent edges + consolidation spectrum + error-gated lr', flush=True)
    print(f'  N=50K, {N_EPOCHS} epochs, checkpoints every {CHECKPOINT_EVERY}', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    cache_type_masks(ns)

    # Add recurrent edges (THE new thing)
    n_recurrent = add_recurrent_edges(graph, k_recurrent=10)
    print(f'  Recurrent: +{n_recurrent:,} driving edges (k=10 local)', flush=True)

    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)

    print(f'  Total: {graph.n_edges():,} edges (+{n_sw:,} SW, +{n_recurrent:,} recurrent)', flush=True)
    print(builder.summary(graph), flush=True)

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
    log = {'epoch': [], 'days_acc': [], 'days_disc': [], 'digits_acc': [], 'digits_disc': [],
           'echo_hl': [], 'avg_stiffness': [], 'n_edges': []}
    for f in sorted(CHECKPOINT_DIR.glob('epoch_*.pt')):
        latest_ckpt = f

    if latest_ckpt is not None:
        print(f'\n  Resuming from {latest_ckpt}', flush=True)
        start_epoch, log = load_checkpoint(latest_ckpt, graph)
        precompute_edge_data(graph, mp)
        print(f'  Resumed at epoch {start_epoch}', flush=True)
    else:
        # Baseline
        day_names = sequences['days']
        print(f'\n--- BASELINE ---', flush=True)
        acc_0, disc_0 = measure_discrimination(symbols, day_names, graph, ns, mp, device,
                                                theta, stp, hom, ip, tau_mult)
        hl, _ = measure_echo_persistence(symbols, 'Mon', graph, ns, mp, device, theta, tau_mult)
        print(f'  Days: acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)
        print(f'  Echo half-life: {hl} steps', flush=True)

    day_names = sequences['days']
    digit_names = sequences['digits']
    seq_keys = list(sequences.keys())
    t0 = time.perf_counter()

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
                    for et in PLASTIC_EDGE_TYPES:
                        if graph.has_edge_type(et):
                            stp.update(graph.edge_store(et), ns, 1.0)
                    apply_memory_learning(graph, lr_scale=1.0, is_replay=False)
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
                    apply_memory_learning(graph, lr_scale=1.0, is_replay=True)
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
                    apply_memory_learning(graph, lr_scale=3.0, is_replay=True)
                    graph.increment_step()

            # Homeostatic downscaling — scale by stiffness
            # Consolidated edges (high post_trace) resist downscaling
            # This is the memory protection: well-learned edges survive sleep
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    # Scale factor: 0.95 for plastic edges, ~1.0 for stiff edges
                    protection = 0.95 + 0.05 * store.post_trace  # range [0.95, 1.0]
                    store.weight *= protection

        # Structural plasticity
        if (epoch + 1) % 200 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]}', flush=True)
                precompute_edge_data(graph, mp)

        # Measure
        if (epoch + 1) % MEASURE_EVERY == 0:
            elapsed = time.perf_counter() - t0
            d_acc, d_disc = measure_discrimination(symbols, day_names, graph, ns, mp, device,
                                                    theta, stp, hom, ip, tau_mult, steps=pd_steps)
            di_acc, di_disc = measure_discrimination(symbols, digit_names, graph, ns, mp, device,
                                                      theta, stp, hom, ip, tau_mult, steps=pd_steps)
            hl, _ = measure_echo_persistence(symbols, 'Mon', graph, ns, mp, device, theta, tau_mult, steps=pd_steps)

            # Average stiffness (consolidation level)
            avg_stiff = 0
            n_edges_total = 0
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    avg_stiff += store.post_trace.sum().item()
                    n_edges_total += store.n_edges
            avg_stiff = avg_stiff / max(n_edges_total, 1)

            log['epoch'].append(epoch + 1)
            log['days_acc'].append(d_acc)
            log['days_disc'].append(d_disc)
            log['digits_acc'].append(di_acc)
            log['digits_disc'].append(di_disc)
            log['echo_hl'].append(hl)
            log['avg_stiffness'].append(avg_stiff)
            log['n_edges'].append(graph.n_edges())

            print(f'  Ep {epoch+1:5d} ({pd_steps}st): days={d_acc:.0f}%/{d_disc:+.1f}% | '
                  f'digits={di_acc:.0f}%/{di_disc:+.1f}% | '
                  f'echo={hl}st stiff={avg_stiff:.4f} edges={graph.n_edges():,} ({elapsed:.0f}s)', flush=True)

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f'epoch_{epoch+1:05d}.pt'
            save_checkpoint(epoch + 1, graph, log, hipp, ckpt_path)
            # Keep only last 3 checkpoints to save disk
            ckpts = sorted(CHECKPOINT_DIR.glob('epoch_*.pt'))
            for old in ckpts[:-3]:
                old.unlink()

    # Transfer test
    print(f'\n--- TRANSFER TEST ---', flush=True)
    novel_acc_before, novel_disc_before = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Before: acc={novel_acc_before:.0f}% disc={novel_disc_before:+.1f}%', flush=True)

    for ep in range(200):
        for name in novel_seq:
            pattern = symbols[name]
            for s in range(50):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pattern.long()] += get_strength()
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                for et in PLASTIC_EDGE_TYPES:
                    if graph.has_edge_type(et):
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_memory_learning(graph, lr_scale=1.0, is_replay=False)
                graph.increment_step()
        if (ep + 1) % 50 == 0:
            nacc, ndisc = measure_discrimination(
                symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            print(f'  Novel ep{ep+1}: acc={nacc:.0f}% disc={ndisc:+.1f}%', flush=True)

    novel_acc_after, novel_disc_after = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    transfer = novel_disc_after - novel_disc_before
    print(f'  Transfer: {novel_disc_before:+.1f}% -> {novel_disc_after:+.1f}% ({transfer:+.1f}%)', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    for key in ['days', 'digits']:
        if log[f'{key}_acc']:
            print(f'  {key}: final={log[f"{key}_acc"][-1]:.0f}%/{log[f"{key}_disc"][-1]:+.1f}% '
                  f'peak_disc={max(log[f"{key}_disc"]):+.1f}%', flush=True)
    if log['echo_hl']:
        print(f'  Echo half-life: {log["echo_hl"][0]} -> {log["echo_hl"][-1]} steps', flush=True)
    if log['avg_stiffness']:
        print(f'  Avg stiffness: {log["avg_stiffness"][0]:.4f} -> {log["avg_stiffness"][-1]:.4f}', flush=True)
    print(f'  Transfer: {transfer:+.1f}%', flush=True)

    torch.save(log, 'memory_substrate_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
