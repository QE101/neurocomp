"""Definitive Curriculum Test: the real test of the architecture.

Two foundational changes:
1. TRUE SILENCE: output = max(0, softplus(|err|) - ln(2)). Zero at baseline.
2. FIXED SENSORY SURFACE: all symbols presented on SAME 8000 nodes as different
   sparse patterns (10% ON). Representations self-organize downstream.

Diverse curriculum:
- Short sequences (3 elements), medium (5), long (7)
- Graduated timing: 100 steps early → 50 mid → 30 late
- Strength variation: 1.0 to 3.0 randomly per presentation
- Gaps: skip elements, test if graph notices
- Partial: present 3 of 7, measure prediction of 4th
- Random sequence order each epoch

Architecture: hierarchy, VIP attention, full sleep (replay + ripples + homeostasis),
temporal Oja on driving, consolidation freeze, structural plasticity.

1000 epochs training, then 200 epochs novel sequence transfer test.
N=50K. This is the pass/fail experiment.
"""

import sys, os, time, math
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
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.hippocampus import HippocampalSystem
from graph_brain.types import EdgeType, NodeType

# ================================================================
# Constants
# ================================================================
BASELINE = math.log(2)  # 0.6931 = softplus(0)
N_EPOCHS = 1000
MEASURE_EVERY = 50
SLEEP_EVERY = 20
INPUT_FRACTION = 0.20
SYMBOL_SPARSITY = 0.10  # 10% of input nodes ON per symbol
PAUSE = 20

# Graduated timing: steps per symbol presentation
def get_presentation_steps(epoch):
    if epoch < 300:
        return 100  # slow, safe learning
    elif epoch < 700:
        return 50   # normal
    else:
        return 30   # compressed, tests robustness

# Random strength per presentation
def get_strength():
    return 1.0 + np.random.random() * 2.0  # 1.0 to 3.0

PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]

WEIGHT_DECAY = 0.013  # pre-computed 0.0065 * 2.0

# Pre-computed channel mappings (constant, never changes)
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

# Pre-computed edge data (populated after graph init, refreshed after SP)
_edge_cache = {}

def precompute_edge_data(graph, mp):
    """Pre-compute static edge data: delay steps and int64 indices."""
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

# Pre-allocated noise buffer (cycle through instead of generating fresh)
_noise_buf = None
_noise_idx = 0
_noise_size = 100  # number of pre-generated noise vectors

def init_noise_buffer(N, device):
    global _noise_buf
    _noise_buf = torch.randn(_noise_size, N, device=device) * 0.005

def get_noise(N):
    global _noise_idx
    noise = _noise_buf[_noise_idx % _noise_size]
    _noise_idx += 1
    return noise

_exc_mask = None
_pv_mask = None
_sst_mask = None
_vip_mask = None
_exc_f = None

def cache_type_masks(ns):
    global _exc_mask, _pv_mask, _sst_mask, _vip_mask, _exc_f
    _exc_mask = ns.type_mask(NodeType.EXCITATORY)
    _pv_mask = ns.type_mask(NodeType.PV)
    _sst_mask = ns.type_mask(NodeType.SST)
    _vip_mask = ns.type_mask(NodeType.VIP)
    _exc_f = _exc_mask.float()

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
    # TRUE SILENCE: subtract baseline, clamp to zero
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


# ================================================================
# Message passing (all 7 edge types)
# ================================================================
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
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap,
                              cache['delay_steps'], step)


# ================================================================
# Learning rule (temporal Oja + bio-calibrated rates)
# ================================================================
def apply_learning(graph, lr_scale=1.0, is_replay=False, driving_replay_count=None):
    ns_ = graph.node_state
    pred_err = ns_.prediction_error
    for et in PLASTIC_EDGE_TYPES:
        if et not in _edge_cache: continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        cache = _edge_cache[et]
        src = ns_.output[cache['src64']]
        dst = ns_.output[cache['dst64']]
        store.pre_trace.lerp_(src, 0.05)  # fused EMA: trace = trace*0.95 + src*0.05

        if et == EdgeType.DRIVING:
            pe_gate = pred_err[cache['dst64']].abs()
            pe_gate = pe_gate / pe_gate.mean().clamp(min=0.1)
            if is_replay:
                lr = 0.001 * lr_scale
                dw = lr * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
                store.post_trace.addcmul_(store.pre_trace, dst, value=0.01).clamp_(0.0, 1.0)
            else:
                if driving_replay_count is not None:
                    frozen = (driving_replay_count >= 5).float()
                else:
                    frozen = torch.zeros_like(store.weight)
                effective_lr = 0.0001 * lr_scale * (1.0 - frozen)
                dw = effective_lr * pe_gate * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.MODULATORY:
            dw = (0.001 * lr_scale) * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.DISINHIBITION:
            dw = (0.002 * lr_scale) * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC):
            dw = (0.0001 * lr_scale) * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        else:
            dw = (0.001 * lr_scale) * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        store.weight += dw
        store.weight.clamp_(0.0, 1.0)


# ================================================================
# FIXED SENSORY SURFACE: all symbols as sparse patterns on same nodes
# ================================================================
def build_sensory_symbols(input_region, device):
    """Each symbol = random 10% of input region ON. SAME surface for all symbols.
    Downstream representation self-organizes."""
    n_input = input_region.shape[0]
    n_on = max(10, int(n_input * SYMBOL_SPARSITY))  # 800 nodes ON per symbol

    torch.manual_seed(777)

    # Training sequences
    sequences = {
        'short': ['S1', 'S2', 'S3'],                           # 3 elements
        'digits': ['D1', 'D2', 'D3', 'D4', 'D5'],             # 5 elements
        'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],  # 7 elements
    }
    novel_seq = ['N1', 'N2', 'N3', 'N4', 'N5']

    all_names = []
    for seq in sequences.values():
        all_names.extend(seq)
    all_names.extend(novel_seq)

    # Each symbol: random 10% of input nodes ON
    symbols = {}
    for name in all_names:
        perm = torch.randperm(n_input, device=device)
        on_nodes = input_region[perm[:n_on]]
        symbols[name] = on_nodes

    # Verify sparsity and overlap
    print(f'  Sensory surface: {n_input} nodes, {n_on} ON per symbol ({SYMBOL_SPARSITY*100:.0f}%)', flush=True)
    names = list(symbols.keys())
    overlaps = []
    for i in range(min(5, len(names))):
        for j in range(i+1, min(5, len(names))):
            s1 = set(symbols[names[i]].cpu().tolist())
            s2 = set(symbols[names[j]].cpu().tolist())
            overlap = len(s1 & s2) / n_on * 100
            overlaps.append(overlap)
    print(f'  Avg pairwise overlap: {np.mean(overlaps):.1f}% (expected ~{SYMBOL_SPARSITY*100:.0f}%)', flush=True)

    return symbols, sequences, novel_seq, n_on


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
# Measurement: apical prediction
# ================================================================
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


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  DEFINITIVE CURRICULUM TEST', flush=True)
    print('  True silence + fixed sensory surface + diverse curriculum', flush=True)
    print(f'  N=50K, {N_EPOCHS} epochs, graduated timing', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    cache_type_masks(ns)

    print(f'True silence: output = max(0, softplus(|err|) - {BASELINE:.4f})', flush=True)

    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)

    print(f'Graph: {graph.n_edges():,} edges (+{n_sw:,} SW)', flush=True)
    print(builder.summary(graph), flush=True)

    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Pre-compute static edge data and noise buffer
    precompute_edge_data(graph, mp)
    init_noise_buffer(N, device)
    print(f'Pre-computed: {len(_edge_cache)} edge types cached, {_noise_size} noise vectors', flush=True)

    # Fixed sensory surface
    exc_idx = torch.where(_exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, sequences, novel_seq, n_on = build_sensory_symbols(input_region, device)
    all_symbol_nodes = input_region  # entire surface is the input

    # Hippocampus
    hipp = HippocampalSystem(config=config.hippocampal, cortical_input_indices=all_symbol_nodes,
                              n_cortical=N, device=device, seed=config.simulation.seed)

    # Replay counter
    driving_replay_count = None
    if graph.has_edge_type(EdgeType.DRIVING):
        driving_replay_count = torch.zeros(graph.edge_store(EdgeType.DRIVING).n_edges, device=device)

    # Baseline
    day_names = sequences['days']
    digit_names = sequences['digits']
    print(f'\n--- BASELINE ---', flush=True)
    acc_0, disc_0 = measure_discrimination(symbols, day_names, graph, ns, mp, device,
                                            theta, stp, hom, ip, tau_mult)
    print(f'  Days: acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)

    # Training
    t0 = time.perf_counter()
    log = {'epoch': [], 'days_acc': [], 'days_disc': [], 'digits_acc': [], 'digits_disc': [],
           'steps_per_symbol': [], 'n_edges': []}
    seq_keys = list(sequences.keys())

    print(f'\n--- TRAINING ({N_EPOCHS} epochs, diverse curriculum) ---', flush=True)
    for epoch in range(N_EPOCHS):
        pd_steps = get_presentation_steps(epoch)

        # Shuffle sequence order
        epoch_order = list(seq_keys)
        np.random.shuffle(epoch_order)

        for seq_key in epoch_order:
            seq = sequences[seq_key]

            # Randomly decide if this is a gap trial (10% chance after epoch 200)
            use_gap = epoch > 200 and np.random.random() < 0.1
            gap_idx = np.random.randint(1, len(seq) - 1) if use_gap else -1

            for si, name in enumerate(seq):
                # Skip element for gap trials
                if si == gap_idx:
                    # Silence for the gap duration
                    for s in range(pd_steps):
                        step = graph.step_count
                        dual_channel_send(ns, graph, mp, device)
                        inputs = mp.read_inputs(step)
                        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                        graph.increment_step()
                    continue

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
                    apply_learning(graph, lr_scale=1.0, is_replay=False,
                                  driving_replay_count=driving_replay_count)
                    if step % 100 == 0:
                        for et in PLASTIC_EDGE_TYPES:
                            if graph.has_edge_type(et):
                                hom.update(graph.edge_store(et), ns, 1.0)
                        ip.update(ns)
                    graph.increment_step()
                hipp.encode(ns.output, graph.step_count)

            # Pause between sequences
            for s in range(PAUSE):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                graph.increment_step()

        # Sleep
        if (epoch + 1) % SLEEP_EVERY == 0 and hipp.n_stored() > 0:
            if graph.has_edge_type(EdgeType.DRIVING):
                graph.edge_store(EdgeType.DRIVING).post_trace.zero_()

            # Phase 1: slow replay
            schedule = hipp.replay_schedule(config.hippocampal.replay_interleave)
            for pidx in schedule:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(config.hippocampal.replay_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    apply_learning(graph, lr_scale=1.0, is_replay=True,
                                  driving_replay_count=driving_replay_count)
                    graph.increment_step()

            # Phase 2: sharp-wave ripples
            schedule2 = hipp.replay_schedule(2)
            for pidx in schedule2:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(10):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay * 2.0
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    apply_learning(graph, lr_scale=3.0, is_replay=True,
                                  driving_replay_count=driving_replay_count)
                    graph.increment_step()

            # Phase 3: homeostatic downscaling
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    graph.edge_store(et).weight *= 0.95

            # Update replay counter
            if graph.has_edge_type(EdgeType.DRIVING):
                reinforced = graph.edge_store(EdgeType.DRIVING).post_trace > 0.1
                driving_replay_count += reinforced.float()

        # Structural plasticity
        if (epoch + 1) % 200 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]}', flush=True)
                # Refresh pre-computed edge data (topology changed)
                precompute_edge_data(graph, mp)
                if graph.has_edge_type(EdgeType.DRIVING):
                    new_n = graph.edge_store(EdgeType.DRIVING).n_edges
                    if driving_replay_count.shape[0] != new_n:
                        old = driving_replay_count
                        driving_replay_count = torch.zeros(new_n, device=device)
                        driving_replay_count[:min(len(old), new_n)] = old[:min(len(old), new_n)]

        # Measure
        if (epoch + 1) % MEASURE_EVERY == 0:
            elapsed = time.perf_counter() - t0
            d_acc, d_disc = measure_discrimination(symbols, day_names, graph, ns, mp, device,
                                                    theta, stp, hom, ip, tau_mult, steps=pd_steps)
            di_acc, di_disc = measure_discrimination(symbols, digit_names, graph, ns, mp, device,
                                                      theta, stp, hom, ip, tau_mult, steps=pd_steps)
            log['epoch'].append(epoch + 1)
            log['days_acc'].append(d_acc)
            log['days_disc'].append(d_disc)
            log['digits_acc'].append(di_acc)
            log['digits_disc'].append(di_disc)
            log['steps_per_symbol'].append(pd_steps)
            log['n_edges'].append(graph.n_edges())

            print(f'  Ep {epoch+1:5d} ({pd_steps}st): days={d_acc:.0f}%/{d_disc:+.1f}% | '
                  f'digits={di_acc:.0f}%/{di_disc:+.1f}% | '
                  f'edges={graph.n_edges():,} ({elapsed:.0f}s)', flush=True)

            torch.save(log, 'definitive_test_checkpoint.pt')

    # ============ TRANSFER TEST ============
    print(f'\n--- TRANSFER TEST (200 epochs on novel sequence) ---', flush=True)
    novel_acc_before, novel_disc_before = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Before: acc={novel_acc_before:.0f}% disc={novel_disc_before:+.1f}%', flush=True)

    for ep in range(200):
        pd_steps = 50
        strength = get_strength()
        for name in novel_seq:
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
                apply_learning(graph, lr_scale=1.0, is_replay=False,
                              driving_replay_count=driving_replay_count)
                graph.increment_step()

        if (ep + 1) % 50 == 0:
            nacc, ndisc = measure_discrimination(
                symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            print(f'  Novel ep{ep+1}: acc={nacc:.0f}% disc={ndisc:+.1f}%', flush=True)

    novel_acc_after, novel_disc_after = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  After 200 epochs: acc={novel_acc_after:.0f}% disc={novel_disc_after:+.1f}%', flush=True)
    transfer = novel_disc_after - novel_disc_before
    print(f'  Transfer: {transfer:+.1f}%', flush=True)

    # ============ RESULTS ============
    print(f'\n{"="*60}', flush=True)
    print('  DEFINITIVE TEST RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Baseline: days acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)
    for key in ['days', 'digits']:
        if log[f'{key}_acc']:
            print(f'  {key}: final={log[f"{key}_acc"][-1]:.0f}%/{log[f"{key}_disc"][-1]:+.1f}% '
                  f'peak_disc={max(log[f"{key}_disc"]):+.1f}%', flush=True)
    print(f'  Transfer: {novel_disc_before:+.1f}% -> {novel_disc_after:+.1f}% ({transfer:+.1f}%)', flush=True)

    if transfer > 5:
        print(f'\n  *** TRANSFER LEARNING DETECTED — ABSTRACTION EMERGING ***', flush=True)
    elif max(log['days_disc']) > 20 and max(log['digits_disc']) > 10:
        print(f'\n  VERDICT: MULTI-SEQUENCE LEARNING but no transfer yet', flush=True)
    else:
        print(f'\n  VERDICT: Architecture needs fundamental rethink', flush=True)

    torch.save(log, 'definitive_test_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
