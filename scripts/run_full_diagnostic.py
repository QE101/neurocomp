"""Full Diagnostic Run: every metric, every epoch, full node vectors.

This is the ONE thorough run. Captures everything — node states, edge stats,
critical edge tracking, sleep effects, inhibitory dynamics, structural changes.
No assumptions about what matters. Let the data speak.

Output: one fat pickle (~3-5GB) with 1000 epoch snapshots.
Full output vectors every epoch. Summary stats per edge type.
Critical edge tracking for sequence-relevant connections.

Same architecture as run_production.py (the validated +5.0% transfer config).
"""

import sys, os, time, math, gc
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import pickle
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
SLEEP_EVERY = 20
INPUT_FRACTION = 0.20
SYMBOL_SPARSITY = 0.10
PAUSE = 20
WEIGHT_DECAY = 0.013

def get_presentation_steps(epoch):
    if epoch < 300: return 100
    elif epoch < 700: return 50
    else: return 30

def get_strength():
    return 1.0 + np.random.random() * 2.0

PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]

OUTPUT_EDGE_CHANNELS = {
    EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
    EdgeType.DISINHIBITION: Channel.VIP_INHIBITION, EdgeType.RETROGRADE: Channel.RETROGRADE,
}
CONTENT_EDGE_CHANNELS = {
    EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION,
}

_exc_mask = _pv_mask = _sst_mask = _vip_mask = _exc_f = None
_edge_cache = {}
_noise_buf = None
_noise_idx = 0

def cache_type_masks(ns):
    global _exc_mask, _pv_mask, _sst_mask, _vip_mask, _exc_f
    _exc_mask = ns.type_mask(NodeType.EXCITATORY)
    _pv_mask = ns.type_mask(NodeType.PV)
    _sst_mask = ns.type_mask(NodeType.SST)
    _vip_mask = ns.type_mask(NodeType.VIP)
    _exc_f = _exc_mask.float()

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
    _noise_buf = torch.randn(100, N, device=device) * 0.005

def get_noise(N):
    global _noise_idx
    noise = _noise_buf[_noise_idx % 100]
    _noise_idx += 1
    return noise


def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    device = ns.device
    N = ns.n_nodes
    exc_mask, pv_mask, sst_mask, vip_mask, exc_f = _exc_mask, _pv_mask, _sst_mask, _vip_mask, _exc_f
    if tau_mult is not None:
        basal_tau, apical_tau, input_norm = 10.0*tau_mult, 20.0*tau_mult, 1.0/tau_mult
    else:
        basal_tau, apical_tau, input_norm = 10.0, 20.0, 1.0
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
            out = out * torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, min=0.0, max=1.0)
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
        store.pre_trace.lerp_(src, 0.05)
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
    return symbols, sequences, novel_seq, n_on


def find_critical_edges(graph, symbols, sequences):
    """Identify edges connecting consecutive symbols in each sequence.
    Vectorised — no Python loops over edges."""
    critical = {}
    device = next(iter(symbols.values())).device

    for seq_name, seq in sequences.items():
        for i in range(len(seq) - 1):
            src_name, dst_name = seq[i], seq[i + 1]
            pair_key = f'{src_name}->{dst_name}'

            # Build boolean node masks (GPU tensor, O(N) not O(E))
            N = graph.n_nodes
            src_mask = torch.zeros(N, dtype=torch.bool, device=device)
            dst_mask = torch.zeros(N, dtype=torch.bool, device=device)
            src_mask[symbols[src_name].long()] = True
            dst_mask[symbols[dst_name].long()] = True

            for et in [EdgeType.DRIVING, EdgeType.MODULATORY]:
                if et not in _edge_cache: continue
                cache = _edge_cache[et]
                # Vectorised: index into boolean masks with edge src/dst
                both = src_mask[cache['src64']] & dst_mask[cache['dst64']]
                n_critical = both.sum().item()
                if n_critical > 0:
                    critical[f'{pair_key}_{et.name}'] = both
    return critical


def capture_snapshot(epoch, graph, ns, symbols, sequences, critical_edges,
                     exc_idx, pv_idx, sst_idx, vip_idx,
                     driving_replay_count,
                     pre_sleep_weights=None, post_sleep_weights=None):
    """READ-ONLY snapshot. Zero simulation steps. Zero perturbation.

    Node state: reads current output/apical/pred_err (whatever training left them as)
    Edge state: reads weights, traces, STP state
    Representation: static weight analysis — where do symbol nodes' edges point?
    """
    snap = {'epoch': epoch}

    # --- Current node state (as-is from training, no extra stimulation) ---
    snap['exc_mean_out'] = ns.output[exc_idx].mean().item()
    snap['exc_std_out'] = ns.output[exc_idx].std().item()
    snap['exc_active_pct'] = (ns.output[exc_idx] > 0).float().mean().item()
    snap['exc_mean_basal'] = ns.basal[exc_idx].mean().item()
    snap['exc_mean_apical'] = ns.apical[exc_idx].mean().item()
    snap['exc_mean_pred_err'] = ns.prediction_error[exc_idx].abs().mean().item()

    # Inhibitory state
    snap['pv_mean_out'] = ns.output[pv_idx].mean().item()
    snap['sst_mean_out'] = ns.output[sst_idx].mean().item()
    snap['vip_mean_out'] = ns.output[vip_idx].mean().item()
    snap['pv_std_out'] = ns.output[pv_idx].std().item()
    snap['sst_std_out'] = ns.output[sst_idx].std().item()
    snap['vip_std_out'] = ns.output[vip_idx].std().item()

    # Level 1 vs Level 2
    l1_mask = _exc_mask & (ns.hierarchy_level == 1)
    l2_mask = _exc_mask & (ns.hierarchy_level == 2)
    snap['l1_mean_out'] = ns.output[l1_mask].mean().item()
    snap['l2_mean_out'] = ns.output[l2_mask].mean().item()
    snap['l1_mean_ema'] = ns.activity_ema[l1_mask].mean().item()
    snap['l2_mean_ema'] = ns.activity_ema[l2_mask].mean().item()

    # Gain
    snap['gain_mean'] = ns.gain[exc_idx].mean().item()
    snap['gain_std'] = ns.gain[exc_idx].std().item()

    # --- Static weight analysis per symbol (zero simulation) ---
    N = graph.n_nodes
    for sym_name, pattern in symbols.items():
        sym_mask = torch.zeros(N, dtype=torch.bool, device=pattern.device)
        sym_mask[pattern.long()] = True

        for et in [EdgeType.DRIVING, EdgeType.MODULATORY]:
            if et not in _edge_cache: continue
            cache = _edge_cache[et]
            store = graph.edge_store(et)
            ename = et.name.lower()

            # Outgoing weight profile: where does this symbol's signal go?
            from_sym = sym_mask[cache['src64']]
            if from_sym.sum() > 0:
                out_w = store.weight[from_sym]
                snap[f'{sym_name}_{ename}_out_w_mean'] = out_w.mean().item()
                snap[f'{sym_name}_{ename}_out_w_sum'] = out_w.sum().item()

            # Incoming weight profile: what predicts this symbol?
            to_sym = sym_mask[cache['dst64']]
            if to_sym.sum() > 0:
                in_w = store.weight[to_sym]
                snap[f'{sym_name}_{ename}_in_w_mean'] = in_w.mean().item()
                snap[f'{sym_name}_{ename}_in_w_sum'] = in_w.sum().item()

    # --- Pairwise weight similarity (static, no simulation) ---
    train_syms = []
    for seq in sequences.values():
        train_syms.extend(seq)
    for i in range(len(train_syms)):
        for j in range(i+1, len(train_syms)):
            # Compare outgoing driving weight profiles
            key_i = f'{train_syms[i]}_driving_out_w_mean'
            key_j = f'{train_syms[j]}_driving_out_w_mean'
            if key_i in snap and key_j in snap:
                snap[f'wsim_{train_syms[i]}_{train_syms[j]}'] = abs(snap[key_i] - snap[key_j])

    # --- Edge stats per type ---
    for et in PLASTIC_EDGE_TYPES:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        w = store.weight
        ename = et.name.lower()
        snap[f'{ename}_w_mean'] = w.mean().item()
        snap[f'{ename}_w_std'] = w.std().item()
        snap[f'{ename}_w_p5'] = w.quantile(0.05).item()
        snap[f'{ename}_w_p25'] = w.quantile(0.25).item()
        snap[f'{ename}_w_p50'] = w.quantile(0.50).item()
        snap[f'{ename}_w_p75'] = w.quantile(0.75).item()
        snap[f'{ename}_w_p95'] = w.quantile(0.95).item()
        snap[f'{ename}_release_mean'] = store.release_prob.mean().item()
        snap[f'{ename}_pre_trace_mean'] = store.pre_trace.mean().item()
        snap[f'{ename}_n_edges'] = store.n_edges

        if et == EdgeType.DRIVING:
            snap['driving_post_trace_mean'] = store.post_trace.mean().item()
            if driving_replay_count is not None:
                snap['driving_n_frozen'] = (driving_replay_count >= 5).sum().item()
                snap['driving_replay_count_mean'] = driving_replay_count.mean().item()
                snap['driving_replay_count_max'] = driving_replay_count.max().item()

    # --- Critical edge tracking ---
    for pair_key, mask in critical_edges.items():
        if mask.sum() == 0: continue
        et_name = pair_key.split('_')[-1]
        et = EdgeType[et_name]
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        crit_w = store.weight[mask]
        snap[f'crit_{pair_key}_w_mean'] = crit_w.mean().item()
        snap[f'crit_{pair_key}_w_std'] = crit_w.std().item()
        snap[f'crit_{pair_key}_n'] = mask.sum().item()
        # Oja force on critical edges (using current dst output)
        if et in _edge_cache:
            cache = _edge_cache[et]
            dst_out = ns.output[cache['dst64']]
            oja_force = (dst_out * dst_out * store.weight)[mask]
            snap[f'crit_{pair_key}_oja_mean'] = oja_force.mean().item()

    # --- Sleep effects ---
    if pre_sleep_weights is not None and post_sleep_weights is not None:
        for et_name in pre_sleep_weights:
            pre = pre_sleep_weights[et_name]
            post = post_sleep_weights[et_name]
            snap[f'sleep_{et_name}_change_mean'] = (post - pre).mean().item()
            snap[f'sleep_{et_name}_change_std'] = (post - pre).std().item()
            snap[f'sleep_{et_name}_ratio'] = (post / pre.clamp(min=1e-8)).mean().item()

    # --- Total edges ---
    snap['total_edges'] = graph.n_edges()

    return snap


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


def main():
    print('=' * 60, flush=True)
    print('  FULL DIAGNOSTIC RUN', flush=True)
    print('  Every metric, every epoch, full node vectors', flush=True)
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

    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)

    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    precompute_edge_data(graph, mp)
    init_noise_buffer(N, device)

    print(f'Graph: {graph.n_edges():,} edges', flush=True)
    print(builder.summary(graph), flush=True)

    # Symbols
    exc_idx = torch.where(_exc_mask)[0]
    pv_idx = torch.where(_pv_mask)[0]
    sst_idx = torch.where(_sst_mask)[0]
    vip_idx = torch.where(_vip_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, sequences, novel_seq, n_on = build_sensory_symbols(input_region, device)
    all_symbol_nodes = input_region

    hipp = HippocampalSystem(config=config.hippocampal, cortical_input_indices=all_symbol_nodes,
                              n_cortical=N, device=device, seed=config.simulation.seed)

    driving_replay_count = None
    if graph.has_edge_type(EdgeType.DRIVING):
        driving_replay_count = torch.zeros(graph.edge_store(EdgeType.DRIVING).n_edges, device=device)

    # Find critical edges
    critical_edges = find_critical_edges(graph, symbols, sequences)
    print(f'Critical edge sets: {len(critical_edges)}', flush=True)
    for k, v in critical_edges.items():
        print(f'  {k}: {v.sum().item()} edges', flush=True)

    # Check for resume
    all_snapshots = []
    start_epoch = 0
    resume_path = 'D:/neurocomp/full_diagnostic_resume.pt'
    if os.path.exists(resume_path):
        print(f'\n--- RESUMING from checkpoint ---', flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        start_epoch = ckpt['epoch']
        graph._step_count = ckpt['step_count']

        # Restore node state
        for key, val in ckpt['node_state'].items():
            getattr(ns, key).copy_(val.to(device))

        # Restore edge state
        for et_name, state in ckpt['graph_state'].items():
            et = EdgeType[et_name]
            if graph.has_edge_type(et):
                store = graph.edge_store(et)
                store.weight.copy_(state['weight'].to(device))
                store.release_prob.copy_(state['release_prob'].to(device))
                store.pre_trace.copy_(state['pre_trace'].to(device))
                store.post_trace.copy_(state['post_trace'].to(device))

        # Restore replay counter
        if ckpt['driving_replay_count'] is not None:
            driving_replay_count = ckpt['driving_replay_count'].to(device)

        # Load existing snapshots
        snap_path = f'D:/neurocomp/full_diagnostic_snapshots_ep{start_epoch}.pkl'
        if os.path.exists(snap_path):
            with open(snap_path, 'rb') as f:
                all_snapshots = pickle.load(f)

        # Refresh edge cache after restore
        precompute_edge_data(graph, mp)
        critical_edges = find_critical_edges(graph, symbols, sequences)

        print(f'  Resumed at epoch {start_epoch}, {len(all_snapshots)} snapshots loaded', flush=True)
    else:
        print(f'\n--- FRESH START ---', flush=True)

    seq_keys = list(sequences.keys())
    t0 = time.perf_counter()

    for epoch in range(start_epoch, N_EPOCHS):
        pd_steps = get_presentation_steps(epoch)
        pre_sleep_weights = None
        post_sleep_weights = None

        # Training
        epoch_order = list(seq_keys)
        np.random.shuffle(epoch_order)
        for seq_key in epoch_order:
            seq = sequences[seq_key]
            use_gap = epoch > 200 and np.random.random() < 0.1
            gap_idx = np.random.randint(1, len(seq) - 1) if use_gap else -1
            for si, name in enumerate(seq):
                if si == gap_idx:
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
                        ns.gain.clamp_(min=0.5)  # gain floor — prevents Oja stabilizer collapse
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
            # Capture pre-sleep weights
            pre_sleep_weights = {}
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    pre_sleep_weights[et.name] = graph.edge_store(et).weight.clone()

            if graph.has_edge_type(EdgeType.DRIVING):
                graph.edge_store(EdgeType.DRIVING).post_trace.zero_()
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
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    graph.edge_store(et).weight *= 0.95
            if graph.has_edge_type(EdgeType.DRIVING):
                reinforced = graph.edge_store(EdgeType.DRIVING).post_trace > 0.5
                driving_replay_count += reinforced.float()

            # Capture post-sleep weights
            post_sleep_weights = {}
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    post_sleep_weights[et.name] = graph.edge_store(et).weight.clone()

        # SP
        if (epoch + 1) % 200 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]}', flush=True)
                precompute_edge_data(graph, mp)
                critical_edges = find_critical_edges(graph, symbols, sequences)
                if graph.has_edge_type(EdgeType.DRIVING):
                    new_n = graph.edge_store(EdgeType.DRIVING).n_edges
                    if driving_replay_count.shape[0] != new_n:
                        old = driving_replay_count
                        driving_replay_count = torch.zeros(new_n, device=device)
                        driving_replay_count[:min(len(old), new_n)] = old[:min(len(old), new_n)]

        # SNAPSHOT every epoch — READ ONLY, zero simulation steps
        snap = capture_snapshot(
            epoch + 1, graph, ns, symbols, sequences, critical_edges,
            exc_idx, pv_idx, sst_idx, vip_idx,
            driving_replay_count,
            pre_sleep_weights, post_sleep_weights)
        snap['pd_steps'] = pd_steps
        all_snapshots.append(snap)

        elapsed = time.perf_counter() - t0
        if (epoch + 1) % 50 == 0:
            n_frozen = snap.get('driving_n_frozen', 0)
            act_pct = snap.get('exc_active_pct', 0)
            drv_w = snap.get('driving_w_mean', 0)
            mon_out = snap.get('Mon_driving_out_w_mean', 0)
            print(f'  Ep {epoch+1:5d} ({pd_steps}st): '
                  f'act={act_pct:.1%} frozen={n_frozen} drv={drv_w:.4f} '
                  f'Mon_drv={mon_out:.4f} edges={snap["total_edges"]:,} ({elapsed:.0f}s)', flush=True)

        # Save every 100 epochs — snapshots + full graph state for resume
        if (epoch + 1) % 100 == 0:
            # Snapshots
            snap_path = f'D:/neurocomp/full_diagnostic_snapshots_ep{epoch+1}.pkl'
            with open(snap_path, 'wb') as f:
                pickle.dump(all_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Full graph state for resume
            resume_path = 'D:/neurocomp/full_diagnostic_resume.pt'
            torch.save({
                'epoch': epoch + 1,
                'graph_state': {
                    et.name: {
                        'weight': graph.edge_store(et).weight.cpu(),
                        'release_prob': graph.edge_store(et).release_prob.cpu(),
                        'pre_trace': graph.edge_store(et).pre_trace.cpu(),
                        'post_trace': graph.edge_store(et).post_trace.cpu(),
                    } for et in EdgeType if graph.has_edge_type(et) and graph.edge_store(et).n_edges > 0
                },
                'node_state': {
                    'basal': ns.basal.cpu(),
                    'apical': ns.apical.cpu(),
                    'output': ns.output.cpu(),
                    'activity_ema': ns.activity_ema.cpu(),
                    'gain': ns.gain.cpu(),
                    'prediction_error': ns.prediction_error.cpu(),
                },
                'driving_replay_count': driving_replay_count.cpu() if driving_replay_count is not None else None,
                'step_count': graph.step_count,
                'n_snapshots': len(all_snapshots),
            }, resume_path)
            size_snap = os.path.getsize(snap_path) / 1e6
            size_resume = os.path.getsize(resume_path) / 1e6
            print(f'  Checkpoint ep{epoch+1}: snapshots={size_snap:.0f}MB, resume={size_resume:.0f}MB', flush=True)

    # Final save
    save_path = 'D:/neurocomp/full_diagnostic_complete.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(all_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f'\nSaved: {save_path} ({size_mb:.0f} MB, {len(all_snapshots)} snapshots)', flush=True)
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
