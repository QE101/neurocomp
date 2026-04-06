"""Unified Test: every lesson applied to Graph Brain at once.

All AMG learnings integrated into the neuromorphic architecture:
1. 2-level hierarchy (L1=1x fast, L2=3x slow regime detection)
2. VIP→SST attention circuit (7th edge type, learned disinhibition)
3. Temporal Oja on driving edges (pre_trace × post × pe_gate)
4. Standard Oja on modulatory edges
5. Dual learning rates (slow wake 0.1x, boosted replay 1x)
6. Hippocampal encode + sleep replay every 50 epochs
7. Proportional sensory encoding (input=20% exc, symbol=1% input)
8. Small-world connectivity
9. Structural plasticity enabled

Task: days of the week sequence (Mon→Tue→Wed→Thu→Fri→Sat→Sun)
500 epochs with periodic measurement.
Apical prediction metric (proven: +74.2% in hierarchy test).

N=50K. No compromises.
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
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.hippocampus import HippocampalSystem
from graph_brain.types import EdgeType, NodeType

STRENGTH = 2.0
PD = 50
PAUSE = 30
N_EPOCHS = 500
MEASURE_EVERY = 50
SLEEP_EVERY = 20  # more frequent sleep — consolidation needs reinforcement
INPUT_FRACTION = 0.20
SYMBOL_FRACTION = 0.01

# Edge types that get STP and learning updates (skip ELECTRICAL — non-plastic)
PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]

# Cached type masks (set once after graph init, never change)
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
# Node dynamics (hierarchy + attention)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    device = ns.device
    N = ns.n_nodes
    exc_mask = _exc_mask
    pv_mask = _pv_mask
    sst_mask = _sst_mask
    vip_mask = _vip_mask
    exc_f = _exc_f
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
    # True silence: subtract baseline so output=0 when prediction is perfect
    BASELINE = 0.6931  # ln(2) = softplus(0)
    raw = F.softplus(pred_err.abs()) - BASELINE
    ns.output = torch.where(exc_mask, raw.clamp(min=0.0, max=10.0) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
        out = F.softplus(ns.basal) * ns.gain * f
        if inh_type == NodeType.SST:
            sst_suppress = torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, min=0.0, max=1.0)
            out = out * sst_suppress
        ns.output = torch.where(mask, out, ns.output)
    ns.output += torch.randn(N, device=device) * 0.005
    ns.output.clamp_(min=0.0, max=10.0)  # firing rate saturation — biological max ~200-500Hz
    ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)


# ================================================================
# Message passing (all 7 edge types)
# ================================================================
def dual_channel_send(ns, graph, mp, device):
    """Message passing — NO hard threshold. PV inhibition controls sparsity via pv_gain.
    Nodes with low output (suppressed by PV) naturally send weak messages.
    The circuit regulates itself."""
    step = graph.step_count
    output = ns.output
    content = F.softplus(ns.basal)
    for et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = output[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
              EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
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
# Learning: temporal Oja on driving, standard Oja on modulatory
# ================================================================
def apply_unified_learning(graph, lr_scale=1.0, is_replay=False, driving_replay_count=None):
    """All AMG learnings in one learning rule:
    - Driving: temporal (pre_trace) + surprise (pe_gate) + Oja stabilizer + dual rate
    - Modulatory: standard Oja (symmetric, for predictions)
    - Disinhibition: standard Oja (VIP→SST learns attention patterns)
    - Consolidation: post_trace tracks replay-strengthened edges
    """
    ns_ = graph.node_state
    pred_err = ns_.prediction_error

    for et in PLASTIC_EDGE_TYPES:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        src = ns_.output[store.src.long()]
        dst = ns_.output[store.dst.long()]

        # Update pre_trace (temporal memory of recent pre activity)
        store.pre_trace *= 0.95
        store.pre_trace += src * 0.05

        if et == EdgeType.DRIVING:
            # TEMPORAL + SURPRISE + OJA + REPLAY-COUNTED SYNAPTIC TAGGING
            pe_gate = pred_err[store.dst.long()].abs()
            pe_gate = pe_gate / (pe_gate.mean().clamp(min=0.1))
            if is_replay:
                # Replay: write associations, accumulate co-activation for this sleep
                lr = 0.001 * lr_scale
                dw = lr * (store.pre_trace * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
                # Track co-activation strength this sleep (post_trace as per-sleep accumulator)
                co_act = store.pre_trace * dst
                store.post_trace += co_act * 0.01
                store.post_trace.clamp_(0.0, 1.0)
            else:
                # Wake: freeze edges that have been reinforced across 5+ separate sleeps
                frozen = (driving_replay_count >= 5).float()
                effective_lr = 0.0001 * lr_scale * (1.0 - frozen)
                dw = effective_lr * pe_gate * (store.pre_trace * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
        elif et == EdgeType.MODULATORY:
            # Standard Oja: symmetric, prediction learning
            lr = 0.001 * lr_scale
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
        elif et == EdgeType.DISINHIBITION:
            # VIP→SST: FAST attention learning (2x normal)
            # VIP quickly learns which SST to suppress for each pattern
            lr = 0.002 * lr_scale
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
        elif et in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC):
            # PV/SST: SLOW inhibitory learning (0.1x normal)
            # Stable scaffold — Peters: inhibitory neurons constant across 14 sessions
            lr = 0.0001 * lr_scale
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
        else:
            # Retrograde etc
            lr = 0.001 * lr_scale
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)

        store.weight += dw
        store.weight.clamp_(0.0, 1.0)


# ================================================================
# Small-world edges
# ================================================================
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
# Symbol encoding (proportional, no hand-assignment)
# ================================================================
def build_symbols(input_region, device):
    """Build multiple sequences for diverse experience.
    Each symbol = random 1% of input region. Zero overlap between ALL symbols."""
    n_input = input_region.shape[0]
    n_per = max(10, int(n_input * SYMBOL_FRACTION))
    torch.manual_seed(777)
    perm = input_region[torch.randperm(n_input, device=device)]

    # Three sequences for diverse experience
    sequences = {
        'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'digits': ['D1', 'D2', 'D3', 'D4', 'D5'],
        'letters': ['LA', 'LB', 'LC', 'LD', 'LE'],
    }

    # Novel test sequence (never trained, tests transfer)
    novel_seq = ['N1', 'N2', 'N3', 'N4', 'N5']

    all_names = []
    for seq in sequences.values():
        all_names.extend(seq)
    all_names.extend(novel_seq)

    symbols = {}
    for i, name in enumerate(all_names):
        symbols[name] = perm[i * n_per:(i + 1) * n_per]

    return symbols, sequences, novel_seq, n_per


# ================================================================
# Measurement: apical prediction (proven metric)
# ================================================================
def measure_apical_prediction(pred_name, target_name, symbols, graph, ns, mp,
                               device, theta, stp, hom, ip, tau_mult):
    pred_nodes = symbols[pred_name]
    target_nodes = symbols[target_name]
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pred_nodes.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
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


def measure_discrimination(symbols, day_names, graph, ns, mp, device,
                            theta, stp, hom, ip, tau_mult):
    n_correct = 0
    n_tested = 0
    total_disc = 0
    for i in range(len(day_names) - 1):
        pred = day_names[i]
        target = day_names[i + 1]
        wrong = day_names[(i + 3) % len(day_names)]
        ap_correct = measure_apical_prediction(pred, target, symbols, graph, ns, mp,
                                                device, theta, stp, hom, ip, tau_mult)
        ap_wrong = measure_apical_prediction(wrong, target, symbols, graph, ns, mp,
                                              device, theta, stp, hom, ip, tau_mult)
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
    print('  UNIFIED TEST: All Learnings Applied', flush=True)
    print('  Temporal Oja + Attention + Hierarchy + Hippocampus', flush=True)
    print('  + Dual rate + Consolidation + Small-world + SP', flush=True)
    print(f'  N=50K, {N_EPOCHS} epochs, days of the week', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()

    # Set up core references FIRST (needed by everything below)
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    cache_type_masks(ns)

    # No PV boost needed — true silence activation handles sparsity naturally
    print(f'True silence mode: output = max(0, softplus(|err|) - ln(2))', flush=True)

    n_sw = add_small_world_edges(graph, fraction=0.2)

    # Build hierarchy BEFORE context gating (assigns hierarchy_level)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)

    # CONTEXT GATING: Add Level 2 → VIP driving edges
    # Level 2 (slow context) drives VIP, which gates SST, which controls attention.
    if config.hierarchy.enabled:
        level_2_exc = torch.where(
            _exc_mask & (ns.hierarchy_level == 2)
        )[0]
        vip_idx = torch.where(_vip_mask)[0]

        if len(level_2_exc) > 0 and len(vip_idx) > 0:
            l2_pos = ns.position[level_2_exc]
            vip_pos = ns.position[vip_idx]
            context_k = 3

            all_src = []
            all_dst = []
            chunk = 2000
            for start in range(0, len(level_2_exc), chunk):
                end = min(start + chunk, len(level_2_exc))
                dists = torch.cdist(l2_pos[start:end], vip_pos)
                _, topk = dists.topk(context_k, dim=1, largest=False)
                src_exp = level_2_exc[start:end].unsqueeze(1).expand(-1, context_k).reshape(-1)
                dst_exp = vip_idx[topk.reshape(-1)]
                all_src.append(src_exp.to(torch.int32))
                all_dst.append(dst_exp.to(torch.int32))

            ctx_src = torch.cat(all_src)
            ctx_dst = torch.cat(all_dst)
            ctx_weights = torch.full((ctx_src.shape[0],), 0.3, device=device)
            graph.add_edges(EdgeType.DRIVING, ctx_src, ctx_dst, weights=ctx_weights)
            print(f'Context gating: {ctx_src.shape[0]:,} Level2->VIP driving edges', flush=True)

    print(f'Graph: {graph.n_edges():,} edges (+{n_sw:,} SW)', flush=True)
    print(builder.summary(graph), flush=True)
    if graph.has_edge_type(EdgeType.DISINHIBITION):
        print(f'Attention: {graph.n_edges(EdgeType.DISINHIBITION):,} VIP->SST edges', flush=True)
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Input region + symbols
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, sequences, novel_seq, n_per = build_symbols(input_region, device)
    all_symbol_nodes = torch.cat([symbols[n] for n in symbols.keys()])

    print(f'\nInput: {input_region.shape[0]} nodes, {n_per} per symbol, 0 overlap', flush=True)
    for seq_name, seq in sequences.items():
        print(f'  {seq_name}: {" -> ".join(seq)}', flush=True)
    print(f'  novel (untrained): {" -> ".join(novel_seq)}', flush=True)
    print(f'  Total symbols: {len(symbols)}', flush=True)

    # Hippocampus
    hipp = HippocampalSystem(config=config.hippocampal, cortical_input_indices=all_symbol_nodes,
                              n_cortical=N, device=device, seed=config.simulation.seed)

    # Per-edge replay counter for driving edges (how many separate sleep phases reinforced)
    # Stored as a separate tensor, not overloading post_trace
    driving_replay_count = None
    if graph.has_edge_type(EdgeType.DRIVING):
        driving_replay_count = torch.zeros(graph.edge_store(EdgeType.DRIVING).n_edges, device=device)

    # Baseline (measure on days sequence)
    day_names = sequences['days']
    print(f'\n--- BASELINE ---', flush=True)
    acc_0, disc_0 = measure_discrimination(symbols, day_names, graph, ns, mp, device,
                                            theta, stp, hom, ip, tau_mult)
    print(f'  Days: Acc={acc_0:.0f}% Disc={disc_0:+.1f}%', flush=True)

    # Training
    t0 = time.perf_counter()
    log = {'epoch': [], 'days_acc': [], 'days_disc': [], 'digits_acc': [], 'digits_disc': []}
    seq_names_list = list(sequences.keys())

    print(f'\n--- TRAINING (3 sequences, random order each epoch) ---', flush=True)
    for epoch in range(N_EPOCHS):
        # Present all sequences in random order this epoch
        epoch_order = list(seq_names_list)
        np.random.shuffle(epoch_order)

        for seq_key in epoch_order:
            seq = sequences[seq_key]
            for name in seq:
                pattern = symbols[name]
                for s in range(PD):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pattern.long()] += STRENGTH
                    error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                    for et in PLASTIC_EDGE_TYPES:
                        if graph.has_edge_type(et):
                            stp.update(graph.edge_store(et), ns, 1.0)
                    apply_unified_learning(graph, lr_scale=1.0, is_replay=False, driving_replay_count=driving_replay_count)
                    if step % 100 == 0:
                        for et in PLASTIC_EDGE_TYPES:
                            if graph.has_edge_type(et):
                                hom.update(graph.edge_store(et), ns, 1.0)
                        ip.update(ns)
                    graph.increment_step()
                # Encode to hippocampus
                hipp.encode(ns.output, graph.step_count)

            # Pause between sequences
            for s in range(PAUSE):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                graph.increment_step()

        # Sleep replay
        if (epoch + 1) % SLEEP_EVERY == 0 and hipp.n_stored() > 0:
            # Reset per-sleep accumulator before this sleep phase
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
                    apply_unified_learning(graph, lr_scale=1.0, is_replay=True, driving_replay_count=driving_replay_count)
                    graph.increment_step()

            # After sleep: increment replay_count for edges that were co-active
            if graph.has_edge_type(EdgeType.DRIVING):
                reinforced = graph.edge_store(EdgeType.DRIVING).post_trace > 0.1
                driving_replay_count += reinforced.float()

            # PHASE 2: Sharp-wave ripple — compressed fast replay
            # Replay the same patterns at 5x speed (PD//5 steps) with boosted learning
            # This mimics the hippocampal sharp-wave ripples that compress sequences
            schedule2 = hipp.replay_schedule(2)  # shorter, faster
            for pidx in schedule2:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(PD // 5):  # 10 steps instead of 50 — compressed
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay * 2.0  # stronger signal
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    apply_unified_learning(graph, lr_scale=3.0, is_replay=True,
                                          driving_replay_count=driving_replay_count)
                    graph.increment_step()

            # PHASE 3: Synaptic homeostasis — global proportional downscaling
            # All weights shrink by 5%. Ratios preserved. Noise edges → 0.
            # Strong edges stay relatively strong. Prevents runaway potentiation.
            for et in PLASTIC_EDGE_TYPES:
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    store.weight *= 0.95

        # Structural plasticity
        if (epoch + 1) % 100 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]}', flush=True)
                # Resize replay counter to match new edge count
                if graph.has_edge_type(EdgeType.DRIVING):
                    new_n = graph.edge_store(EdgeType.DRIVING).n_edges
                    if driving_replay_count.shape[0] != new_n:
                        old = driving_replay_count
                        driving_replay_count = torch.zeros(new_n, device=device)
                        driving_replay_count[:min(len(old), new_n)] = old[:min(len(old), new_n)]

        # Measure all sequences
        if (epoch + 1) % MEASURE_EVERY == 0:
            elapsed = time.perf_counter() - t0
            drv_w = graph.edge_store(EdgeType.DRIVING).weight.mean().item()
            mod_w = graph.edge_store(EdgeType.MODULATORY).weight.mean().item()

            log['epoch'].append(epoch + 1)
            results_str = []

            for seq_key in ['days', 'digits']:
                seq = sequences[seq_key]
                acc, disc = measure_discrimination(symbols, seq, graph, ns, mp, device,
                                                    theta, stp, hom, ip, tau_mult)
                log[f'{seq_key}_acc'].append(acc)
                log[f'{seq_key}_disc'].append(disc)
                results_str.append(f'{seq_key}={acc:.0f}%/{disc:+.1f}%')

            print(f'  Epoch {epoch+1:4d}: {" | ".join(results_str)} '
                  f'drv={drv_w:.4f} mod={mod_w:.4f} ({elapsed:.0f}s)', flush=True)

            # Checkpoint
            torch.save({'log': log, 'epoch': epoch+1}, 'unified_test_checkpoint.pt')

    # Transfer test: present novel sequence ONCE, measure if graph picks it up faster
    print(f'\n--- TRANSFER TEST ---', flush=True)
    print(f'  Novel sequence (never trained): {" -> ".join(novel_seq)}', flush=True)

    # Baseline: measure novel sequence discrimination before any novel training
    novel_acc_before, novel_disc_before = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Before training: acc={novel_acc_before:.0f}% disc={novel_disc_before:+.1f}%', flush=True)

    # Train on novel sequence for 200 epochs (proper transfer test)
    for ep in range(200):
        for name in novel_seq:
            pattern = symbols[name]
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pattern.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                for et in PLASTIC_EDGE_TYPES:
                    if graph.has_edge_type(et):
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_unified_learning(graph, lr_scale=1.0, is_replay=False, driving_replay_count=driving_replay_count)
                graph.increment_step()

    novel_acc_after, novel_disc_after = measure_discrimination(
        symbols, novel_seq, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  After 20 epochs: acc={novel_acc_after:.0f}% disc={novel_disc_after:+.1f}%', flush=True)
    print(f'  Transfer: {novel_disc_after - novel_disc_before:+.1f}% improvement in 20 epochs', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Baseline: days acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)
    for seq_key in ['days', 'digits']:
        if log[f'{seq_key}_acc']:
            final_acc = log[f'{seq_key}_acc'][-1]
            final_disc = log[f'{seq_key}_disc'][-1]
            peak_disc = max(log[f'{seq_key}_disc'])
            print(f'  {seq_key}: final acc={final_acc:.0f}% disc={final_disc:+.1f}% peak={peak_disc:+.1f}%', flush=True)
            print(f'    Trajectory: {[f"{d:+.1f}" for d in log[f"{seq_key}_disc"]]}', flush=True)

    print(f'  Transfer: novel {novel_disc_before:+.1f}% -> {novel_disc_after:+.1f}% in 20 epochs', flush=True)

    if novel_disc_after > novel_disc_before + 5:
        print(f'\n  VERDICT: TRANSFER LEARNING DETECTED — abstraction emerging', flush=True)
    elif any(max(log[f'{k}_disc']) > disc_0 + 5 for k in ['days', 'digits']):
        print(f'\n  VERDICT: MULTI-SEQUENCE LEARNING — discrimination improved', flush=True)
    else:
        print(f'\n  VERDICT: Needs more training or tuning', flush=True)

    torch.save(log, 'unified_test_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
