"""Sequence Prediction with Hierarchy: the temporal abstraction test.

2-level hierarchy on N=50K:
- Level 1 (bottom z-half): fast (tau_mult=1.0), learns sensory patterns
- Level 2 (top z-half): slow (tau_mult=3.0), learns sequence transitions

Bottom-up driving edges carry errors from L1 -> L2.
Top-down modulatory edges carry predictions from L2 -> L1.

When A is presented, Level 2 (slow) still holds A's representation.
It sends a prediction of B to Level 1. When B arrives, error is LOW.
When C arrives instead, error is HIGH.

A flat graph can't do this — all nodes have the same time constant.

Oja stabilizer + theta + small-world + structural plasticity + hippocampus.
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
N_SEQUENCES = 500
MEASURE_EVERY = 50

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
            'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'],
                          'target_types': ['PV'], 'constant_k': 5},
            'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 10},
            'max_radius': 0.5,
        },
        'structural': {'enabled': True, 'update_interval': 500, 'growth_rate': 0.1,
                        'prune_threshold': 0.01, 'edge_cost': 1e-5, 'max_degree': 5000},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {
        'enabled': True,
        'n_levels': 2,
        'split_axis': 2,        # z-axis (orthogonal to x-axis pattern regions)
        'time_scale_factor': 3.0,  # L1=1x, L2=3x (validated sweet spot)
        'inter_level_k': 5,     # gentle inter-level connectivity
        'inter_level_sigma': 0.5,
        'inter_level_init_weight': 0.02,  # whisper, not shout
    },
    'hippocampal': {
        'enabled': True, 'n_dg': 2000, 'n_ca3': 500,
        'dg_sparsity': 0.02, 'dg_fan_in': 2000, 'ca3_sparsity': 0.10,
        'encoding_lr': 0.5, 'replay_strength': 0.5, 'replay_lr_scale': 0.1,
        'max_patterns': 20, 'replay_interleave': 3, 'replay_steps': 30,
    },
}


# ================================================================
# Core dynamics (copy-pasted, DO NOT MODIFY except tau_multiplier)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()

    # Time constants: scaled by hierarchy level
    if tau_mult is not None:
        basal_tau = 10.0 * tau_mult
        apical_tau = 20.0 * tau_mult
    else:
        basal_tau = 10.0
        apical_tau = 20.0

    # Normalize input by tau_mult so equilibrium is the same across levels
    # (slower integration, NOT higher gain)
    if tau_mult is not None:
        input_norm = 1.0 / tau_mult  # Level 1: 1.0, Level 2: 1/3
    else:
        input_norm = 1.0
    ns.basal += 1.0 * (-ns.basal / basal_tau + inputs.basal * theta_mod * input_norm) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += 1.0 * (-ns.apical / apical_tau + inputs.apical * (1.0 - sst_gate) * input_norm) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(exc_mask, F.softplus(pred_err.abs()) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f  # inhibitory always fast
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


def apply_hebbian_oja(graph, lr_scale=1.0):
    ns_ = graph.node_state
    lr = 0.001 * lr_scale
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def add_small_world_edges(graph, fraction=0.2):
    ns = graph.node_state
    device = ns.position.device
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    n_exc = exc_idx.shape[0]
    n_existing = graph.n_edges(EdgeType.MODULATORY)
    n_long_range = int(n_existing * fraction)
    src_local = torch.randint(0, n_exc, (n_long_range,), device=device)
    dst_local = torch.randint(0, n_exc, (n_long_range,), device=device)
    valid = src_local != dst_local
    new_src = exc_idx[src_local[valid]]
    new_dst = exc_idx[dst_local[valid]]
    graph.add_edges(EdgeType.MODULATORY, new_src, new_dst,
                    weights=torch.full((new_src.shape[0],), 0.05, device=device))
    return new_src.shape[0]


def measure_transition_error(graph, ns, mp, device, theta, pat_from, pat_to,
                              stp, hom, ip, tau_mult):
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pat_from.long()] += STRENGTH
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

    errors = []
    for s in range(20):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pat_to.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        graph.increment_step()
        pe = ns.prediction_error[pat_to.long()].abs().mean().item()
        errors.append(pe)
    return float(np.mean(errors))


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  SEQUENCE PREDICTION WITH HIERARCHY', flush=True)
    print('  2-level PC: L1 fast (1x), L2 slow (3x)', flush=True)
    print('  N=50K, Oja + theta + small-world + hippocampus', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)

    # Build graph
    print('Building N=50K graph...', flush=True)
    t_build = time.perf_counter()
    graph = NeuromorphicGraph(config)
    graph.initialize()
    print(f'Built in {time.perf_counter()-t_build:.1f}s ({graph.n_edges():,} edges)', flush=True)

    # Add small-world edges
    n_sw = add_small_world_edges(graph, fraction=0.2)
    print(f'Small-world: +{n_sw:,} edges', flush=True)

    # Build hierarchy
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)
    print(builder.summary(graph), flush=True)
    print(f'Inter-level: {h_stats["ff_edges"]:,} FF + {h_stats["fb_edges"]:,} FB', flush=True)
    print(f'Total edges: {graph.n_edges():,}', flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Co-located patterns: same spatial region, different random subsets
    # At N=50K: 100 nodes out of ~8000 in input region = 1.25% activation
    # Near-zero overlap, no spatial confound
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(0.2)]  # bottom 20% by z = ~8000 nodes
    n_input = input_region.shape[0]

    torch.manual_seed(42)
    shuffled = input_region[torch.randperm(n_input)]
    pa = shuffled[:100]
    pb = shuffled[100:200]
    pc = shuffled[200:300]
    all_input = torch.cat([pa, pb, pc])
    sequence = [pa, pb, pc]

    # Verify near-zero overlap
    a_set = set(pa.cpu().tolist())
    b_set = set(pb.cpu().tolist())
    c_set = set(pc.cpu().tolist())
    ab_overlap = len(a_set & b_set)
    bc_overlap = len(b_set & c_set)
    ac_overlap = len(a_set & c_set)

    print(f'\nInput region: {n_input} nodes (bottom 20% z)', flush=True)
    print(f'Patterns: A={pa.shape[0]}, B={pb.shape[0]}, C={pc.shape[0]} ({100/n_input*100:.1f}% activation)', flush=True)
    print(f'Overlap: A&B={ab_overlap}, B&C={bc_overlap}, A&C={ac_overlap}', flush=True)

    # Hippocampus
    hipp = HippocampalSystem(config=config.hippocampal, cortical_input_indices=all_input,
                              n_cortical=N, device=device, seed=config.simulation.seed)

    # Clean metric: measure APICAL predictions at target nodes
    # during predecessor presentation (no target presented).
    # If system predicts B after A: apical_B rises during A.
    def measure_prediction_signal():
        """Measure apical prediction at next-pattern nodes during current pattern.

        Present A for PD steps, read apical at B-nodes (should be high if A→B learned).
        Present C for PD steps, read apical at B-nodes (should be low if A→B specific).

        Returns: apical_B_during_A, apical_B_during_C, discrimination.
        """
        # Present A, measure apical at B-nodes in last 10 steps
        ap_b_vals = []
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
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
            if s >= PD - 10:  # last 10 steps
                ap_b_vals.append(ns.apical[pb.long()].mean().item())
        ap_b_after_a = float(np.mean(ap_b_vals))

        # Flush (20 steps no input to reset)
        for s in range(20):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()

        # Present C, measure apical at B-nodes in last 10 steps
        ap_b_vals2 = []
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[pc.long()] += STRENGTH
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
            if s >= PD - 10:
                ap_b_vals2.append(ns.apical[pb.long()].mean().item())
        ap_b_after_c = float(np.mean(ap_b_vals2))

        # Discrimination: positive = system predicts B after A more than after C
        if ap_b_after_c > 1e-8:
            disc = (ap_b_after_a - ap_b_after_c) / ap_b_after_c * 100
        else:
            disc = 0.0

        return ap_b_after_a, ap_b_after_c, disc

    # Baseline: apical prediction signal
    print(f'\n--- BASELINE ---', flush=True)
    ap_ba_0, ap_bc_0, disc_0 = measure_prediction_signal()
    print(f'  Apical_B during A: {ap_ba_0:.4f}', flush=True)
    print(f'  Apical_B during C: {ap_bc_0:.4f}', flush=True)
    print(f'  Discrimination: {disc_0:+.1f}%', flush=True)

    # Training
    t0 = time.perf_counter()
    log = {'ap_ba': [ap_ba_0], 'ap_bc': [ap_bc_0], 'disc': [disc_0], 'seq': [0]}

    print(f'\n--- TRAINING: {N_SEQUENCES} sequences ---', flush=True)
    for seq_i in range(N_SEQUENCES):
        for pat in sequence:
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_hebbian_oja(graph, lr_scale=1.0)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
            hipp.encode(ns.output, graph.step_count)

        # Structural plasticity every 100 sequences
        if (seq_i + 1) % 100 == 0:
            sp_stats = sp.update(graph)
            print(f'  SP: grown={sp_stats["grown"]}, pruned={sp_stats["pruned"]}', flush=True)

        # Sleep replay every 50 sequences
        if (seq_i + 1) % 50 == 0 and hipp.n_stored() > 0:
            schedule = hipp.replay_schedule(config.hippocampal.replay_interleave)
            for pidx in schedule:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(config.hippocampal.replay_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_input.long()] += replay
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    apply_hebbian_oja(graph, lr_scale=config.hippocampal.replay_lr_scale)
                    graph.increment_step()

        # Measure apical prediction signal
        if (seq_i + 1) % MEASURE_EVERY == 0:
            ap_ba, ap_bc, disc = measure_prediction_signal()
            log['ap_ba'].append(ap_ba); log['ap_bc'].append(ap_bc)
            log['disc'].append(disc); log['seq'].append(seq_i + 1)

            elapsed = time.perf_counter() - t0
            trend = 'UP' if disc > disc_0 else 'down'
            print(f'  Seq {seq_i+1:4d}: ap_B|A={ap_ba:.4f} ap_B|C={ap_bc:.4f} disc={disc:+.1f}% ({trend}) ({elapsed:.0f}s)', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS (apical prediction metric)', flush=True)
    print(f'{"="*60}', flush=True)

    avg_disc = np.mean(log['disc'][-5:])
    all_disc = log['disc']
    print(f'  Baseline discrimination: {disc_0:+.1f}%', flush=True)
    print(f'  Last-5 avg discrimination: {avg_disc:+.1f}%', flush=True)
    print(f'  All disc: {[f"{d:+.1f}" for d in all_disc]}', flush=True)

    # Trend: is discrimination increasing over training?
    if len(all_disc) >= 4:
        first_half = np.mean(all_disc[:len(all_disc)//2])
        second_half = np.mean(all_disc[len(all_disc)//2:])
        print(f'  First half avg: {first_half:+.1f}%, Second half avg: {second_half:+.1f}%', flush=True)

    if avg_disc > 10:
        print(f'\n  VERDICT: SEQUENCE LEARNED -- apical predicts B after A (>{10}% disc)', flush=True)
    elif avg_disc > 0:
        print(f'\n  VERDICT: PARTIAL -- positive discrimination emerging', flush=True)
    else:
        print(f'\n  VERDICT: NOT YET -- no apical prediction signal', flush=True)

    torch.save(log, 'sequence_hierarchy_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
