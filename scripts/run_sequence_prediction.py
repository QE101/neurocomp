"""Sequence Prediction: the first real cognitive test.

Present A -> B -> C in order, sustained (200 steps each). Repeat the sequence.
After learning, when A appears, does the system predict B is coming next?

Measured by: prediction error when B follows A should DROP over training.
If the system learns the sequence, it "expects" B after A and C after B.

N=50K, Oja stabilizer, theta, hippocampal encoding.
Sustained presentations (200 steps) — biologically realistic.

Three spatially separated patterns to avoid the overlap problem.
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
from graph_brain.hippocampus import HippocampalSystem
from graph_brain.types import EdgeType, NodeType

STRENGTH = 2.0
PRESENT_STEPS = 50    # steps per pattern (matches validated N=50K dynamics)
N_SEQUENCES = 200     # how many A->B->C sequences to train on
MEASURE_EVERY = 20    # measure prediction error every N sequences

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
        'structural': {'enabled': False},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {'enabled': False},
    'hippocampal': {
        'enabled': True,
        'n_dg': 2000,
        'n_ca3': 500,
        'dg_sparsity': 0.02,
        'dg_fan_in': 2000,
        'ca3_sparsity': 0.10,
        'encoding_lr': 0.5,
        'replay_strength': 0.5,
        'replay_lr_scale': 0.1,
        'max_patterns': 20,
        'replay_interleave': 3,
        'replay_steps': 30,
    },
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


# ================================================================
# PATTERN SETUP: 3 spatially separated patterns
# ================================================================
def setup_patterns(graph):
    """Create 3 spatially separated patterns using different spatial regions.

    Pattern A: 100 nodes from x < 0.33
    Pattern B: 100 nodes from 0.33 < x < 0.66
    Pattern C: 100 nodes from x > 0.66

    Spatially separated -> minimal KNN overlap -> clean test.
    """
    ns = graph.node_state
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_x = ns.position[exc_idx, 0]  # use x-axis for separation

    # Three spatial regions
    region_a = exc_idx[exc_x < 0.33]
    region_b = exc_idx[(exc_x >= 0.33) & (exc_x < 0.66)]
    region_c = exc_idx[exc_x >= 0.66]

    # Take 100 from each (random subset)
    torch.manual_seed(42)
    pa = region_a[torch.randperm(region_a.shape[0])[:100]]
    pb = region_b[torch.randperm(region_b.shape[0])[:100]]
    pc = region_c[torch.randperm(region_c.shape[0])[:100]]

    return pa, pb, pc, exc_idx


# ================================================================
# MEASURE: Prediction error at transition points
# ================================================================
def measure_transition_error(graph, ns, mp, device, theta, pat_from, pat_to, stp, hom, ip):
    """Measure prediction error at the transition from one pattern to another.

    Present pat_from for PRESENT_STEPS, then switch to pat_to.
    Measure prediction error during the FIRST 20 steps of pat_to.
    If the system predicts the transition, error should be LOW.
    Includes all maintenance ops (STP, homeostatic, IP) to prevent divergence.
    No Hebbian learning during measurement.
    """
    # Establish pat_from
    for s in range(PRESENT_STEPS):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pat_from.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()

    # Switch to pat_to — measure prediction error during first 20 steps
    errors = []
    for s in range(20):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pat_to.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
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
    print('  SEQUENCE PREDICTION: A -> B -> C', flush=True)
    print(f'  N=50K, {PRESENT_STEPS} steps/pattern, {N_SEQUENCES} sequences', flush=True)
    print('  Oja stabilizer + theta + hippocampus', flush=True)
    print('  Measure: prediction error at transitions', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    print('Building N=50K graph...', flush=True)
    t_build = time.perf_counter()
    graph = NeuromorphicGraph(config)
    graph.initialize()
    print(f'Built in {time.perf_counter()-t_build:.1f}s ({graph.n_edges():,} edges)', flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    pa, pb, pc, exc_idx = setup_patterns(graph)
    all_input_nodes = torch.cat([pa, pb, pc])
    sequence = [pa, pb, pc]
    names = ['A', 'B', 'C']

    print(f'  Pattern A: {pa.shape[0]} nodes (x < 0.33)', flush=True)
    print(f'  Pattern B: {pb.shape[0]} nodes (0.33 < x < 0.66)', flush=True)
    print(f'  Pattern C: {pc.shape[0]} nodes (x > 0.66)', flush=True)

    # Spatial separation check
    xa = ns.position[pa, 0]
    xb = ns.position[pb, 0]
    xc = ns.position[pc, 0]
    print(f'  A x-range: [{xa.min():.3f}, {xa.max():.3f}]', flush=True)
    print(f'  B x-range: [{xb.min():.3f}, {xb.max():.3f}]', flush=True)
    print(f'  C x-range: [{xc.min():.3f}, {xc.max():.3f}]', flush=True)

    # Initialize hippocampus
    hipp = HippocampalSystem(
        config=config.hippocampal,
        cortical_input_indices=all_input_nodes,
        n_cortical=N,
        device=device,
        seed=config.simulation.seed,
    )

    # ============ BASELINE MEASUREMENT ============
    print(f'\n  --- BASELINE (before any training) ---', flush=True)
    pe_ab_0 = measure_transition_error(graph, ns, mp, device, theta, pa, pb, stp, hom, ip)
    pe_bc_0 = measure_transition_error(graph, ns, mp, device, theta, pb, pc, stp, hom, ip)
    pe_ca_0 = measure_transition_error(graph, ns, mp, device, theta, pc, pa, stp, hom, ip)
    # Control: wrong transition (A->C instead of A->B)
    pe_ac_0 = measure_transition_error(graph, ns, mp, device, theta, pa, pc, stp, hom, ip)
    print(f'  A->B: {pe_ab_0:.4f}  B->C: {pe_bc_0:.4f}  C->A: {pe_ca_0:.4f}  A->C(ctrl): {pe_ac_0:.4f}', flush=True)

    # ============ TRAINING ============
    t0 = time.perf_counter()
    log = {'pe_ab': [pe_ab_0], 'pe_bc': [pe_bc_0], 'pe_ca': [pe_ca_0], 'pe_ac': [pe_ac_0], 'seq': [0]}

    print(f'\n  --- TRAINING: {N_SEQUENCES} sequences of A->B->C ---', flush=True)
    for seq_i in range(N_SEQUENCES):
        # Present A -> B -> C with sustained presentation
        for pat_idx, pat in enumerate(sequence):
            for s in range(PRESENT_STEPS):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
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

            # Encode each pattern transition to hippocampus
            hipp.encode(ns.output, graph.step_count)

        # Sleep replay every 50 sequences
        if (seq_i + 1) % 50 == 0 and hipp.n_stored() > 0:
            schedule = hipp.replay_schedule(config.hippocampal.replay_interleave)
            for pat_idx in schedule:
                replay = hipp.get_replay_pattern(pat_idx)
                for s in range(config.hippocampal.replay_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_input_nodes.long()] += replay
                    error_node_update(ns, inputs, theta_mod=1.0)
                    apply_hebbian_oja(graph, lr_scale=config.hippocampal.replay_lr_scale)
                    graph.increment_step()

        # Measure every MEASURE_EVERY sequences
        if (seq_i + 1) % MEASURE_EVERY == 0:
            pe_ab = measure_transition_error(graph, ns, mp, device, theta, pa, pb, stp, hom, ip)
            pe_bc = measure_transition_error(graph, ns, mp, device, theta, pb, pc, stp, hom, ip)
            pe_ca = measure_transition_error(graph, ns, mp, device, theta, pc, pa, stp, hom, ip)
            pe_ac = measure_transition_error(graph, ns, mp, device, theta, pa, pc, stp, hom, ip)

            log['pe_ab'].append(pe_ab)
            log['pe_bc'].append(pe_bc)
            log['pe_ca'].append(pe_ca)
            log['pe_ac'].append(pe_ac)
            log['seq'].append(seq_i + 1)

            elapsed = time.perf_counter() - t0
            # Reduction from baseline
            red_ab = (1 - pe_ab / pe_ab_0) * 100 if pe_ab_0 > 0 else 0
            red_bc = (1 - pe_bc / pe_bc_0) * 100 if pe_bc_0 > 0 else 0
            red_ca = (1 - pe_ca / pe_ca_0) * 100 if pe_ca_0 > 0 else 0
            red_ac = (1 - pe_ac / pe_ac_0) * 100 if pe_ac_0 > 0 else 0

            print(f'  Seq {seq_i+1:4d}: '
                  f'A->B={pe_ab:.4f}({red_ab:+.0f}%) '
                  f'B->C={pe_bc:.4f}({red_bc:+.0f}%) '
                  f'C->A={pe_ca:.4f}({red_ca:+.0f}%) '
                  f'A->C={pe_ac:.4f}({red_ac:+.0f}%) '
                  f'({elapsed:.0f}s)', flush=True)

    # ============ FINAL ANALYSIS ============
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)

    final_ab = log['pe_ab'][-1]
    final_bc = log['pe_bc'][-1]
    final_ca = log['pe_ca'][-1]
    final_ac = log['pe_ac'][-1]

    red_ab = (1 - final_ab / pe_ab_0) * 100
    red_bc = (1 - final_bc / pe_bc_0) * 100
    red_ca = (1 - final_ca / pe_ca_0) * 100
    red_ac = (1 - final_ac / pe_ac_0) * 100

    print(f'  Transition  | Baseline | Final    | Reduction', flush=True)
    print(f'  ------------|----------|----------|----------', flush=True)
    print(f'  A->B (seq)  | {pe_ab_0:.4f}   | {final_ab:.4f}   | {red_ab:+.1f}%', flush=True)
    print(f'  B->C (seq)  | {pe_bc_0:.4f}   | {final_bc:.4f}   | {red_bc:+.1f}%', flush=True)
    print(f'  C->A (seq)  | {pe_ca_0:.4f}   | {final_ca:.4f}   | {red_ca:+.1f}%', flush=True)
    print(f'  A->C (ctrl) | {pe_ac_0:.4f}   | {final_ac:.4f}   | {red_ac:+.1f}%', flush=True)

    # Sequence learning = sequential transitions have lower error than control
    if final_ab < final_ac and final_bc < final_ac:
        print(f'\n  VERDICT: SEQUENCE LEARNED -- A->B and B->C have lower error than A->C', flush=True)
    elif red_ab > 10 or red_bc > 10:
        print(f'\n  VERDICT: PARTIAL LEARNING -- some error reduction at transitions', flush=True)
    else:
        print(f'\n  VERDICT: NO LEARNING -- transitions not predicted', flush=True)

    # Weight health
    drv_w = graph.edge_store(EdgeType.DRIVING).weight.mean().item()
    mod_w = graph.edge_store(EdgeType.MODULATORY).weight.mean().item()
    print(f'\n  Weights: drv={drv_w:.4f} mod={mod_w:.4f}', flush=True)

    torch.save(log, 'sequence_prediction_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
