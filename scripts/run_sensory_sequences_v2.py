"""Sensory Sequence Learning v2: proportional encoding, no magic numbers.

Input region = 20% of excitatory (scales with N).
Symbol activation = 1% of input region (scales with input region).
Each symbol = different random sparse subset of the input surface.
Downstream representation is 100% self-organized.

Encoding is the "retina" — fixed, not learned. Everything downstream learns.

Days of the week, 500 epochs, with attention circuit (VIP->SST disinhibition).
N=50K, 2-level hierarchy, Oja stabilizer, theta, small-world.

Measurement: apical prediction at next-symbol's nodes during current symbol.
(Proven metric from run_sequence_hierarchy.py: +74.2% discrimination.)
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
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.types import EdgeType, NodeType

STRENGTH = 2.0
PD = 50
PAUSE = 30
N_EPOCHS = 500
INPUT_FRACTION = 0.20   # input region = 20% of excitatory
SYMBOL_FRACTION = 0.01  # each symbol = 1% of input region

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
        'structural': {'enabled': False},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {
        'enabled': True, 'n_levels': 2, 'split_axis': 2,
        'time_scale_factor': 3.0, 'inter_level_k': 5,
        'inter_level_sigma': 0.5, 'inter_level_init_weight': 0.02,
    },
}


# ================================================================
# Core dynamics (with attention circuit)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0, tau_mult=None):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()
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
    ns.output = torch.where(exc_mask, F.softplus(pred_err.abs()) * pv_gain * ns.gain, ns.output)
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
    ns.output.clamp_(min=0.0)
    ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)


def dual_channel_send(ns, graph, mp, device):
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


def apply_surprise_gated_oja(graph, lr_scale=1.0):
    """Surprise-gated temporal Oja: learn when prediction error is high.

    Driving edges: pre_trace × post × pe_gate (temporal + surprise)
      - pre_trace encodes recent pre history (temporal direction)
      - pe_gate = |prediction_error| at dst (high at transitions, low at steady-state)
      - Only learns strongly when something CHANGES (transition moments)

    Modulatory edges: standard Oja (symmetric, for continuous prediction learning)
    """
    ns_ = graph.node_state
    lr = 0.001 * lr_scale
    pred_err = ns_.prediction_error

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            # Update pre_trace
            store.pre_trace *= 0.95
            store.pre_trace += src * 0.05

            if et == EdgeType.DRIVING:
                # Surprise gate: prediction error at destination
                pe_gate = pred_err[store.dst.long()].abs()
                # Normalize pe_gate so it scales the learning rate, not explodes it
                pe_gate = pe_gate / (pe_gate.mean().clamp(min=0.1))

                # Temporal + surprise: learn transitions, not steady-state
                dw = lr * pe_gate * (store.pre_trace * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
            else:
                # Standard Oja for modulatory/inhibitory (continuous prediction learning)
                dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)

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


# ================================================================
# PROPORTIONAL SYMBOL ENCODING
# ================================================================
def build_symbols(input_region, device):
    """Each symbol = random 1% of input region. Zero overlap. Scales with N."""
    n_input = input_region.shape[0]
    n_per_symbol = max(10, int(n_input * SYMBOL_FRACTION))

    torch.manual_seed(777)
    perm = input_region[torch.randperm(n_input, device=device)]

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    symbols = {}
    for i, name in enumerate(days):
        start = i * n_per_symbol
        symbols[name] = perm[start:start + n_per_symbol]

    # Verify zero overlap
    for i, n1 in enumerate(days):
        for j, n2 in enumerate(days):
            if i >= j: continue
            overlap = len(set(symbols[n1].cpu().tolist()) & set(symbols[n2].cpu().tolist()))
            if overlap > 0:
                print(f'  WARNING: {n1}-{n2} overlap={overlap}', flush=True)

    return symbols, days, n_per_symbol


# ================================================================
# MEASUREMENT: apical prediction (proven metric)
# ================================================================
def measure_apical_prediction(pred_name, target_name, symbols, graph, ns, mp,
                               device, theta, stp, hom, ip, tau_mult):
    """Present predecessor, measure apical at target's nodes during last 10 steps.
    No target presented. Pure prediction signal."""
    pred_nodes = symbols[pred_name]
    target_nodes = symbols[target_name]

    # Present predecessor
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

    # Read apical at target nodes (last 5 steps, no input)
    ap_vals = []
    for s in range(5):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        ap_vals.append(ns.apical[target_nodes.long()].mean().item())

    return float(np.mean(ap_vals))


def measure_sequence_discrimination(symbols, day_names, graph, ns, mp, device,
                                     theta, stp, hom, ip, tau_mult):
    """For each transition, measure: is apical at target higher after correct
    predecessor than after wrong predecessor?

    Returns accuracy (% of transitions where correct pred > wrong pred) and avg discrimination.
    """
    n_correct = 0
    n_tested = 0
    total_disc = 0

    for i in range(len(day_names) - 1):
        pred = day_names[i]
        target = day_names[i + 1]
        # Pick a wrong predecessor (2 positions away in the sequence)
        wrong = day_names[(i + 3) % len(day_names)]

        # Apical at target after correct predecessor
        ap_correct = measure_apical_prediction(pred, target, symbols, graph, ns, mp,
                                                device, theta, stp, hom, ip, tau_mult)
        # Apical at target after wrong predecessor
        ap_wrong = measure_apical_prediction(wrong, target, symbols, graph, ns, mp,
                                              device, theta, stp, hom, ip, tau_mult)

        correct = ap_correct > ap_wrong
        n_correct += int(correct)
        n_tested += 1

        disc = (ap_correct - ap_wrong) / max(abs(ap_wrong), 1e-8) * 100
        total_disc += disc

    acc = n_correct / max(n_tested, 1) * 100
    avg_disc = total_disc / max(n_tested, 1)
    return acc, avg_disc


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  SENSORY SEQUENCE LEARNING v2', flush=True)
    print('  Proportional encoding: input=20% exc, symbol=1% input', flush=True)
    print('  Apical prediction metric. Attention circuit active.', flush=True)
    print(f'  {N_EPOCHS} epochs of Mon->Tue->...->Sun', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)
    print(f'Graph: {graph.n_edges():,} edges (+{n_sw:,} small-world)', flush=True)
    print(builder.summary(graph), flush=True)
    if graph.has_edge_type(EdgeType.DISINHIBITION):
        print(f'Attention: {graph.n_edges(EdgeType.DISINHIBITION):,} VIP->SST edges', flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Input region: bottom 20% of excitatory by z
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, day_names, n_per_symbol = build_symbols(input_region, device)

    print(f'\nInput region: {input_region.shape[0]} nodes ({INPUT_FRACTION*100:.0f}% of exc)', flush=True)
    print(f'Symbol size: {n_per_symbol} nodes ({SYMBOL_FRACTION*100:.0f}% of input region)', flush=True)
    print(f'Sequence: {" -> ".join(day_names)}', flush=True)

    # Baseline
    print(f'\n--- BASELINE ---', flush=True)
    acc_0, disc_0 = measure_sequence_discrimination(
        symbols, day_names, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Accuracy: {acc_0:.0f}% Discrimination: {disc_0:+.1f}%', flush=True)

    # Training
    t0 = time.perf_counter()
    log = {'epoch': [], 'acc': [], 'disc': []}

    print(f'\n--- TRAINING ---', flush=True)
    for epoch in range(N_EPOCHS):
        # Present full sequence
        for name in day_names:
            pattern = symbols[name]
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pattern.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_surprise_gated_oja(graph, lr_scale=1.0)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()

        # Pause between sequences
        for s in range(PAUSE):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()

        # Measure every 50 epochs
        if (epoch + 1) % 50 == 0:
            acc, disc = measure_sequence_discrimination(
                symbols, day_names, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            log['epoch'].append(epoch + 1)
            log['acc'].append(acc)
            log['disc'].append(disc)

            elapsed = time.perf_counter() - t0
            trend = 'UP' if disc > disc_0 else 'down'
            print(f'  Epoch {epoch+1:4d}: acc={acc:.0f}% disc={disc:+.1f}% ({trend}) ({elapsed:.0f}s)', flush=True)

    # Offset test
    print(f'\n--- OFFSET TEST ---', flush=True)
    ap_wed_thu = measure_apical_prediction('Wed', 'Thu', symbols, graph, ns, mp,
                                            device, theta, stp, hom, ip, tau_mult)
    ap_mon_thu = measure_apical_prediction('Mon', 'Thu', symbols, graph, ns, mp,
                                            device, theta, stp, hom, ip, tau_mult)
    print(f'  Apical at Thu after Wed (correct): {ap_wed_thu:.4f}', flush=True)
    print(f'  Apical at Thu after Mon (wrong):   {ap_mon_thu:.4f}', flush=True)
    disc_offset = (ap_wed_thu - ap_mon_thu) / max(abs(ap_mon_thu), 1e-8) * 100
    print(f'  Offset discrimination: {disc_offset:+.1f}%', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    final_acc = log['acc'][-1] if log['acc'] else 0
    final_disc = log['disc'][-1] if log['disc'] else 0
    print(f'  Baseline: acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)
    print(f'  Final:    acc={final_acc:.0f}% disc={final_disc:+.1f}%', flush=True)
    print(f'  Trajectory: {[f"{a:.0f}%" for a in log["acc"]]}', flush=True)
    print(f'  Offset test: {disc_offset:+.1f}%', flush=True)

    if final_acc > 80:
        print(f'\n  VERDICT: SEQUENCES LEARNED -- high accuracy prediction', flush=True)
    elif final_acc > 50:
        print(f'\n  VERDICT: PARTIAL -- above chance, learning detected', flush=True)
    elif final_disc > 10:
        print(f'\n  VERDICT: WEAK SIGNAL -- discrimination positive but accuracy low', flush=True)
    else:
        print(f'\n  VERDICT: NOT YET', flush=True)

    torch.save(log, 'sensory_sequences_v2_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
