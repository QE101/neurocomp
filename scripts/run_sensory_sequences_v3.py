"""Sensory Sequence Learning v3: Hippocampus + Structural Plasticity.

The connectivity problem: only ~24 driving edges connect Mon→Tue at N=50K.
No learning rule can find 24 needles in 1.3M edges from correlation alone.

Solution: hippocampus creates the ASSOCIATION, structural plasticity wires it in.
1. Present sequence, hippocampus encodes each transition
2. During replay, hippocampus co-activates Mon and Tue nodes simultaneously
3. Hebbian learning strengthens the ~80 existing cross-pattern edges
4. Structural plasticity grows NEW edges where correlated activity has no edge yet

Over time, the cortex develops its own Mon→Tue wiring, independent of hippocampus.

N=50K, proportional encoding, 2-level hierarchy, Oja + surprise gate,
attention circuit, theta.
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
N_EPOCHS = 300
INPUT_FRACTION = 0.20
SYMBOL_FRACTION = 0.01
WAKE_PER_SLEEP = 10   # sleep every 10 epochs

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
        'structural': {'enabled': True, 'update_interval': 200, 'growth_rate': 0.2,
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
# Core dynamics (with attention)
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


def apply_dual_rate_oja(graph, lr_scale=1.0, is_replay=False):
    """Dual learning rate Oja with synaptic consolidation.

    Driving edges: slow during wake (0.1x), boosted during replay.
    Consolidation: edges strengthened by replay accumulate a consolidation score
    (stored in post_trace, repurposed). Higher consolidation = slower Oja decay.
    After enough replay cycles, consolidated edges become nearly permanent.

    Modulatory: normal (1x), fast prediction adaptation.
    """
    ns_ = graph.node_state
    pred_err = ns_.prediction_error

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            # Update pre_trace for temporal signal
            store.pre_trace *= 0.95
            store.pre_trace += src * 0.05

            if et == EdgeType.DRIVING:
                pe_gate = pred_err[store.dst.long()].abs()
                pe_gate = pe_gate / (pe_gate.mean().clamp(min=0.1))

                if is_replay:
                    # REPLAY: boost learning, build consolidation
                    lr = 0.001 * lr_scale
                    dw = lr * (store.pre_trace * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)

                    # Consolidation: edges with strong co-activation during replay
                    # accumulate consolidation score (stored in post_trace)
                    co_act = store.pre_trace * dst
                    store.post_trace += co_act * 0.01  # slow accumulation
                    store.post_trace.clamp_(0.0, 1.0)
                else:
                    # WAKE: slow learning, consolidated edges resist change
                    # consolidation 0.0 = normal decay, 1.0 = nearly frozen
                    consolidation = store.post_trace
                    effective_lr = 0.0001 * lr_scale * (1.0 - 0.95 * consolidation)

                    dw = effective_lr * pe_gate * (store.pre_trace * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)

                    # Slow consolidation decay (memories fade if not replayed)
                    store.post_trace *= 0.9999
            elif et == EdgeType.MODULATORY:
                lr = 0.001 * lr_scale
                dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
            else:
                lr = 0.001 * lr_scale
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
# Encoding + Measurement (from v2)
# ================================================================
def build_symbols(input_region, device):
    n_input = input_region.shape[0]
    n_per_symbol = max(10, int(n_input * SYMBOL_FRACTION))
    torch.manual_seed(777)
    perm = input_region[torch.randperm(n_input, device=device)]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    symbols = {}
    for i, name in enumerate(days):
        symbols[name] = perm[i * n_per_symbol:(i + 1) * n_per_symbol]
    return symbols, days, n_per_symbol


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
    print('  SEQUENCE LEARNING v3: Hippocampus + Structural Plasticity', flush=True)
    print('  Hippocampus creates associations, SP wires them into cortex.', flush=True)
    print(f'  {N_EPOCHS} epochs, sleep every {WAKE_PER_SLEEP}, N=50K', flush=True)
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
    sp = StructuralPlasticity(config)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Input region + symbols
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(INPUT_FRACTION)]
    symbols, day_names, n_per_symbol = build_symbols(input_region, device)
    all_symbol_nodes = torch.cat([symbols[n] for n in day_names])

    print(f'\nInput region: {input_region.shape[0]} nodes', flush=True)
    print(f'Symbol size: {n_per_symbol} nodes', flush=True)

    # Hippocampus
    hipp = HippocampalSystem(
        config=config.hippocampal, cortical_input_indices=all_symbol_nodes,
        n_cortical=N, device=device, seed=config.simulation.seed)
    print(f'Hippocampus: DG={config.hippocampal.n_dg}, CA3={config.hippocampal.n_ca3}', flush=True)

    # Baseline
    print(f'\n--- BASELINE ---', flush=True)
    acc_0, disc_0 = measure_discrimination(
        symbols, day_names, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
    print(f'  Acc={acc_0:.0f}% Disc={disc_0:+.1f}%', flush=True)

    # Training
    t0 = time.perf_counter()
    log = {'epoch': [], 'acc': [], 'disc': [], 'edges': []}

    print(f'\n--- TRAINING ---', flush=True)
    for epoch in range(N_EPOCHS):
        # Wake: present sequence
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
                apply_dual_rate_oja(graph, lr_scale=1.0, is_replay=False)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
            # Encode each pattern to hippocampus
            hipp.encode(ns.output, graph.step_count)

        # Pause
        for s in range(PAUSE):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()

        # Sleep: hippocampal replay every WAKE_PER_SLEEP epochs
        if (epoch + 1) % WAKE_PER_SLEEP == 0 and hipp.n_stored() > 0:
            schedule = hipp.replay_schedule(config.hippocampal.replay_interleave)
            for pidx in schedule:
                replay = hipp.get_replay_pattern(pidx)
                for s in range(config.hippocampal.replay_steps):
                    step = graph.step_count
                    dual_channel_send(ns, graph, mp, device)
                    inputs = mp.read_inputs(step)
                    inputs.basal[all_symbol_nodes.long()] += replay
                    error_node_update(ns, inputs, theta_mod=1.0, tau_mult=tau_mult)
                    # During replay: driving edges learn at BOOSTED rate (10x)
                    # This is when hippocampus explicitly co-activates predecessor + successor
                    apply_dual_rate_oja(graph, lr_scale=1.0, is_replay=True)
                    graph.increment_step()

        # Structural plasticity every 30 epochs
        if (epoch + 1) % 30 == 0:
            sp_stats = sp.update(graph)
            if sp_stats['grown'] > 0 or sp_stats['pruned'] > 0:
                print(f'  SP ep{epoch+1}: +{sp_stats["grown"]} -{sp_stats["pruned"]} = {graph.n_edges():,}', flush=True)

        # Measure every 30 epochs
        if (epoch + 1) % 30 == 0:
            acc, disc = measure_discrimination(
                symbols, day_names, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            log['epoch'].append(epoch + 1)
            log['acc'].append(acc)
            log['disc'].append(disc)
            log['edges'].append(graph.n_edges())

            elapsed = time.perf_counter() - t0
            trend = 'UP' if disc > disc_0 else 'down'
            print(f'  Epoch {epoch+1:4d}: acc={acc:.0f}% disc={disc:+.1f}% ({trend}) edges={graph.n_edges():,} ({elapsed:.0f}s)', flush=True)

    # Offset test
    print(f'\n--- OFFSET TEST ---', flush=True)
    ap_wed_thu = measure_apical_prediction('Wed', 'Thu', symbols, graph, ns, mp,
                                            device, theta, stp, hom, ip, tau_mult)
    ap_mon_thu = measure_apical_prediction('Mon', 'Thu', symbols, graph, ns, mp,
                                            device, theta, stp, hom, ip, tau_mult)
    disc_offset = (ap_wed_thu - ap_mon_thu) / max(abs(ap_mon_thu), 1e-8) * 100
    print(f'  Wed->Thu (correct): {ap_wed_thu:.4f}', flush=True)
    print(f'  Mon->Thu (wrong):   {ap_mon_thu:.4f}', flush=True)
    print(f'  Offset disc: {disc_offset:+.1f}%', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    final_acc = log['acc'][-1] if log['acc'] else 0
    final_disc = log['disc'][-1] if log['disc'] else 0
    print(f'  Baseline: acc={acc_0:.0f}% disc={disc_0:+.1f}%', flush=True)
    print(f'  Final:    acc={final_acc:.0f}% disc={final_disc:+.1f}%', flush=True)
    print(f'  Trajectory: {[f"{a:.0f}%" for a in log["acc"]]}', flush=True)
    print(f'  Disc trend: {[f"{d:+.1f}" for d in log["disc"]]}', flush=True)
    print(f'  Offset: {disc_offset:+.1f}%', flush=True)

    if final_acc > 80 and final_disc > 10:
        print(f'\n  VERDICT: SEQUENCES LEARNED with hippocampal consolidation', flush=True)
    elif final_acc > 50 or final_disc > 5:
        print(f'\n  VERDICT: PARTIAL -- learning detected', flush=True)
    else:
        print(f'\n  VERDICT: NOT YET', flush=True)

    torch.save(log, 'sensory_sequences_v3_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
