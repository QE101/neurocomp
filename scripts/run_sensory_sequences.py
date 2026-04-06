"""Sensory Bottleneck Sequence Learning.

Small sensory input (50 nodes), different activation vectors per symbol.
The graph self-organizes internal representations through propagation + learning.
No hand-assigned symbol nodes. No pre-encoded categories.

Curriculum: days of the week (7 symbols), 500 epochs.
Each symbol is a unique activation vector on the same 50 sensory nodes.
Measurement: after presenting Mon, does the graph's internal state
predict Tue better than other days?

N=50K, 2-level hierarchy, Oja stabilizer, theta, small-world.
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
N_SENSORY = 50  # sensory bottleneck size

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
# Core dynamics
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
            # SST gated by both self-inhibition AND VIP disinhibition
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
    # Output-based messages: driving, perisomatic inhibition, disinhibition, retrograde
    for et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = output[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
              EdgeType.DISINHIBITION: Channel.VIP_INHIBITION,
              EdgeType.RETROGRADE: Channel.RETROGRADE}[et]
        mp.delay_buffer.write(ch, store.dst, msg, d, step)
    # Content-based messages: modulatory, dendritic inhibition
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
    n_add = int(graph.n_edges(EdgeType.MODULATORY) * fraction)
    src = torch.randint(0, n_exc, (n_add,), device=device)
    dst = torch.randint(0, n_exc, (n_add,), device=device)
    valid = src != dst
    graph.add_edges(EdgeType.MODULATORY, exc_idx[src[valid]], exc_idx[dst[valid]],
                    weights=torch.full((valid.sum().item(),), 0.05, device=device))
    return valid.sum().item()


# ================================================================
# SENSORY ENCODING: activation vectors on bottleneck nodes
# ================================================================
def build_sensory_symbols(sensory_nodes, device):
    """Build symbols as unique activation vectors on sensory nodes.

    Each symbol is a distinct pattern of activation strengths on the
    same N_SENSORY nodes. No hand-assigned downstream nodes.
    The graph discovers internal representations through propagation.
    """
    n = sensory_nodes.shape[0]
    torch.manual_seed(777)

    symbols = {}
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Sparse binary encoding: each day activates a DIFFERENT subset of sensory nodes
    # 10 of 50 nodes ON per symbol, rest OFF. Near-orthogonal by construction.
    perm = torch.randperm(n, device=device)
    for i, name in enumerate(days):
        vec = torch.zeros(n, device=device)
        # Each day gets 7 unique nodes ON (no overlap between days)
        start = i * 7
        vec[perm[start:start + 7]] = STRENGTH
        # Plus 3 random shared nodes for some base activation
        vec[perm[49]] = STRENGTH * 0.5  # shared node (all days)
        vec[perm[48]] = STRENGTH * 0.3
        vec[perm[47]] = STRENGTH * 0.2
        symbols[name] = vec

    # Verify symbols are distinct (pairwise cosine similarity)
    names = list(symbols.keys())
    print(f'  Symbol similarities:', flush=True)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cos = F.cosine_similarity(symbols[names[i]].unsqueeze(0),
                                       symbols[names[j]].unsqueeze(0)).item()
            if i < 3 and j < 4:  # print first few
                print(f'    {names[i]}-{names[j]}: {cos:.3f}', flush=True)

    return symbols, days


# ================================================================
# PRESENT + MEASURE
# ================================================================
def present_symbol(symbol_vec, sensory_nodes, graph, ns, mp, device, theta, stp, hom, ip, tau_mult):
    """Present one symbol for PD steps with learning."""
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        # Inject activation vector at sensory nodes
        inputs.basal[sensory_nodes.long()] += symbol_vec
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


def present_pause(graph, ns, mp, device, theta, tau_mult):
    for s in range(PAUSE):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()


def snapshot_internal_state(ns, exc_idx):
    """Capture the graph's internal state (excitatory output) as a fingerprint."""
    return ns.output[exc_idx].detach().clone()


def find_responsive_nodes(symbols, day_names, sensory_nodes, exc_idx, graph, ns, mp,
                           device, theta, stp, hom, ip, tau_mult):
    """Find nodes whose output varies across different inputs.

    Present each symbol, record internal state. Nodes with high variance
    across symbols are "responsive" — they self-selected as relevant.
    """
    states = []
    for name in day_names:
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[sensory_nodes.long()] += symbols[name]
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
        states.append(ns.output[exc_idx].detach().clone())
        # brief pause
        for s in range(10):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()

    # Variance across symbols per node
    state_stack = torch.stack(states)  # [n_symbols, n_exc]
    variance = state_stack.var(dim=0)  # [n_exc]

    # Top responsive nodes (above median variance, or top 10%)
    threshold = variance.quantile(0.90)
    responsive = variance > threshold
    responsive_idx = torch.where(responsive)[0]

    return responsive_idx, variance


def measure_sequence_prediction(symbols, day_names, sensory_nodes, exc_idx,
                                 graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
                                 responsive_idx=None):
    """For each adjacent pair, present A then check:
    does the internal state after A resemble the state DURING B
    more than the state during other days?

    Only compares RESPONSIVE nodes (nodes that self-selected as input-dependent).
    """
    # Which nodes to compare
    if responsive_idx is not None and responsive_idx.numel() > 0:
        measure_idx = responsive_idx
    else:
        measure_idx = torch.arange(exc_idx.shape[0], device=device)

    # Build reference fingerprints (responsive nodes only)
    refs = {}
    for name in day_names:
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[sensory_nodes.long()] += symbols[name]
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
        full_state = ns.output[exc_idx].detach().clone()
        refs[name] = full_state[measure_idx]  # only responsive nodes
        present_pause(graph, ns, mp, device, theta, tau_mult)

    # Test each transition
    n_correct = 0
    n_tested = 0
    total_disc = 0

    for i in range(len(day_names) - 1):
        pred_name = day_names[i]
        target_name = day_names[i + 1]

        # Present predecessor
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[sensory_nodes.long()] += symbols[pred_name]
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

        # Post-predecessor state (5 steps, no input, responsive nodes only)
        post_states = []
        for s in range(5):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()
            full_state = ns.output[exc_idx].detach().clone()
            post_states.append(full_state[measure_idx])
        post_state = torch.stack(post_states).mean(dim=0)

        # Compare to all references (except predecessor)
        sims = {}
        for name, ref in refs.items():
            if name == pred_name:
                continue
            sim = F.cosine_similarity(post_state.unsqueeze(0), ref.unsqueeze(0)).item()
            sims[name] = sim

        best_match = max(sims, key=sims.get)
        correct = (best_match == target_name)
        n_correct += int(correct)
        n_tested += 1

        target_sim = sims[target_name]
        other_sims = [v for k, v in sims.items() if k != target_name]
        other_mean = np.mean(other_sims)
        disc = (target_sim - other_mean) / max(abs(other_mean), 1e-8) * 100
        total_disc += disc

        present_pause(graph, ns, mp, device, theta, tau_mult)

    acc = n_correct / max(n_tested, 1) * 100
    avg_disc = total_disc / max(n_tested, 1)
    return acc, avg_disc, n_correct, n_tested


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  SENSORY BOTTLENECK SEQUENCE LEARNING', flush=True)
    print(f'  {N_SENSORY} sensory nodes, days of the week, {N_EPOCHS} epochs', flush=True)
    print('  Self-organized internal representations', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)
    print(f'Graph: {graph.n_edges():,} edges', flush=True)
    print(builder.summary(graph), flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Sensory bottleneck: 50 nodes at the bottom of the graph (lowest z)
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    _, sort_idx = exc_z.sort()
    sensory_nodes = exc_idx[sort_idx[:N_SENSORY]]
    print(f'\n  Sensory nodes: {N_SENSORY} (lowest z)', flush=True)
    print(f'  z-range: [{ns.position[sensory_nodes, 2].min():.4f}, {ns.position[sensory_nodes, 2].max():.4f}]', flush=True)

    symbols, day_names = build_sensory_symbols(sensory_nodes, device)

    # Training
    t0 = time.perf_counter()
    log = {'epoch': [], 'acc': [], 'disc': [], 'n_responsive': []}
    responsive_idx = None

    print(f'\n--- TRAINING: {N_EPOCHS} epochs of Mon->Tue->...->Sun ---', flush=True)
    for epoch in range(N_EPOCHS):
        # Present full week sequence
        for name in day_names:
            present_symbol(symbols[name], sensory_nodes, graph, ns, mp, device,
                          theta, stp, hom, ip, tau_mult)
        present_pause(graph, ns, mp, device, theta, tau_mult)

        # Find responsive nodes and measure every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Identify which nodes care about input
            responsive_idx, variance = find_responsive_nodes(
                symbols, day_names, sensory_nodes, exc_idx,
                graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            n_resp = responsive_idx.shape[0]

            acc, disc, nc, nt = measure_sequence_prediction(
                symbols, day_names, sensory_nodes, exc_idx,
                graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
                responsive_idx=responsive_idx)
            log['epoch'].append(epoch + 1)
            log['acc'].append(acc)
            log['disc'].append(disc)
            log['n_responsive'].append(n_resp)

            elapsed = time.perf_counter() - t0
            print(f'  Epoch {epoch+1:4d}: acc={acc:.0f}% ({nc}/{nt}) disc={disc:+.1f}% resp={n_resp} ({elapsed:.0f}s)', flush=True)

    # Offset test
    print(f'\n--- OFFSET TEST ---', flush=True)
    print('  Trained on: Mon->Tue->Wed->Thu->Fri->Sat->Sun', flush=True)
    print('  Testing: present Wed alone, does post-Wed state match Thu?', flush=True)

    # Present Wed, check if internal state points to Thu
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[sensory_nodes.long()] += symbols['Wed']
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        graph.increment_step()

    # Read post-Wed state
    post_states = []
    for s in range(5):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()
        post_states.append(ns.output[exc_idx].detach().clone())
    post_wed = torch.stack(post_states).mean(dim=0)

    # Compare to all day references (build fresh)
    print('  Building reference fingerprints...', flush=True)
    refs = {}
    for name in day_names:
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(ns, graph, mp, device)
            inputs = mp.read_inputs(step)
            inputs.basal[sensory_nodes.long()] += symbols[name]
            error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
            graph.increment_step()
        refs[name] = ns.output[exc_idx].detach().clone()
        present_pause(graph, ns, mp, device, theta, tau_mult)

    sims = {}
    for name, ref in refs.items():
        if name == 'Wed':
            continue
        sim = F.cosine_similarity(post_wed.unsqueeze(0), ref.unsqueeze(0)).item()
        sims[name] = sim
        print(f'  Post-Wed vs {name}: {sim:.4f}', flush=True)

    best = max(sims, key=sims.get)
    print(f'  Best match: {best} (should be Thu)', flush=True)
    print(f'  Wed->Thu prediction: {"CORRECT" if best == "Thu" else "WRONG"}', flush=True)

    # Results
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    final_acc = log['acc'][-1] if log['acc'] else 0
    final_disc = log['disc'][-1] if log['disc'] else 0
    print(f'  Final accuracy: {final_acc:.0f}%', flush=True)
    print(f'  Final discrimination: {final_disc:+.1f}%', flush=True)
    print(f'  Accuracy trajectory: {[f"{a:.0f}%" for a in log["acc"]]}', flush=True)

    if final_acc > 50:
        print(f'\n  VERDICT: SEQUENCES LEARNED from sensory bottleneck', flush=True)
    elif final_acc > 17:  # chance = 1/6 = 17%
        print(f'\n  VERDICT: PARTIAL -- above chance ({final_acc:.0f}% vs 17% chance)', flush=True)
    else:
        print(f'\n  VERDICT: NOT YET -- at chance level', flush=True)

    torch.save(log, 'sensory_sequences_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
