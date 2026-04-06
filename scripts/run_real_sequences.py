"""Real Sequence Learning: deterministic structured data.

Encode real-world sequences as spatial patterns with category structure:
- Days of the week (Mon-Tue-Wed-Thu-Fri-Sat-Sun)
- Digits 0-9
- Months (Jan-Feb-Mar-...-Dec)

Each symbol: 25 category-shared nodes + 25 unique nodes = 50 active nodes
Category membership is encoded in the shared nodes.

Training: present complete sequences with pauses between types.
Testing:
1. Prediction accuracy: does the graph predict the next element?
2. Offset recognition: present Tue-Wed-Thu, does it quickly predict Fri?
3. Category: does any day activate "day" category nodes?

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
PD = 50       # steps per symbol presentation
PAUSE = 30    # steps of no input between sequences
N_EPOCHS = 50 # times through the full curriculum

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
    'hierarchy': {
        'enabled': True, 'n_levels': 2, 'split_axis': 2,
        'time_scale_factor': 3.0, 'inter_level_k': 5,
        'inter_level_sigma': 0.5, 'inter_level_init_weight': 0.02,
    },
}


# ================================================================
# Core dynamics (tau_mult version)
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
    n_add = int(n_existing * fraction)
    src = torch.randint(0, n_exc, (n_add,), device=device)
    dst = torch.randint(0, n_exc, (n_add,), device=device)
    valid = src != dst
    graph.add_edges(EdgeType.MODULATORY, exc_idx[src[valid]], exc_idx[dst[valid]],
                    weights=torch.full((valid.sum().item(),), 0.05, device=device))
    return valid.sum().item()


# ================================================================
# SYMBOL ENCODING: category-structured patterns
# ================================================================
def build_symbols(input_region, device):
    """Build structured symbol encodings.

    Each symbol = 25 category-shared nodes + 25 unique nodes = 50 active nodes.
    Categories: days (7), digits (10), months (12) = 29 symbols.
    """
    n_input = input_region.shape[0]
    torch.manual_seed(123)  # deterministic encoding
    shuffled = input_region[torch.randperm(n_input, device=device)]

    ptr = 0
    def alloc(n):
        nonlocal ptr
        nodes = shuffled[ptr:ptr+n]
        ptr += n
        return nodes

    symbols = {}  # name -> tensor of node indices
    categories = {}  # category_name -> list of symbol names
    cat_nodes = {}  # category_name -> shared node indices

    # Days of the week
    day_shared = alloc(25)
    cat_nodes['days'] = day_shared
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    categories['days'] = days
    for name in days:
        unique = alloc(25)
        symbols[name] = torch.cat([day_shared, unique])

    # Digits 0-9
    digit_shared = alloc(25)
    cat_nodes['digits'] = digit_shared
    digits = [str(i) for i in range(10)]
    categories['digits'] = digits
    for name in digits:
        unique = alloc(25)
        symbols[name] = torch.cat([digit_shared, unique])

    # Months
    month_shared = alloc(25)
    cat_nodes['months'] = month_shared
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    categories['months'] = months
    for name in months:
        unique = alloc(25)
        symbols[name] = torch.cat([month_shared, unique])

    print(f'  Encoded {len(symbols)} symbols using {ptr}/{n_input} nodes', flush=True)
    print(f'  Categories: days({len(days)}), digits({len(digits)}), months({len(months)})', flush=True)

    return symbols, categories, cat_nodes


def build_curriculum():
    """Build the training curriculum: ordered sequences with pauses."""
    sequences = [
        ('days_full', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
        ('digits_full', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
        ('months_full', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
    ]
    return sequences


# ================================================================
# PRESENT a sequence with learning
# ================================================================
def present_sequence(seq_names, symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult):
    """Present a sequence of symbols, learning at each step."""
    for name in seq_names:
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
            apply_hebbian_oja(graph, lr_scale=1.0)
            if step % 100 == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()


def present_pause(graph, ns, mp, device, theta, tau_mult, steps=None):
    """Run steps with no input (pause between sequences)."""
    for s in range(steps or PAUSE):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()


# ================================================================
# MEASUREMENT: prediction accuracy
# ================================================================
def measure_prediction_accuracy(predecessor_name, target_name, symbols, graph, ns, mp,
                                 device, theta, stp, hom, ip, tau_mult, all_same_cat_names):
    """Present predecessor, then measure: does apical at target > apical at other same-category symbols?

    Accuracy = 1 if target has highest apical among its category, 0 otherwise.
    Also returns: apical at target, mean apical at other same-category, discrimination.
    """
    # Present predecessor
    pred_pattern = symbols[predecessor_name]
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pred_pattern.long()] += STRENGTH
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

    # Read apical at all same-category symbol nodes (last 5 steps average)
    apicals = {}
    for s in range(5):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        # NO input — pure prediction
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()

        for name in all_same_cat_names:
            ap = ns.apical[symbols[name].long()].mean().item()
            if name not in apicals:
                apicals[name] = []
            apicals[name].append(ap)

    # Average over 5 steps
    mean_apicals = {name: np.mean(vals) for name, vals in apicals.items()}
    target_ap = mean_apicals[target_name]
    others = [v for k, v in mean_apicals.items() if k != target_name and k != predecessor_name]
    other_mean = np.mean(others) if others else 0

    # Accuracy: is target the highest among non-predecessor symbols?
    non_pred = {k: v for k, v in mean_apicals.items() if k != predecessor_name}
    if non_pred:
        best = max(non_pred, key=non_pred.get)
        correct = (best == target_name)
    else:
        correct = False

    disc = (target_ap - other_mean) / max(abs(other_mean), 1e-8) * 100

    return correct, target_ap, other_mean, disc


# ================================================================
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  REAL SEQUENCE LEARNING', flush=True)
    print('  Days, Digits, Months — structured deterministic sequences', flush=True)
    print(f'  N=50K, 2-level hierarchy, {N_EPOCHS} epochs', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG_50K)
    print('Building graph...', flush=True)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    n_sw = add_small_world_edges(graph, fraction=0.2)
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)
    print(f'Graph: {graph.n_edges():,} edges, +{n_sw:,} small-world', flush=True)
    print(builder.summary(graph), flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    # Build symbols in input region
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(0.2)]
    symbols, categories, cat_nodes = build_symbols(input_region, device)

    curriculum = build_curriculum()
    print(f'  Curriculum: {len(curriculum)} sequences', flush=True)
    for name, seq in curriculum:
        print(f'    {name}: {" -> ".join(seq)}', flush=True)

    # ============ BASELINE ============
    print(f'\n--- BASELINE ---', flush=True)
    # Test: after Mon, is Tue predicted?
    correct, ap_tgt, ap_oth, disc = measure_prediction_accuracy(
        'Mon', 'Tue', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['days'])
    print(f'  Mon->Tue: correct={correct}, ap_tgt={ap_tgt:.4f}, ap_other={ap_oth:.4f}, disc={disc:+.1f}%', flush=True)
    correct2, ap2, ao2, d2 = measure_prediction_accuracy(
        '3', '4', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['digits'])
    print(f'  3->4: correct={correct2}, ap_tgt={ap2:.4f}, ap_other={ao2:.4f}, disc={d2:+.1f}%', flush=True)

    # ============ TRAINING ============
    t0 = time.perf_counter()
    log = {'epoch': [], 'days_acc': [], 'digits_acc': [], 'months_acc': [],
           'days_disc': [], 'digits_disc': [], 'months_disc': []}

    print(f'\n--- TRAINING: {N_EPOCHS} epochs ---', flush=True)
    for epoch in range(N_EPOCHS):
        # Present each sequence in the curriculum with pauses
        for seq_name, seq in curriculum:
            present_sequence(seq, symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult)
            present_pause(graph, ns, mp, device, theta, tau_mult)

        # Measure every 5 epochs
        if (epoch + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0

            # Test all transitions in each category
            results = {}
            for cat_name, cat_symbols in categories.items():
                n_correct = 0
                total_disc = 0
                n_tested = 0
                for i in range(len(cat_symbols) - 1):
                    pred = cat_symbols[i]
                    tgt = cat_symbols[i + 1]
                    c, at, ao, d = measure_prediction_accuracy(
                        pred, tgt, symbols, graph, ns, mp, device, theta,
                        stp, hom, ip, tau_mult, cat_symbols)
                    n_correct += int(c)
                    total_disc += d
                    n_tested += 1
                acc = n_correct / max(n_tested, 1) * 100
                avg_disc = total_disc / max(n_tested, 1)
                results[cat_name] = (acc, avg_disc)

            log['epoch'].append(epoch + 1)
            log['days_acc'].append(results['days'][0])
            log['digits_acc'].append(results['digits'][0])
            log['months_acc'].append(results['months'][0])
            log['days_disc'].append(results['days'][1])
            log['digits_disc'].append(results['digits'][1])
            log['months_disc'].append(results['months'][1])

            d_a, d_d = results['days']
            di_a, di_d = results['digits']
            m_a, m_d = results['months']
            print(f'  Epoch {epoch+1:3d}: '
                  f'Days={d_a:.0f}%({d_d:+.0f}%) '
                  f'Digits={di_a:.0f}%({di_d:+.0f}%) '
                  f'Months={m_a:.0f}%({m_d:+.0f}%) '
                  f'({elapsed:.0f}s)', flush=True)

    # ============ OFFSET RECOGNITION TEST ============
    print(f'\n--- OFFSET TEST ---', flush=True)
    print('  Training was: Mon->Tue->Wed->Thu->Fri->Sat->Sun', flush=True)
    print('  Testing: present Wed, does it predict Thu?', flush=True)

    # Present Wed only (never started a sequence with Wed during training)
    c, at, ao, d = measure_prediction_accuracy(
        'Wed', 'Thu', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['days'])
    print(f'  Wed->Thu: correct={c}, disc={d:+.1f}%', flush=True)

    c2, at2, ao2, d2 = measure_prediction_accuracy(
        'Fri', 'Sat', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['days'])
    print(f'  Fri->Sat: correct={c2}, disc={d2:+.1f}%', flush=True)

    c3, at3, ao3, d3 = measure_prediction_accuracy(
        '5', '6', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['digits'])
    print(f'  5->6: correct={c3}, disc={d3:+.1f}%', flush=True)

    c4, at4, ao4, d4 = measure_prediction_accuracy(
        'Aug', 'Sep', symbols, graph, ns, mp, device, theta, stp, hom, ip, tau_mult,
        categories['months'])
    print(f'  Aug->Sep: correct={c4}, disc={d4:+.1f}%', flush=True)

    # ============ RESULTS ============
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)

    final_days = log['days_acc'][-1] if log['days_acc'] else 0
    final_digits = log['digits_acc'][-1] if log['digits_acc'] else 0
    final_months = log['months_acc'][-1] if log['months_acc'] else 0
    avg_acc = (final_days + final_digits + final_months) / 3

    print(f'  Final accuracy: Days={final_days:.0f}% Digits={final_digits:.0f}% Months={final_months:.0f}%', flush=True)
    print(f'  Average: {avg_acc:.0f}%', flush=True)

    if avg_acc > 50:
        print(f'\n  VERDICT: SEQUENCES LEARNED -- majority correct predictions', flush=True)
    elif avg_acc > 25:
        print(f'\n  VERDICT: PARTIAL -- above chance, learning detected', flush=True)
    else:
        print(f'\n  VERDICT: NOT YET -- at or below chance', flush=True)

    torch.save(log, 'real_sequences_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
