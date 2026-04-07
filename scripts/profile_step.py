"""Profile every component of a training step at N=50K.

Uses CUDA events for accurate GPU timing without sync overhead.
Runs the actual training-loop operations (not Simulator.step()).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.types import EdgeType, NodeType

BASELINE = math.log(2)

# Same config as run_adaptive_consolidation.py
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
}

PLASTIC_EDGE_TYPES = [EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
                      EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE]

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

WEIGHT_DECAY = 0.013


def main():
    print("=" * 60)
    print("  STEP PROFILER — N=50K training loop")
    print("=" * 60)

    # Build graph
    print("\nBuilding graph...", flush=True)
    cfg = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(cfg)
    graph.initialize()
    device = graph.device
    ns = graph.node_state
    N = graph.n_nodes

    # Build hierarchy
    from graph_brain.hierarchy import HierarchyBuilder
    hb = HierarchyBuilder(cfg)
    tau_mult = hb.build(graph)

    mp = TypedMessagePasser(cfg, N, device)
    stp = ShortTermPlasticity(cfg.edges.stp)
    hom = HomeostaticScaling(cfg.edges.homeostatic)
    ip = IntrinsicPlasticity(cfg.nodes)

    # Pre-compute edge data (same as training script)
    edge_cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        edge_cache[et] = {
            'src64': store.src.long(),
            'dst64': store.dst.long(),
            'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps),
        }

    # Cache type masks
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()

    # Pre-allocate noise buffer
    noise_buf = torch.randn(100, N, device=device) * 0.005

    # Print graph stats
    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges")

    # Inject some activity so we measure realistic sparsity
    print("\nWarming up (100 steps)...", flush=True)
    input_region = torch.where(exc_mask)[0][:int(0.2 * 40000)]
    n_on = max(10, int(input_region.shape[0] * 0.10))
    pattern = input_region[torch.randperm(input_region.shape[0], device=device)[:n_on]]

    for _ in range(100):
        step = graph.step_count
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        # Send
        for et, ch in OUTPUT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]
            store = graph.edge_store(et)
            msg = output[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        for et, ch in CONTENT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]
            store = graph.edge_store(et)
            msg = content[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[c['src64']] - output[c['dst64']])
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, c['delay_steps'], step)
        # Read
        inputs = mp.read_inputs(step)
        inputs.basal[pattern.long()] += 2.0
        # Node update
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(min=0.0, max=10.0) * pv_gain * ns.gain, ns.output)
        ns.output += noise_buf[_ % 100]
        ns.output.clamp_(min=0.0, max=10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)
        # STP
        for et in PLASTIC_EDGE_TYPES:
            if graph.has_edge_type(et):
                stp.update(graph.edge_store(et), ns, 1.0)
        graph.increment_step()

    active_frac = (ns.output > 0).float().mean().item()
    print(f"  Activity: {active_frac*100:.1f}% nodes active")

    # ================================================================
    # PROFILING
    # ================================================================
    N_STEPS = 500
    print(f"\nProfiling {N_STEPS} steps...", flush=True)

    # CUDA events for each section
    sections = [
        'send_output', 'send_content', 'send_electrical',
        'read_buffer', 'node_update_exc', 'node_update_inh',
        'noise_clamp', 'stp_update', 'learning', 'total',
    ]
    times = {s: [] for s in sections}

    def cuda_time(start_evt, end_evt):
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt)  # ms

    for i in range(N_STEPS):
        step = graph.step_count
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)

        t_total_s = torch.cuda.Event(enable_timing=True)
        t_total_e = torch.cuda.Event(enable_timing=True)

        t_total_s.record()

        # --- SEND OUTPUT CHANNELS ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for et, ch in OUTPUT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]
            store = graph.edge_store(et)
            msg = output[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        t1.record()
        times['send_output'].append((t0, t1))

        # --- SEND CONTENT CHANNELS ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for et, ch in CONTENT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]
            store = graph.edge_store(et)
            msg = content[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        t1.record()
        times['send_content'].append((t0, t1))

        # --- SEND ELECTRICAL ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[c['src64']] - output[c['dst64']])
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, c['delay_steps'], step)
        t1.record()
        times['send_electrical'].append((t0, t1))

        # --- READ BUFFER ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        inputs = mp.read_inputs(step)
        if i % 3 == 0:  # inject input periodically
            inputs.basal[pattern.long()] += 2.0
        t1.record()
        times['read_buffer'].append((t0, t1))

        # --- NODE UPDATE: EXCITATORY ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        ns.basal += 1.0 * (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += 1.0 * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(min=0.0, max=10.0) * pv_gain * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        t1.record()
        times['node_update_exc'].append((t0, t1))

        # --- NODE UPDATE: INHIBITORY ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
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
        t1.record()
        times['node_update_inh'].append((t0, t1))

        # --- NOISE + CLAMP + EMA ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        ns.output += noise_buf[i % 100]
        ns.output.clamp_(min=0.0, max=10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)
        t1.record()
        times['noise_clamp'].append((t0, t1))

        # --- STP UPDATE ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for et in PLASTIC_EDGE_TYPES:
            if graph.has_edge_type(et):
                stp.update(graph.edge_store(et), ns, 1.0)
        t1.record()
        times['stp_update'].append((t0, t1))

        # --- LEARNING (apply_memory_learning equivalent) ---
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        pred_err_cached = ns.prediction_error
        global_novelty = pred_err_cached[exc_mask].abs().mean().clamp(min=0.01)
        for et in PLASTIC_EDGE_TYPES:
            if et not in edge_cache: continue
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            c = edge_cache[et]
            src = ns.output[c['src64']]
            dst = ns.output[c['dst64']]
            store.pre_trace.lerp_(src, 0.05)
            co_act = src * dst
            store.post_trace *= 0.999
            store.post_trace += co_act * 0.0001
            store.post_trace.clamp_(0.0, 1.0)
            stiffness = store.post_trace
            dst_error = pred_err_cached[c['dst64']].abs()
            error_gate = (dst_error / global_novelty).clamp(0.0, 3.0)
            error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
            unconsolidate_amount = 0.0001 * error_signal * stiffness * stiffness
            store.post_trace -= unconsolidate_amount
            store.post_trace.clamp_(0.0, 1.0)
            stiffness = store.post_trace
            plasticity = error_gate * (1.0 - 0.9 * stiffness)
            lr = 0.001
            dw = lr * plasticity * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)
        t1.record()
        times['learning'].append((t0, t1))

        t_total_e.record()
        times['total'].append((t_total_s, t_total_e))

        graph.increment_step()

    # ================================================================
    # RESULTS
    # ================================================================
    torch.cuda.synchronize()

    print("\n" + "=" * 60)
    print("  RESULTS (mean over %d steps)" % N_STEPS)
    print("=" * 60)

    # Skip first 50 steps (warmup)
    skip = 50
    results = {}
    for name in sections:
        evts = times[name][skip:]
        ms_list = [s.elapsed_time(e) for s, e in evts]
        results[name] = np.mean(ms_list)

    total = results['total']
    accounted = sum(results[s] for s in sections if s != 'total')

    # Sort by time (descending)
    ranked = sorted(
        [(s, results[s]) for s in sections if s != 'total'],
        key=lambda x: -x[1]
    )

    print(f"\n  {'Section':<22} {'Time (ms)':>10} {'% of step':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    for name, ms in ranked:
        pct = ms / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {name:<22} {ms:>10.3f} {pct:>9.1f}%  {bar}")

    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'accounted':<22} {accounted:>10.3f} {accounted/total*100:>9.1f}%")
    print(f"  {'total step':<22} {total:>10.3f} {'100.0':>9}%")
    print(f"  {'overhead (python)':<22} {total - accounted:>10.3f} {(total-accounted)/total*100:>9.1f}%")

    # Throughput
    steps_per_sec = 1000.0 / total
    print(f"\n  {steps_per_sec:.0f} steps/sec")
    print(f"  {total:.2f} ms/step")
    print(f"  100 steps/epoch x 1000 epochs = {100 * 1000 * total / 1000 / 60:.1f} min")

    # Message passing breakdown
    mp_total = results['send_output'] + results['send_content'] + results['send_electrical'] + results['read_buffer']
    node_total = results['node_update_exc'] + results['node_update_inh'] + results['noise_clamp']
    print(f"\n  Message passing total: {mp_total:.3f} ms ({mp_total/total*100:.1f}%)")
    print(f"  Node update total:    {node_total:.3f} ms ({node_total/total*100:.1f}%)")
    print(f"  STP total:            {results['stp_update']:.3f} ms ({results['stp_update']/total*100:.1f}%)")
    print(f"  Learning total:       {results['learning']:.3f} ms ({results['learning']/total*100:.1f}%)")


if __name__ == '__main__':
    main()
