"""Profile step v3: after pre-allocated temps + fused send."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import time
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.engine.fused_plasticity import FusedPlasticity, PLASTIC_EDGE_TYPES
from graph_brain.types import EdgeType, NodeType

BASELINE = math.log(2)

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
        'structural': {'enabled': True, 'update_interval': 500},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {
        'enabled': True, 'n_levels': 2, 'split_axis': 2,
        'time_scale_factor': 3.0, 'inter_level_k': 5,
        'inter_level_sigma': 0.5, 'inter_level_init_weight': 0.02,
    },
}

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


def main():
    print("=" * 60)
    print("  STEP PROFILER v3: pre-alloc temps + fused send")
    print("=" * 60)

    cfg = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(cfg)
    graph.initialize()
    device = graph.device
    ns = graph.node_state
    N = graph.n_nodes

    from graph_brain.hierarchy import HierarchyBuilder
    hb = HierarchyBuilder(cfg)
    hb.build(graph)

    mp = TypedMessagePasser(cfg, N, device)

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    sst_mask = ns.type_mask(NodeType.SST)
    exc_f = exc_mask.float()
    inh_mask = ~exc_mask
    inh_f = inh_mask.float()
    pv_f = ns.type_mask(NodeType.PV).float()

    edge_cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        edge_cache[et] = {'src64': store.src.long(), 'dst64': store.dst.long(),
                          'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)}

    noise_buf = torch.randn(100, N, device=device) * 0.005
    fused = FusedPlasticity(graph, cfg.edges.stp, mp.delay_buffer)

    input_region = torch.where(exc_mask)[0][:int(0.2 * 40000)]
    n_on = max(10, int(input_region.shape[0] * 0.10))
    pattern = input_region[torch.randperm(input_region.shape[0], device=device)[:n_on]]

    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges")

    N_ITER = 500
    WARMUP = 100
    SKIP = 50

    # Warmup
    print("\nWarming up...", flush=True)
    for i in range(WARMUP):
        step = graph.step_count
        fused.send(ns, mp.delay_buffer, step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (ns.output[c['src64']] - ns.output[c['dst64']])
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, c['delay_steps'], step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0) * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        inh_input = inputs.basal + inputs.electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * inh_f
        inh_out = torch.where(sst_mask, inh_out * torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0), inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)
        ns.output += noise_buf[i % 100]; ns.output.clamp_(0.0, 10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)
        fused.stp(ns.output)
        fused.learn(ns, exc_mask, 0.999)
        graph.increment_step()

    # Profile
    print(f"\nProfiling {N_ITER} steps...", flush=True)
    sections = ['fused_send', 'send_electrical', 'read_buffer',
                'node_exc', 'node_inh', 'noise_ema',
                'fused_stp', 'fused_learn', 'total']
    times = {s: [] for s in sections}

    for i in range(N_ITER):
        step = graph.step_count
        ts = torch.cuda.Event(enable_timing=True); te = torch.cuda.Event(enable_timing=True)
        ts.record()

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fused.send(ns, mp.delay_buffer, step)
        t1.record()
        times['fused_send'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (ns.output[c['src64']] - ns.output[c['dst64']])
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, c['delay_steps'], step)
        t1.record()
        times['send_electrical'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        t1.record()
        times['read_buffer'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * pv_gain * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        t1.record()
        times['node_exc'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        inh_input = inputs.basal + inputs.electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * inh_f
        sst_suppress = torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0)
        inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)
        t1.record()
        times['node_inh'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        ns.output += noise_buf[i % 100]; ns.output.clamp_(0.0, 10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)
        t1.record()
        times['noise_ema'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fused.stp(ns.output)
        t1.record()
        times['fused_stp'].append((t0, t1))

        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fused.learn(ns, exc_mask, 0.999)
        t1.record()
        times['fused_learn'].append((t0, t1))

        te.record()
        times['total'].append((ts, te))
        graph.increment_step()

    torch.cuda.synchronize()
    print(f"\n{'='*60}")
    print(f"  POST-v2 BREAKDOWN")
    print(f"{'='*60}")

    results = {}
    for name in sections:
        evts = times[name][SKIP:]
        results[name] = np.mean([s.elapsed_time(e) for s, e in evts])

    total = results['total']
    ranked = sorted([(s, results[s]) for s in sections if s != 'total'], key=lambda x: -x[1])

    print(f"\n  {'Section':<22} {'Time (ms)':>10} {'% of step':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    for name, ms in ranked:
        pct = ms / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {name:<22} {ms:>10.3f} {pct:>9.1f}%  {bar}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'total step':<22} {total:>10.3f}")

    mp_total = results['fused_send'] + results['send_electrical'] + results['read_buffer']
    node_total = results['node_exc'] + results['node_inh'] + results['noise_ema']
    plast_total = results['fused_stp'] + results['fused_learn']
    print(f"\n  Message passing: {mp_total:.3f} ms ({mp_total/total*100:.1f}%)")
    print(f"  Node dynamics:   {node_total:.3f} ms ({node_total/total*100:.1f}%)")
    print(f"  Plasticity:      {plast_total:.3f} ms ({plast_total/total*100:.1f}%)")
    print(f"\n  {1000/total:.0f} steps/sec | {total:.2f} ms/step")
    print(f"  1000ep x 100st = {1000*100*total/1000/60:.1f} min")

    # Compare to original 9.68 ms
    print(f"\n  vs original 9.68 ms: {9.68/total:.2f}x overall speedup")


if __name__ == '__main__':
    main()
