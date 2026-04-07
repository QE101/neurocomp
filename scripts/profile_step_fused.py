"""Profile a full training step with fused plasticity.

Compares old vs new step time end-to-end.
"""

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
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.engine.fused_plasticity import FusedPlasticity, PLASTIC_EDGE_TYPES, WEIGHT_DECAY
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
    print("  END-TO-END STEP PROFILER: OLD vs FUSED")
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
    stp_obj = ShortTermPlasticity(cfg.edges.stp)

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()
    inh_mask = ~exc_mask
    inh_f = inh_mask.float()
    pv_f = pv_mask.float()

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

    input_region = torch.where(exc_mask)[0][:int(0.2 * 40000)]
    n_on = max(10, int(input_region.shape[0] * 0.10))
    pattern = input_region[torch.randperm(input_region.shape[0], device=device)[:n_on]]

    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges\n")

    N_ITER = 500
    WARMUP = 100

    def send_messages(step):
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        for et, ch in OUTPUT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            msg = output[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        for et, ch in CONTENT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            msg = content[c['src64']] * store.release_prob * store.weight
            mp.delay_buffer.write(ch, store.dst, msg, c['delay_steps'], step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[c['src64']] - output[c['dst64']])
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, c['delay_steps'], step)

    def node_update_old(inputs, i):
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * pv_gain * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        # OLD inhibitory loop
        for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
            f = mask.float()
            inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
            ns.basal += (-ns.basal / 10.0 + inp) * f
            inh_raw = F.softplus(ns.basal) - BASELINE
            out = inh_raw.clamp(0.0, 10.0) * ns.gain * f
            if inh_type == NodeType.SST:
                out = out * torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0)
            ns.output = torch.where(mask, out, ns.output)
        ns.output += noise_buf[i % 100]
        ns.output.clamp_(0.0, 10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)

    def node_update_fused(inputs, i):
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * pv_gain * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        # FUSED inhibitory (vectorized)
        inh_input = inputs.basal + inputs.electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * inh_f
        sst_suppress = torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0)
        inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)
        ns.output += noise_buf[i % 100]
        ns.output.clamp_(0.0, 10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)

    def old_stp_learn():
        for et in PLASTIC_EDGE_TYPES:
            if graph.has_edge_type(et):
                stp_obj.update(graph.edge_store(et), ns, 1.0)
        pred_err = ns.prediction_error
        global_novelty = pred_err[exc_mask].abs().mean().clamp(min=0.01)
        for et in PLASTIC_EDGE_TYPES:
            if et not in edge_cache: continue
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            c = edge_cache[et]
            src = ns.output[c['src64']]; dst = ns.output[c['dst64']]
            store.pre_trace.lerp_(src, 0.05)
            co_act = src * dst
            store.post_trace *= 0.999
            store.post_trace += co_act * 0.0001
            store.post_trace.clamp_(0.0, 1.0)
            stiffness = store.post_trace
            dst_error = pred_err[c['dst64']].abs()
            error_gate = (dst_error / global_novelty).clamp(0.0, 3.0)
            error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
            store.post_trace -= 0.0001 * error_signal * stiffness * stiffness
            store.post_trace.clamp_(0.0, 1.0)
            plasticity = error_gate * (1.0 - 0.9 * store.post_trace)
            lr = 0.001
            dw = lr * plasticity * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

    # ================================================================
    # OLD STEP
    # ================================================================
    print("Benchmarking OLD step...", flush=True)
    for i in range(WARMUP):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update_old(inputs, i)
        old_stp_learn()
        graph.increment_step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update_old(inputs, i)
        old_stp_learn()
        graph.increment_step()
    torch.cuda.synchronize()
    t_old = (time.perf_counter() - t0) / N_ITER * 1000

    # ================================================================
    # FUSED STEP
    # ================================================================
    print("Benchmarking FUSED step...", flush=True)

    # Rebuild fused (edges may have drifted during old benchmark)
    fused = FusedPlasticity(graph, cfg.edges.stp)

    for i in range(WARMUP):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update_fused(inputs, i)
        fused.stp(ns.output)
        fused.learn(ns, exc_mask, 0.999)
        graph.increment_step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update_fused(inputs, i)
        fused.stp(ns.output)
        fused.learn(ns, exc_mask, 0.999)
        graph.increment_step()
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITER * 1000

    # ================================================================
    # RESULTS
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  END-TO-END RESULTS ({N_ITER} steps)")
    print(f"{'='*60}")
    print(f"  Old step:    {t_old:.2f} ms  ({1000/t_old:.0f} steps/sec)")
    print(f"  Fused step:  {t_fused:.2f} ms  ({1000/t_fused:.0f} steps/sec)")
    print(f"  Speedup:     {t_old/t_fused:.2f}x")
    print(f"  Saved:       {t_old - t_fused:.2f} ms/step")
    print(f"\n  1000 epochs x 100 steps:")
    print(f"    Old:   {1000 * 100 * t_old / 1000 / 60:.1f} min")
    print(f"    Fused: {1000 * 100 * t_fused / 1000 / 60:.1f} min")
    print(f"    Saved: {1000 * 100 * (t_old - t_fused) / 1000 / 60:.1f} min")


if __name__ == '__main__':
    main()
