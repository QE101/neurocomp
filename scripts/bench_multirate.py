"""Benchmark: multi-rate time stepping.

Fast path (every step): message passing + node update
Medium path (every N_STP steps): STP update
Slow path (every N_LEARN steps): full learning

Tests correctness (drift from every-step baseline) and speed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import time
import copy

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

OUTPUT_CH = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
             EdgeType.DISINHIBITION: Channel.VIP_INHIBITION, EdgeType.RETROGRADE: Channel.RETROGRADE}
CONTENT_CH = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}


def main():
    print("=" * 60)
    print("  MULTI-RATE TIME STEPPING BENCHMARK")
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
    pattern = torch.where(exc_mask)[0][:800]
    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges")

    # Inject activity
    ns.output = torch.rand(N, device=device) * 0.5
    ns.output[torch.rand(N, device=device) > 0.17] = 0.0
    ns.prediction_error = torch.randn(N, device=device) * 0.1

    fused = FusedPlasticity(graph, cfg.edges.stp)
    fused.enable_compile()

    def send_messages(step):
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        for et, ch in OUTPUT_CH.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, output[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        for et, ch in CONTENT_CH.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, content[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, store.weight * (output[c['src64']] - output[c['dst64']]), c['delay_steps'], step)

    def node_update(inputs, i):
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0) * ns.gain, ns.output)
        ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
        inh_input = inputs.basal + inputs.electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_out = (F.softplus(ns.basal) - BASELINE).clamp(0.0, 10.0) * ns.gain * inh_f
        inh_out = torch.where(sst_mask, inh_out * torch.clamp(1.0 - inputs.sst_inhibition - inputs.vip_inhibition, 0.0, 1.0), inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)
        ns.output += noise_buf[i % 100]; ns.output.clamp_(0.0, 10.0)
        ns.activity_ema.lerp_(ns.output, 0.001)

    # ================================================================
    # Test different multi-rate configs
    # ================================================================
    configs = [
        ('every-step (baseline)', 1, 1),
        ('STP/2, learn/5', 2, 5),
        ('STP/3, learn/5', 3, 5),
        ('STP/5, learn/10', 5, 10),
        ('STP/5, learn/20', 5, 20),
        ('STP/10, learn/20', 10, 20),
    ]

    N_ITER = 500
    WARMUP = 100

    for label, n_stp, n_learn in configs:
        # Reset state for fair comparison
        ns.output = torch.rand(N, device=device) * 0.5
        ns.output[torch.rand(N, device=device) > 0.17] = 0.0
        ns.prediction_error = torch.randn(N, device=device) * 0.1
        ns.basal.zero_(); ns.apical.zero_()
        mp.delay_buffer.reset()
        # Reset fused state
        fused.f_facilitation.zero_()
        fused.f_depression.fill_(1.0)
        fused.f_release_prob.fill_(0.1)
        fused.f_pre_trace.zero_()
        fused.f_post_trace.zero_()
        # Reset weights to a known state
        torch.manual_seed(42)
        fused.f_weight.uniform_(0.01, 0.3)

        # Warmup
        for i in range(WARMUP):
            step = graph.step_count
            send_messages(step)
            inputs = mp.read_inputs(step)
            if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
            node_update(inputs, i)
            if i % n_stp == 0:
                fused.stp(ns.output)
            if i % n_learn == 0:
                fused.learn(ns, exc_mask, 0.999)
            graph.increment_step()

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N_ITER):
            step = graph.step_count
            send_messages(step)
            inputs = mp.read_inputs(step)
            if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
            node_update(inputs, i)
            if i % n_stp == 0:
                fused.stp(ns.output)
            if i % n_learn == 0:
                fused.learn(ns, exc_mask, 0.999)
            graph.increment_step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITER * 1000

        # Check activity stats
        active = (ns.output > 0).float().mean().item() * 100
        out_mean = ns.output.mean().item()
        w_mean = fused.f_weight.mean().item()
        rp_mean = fused.f_release_prob.mean().item()

        print(f"  {label:30s}  {ms:.2f} ms  {1000/ms:4.0f} stp/s  "
              f"active={active:.0f}%  out={out_mean:.4f}  w={w_mean:.4f}  rp={rp_mean:.4f}")

    # ================================================================
    # Correctness: compare weight/trace drift vs baseline after 500 steps
    # ================================================================
    print(f"\n--- CORRECTNESS (weight drift after 500 steps) ---")

    for label, n_stp, n_learn in [('every-step', 1, 1), ('STP/5 learn/10', 5, 10)]:
        torch.manual_seed(123)
        ns.output = torch.rand(N, device=device) * 0.5
        ns.output[torch.rand(N, device=device) > 0.17] = 0.0
        ns.prediction_error = torch.randn(N, device=device) * 0.1
        ns.basal.zero_(); ns.apical.zero_()
        mp.delay_buffer.reset()
        fused.f_facilitation.zero_()
        fused.f_depression.fill_(1.0)
        fused.f_release_prob.fill_(0.1)
        fused.f_pre_trace.zero_()
        fused.f_post_trace.zero_()
        torch.manual_seed(42)
        fused.f_weight.uniform_(0.01, 0.3)

        w_initial = fused.f_weight.clone()

        for i in range(500):
            step = graph.step_count
            send_messages(step)
            inputs = mp.read_inputs(step)
            if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
            node_update(inputs, i)
            if i % n_stp == 0:
                fused.stp(ns.output)
            if i % n_learn == 0:
                fused.learn(ns, exc_mask, 0.999)
            graph.increment_step()

        w_delta = (fused.f_weight - w_initial).abs()
        print(f"  {label:20s}  w_change: mean={w_delta.mean().item():.6f}  "
              f"max={w_delta.max().item():.6f}  "
              f"rp={fused.f_release_prob.mean().item():.4f}  "
              f"fac={fused.f_facilitation.mean().item():.4f}")


if __name__ == '__main__':
    main()
