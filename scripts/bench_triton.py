"""Benchmark: old scatter vs Triton fused source-parallel kernel.

Run in WSL: source ~/gb_env/bin/activate && cd /mnt/c/Graph_Brain && python3 scripts/bench_triton.py
"""

import sys, time, math
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.types import EdgeType
from graph_brain.engine.triton_engine import TritonEngine

BASELINE = math.log(2)

CONFIG_50K = {
    'nodes': {'n_excitatory': 40000, 'n_pv': 3500, 'n_sst': 3500, 'n_vip': 3000, 'noise_std': 0.005},
    'edges': {'connectivity': {
        'driving': {'p_max': 0.3, 'sigma': 0.15, 'source_types': ['EXCITATORY'], 'target_types': ['EXCITATORY'], 'constant_k': 30},
        'modulatory': {'p_max': 0.2, 'sigma': 0.25, 'source_types': ['EXCITATORY'], 'target_types': ['EXCITATORY'], 'constant_k': 70},
        'inhib_perisomatic': {'p_max': 0.5, 'sigma': 0.10, 'source_types': ['PV'], 'target_types': ['EXCITATORY'], 'constant_k': 5},
        'inhib_dendritic': {'p_max': 0.4, 'sigma': 0.12, 'source_types': ['SST'], 'target_types': ['EXCITATORY', 'VIP'], 'constant_k': 5},
        'disinhibition': {'p_max': 0.4, 'sigma': 0.10, 'source_types': ['VIP'], 'target_types': ['SST'], 'constant_k': 10},
        'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'], 'target_types': ['PV'], 'constant_k': 5},
        'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'], 'target_types': ['EXCITATORY'], 'constant_k': 10},
        'max_radius': 0.5,
    }},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': True, 'n_levels': 2, 'split_axis': 2, 'time_scale_factor': 3.0, 'inter_level_k': 5, 'inter_level_sigma': 0.5, 'inter_level_init_weight': 0.02},
}

OUTPUT_CHS = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION, EdgeType.DISINHIBITION: Channel.VIP_INHIBITION, EdgeType.RETROGRADE: Channel.RETROGRADE}
CONTENT_CHS = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}

def precompute(graph, mp):
    cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        cache[et] = {'src64': store.src.long(), 'dst64': store.dst.long(), 'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)}
    return cache

def old_send(ns, graph, mp, cache):
    step = graph.step_count
    out, con = ns.output, F.softplus(ns.basal).clamp(max=10.0)
    for et, ch in OUTPUT_CHS.items():
        if et not in cache: continue
        c = cache[et]; s = graph.edge_store(et)
        mp.delay_buffer.write(ch, s.dst, out[c['src64']] * s.release_prob * s.weight, c['delay_steps'], step)
    for et, ch in CONTENT_CHS.items():
        if et not in cache: continue
        c = cache[et]; s = graph.edge_store(et)
        mp.delay_buffer.write(ch, s.dst, con[c['src64']] * s.release_prob * s.weight, c['delay_steps'], step)
    if EdgeType.ELECTRICAL in cache:
        c = cache[EdgeType.ELECTRICAL]; s = graph.edge_store(EdgeType.ELECTRICAL)
        mp.delay_buffer.write(Channel.ELECTRICAL, s.dst, s.weight * (out[c['src64']] - out[c['dst64']]), c['delay_steps'], step)

def make_sparse_activity(N, device, sparsity=0.17):
    output = torch.zeros(N, device=device)
    n_active = int(N * sparsity)
    active_idx = torch.randperm(N, device=device)[:n_active]
    output[active_idx] = torch.rand(n_active, device=device) * 2.0
    return output

def main():
    print('=' * 60)
    print('  TRITON BENCHMARK: scatter vs fused source-parallel')
    print('  N=50K, ~17% node activity (true silence)')
    print('=' * 60)

    print('\nBuilding graph...', flush=True)
    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config); graph.initialize()
    ns = graph.node_state; N = graph.n_nodes; device = graph.device
    HierarchyBuilder(config).build(graph)
    print(f'  {N:,} nodes, {graph.n_edges():,} edges')

    # Build all engines
    mp = TypedMessagePasser(config, N, device)
    cache = precompute(graph, mp)

    engine_src = TritonEngine(graph, mode='src')
    engine_dst = TritonEngine(graph, mode='dst')
    stats = engine_dst.stats()
    print(f'  Triton src: {engine_src.stats()["n_edge_types"]} types')
    print(f'  Triton dst: {stats["n_edge_types"]} types, {stats["total_edges"]:,} edges')

    # Init STP
    ns.output = make_sparse_activity(N, device)
    ns.basal = torch.randn(N, device=device) * 0.5
    stp = ShortTermPlasticity(config.edges.stp)
    for _ in range(10):
        for et in EdgeType:
            if graph.has_edge_type(et): stp.update(graph.edge_store(et), ns, 1.0)

    # ================================================================
    # CORRECTNESS
    # ================================================================
    print('\n--- CORRECTNESS CHECK ---')
    for label, engine in [('src', engine_src), ('dst', engine_dst)]:
        mp.delay_buffer.reset(); engine.reset()
        max_diff = 0.0
        for s in range(5):
            ns.output = make_sparse_activity(N, device)
            ns.basal = torch.randn(N, device=device) * 0.5
            output = ns.output
            content = F.softplus(ns.basal).clamp(max=10.0)
            graph._step_count = s
            old_send(ns, graph, mp, cache); old_in = mp.read_inputs(s)
            engine.send(graph, output, content, s); new_in = engine.read(s)
            for name in ['basal','apical','pv_inhibition','sst_inhibition','vip_inhibition','electrical','retrograde']:
                d = (getattr(old_in, name) - getattr(new_in, name)).abs().max().item()
                max_diff = max(max_diff, d)
        print(f'  Triton {label}: {"PASS" if max_diff < 1e-3 else "FAIL"} (max diff = {max_diff:.2e})')

    # ================================================================
    # PERFORMANCE
    # ================================================================
    n_warm, n_bench = 30, 300

    def bench(label, send_fn, read_fn, reset_fn, sparsity=0.17):
        reset_fn()
        for s in range(n_warm):
            if sparsity < 0.5:
                ns.output = make_sparse_activity(N, device, sparsity)
            else:
                ns.output = torch.randn(N, device=device).clamp(min=0, max=2.0)
            ns.basal = torch.randn(N, device=device) * 0.5
            send_fn(s); read_fn(s)
        reset_fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for s in range(n_bench):
            if sparsity < 0.5:
                ns.output = make_sparse_activity(N, device, sparsity)
            else:
                ns.output = torch.randn(N, device=device).clamp(min=0, max=2.0)
            ns.basal = torch.randn(N, device=device) * 0.5
            send_fn(s); read_fn(s)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_bench * 1000

    for sparsity_label, sp in [('17% sparse', 0.17), ('50% dense', 0.50)]:
        print(f'\n--- PERFORMANCE ({sparsity_label}) ---')
        old_ms = bench('old',
            lambda s: (setattr(graph, '_step_count', s), old_send(ns, graph, mp, cache)),
            lambda s: mp.read_inputs(s), lambda: mp.delay_buffer.reset(), sp)
        src_ms = bench('src',
            lambda s: engine_src.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s),
            lambda s: engine_src.read(s), lambda: engine_src.reset(), sp)
        dst_ms = bench('dst',
            lambda s: engine_dst.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s),
            lambda s: engine_dst.read(s), lambda: engine_dst.reset(), sp)
        print(f'  Old (scatter):     {old_ms:.2f} ms/step')
        print(f'  Triton src:        {src_ms:.2f} ms/step  ({old_ms/src_ms:.2f}x)')
        print(f'  Triton dst:        {dst_ms:.2f} ms/step  ({old_ms/dst_ms:.2f}x)')

    print(f'\n{"="*60}')
    print(f'  DONE')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
