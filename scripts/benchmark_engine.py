"""Benchmark: Old scatter vs SpMV vs Active-gated message passing.

Tests correctness and performance of all three engines at N=50K
with realistic true-silence sparsity (~17% active nodes).
"""

import sys, os, time, math
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.types import EdgeType, NodeType
from graph_brain.engine import SparseEngine, ActiveEngine

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
    },
    'simulation': {'device': 'cuda', 'seed': 42},
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


def precompute_edge_data(graph, mp):
    cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        cache[et] = {
            'src64': store.src.long(),
            'dst64': store.dst.long(),
            'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps),
        }
    return cache


def old_send(ns, graph, mp, edge_cache, device):
    step = graph.step_count
    output = ns.output
    content = F.softplus(ns.basal).clamp(max=10.0)
    for et, ch in OUTPUT_EDGE_CHANNELS.items():
        if et not in edge_cache: continue
        cache = edge_cache[et]
        store = graph.edge_store(et)
        msg = output[cache['src64']] * store.release_prob * store.weight
        mp.delay_buffer.write(ch, store.dst, msg, cache['delay_steps'], step)
    for et, ch in CONTENT_EDGE_CHANNELS.items():
        if et not in edge_cache: continue
        cache = edge_cache[et]
        store = graph.edge_store(et)
        msg = content[cache['src64']] * store.release_prob * store.weight
        mp.delay_buffer.write(ch, store.dst, msg, cache['delay_steps'], step)
    if EdgeType.ELECTRICAL in edge_cache:
        cache = edge_cache[EdgeType.ELECTRICAL]
        store = graph.edge_store(EdgeType.ELECTRICAL)
        gap = store.weight * (output[cache['src64']] - output[cache['dst64']])
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap, cache['delay_steps'], step)


def make_sparse_activity(N, device, sparsity=0.17):
    """Create realistic true-silence activity: ~17% non-zero."""
    output = torch.zeros(N, device=device)
    n_active = int(N * sparsity)
    active_idx = torch.randperm(N, device=device)[:n_active]
    output[active_idx] = torch.rand(n_active, device=device) * 2.0
    return output


def bench_engine(name, send_fn, read_fn, reset_fn, graph, ns, N, device, n_warmup=20, n_bench=200):
    """Benchmark one engine. Returns ms/step."""
    reset_fn()
    for s in range(n_warmup):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        send_fn(s)
        read_fn(s)

    reset_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(n_bench):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        send_fn(s)
        read_fn(s)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_bench * 1000


def main():
    print('=' * 60)
    print('  ENGINE BENCHMARK: scatter vs SpMV vs Active-gated')
    print('  N=50K, ~17% node activity (true silence)')
    print('=' * 60)

    print('\nBuilding graph...', flush=True)
    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    builder = HierarchyBuilder(config)
    builder.build(graph)

    print(f'  N={N:,} nodes, {graph.n_edges():,} edges')
    for et in EdgeType:
        if graph.has_edge_type(et):
            print(f'    {et.name}: {graph.n_edges(et):,}')

    # Build all three engines
    print('\nBuilding engines...', flush=True)

    mp = TypedMessagePasser(config, N, device)
    edge_cache = precompute_edge_data(graph, mp)
    print(f'  Old (scatter): ready')

    spmv_engine = SparseEngine(graph)
    stats = spmv_engine.stats()
    print(f'  SpMV: {stats["n_groups"]} CSR groups, {stats["total_nnz"]:,} nnz')

    active_engine = ActiveEngine(graph)
    stats_a = active_engine.stats()
    print(f'  Active: {stats_a["n_edge_types"]} edge types, {stats_a["total_edges"]:,} edges')

    # Init STP
    ns.output = make_sparse_activity(N, device)
    ns.basal = torch.randn(N, device=device) * 0.5
    stp = ShortTermPlasticity(config.edges.stp)
    for _ in range(10):
        for et in EdgeType:
            if graph.has_edge_type(et):
                stp.update(graph.edge_store(et), ns, 1.0)

    # Check sparsity
    pct_active = (ns.output > 0).float().mean().item() * 100
    print(f'  Activity: {pct_active:.1f}% nodes active')

    # ================================================================
    # CORRECTNESS CHECK
    # ================================================================
    print('\n--- CORRECTNESS CHECK ---')

    # Compare all three engines over 5 steps
    for engine_name, engine_obj in [('SpMV', spmv_engine), ('Active', active_engine)]:
        mp.delay_buffer.reset()
        engine_obj.reset()
        max_diff = 0.0

        for s in range(5):
            ns.output = make_sparse_activity(N, device)
            ns.basal = torch.randn(N, device=device) * 0.5
            output = ns.output
            content = F.softplus(ns.basal).clamp(max=10.0)

            graph._step_count = s
            old_send(ns, graph, mp, edge_cache, device)
            old_inputs = mp.read_inputs(s)

            engine_obj.send(graph, output, content, s)
            new_inputs = engine_obj.read(s)

            for name in ['basal', 'apical', 'pv_inhibition', 'sst_inhibition',
                          'vip_inhibition', 'electrical', 'retrograde']:
                old_val = getattr(old_inputs, name)
                new_val = getattr(new_inputs, name)
                diff = (old_val - new_val).abs().max().item()
                max_diff = max(max_diff, diff)

        status = 'PASS' if max_diff < 1e-4 else 'FAIL'
        print(f'  {engine_name}: {status} (max diff = {max_diff:.2e})')

    # ================================================================
    # PERFORMANCE BENCHMARK
    # ================================================================
    print('\n--- PERFORMANCE BENCHMARK (17% sparsity) ---')

    n_warmup = 20
    n_bench = 200

    # Old engine
    old_ms = bench_engine(
        'Old',
        lambda s: old_send(ns, graph, mp, edge_cache, device) or setattr(graph, '_step_count', s),
        lambda s: mp.read_inputs(s),
        lambda: mp.delay_buffer.reset(),
        graph, ns, N, device, n_warmup, n_bench,
    )

    # SpMV engine
    spmv_ms = bench_engine(
        'SpMV',
        lambda s: spmv_engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s),
        lambda s: spmv_engine.read(s),
        lambda: spmv_engine.reset(),
        graph, ns, N, device, n_warmup, n_bench,
    )

    # Active engine
    active_ms = bench_engine(
        'Active',
        lambda s: active_engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s),
        lambda s: active_engine.read(s),
        lambda: active_engine.reset(),
        graph, ns, N, device, n_warmup, n_bench,
    )

    print(f'  Old (scatter):  {old_ms:.2f} ms/step')
    print(f'  SpMV (CSR):     {spmv_ms:.2f} ms/step  ({old_ms/spmv_ms:.2f}x)')
    print(f'  Active (gated): {active_ms:.2f} ms/step  ({old_ms/active_ms:.2f}x)')

    # Also test at 50% activity (non-sparse regime) for comparison
    print('\n--- PERFORMANCE BENCHMARK (50% sparsity) ---')

    def make_dense():
        ns.output = torch.randn(N, device=device).clamp(min=0.0, max=2.0)
        ns.basal = torch.randn(N, device=device) * 0.5

    old_dense = bench_engine(
        'Old-dense',
        lambda s: old_send(ns, graph, mp, edge_cache, device) or setattr(graph, '_step_count', s),
        lambda s: mp.read_inputs(s),
        lambda: mp.delay_buffer.reset(),
        graph, ns, N, device, n_warmup, n_bench,
    )
    # Override the activity function for active engine test
    active_dense_reset = lambda: active_engine.reset()
    active_engine.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(n_bench):
        ns.output = torch.randn(N, device=device).clamp(min=0.0, max=2.0)
        ns.basal = torch.randn(N, device=device) * 0.5
        active_engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s)
        active_engine.read(s)
    torch.cuda.synchronize()
    active_dense = (time.perf_counter() - t0) / n_bench * 1000

    print(f'  Old (scatter):  {old_dense:.2f} ms/step')
    print(f'  Active (gated): {active_dense:.2f} ms/step  ({old_dense/active_dense:.2f}x)')

    # Summary
    print(f'\n{"="*60}')
    print(f'  RESULTS SUMMARY')
    print(f'  At 17% activity (true silence):')
    print(f'    Old:    {old_ms:.2f} ms/step')
    print(f'    Active: {active_ms:.2f} ms/step ({old_ms/active_ms:.2f}x)')
    if old_ms/active_ms > 1.5:
        epochs_old = old_ms * 1500 / 1000
        epochs_new = active_ms * 1500 / 1000
        print(f'    Per epoch: {epochs_old:.1f}s -> {epochs_new:.1f}s')
    print(f'  At 50% activity (dense):')
    print(f'    Old:    {old_dense:.2f} ms/step')
    print(f'    Active: {active_dense:.2f} ms/step ({old_dense/active_dense:.2f}x)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
