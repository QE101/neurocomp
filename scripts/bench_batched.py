"""Quick benchmark: old scatter vs batched engine."""
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
from graph_brain.engine.batched_engine import BatchedEngine

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

def main():
    print('Building graph...', flush=True)
    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config); graph.initialize()
    ns = graph.node_state; N = graph.n_nodes; device = graph.device
    HierarchyBuilder(config).build(graph)
    print(f'  {N:,} nodes, {graph.n_edges():,} edges')

    mp = TypedMessagePasser(config, N, device)
    cache = precompute(graph, mp)
    engine = BatchedEngine(graph)
    stats = engine.stats()
    print(f'  Batched: {stats["output_edges"]:,} output + {stats["content_edges"]:,} content, {stats["kernel_launches_per_step"]} launches/step')

    # Init STP
    stp = ShortTermPlasticity(config.edges.stp)
    ns.output = torch.randn(N, device=device).clamp(min=0.0, max=2.0)
    ns.basal = torch.randn(N, device=device) * 0.5
    for _ in range(10):
        for et in EdgeType:
            if graph.has_edge_type(et): stp.update(graph.edge_store(et), ns, 1.0)

    # Correctness
    print('\nCorrectness...', flush=True)
    mp.delay_buffer.reset(); engine.reset()
    max_diff = 0.0
    for s in range(5):
        ns.output = torch.randn(N, device=device).clamp(min=0.0, max=2.0)
        ns.basal = torch.randn(N, device=device) * 0.5
        graph._step_count = s
        old_send(ns, graph, mp, cache); old_in = mp.read_inputs(s)
        engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s); new_in = engine.read(s)
        for name in ['basal','apical','pv_inhibition','sst_inhibition','vip_inhibition','electrical','retrograde']:
            d = (getattr(old_in, name) - getattr(new_in, name)).abs().max().item()
            max_diff = max(max_diff, d)
    print(f'  {"PASS" if max_diff < 1e-4 else "FAIL"}: max diff = {max_diff:.2e}')

    # Benchmark
    n_warm, n_bench = 20, 300
    def rand_activity():
        ns.output = torch.randn(N, device=device).clamp(min=0.0, max=2.0)
        ns.basal = torch.randn(N, device=device) * 0.5

    # Old
    mp.delay_buffer.reset()
    for s in range(n_warm): rand_activity(); graph._step_count = s; old_send(ns, graph, mp, cache); mp.read_inputs(s)
    mp.delay_buffer.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(n_bench): rand_activity(); graph._step_count = s; old_send(ns, graph, mp, cache); mp.read_inputs(s)
    torch.cuda.synchronize(); old_ms = (time.perf_counter() - t0) / n_bench * 1000

    # Batched
    engine.reset()
    for s in range(n_warm): rand_activity(); engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s); engine.read(s)
    engine.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(n_bench): rand_activity(); engine.send(graph, ns.output, F.softplus(ns.basal).clamp(max=10.0), s); engine.read(s)
    torch.cuda.synchronize(); new_ms = (time.perf_counter() - t0) / n_bench * 1000

    print(f'\nOld:     {old_ms:.2f} ms/step')
    print(f'Batched: {new_ms:.2f} ms/step  ({old_ms/new_ms:.2f}x)')

if __name__ == '__main__':
    main()
