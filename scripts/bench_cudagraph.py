"""Benchmark: CUDA Graphs — eliminate CPU-GPU round trips.

If the GPU fans aren't spinning, the bottleneck is Python/CPU dispatch latency,
not GPU compute. CUDA Graphs capture a sequence of kernel launches and replay
them as a single GPU-side operation with zero CPU overhead.

Run in WSL: source ~/gb_env/bin/activate && cd /mnt/c/Graph_Brain && python3 scripts/bench_cudagraph.py
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


def make_sparse_activity(N, device, sparsity=0.17):
    output = torch.zeros(N, device=device)
    n_active = int(N * sparsity)
    active_idx = torch.randperm(N, device=device)[:n_active]
    output[active_idx] = torch.rand(n_active, device=device) * 2.0
    return output


class CUDAGraphEngine:
    """Message passing captured as a CUDA Graph for zero-overhead replay.

    CUDA Graphs record GPU operations once, then replay the exact same
    sequence with no CPU involvement. The GPU runs the full send+read
    pipeline as a single monolithic operation.

    Constraint: all tensor sizes must be fixed (no dynamic shapes).
    This is fine for us — edge counts only change on structural plasticity.
    """

    def __init__(self, graph, config, dt=1.0):
        self.N = graph.n_nodes
        self.device = graph.device
        self.dt = dt

        max_delay_ms = config.edges.connectivity.max_radius * 10.0
        self.max_delay_steps = int(max_delay_ms / dt) + 2
        self.buf_len = self.max_delay_steps + 1

        # Delay buffer
        self._buffer = torch.zeros(N_CHANNELS, self.buf_len, self.N, device=self.device)

        # Pre-compute edge cache (same as test scripts)
        self._edge_data = {}
        for et in EdgeType:
            if not graph.has_edge_type(et): continue
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            self._edge_data[et] = {
                'src64': store.src.long(),
                'dst64': store.dst.long(),
                'delay_steps': (store.delay / dt).ceil().long().clamp(1, self.max_delay_steps),
            }

        # Static input tensors that the graph reads from (we write to these before replay)
        self._output = torch.zeros(self.N, device=self.device)
        self._content = torch.zeros(self.N, device=self.device)
        self._step_tensor = torch.tensor(0, dtype=torch.long, device=self.device)

        # Output tensor for read results
        self._read_result = torch.zeros(N_CHANNELS, self.N, device=self.device)

        # The captured graph
        self._graph_send = None
        self._graph_read = None

        self._capture(graph)

    def _send_impl(self, graph):
        """The actual send logic — called during capture and for correctness checks."""
        output = self._output
        content = self._content
        step = self._step_tensor

        for et, ch in OUTPUT_CHS.items():
            if et not in self._edge_data: continue
            cache = self._edge_data[et]
            store = graph.edge_store(et)
            msg = output[cache['src64']] * store.release_prob * store.weight
            target_steps = step + cache['delay_steps']
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + cache['dst64']
            self._buffer[ch].reshape(-1).index_add_(0, flat_idx, msg)

        for et, ch in CONTENT_CHS.items():
            if et not in self._edge_data: continue
            cache = self._edge_data[et]
            store = graph.edge_store(et)
            msg = content[cache['src64']] * store.release_prob * store.weight
            target_steps = step + cache['delay_steps']
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + cache['dst64']
            self._buffer[ch].reshape(-1).index_add_(0, flat_idx, msg)

        if EdgeType.ELECTRICAL in self._edge_data:
            cache = self._edge_data[EdgeType.ELECTRICAL]
            store = graph.edge_store(EdgeType.ELECTRICAL)
            gap = store.weight * (output[cache['src64']] - output[cache['dst64']])
            target_steps = step + cache['delay_steps']
            buf_indices = target_steps % self.buf_len
            flat_idx = buf_indices * self.N + cache['dst64']
            self._buffer[Channel.ELECTRICAL].reshape(-1).index_add_(0, flat_idx, gap)

    def _read_impl(self):
        """Read from delay buffer into static output tensor.
        Uses gather/scatter instead of fancy indexing (capturable by CUDA Graph).
        """
        buf_idx = self._step_tensor % self.buf_len
        # Expand buf_idx to gather shape: [N_CHANNELS, 1, N]
        idx = buf_idx.view(1, 1, 1).expand(N_CHANNELS, 1, self.N)
        # Gather one slice from dim=1: [N_CHANNELS, 1, N] -> squeeze -> [N_CHANNELS, N]
        self._read_result.copy_(self._buffer.gather(1, idx).squeeze(1))
        # Zero the read slot
        self._buffer.scatter_(1, idx, torch.zeros_like(idx, dtype=self._buffer.dtype))

    def _capture(self, graph):
        """Capture send and read as CUDA Graphs."""
        # Warm up (CUDA Graphs need a warm-up run to allocate intermediates)
        self._output.copy_(make_sparse_activity(self.N, self.device))
        self._content.copy_(torch.randn(self.N, device=self.device).abs())
        self._step_tensor.fill_(0)

        # Warm-up runs (ensures all kernels are compiled)
        for _ in range(3):
            self._send_impl(graph)
            self._read_impl()
        self._buffer.zero_()

        # Capture SEND graph
        self._graph_send = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph_send):
            self._send_impl(graph)

        # Capture READ graph
        self._graph_read = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph_read):
            self._read_impl()

    def send(self, output, content, step):
        """Copy inputs and replay captured graph."""
        self._output.copy_(output)
        self._content.copy_(content)
        self._step_tensor.fill_(step)
        self._graph_send.replay()

    def read(self, step):
        """Replay read graph and return CompartmentInputs."""
        self._step_tensor.fill_(step)
        self._graph_read.replay()
        from graph_brain.core.message_passing import CompartmentInputs
        return CompartmentInputs(
            basal=self._read_result[Channel.BASAL].clone(),
            apical=self._read_result[Channel.APICAL].clone(),
            pv_inhibition=self._read_result[Channel.PV_INHIBITION].clone(),
            sst_inhibition=self._read_result[Channel.SST_INHIBITION].clone(),
            vip_inhibition=self._read_result[Channel.VIP_INHIBITION].clone(),
            electrical=self._read_result[Channel.ELECTRICAL].clone(),
            retrograde=self._read_result[Channel.RETROGRADE].clone(),
        )

    def reset(self):
        self._buffer.zero_()


N_CHANNELS = 7


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
    print('=' * 60)
    print('  CUDA GRAPH BENCHMARK')
    print('  Zero CPU overhead — replay captured GPU operations')
    print('=' * 60)

    print('\nBuilding graph...', flush=True)
    config = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(config); graph.initialize()
    ns = graph.node_state; N = graph.n_nodes; device = graph.device
    HierarchyBuilder(config).build(graph)
    print(f'  {N:,} nodes, {graph.n_edges():,} edges')

    mp = TypedMessagePasser(config, N, device)
    cache = precompute(graph, mp)

    # Init STP
    ns.output = make_sparse_activity(N, device)
    ns.basal = torch.randn(N, device=device) * 0.5
    stp = ShortTermPlasticity(config.edges.stp)
    for _ in range(10):
        for et in EdgeType:
            if graph.has_edge_type(et): stp.update(graph.edge_store(et), ns, 1.0)

    print('\nCapturing CUDA Graph...', flush=True)
    cg_engine = CUDAGraphEngine(graph, config)
    print('  Captured.')

    # ================================================================
    # CORRECTNESS
    # ================================================================
    print('\n--- CORRECTNESS CHECK ---')
    mp.delay_buffer.reset(); cg_engine.reset()
    max_diff = 0.0
    for s in range(5):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        graph._step_count = s
        old_send(ns, graph, mp, cache); old_in = mp.read_inputs(s)
        cg_engine.send(output, content, s); new_in = cg_engine.read(s)
        for name in ['basal','apical','pv_inhibition','sst_inhibition','vip_inhibition','electrical','retrograde']:
            d = (getattr(old_in, name) - getattr(new_in, name)).abs().max().item()
            max_diff = max(max_diff, d)
    print(f'  {"PASS" if max_diff < 1e-3 else "FAIL"}: max diff = {max_diff:.2e}')

    # ================================================================
    # PERFORMANCE
    # ================================================================
    n_warm, n_bench = 50, 500

    # Old engine
    mp.delay_buffer.reset()
    for s in range(n_warm):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        graph._step_count = s; old_send(ns, graph, mp, cache); mp.read_inputs(s)
    mp.delay_buffer.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(n_bench):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        graph._step_count = s; old_send(ns, graph, mp, cache); mp.read_inputs(s)
    torch.cuda.synchronize(); old_ms = (time.perf_counter() - t0) / n_bench * 1000

    # CUDA Graph engine
    cg_engine.reset()
    for s in range(n_warm):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        cg_engine.send(ns.output, F.softplus(ns.basal).clamp(max=10.0), s)
        cg_engine.read(s)
    cg_engine.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(n_bench):
        ns.output = make_sparse_activity(N, device)
        ns.basal = torch.randn(N, device=device) * 0.5
        cg_engine.send(ns.output, F.softplus(ns.basal).clamp(max=10.0), s)
        cg_engine.read(s)
    torch.cuda.synchronize(); cg_ms = (time.perf_counter() - t0) / n_bench * 1000

    speedup = old_ms / cg_ms

    print(f'\n--- RESULTS ---')
    print(f'  Old (scatter):  {old_ms:.2f} ms/step')
    print(f'  CUDA Graph:     {cg_ms:.2f} ms/step')
    print(f'  Speedup:        {speedup:.2f}x')
    if speedup > 1.5:
        print(f'  Per epoch: {old_ms*1500/1000:.1f}s -> {cg_ms*1500/1000:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
