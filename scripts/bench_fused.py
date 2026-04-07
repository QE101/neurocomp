"""Benchmark: original per-type loops vs fused plasticity.

Verifies correctness (identical results) and measures speedup.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import copy
import time

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
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


def original_stp(graph, stp, ns):
    """Original per-type STP loop."""
    for et in PLASTIC_EDGE_TYPES:
        if graph.has_edge_type(et):
            stp.update(graph.edge_store(et), ns, 1.0)


def original_learn(graph, ns, edge_cache, exc_mask, current_decay):
    """Original per-type learning loop (from run_adaptive_consolidation.py)."""
    pred_err = ns.prediction_error
    global_novelty = pred_err[exc_mask].abs().mean().clamp(min=0.01)

    for et in PLASTIC_EDGE_TYPES:
        if et not in edge_cache:
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        cache = edge_cache[et]
        src = ns.output[cache['src64']]
        dst = ns.output[cache['dst64']]

        store.pre_trace.lerp_(src, 0.05)

        co_act = src * dst
        store.post_trace *= current_decay
        store.post_trace += co_act * 0.0001
        store.post_trace.clamp_(0.0, 1.0)

        stiffness = store.post_trace
        dst_error = pred_err[cache['dst64']].abs()
        error_gate = (dst_error / global_novelty).clamp(0.0, 3.0)

        error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
        unconsolidate_amount = 0.0001 * error_signal * stiffness * stiffness
        store.post_trace -= unconsolidate_amount
        store.post_trace.clamp_(0.0, 1.0)
        stiffness = store.post_trace

        plasticity = error_gate * (1.0 - 0.9 * stiffness)

        if et == EdgeType.DRIVING:
            lr = 0.0001
            dw = lr * plasticity * (store.pre_trace * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.MODULATORY:
            lr = 0.001
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et == EdgeType.DISINHIBITION:
            lr = 0.002
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        elif et in (EdgeType.INHIB_PERISOMATIC, EdgeType.INHIB_DENDRITIC):
            lr = 0.0001
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)
        else:
            lr = 0.001
            dw = lr * plasticity * (src * dst - WEIGHT_DECAY * store.weight - dst * dst * store.weight)

        store.weight += dw
        store.weight.clamp_(0.0, 1.0)


def main():
    print("=" * 60)
    print("  FUSED PLASTICITY BENCHMARK")
    print("=" * 60)

    # Build graph
    print("\nBuilding graph...", flush=True)
    cfg = GraphBrainConfig.from_dict(CONFIG_50K)
    graph = NeuromorphicGraph(cfg)
    graph.initialize()
    device = graph.device
    ns = graph.node_state
    N = graph.n_nodes

    from graph_brain.hierarchy import HierarchyBuilder
    hb = HierarchyBuilder(cfg)
    hb.build(graph)

    stp = ShortTermPlasticity(cfg.edges.stp)
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)

    # Edge cache for original learning
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
        }

    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges")

    # Inject some activity
    input_region = torch.where(exc_mask)[0][:int(0.2 * 40000)]
    n_on = max(10, int(input_region.shape[0] * 0.10))
    pattern = input_region[torch.randperm(input_region.shape[0], device=device)[:n_on]]
    ns.output[pattern.long()] = 2.0
    ns.prediction_error[pattern.long()] = 0.5

    # ================================================================
    # CORRECTNESS CHECK
    # ================================================================
    print("\n--- CORRECTNESS CHECK ---")

    # Save original state
    orig_state = {}
    for et in PLASTIC_EDGE_TYPES:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        orig_state[et] = {
            'facilitation': store.facilitation.clone(),
            'depression': store.depression.clone(),
            'release_prob': store.release_prob.clone(),
            'weight': store.weight.clone(),
            'pre_trace': store.pre_trace.clone(),
            'post_trace': store.post_trace.clone(),
        }

    # Run original STP + learning
    original_stp(graph, stp, ns)
    original_learn(graph, ns, edge_cache, exc_mask, current_decay=0.999)

    # Save original results
    orig_results = {}
    for et in PLASTIC_EDGE_TYPES:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        orig_results[et] = {
            'facilitation': store.facilitation.clone(),
            'depression': store.depression.clone(),
            'release_prob': store.release_prob.clone(),
            'weight': store.weight.clone(),
            'pre_trace': store.pre_trace.clone(),
            'post_trace': store.post_trace.clone(),
        }

    # Restore original state
    for et in PLASTIC_EDGE_TYPES:
        if et not in orig_state:
            continue
        store = graph.edge_store(et)
        for field, val in orig_state[et].items():
            getattr(store, field).copy_(val) if hasattr(store, field) else setattr(store, field, val.clone())
        # Need to handle the case where fields might have been replaced
        store.facilitation = orig_state[et]['facilitation'].clone()
        store.depression = orig_state[et]['depression'].clone()
        store.release_prob = orig_state[et]['release_prob'].clone()
        store.weight = orig_state[et]['weight'].clone()
        store.pre_trace = orig_state[et]['pre_trace'].clone()
        store.post_trace = orig_state[et]['post_trace'].clone()

    # Run fused STP + learning
    fused = FusedPlasticity(graph, cfg.edges.stp)
    fused.stp(ns.output, dt=1.0)
    fused.learn(ns, exc_mask, current_decay=0.999)

    # Compare results
    max_diffs = {}
    all_pass = True
    for et in PLASTIC_EDGE_TYPES:
        if et not in orig_results:
            continue
        store = graph.edge_store(et)
        for field in ['facilitation', 'depression', 'release_prob', 'weight', 'pre_trace', 'post_trace']:
            orig_val = orig_results[et][field]
            fused_val = getattr(store, field)
            diff = (orig_val - fused_val).abs().max().item()
            key = f"{et.name}.{field}"
            max_diffs[key] = diff
            if diff > 1e-4:
                print(f"  FAIL: {key} max diff = {diff:.6e}")
                all_pass = False

    if all_pass:
        worst = max(max_diffs.values()) if max_diffs else 0
        print(f"  PASS: all fields match (worst diff = {worst:.2e})")
    else:
        print("  CORRECTNESS FAILED — aborting benchmark")
        return

    # ================================================================
    # SPEED BENCHMARK
    # ================================================================
    print("\n--- SPEED BENCHMARK ---")

    N_ITER = 300
    WARMUP = 50

    # --- Benchmark ORIGINAL ---
    # Restore state
    for et in PLASTIC_EDGE_TYPES:
        if et not in orig_state:
            continue
        store = graph.edge_store(et)
        store.facilitation = orig_state[et]['facilitation'].clone()
        store.depression = orig_state[et]['depression'].clone()
        store.release_prob = orig_state[et]['release_prob'].clone()
        store.weight = orig_state[et]['weight'].clone()
        store.pre_trace = orig_state[et]['pre_trace'].clone()
        store.post_trace = orig_state[et]['post_trace'].clone()

    # Re-create edge_cache (src/dst pointers may have changed)
    edge_cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        edge_cache[et] = {'src64': store.src.long(), 'dst64': store.dst.long()}

    # Warmup
    for _ in range(WARMUP):
        original_stp(graph, stp, ns)
        original_learn(graph, ns, edge_cache, exc_mask, 0.999)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        original_stp(graph, stp, ns)
        original_learn(graph, ns, edge_cache, exc_mask, 0.999)
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / N_ITER * 1000  # ms

    # --- Benchmark FUSED ---
    # Restore state and build fused
    for et in PLASTIC_EDGE_TYPES:
        if et not in orig_state:
            continue
        store = graph.edge_store(et)
        store.facilitation = orig_state[et]['facilitation'].clone()
        store.depression = orig_state[et]['depression'].clone()
        store.release_prob = orig_state[et]['release_prob'].clone()
        store.weight = orig_state[et]['weight'].clone()
        store.pre_trace = orig_state[et]['pre_trace'].clone()
        store.post_trace = orig_state[et]['post_trace'].clone()

    fused = FusedPlasticity(graph, cfg.edges.stp)

    # Warmup
    for _ in range(WARMUP):
        fused.stp(ns.output, dt=1.0)
        fused.learn(ns, exc_mask, 0.999)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fused.stp(ns.output, dt=1.0)
        fused.learn(ns, exc_mask, 0.999)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / N_ITER * 1000  # ms

    # --- Benchmark INHIBITORY NODE UPDATE ---
    # Original (loop)
    noise_buf = torch.randn(100, N, device=device) * 0.005
    inputs_basal = torch.randn(N, device=device) * 0.1
    inputs_electrical = torch.randn(N, device=device) * 0.01
    inputs_sst_inhibition = torch.randn(N, device=device).abs() * 0.1
    inputs_vip_inhibition = torch.randn(N, device=device).abs() * 0.1

    # Warmup
    for _ in range(WARMUP):
        for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
            f = mask.float()
            inp = inputs_basal + (inputs_electrical if inh_type == NodeType.PV else torch.zeros_like(inputs_basal))
            ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
            inh_raw = F.softplus(ns.basal) - BASELINE
            out = inh_raw.clamp(min=0.0, max=10.0) * ns.gain * f
            if inh_type == NodeType.SST:
                sst_suppress = torch.clamp(1.0 - inputs_sst_inhibition - inputs_vip_inhibition, min=0.0, max=1.0)
                out = out * sst_suppress
            ns.output = torch.where(mask, out, ns.output)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
            f = mask.float()
            inp = inputs_basal + (inputs_electrical if inh_type == NodeType.PV else torch.zeros_like(inputs_basal))
            ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
            inh_raw = F.softplus(ns.basal) - BASELINE
            out = inh_raw.clamp(min=0.0, max=10.0) * ns.gain * f
            if inh_type == NodeType.SST:
                sst_suppress = torch.clamp(1.0 - inputs_sst_inhibition - inputs_vip_inhibition, min=0.0, max=1.0)
                out = out * sst_suppress
            ns.output = torch.where(mask, out, ns.output)
    torch.cuda.synchronize()
    t_inh_orig = (time.perf_counter() - t0) / N_ITER * 1000

    # Fused inhibitory (vectorized, no loop)
    inh_mask = ~exc_mask
    inh_f = inh_mask.float()
    pv_f = pv_mask.float()
    sst_f = sst_mask.float()
    # Pre-allocate
    _sst_suppress_mask = sst_mask.clone()

    for _ in range(WARMUP):
        inh_input = inputs_basal + inputs_electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * inh_f
        sst_suppress = torch.clamp(1.0 - inputs_sst_inhibition - inputs_vip_inhibition, 0.0, 1.0)
        inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        inh_input = inputs_basal + inputs_electrical * pv_f
        ns.basal += (-ns.basal / 10.0 + inh_input) * inh_f
        inh_raw = F.softplus(ns.basal) - BASELINE
        inh_out = inh_raw.clamp(0.0, 10.0) * ns.gain * inh_f
        sst_suppress = torch.clamp(1.0 - inputs_sst_inhibition - inputs_vip_inhibition, 0.0, 1.0)
        inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
        ns.output = torch.where(inh_mask, inh_out, ns.output)
    torch.cuda.synchronize()
    t_inh_fused = (time.perf_counter() - t0) / N_ITER * 1000

    # ================================================================
    # RESULTS
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  STP + Learning:")
    print(f"    Original:  {t_orig:.2f} ms/step")
    print(f"    Fused:     {t_fused:.2f} ms/step")
    print(f"    Speedup:   {t_orig/t_fused:.2f}x")
    print(f"    Saved:     {t_orig - t_fused:.2f} ms/step")

    print(f"\n  Inhibitory node update:")
    print(f"    Original:  {t_inh_orig:.2f} ms/step")
    print(f"    Fused:     {t_inh_fused:.2f} ms/step")
    print(f"    Speedup:   {t_inh_orig/t_inh_fused:.2f}x")
    print(f"    Saved:     {t_inh_orig - t_inh_fused:.2f} ms/step")

    total_saved = (t_orig - t_fused) + (t_inh_orig - t_inh_fused)
    print(f"\n  Total saved: {total_saved:.2f} ms/step")
    # Original full step was ~10.5 ms
    orig_step = 10.54  # from profiler
    new_step = orig_step - total_saved
    print(f"  Original step: {orig_step:.2f} ms")
    print(f"  New step:      ~{new_step:.2f} ms")
    print(f"  Step speedup:  ~{orig_step/new_step:.2f}x")
    print(f"  Experiment:    ~{100 * 1000 * new_step / 1000 / 60:.1f} min (was 17.6 min)")


if __name__ == '__main__':
    main()
