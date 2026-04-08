"""Benchmark: Morton reordering + CUDA stream overlap + shared gather.

Tests each optimization individually and combined.
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
from graph_brain.engine.fused_plasticity import (
    FusedPlasticity, PLASTIC_EDGE_TYPES, _stp_core, _learn_core
)
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


# ================================================================
# MORTON CODE (Z-order curve) for 3D spatial locality
# ================================================================

def spread_bits_32(v):
    """Spread 10-bit integer into every 3rd bit position for 3D Morton code."""
    v = v & 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def morton_code_3d(positions, bits=10):
    """Compute Morton (Z-order) codes for 3D positions in [0,1]^3.

    Args:
        positions: [N, 3] float tensor in [0, 1]
        bits: precision (10 bits = 1024 levels per axis)

    Returns:
        [N] int64 Morton codes
    """
    scale = (1 << bits) - 1
    # Quantize to integer grid
    ix = (positions[:, 0].clamp(0, 1) * scale).long().cpu().numpy()
    iy = (positions[:, 1].clamp(0, 1) * scale).long().cpu().numpy()
    iz = (positions[:, 2].clamp(0, 1) * scale).long().cpu().numpy()

    codes = np.zeros(len(ix), dtype=np.int64)
    for i in range(len(ix)):
        codes[i] = spread_bits_32(int(ix[i])) | (spread_bits_32(int(iy[i])) << 1) | (spread_bits_32(int(iz[i])) << 2)
    return codes


def reorder_graph_morton(graph):
    """Reorder all node indices by Morton code for cache-friendly access.

    One-time cost. All edge src/dst indices are remapped.
    Returns the permutation for reference.
    """
    ns = graph.node_state
    N = ns.n_nodes
    device = ns.position.device

    print("  Computing Morton codes...", flush=True)
    codes = morton_code_3d(ns.position)
    perm = np.argsort(codes)
    perm_t = torch.from_numpy(perm).long().to(device)
    inv_perm = torch.zeros(N, dtype=torch.int64, device=device)
    inv_perm[perm_t] = torch.arange(N, dtype=torch.int64, device=device)

    # Reorder ALL node state arrays
    print("  Reordering node state...", flush=True)
    ns.position = ns.position[perm_t]
    ns.basal = ns.basal[perm_t]
    ns.apical = ns.apical[perm_t]
    ns.output = ns.output[perm_t]
    ns.prediction_error = ns.prediction_error[perm_t]
    ns.gain = ns.gain[perm_t]
    ns.activity_ema = ns.activity_ema[perm_t]
    ns.threshold = ns.threshold[perm_t]
    ns.last_spike_time = ns.last_spike_time[perm_t]
    ns.node_type = ns.node_type[perm_t]
    if hasattr(ns, 'hierarchy_level'):
        ns.hierarchy_level = ns.hierarchy_level[perm_t]

    # Remap ALL edge src/dst indices
    print("  Remapping edge indices...", flush=True)
    for et in EdgeType:
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        # Remap: new_idx = inv_perm[old_idx]
        store.src = inv_perm[store.src.long()].to(store.src.dtype)
        store.dst = inv_perm[store.dst.long()].to(store.dst.dtype)
        # Re-sort by destination (required for CSR dst_ptr)
        sort_order = torch.argsort(store.dst.long())
        store.src = store.src[sort_order]
        store.dst = store.dst[sort_order]
        store.weight = store.weight[sort_order]
        store.delay = store.delay[sort_order]
        store.release_prob = store.release_prob[sort_order]
        store.facilitation = store.facilitation[sort_order]
        store.depression = store.depression[sort_order]
        store.pre_trace = store.pre_trace[sort_order]
        store.post_trace = store.post_trace[sort_order]
        # Rebuild CSR pointer
        from graph_brain.core.graph import build_dst_ptr
        store.dst_ptr = build_dst_ptr(store.dst, N)

    print(f"  Morton reorder complete ({N:,} nodes remapped)", flush=True)
    return perm_t


def main():
    print("=" * 60)
    print("  OPTIMIZATION BENCHMARK")
    print("  Morton reorder + Stream overlap + Shared gather")
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

    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges\n")

    # Inject realistic activity (17% active, true silence)
    ns.output = torch.rand(N, device=device) * 0.5
    ns.output[torch.rand(N, device=device) > 0.17] = 0.0
    ns.prediction_error = torch.randn(N, device=device) * 0.1

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

    OUTPUT_CH = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
                 EdgeType.DISINHIBITION: Channel.VIP_INHIBITION, EdgeType.RETROGRADE: Channel.RETROGRADE}
    CONTENT_CH = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}

    N_ITER = 300
    WARMUP = 100

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
    # BASELINE: current fused + compiled (no Morton, no streams)
    # ================================================================
    print("--- BASELINE (fused + compiled) ---")
    fused = FusedPlasticity(graph, cfg.edges.stp)
    fused.enable_compile()

    pattern = torch.where(exc_mask)[0][:800]
    for i in range(WARMUP):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update(inputs, i)
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
        node_update(inputs, i)
        fused.stp(ns.output)
        fused.learn(ns, exc_mask, 0.999)
        graph.increment_step()
    torch.cuda.synchronize()
    t_baseline = (time.perf_counter() - t0) / N_ITER * 1000
    print(f"  {t_baseline:.2f} ms/step ({1000/t_baseline:.0f} steps/sec)")

    # ================================================================
    # OPT 1: CUDA stream overlap (STP || learn)
    # ================================================================
    print("\n--- + STREAM OVERLAP (STP || learn) ---")

    stp_stream = torch.cuda.Stream()
    learn_stream = torch.cuda.Stream()

    compiled_stp = fused._stp_fn if fused._compiled else None
    compiled_learn = fused._learn_fn if fused._compiled else None

    def stp_and_learn_overlap():
        """Run STP and learning concurrently on separate streams."""
        output = ns.output
        # Shared gather (default stream)
        torch.index_select(output, 0, fused.f_src64, out=fused.f_src_out)
        torch.index_select(output, 0, fused.f_dst64, out=fused.f_dst_out)
        pre_act = fused.f_src_out.half() if fused._fp16_stp else fused.f_src_out.clone()
        pred_err_dst = ns.prediction_error[fused.f_dst64]
        global_nov = ns.prediction_error[exc_mask].abs().mean().clamp(min=0.01)

        # Record gather completion
        gather_done = torch.cuda.Event()
        gather_done.record()

        # STP on stream A
        with torch.cuda.stream(stp_stream):
            stp_stream.wait_event(gather_done)
            if compiled_stp:
                compiled_stp(fused.f_facilitation, fused.f_depression, fused.f_release_prob,
                             pre_act, fused.U, fused.tau_f, fused.tau_d)
            else:
                _stp_core(fused.f_facilitation, fused.f_depression, fused.f_release_prob,
                          pre_act, fused.U, fused.tau_f, fused.tau_d)

        # Learning on stream B
        with torch.cuda.stream(learn_stream):
            learn_stream.wait_event(gather_done)
            if compiled_learn:
                compiled_learn(fused.f_pre_trace, fused.f_post_trace, fused.f_weight,
                               fused.f_lr, fused.f_use_pretrace,
                               fused.f_src_out, fused.f_dst_out, pred_err_dst,
                               global_nov, 0.999)
            else:
                _learn_core(fused.f_pre_trace, fused.f_post_trace, fused.f_weight,
                            fused.f_lr, fused.f_use_pretrace,
                            fused.f_src_out, fused.f_dst_out, pred_err_dst,
                            global_nov, 0.999)

        # Default stream waits for both
        stp_done = torch.cuda.Event()
        stp_done.record(stp_stream)
        learn_done = torch.cuda.Event()
        learn_done.record(learn_stream)
        torch.cuda.current_stream().wait_event(stp_done)
        torch.cuda.current_stream().wait_event(learn_done)

    for i in range(WARMUP):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update(inputs, i)
        stp_and_learn_overlap()
        graph.increment_step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update(inputs, i)
        stp_and_learn_overlap()
        graph.increment_step()
    torch.cuda.synchronize()
    t_streams = (time.perf_counter() - t0) / N_ITER * 1000
    print(f"  {t_streams:.2f} ms/step ({1000/t_streams:.0f} steps/sec)")
    print(f"  vs baseline: {t_baseline/t_streams:.2f}x")

    # ================================================================
    # OPT 2: MORTON REORDER (cache-friendly gathers)
    # ================================================================
    print("\n--- + MORTON REORDER ---")
    reorder_graph_morton(graph)

    # Rebuild everything after reorder
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    sst_mask = ns.type_mask(NodeType.SST)
    exc_f = exc_mask.float()
    inh_mask = ~exc_mask
    inh_f = inh_mask.float()
    pv_f = ns.type_mask(NodeType.PV).float()
    pattern = torch.where(exc_mask)[0][:800]

    edge_cache = {}
    for et in EdgeType:
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        edge_cache[et] = {'src64': store.src.long(), 'dst64': store.dst.long(),
                          'delay_steps': (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)}

    mp.delay_buffer.reset()
    fused_morton = FusedPlasticity(graph, cfg.edges.stp)
    fused_morton.enable_compile()

    # Update stream refs
    compiled_stp = fused_morton._stp_fn if fused_morton._compiled else None
    compiled_learn = fused_morton._learn_fn if fused_morton._compiled else None

    # Re-inject activity
    ns.output = torch.rand(N, device=device) * 0.5
    ns.output[torch.rand(N, device=device) > 0.17] = 0.0
    ns.prediction_error = torch.randn(N, device=device) * 0.1

    def stp_and_learn_morton():
        output = ns.output
        torch.index_select(output, 0, fused_morton.f_src64, out=fused_morton.f_src_out)
        torch.index_select(output, 0, fused_morton.f_dst64, out=fused_morton.f_dst_out)
        pre_act = fused_morton.f_src_out.half() if fused_morton._fp16_stp else fused_morton.f_src_out.clone()
        pred_err_dst = ns.prediction_error[fused_morton.f_dst64]
        global_nov = ns.prediction_error[exc_mask].abs().mean().clamp(min=0.01)
        gather_done = torch.cuda.Event()
        gather_done.record()
        with torch.cuda.stream(stp_stream):
            stp_stream.wait_event(gather_done)
            if compiled_stp:
                compiled_stp(fused_morton.f_facilitation, fused_morton.f_depression, fused_morton.f_release_prob,
                             pre_act, fused_morton.U, fused_morton.tau_f, fused_morton.tau_d)
            else:
                _stp_core(fused_morton.f_facilitation, fused_morton.f_depression, fused_morton.f_release_prob,
                          pre_act, fused_morton.U, fused_morton.tau_f, fused_morton.tau_d)
        with torch.cuda.stream(learn_stream):
            learn_stream.wait_event(gather_done)
            if compiled_learn:
                compiled_learn(fused_morton.f_pre_trace, fused_morton.f_post_trace, fused_morton.f_weight,
                               fused_morton.f_lr, fused_morton.f_use_pretrace,
                               fused_morton.f_src_out, fused_morton.f_dst_out, pred_err_dst,
                               global_nov, 0.999)
            else:
                _learn_core(fused_morton.f_pre_trace, fused_morton.f_post_trace, fused_morton.f_weight,
                            fused_morton.f_lr, fused_morton.f_use_pretrace,
                            fused_morton.f_src_out, fused_morton.f_dst_out, pred_err_dst,
                            global_nov, 0.999)
        stp_done = torch.cuda.Event()
        stp_done.record(stp_stream)
        learn_done = torch.cuda.Event()
        learn_done.record(learn_stream)
        torch.cuda.current_stream().wait_event(stp_done)
        torch.cuda.current_stream().wait_event(learn_done)

    for i in range(WARMUP):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update(inputs, i)
        stp_and_learn_morton()
        graph.increment_step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        step = graph.step_count
        send_messages(step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        node_update(inputs, i)
        stp_and_learn_morton()
        graph.increment_step()
    torch.cuda.synchronize()
    t_morton = (time.perf_counter() - t0) / N_ITER * 1000
    print(f"  {t_morton:.2f} ms/step ({1000/t_morton:.0f} steps/sec)")
    print(f"  vs baseline: {t_baseline/t_morton:.2f}x")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (fused+compiled): {t_baseline:.2f} ms")
    print(f"  + Stream overlap:          {t_streams:.2f} ms ({t_baseline/t_streams:.2f}x)")
    print(f"  + Morton + streams:        {t_morton:.2f} ms ({t_baseline/t_morton:.2f}x)")
    print(f"\n  vs original 9.68 ms:       {9.68/t_morton:.2f}x total speedup")
    print(f"  {1000/t_morton:.0f} steps/sec")


if __name__ == '__main__':
    main()
