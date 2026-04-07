"""Benchmark torch.compile on fused STP + learning.

torch.compile traces the Python ops and fuses elementwise chains into
single CUDA kernels, cutting memory traffic by 3-5x.

Must run in WSL (torch.compile needs Linux).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import time

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
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
    print("  torch.compile BENCHMARK")
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
    fused = FusedPlasticity(graph, cfg.edges.stp)

    input_region = torch.where(exc_mask)[0][:int(0.2 * 40000)]
    n_on = max(10, int(input_region.shape[0] * 0.10))
    pattern = input_region[torch.randperm(input_region.shape[0], device=device)[:n_on]]

    total_edges = sum(graph.edge_store(et).n_edges for et in EdgeType if graph.has_edge_type(et))
    print(f"  {N:,} nodes, {total_edges:,} edges")

    # ================================================================
    # Compilable functions (pure tensor ops, no Python control flow)
    # ================================================================

    def stp_core(f_facilitation, f_depression, f_release_prob, pre_activity, U, tau_f, tau_d):
        """STP math — in-place ops, compilable."""
        du = -f_facilitation / tau_f + U * (1.0 - f_facilitation) * pre_activity
        f_facilitation.add_(du).clamp_(0.0, 1.0)

        dx = (1.0 - f_depression) / tau_d - f_facilitation * f_depression * pre_activity
        f_depression.add_(dx).clamp_(0.0, 1.0)

        f_release_prob.copy_((f_facilitation + U) * f_depression).clamp_(0.0, 1.0)

    def learn_core(
        f_pre_trace, f_post_trace, f_weight, f_lr, f_use_pretrace,
        src, dst, pred_err_dst, global_novelty, current_decay
    ):
        """Learning math — in-place where possible, compilable."""
        # Pre-trace
        f_pre_trace.lerp_(src, 0.05)

        # Consolidation
        f_post_trace.mul_(current_decay).addcmul_(src, dst, value=0.0001).clamp_(0.0, 1.0)

        # Error gate
        error_gate = (pred_err_dst.abs() / global_novelty).clamp(0.0, 3.0)

        # Un-consolidation
        error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
        f_post_trace.sub_(0.0001 * error_signal * f_post_trace * f_post_trace).clamp_(0.0, 1.0)

        # Plasticity
        plasticity = error_gate * (1.0 - 0.9 * f_post_trace)

        # Hebbian source
        hebbian_src = torch.where(f_use_pretrace, f_pre_trace, src)

        # Weight update
        dw = f_lr * plasticity * (
            hebbian_src * dst - WEIGHT_DECAY * f_weight - dst * dst * f_weight
        )
        f_weight.add_(dw).clamp_(0.0, 1.0)

    def node_update(basal, apical, output, prediction_error, gain,
                    inp_basal, inp_apical, inp_sst, inp_pv, inp_electrical, inp_vip,
                    exc_mask, exc_f, inh_mask, inh_f, sst_mask, pv_f, noise):
        """Full node update — compilable."""
        # Excitatory
        basal = basal + (-basal / 10.0 + inp_basal) * exc_f
        sst_gate = torch.sigmoid(inp_sst * 5.0)
        apical = apical + (-apical / 20.0 + inp_apical * (1.0 - sst_gate)) * exc_f
        pred_err = basal - apical
        pv_gain = (1.0 - inp_pv).clamp(0.0, 1.0)
        raw = (F.softplus(pred_err.abs()) - BASELINE).clamp(0.0, 10.0)
        output = torch.where(exc_mask, raw * pv_gain * gain, output)
        prediction_error = torch.where(exc_mask, pred_err, prediction_error)

        # Inhibitory (vectorized)
        inh_input = inp_basal + inp_electrical * pv_f
        basal = basal + (-basal / 10.0 + inh_input) * inh_f
        inh_raw = (F.softplus(basal) - BASELINE).clamp(0.0, 10.0)
        inh_out = inh_raw * gain * inh_f
        sst_suppress = (1.0 - inp_sst - inp_vip).clamp(0.0, 1.0)
        inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
        output = torch.where(inh_mask, inh_out, output)

        output = (output + noise).clamp(0.0, 10.0)
        return basal, apical, output, prediction_error

    # ================================================================
    # Compile
    # ================================================================
    print("\nCompiling...", flush=True)
    t_compile_start = time.perf_counter()

    compiled_stp = torch.compile(stp_core, mode='default')
    compiled_learn = torch.compile(learn_core, mode='default')
    compiled_node = torch.compile(node_update, mode='default')

    # Trigger compilation with warmup
    print("  Warming up compiled kernels (first call triggers JIT)...", flush=True)

    # STP warmup
    pre_act = ns.output[fused.f_src64]
    compiled_stp(
        fused.f_facilitation, fused.f_depression, fused.f_release_prob,
        pre_act, fused.U, fused.tau_f, fused.tau_d
    )
    torch.cuda.synchronize()
    print("  STP compiled.", flush=True)

    # Learn warmup
    global_nov = ns.prediction_error[exc_mask].abs().mean().clamp(min=0.01)
    torch.index_select(ns.output, 0, fused.f_src64, out=fused.f_src_out)
    torch.index_select(ns.output, 0, fused.f_dst64, out=fused.f_dst_out)
    pred_err_dst = ns.prediction_error[fused.f_dst64]
    compiled_learn(
        fused.f_pre_trace, fused.f_post_trace, fused.f_weight, fused.f_lr, fused.f_use_pretrace,
        fused.f_src_out, fused.f_dst_out, pred_err_dst, global_nov, 0.999
    )
    torch.cuda.synchronize()
    print("  Learn compiled.", flush=True)

    # Node update warmup
    step = graph.step_count
    for et, ch in OUTPUT_EDGE_CHANNELS.items():
        if et not in edge_cache: continue
        c = edge_cache[et]; store = graph.edge_store(et)
        mp.delay_buffer.write(ch, store.dst, ns.output[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
    for et, ch in CONTENT_EDGE_CHANNELS.items():
        if et not in edge_cache: continue
        c = edge_cache[et]; store = graph.edge_store(et)
        mp.delay_buffer.write(ch, store.dst, F.softplus(ns.basal).clamp(max=10.0)[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
    inputs = mp.read_inputs(step)
    ns.basal, ns.apical, ns.output, ns.prediction_error = compiled_node(
        ns.basal, ns.apical, ns.output, ns.prediction_error, ns.gain,
        inputs.basal, inputs.apical, inputs.sst_inhibition, inputs.pv_inhibition,
        inputs.electrical, inputs.vip_inhibition,
        exc_mask, exc_f, inh_mask, inh_f, sst_mask, pv_f, noise_buf[0]
    )
    torch.cuda.synchronize()
    graph.increment_step()
    t_compile = time.perf_counter() - t_compile_start
    print(f"  Compilation done in {t_compile:.1f}s", flush=True)

    # ================================================================
    # Benchmark: eager (fused v1) vs compiled
    # ================================================================
    N_ITER = 300
    WARMUP = 100

    def run_step_eager(i):
        step = graph.step_count
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        for et, ch in OUTPUT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, output[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        for et, ch in CONTENT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, content[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, store.weight * (output[c['src64']] - output[c['dst64']]), c['delay_steps'], step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        # Node update (inline)
        ns.basal += (-ns.basal / 10.0 + inputs.basal) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        pred_err = ns.basal - ns.apical
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, 0.0, 1.0)
        raw = F.softplus(pred_err.abs()) - BASELINE
        ns.output = torch.where(exc_mask, raw.clamp(0.0, 10.0) * pv_gain * ns.gain, ns.output)
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

    def run_step_compiled(i):
        step = graph.step_count
        output = ns.output
        content = F.softplus(ns.basal).clamp(max=10.0)
        for et, ch in OUTPUT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, output[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        for et, ch in CONTENT_EDGE_CHANNELS.items():
            if et not in edge_cache: continue
            c = edge_cache[et]; store = graph.edge_store(et)
            mp.delay_buffer.write(ch, store.dst, content[c['src64']] * store.release_prob * store.weight, c['delay_steps'], step)
        if EdgeType.ELECTRICAL in edge_cache:
            c = edge_cache[EdgeType.ELECTRICAL]; store = graph.edge_store(EdgeType.ELECTRICAL)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, store.weight * (output[c['src64']] - output[c['dst64']]), c['delay_steps'], step)
        inputs = mp.read_inputs(step)
        if i % 3 == 0: inputs.basal[pattern.long()] += 2.0
        # Compiled node update
        ns.basal, ns.apical, ns.output, ns.prediction_error = compiled_node(
            ns.basal, ns.apical, ns.output, ns.prediction_error, ns.gain,
            inputs.basal, inputs.apical, inputs.sst_inhibition, inputs.pv_inhibition,
            inputs.electrical, inputs.vip_inhibition,
            exc_mask, exc_f, inh_mask, inh_f, sst_mask, pv_f, noise_buf[i % 100]
        )
        ns.activity_ema.lerp_(ns.output, 0.001)
        # Compiled STP
        pre_act = ns.output[fused.f_src64]
        compiled_stp(
            fused.f_facilitation, fused.f_depression, fused.f_release_prob,
            pre_act, fused.U, fused.tau_f, fused.tau_d
        )
        # Compiled learning
        global_nov = ns.prediction_error[exc_mask].abs().mean().clamp(min=0.01)
        torch.index_select(ns.output, 0, fused.f_src64, out=fused.f_src_out)
        torch.index_select(ns.output, 0, fused.f_dst64, out=fused.f_dst_out)
        pred_err_dst = ns.prediction_error[fused.f_dst64]
        compiled_learn(
            fused.f_pre_trace, fused.f_post_trace, fused.f_weight, fused.f_lr, fused.f_use_pretrace,
            fused.f_src_out, fused.f_dst_out, pred_err_dst, global_nov, 0.999
        )
        graph.increment_step()

    # --- Eager benchmark ---
    print("\nBenchmarking eager (fused v1)...", flush=True)
    for i in range(WARMUP):
        run_step_eager(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        run_step_eager(i)
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / N_ITER * 1000

    # --- Compiled benchmark ---
    print("Benchmarking compiled...", flush=True)
    for i in range(WARMUP):
        run_step_compiled(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITER):
        run_step_compiled(i)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / N_ITER * 1000

    # ================================================================
    # Results
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Eager (fused v1): {t_eager:.2f} ms  ({1000/t_eager:.0f} steps/sec)")
    print(f"  Compiled:         {t_compiled:.2f} ms  ({1000/t_compiled:.0f} steps/sec)")
    print(f"  Speedup:          {t_eager/t_compiled:.2f}x")
    print(f"  vs original 9.68: {9.68/t_compiled:.2f}x total speedup")
    print(f"\n  1000ep x 100st:")
    print(f"    Eager:    {1000*100*t_eager/1000/60:.1f} min")
    print(f"    Compiled: {1000*100*t_compiled/1000/60:.1f} min")


if __name__ == '__main__':
    main()
