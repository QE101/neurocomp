"""Diagnostic run: instrument EVERYTHING during the oscillation.

Standard undamped Hebbian + theta on A-B. Every 10 cycles, dump every metric
we can think of. Goal: SEE what happens when the mismatch crashes.

2000 cycles, measurement every 10 cycles, mismatch every 100 cycles.
Saves full log to diagnostic_log.pt for offline analysis.
"""

import sys, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1
STRENGTH = 2.0
PD = 50
N_CYCLES = 2000

SMALL_CONFIG = {
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}


# ================================================================
# EXACT copy-paste from run_stability_battery.py (DO NOT MODIFY)
# ================================================================
def error_node_update(ns, inputs, theta_mod=1.0):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()
    ns.basal += 1.0 * (-ns.basal / 10.0 + inputs.basal * theta_mod) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += 1.0 * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(exc_mask, F.softplus(pred_err.abs()) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += 1.0 * (-ns.basal / 10.0 + inp) * f
        out = F.softplus(ns.basal) * ns.gain * f
        if inh_type == NodeType.SST:
            out = out * torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        ns.output = torch.where(mask, out, ns.output)
    ns.output += torch.randn(N, device=device) * 0.005
    ns.output.clamp_(min=0.0)
    ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)


def dual_channel_send(ns, graph, mp, device):
    step = graph.step_count
    output = ns.output
    content = F.softplus(ns.basal)
    for et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = output[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
              EdgeType.RETROGRADE: Channel.RETROGRADE}[et]
        mp.delay_buffer.write(ch, store.dst, msg, d, step)
    for et in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
        if not graph.has_edge_type(et): continue
        store = graph.edge_store(et)
        msg = content[store.src.long()] * store.release_prob * store.weight
        d = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[et]
        mp.delay_buffer.write(ch, store.dst, msg, d, step)
    if graph.has_edge_type(EdgeType.ELECTRICAL):
        store = graph.edge_store(EdgeType.ELECTRICAL)
        gap = store.weight * (output[store.src.long()] - output[store.dst.long()])
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap,
                              torch.ones(store.n_edges, dtype=torch.long, device=device), step)


# Standard Hebbian (the one that oscillates)
def apply_hebbian(graph, la):
    ns_ = graph.node_state
    dw_stats = {}
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]
            dw = 0.001 * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
            dw_stats[et] = dw.detach().clone()
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)
    return dw_stats


def run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes):
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
    bl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        bl.append(ns.output[input_nodes].mean().item())
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
    vl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(ns, graph, mp, device)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        vl.append(ns.output[input_nodes].mean().item())
    return float(np.mean(bl)), float(np.mean(vl))


def main():
    print('=' * 60, flush=True)
    print('  DIAGNOSTIC RUN: Full Instrumentation', flush=True)
    print('  Standard Hebbian + theta. Every metric, every 10 cycles.', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    config = GraphBrainConfig.from_dict(SMALL_CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    # Identify edge subsets for pattern-specific analysis
    pa_set = set(pa.cpu().tolist())
    pb_set = set(pb.cpu().tolist())

    edge_masks = {}
    for et in EdgeType:
        if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0: continue
        src_np = store.src.cpu().numpy()
        dst_np = store.dst.cpu().numpy()
        # Edges FROM pattern A nodes
        from_a = torch.tensor([s in pa_set for s in src_np], device=device)
        from_b = torch.tensor([s in pb_set for s in src_np], device=device)
        to_a = torch.tensor([d in pa_set for d in dst_np], device=device)
        to_b = torch.tensor([d in pb_set for d in dst_np], device=device)
        # Shared: from A to B or from B to A
        shared = (from_a & to_b) | (from_b & to_a)
        a_only = from_a & ~from_b & ~shared
        b_only = from_b & ~from_a & ~shared
        edge_masks[et] = {'from_a': from_a, 'from_b': from_b, 'to_a': to_a, 'to_b': to_b,
                          'shared': shared, 'a_only': a_only, 'b_only': b_only}

    print(f'  Patterns: A={pa.shape[0]}, B={pb.shape[0]}', flush=True)
    for et, masks in edge_masks.items():
        n_e = graph.edge_store(et).n_edges
        print(f'  {et.name}: {n_e} edges, from_A={masks["from_a"].sum()}, '
              f'from_B={masks["from_b"].sum()}, shared={masks["shared"].sum()}', flush=True)

    log = []  # full diagnostic log
    t0 = time.perf_counter()

    for cycle in range(N_CYCLES):
        # ---- Run one A-B cycle, collecting per-step dw ----
        cycle_dw_accum = {}  # accumulate |dw| across the cycle
        err_sum, n_steps = 0.0, 0

        for pat_idx, pat in enumerate([pa, pb]):
            pat_name = 'A' if pat_idx == 0 else 'B'
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)

                dw_stats = apply_hebbian(graph, LAMBDA_ACT)

                # Accumulate dw stats
                for et, dw in dw_stats.items():
                    if et not in cycle_dw_accum:
                        cycle_dw_accum[et] = {'pos': 0.0, 'neg': 0.0, 'abs_sum': 0.0, 'count': 0}
                    cycle_dw_accum[et]['pos'] += (dw > 0).sum().item()
                    cycle_dw_accum[et]['neg'] += (dw < 0).sum().item()
                    cycle_dw_accum[et]['abs_sum'] += dw.abs().sum().item()
                    cycle_dw_accum[et]['count'] += 1

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n_steps += 1

        # ---- Every 10 cycles: full diagnostic snapshot ----
        if (cycle + 1) % 10 == 0:
            snap = {'cycle': cycle + 1, 'time': time.perf_counter() - t0}

            # Node metrics by group
            for name, idx in [('pa', pa), ('pb', pb), ('exc', exc_idx)]:
                snap[f'{name}_output_mean'] = ns.output[idx].mean().item()
                snap[f'{name}_output_std'] = ns.output[idx].std().item()
                snap[f'{name}_basal_mean'] = ns.basal[idx].mean().item()
                snap[f'{name}_apical_mean'] = ns.apical[idx].mean().item()
                snap[f'{name}_pred_err_mean'] = ns.prediction_error[idx].mean().item()
                snap[f'{name}_pred_err_abs'] = ns.prediction_error[idx].abs().mean().item()
                snap[f'{name}_ema_mean'] = ns.activity_ema[idx].mean().item()
                snap[f'{name}_gain_mean'] = ns.gain[idx].mean().item()

            # PV and SST
            pv_idx = torch.where(pv_mask)[0]
            sst_idx = torch.where(sst_mask)[0]
            snap['pv_output_mean'] = ns.output[pv_idx].mean().item()
            snap['sst_output_mean'] = ns.output[sst_idx].mean().item()

            # Edge metrics by type + pattern subset
            for et in EdgeType:
                if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
                    continue
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                ename = et.name.lower()

                # Overall weight stats
                w = store.weight
                snap[f'{ename}_w_mean'] = w.mean().item()
                snap[f'{ename}_w_std'] = w.std().item()
                snap[f'{ename}_w_min'] = w.min().item()
                snap[f'{ename}_w_max'] = w.max().item()
                snap[f'{ename}_w_p25'] = w.quantile(0.25).item()
                snap[f'{ename}_w_p75'] = w.quantile(0.75).item()

                # STP state
                snap[f'{ename}_release_mean'] = store.release_prob.mean().item()
                snap[f'{ename}_pre_trace_mean'] = store.pre_trace.mean().item()
                snap[f'{ename}_post_trace_mean'] = store.post_trace.mean().item()

                # Weight sum per destination (budget)
                budget = torch.zeros(N, device=device)
                budget.scatter_add_(0, store.dst.long(), w)
                snap[f'{ename}_budget_mean'] = budget[budget > 0].mean().item()
                snap[f'{ename}_budget_std'] = budget[budget > 0].std().item()

                # Pattern-specific weight stats
                if et in edge_masks:
                    masks = edge_masks[et]
                    for subset_name in ['from_a', 'from_b', 'shared', 'a_only', 'b_only']:
                        m = masks[subset_name]
                        if m.sum() > 0:
                            snap[f'{ename}_{subset_name}_w_mean'] = w[m].mean().item()
                        else:
                            snap[f'{ename}_{subset_name}_w_mean'] = 0.0

                # dw stats from this cycle
                if et in cycle_dw_accum:
                    info = cycle_dw_accum[et]
                    total = info['pos'] + info['neg']
                    snap[f'{ename}_dw_pct_pos'] = info['pos'] / max(total, 1) * 100
                    snap[f'{ename}_dw_abs_mean'] = info['abs_sum'] / max(info['count'] * store.n_edges, 1)

            # Theta state
            snap['theta_mod'] = theta.get_modulation(graph.step_count)

            # Error
            snap['avg_error'] = err_sum / n_steps

            log.append(snap)

            # Print compact summary every 10 cycles
            if (cycle + 1) % 100 == 0:
                drv = snap.get('driving_w_mean', 0)
                mod = snap.get('modulatory_w_mean', 0)
                drv_a = snap.get('driving_from_a_w_mean', 0)
                drv_b = snap.get('driving_from_b_w_mean', 0)
                pe_a = snap['pa_pred_err_abs']
                pe_b = snap['pb_pred_err_abs']
                ba_a = snap['pa_basal_mean']
                ap_a = snap['pa_apical_mean']
                ba_b = snap['pb_basal_mean']
                ap_b = snap['pb_apical_mean']
                out_a = snap['pa_output_mean']
                out_b = snap['pb_output_mean']
                pv = snap['pv_output_mean']

                print(f'  Cyc {cycle+1:5d}: '
                      f'out_A={out_a:.3f} out_B={out_b:.3f} | '
                      f'bas_A={ba_a:.3f} ap_A={ap_a:.3f} pe_A={pe_a:.3f} | '
                      f'bas_B={ba_b:.3f} ap_B={ap_b:.3f} pe_B={pe_b:.3f} | '
                      f'drv={drv:.4f} mod={mod:.4f} | '
                      f'drv_fromA={drv_a:.4f} drv_fromB={drv_b:.4f} | '
                      f'PV={pv:.3f}', flush=True)

        # Mismatch every 200 cycles
        if (cycle + 1) % 200 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            elapsed = time.perf_counter() - t0
            print(f'  *** MISMATCH Cyc {cycle+1}: {mm} (bl={bl:.4f} vl={vl:.4f}) ({elapsed:.0f}s) ***', flush=True)

    # Save everything
    torch.save(log, 'diagnostic_log.pt')
    print(f'\nSaved {len(log)} snapshots to diagnostic_log.pt', flush=True)
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'Load with: log = torch.load("diagnostic_log.pt"); import pandas as pd; df = pd.DataFrame(log)', flush=True)


if __name__ == '__main__':
    main()
