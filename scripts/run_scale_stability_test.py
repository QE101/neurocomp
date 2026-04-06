"""The Scale Hypothesis Test: does N=50K resolve the theta oscillation?

At N=1250, patterns share edges → Hebbian interference → oscillation.
At N=50K with constant k, patterns activate different nodes → orthogonal representations → stable?

Two conditions:
A) N=50K + theta + universal error model (does oscillation stop?)
B) N=50K + NO theta + universal error model (baseline — should be stable like N=1250)

2000 A-B cycles each (enough to see 1-2 full oscillation periods if they exist).
Unbuffered output.
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
CONFIG = {
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
            'electrical': {'p_max': 0.3, 'sigma': 0.05, 'source_types': ['PV'],
                          'target_types': ['PV'], 'constant_k': 5},
            'retrograde': {'p_max': 0.1, 'sigma': 0.15, 'source_types': ['EXCITATORY'],
                           'target_types': ['EXCITATORY'], 'constant_k': 10},
            'max_radius': 0.5,
        },
        'structural': {'enabled': False},
    },
    'simulation': {'device': 'cuda', 'seed': 42, 'record_interval': 100},
    'hierarchy': {'enabled': False},
}


def run_condition(name, use_theta, n_cycles=2000):
    print(f'\n{"="*60}', flush=True)
    print(f'  {name} ({n_cycles} cycles)', flush=True)
    print(f'{"="*60}', flush=True)

    config = GraphBrainConfig.from_dict(CONFIG)
    print('Building N=50K graph...', flush=True)
    t_build = time.perf_counter()
    graph = NeuromorphicGraph(config)
    graph.initialize()
    print(f'Built in {time.perf_counter()-t_build:.1f}s ({graph.n_edges():,} edges)', flush=True)

    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5) if use_theta else None

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0

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

    def dual_channel_send(step):
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

    def apply_hebbian(la):
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src = ns.output[store.src.long()]
                dst = ns.output[store.dst.long()]
                dw = 0.001 * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

    def run_step(pat):
        step = graph.step_count
        tm = theta.get_modulation(step) if theta else 1.0
        dual_channel_send(step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=tm)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian(LAMBDA_ACT)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()

    def run_mismatch():
        for s in range(PD):
            run_step(pa)
        bl = []
        for s in range(PD):
            step = graph.step_count
            tm = theta.get_modulation(step) if theta else 1.0
            dual_channel_send(step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            error_node_update(ns, inputs, theta_mod=tm)
            graph.increment_step()
            bl.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        vl = []
        for s in range(PD):
            step = graph.step_count
            tm = theta.get_modulation(step) if theta else 1.0
            dual_channel_send(step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            error_node_update(ns, inputs, theta_mod=tm)
            graph.increment_step()
            vl.append(ns.output[input_nodes].mean().item())
        return float(np.mean(bl)), float(np.mean(vl))

    print(f'Running {n_cycles} A-B cycles...', flush=True)
    t0 = time.perf_counter()
    errors = []
    all_mm = []

    for cycle in range(n_cycles):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        if (cycle + 1) % 250 == 0:
            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            sup = (1 - errors[-1] / errors[0]) * 100
            ap = ns.apical[exc_idx].std().item()
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: err={errors[-1]:.4f} sup={sup:.1f}% '
                  f'ap={ap:.4f} mm={mm} ({elapsed:.0f}s)', flush=True)

    if len(all_mm) >= 4:
        first_half = all_mm[:len(all_mm)//2]
        second_half = all_mm[len(all_mm)//2:]
        fr = max(first_half) - min(first_half)
        sr = max(second_half) - min(second_half)
        damping = "DAMPED" if sr < fr * 0.5 else ("PARTIAL" if sr < fr * 0.9 else "NO DAMPING")
    else:
        fr = sr = 0
        damping = "N/A"

    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'\n  DONE: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    return {'name': name, 'best': best, 'final': final, 'damping': damping,
            'first_range': fr, 'second_range': sr, 'mm': all_mm}


def main():
    print('=' * 60, flush=True)
    print('  SCALE HYPOTHESIS: Does N=50K resolve the oscillation?', flush=True)
    print('=' * 60, flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    r_theta = run_condition('N=50K + Theta', use_theta=True, n_cycles=2000)
    r_baseline = run_condition('N=50K Baseline', use_theta=False, n_cycles=2000)

    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  N=50K + Theta:    best={r_theta["best"]:.3f}x final={r_theta["final"]:.3f}x {r_theta["damping"]}', flush=True)
    print(f'  N=50K Baseline:   best={r_baseline["best"]:.3f}x final={r_baseline["final"]:.3f}x {r_baseline["damping"]}', flush=True)

    if r_theta['best'] > 1.1 and r_theta['damping'] in ('DAMPED', 'N/A'):
        print(f'\n  SCALE HYPOTHESIS: CONFIRMED — theta is stable at N=50K', flush=True)
    elif r_theta['best'] > 1.1:
        print(f'\n  SCALE HYPOTHESIS: PARTIAL — theta works but still oscillates at N=50K', flush=True)
    else:
        print(f'\n  SCALE HYPOTHESIS: FAILED — N=50K doesn\'t produce mismatch', flush=True)

    torch.save({'theta': r_theta, 'baseline': r_baseline}, 'scale_stability_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

if __name__ == '__main__':
    main()
