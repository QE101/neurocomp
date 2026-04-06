"""Normalised Hebbian: decouple activity magnitude from learning rate.

Two conditions:
A) Theta + normalised Hebbian on A-B (does oscillation stop?)
B) No theta + normalised Hebbian on A-B-C-D (does stability hold under load?)

The change: dw = (pre × post) / (mean_activity + eps) - decay - penalty

Direction of learning preserved (which edges are co-active).
Magnitude normalised (doesn't scale with activity level).
Theta can modulate dynamics without amplifying learning rate.
More patterns can't amplify learning through increased total activity.

5000 cycles each, sequential, unbuffered.
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
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
}


def setup_graph():
    config = GraphBrainConfig.from_dict(CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]

    return graph, ns, N, device, mp, stp, hom, ip, exc_idx, input_nodes


def make_patterns(input_nodes, ns, n_patterns):
    """Split input nodes into n_patterns spatial quadrants."""
    pos = ns.position[input_nodes.long()]
    x, y = pos[:, 0], pos[:, 1]
    med_x, med_y = x.median(), y.median()
    if n_patterns == 2:
        return {
            0: input_nodes[x < med_x],
            1: input_nodes[x >= med_x],
        }
    elif n_patterns == 4:
        return {
            0: input_nodes[(x < med_x) & (y < med_y)],
            1: input_nodes[(x >= med_x) & (y < med_y)],
            2: input_nodes[(x < med_x) & (y >= med_y)],
            3: input_nodes[(x >= med_x) & (y >= med_y)],
        }


def error_node_update(ns, inputs, dt=1.0, noise_std=0.005, theta_mod=1.0):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal * theta_mod) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    prediction_error = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(exc_mask, F.softplus(prediction_error.abs()) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, prediction_error, ns.prediction_error)
    pv_f = pv_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal + inputs.electrical) * pv_f
    ns.output = torch.where(pv_mask, F.softplus(ns.basal) * ns.gain * pv_f, ns.output)
    sst_f = sst_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * sst_f
    vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(sst_mask, F.softplus(ns.basal) * ns.gain * vip_inhib * sst_f, ns.output)
    vip_f = vip_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * vip_f
    ns.output = torch.where(vip_mask, F.softplus(ns.basal) * ns.gain * vip_f, ns.output)
    ns.output += torch.randn(N, device=device) * noise_std
    ns.output.clamp_(min=0.0)
    ns.activity_ema.lerp_(ns.output, dt / 1000.0)


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


def apply_normalised_hebbian(graph, la):
    """Normalised Hebbian: direction preserved, magnitude normalised.

    dw = base_lr * (pre * post) / (mean_pre_post + eps) - decay - penalty

    The normalisation ensures that the Hebbian signal's DIRECTION (which edges
    are co-active) drives learning, but the MAGNITUDE doesn't scale with
    overall activity level. Theta or more patterns can't amplify learning rate.
    """
    ns_ = graph.node_state
    exc_mask = ns_.type_mask(NodeType.EXCITATORY)
    mean_activity = ns_.output[exc_mask].mean().clamp(min=0.01)

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            # Normalised Hebbian: divide by mean activity squared
            hebbian = (src * dst) / (mean_activity * mean_activity)
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src + dst) / (mean_activity * 2.0) * store.weight
            dw = 0.001 * (hebbian - weight_decay - activity_penalty)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def run_condition(name, n_patterns, use_theta, n_cycles=5000):
    """Run one experimental condition."""
    print(f'\n  --- {name} ({n_cycles} cycles) ---', flush=True)

    graph, ns, N, device, mp, stp, hom, ip, exc_idx, input_nodes = setup_graph()
    patterns = make_patterns(input_nodes, ns, n_patterns)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5) if use_theta else None
    PD, STRENGTH = 50, 2.0

    # Use first two patterns for mismatch test regardless of n_patterns
    pa = patterns[0]
    pb = patterns[1]

    t0 = time.perf_counter()
    errors = []
    all_mm = []

    for cycle in range(n_cycles):
        err_sum, n = 0.0, 0
        for pid in range(n_patterns):
            pat = patterns[pid]
            for s in range(PD):
                step = graph.step_count
                theta_mod = theta.get_modulation(step) if theta else 1.0
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta_mod)
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_normalised_hebbian(graph, LAMBDA_ACT)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        if (cycle + 1) % 250 == 0:
            # Mismatch test (always A vs violation A-A)
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pa.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step) if theta else 1.0)
                graph.increment_step()
            bl = []
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pb.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step) if theta else 1.0)
                graph.increment_step()
                bl.append(ns.output[input_nodes].mean().item())
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pa.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step) if theta else 1.0)
                graph.increment_step()
            vl = []
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pa.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step) if theta else 1.0)
                graph.increment_step()
                vl.append(ns.output[input_nodes].mean().item())

            ratio = float(np.mean(vl)) / max(float(np.mean(bl)), 1e-8)
            all_mm.append(ratio)
            sup = (1 - errors[-1] / errors[0]) * 100
            ap = ns.apical[exc_idx].std().item()
            elapsed = time.perf_counter() - t0
            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'    Cyc {cycle+1:5d}: err={errors[-1]:.4f} sup={sup:.1f}% '
                  f'ap={ap:.4f} mm={mm} ({elapsed:.0f}s)', flush=True)

    # Oscillation analysis
    best_mm = max(all_mm)
    final_mm = all_mm[-1]
    if len(all_mm) >= 8:
        first_half = all_mm[:len(all_mm)//2]
        second_half = all_mm[len(all_mm)//2:]
        first_range = max(first_half) - min(first_half)
        second_range = max(second_half) - min(second_half)
        damping = "DAMPED" if second_range < first_range * 0.5 else (
            "PARTIAL" if second_range < first_range * 0.9 else "NO DAMPING")
    else:
        first_range = second_range = 0
        damping = "N/A"

    print(f'    DONE: best={best_mm:.3f}x final={final_mm:.3f}x '
          f'osc_range={first_range:.3f}/{second_range:.3f} {damping}', flush=True)

    return {
        'name': name, 'best_mm': best_mm, 'final_mm': final_mm,
        'damping': damping, 'first_range': first_range, 'second_range': second_range,
        'mismatch_history': all_mm,
    }


def main():
    print('=' * 60, flush=True)
    print('  NORMALISED HEBBIAN: Activity-invariant learning rate', flush=True)
    print('=' * 60, flush=True)
    print(f'dw = (pre*post)/(mean_activity^2) - normalised_decay', flush=True)
    print(f'Direction preserved, magnitude normalised', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n', flush=True)

    results = []

    # Condition A: Theta + normalised Hebbian on A-B
    r = run_condition('Theta+Norm (A-B)', n_patterns=2, use_theta=True, n_cycles=5000)
    results.append(r)

    # Condition B: No theta + normalised Hebbian on A-B-C-D
    r = run_condition('NoTheta+Norm (ABCD)', n_patterns=4, use_theta=False, n_cycles=5000)
    results.append(r)

    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    for r in results:
        print(f'  {r["name"]}: best={r["best_mm"]:.3f}x final={r["final_mm"]:.3f}x {r["damping"]}', flush=True)

    torch.save(results, 'normalised_hebbian_results.pt')
    print(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
