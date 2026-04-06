"""Theta + Slow Consolidation: two-timescale eligibility for stable learning.

Fast trace: accumulates co-activation within ~50 steps (one pattern presentation)
Slow trace: accumulates CONSISTENT fast-trace over ~2000 steps (many A/B cycles)
Weight update: proportional to slow trace only

Transient co-activation (one pattern) builds fast trace but gets reset by the
next pattern before reaching slow trace. Only CONSISTENT signal (edges useful
across both patterns / overall prediction structure) accumulates and updates weights.

A/B oscillation cancels in slow trace: A pushes one direction, B pushes the other,
only the shared prediction signal survives.

5000 cycles, unbuffered output.
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
config = GraphBrainConfig.from_dict({
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
})

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
exc_z = ns.position[exc_idx, 2]
input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
pa = input_nodes[:input_nodes.shape[0] // 2]
pb = input_nodes[input_nodes.shape[0] // 2:]
PD, STRENGTH = 50, 2.0

# Two-timescale trace state
fast_trace = {}   # per-edge-type: fast accumulation of Hebbian signal
slow_trace = {}   # per-edge-type: slow accumulation of consistent signal
FAST_TAU = 50.0   # fast trace half-life: ~50 steps (within one pattern)
SLOW_TAU = 2000.0 # slow trace half-life: ~2000 steps (across many A/B cycles)
SLOW_LR = 0.005   # weight update rate from slow trace


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


def dual_channel_send(graph, step):
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


def apply_hebbian_slow_consolidation(graph, la):
    """Two-timescale Hebbian: fast trace captures instant signal, slow trace
    accumulates consistent signal, weight updates from slow trace only.

    Fast trace: EMA of raw Hebbian signal (tau=50 steps)
        - Captures within-pattern co-activation
        - Resets when pattern changes (opposing signal cancels)

    Slow trace: EMA of fast trace (tau=2000 steps)
        - Only accumulates if fast trace is CONSISTENTLY positive or negative
        - Transient signals from alternating patterns cancel out
        - Only the SHARED prediction structure survives

    Weight update: dw = SLOW_LR * slow_trace - weight_decay - activity_penalty
    """
    global fast_trace, slow_trace
    ns_ = graph.node_state

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue

            if et not in fast_trace:
                fast_trace[et] = torch.zeros_like(store.weight)
                slow_trace[et] = torch.zeros_like(store.weight)

            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            # Raw Hebbian signal (same as always)
            hebbian = src * dst

            # Fast trace: EMA of raw signal (captures within-pattern structure)
            alpha_fast = 1.0 / FAST_TAU
            fast_trace[et] = fast_trace[et] * (1 - alpha_fast) + hebbian * alpha_fast

            # Slow trace: EMA of fast trace (captures consistent-across-patterns structure)
            alpha_slow = 1.0 / SLOW_TAU
            slow_trace[et] = slow_trace[et] * (1 - alpha_slow) + fast_trace[et] * alpha_slow

            # Weight update from SLOW trace only (not raw Hebbian)
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src + dst) * store.weight
            dw = SLOW_LR * slow_trace[et] - 0.001 * (weight_decay + activity_penalty)

            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def run_step(pat):
    step = graph.step_count
    dual_channel_send(graph, step)
    inputs = mp.read_inputs(step)
    inputs.basal[pat.long()] += STRENGTH
    error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            stp.update(graph.edge_store(et), ns, 1.0)
    apply_hebbian_slow_consolidation(graph, LAMBDA_ACT)
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
        dual_channel_send(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        bl.append(ns.output[input_nodes].mean().item())
    for s in range(PD):
        run_step(pa)
    vl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += STRENGTH
        error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
        graph.increment_step()
        vl.append(ns.output[input_nodes].mean().item())
    return float(np.mean(bl)), float(np.mean(vl))


print('Theta + Slow Consolidation: 5000 cycles', flush=True)
print(f'Theta=6Hz amp=0.5 | Fast tau={FAST_TAU} | Slow tau={SLOW_TAU} | Slow LR={SLOW_LR}', flush=True)
print(f'Only CONSISTENT signal across patterns reaches weight updates', flush=True)
print(f'{"Cyc":>5} | {"Err":>7} | {"Sup%":>5} | {"Ap_std":>7} | {"SlowMn":>7} | {"FastMn":>7} | {"MM":>8} | {"Time":>5}', flush=True)
print('-' * 70, flush=True)

t0 = time.perf_counter()
errors = []
all_mm = []

for cycle in range(5000):
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
        # Mean traces
        slow_mn = np.mean([t.abs().mean().item() for t in slow_trace.values()]) if slow_trace else 0
        fast_mn = np.mean([t.abs().mean().item() for t in fast_trace.values()]) if fast_trace else 0
        elapsed = time.perf_counter() - t0
        mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
        print(f'{cycle+1:5d} | {errors[-1]:.4f} | {sup:.1f}% | {ap:.4f} | {slow_mn:.5f} | '
              f'{fast_mn:.5f} | {mm:>8} | {elapsed:.0f}s', flush=True)

print(f'\nBest mismatch: {max(all_mm):.3f}x', flush=True)
print(f'Final mismatch: {all_mm[-1]:.3f}x', flush=True)
print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
