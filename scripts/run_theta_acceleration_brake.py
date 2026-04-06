"""Phase 1B + Theta + Precision-gated learning: 5000 cycles.

Three additions to the validated Phase 1B substrate:
1. Theta modulation (6Hz, amplitude 0.5)
2. Precision computation (tau=1000ms, across theta cycles)
3. Precision-gated Hebbian learning rate

The governor: high precision → slow learning → stable weights.
The turbo: theta → amplified activity → faster initial learning.
Combined: learn fast when uncertain, stabilise when confident.
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
PRECISION_TAU = 1000.0  # 1000ms — averages across ~6 theta cycles


def error_node_update(ns, inputs, dt=1.0, noise_std=0.005, theta_mod=1.0):
    device = ns.device
    N = ns.n_nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    exc_f = exc_mask.float()

    # Excitatory: universal error model with theta
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal * theta_mod) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    prediction_error = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    exc_output = F.softplus(prediction_error.abs()) * pv_gain * ns.gain
    ns.output = torch.where(exc_mask, exc_output, ns.output)
    ns.prediction_error = torch.where(exc_mask, prediction_error, ns.prediction_error)

    # === PRECISION UPDATE (new) ===
    # Slow EMA of absolute error — averages across many theta cycles
    alpha_p = dt / PRECISION_TAU
    ns.error_mean_ema = torch.where(
        exc_mask,
        ns.error_mean_ema * (1 - alpha_p) + prediction_error.abs() * alpha_p,
        ns.error_mean_ema,
    )
    ns.precision = torch.where(
        exc_mask,
        (1.0 / (ns.error_mean_ema + 0.1)).clamp(0.1, 50.0),
        ns.precision,
    )

    # PV
    pv_f = pv_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal + inputs.electrical) * pv_f
    ns.output = torch.where(pv_mask, F.softplus(ns.basal) * ns.gain * pv_f, ns.output)
    # SST
    sst_f = sst_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * sst_f
    vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(sst_mask, F.softplus(ns.basal) * ns.gain * vip_inhib * sst_f, ns.output)
    # VIP
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


weight_velocity = {}  # per-edge-type EMA of |dw| — tracks how fast weights are changing
VELOCITY_TAU = 500.0  # 500ms EMA — responds within a few hundred cycles
VELOCITY_SCALE = 100.0  # scaling factor: velocity of 0.01 → 2x brake

def apply_hebbian_acceleration_brake(graph, la):
    """Hebbian with acceleration-proportional braking.

    Track how fast weights are changing (velocity = EMA of |dw|).
    Faster change → stronger brake → prevents overshoot.
    Stable weights → no brake → normal learning.

    Like a shock absorber: resists rapid change, allows slow adjustment.
    """
    global weight_velocity
    ns_ = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue

            # Initialize velocity tracker if needed
            if et not in weight_velocity:
                weight_velocity[et] = torch.zeros_like(store.weight)

            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]

            # Compute raw dw (unbraked)
            hebbian = src * dst
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src + dst) * store.weight
            raw_dw = 0.001 * (hebbian - weight_decay - activity_penalty)

            # Brake proportional to velocity (how fast weights have been changing)
            velocity = weight_velocity[et]
            brake = 1.0 / (velocity * VELOCITY_SCALE + 1.0)
            # velocity=0 → brake=1.0 (no braking)
            # velocity=0.01 → brake=1/(1+1)=0.5 (50% braking)
            # velocity=0.05 → brake=1/(1+5)=0.17 (83% braking)
            # velocity=0.1 → brake=1/(1+10)=0.09 (91% braking)

            dw = raw_dw * brake
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

            # Update velocity: EMA of |dw|
            alpha_v = 1.0 / VELOCITY_TAU
            weight_velocity[et] = velocity * (1 - alpha_v) + raw_dw.abs() * alpha_v


def run_step(pat):
    step = graph.step_count
    dual_channel_send(graph, step)
    inputs = mp.read_inputs(step)
    inputs.basal[pat.long()] += STRENGTH
    error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            stp.update(graph.edge_store(et), ns, 1.0)
    apply_hebbian_acceleration_brake(graph, LAMBDA_ACT)
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


print('Theta + Acceleration brake: 5000 cycles', flush=True)
print('Theta=6Hz amp=0.5 | Brake proportional to weight change velocity', flush=True)
print(f'{"Cyc":>5} | {"Err":>7} | {"Sup%":>5} | {"Ap_std":>7} | {"Vel":>9} | {"Brake":>6} | {"MM":>8} | {"Time":>5}', flush=True)
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
        # Mean weight velocity across all edge types
        vel_mn = np.mean([v.mean().item() for v in weight_velocity.values()]) if weight_velocity else 0
        brake_mn = 1.0 / (vel_mn * VELOCITY_SCALE + 1.0)
        elapsed = time.perf_counter() - t0
        mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
        print(f'{cycle+1:5d} | {errors[-1]:.4f} | {sup:.1f}% | {ap:.4f} | {vel_mn:.6f} | '
              f'{brake_mn:.4f} | {mm:>8} | {elapsed:.0f}s', flush=True)

print(f'\nBest mismatch: {max(all_mm):.3f}x', flush=True)
print(f'Final mismatch: {all_mm[-1]:.3f}x', flush=True)
vel_final = np.mean([v.mean().item() for v in weight_velocity.values()]) if weight_velocity else 0
print(f'Final velocity: {vel_final:.6f} (brake: {1.0/(vel_final*VELOCITY_SCALE+1.0):.4f})', flush=True)
print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
