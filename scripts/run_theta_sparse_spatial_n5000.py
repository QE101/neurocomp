"""Theta + Sparse Coding + Spatial Gating: structural solution to interference.

Option C: both mechanisms combined.

Sparse coding (PV winner-take-all):
  - Boost PV→EXC inhibition so each pattern activates a small, distinct set of nodes
  - Competitive exclusion: when A fires, PV suppresses non-A nodes
  - Different patterns → different active populations → different edges

Spatial gating:
  - Only update edges near currently-active input nodes
  - Pattern A (bottom-left) → only bottom-left edges learn
  - Pattern B (bottom-right) → only bottom-right edges learn
  - Interference impossible at the input level

Combined: spatial gating separates learning at input, PV competition
separates representations at relay/prediction level.

5000 cycles with theta, unbuffered.
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
SPATIAL_RADIUS = 0.1  # only update edges with src or dst within this radius of active input
PV_BOOST = 5.0  # boost PV→EXC inhibition for winner-take-all

config = GraphBrainConfig.from_dict({
    'nodes': {'n_excitatory': 4000, 'n_pv': 350, 'n_sst': 350, 'n_vip': 300, 'noise_std': 0.005},
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

# === SPARSE CODING: Boost PV inhibition ===
if graph.has_edge_type(EdgeType.INHIB_PERISOMATIC):
    store = graph.edge_store(EdgeType.INHIB_PERISOMATIC)
    store.weight *= PV_BOOST
    store.weight.clamp_(0.0, 1.0)
    print(f'PV inhibition boosted {PV_BOOST}x (mean={store.weight.mean():.3f})', flush=True)

# === SPATIAL GATING: Pre-compute proximity masks per pattern ===
# For each edge type, compute which edges are "near" pattern A or B input nodes
positions = ns.position

def compute_spatial_mask(pattern_nodes, radius):
    """For each edge, check if src OR dst is within radius of any pattern node."""
    pat_pos = positions[pattern_nodes.long()]  # [K, 3]
    masks = {}
    for et in EdgeType:
        if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            masks[et] = torch.zeros(0, dtype=torch.bool, device=device)
            continue
        # Check if src or dst of each edge is near any pattern node
        src_pos = positions[store.src.long()]  # [E, 3]
        dst_pos = positions[store.dst.long()]  # [E, 3]
        # Min distance from any pattern node to src/dst
        # Efficient: for each edge endpoint, check distance to ALL pattern nodes
        # src_pos: [E, 3], pat_pos: [K, 3]
        # dist_src: [E, K]
        # Use chunked computation to avoid OOM
        E = store.n_edges
        K = pat_pos.shape[0]
        near_src = torch.zeros(E, dtype=torch.bool, device=device)
        near_dst = torch.zeros(E, dtype=torch.bool, device=device)
        chunk = 5000
        for i in range(0, E, chunk):
            end = min(i + chunk, E)
            d_src = torch.cdist(src_pos[i:end], pat_pos)  # [chunk, K]
            d_dst = torch.cdist(dst_pos[i:end], pat_pos)
            near_src[i:end] = (d_src.min(dim=1).values < radius)
            near_dst[i:end] = (d_dst.min(dim=1).values < radius)
        masks[et] = near_src | near_dst
    return masks

print('Pre-computing spatial masks...', flush=True)
mask_a = compute_spatial_mask(pa, SPATIAL_RADIUS)
mask_b = compute_spatial_mask(pb, SPATIAL_RADIUS)
for et in mask_a:
    n_a = mask_a[et].sum().item()
    n_b = mask_b[et].sum().item()
    n_both = (mask_a[et] & mask_b[et]).sum().item()
    n_total = mask_a[et].shape[0]
    print(f'  {et.name}: A={n_a} B={n_b} overlap={n_both} total={n_total} '
          f'({n_both/max(n_total,1)*100:.1f}% overlap)', flush=True)

current_mask = mask_a  # which spatial mask is active


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


def apply_hebbian_spatially_gated(graph, la):
    """Hebbian with spatial gating: only update edges near active input pattern."""
    ns_ = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            if et not in current_mask: continue

            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]
            hebbian = src * dst
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src + dst) * store.weight
            raw_dw = 0.001 * (hebbian - weight_decay - activity_penalty)

            # Spatial gate: only apply dw to edges near active pattern
            gate = current_mask[et].float()
            dw = raw_dw * gate
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def run_step(pat, pat_mask):
    global current_mask
    current_mask = pat_mask
    step = graph.step_count
    dual_channel_send(graph, step)
    inputs = mp.read_inputs(step)
    inputs.basal[pat.long()] += STRENGTH
    error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            stp.update(graph.edge_store(et), ns, 1.0)
    apply_hebbian_spatially_gated(graph, LAMBDA_ACT)
    if step % 100 == 0:
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                hom.update(graph.edge_store(et), ns, 1.0)
        ip.update(ns)
    graph.increment_step()


def run_mismatch():
    for s in range(PD):
        run_step(pa, mask_a)
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
        run_step(pa, mask_a)
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


print(f'\nTheta + Sparse Coding + Spatial Gating: 5000 cycles', flush=True)
print(f'PV boost={PV_BOOST}x | Spatial radius={SPATIAL_RADIUS}', flush=True)
print(f'{"Cyc":>5} | {"Err":>7} | {"Sup%":>5} | {"Ap_std":>7} | {"MM":>8} | {"Time":>5}', flush=True)
print('-' * 55, flush=True)

t0 = time.perf_counter()
errors = []
all_mm = []

for cycle in range(5000):
    err_sum, n = 0.0, 0
    for pat, mask in [(pa, mask_a), (pb, mask_b)]:
        for s in range(PD):
            run_step(pat, mask)
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
        print(f'{cycle+1:5d} | {errors[-1]:.4f} | {sup:.1f}% | {ap:.4f} | {mm:>8} | {elapsed:.0f}s', flush=True)

print(f'\nBest mismatch: {max(all_mm):.3f}x', flush=True)
print(f'Final mismatch: {all_mm[-1]:.3f}x', flush=True)

# Check if oscillation is damped
if len(all_mm) >= 8:
    first_half_range = max(all_mm[:len(all_mm)//2]) - min(all_mm[:len(all_mm)//2])
    second_half_range = max(all_mm[len(all_mm)//2:]) - min(all_mm[len(all_mm)//2:])
    print(f'Oscillation range: first half={first_half_range:.3f} second half={second_half_range:.3f}', flush=True)
    if second_half_range < first_half_range * 0.5:
        print('DAMPING DETECTED — oscillation is reducing', flush=True)
    elif second_half_range < first_half_range * 0.9:
        print('PARTIAL DAMPING — oscillation slightly reducing', flush=True)
    else:
        print('NO DAMPING — oscillation sustained', flush=True)

print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
