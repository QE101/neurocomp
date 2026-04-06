"""Hard Weight Normalisation: PREVENTION not damping.

Total weight into each node is fixed at its initial value (the "budget").
Hebbian redistributes within that budget — strengthening A→i necessarily weakens B→i.
Oscillation cannot form because there's no envelope to overshoot.

Copy-pasted core functions from run_stability_battery.py (never reimplement).
Only change: normalise_weights() called after apply_hebbian().
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
N_CYCLES = 5000

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


# ================================================================
# THE NEW BIT: Hard weight normalisation using CSR dst_ptr
# ================================================================
def compute_weight_budgets(graph):
    """Capture the initial weight budget per node per edge type.
    Budget = sum of weights into each node at initialisation.
    """
    budgets = {}
    N = graph.n_nodes
    for et in EdgeType:
        if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        # Use dst_ptr (CSR) to compute per-node sum of incoming weights
        # dst_ptr[i]:dst_ptr[i+1] spans edges targeting node i
        budget = torch.zeros(N, device=store.weight.device)
        ptr = store.dst_ptr.long()
        for i in range(N):
            start, end = ptr[i], ptr[i + 1]
            if end > start:
                budget[i] = store.weight[start:end].sum()
        budgets[et] = budget
    return budgets


def compute_weight_budgets_fast(graph):
    """Vectorised version using scatter_add — O(E) not O(N)."""
    budgets = {}
    N = graph.n_nodes
    for et in EdgeType:
        if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue
        budget = torch.zeros(N, device=store.weight.device)
        budget.scatter_add_(0, store.dst.long(), store.weight)
        budgets[et] = budget
    return budgets


def normalise_weights(graph, budgets):
    """Hard normalisation: rescale incoming weights to each node to match budget.

    w_into_i *= (budget_i / current_sum_i)

    Uses CSR dst_ptr for efficient per-node access.
    Nodes with zero budget (no incoming edges of this type) are skipped.
    """
    for et, budget in budgets.items():
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        if store.n_edges == 0:
            continue

        # Compute current sum per destination node
        N = graph.n_nodes
        current_sum = torch.zeros(N, device=store.weight.device)
        current_sum.scatter_add_(0, store.dst.long(), store.weight)

        # Scale factor per node: budget / current (avoid div by zero)
        scale = torch.ones(N, device=store.weight.device)
        nonzero = current_sum > 1e-8
        scale[nonzero] = budget[nonzero] / current_sum[nonzero]

        # Apply per-edge: each edge gets its destination node's scale factor
        store.weight *= scale[store.dst.long()]
        store.weight.clamp_(0.0, 1.0)


# ================================================================
# Standard Hebbian (exact copy from battery)
# ================================================================
def apply_hebbian(graph, la):
    ns_ = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]
            dw = 0.001 * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


# ================================================================
# MAIN TEST
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  HARD WEIGHT NORMALISATION: Prevention Test', flush=True)
    print('  Budget per node fixed at init. Hebbian redistributes.', flush=True)
    print('  Theta + A-B, N=1250, 5000 cycles.', flush=True)
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
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    # Capture initial weight budgets BEFORE any learning
    budgets = compute_weight_budgets_fast(graph)
    for et, b in budgets.items():
        nonzero = (b > 0).sum().item()
        print(f'  Budget {et.name}: {nonzero} nodes, mean={b[b>0].mean():.4f}', flush=True)

    t0 = time.perf_counter()
    errors, all_mm = [], []

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                error_node_update(ns, inputs, theta_mod=theta.get_modulation(step))
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)

                # Standard Hebbian THEN hard normalisation
                apply_hebbian(graph, LAMBDA_ACT)
                normalise_weights(graph, budgets)

                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

        if (cycle + 1) % 500 == 0:
            bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
            ratio = vl / max(bl, 1e-8)
            all_mm.append(ratio)
            sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0
            elapsed = time.perf_counter() - t0

            # Check weight budget drift (should be ~0)
            drift_pct = 0.0
            n_checked = 0
            for et, budget in budgets.items():
                if not graph.has_edge_type(et): continue
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                current = torch.zeros(N, device=device)
                current.scatter_add_(0, store.dst.long(), store.weight)
                nonzero = budget > 1e-8
                if nonzero.any():
                    drift = ((current[nonzero] - budget[nonzero]).abs() / budget[nonzero]).mean().item()
                    drift_pct += drift
                    n_checked += 1
            drift_pct = (drift_pct / max(n_checked, 1)) * 100

            mm = f'{ratio:.3f}x' + (' **' if ratio > 1.1 else '')
            print(f'  Cyc {cycle+1:5d}: mm={mm} sup={sup:.1f}% drift={drift_pct:.2f}% ({elapsed:.0f}s)', flush=True)

    # Oscillation analysis
    if len(all_mm) >= 4:
        half = len(all_mm) // 2
        fr = max(all_mm[:half]) - min(all_mm[:half])
        sr = max(all_mm[half:]) - min(all_mm[half:])
        if sr < fr * 0.5:
            damping = "DAMPED"
        elif sr < fr * 0.9:
            damping = "PARTIAL"
        else:
            damping = "NO DAMPING"
    else:
        fr = sr = 0
        damping = "N/A"

    best = max(all_mm) if all_mm else 0
    final = all_mm[-1] if all_mm else 0
    print(f'\n  RESULT: best={best:.3f}x final={final:.3f}x osc={fr:.3f}/{sr:.3f} {damping}', flush=True)
    print(f'  All mismatch: {[f"{m:.3f}" for m in all_mm]}', flush=True)

    if damping == "NO DAMPING" or sr > 0.5:
        print(f'\n  VERDICT: Oscillation PERSISTS — prevention failed', flush=True)
    elif best > 1.1 and sr < 0.3:
        print(f'\n  VERDICT: STABLE MISMATCH — oscillation PREVENTED', flush=True)
    elif best > 1.1 and damping == "DAMPED":
        print(f'\n  VERDICT: CONVERGING — oscillation being eliminated', flush=True)
    else:
        print(f'\n  VERDICT: Mismatch too weak (best={best:.3f}x)', flush=True)

    torch.save({'all_mm': all_mm, 'errors': errors, 'damping': damping,
                'best': best, 'final': final, 'fr': fr, 'sr': sr},
               'hard_normalisation_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
