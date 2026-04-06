"""Phase 1B + Phase 2: Oscillatory-driven PC emergence.

PV interneurons coupled by gap junctions → gamma oscillation.
The oscillation gates excitatory output:
  - Low PV phase: inhibition drops → nodes free to predict
  - High PV phase: inhibition rises → only errors punch through

This creates natural predict/observe temporal separation WITHOUT
an artificial protocol. The energy constraint + sparsity builds topology,
the oscillation provides the temporal structure for functional PC.

5000 cycles, comprehensive metrics, unbuffered output.
"""

import sys
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph, EdgeStore, build_dst_ptr
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.energy import EnergyGenome, TemporalState, apply_energy_gradient
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType


def strengthen_pv_coupling(graph, coupling_strength=0.5):
    """Strengthen PV-PV gap junctions to induce gamma oscillations.

    If not enough electrical edges exist, create dense PV-PV connections.
    Gap junctions are bidirectional and non-plastic.
    """
    ns = graph.node_state
    device = ns.device
    pv_mask = ns.type_mask(NodeType.PV)
    pv_idx = torch.where(pv_mask)[0]
    n_pv = pv_idx.shape[0]

    if n_pv < 2:
        print(f"  Only {n_pv} PV nodes — can't create oscillatory circuit", flush=True)
        return

    # Create dense PV-PV connections (all-to-all within distance)
    positions = ns.position
    pv_pos = positions[pv_idx]

    # All-pairs distance between PV nodes
    diff = pv_pos.unsqueeze(1) - pv_pos.unsqueeze(0)
    dist = (diff * diff).sum(dim=2).sqrt()

    # Connect PV nodes within radius 0.3 (local coupling)
    radius = 0.3
    connected = (dist < radius) & (dist > 0)  # exclude self

    if not connected.any():
        # If PV nodes are too spread, connect nearest neighbors
        print(f"  No PV pairs within radius {radius}, connecting 5 nearest each", flush=True)
        for i in range(n_pv):
            _, nearest = dist[i].topk(min(6, n_pv), largest=False)
            nearest = nearest[nearest != i][:5]
            connected[i, nearest] = True
            connected[nearest, i] = True  # bidirectional

    src_local, dst_local = torch.where(connected)
    src_global = pv_idx[src_local].to(torch.int32)
    dst_global = pv_idx[dst_local].to(torch.int32)
    n_new = src_global.shape[0]

    if n_new == 0:
        return

    # Remove existing electrical edges and replace with strong coupling
    if graph.has_edge_type(EdgeType.ELECTRICAL):
        old = graph.edge_store(EdgeType.ELECTRICAL)
        graph.remove_edges(EdgeType.ELECTRICAL, torch.ones(old.n_edges, dtype=torch.bool, device=device))

    # Create new strong gap junctions
    weights = torch.full((n_new,), coupling_strength, device=device)
    graph.add_edges(EdgeType.ELECTRICAL, src_global, dst_global, weights=weights)

    print(f"  Created {n_new} PV-PV gap junctions (coupling={coupling_strength})", flush=True)


def add_pv_drive(graph, drive_strength=0.5, drive_frequency=0.05):
    """Add oscillatory drive to PV population.

    A simple sinusoidal modulation of PV excitability creates the
    seed oscillation. Gap junction coupling then synchronises it
    into coherent gamma.

    Returns a function that computes the drive at each timestep.
    """
    ns = graph.node_state
    pv_idx = torch.where(ns.type_mask(NodeType.PV))[0]

    def get_drive(step):
        """Sinusoidal drive to PV neurons. Returns [N] tensor."""
        phase = 2.0 * np.pi * drive_frequency * step
        drive_value = drive_strength * (1.0 + np.sin(phase)) / 2.0  # 0 to drive_strength
        drive = torch.zeros(ns.n_nodes, device=ns.device)
        drive[pv_idx] = drive_value
        return drive

    return get_drive, pv_idx


def main():
    print("=" * 70, flush=True)
    print("  PHASE 1B+2: OSCILLATORY-DRIVEN PC EMERGENCE", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": 42},
        "hierarchy": {"enabled": False},
    })

    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    # Precision-gated genome (attempt 7's setup)
    genome = EnergyGenome(
        lambda_activity=3.1, lambda_weight=0.0065, lambda_edge=0.00001,
        lambda_prediction=2.0, lambda_reconstruction=0.27, lambda_mi=2.3,
        lambda_compartment=2.0,
    )

    # Strengthen PV-PV coupling for gamma oscillations
    print("\nSetting up oscillatory circuit...", flush=True)
    strengthen_pv_coupling(graph, coupling_strength=0.5)

    # Oscillatory drive to PV population
    # Frequency: 0.05 cycles/step = period of 20 steps
    # At dt=1ms, this is 50Hz gamma
    get_pv_drive, pv_idx = add_pv_drive(graph, drive_strength=1.0, drive_frequency=0.05)
    print(f"  PV drive: 50Hz gamma, {pv_idx.shape[0]} PV neurons", flush=True)

    # Components
    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    ts = TemporalState(N, device)

    # Input setup
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD = 50
    STRENGTH = 2.0
    N_CYCLES = 5000
    CHECKPOINT = 250

    print(f"\nGraph: {N} nodes, {graph.n_edges()} edges", flush=True)
    print(f"PV oscillation provides natural predict/observe gating", flush=True)
    print(f"+ Precision-gated per-node sparsity from attempt 7\n", flush=True)

    def run_step(pat):
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        # Inject pattern
        inputs.basal[pat.long()] += STRENGTH
        # Add oscillatory PV drive
        pv_drive = get_pv_drive(step)
        inputs.basal += pv_drive  # PV drive adds to basal of PV nodes
        # Node update
        nm.step(ns, inputs, float(step))
        # Plasticity
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_energy_gradient(graph, genome, ts, 1.0)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, 1.0)
            ip.update(ns)
        graph.increment_step()

    def run_mismatch_test():
        for s in range(PD):
            run_step(pa)
        baseline = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            inputs.basal += get_pv_drive(step)
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            baseline.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        violation = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            inputs.basal += get_pv_drive(step)
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            violation.append(ns.output[input_nodes].mean().item())
        return float(np.mean(baseline)), float(np.mean(violation))

    # Header
    print(f"{'Cyc':>5} | {'Err':>8} | {'Sup%':>5} | {'Ap_std':>7} | {'PV_std':>7} | "
          f"{'Prec_mn':>7} | {'Prec_sd':>7} | {'Asym':>6} | {'MM':>7} | {'Time':>5}", flush=True)
    print("-" * 100, flush=True)

    t0 = time.perf_counter()
    all_errors = []
    all_mismatch = []

    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        all_errors.append(err_sum / n)

        if (cycle + 1) % CHECKPOINT == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - all_errors[-1] / all_errors[0]) * 100 if all_errors[0] > 0 else 0
            ap_std = ns.apical[exc_idx].std().item()
            pv_out_std = ns.output[pv_idx].std().item()
            exc_prec = ns.precision[exc_idx]

            # Asymmetry
            up_w, down_w, n_up, n_down = 0.0, 0.0, 0, 0
            for et in (EdgeType.DRIVING, EdgeType.MODULATORY):
                if graph.has_edge_type(et):
                    store = graph.edge_store(et)
                    sz = ns.position[store.src.long(), 2]
                    dz = ns.position[store.dst.long(), 2]
                    up = dz > sz
                    dn = dz < sz
                    if up.any():
                        up_w += float(store.weight[up].sum())
                        n_up += int(up.sum())
                    if dn.any():
                        down_w += float(store.weight[dn].sum())
                        n_down += int(dn.sum())
            um = up_w / max(n_up, 1)
            dm = down_w / max(n_down, 1)
            asym = abs(um - dm) / (max(um, dm) + 1e-6)

            bl, vl = run_mismatch_test()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " *"

            print(f"{cycle+1:5d} | {all_errors[-1]:8.4f} | {sup:4.1f}% | {ap_std:7.4f} | {pv_out_std:7.4f} | "
                  f"{exc_prec.mean().item():7.3f} | {exc_prec.std().item():7.3f} | "
                  f"{asym:6.4f} | {mm_str:>7} | {elapsed:5.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Time: {total:.0f}s ({total/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"Apical std: {ns.apical[exc_idx].std().item():.4f}", flush=True)
    print(f"PV output std: {ns.output[pv_idx].std().item():.4f} (oscillation strength)", flush=True)
    print(f"Best mismatch: {max(all_mismatch):.3f}x", flush=True)
    print(f"Final mismatch: {all_mismatch[-1]:.3f}x", flush=True)

    if max(all_mismatch) > 1.1:
        print(f"MISMATCH DETECTED", flush=True)
    else:
        print(f"NO MISMATCH DETECTED", flush=True)

    torch.save({
        "errors": all_errors, "mismatch_ratios": all_mismatch,
        "genome": genome.to_dict(),
    }, "oscillatory_emergence_results.pt")
    print(f"\nSaved to oscillatory_emergence_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
