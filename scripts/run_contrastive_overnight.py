"""Phase 1B: Contrastive phases overnight test suite.

Runs the full experiment with detailed metrics at every checkpoint.
Unbuffered output for live monitoring.

Protocol per A-B cycle:
  1. PREDICT phase A (50 steps, no input) — system generates prediction
  2. OBSERVE phase A (50 steps, pattern A injected) — compare to prediction
  3. Contrastive weight update from A
  4. PREDICT phase B (50 steps, no input)
  5. OBSERVE phase B (50 steps, pattern B injected)
  6. Contrastive weight update from B

Tests:
  - 5000 A-B cycles with contrastive learning
  - Mismatch test every 500 cycles (A-B-A-B → A-A violation)
  - Full metrics: suppression, apical_std, match quality, weight asymmetry
  - Final comparison against hand-built Phase 1A baseline
"""

import sys
import os
import time

# Force unbuffered output everywhere
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.contrastive import ContrastiveLearning, PhaseSnapshot
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.energy import EnergyGenome, TemporalState, apply_energy_gradient
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType


def main():
    print("=" * 70, flush=True)
    print("  PHASE 1B: CONTRASTIVE PHASES — OVERNIGHT TEST", flush=True)
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

    # Winning genome from evolution + stronger contrastive signal
    genome = EnergyGenome(
        lambda_activity=3.1, lambda_weight=0.0065, lambda_edge=0.00001,
        lambda_prediction=2.0, lambda_reconstruction=0.27, lambda_mi=2.3,
        lambda_compartment=2.0,
    )

    # Components
    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    ts = TemporalState(N, device)
    contrastive = ContrastiveLearning(genome)

    # Input nodes
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD = 50  # steps per phase
    STRENGTH = 2.0
    N_CYCLES = 5000
    CHECKPOINT_EVERY = 250

    print(f"\nGraph: {N} nodes, {graph.n_edges()} edges", flush=True)
    print(f"Input nodes: {input_nodes.shape[0]} (pattern A: {pa.shape[0]}, B: {pb.shape[0]})", flush=True)
    print(f"Protocol: {PD} predict steps + {PD} observe steps per pattern", flush=True)
    print(f"Cycles: {N_CYCLES}, checkpoints every {CHECKPOINT_EVERY}", flush=True)
    print(f"Genome: act={genome.lambda_activity} pred={genome.lambda_prediction} comp={genome.lambda_compartment}", flush=True)

    def run_phase(pat_nodes, inject_input):
        """Run one phase (predict or observe) for PD steps."""
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            if inject_input and pat_nodes is not None:
                inputs.basal[pat_nodes.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    stp.update(graph.edge_store(et), ns, 1.0)
            # Standard energy gradient still runs (temporal Hebbian + metabolic)
            apply_energy_gradient(graph, genome, ts, 1.0)
            if step % 100 == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()

    def run_mismatch_test():
        """Quick mismatch test: A-B baseline then A-A violation."""
        # Baseline: present A then B (expected)
        run_phase(pa, inject_input=True)
        baseline_snap = contrastive.capture_snapshot(graph)
        baseline_vals = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            baseline_vals.append(ns.output[input_nodes].mean().item())

        # Violation: present A then A (unexpected — expected B)
        run_phase(pa, inject_input=True)
        violation_vals = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            violation_vals.append(ns.output[input_nodes].mean().item())

        bl = float(np.mean(baseline_vals))
        vl = float(np.mean(violation_vals))
        return bl, vl

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    print(f"\n{'='*70}", flush=True)
    print("  TRAINING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Cycle':>6} | {'Err':>8} | {'Sup%':>6} | {'Match':>7} | {'Ap_std':>7} | {'Ba_std':>7} | "
          f"{'Asym':>6} | {'Mismatch':>9} | {'Time':>5}", flush=True)
    print("-" * 85, flush=True)

    t0 = time.perf_counter()
    all_errors = []
    all_match = []
    all_mismatch_ratios = []

    for cycle in range(N_CYCLES):
        cycle_err = 0.0
        cycle_match = 0.0
        n_samples = 0

        for pat in [pa, pb]:
            # PREDICT phase: no input, system generates from internal state
            run_phase(pat, inject_input=False)
            predict_snap = contrastive.capture_snapshot(graph)

            # OBSERVE phase: input arrives
            run_phase(pat, inject_input=True)
            observe_snap = contrastive.capture_snapshot(graph)

            # Contrastive weight update
            metrics = contrastive.update_weights(graph, predict_snap, observe_snap)
            cycle_match += metrics["match"]

            # Track output error during observe phase
            cycle_err += ns.output[input_nodes].mean().item()
            n_samples += 1

        all_errors.append(cycle_err / n_samples)
        all_match.append(cycle_match / n_samples)

        # Checkpoint
        if (cycle + 1) % CHECKPOINT_EVERY == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - all_errors[-1] / all_errors[0]) * 100 if all_errors[0] > 0 else 0

            # Diagnostics
            ap_std = ns.apical[exc_idx].std().item()
            ba_std = ns.basal[exc_idx].std().item()

            # Weight asymmetry
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

            # Mismatch test
            bl, vl = run_mismatch_test()
            ratio = vl / max(bl, 1e-8)
            all_mismatch_ratios.append(ratio)

            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " *"

            print(f"{cycle+1:6d} | {all_errors[-1]:8.4f} | {sup:5.1f}% | {all_match[-1]:7.4f} | "
                  f"{ap_std:7.4f} | {ba_std:7.4f} | {asym:6.4f} | {mm_str:>9} | {elapsed:5.0f}s",
                  flush=True)

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    total_time = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"Apical std: {ns.apical[exc_idx].std().item():.4f}", flush=True)
    print(f"Best mismatch ratio: {max(all_mismatch_ratios):.3f}x", flush=True)
    print(f"Final mismatch ratio: {all_mismatch_ratios[-1]:.3f}x", flush=True)

    if max(all_mismatch_ratios) > 1.1:
        print(f"\nMISMATCH DETECTED at checkpoint {(all_mismatch_ratios.index(max(all_mismatch_ratios))+1)*CHECKPOINT_EVERY}", flush=True)
    else:
        print(f"\nNO MISMATCH DETECTED across {N_CYCLES} cycles", flush=True)

    # Save everything
    torch.save({
        "errors": all_errors,
        "match_quality": all_match,
        "mismatch_ratios": all_mismatch_ratios,
        "genome": genome.to_dict(),
        "config": config.model_dump(),
    }, "contrastive_overnight_results.pt")
    print(f"\nSaved to contrastive_overnight_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
