"""Phase 3 (minimal): RL as the stabiliser for emergent PC.

Hypothesis: the system discovers PC transiently via energy constraint,
and RL locks it in by rewarding correct predictions.

Minimal RL: a simple reward signal that fires when predictions are correct.
Not a full basal ganglia — just the stabilising mechanism.

Protocol:
  - Self-organised graph (no hand-built hierarchy)
  - Simultaneous Hebbian + energy constraint (the code that produces transient PC)
  - PLUS: reward signal when error nodes are quiet during expected input
    (correct prediction → low error → reward → protect prediction edges)
  - The reward counteracts the sparsity pressure on prediction-generating nodes

5000 cycles, mismatch test every 250 cycles.
"""

import sys
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType


def apply_hebbian_with_reward(graph, la, reward_signal, reward_strength=0.5):
    """Simultaneous Hebbian + RL reward for prediction accuracy.

    The reward_signal is a per-node scalar: high when the node successfully
    predicted (low output change when input arrived). Edges FROM high-reward
    nodes get a bonus that counteracts the sparsity pressure.
    """
    ns = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            src_out = ns.output[store.src.long()]
            dst_out = ns.output[store.dst.long()]

            # Standard Hebbian
            hebbian = src_out * dst_out

            # RL reward: protect edges from nodes that predicted well
            # reward_signal[src] high → this source made good predictions
            # Strengthen its outgoing edges to preserve the prediction circuit
            src_reward = reward_signal[store.src.long()]
            reward_bonus = src_reward * store.weight * reward_strength

            # Standard penalties
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src_out + dst_out) * store.weight

            dw = 0.001 * (hebbian + reward_bonus - weight_decay - activity_penalty)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def compute_reward(ns, prev_output, input_nodes, current_pat):
    """Compute per-node reward signal based on prediction accuracy.

    Reward = how little the node's output changed when input arrived.
    Nodes that predicted correctly have small output change → high reward.
    Nodes that were surprised have large output change → low reward.

    This is a minimal RL signal — not a full actor-critic, just
    "correct predictions are rewarded."
    """
    output_change = (ns.output - prev_output).abs()

    # Normalize: low change = high reward, high change = low reward
    max_change = output_change.max() + 1e-6
    reward = 1.0 - (output_change / max_change)
    reward = reward.clamp(0.0, 1.0)

    # Only reward excitatory nodes (they generate predictions)
    exc_mask = ns.type_mask(NodeType.EXCITATORY).float()
    reward = reward * exc_mask

    return reward


def main():
    print("=" * 70, flush=True)
    print("  PHASE 3 (MINIMAL): RL STABILISER FOR EMERGENT PC", flush=True)
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

    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0
    LAMBDA_ACT = 3.1
    REWARD_STRENGTH = 1.0
    N_CYCLES = 5000
    CHECKPOINT = 250

    print(f"Graph: {N} nodes, no hierarchy (self-organised)", flush=True)
    print(f"Lambda_activity: {LAMBDA_ACT}", flush=True)
    print(f"Reward strength: {REWARD_STRENGTH}", flush=True)
    print(f"Reward = 1 - |output_change|/max_change (correct prediction = high reward)\n", flush=True)

    prev_output = ns.output.clone()

    def run_step(pat):
        nonlocal prev_output
        step = graph.step_count
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH

        # Snapshot before update
        prev_output = ns.output.detach().clone()

        nm.step(ns, inputs, float(step))

        # Compute reward from prediction accuracy
        reward = compute_reward(ns, prev_output, input_nodes, pat)

        # STP
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)

        # Hebbian + reward
        apply_hebbian_with_reward(graph, LAMBDA_ACT, reward, REWARD_STRENGTH)

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
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            bl.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        vl = []
        for s in range(PD):
            step = graph.step_count
            mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            nm.step(ns, inputs, float(step))
            graph.increment_step()
            vl.append(ns.output[input_nodes].mean().item())
        return float(np.mean(bl)), float(np.mean(vl))

    print(f"{'Cyc':>5} | {'Err':>8} | {'Sup%':>5} | {'Ap_std':>7} | {'Reward':>7} | "
          f"{'MM':>8} | {'Time':>5}", flush=True)
    print("-" * 65, flush=True)

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

            # Average reward
            reward = compute_reward(ns, prev_output, input_nodes, pa)
            avg_reward = reward[exc_idx].mean().item()

            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " **"

            print(f"{cycle+1:5d} | {all_errors[-1]:8.4f} | {sup:4.1f}% | {ap_std:7.4f} | "
                  f"{avg_reward:7.4f} | {mm_str:>8} | {elapsed:5.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Time: {total:.0f}s ({total/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"Best mismatch: {max(all_mismatch):.3f}x", flush=True)
    print(f"Final mismatch: {all_mismatch[-1]:.3f}x", flush=True)

    if max(all_mismatch) > 1.1:
        best_idx = all_mismatch.index(max(all_mismatch))
        print(f"MISMATCH DETECTED at cycle {(best_idx+1)*CHECKPOINT}", flush=True)
    else:
        print("NO MISMATCH DETECTED", flush=True)

    torch.save({
        "errors": all_errors, "mismatch_ratios": all_mismatch,
    }, "rl_stabiliser_results.pt")
    print(f"\nSaved to rl_stabiliser_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
