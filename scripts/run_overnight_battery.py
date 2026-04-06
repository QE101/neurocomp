"""Overnight battery: stress-test existing systems, no new features.

Test 1: Phase 1B seed robustness — 5 seeds of universal error model + dual channel
Test 2: Phase 1B at N=5000 — does it scale?
Test 3: Phase 3 extended — 3000 trials, does reversal eventually learn?
Test 4: Phase 3 seed robustness — 3 seeds

All sequential on GPU (no contention = faster per-experiment).
Unbuffered output throughout.
"""

import sys
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

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
from graph_brain.reward import RewardSystem
from graph_brain.episodic import EpisodicMemory
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1


def error_node_update(ns, inputs, dt=1.0):
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    device = ns.device
    exc_f = exc_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    ns.output = torch.where(exc_mask, F.softplus(pred_err.abs()) * pv_gain * ns.gain, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)
    for inh_type, mask in [(NodeType.PV, ns.type_mask(NodeType.PV)),
                            (NodeType.SST, ns.type_mask(NodeType.SST)),
                            (NodeType.VIP, ns.type_mask(NodeType.VIP))]:
        f = mask.float()
        inp = inputs.basal + (inputs.electrical if inh_type == NodeType.PV else torch.zeros_like(inputs.basal))
        ns.basal += dt * (-ns.basal / 10.0 + inp) * f
        out = F.softplus(ns.basal) * ns.gain * f
        if inh_type == NodeType.SST:
            out = out * torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        ns.output = torch.where(mask, out, ns.output)
    ns.output += torch.randn(ns.n_nodes, device=device) * 0.005
    ns.output.clamp_(min=0.0)
    ns.activity_ema.lerp_(ns.output, dt / 1000.0)


def dual_channel_send(graph, mp, step):
    ns = graph.node_state
    output = ns.output
    content = F.softplus(ns.basal)
    device = graph.device
    for et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        msg = output[store.src.long()] * store.release_prob * store.weight
        delay = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
              EdgeType.RETROGRADE: Channel.RETROGRADE}[et]
        mp.delay_buffer.write(ch, store.dst, msg, delay, step)
    for et in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
        if not graph.has_edge_type(et):
            continue
        store = graph.edge_store(et)
        msg = content[store.src.long()] * store.release_prob * store.weight
        delay = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
        ch = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[et]
        mp.delay_buffer.write(ch, store.dst, msg, delay, step)
    if graph.has_edge_type(EdgeType.ELECTRICAL):
        store = graph.edge_store(EdgeType.ELECTRICAL)
        gap = store.weight * (output[store.src.long()] - output[store.dst.long()])
        mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap,
                              torch.ones(store.n_edges, dtype=torch.long, device=device), step)


def apply_hebbian(graph, la):
    ns = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue
            src = ns.output[store.src.long()]
            dst = ns.output[store.dst.long()]
            dw = 0.001 * (src * dst - 0.0065 * 2.0 * store.weight - la * (src + dst) * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


# ==============================================================
# TEST 1: Phase 1B seed robustness
# ==============================================================
def test_1b_seed(seed, n_exc=1000, n_cycles=3000):
    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": n_exc, "n_pv": int(n_exc*0.09), "n_sst": int(n_exc*0.09),
                  "n_vip": int(n_exc*0.07), "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": seed},
        "hierarchy": {"enabled": False},
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

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD = 50

    t0 = time.perf_counter()
    errors = []

    for cycle in range(n_cycles):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                step = graph.step_count
                dual_channel_send(graph, mp, step)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += 2.0
                error_node_update(ns, inputs)
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        stp.update(graph.edge_store(et), ns, 1.0)
                apply_hebbian(graph, LAMBDA_ACT)
                if step % 100 == 0:
                    for et in EdgeType:
                        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            hom.update(graph.edge_store(et), ns, 1.0)
                    ip.update(ns)
                graph.increment_step()
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        errors.append(err_sum / n)

    # Mismatch test
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += 2.0
        error_node_update(ns, inputs)
        graph.increment_step()
    bl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pb.long()] += 2.0
        error_node_update(ns, inputs)
        graph.increment_step()
        bl.append(ns.output[input_nodes].mean().item())
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += 2.0
        error_node_update(ns, inputs)
        graph.increment_step()
    vl = []
    for s in range(PD):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pa.long()] += 2.0
        error_node_update(ns, inputs)
        graph.increment_step()
        vl.append(ns.output[input_nodes].mean().item())

    baseline = float(np.mean(bl))
    violation = float(np.mean(vl))
    ratio = violation / max(baseline, 1e-8)
    sup = (1 - errors[-1] / errors[0]) * 100
    elapsed = time.perf_counter() - t0
    ap = ns.apical[exc_idx].std().item()

    return {"seed": seed, "mismatch": ratio, "suppression": sup,
            "apical_std": ap, "time": elapsed, "n_exc": n_exc}


# ==============================================================
# TEST 3: Phase 3 extended (3000 trials)
# ==============================================================
def test_phase3_extended(seed, n_trials=3000):
    config = GraphBrainConfig.from_dict({
        "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
        "edges": {"structural": {"enabled": False}},
        "simulation": {"device": "cuda", "seed": seed},
        "hierarchy": {"enabled": False},
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

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    exc_x = ns.position[exc_idx, 0]

    sensory_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    sens_x = ns.position[sensory_nodes, 0]
    sens_y = ns.position[sensory_nodes, 1]
    med_sx, med_sy = sens_x.median(), sens_y.median()
    patterns = {
        0: sensory_nodes[(sens_x < med_sx) & (sens_y < med_sy)],
        1: sensory_nodes[(sens_x >= med_sx) & (sens_y < med_sy)],
        2: sensory_nodes[(sens_x < med_sx) & (sens_y >= med_sy)],
        3: sensory_nodes[(sens_x >= med_sx) & (sens_y >= med_sy)],
    }

    motor_nodes = exc_idx[exc_z >= exc_z.quantile(0.9)]
    motor_x = ns.position[motor_nodes, 0]
    med_mx = motor_x.median()
    left_nodes = motor_nodes[motor_x < med_mx]
    right_nodes = motor_nodes[motor_x >= med_mx]

    # Seed long-range edges
    for motor_pop in [left_nodes, right_nodes]:
        src = sensory_nodes[torch.randint(0, sensory_nodes.shape[0], (30,))]
        dst = motor_pop[torch.randint(0, motor_pop.shape[0], (30,))]
        graph.add_edges(EdgeType.DRIVING, src.to(torch.int32), dst.to(torch.int32),
                        torch.full((30,), 0.1, device=device))
        src2 = motor_pop[torch.randint(0, motor_pop.shape[0], (30,))]
        dst2 = sensory_nodes[torch.randint(0, sensory_nodes.shape[0], (30,))]
        graph.add_edges(EdgeType.MODULATORY, src2.to(torch.int32), dst2.to(torch.int32),
                        torch.full((30,), 0.1, device=device))

    reward_sys = RewardSystem(graph, reward_lr=0.05, eligibility_decay=0.99,
                               lambda_modulation=0.2, modulation_decay=0.85)

    # Warmup
    for w in range(300):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[patterns[w % 4].long()] += 2.0
        error_node_update(ns, inputs)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian(graph, LAMBDA_ACT)
        graph.increment_step()

    reward_map = {0: "LEFT", 1: "RIGHT", 2: "LEFT", 3: "RIGHT"}
    def rev(m):
        return {k: ("RIGHT" if v == "LEFT" else "LEFT") for k, v in m.items()}

    t0 = time.perf_counter()
    results = []

    for trial in range(n_trials):
        pid = trial % 4
        if trial == 500:
            reward_map = rev(reward_map)
        elif trial == 1000:
            reward_map = rev(reward_map)
        elif trial == 1500:
            reward_map = rev(reward_map)
        elif trial == 2000:
            reward_map = rev(reward_map)
        correct_action = reward_map[pid]

        # Present (30 steps)
        for s in range(30):
            step = graph.step_count
            dual_channel_send(graph, mp, step)
            inputs = mp.read_inputs(step)
            inputs.basal[patterns[pid].long()] += 2.0
            error_node_update(ns, inputs)
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    stp.update(graph.edge_store(et), ns, 1.0)
            eff_la = reward_sys.effective_lambda(LAMBDA_ACT)
            apply_hebbian(graph, eff_la)
            reward_sys.update_eligibility(graph)
            reward_sys.step_modulation()
            if step % 100 == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()

        # Decide (10 steps)
        for s in range(10):
            step = graph.step_count
            dual_channel_send(graph, mp, step)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs)
            reward_sys.update_eligibility(graph)
            graph.increment_step()

        left_act = ns.output[left_nodes].mean().item()
        right_act = ns.output[right_nodes].mean().item()
        chosen = "LEFT" if left_act > right_act else "RIGHT"
        correct = chosen == correct_action

        # Reward (10 steps)
        rv = 1.0 if correct else -0.3
        for s in range(10):
            step = graph.step_count
            dual_channel_send(graph, mp, step)
            inputs = mp.read_inputs(step)
            if s == 0:
                motor_all = torch.cat([left_nodes, right_nodes])
                inputs.basal[motor_all.long()] += rv * 3.0
                inputs.basal[exc_idx.long()] += rv * 0.3
            error_node_update(ns, inputs)
            reward_sys.apply_reward(graph, rv)
            reward_sys.step_modulation()
            graph.increment_step()

        results.append(correct)

        if (trial + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            w = results[-100:]
            acc = sum(w) / len(w)
            phase_names = {0: "learn-1", 500: "rev-1", 1000: "learn-2", 1500: "rev-2", 2000: "learn-3"}
            phase = "learn-1"
            for t, p in sorted(phase_names.items()):
                if trial >= t:
                    phase = p
            print(f"    Trial {trial+1:4d} ({phase:>8}): acc={acc:.0%} ({elapsed:.0f}s)", flush=True)

    total = time.perf_counter() - t0
    return {"seed": seed, "n_trials": n_trials, "results": results, "time": total}


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 70, flush=True)
    print("  OVERNIGHT BATTERY: STRESS-TESTING EXISTING SYSTEMS", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Sequential on GPU (no contention)\n", flush=True)

    all_results = {}

    # ---- TEST 1: Phase 1B seed robustness (5 seeds) ----
    print("=" * 50, flush=True)
    print("TEST 1: Phase 1B seed robustness (N=1000, 3000 cycles)", flush=True)
    print("=" * 50, flush=True)
    seed_results = []
    for seed in [42, 123, 456, 789, 1337]:
        print(f"  Seed {seed}...", flush=True)
        r = test_1b_seed(seed, n_exc=1000, n_cycles=3000)
        print(f"    mm={r['mismatch']:.3f}x sup={r['suppression']:.1f}% "
              f"ap={r['apical_std']:.4f} ({r['time']:.0f}s)", flush=True)
        seed_results.append(r)

    mm_vals = [r["mismatch"] for r in seed_results]
    print(f"\n  Mismatch: {np.mean(mm_vals):.3f}x +/- {np.std(mm_vals):.3f}", flush=True)
    print(f"  All above 1.1x: {all(m > 1.1 for m in mm_vals)}", flush=True)
    all_results["test1_seeds"] = seed_results

    # ---- TEST 2: Phase 1B at N=5000 ----
    print(f"\n{'='*50}", flush=True)
    print("TEST 2: Phase 1B at N=5000 (3000 cycles)", flush=True)
    print("=" * 50, flush=True)
    r = test_1b_seed(42, n_exc=4000, n_cycles=3000)
    print(f"  N=5000: mm={r['mismatch']:.3f}x sup={r['suppression']:.1f}% "
          f"ap={r['apical_std']:.4f} ({r['time']:.0f}s)", flush=True)
    all_results["test2_scale"] = r

    # ---- TEST 3: Phase 3 extended (3000 trials, multiple reversals) ----
    print(f"\n{'='*50}", flush=True)
    print("TEST 3: Phase 3 extended (3000 trials, 4 reversals)", flush=True)
    print("=" * 50, flush=True)
    r = test_phase3_extended(42, n_trials=3000)
    # Compute per-phase accuracy
    for phase_start, phase_name in [(0, "learn-1"), (500, "rev-1"), (1000, "learn-2"),
                                      (1500, "rev-2"), (2000, "learn-3")]:
        phase_end = min(phase_start + 500, len(r["results"]))
        phase_results = r["results"][phase_start:phase_end]
        acc = sum(phase_results) / len(phase_results) if phase_results else 0
        print(f"  {phase_name}: {acc:.0%}", flush=True)
    all_results["test3_extended"] = {"seed": r["seed"], "time": r["time"],
                                      "results": r["results"]}

    # ---- TEST 4: Phase 3 seed robustness ----
    print(f"\n{'='*50}", flush=True)
    print("TEST 4: Phase 3 seed robustness (3 seeds, 1000 trials)", flush=True)
    print("=" * 50, flush=True)
    p3_seeds = []
    for seed in [42, 123, 789]:
        print(f"  Seed {seed}...", flush=True)
        r = test_phase3_extended(seed, n_trials=1000)
        learn_acc = sum(r["results"][200:300]) / 100
        print(f"    learn_acc={learn_acc:.0%} ({r['time']:.0f}s)", flush=True)
        p3_seeds.append({"seed": seed, "learn_acc": learn_acc, "time": r["time"]})
    acc_vals = [r["learn_acc"] for r in p3_seeds]
    print(f"\n  RL accuracy: {np.mean(acc_vals):.0%} +/- {np.std(acc_vals):.0%}", flush=True)
    all_results["test4_p3_seeds"] = p3_seeds

    # ---- SUMMARY ----
    print(f"\n{'='*70}", flush=True)
    print("  OVERNIGHT BATTERY SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Test 1 (1B seeds):  mm={np.mean(mm_vals):.3f}x +/- {np.std(mm_vals):.3f} "
          f"({'ROBUST' if all(m > 1.1 for m in mm_vals) else 'FRAGILE'})", flush=True)
    print(f"  Test 2 (1B scale):  mm={all_results['test2_scale']['mismatch']:.3f}x at N=5000 "
          f"({'SCALES' if all_results['test2_scale']['mismatch'] > 1.1 else 'FAILS TO SCALE'})", flush=True)
    print(f"  Test 3 (P3 ext):    see per-phase accuracy above", flush=True)
    print(f"  Test 4 (P3 seeds):  acc={np.mean(acc_vals):.0%} +/- {np.std(acc_vals):.0%} "
          f"({'ROBUST' if all(a > 0.6 for a in acc_vals) else 'FRAGILE'})", flush=True)

    torch.save(all_results, "overnight_battery_results.pt")
    print(f"\nSaved to overnight_battery_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
