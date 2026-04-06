"""Phase 3 v2: Fixed RL mechanism.

Three fixes from v1 diagnosis:
1. Eligibility decay 0.99 (was 0.95) — traces survive the sensory→motor path
2. Reward targeted at motor region (was global) — focused learning signal
3. Seed long-range sensory→motor edges — bridge the 70% spatial gap

Also: more trials (1000) and stronger reward signal.
Unbuffered output.
"""

import sys
import os
import time
from multiprocessing import Process, Queue

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
from graph_brain.episodic import EpisodicMemory
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.reward import RewardSystem
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1
STRENGTH = 2.0
REWARD_STRENGTH = 3.0  # stronger (was 1.5)
CUE_STRENGTH = 1.5
N_TRIALS = 1000  # more trials (was 600)

CONFIG_DICT = {
    "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
    "edges": {"structural": {"enabled": False}},
    "simulation": {"device": "cuda", "seed": 42},
    "hierarchy": {"enabled": False},
}


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


def seed_long_range_edges(graph, sensory_nodes, left_nodes, right_nodes, n_per_pair=20):
    """Fix 3: Seed a few long-range DRIVING edges from sensory to motor regions.

    Without these, there's no pathway for sensory→motor signal flow.
    The RL system will strengthen useful ones and let useless ones decay.
    """
    device = graph.device
    # Random sensory→left and sensory→right edges
    for motor_pop in [left_nodes, right_nodes]:
        src_idx = sensory_nodes[torch.randint(0, sensory_nodes.shape[0], (n_per_pair,))]
        dst_idx = motor_pop[torch.randint(0, motor_pop.shape[0], (n_per_pair,))]
        weights = torch.full((n_per_pair,), 0.1, device=device)
        graph.add_edges(EdgeType.DRIVING, src_idx.to(torch.int32), dst_idx.to(torch.int32), weights)

    # Also seed modulatory motor→sensory (for prediction feedback)
    for motor_pop in [left_nodes, right_nodes]:
        src_idx = motor_pop[torch.randint(0, motor_pop.shape[0], (n_per_pair,))]
        dst_idx = sensory_nodes[torch.randint(0, sensory_nodes.shape[0], (n_per_pair,))]
        weights = torch.full((n_per_pair,), 0.1, device=device)
        graph.add_edges(EdgeType.MODULATORY, src_idx.to(torch.int32), dst_idx.to(torch.int32), weights)


def run_condition(name, use_rl, use_memory, result_queue):
    config = GraphBrainConfig.from_dict(CONFIG_DICT)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    mp = TypedMessagePasser(config, N, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    # Regions
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    exc_x = ns.position[exc_idx, 0]
    exc_y = ns.position[exc_idx, 1]

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

    # Fix 3: Seed long-range edges
    seed_long_range_edges(graph, sensory_nodes, left_nodes, right_nodes, n_per_pair=30)

    # Memory
    memory = EpisodicMemory(graph) if use_memory else None

    # Fix 1: Slower eligibility decay (0.99 not 0.95)
    # Fix 2: Stronger reward lr
    reward_sys = RewardSystem(
        graph, reward_lr=0.05, eligibility_decay=0.99,
        lambda_modulation=0.2, modulation_decay=0.85,
    ) if use_rl else None

    reward_map = {0: "LEFT", 1: "RIGHT", 2: "LEFT", 3: "RIGHT"}

    def reverse_map(m):
        return {k: ("RIGHT" if v == "LEFT" else "LEFT") for k, v in m.items()}

    # Warmup PC (300 steps)
    for w in range(300):
        step = graph.step_count
        dual_channel_send(graph, mp, step)
        inputs = mp.read_inputs(step)
        inputs.basal[patterns[w % 4].long()] += STRENGTH
        error_node_update(ns, inputs)
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, 1.0)
        apply_hebbian(graph, LAMBDA_ACT)
        graph.increment_step()

    t0 = time.perf_counter()
    trial_results = []

    for trial in range(N_TRIALS):
        pattern_id = trial % 4
        pat_nodes = patterns[pattern_id]

        if trial == 300:
            reward_map = reverse_map(reward_map)
        elif trial == 600:
            reward_map = reverse_map(reward_map)
        correct_action = reward_map[pattern_id]

        # Presentation (30 steps)
        for s in range(30):
            step = graph.step_count
            dual_channel_send(graph, mp, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pat_nodes.long()] += STRENGTH
            error_node_update(ns, inputs)
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    stp.update(graph.edge_store(et), ns, 1.0)
            eff_la = reward_sys.effective_lambda(LAMBDA_ACT) if reward_sys else LAMBDA_ACT
            apply_hebbian(graph, eff_la)
            if reward_sys:
                reward_sys.update_eligibility(graph)
                reward_sys.step_modulation()
            if step % 100 == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()

        # Memory cue (5 steps)
        if memory:
            cue_nodes = memory.cue(graph, pattern_id)
            for s in range(5):
                step = graph.step_count
                dual_channel_send(graph, mp, step)
                inputs = mp.read_inputs(step)
                if cue_nodes.numel() > 0:
                    inputs.basal[cue_nodes.long()] += CUE_STRENGTH
                error_node_update(ns, inputs)
                graph.increment_step()

        # Decision (10 steps)
        for s in range(10):
            step = graph.step_count
            dual_channel_send(graph, mp, step)
            inputs = mp.read_inputs(step)
            error_node_update(ns, inputs)
            if reward_sys:
                reward_sys.update_eligibility(graph)
            graph.increment_step()

        # Read action
        left_act = ns.output[left_nodes].mean().item()
        right_act = ns.output[right_nodes].mean().item()
        chosen = "LEFT" if left_act > right_act else "RIGHT"
        correct = (chosen == correct_action)

        # Reward (10 steps) — Fix 2: targeted at motor region
        if use_rl:
            reward_val = 1.0 if correct else -0.3
            for s in range(10):
                step = graph.step_count
                dual_channel_send(graph, mp, step)
                inputs = mp.read_inputs(step)
                if s == 0:
                    # Targeted reward: inject into MOTOR nodes, not all excitatory
                    motor_all = torch.cat([left_nodes, right_nodes])
                    inputs.basal[motor_all.long()] += reward_val * REWARD_STRENGTH
                    # Smaller global signal for context
                    inputs.basal[exc_idx.long()] += reward_val * REWARD_STRENGTH * 0.1
                error_node_update(ns, inputs)
                reward_sys.apply_reward(graph, reward_val)
                reward_sys.step_modulation()
                graph.increment_step()
        else:
            for s in range(10):
                step = graph.step_count
                dual_channel_send(graph, mp, step)
                inputs = mp.read_inputs(step)
                error_node_update(ns, inputs)
                graph.increment_step()

        # Memory storage
        if memory and correct:  # only store successful episodes
            memory.encode(graph, 5)

        trial_results.append({
            "trial": trial, "correct": correct,
            "left_act": left_act, "right_act": right_act,
            "apical_std": ns.apical[exc_idx].std().item(),
        })

        if (trial + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            window = trial_results[-50:]
            acc = sum(1 for t in window if t["correct"]) / len(window)
            ap = np.mean([t["apical_std"] for t in window])
            phase = "learn" if trial < 300 else ("reversal" if trial < 600 else "re-reversal")
            print(f"  [{name}] Trial {trial+1:4d} ({phase:>11}): acc={acc:.0%} "
                  f"apical={ap:.4f} ({elapsed:.0f}s)", flush=True)

    total_time = time.perf_counter() - t0

    def window_acc(start, end):
        w = [t for t in trial_results if start <= t["trial"] < end]
        return sum(1 for t in w if t["correct"]) / max(len(w), 1)

    learn_acc = window_acc(200, 300)
    # Reversal speed
    rev_speed = N_TRIALS
    for i in range(300, 600, 20):
        if window_acc(i, i + 20) >= 0.6:
            rev_speed = i - 300
            break
    # Re-reversal speed
    rerev_speed = N_TRIALS
    for i in range(600, 1000, 20):
        if window_acc(i, i + 20) >= 0.6:
            rerev_speed = i - 600
            break

    ap_mean = np.mean([t["apical_std"] for t in trial_results])
    result = {
        "name": name, "learn_acc": learn_acc,
        "rev_speed": rev_speed, "rerev_speed": rerev_speed,
        "apical_std_mean": ap_mean, "time": total_time,
        "trial_results": trial_results,
    }
    print(f"  [{name}] DONE: learn={learn_acc:.0%} rev={rev_speed} rerev={rerev_speed} "
          f"ap={ap_mean:.4f} ({total_time:.0f}s)", flush=True)
    result_queue.put(result)


def main():
    print("=" * 70, flush=True)
    print("  PHASE 3 v2: FIXED RL MECHANISM", flush=True)
    print("=" * 70, flush=True)
    print(f"Fixes: eligibility_decay=0.99, targeted motor reward, seeded long-range edges", flush=True)
    print(f"Trials: {N_TRIALS} (300 learn + 300 reversal + 400 re-reversal)", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    conditions = [
        ("PC+RL", True, False),
        ("PC+RL+Mem", True, True),
    ]

    queue = Queue()
    procs = []
    for name, rl, mem in conditions:
        p = Process(target=run_condition, args=(name, rl, mem, queue))
        p.start()
        procs.append(p)
        print(f"  Launched: {name}", flush=True)

    for p in procs:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r["name"])

    print(f"\n{'='*70}", flush=True)
    print("  RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    for r in results:
        rev = f"{r['rev_speed']}" if r['rev_speed'] < N_TRIALS else "never"
        rerev = f"{r['rerev_speed']}" if r['rerev_speed'] < N_TRIALS else "never"
        print(f"  {r['name']:<12}: learn={r['learn_acc']:.0%} rev_speed={rev} "
              f"rerev_speed={rerev} apical={r['apical_std_mean']:.4f}", flush=True)

    # Check RL learning
    pc_rl = next((r for r in results if r["name"] == "PC+RL"), None)
    full = next((r for r in results if r["name"] == "PC+RL+Mem"), None)

    if pc_rl:
        print(f"\n  RL learning: {'PASS' if pc_rl['learn_acc'] > 0.6 else 'FAIL'} "
              f"({pc_rl['learn_acc']:.0%})", flush=True)
        print(f"  PC survival: {'PASS' if pc_rl['apical_std_mean'] > 0.05 else 'FAIL'} "
              f"(apical={pc_rl['apical_std_mean']:.4f})", flush=True)
    if full and pc_rl:
        mem_helps = full["rerev_speed"] < pc_rl["rev_speed"] * 0.7 if pc_rl["rev_speed"] < N_TRIALS else False
        print(f"  Memory helps: {'PASS' if mem_helps else 'FAIL'}", flush=True)

    torch.save(results, "phase3_v2_results.pt")
    print(f"\nSaved to phase3_v2_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
