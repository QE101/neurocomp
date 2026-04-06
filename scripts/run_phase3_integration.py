"""Phase 3: Multi-System Integration — The Last Existential Test.

Can PC + RL + Episodic Memory coexist on the same graph?

Four conditions:
  1. PC-only: baseline suppression/mismatch
  2. PC + RL: action selection with reward
  3. PC + RL + Memory: full system with hippocampal episodic memory
  4. RL-only: standard node model (no universal error)

Task: rewarded pattern navigation with reversal.
600 trials, 50 steps each, ~10 min total.

Unbuffered output throughout.
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
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.reward import RewardSystem
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1
STRENGTH = 2.0
REWARD_STRENGTH = 1.5
CUE_STRENGTH = 1.0
N_TRIALS = 600
PRESENTATION_STEPS = 30
CUE_STEPS = 5
DECISION_STEPS = 10
REWARD_STEPS = 10
ENCODE_STEPS = 5

CONFIG_DICT = {
    "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
    "edges": {"structural": {"enabled": False}},
    "simulation": {"device": "cuda", "seed": 42},
    "hierarchy": {"enabled": False},
}


def error_node_update(ns, inputs, dt=1.0):
    """Universal error model from Phase 1B attempt 15."""
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)
    device = ns.device

    exc_f = exc_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * exc_f
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
    pred_err = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    exc_out = F.softplus(pred_err.abs()) * pv_gain * ns.gain
    ns.output = torch.where(exc_mask, exc_out, ns.output)
    ns.prediction_error = torch.where(exc_mask, pred_err, ns.prediction_error)

    for inh_type, mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
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


def standard_node_update(ns, inputs, dt=1.0):
    """Standard two-compartment model (for RL-only condition)."""
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    device = ns.device

    exc_f = exc_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * exc_f
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical) * exc_f
    out = F.softplus(ns.basal) * ns.gain * exc_f
    ns.output = torch.where(exc_mask, out, ns.output)

    for mask in [ns.type_mask(NodeType.PV), ns.type_mask(NodeType.SST), ns.type_mask(NodeType.VIP)]:
        f = mask.float()
        ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * f
        ns.output = torch.where(mask, F.softplus(ns.basal) * ns.gain * f, ns.output)

    ns.output += torch.randn(ns.n_nodes, device=device) * 0.005
    ns.output.clamp_(min=0.0)


def dual_channel_send(graph, mp, step):
    """Dual-channel routing from Phase 1B attempt 15."""
    ns = graph.node_state
    output = ns.output
    content = F.softplus(ns.basal)

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
                              torch.ones(store.n_edges, dtype=torch.long, device=graph.device), step)


def apply_hebbian(graph, la):
    """Simultaneous Hebbian from Phase 1B."""
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


def run_condition(name, use_rl, use_memory, use_error_model, result_queue):
    """Run one experimental condition."""
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
    node_update = error_node_update if use_error_model else standard_node_update

    # Setup reward system
    reward_sys = RewardSystem(graph) if use_rl else None

    # Setup regions
    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    exc_x = ns.position[exc_idx, 0]
    exc_y = ns.position[exc_idx, 1]

    # Sensory (bottom 20%)
    sensory_mask = exc_z <= exc_z.quantile(0.2)
    sensory_nodes = exc_idx[sensory_mask]
    sens_x = ns.position[sensory_nodes, 0]
    sens_y = ns.position[sensory_nodes, 1]
    med_sx = sens_x.median()
    med_sy = sens_y.median()

    patterns = {
        0: sensory_nodes[(sens_x < med_sx) & (sens_y < med_sy)],
        1: sensory_nodes[(sens_x >= med_sx) & (sens_y < med_sy)],
        2: sensory_nodes[(sens_x < med_sx) & (sens_y >= med_sy)],
        3: sensory_nodes[(sens_x >= med_sx) & (sens_y >= med_sy)],
    }

    # Motor (top 10%)
    motor_mask = exc_z >= exc_z.quantile(0.9)
    motor_nodes = exc_idx[motor_mask]
    motor_x = ns.position[motor_nodes, 0]
    med_mx = motor_x.median()
    left_nodes = motor_nodes[motor_x < med_mx]
    right_nodes = motor_nodes[motor_x >= med_mx]

    # Episodic memory
    memory = EpisodicMemory(graph) if use_memory else None
    if memory:
        # Rebuild reward system eligibility after memory added new edges
        reward_sys = RewardSystem(graph) if use_rl else None

    # Reward mapping
    reward_map = {0: "LEFT", 1: "RIGHT", 2: "LEFT", 3: "RIGHT"}

    def reverse_map(m):
        return {k: ("RIGHT" if v == "LEFT" else "LEFT") for k, v in m.items()}

    # Pre-training: let PC settle (200 steps with random patterns)
    for warmup in range(200):
        pid = warmup % 4
        step = graph.step_count
        dual_channel_send(graph, mp, step) if use_error_model else mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[patterns[pid].long()] += STRENGTH
        node_update(ns, inputs)
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
        correct_action = reward_map[pattern_id]

        # Reversal at trial 200 and 400
        if trial == 200:
            reward_map = reverse_map(reward_map)
        elif trial == 400:
            reward_map = reverse_map(reward_map)
            correct_action = reward_map[pattern_id]

        correct_action = reward_map[pattern_id]

        # Phase A: Presentation
        for s in range(PRESENTATION_STEPS):
            step = graph.step_count
            dual_channel_send(graph, mp, step) if use_error_model else mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pat_nodes.long()] += STRENGTH
            node_update(ns, inputs)
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    stp.update(graph.edge_store(et), ns, 1.0)
            effective_la = reward_sys.effective_lambda(LAMBDA_ACT) if reward_sys else LAMBDA_ACT
            apply_hebbian(graph, effective_la)
            if reward_sys:
                reward_sys.update_eligibility(graph)
                reward_sys.step_modulation()
            if step % 100 == 0:
                for et in EdgeType:
                    if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                        hom.update(graph.edge_store(et), ns, 1.0)
                ip.update(ns)
            graph.increment_step()

        # Phase B: Memory cue
        if memory:
            cue_nodes = memory.cue(graph, pattern_id)
            for s in range(CUE_STEPS):
                step = graph.step_count
                dual_channel_send(graph, mp, step)
                inputs = mp.read_inputs(step)
                if cue_nodes.numel() > 0:
                    inputs.basal[cue_nodes.long()] += CUE_STRENGTH
                node_update(ns, inputs)
                graph.increment_step()

        # Phase C: Decision (no input)
        for s in range(DECISION_STEPS):
            step = graph.step_count
            dual_channel_send(graph, mp, step) if use_error_model else mp.send_messages(graph, step)
            inputs = mp.read_inputs(step)
            node_update(ns, inputs)
            if reward_sys:
                reward_sys.update_eligibility(graph)
            graph.increment_step()

        # Read action
        left_act = ns.output[left_nodes].mean().item()
        right_act = ns.output[right_nodes].mean().item()
        chosen = "LEFT" if left_act > right_act else "RIGHT"
        correct = (chosen == correct_action)

        # Phase D: Reward
        if use_rl:
            reward_val = 1.0 if correct else -0.3
            for s in range(REWARD_STEPS):
                step = graph.step_count
                dual_channel_send(graph, mp, step) if use_error_model else mp.send_messages(graph, step)
                inputs = mp.read_inputs(step)
                if s == 0:
                    inputs.basal[exc_idx.long()] += reward_val * REWARD_STRENGTH
                node_update(ns, inputs)
                reward_sys.apply_reward(graph, reward_val)
                reward_sys.step_modulation()
                graph.increment_step()
        else:
            for s in range(REWARD_STEPS):
                step = graph.step_count
                dual_channel_send(graph, mp, step) if use_error_model else mp.send_messages(graph, step)
                inputs = mp.read_inputs(step)
                node_update(ns, inputs)
                graph.increment_step()

        # Phase E: Memory storage
        if memory and trial % 5 == 0:
            memory.encode(graph, ENCODE_STEPS)

        # Record
        trial_results.append({
            "trial": trial,
            "pattern": pattern_id,
            "chosen": chosen,
            "correct": correct,
            "left_act": left_act,
            "right_act": right_act,
            "apical_std": ns.apical[exc_idx].std().item(),
            "output_mean": ns.output[exc_idx].mean().item(),
        })

        # Progress
        if (trial + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            window = trial_results[-50:]
            acc = sum(1 for t in window if t["correct"]) / len(window)
            ap = np.mean([t["apical_std"] for t in window])
            phase = "learn" if trial < 200 else ("reversal" if trial < 400 else "re-reversal")
            print(f"  [{name}] Trial {trial+1:3d} ({phase}): acc={acc:.0%} "
                  f"apical={ap:.4f} ({elapsed:.0f}s)", flush=True)

    # Compute summary
    total_time = time.perf_counter() - t0

    def window_acc(start, end):
        w = [t for t in trial_results if start <= t["trial"] < end]
        return sum(1 for t in w if t["correct"]) / max(len(w), 1)

    # Learning accuracy (last 100 of phase 1)
    learn_acc = window_acc(100, 200)
    # Reversal: trials to 60% (in windows of 20)
    rev_speed = N_TRIALS  # default: never reached
    for i in range(200, 400, 20):
        if window_acc(i, i + 20) >= 0.6:
            rev_speed = i - 200
            break
    # Re-reversal speed
    rerev_speed = N_TRIALS
    for i in range(400, 600, 20):
        if window_acc(i, i + 20) >= 0.6:
            rerev_speed = i - 400
            break

    ap_mean = np.mean([t["apical_std"] for t in trial_results])
    out_mean = np.mean([t["output_mean"] for t in trial_results])

    result = {
        "name": name, "learn_acc": learn_acc,
        "rev_speed": rev_speed, "rerev_speed": rerev_speed,
        "apical_std_mean": ap_mean, "output_mean": out_mean,
        "time": total_time, "trial_results": trial_results,
    }

    print(f"  [{name}] DONE: learn_acc={learn_acc:.0%} rev_speed={rev_speed} "
          f"rerev_speed={rerev_speed} apical={ap_mean:.4f} ({total_time:.0f}s)", flush=True)
    result_queue.put(result)


def main():
    print("=" * 70, flush=True)
    print("  PHASE 3: MULTI-SYSTEM INTEGRATION", flush=True)
    print("  Can PC + RL + Episodic Memory coexist?", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Task: rewarded pattern navigation with reversal", flush=True)
    print(f"Trials: {N_TRIALS} (200 learn + 200 reversal + 200 re-reversal)\n", flush=True)

    conditions = [
        ("PC-only", False, False, True),
        ("PC+RL", True, False, True),
        ("PC+RL+Mem", True, True, True),
        ("RL-only", True, False, False),
    ]

    queue = Queue()
    procs = []
    for name, rl, mem, err_model in conditions:
        p = Process(target=run_condition, args=(name, rl, mem, err_model, queue))
        p.start()
        procs.append(p)
        print(f"  Launched: {name}", flush=True)

    for p in procs:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r["name"])

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("  PHASE 3 RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Condition':<12} | {'Learn%':>6} | {'RevSpd':>6} | {'ReRevSpd':>8} | "
          f"{'Apical':>7} | {'Time':>5}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        rev = f"{r['rev_speed']}" if r['rev_speed'] < N_TRIALS else "never"
        rerev = f"{r['rerev_speed']}" if r['rerev_speed'] < N_TRIALS else "never"
        print(f"{r['name']:<12} | {r['learn_acc']:5.0%} | {rev:>6} | {rerev:>8} | "
              f"{r['apical_std_mean']:7.4f} | {r['time']:5.0f}s", flush=True)

    # Success criteria check
    print(f"\n{'='*80}", flush=True)
    print("  SUCCESS CRITERIA", flush=True)
    print(f"{'='*80}", flush=True)

    pc_rl = next((r for r in results if r["name"] == "PC+RL"), None)
    pc_only = next((r for r in results if r["name"] == "PC-only"), None)
    full = next((r for r in results if r["name"] == "PC+RL+Mem"), None)
    rl_only = next((r for r in results if r["name"] == "RL-only"), None)

    checks = {}
    if pc_rl:
        checks["RL works (>65%)"] = pc_rl["learn_acc"] > 0.65
        checks["PC survives (apical>0.05)"] = pc_rl["apical_std_mean"] > 0.05
    if full and pc_rl:
        checks["Memory helps (30% faster)"] = (full["rerev_speed"] < pc_rl["rev_speed"] * 0.7
                                                 if pc_rl["rev_speed"] < N_TRIALS else False)
    if pc_rl and pc_only:
        interference = abs(pc_rl["apical_std_mean"] - pc_only["apical_std_mean"]) / max(pc_only["apical_std_mean"], 0.001)
        checks["No interference (<20%)"] = interference < 0.2
    if pc_rl and rl_only:
        checks["Error model helps RL"] = pc_rl["learn_acc"] > rl_only["learn_acc"]

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}", flush=True)

    all_pass = all(checks.values()) if checks else False
    print(f"\n  OVERALL: {'Q3 PASSED — SYSTEMS COEXIST' if all_pass else 'Q3 NEEDS WORK'}", flush=True)
    print(f"{'='*80}", flush=True)

    torch.save(results, "phase3_integration_results.pt")
    print(f"\nSaved to phase3_integration_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
