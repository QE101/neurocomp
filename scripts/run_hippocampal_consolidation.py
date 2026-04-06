"""Hippocampal Consolidation: the biologically correct test.

Wake phase: present patterns, encode to hippocampus, cortex learns at full rate.
Sleep phase: replay stored patterns interleaved at 0.1x rate.
The cortex NEVER faces rapid A-B alternation — it gets curated replays.

Oja stabilizer learning rule for cortex.
N=1250 first (fast iteration), then N=50K.

Copy-pasted core functions from run_stability_battery.py (never reimplement).
"""

import sys, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from graph_brain.config import GraphBrainConfig, HippocampalConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.hippocampus import HippocampalSystem
from graph_brain.types import EdgeType, NodeType

STRENGTH = 2.0
PD = 50

SMALL_CONFIG = {
    'nodes': {'n_excitatory': 1000, 'n_pv': 90, 'n_sst': 90, 'n_vip': 70, 'noise_std': 0.005},
    'edges': {'structural': {'enabled': False}},
    'simulation': {'device': 'cuda', 'seed': 42},
    'hierarchy': {'enabled': False},
    'hippocampal': {
        'enabled': True,
        'n_dg': 2000,
        'n_ca3': 500,
        'dg_sparsity': 0.02,
        'dg_fan_in': 200,
        'ca3_sparsity': 0.10,
        'encoding_lr': 0.5,
        'replay_strength': 0.5,
        'replay_lr_scale': 0.1,
        'max_patterns': 20,
        'replay_interleave': 5,
        'replay_steps': 30,
    },
}

# Wake-sleep schedule
WAKE_CYCLES = 200     # present each pattern for 200 cycles before sleep
SLEEP_REPLAYS = 5     # replay cycles per sleep phase
REPLAY_STEPS = 30     # steps per pattern replay
TOTAL_WAKE = 2000     # total wake cycles
N_SLEEP_PHASES = TOTAL_WAKE // WAKE_CYCLES  # = 10 sleep phases


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


# Oja stabilizer (validated learning rule)
def apply_hebbian_oja(graph, lr_scale=1.0):
    ns_ = graph.node_state
    lr = 0.001 * lr_scale
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0: continue
            src = ns_.output[store.src.long()]
            dst = ns_.output[store.dst.long()]
            dw = lr * (src * dst - 0.0065 * 2.0 * store.weight - dst * dst * store.weight)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


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
# MAIN
# ================================================================
def main():
    print('=' * 60, flush=True)
    print('  HIPPOCAMPAL CONSOLIDATION TEST', flush=True)
    print(f'  Wake: {WAKE_CYCLES} cycles, then sleep: {SLEEP_REPLAYS} replay cycles', flush=True)
    print(f'  {N_SLEEP_PHASES} sleep phases across {TOTAL_WAKE} wake cycles', flush=True)
    print('  Oja stabilizer + theta. N=1250.', flush=True)
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

    # Initialize hippocampal system
    hipp = HippocampalSystem(
        config=config.hippocampal,
        cortical_input_indices=input_nodes,
        n_cortical=N,
        device=device,
        seed=config.simulation.seed,
    )

    print(f'  Patterns: A={pa.shape[0]}, B={pb.shape[0]}', flush=True)
    print(f'  Hippocampus: DG={config.hippocampal.n_dg}, CA3={config.hippocampal.n_ca3}', flush=True)
    print(f'  DG sparsity: {config.hippocampal.dg_sparsity} (k={hipp.dg.k})', flush=True)
    print(f'  CA3 sparsity: {config.hippocampal.ca3_sparsity} (k={hipp.ca3.k})', flush=True)

    t0 = time.perf_counter()
    all_mm = []
    wake_cycle = 0

    for phase in range(N_SLEEP_PHASES):
        # ============ WAKE PHASE ============
        print(f'\n  --- WAKE PHASE {phase+1}/{N_SLEEP_PHASES} ({WAKE_CYCLES} cycles) ---', flush=True)
        for cyc in range(WAKE_CYCLES):
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
                    apply_hebbian_oja(graph, lr_scale=1.0)
                    if step % 100 == 0:
                        for et in EdgeType:
                            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                                hom.update(graph.edge_store(et), ns, 1.0)
                        ip.update(ns)
                    graph.increment_step()

                # Encode to hippocampus after each pattern presentation
                hipp.encode(ns.output, graph.step_count)

            wake_cycle += 1

        # Mismatch BEFORE sleep
        bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
        mm_before = vl / max(bl, 1e-8)
        elapsed = time.perf_counter() - t0
        mm_str = f'{mm_before:.3f}x' + (' **' if mm_before > 1.1 else '')
        print(f'  Pre-sleep mm={mm_str} (wake_cyc={wake_cycle}) ({elapsed:.0f}s)', flush=True)

        # ============ SLEEP PHASE ============
        print(f'  --- SLEEP PHASE {phase+1} ({SLEEP_REPLAYS} replay cycles, {hipp.n_stored()} patterns stored) ---', flush=True)
        schedule = hipp.replay_schedule(SLEEP_REPLAYS)

        for pat_idx in schedule:
            replay_activation = hipp.get_replay_pattern(pat_idx)
            for s in range(REPLAY_STEPS):
                step = graph.step_count
                dual_channel_send(ns, graph, mp, device)
                inputs = mp.read_inputs(step)
                # Inject replay into cortical input region
                inputs.basal[input_nodes.long()] += replay_activation
                # No theta during sleep (just like biology)
                error_node_update(ns, inputs, theta_mod=1.0)
                # Learn at reduced rate
                apply_hebbian_oja(graph, lr_scale=config.hippocampal.replay_lr_scale)
                graph.increment_step()

        # Mismatch AFTER sleep
        bl, vl = run_mismatch(graph, ns, mp, device, theta, pa, pb, input_nodes)
        mm_after = vl / max(bl, 1e-8)
        all_mm.append(mm_after)
        elapsed = time.perf_counter() - t0
        mm_str = f'{mm_after:.3f}x' + (' **' if mm_after > 1.1 else '')
        delta = mm_after - mm_before
        print(f'  Post-sleep mm={mm_str} (delta={delta:+.3f}) ({elapsed:.0f}s)', flush=True)

        # Weight health check
        drv_w = graph.edge_store(EdgeType.DRIVING).weight.mean().item()
        mod_w = graph.edge_store(EdgeType.MODULATORY).weight.mean().item() if graph.has_edge_type(EdgeType.MODULATORY) else 0
        print(f'  Weights: drv={drv_w:.4f} mod={mod_w:.4f}', flush=True)

    # ============ FINAL ANALYSIS ============
    print(f'\n{"="*60}', flush=True)
    print('  RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  All post-sleep mismatch: {[f"{m:.3f}" for m in all_mm]}', flush=True)

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

    # Flatness
    if len(all_mm) >= 3:
        diffs = [abs(all_mm[i+1] - all_mm[i]) for i in range(len(all_mm)-1)]
        max_diff = max(diffs)
        avg_diff = np.mean(diffs)
        print(f'  Oscillation: first_half={fr:.3f} second_half={sr:.3f} {damping}', flush=True)
        print(f'  Trajectory: max_step={max_diff:.3f} avg_step={avg_diff:.3f}', flush=True)

        if max_diff < 0.3 and best > 1.1:
            print(f'\n  VERDICT: STABLE MISMATCH -- hippocampal consolidation WORKS', flush=True)
        elif best > 1.1 and sr < 0.3:
            print(f'\n  VERDICT: CONVERGING -- consolidation stabilizing', flush=True)
        elif best < 1.05:
            print(f'\n  VERDICT: No mismatch (replay too weak?)', flush=True)
        else:
            print(f'\n  VERDICT: {damping} (max_step={max_diff:.3f})', flush=True)

    torch.save({'all_mm': all_mm, 'damping': damping, 'best': best, 'final': final},
               'hippocampal_consolidation_results.pt')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
