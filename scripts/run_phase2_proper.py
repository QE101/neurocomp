"""Phase 2: Proper Oscillatory Dynamics — Calibration + 4 Challenge Tests.

Sequential on GPU, unbuffered output, exact functions from working scripts.
"""

import sys
import os
import time
import math

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
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.predictive_coding import PredictiveCodingModel, PCWeightUpdate
from graph_brain.oscillations import PINGMechanism, ThetaDrive, OscillationAnalyzer
from graph_brain.reward import RewardSystem
from graph_brain.types import EdgeType, HierarchyLevel, NodeRole, NodeType

LAMBDA_ACT = 3.1
STRENGTH = 2.0
PD = 50

CONFIG = {
    "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
    "edges": {"structural": {"enabled": False}},
    "simulation": {"device": "cuda", "seed": 42},
    "hierarchy": {"enabled": False},
}


# === EXACT functions from run_error_model_emergence.py (Phase 1B working code) ===

def make_phase1b_helpers(graph, mp):
    """Create closure-based helpers exactly matching the working Phase 1B script."""
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    stp = ShortTermPlasticity(graph.config.edges.stp)
    hom = HomeostaticScaling(graph.config.edges.homeostatic)
    ip = IntrinsicPlasticity(graph.config.nodes)

    exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    def error_node_update(ns, inputs, dt=1.0, noise_std=0.005,
                          ping=None, theta_mod=1.0):
        """Phase 1B node update with optional oscillatory modifications."""
        device = ns.device
        N = ns.n_nodes
        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        pv_mask = ns.type_mask(NodeType.PV)
        sst_mask = ns.type_mask(NodeType.SST)
        vip_mask = ns.type_mask(NodeType.VIP)

        exc_f = exc_mask.float()

        # Theta modulation on excitatory basal input
        exc_basal_input = inputs.basal * theta_mod if theta_mod != 1.0 else inputs.basal

        # Add local field coupling for PV (EXC→PV drive)
        if ping is not None:
            field_drive = ping.compute_field_drive(graph)
            inputs.basal = inputs.basal + field_drive

        # Per-node tau (PV gets fast tau if PING active)
        if ping is not None:
            exc_tau = 10.0
            pv_tau_val = ping.pv_tau
        else:
            exc_tau = 10.0
            pv_tau_val = 10.0

        # EXCITATORY
        ns.basal += 1.0 * (-ns.basal / exc_tau + exc_basal_input) * exc_f
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
        ns.apical += 1.0 * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f
        prediction_error = ns.basal - ns.apical

        # PV gain: GABA dynamics if PING, instantaneous otherwise
        if ping is not None:
            pv_gain = ping.update_gaba(inputs.pv_inhibition)
        else:
            pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)

        exc_output = F.softplus(prediction_error.abs()) * pv_gain * ns.gain
        ns.output = torch.where(exc_mask, exc_output, ns.output)
        ns.prediction_error = torch.where(exc_mask, prediction_error, ns.prediction_error)

        # PV
        pv_f = pv_mask.float()
        ns.basal += 1.0 * (-ns.basal / pv_tau_val + inputs.basal + inputs.electrical) * pv_f
        pv_out = F.softplus(ns.basal) * ns.gain * pv_f
        ns.output = torch.where(pv_mask, pv_out, ns.output)

        # SST
        sst_f = sst_mask.float()
        ns.basal += 1.0 * (-ns.basal / 10.0 + inputs.basal) * sst_f
        vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        sst_out = F.softplus(ns.basal) * ns.gain * vip_inhib * sst_f
        ns.output = torch.where(sst_mask, sst_out, ns.output)

        # VIP
        vip_f = vip_mask.float()
        ns.basal += 1.0 * (-ns.basal / 10.0 + inputs.basal) * vip_f
        vip_out = F.softplus(ns.basal) * ns.gain * vip_f
        ns.output = torch.where(vip_mask, vip_out, ns.output)

        ns.output += torch.randn(N, device=device) * noise_std
        ns.output.clamp_(min=0.0)
        ns.activity_ema.lerp_(ns.output, 1.0 / 1000.0)

    def dual_channel_send(graph, step):
        output = ns.output
        content = F.softplus(ns.basal)
        for edge_type in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
            if not graph.has_edge_type(edge_type): continue
            store = graph.edge_store(edge_type)
            msg = output[store.src.long()] * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.DRIVING: Channel.BASAL, EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
                       EdgeType.RETROGRADE: Channel.RETROGRADE}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)
        for edge_type in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
            if not graph.has_edge_type(edge_type): continue
            store = graph.edge_store(edge_type)
            msg = content[store.src.long()] * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.MODULATORY: Channel.APICAL, EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            src_out = output[store.src.long()]
            dst_out = output[store.dst.long()]
            gap_current = store.weight * (src_out - dst_out)
            delay_s = torch.ones(store.n_edges, dtype=torch.long, device=device)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap_current, delay_s, step)

    def apply_hebbian(graph, la):
        ns_ = graph.node_state
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                store = graph.edge_store(et)
                if store.n_edges == 0: continue
                src_out = ns_.output[store.src.long()]
                dst_out = ns_.output[store.dst.long()]
                dw = 0.001 * (src_out * dst_out - 0.0065 * 2.0 * store.weight - la * (src_out + dst_out) * store.weight)
                store.weight += dw
                store.weight.clamp_(0.0, 1.0)

    return (error_node_update, dual_channel_send, apply_hebbian,
            stp, hom, ip, exc_idx, input_nodes, pa, pb)


# ==============================================================
# CALIBRATION
# ==============================================================
def calibrate_ping():
    """Find PING parameters that produce endogenous gamma."""
    print("\n" + "=" * 60, flush=True)
    print("  CALIBRATION: Finding endogenous PING gamma", flush=True)
    print("=" * 60, flush=True)

    config = GraphBrainConfig.from_dict(CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device
    mp = TypedMessagePasser(config, N, device)

    helpers = make_phase1b_helpers(graph, mp)
    (node_update, send, hebbian, stp, hom, ip, exc_idx, input_nodes, pa, pb) = helpers

    best = {"coupling": 0.3, "pv_tau": 5.0, "boost": 3.0, "found": False, "freq": 0, "snr": 0}

    for coupling in [0.3, 0.5, 1.0]:
        for pv_tau in [5.0, 7.0]:
            for boost in [3.0, 5.0]:
                # Fresh graph for each config
                config = GraphBrainConfig.from_dict(CONFIG)
                g = NeuromorphicGraph(config)
                g.initialize()
                ns = g.node_state
                mp_cal = TypedMessagePasser(config, g.n_nodes, device)
                h = make_phase1b_helpers(g, mp_cal)
                (nu, snd, heb, st, ho, ip2, ei, inn, p_a, p_b) = h

                ping = PINGMechanism(g, coupling_strength=coupling, pv_tau=pv_tau,
                                     inhib_boost=boost)
                analyzer = OscillationAnalyzer()

                # Run 2000 steps with constant input
                for s in range(2000):
                    step = g.step_count
                    snd(g, step)
                    inputs = mp_cal.read_inputs(step)
                    inputs.basal[p_a.long()] += STRENGTH
                    nu(ns, inputs, ping=ping)
                    for et in EdgeType:
                        if g.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                            st.update(g.edge_store(et), ns, 1.0)
                    heb(g, LAMBDA_ACT)
                    g.increment_step()

                    if s >= 500:  # skip transient
                        pv_mean = ns.output[ping.pv_idx].mean().item()
                        analyzer.record(pv_mean, ns.output[ei].mean().item())

                result = analyzer.detect_gamma()
                found = "YES" if result["found"] else "no"
                print(f"  coupling={coupling} pv_tau={pv_tau} boost={boost}: "
                      f"gamma={found} freq={result['frequency']:.1f}Hz snr={result.get('snr_db',0):.1f}dB",
                      flush=True)

                if result["found"] and result.get("snr_db", 0) > best.get("snr", 0):
                    best = {"coupling": coupling, "pv_tau": pv_tau, "boost": boost,
                            "found": True, "freq": result["frequency"],
                            "snr": result.get("snr_db", 0)}

    print(f"\n  Best: coupling={best['coupling']} pv_tau={best['pv_tau']} boost={best['boost']} "
          f"freq={best['freq']:.1f}Hz snr={best['snr']:.1f}dB", flush=True)
    return best


# ==============================================================
# TEST C: Oscillation-PC Interaction (the key test)
# ==============================================================
def test_pc_interaction(ping_params):
    """5 conditions: 1B baseline, +PING, +Theta, +Both, 1A+Both."""
    print("\n" + "=" * 60, flush=True)
    print("  TEST C: Oscillation-PC Interaction", flush=True)
    print("=" * 60, flush=True)

    conditions = [
        ("1B Baseline", False, False, True),
        ("1B+PING", True, False, True),
        ("1B+Theta", False, True, True),
        ("1B+Both", True, True, True),
        ("1A+Both", True, True, False),  # Hand-built hierarchy
    ]

    results = {}
    n_cycles = 1000

    for cond_name, use_ping, use_theta, use_1b in conditions:
        print(f"\n  --- {cond_name} ({n_cycles} cycles) ---", flush=True)

        if use_1b:
            config = GraphBrainConfig.from_dict(CONFIG)
        else:
            cfg_1a = dict(CONFIG)
            cfg_1a["hierarchy"] = {"enabled": True, "error_ratio": 0.4, "pc_learning_rate": 0.1,
                                    "inter_level_p": 0.3, "inter_level_sigma": 0.5,
                                    "pattern_duration": 50, "input_strength": 2.0}
            config = GraphBrainConfig.from_dict(cfg_1a)

        graph = NeuromorphicGraph(config)
        graph.initialize()
        ns = graph.node_state
        N = graph.n_nodes
        device = graph.device
        mp = TypedMessagePasser(config, N, device)

        if not use_1b:
            HierarchyBuilder(config).build(graph)

        # Setup oscillations
        ping = PINGMechanism(graph, coupling_strength=ping_params["coupling"],
                             pv_tau=ping_params["pv_tau"],
                             inhib_boost=ping_params["boost"]) if use_ping else None
        theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5) if use_theta else None

        if use_1b:
            helpers = make_phase1b_helpers(graph, mp)
            (node_update, send, hebbian, stp_h, hom_h, ip_h, exc_idx, input_nodes, pa, pb) = helpers
        else:
            # 1A setup
            stp_h = ShortTermPlasticity(config.edges.stp)
            hom_h = HomeostaticScaling(config.edges.homeostatic)
            ip_h = IntrinsicPlasticity(config.nodes)
            pcw = PCWeightUpdate(config)
            nm_1a = PredictiveCodingModel(config)
            exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]
            l1_err = torch.where(ns.role_level_mask(NodeRole.ERROR, HierarchyLevel.LEVEL_1))[0]
            input_nodes = l1_err
            pa = l1_err[:l1_err.shape[0] // 2]
            pb = l1_err[l1_err.shape[0] // 2:]

        t0 = time.perf_counter()
        errors = []
        mismatch_history = []

        for cycle in range(n_cycles):
            err_sum, n = 0.0, 0
            for pat in [pa, pb]:
                for s in range(PD):
                    step = graph.step_count
                    theta_mod = theta.get_modulation(step) if theta else 1.0

                    if use_1b:
                        send(graph, step)
                        inputs = mp.read_inputs(step)
                        inputs.basal[pat.long()] += STRENGTH
                        node_update(ns, inputs, ping=ping, theta_mod=theta_mod)
                        for et in EdgeType:
                            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                                stp_h.update(graph.edge_store(et), ns, 1.0)
                        hebbian(graph, LAMBDA_ACT)
                    else:
                        mp.send_messages(graph, step)
                        inputs = mp.read_inputs(step)
                        inputs.basal[pat.long()] += STRENGTH
                        if theta_mod != 1.0:
                            exc_mask = ns.type_mask(NodeType.EXCITATORY)
                            inputs.basal = inputs.basal * (exc_mask.float() * theta_mod + (~exc_mask).float())
                        nm_1a.step(ns, inputs, float(step))
                        for et in EdgeType:
                            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                                stp_h.update(graph.edge_store(et), ns, 1.0)
                        if graph.has_edge_type(EdgeType.MODULATORY):
                            pcw.update(graph.edge_store(EdgeType.MODULATORY), ns)

                    if step % 100 == 0:
                        for et in EdgeType:
                            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                                hom_h.update(graph.edge_store(et), ns, 1.0)
                        ip_h.update(ns)
                    graph.increment_step()
                    err_sum += ns.output[input_nodes].mean().item()
                    n += 1
            errors.append(err_sum / n)

            if (cycle + 1) % 250 == 0:
                # Mismatch test
                for s in range(PD):
                    step = graph.step_count
                    if use_1b:
                        send(graph, step)
                    else:
                        mp.send_messages(graph, step)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pa.long()] += STRENGTH
                    if use_1b:
                        node_update(ns, inputs, ping=ping, theta_mod=theta.get_modulation(step) if theta else 1.0)
                    else:
                        nm_1a.step(ns, inputs, float(step))
                    graph.increment_step()
                bl = []
                for s in range(PD):
                    step = graph.step_count
                    if use_1b:
                        send(graph, step)
                    else:
                        mp.send_messages(graph, step)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pb.long()] += STRENGTH
                    if use_1b:
                        node_update(ns, inputs, ping=ping, theta_mod=theta.get_modulation(step) if theta else 1.0)
                    else:
                        nm_1a.step(ns, inputs, float(step))
                    graph.increment_step()
                    bl.append(ns.output[input_nodes].mean().item())
                for s in range(PD):
                    step = graph.step_count
                    if use_1b:
                        send(graph, step)
                    else:
                        mp.send_messages(graph, step)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pa.long()] += STRENGTH
                    if use_1b:
                        node_update(ns, inputs, ping=ping, theta_mod=theta.get_modulation(step) if theta else 1.0)
                    else:
                        nm_1a.step(ns, inputs, float(step))
                    graph.increment_step()
                vl = []
                for s in range(PD):
                    step = graph.step_count
                    if use_1b:
                        send(graph, step)
                    else:
                        mp.send_messages(graph, step)
                    inputs = mp.read_inputs(step)
                    inputs.basal[pa.long()] += STRENGTH
                    if use_1b:
                        node_update(ns, inputs, ping=ping, theta_mod=theta.get_modulation(step) if theta else 1.0)
                    else:
                        nm_1a.step(ns, inputs, float(step))
                    graph.increment_step()
                    vl.append(ns.output[input_nodes].mean().item())

                baseline = float(np.mean(bl))
                violation = float(np.mean(vl))
                ratio = violation / max(baseline, 1e-8)
                mismatch_history.append(ratio)
                sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0
                elapsed = time.perf_counter() - t0
                mm_str = f"{ratio:.3f}x" + (" **" if ratio > 1.1 else "")
                print(f"    Cycle {cycle+1:4d}: sup={sup:.1f}% mm={mm_str} "
                      f"ap={ns.apical[exc_idx].std().item():.4f} ({elapsed:.0f}s)", flush=True)

        best_mm = max(mismatch_history) if mismatch_history else 0
        final_mm = mismatch_history[-1] if mismatch_history else 0
        sup = (1 - errors[-1] / errors[0]) * 100 if errors[0] > 0 else 0

        results[cond_name] = {
            "suppression": sup, "best_mm": best_mm, "final_mm": final_mm,
            "apical_std": ns.apical[exc_idx].std().item(),
            "time": time.perf_counter() - t0,
        }
        print(f"    DONE: sup={sup:.1f}% best_mm={best_mm:.3f}x final_mm={final_mm:.3f}x", flush=True)

    # Summary
    print(f"\n  {'Condition':<15} | {'Sup%':>5} | {'BestMM':>7} | {'FinalMM':>8} | {'Apical':>7}", flush=True)
    print("  " + "-" * 55, flush=True)
    for name, r in results.items():
        star = " **" if r["best_mm"] > 1.1 else ""
        print(f"  {name:<15} | {r['suppression']:4.1f}% | {r['best_mm']:.3f}x | {r['final_mm']:.3f}x | "
              f"{r['apical_std']:.4f}{star}", flush=True)

    return results


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60, flush=True)
    print("  PHASE 2: PROPER OSCILLATORY DYNAMICS", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Calibrate PING
    ping_params = calibrate_ping()

    if not ping_params["found"]:
        print("\n  WARNING: No gamma peak found. Running tests with best params anyway.", flush=True)

    # Test C is the most important — run it first
    results_c = test_pc_interaction(ping_params)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("  PHASE 2 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  PING calibration: {'PASS' if ping_params['found'] else 'FAIL'} "
          f"({ping_params['freq']:.0f}Hz, {ping_params['snr']:.1f}dB)", flush=True)

    # PC interaction
    baseline = results_c.get("1B Baseline", {})
    for name, r in results_c.items():
        if name == "1B Baseline":
            continue
        vs_baseline = r["best_mm"] / max(baseline.get("best_mm", 1), 0.01) if baseline else 0
        print(f"  {name}: {r['best_mm']:.3f}x (vs baseline {baseline.get('best_mm', 0):.3f}x = "
              f"{vs_baseline:.0%})", flush=True)

    torch.save({
        "ping_params": ping_params,
        "test_c": results_c,
    }, "phase2_proper_results.pt")
    print(f"\nSaved to phase2_proper_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
