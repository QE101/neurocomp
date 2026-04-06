"""Phase 1B: Universal error model — silence requires prediction.

THE KEY CHANGE: output = f(|basal - apical|) for ALL excitatory nodes.

With external input forcing basal non-zero, the only way to reduce
output (save energy) is to match apical to basal. Matching apical
to basal IS prediction. Silence without prediction is impossible.

Same energy functional, same Hebbian rule, same graph.
Different node model. One change.

5000 cycles, unbuffered, comprehensive metrics.
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
from graph_brain.core.message_passing import TypedMessagePasser, CompartmentInputs
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.energy import TemporalState
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.types import EdgeType, NodeType

LAMBDA_ACT = 3.1


def error_node_update(ns, inputs, dt, noise_std=0.005):
    """Universal error model with dual-channel output.

    Every excitatory node computes TWO signals:
      - output (error): f(|basal - apical|) — used by DRIVING edges (feedforward)
      - basal value (content): the raw evidence — used by MODULATORY edges (feedback)

    output goes to graph.node_state.output (read by driving edges in message passing)
    basal content goes to graph.node_state.apical... NO — we need a separate channel.

    SIMPLER: we store the content signal in node_state.gain temporarily and
    have the message passing read from the right field per edge type.
    Actually simplest: override in the message passing step below.

    For now: output = f(|basal - apical|). The dual-channel routing happens
    in the custom message passing function, not here.
    """
    device = ns.device
    N = ns.n_nodes

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    pv_mask = ns.type_mask(NodeType.PV)
    sst_mask = ns.type_mask(NodeType.SST)
    vip_mask = ns.type_mask(NodeType.VIP)

    # === EXCITATORY: universal error model ===
    exc_f = exc_mask.float()

    # Basal: leaky integration of driving input (evidence)
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * exc_f

    # Apical: leaky integration of modulatory input (prediction)
    sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
    ns.apical += dt * (-ns.apical / 20.0 + inputs.apical * (1.0 - sst_gate)) * exc_f

    # OUTPUT = f(|basal - apical|) — the error signal
    prediction_error = ns.basal - ns.apical
    pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
    exc_output = F.softplus(prediction_error.abs()) * pv_gain * ns.gain

    ns.output = torch.where(exc_mask, exc_output, ns.output)

    # Store prediction error for diagnostics
    ns.prediction_error = torch.where(exc_mask, prediction_error, ns.prediction_error)

    # === PV interneurons ===
    pv_f = pv_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal + inputs.electrical) * pv_f
    pv_out = F.softplus(ns.basal) * ns.gain * pv_f
    ns.output = torch.where(pv_mask, pv_out, ns.output)

    # === SST interneurons ===
    sst_f = sst_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * sst_f
    vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
    sst_out = F.softplus(ns.basal) * ns.gain * vip_inhib * sst_f
    ns.output = torch.where(sst_mask, sst_out, ns.output)

    # === VIP interneurons ===
    vip_f = vip_mask.float()
    ns.basal += dt * (-ns.basal / 10.0 + inputs.basal) * vip_f
    vip_out = F.softplus(ns.basal) * ns.gain * vip_f
    ns.output = torch.where(vip_mask, vip_out, ns.output)

    # Noise + clamp
    ns.output += torch.randn(N, device=device) * noise_std
    ns.output.clamp_(min=0.0)

    # Activity EMA
    ns.activity_ema.lerp_(ns.output, dt / 1000.0)


def apply_hebbian(graph, la):
    """Standard simultaneous Hebbian + energy penalty."""
    ns = graph.node_state
    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue
            src_out = ns.output[store.src.long()]
            dst_out = ns.output[store.dst.long()]
            hebbian = src_out * dst_out
            weight_decay = 0.0065 * 2.0 * store.weight
            activity_penalty = la * (src_out + dst_out) * store.weight
            dw = 0.001 * (hebbian - weight_decay - activity_penalty)
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)


def main():
    print("=" * 70, flush=True)
    print("  PHASE 1B: UNIVERSAL ERROR MODEL", flush=True)
    print("  output = f(|basal - apical|) — silence requires prediction", flush=True)
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
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]
    PD, STRENGTH = 50, 2.0
    N_CYCLES = 5000
    CHECKPOINT = 250

    print(f"\nGraph: {N} nodes, {graph.n_edges()} edges", flush=True)
    print(f"Node model: output = softplus(|basal - apical|) * pv_gain", flush=True)
    print(f"Hebbian + lambda_activity={LAMBDA_ACT}", flush=True)
    print(f"Key insight: silence is impossible without matching apical to basal\n", flush=True)

    def dual_channel_send(graph, step):
        """Message passing with dual-channel output.

        Driving edges: transmit output (= error signal = |B-A|)
        Modulatory edges: transmit basal (= content signal = what I'm seeing)
        Other edges: transmit output (standard)

        This routes the RIGHT signal through the RIGHT pathway without
        role assignment. Every node has both signals. Edge type determines
        which one flows.
        """
        from graph_brain.core.delay_buffer import Channel

        output = ns.output          # error signal: |B-A|
        content = F.softplus(ns.basal)  # content signal: what the node is seeing

        for edge_type in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC, EdgeType.RETROGRADE):
            if not graph.has_edge_type(edge_type):
                continue
            store = graph.edge_store(edge_type)
            # Driving edges transmit ERROR (output = |B-A|)
            src_signal = output[store.src.long()]
            msg = src_signal * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.DRIVING: Channel.BASAL,
                       EdgeType.INHIB_PERISOMATIC: Channel.PV_INHIBITION,
                       EdgeType.RETROGRADE: Channel.RETROGRADE}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)

        for edge_type in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
            if not graph.has_edge_type(edge_type):
                continue
            store = graph.edge_store(edge_type)
            # Modulatory edges transmit CONTENT (basal value, not error)
            src_signal = content[store.src.long()]
            msg = src_signal * store.release_prob * store.weight
            delay_steps = (store.delay / 1.0).ceil().long().clamp(1, mp.max_delay_steps)
            channel = {EdgeType.MODULATORY: Channel.APICAL,
                       EdgeType.INHIB_DENDRITIC: Channel.SST_INHIBITION}[edge_type]
            mp.delay_buffer.write(channel, store.dst, msg, delay_steps, step)

        # Electrical (gap junctions)
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            store = graph.edge_store(EdgeType.ELECTRICAL)
            src_out = output[store.src.long()]
            dst_out = output[store.dst.long()]
            gap_current = store.weight * (src_out - dst_out)
            delay_steps = torch.ones(store.n_edges, dtype=torch.long, device=device)
            mp.delay_buffer.write(Channel.ELECTRICAL, store.dst, gap_current, delay_steps, step)

    def run_step(pat):
        step = graph.step_count
        dual_channel_send(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat.long()] += STRENGTH
        error_node_update(ns, inputs, 1.0)
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

    def run_mismatch():
        for s in range(PD):
            run_step(pa)
        bl = []
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pb.long()] += STRENGTH
            error_node_update(ns, inputs, 1.0)
            graph.increment_step()
            bl.append(ns.output[input_nodes].mean().item())
        for s in range(PD):
            run_step(pa)
        vl = []
        for s in range(PD):
            step = graph.step_count
            dual_channel_send(graph, step)
            inputs = mp.read_inputs(step)
            inputs.basal[pa.long()] += STRENGTH
            error_node_update(ns, inputs, 1.0)
            graph.increment_step()
            vl.append(ns.output[input_nodes].mean().item())
        return float(np.mean(bl)), float(np.mean(vl))

    print(f"{'Cyc':>5} | {'Err':>8} | {'Sup%':>5} | {'Ap_std':>7} | {'Ba_std':>7} | "
          f"{'|B-A|':>7} | {'Asym':>6} | {'MM':>8} | {'Time':>5}", flush=True)
    print("-" * 80, flush=True)

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
            ba_std = ns.basal[exc_idx].std().item()
            ba_ap_diff = (ns.basal[exc_idx] - ns.apical[exc_idx]).abs().mean().item()

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

            bl, vl = run_mismatch()
            ratio = vl / max(bl, 1e-8)
            all_mismatch.append(ratio)
            mm_str = f"{ratio:.3f}x"
            if ratio > 1.1:
                mm_str += " **"

            print(f"{cycle+1:5d} | {all_errors[-1]:8.4f} | {sup:4.1f}% | {ap_std:7.4f} | {ba_std:7.4f} | "
                  f"{ba_ap_diff:7.4f} | {asym:6.4f} | {mm_str:>8} | {elapsed:5.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"\n{'='*70}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Time: {total:.0f}s ({total/3600:.1f}h)", flush=True)
    print(f"Error: {all_errors[0]:.4f} -> {all_errors[-1]:.4f}", flush=True)
    print(f"Suppression: {(1-all_errors[-1]/all_errors[0])*100:.1f}%", flush=True)
    print(f"|basal - apical|: {(ns.basal[exc_idx] - ns.apical[exc_idx]).abs().mean().item():.4f}", flush=True)
    print(f"Apical std: {ns.apical[exc_idx].std().item():.4f}", flush=True)
    print(f"Best mismatch: {max(all_mismatch):.3f}x", flush=True)
    print(f"Final mismatch: {all_mismatch[-1]:.3f}x", flush=True)

    if max(all_mismatch) > 1.1:
        best_idx = all_mismatch.index(max(all_mismatch))
        print(f"MISMATCH DETECTED at cycle {(best_idx+1)*CHECKPOINT}", flush=True)
    else:
        print("NO MISMATCH DETECTED", flush=True)

    torch.save({
        "errors": all_errors, "mismatch_ratios": all_mismatch,
    }, "error_model_emergence_results.pt")
    print(f"\nSaved to error_model_emergence_results.pt", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
