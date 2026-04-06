"""Phase 1B Validation: Three independent tests run in parallel.

Test 1: Functional mismatch — A-B-A-B → A-A on self-organised graph (no hierarchy builder)
Test 2: Reproducibility — 5 seeds with winning genome's lambda region
Test 3: Ablation — winning genome with lambda_activity reset to 0.01

All run in separate processes for parallelism.
"""

import sys
import time
from multiprocessing import Process, Queue

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

# Winning genome from evolution (generation 6-10)
WINNING_LAMBDAS = {
    "lambda_activity": 3.1,
    "lambda_weight": 0.0065,
    "lambda_edge": 0.00001,
    "lambda_prediction": 2.0,
    "lambda_reconstruction": 0.27,
    "lambda_mi": 2.3,
    "lambda_compartment": 0.1,
}

BASE_CONFIG = {
    "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70, "noise_std": 0.005},
    "edges": {"structural": {"enabled": True, "update_interval": 500, "growth_rate": 0.1,
                               "prune_threshold": 0.005, "edge_cost": 0.00001, "max_degree": 2000}},
    "simulation": {"device": "cuda", "seed": 42},
    "hierarchy": {"enabled": False},
}


def run_functional_mismatch(result_queue):
    """Test 1: A-B-A-B mismatch test on self-organised graph."""
    import torch
    from graph_brain.config import GraphBrainConfig
    from graph_brain.core.graph import NeuromorphicGraph
    from graph_brain.core.message_passing import TypedMessagePasser
    from graph_brain.edges.homeostatic import HomeostaticScaling
    from graph_brain.edges.short_term import ShortTermPlasticity
    from graph_brain.energy import EnergyGenome, TemporalHebbianState, apply_energy_gradient
    from graph_brain.nodes.intrinsic import IntrinsicPlasticity
    from graph_brain.nodes.model import TwoCompartmentModel
    from graph_brain.types import EdgeType, NodeType

    config = GraphBrainConfig.from_dict(BASE_CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    genome = EnergyGenome(**WINNING_LAMBDAS)
    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    ts = TemporalHebbianState(N, device)

    # Input nodes: bottom 20% excitatory by z
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    PD = 50
    STRENGTH = 2.0

    def run_step(step, pat_nodes):
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)
        inputs.basal[pat_nodes.long()] += STRENGTH
        nm.step(ns, inputs, float(step))
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

    # Phase 1: Train on A-B for 5000 cycles (self-organisation)
    print("[MISMATCH] Training 5000 A-B cycles...", flush=True)
    t0 = time.perf_counter()
    cycle_errors = []
    for cycle in range(5000):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                run_step(graph.step_count, pat)
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        cycle_errors.append(err_sum / n)
        if (cycle + 1) % 1000 == 0:
            print(f"[MISMATCH] Cycle {cycle+1}: err={cycle_errors[-1]:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Phase 2: Baseline A-B
    for s in range(PD):
        run_step(graph.step_count, pa)
    baseline = []
    for s in range(PD):
        run_step(graph.step_count, pb)
        baseline.append(ns.output[input_nodes].mean().item())

    # Phase 3: Violation A-A
    for s in range(PD):
        run_step(graph.step_count, pa)
    violation = []
    for s in range(PD):
        run_step(graph.step_count, pa)
        violation.append(ns.output[input_nodes].mean().item())

    bl = float(np.mean(baseline))
    vl = float(np.mean(violation))
    ratio = vl / max(bl, 1e-8)
    sup = (1 - cycle_errors[-1] / cycle_errors[0]) * 100

    print(f"[MISMATCH] DONE: sup={sup:.1f}% baseline={bl:.4f} violation={vl:.4f} ratio={ratio:.3f}x", flush=True)
    result_queue.put(("mismatch", {
        "suppression": sup, "baseline": bl, "violation": vl, "ratio": ratio,
        "error_start": cycle_errors[0], "error_end": cycle_errors[-1],
    }))


def run_reproducibility(seed, result_queue):
    """Test 2: Single seed of winning genome."""
    import torch
    from graph_brain.config import GraphBrainConfig
    from graph_brain.core.graph import NeuromorphicGraph
    from graph_brain.core.message_passing import TypedMessagePasser
    from graph_brain.edges.homeostatic import HomeostaticScaling
    from graph_brain.edges.short_term import ShortTermPlasticity
    from graph_brain.energy import EnergyGenome, TemporalHebbianState, apply_energy_gradient
    from graph_brain.nodes.intrinsic import IntrinsicPlasticity
    from graph_brain.nodes.model import TwoCompartmentModel
    from graph_brain.types import EdgeType, NodeType

    cfg = dict(BASE_CONFIG)
    cfg["simulation"] = dict(cfg["simulation"])
    cfg["simulation"]["seed"] = seed
    config = GraphBrainConfig.from_dict(cfg)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    N = graph.n_nodes
    device = graph.device

    genome = EnergyGenome(**WINNING_LAMBDAS)
    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    ts = TemporalHebbianState(N, device)

    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_idx = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_nodes = exc_idx[exc_z <= exc_z.quantile(0.2)]
    pa = input_nodes[:input_nodes.shape[0] // 2]
    pb = input_nodes[input_nodes.shape[0] // 2:]

    PD, STRENGTH = 50, 2.0
    t0 = time.perf_counter()

    # Train 3000 cycles
    cycle_errors = []
    for cycle in range(3000):
        err_sum, n = 0.0, 0
        for pat in [pa, pb]:
            for s in range(PD):
                step = graph.step_count
                mp.send_messages(graph, step)
                inputs = mp.read_inputs(step)
                inputs.basal[pat.long()] += STRENGTH
                nm.step(ns, inputs, float(step))
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
                err_sum += ns.output[input_nodes].mean().item()
                n += 1
        cycle_errors.append(err_sum / n)

    # Measure suppression metric (high apical → lower output?)
    exc_apical = ns.apical[exc_idx].abs()
    exc_output = ns.output[exc_idx]
    high_ap = exc_apical > exc_apical.median()
    low_ap = ~high_ap
    if high_ap.any() and low_ap.any():
        hi_out = float(exc_output[high_ap].mean())
        lo_out = float(exc_output[low_ap].mean())
        supp_ratio = max(0.0, (lo_out - hi_out) / (lo_out + 1e-6))
    else:
        supp_ratio = 0.0

    # Weight asymmetry
    from graph_brain.types import EdgeType as ET
    up_w, down_w, n_up, n_down = 0.0, 0.0, 0, 0
    for et in (ET.DRIVING, ET.MODULATORY):
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            src_z = ns.position[store.src.long(), 2]
            dst_z = ns.position[store.dst.long(), 2]
            up = dst_z > src_z
            dn = dst_z < src_z
            if up.any():
                up_w += float(store.weight[up].sum())
                n_up += int(up.sum())
            if dn.any():
                down_w += float(store.weight[dn].sum())
                n_down += int(dn.sum())
    um = up_w / max(n_up, 1)
    dm = down_w / max(n_down, 1)
    asym = abs(um - dm) / (max(um, dm) + 1e-6)

    elapsed = time.perf_counter() - t0
    sup_pct = (1 - cycle_errors[-1] / cycle_errors[0]) * 100
    print(f"[SEED {seed}] sup_ratio={supp_ratio:.4f} asym={asym:.4f} error_sup={sup_pct:.1f}% ({elapsed:.0f}s)", flush=True)

    result_queue.put(("seed", {
        "seed": seed, "suppression_ratio": supp_ratio, "asymmetry": asym,
        "error_suppression": sup_pct, "error_start": cycle_errors[0], "error_end": cycle_errors[-1],
    }))


def run_ablation(result_queue):
    """Test 3: Winning genome but lambda_activity = 0.01 (original default)."""
    import torch
    from graph_brain.config import GraphBrainConfig
    from graph_brain.energy import EnergyGenome
    from graph_brain.evolution import evaluate_individual

    ablated = dict(WINNING_LAMBDAS)
    ablated["lambda_activity"] = 0.01  # reset to original default

    config = GraphBrainConfig.from_dict(BASE_CONFIG)
    genome = EnergyGenome(**ablated)

    print("[ABLATION] Running with lambda_activity=0.01 (3000 steps)...", flush=True)
    t0 = time.perf_counter()
    result = evaluate_individual(genome, config, n_steps=3000)
    elapsed = time.perf_counter() - t0

    print(f"[ABLATION] DONE: supp={result.suppression_ratio:.4f} asym={result.weight_asymmetry:.4f} "
          f"diff={result.output_differentiation:.4f} ({elapsed:.0f}s)", flush=True)

    result_queue.put(("ablation", {
        "suppression_ratio": result.suppression_ratio,
        "asymmetry": result.weight_asymmetry,
        "differentiation": result.output_differentiation,
        "prediction_loss": result.prediction_loss,
    }))


def main():
    print("=" * 70, flush=True)
    print("  PHASE 1B VALIDATION — THREE INDEPENDENT TESTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Winning genome: lambda_activity={WINNING_LAMBDAS['lambda_activity']}", flush=True)

    queue = Queue()
    procs = []

    # Test 1: Functional mismatch
    p = Process(target=run_functional_mismatch, args=(queue,))
    p.start()
    procs.append(("MISMATCH", p))

    # Test 2: Reproducibility — 5 seeds
    for seed in [42, 123, 456, 789, 1337]:
        p = Process(target=run_reproducibility, args=(seed, queue))
        p.start()
        procs.append((f"SEED-{seed}", p))

    # Test 3: Ablation
    p = Process(target=run_ablation, args=(queue,))
    p.start()
    procs.append(("ABLATION", p))

    # Wait for all
    for name, p in procs:
        p.join()
        print(f"  {name} finished (exit={p.exitcode})", flush=True)

    # Collect results
    results = {"mismatch": None, "seeds": [], "ablation": None}
    while not queue.empty():
        test_type, data = queue.get()
        if test_type == "mismatch":
            results["mismatch"] = data
        elif test_type == "seed":
            results["seeds"].append(data)
        elif test_type == "ablation":
            results["ablation"] = data

    # Print summary
    print(f"\n{'=' * 70}", flush=True)
    print("  VALIDATION RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print("\n--- Test 1: Functional Mismatch (A-B-A-B -> A-A) ---", flush=True)
    if results["mismatch"]:
        m = results["mismatch"]
        print(f"  Error suppression: {m['suppression']:.1f}%", flush=True)
        print(f"  Baseline (expected B): {m['baseline']:.4f}", flush=True)
        print(f"  Violation (unexpected A): {m['violation']:.4f}", flush=True)
        print(f"  Violation/baseline: {m['ratio']:.3f}x", flush=True)
        if m['ratio'] > 1.1:
            print(f"  RESULT: MISMATCH DETECTED", flush=True)
        else:
            print(f"  RESULT: NO MISMATCH", flush=True)

    print("\n--- Test 2: Reproducibility (5 seeds) ---", flush=True)
    seeds_data = sorted(results["seeds"], key=lambda x: x["seed"])
    supp_ratios = []
    asymmetries = []
    for s in seeds_data:
        print(f"  Seed {s['seed']}: supp_ratio={s['suppression_ratio']:.4f} "
              f"asym={s['asymmetry']:.4f} error_sup={s['error_suppression']:.1f}%", flush=True)
        supp_ratios.append(s["suppression_ratio"])
        asymmetries.append(s["asymmetry"])
    if supp_ratios:
        print(f"  Mean suppression: {np.mean(supp_ratios):.4f} +/- {np.std(supp_ratios):.4f}", flush=True)
        print(f"  Mean asymmetry:   {np.mean(asymmetries):.4f} +/- {np.std(asymmetries):.4f}", flush=True)
        n_above = sum(1 for s in supp_ratios if s > 0.5)
        print(f"  Seeds with suppression > 0.5: {n_above}/5", flush=True)

    print("\n--- Test 3: Ablation (lambda_activity = 0.01) ---", flush=True)
    if results["ablation"]:
        a = results["ablation"]
        print(f"  Suppression ratio: {a['suppression_ratio']:.4f}", flush=True)
        print(f"  Weight asymmetry:  {a['asymmetry']:.4f}", flush=True)
        print(f"  Differentiation:   {a['differentiation']:.4f}", flush=True)
        if results["seeds"]:
            mean_sup = np.mean(supp_ratios)
            print(f"  vs winning genome mean: {mean_sup:.4f}", flush=True)
            if a['suppression_ratio'] < mean_sup * 0.5:
                print(f"  RESULT: ABLATION CONFIRMS — sparsity is causal", flush=True)
            else:
                print(f"  RESULT: ABLATION INCONCLUSIVE", flush=True)

    # Overall verdict
    print(f"\n{'=' * 70}", flush=True)
    mismatch_pass = results["mismatch"] and results["mismatch"]["ratio"] > 1.1
    repro_pass = len([s for s in supp_ratios if s > 0.3]) >= 3 if supp_ratios else False
    ablation_pass = results["ablation"] and results["ablation"]["suppression_ratio"] < 0.3 if results["ablation"] else False

    print(f"  Functional mismatch:  {'PASS' if mismatch_pass else 'FAIL'}", flush=True)
    print(f"  Reproducibility:      {'PASS' if repro_pass else 'FAIL'} ({len([s for s in supp_ratios if s > 0.3])}/5 seeds)", flush=True)
    print(f"  Ablation:             {'PASS' if ablation_pass else 'FAIL'}", flush=True)
    print(f"  OVERALL:              {'VALIDATED' if (mismatch_pass and repro_pass and ablation_pass) else 'NEEDS WORK'}", flush=True)
    print(f"{'=' * 70}", flush=True)

    import torch
    torch.save(results, "validation_1b_results.pt")
    print("Saved to validation_1b_results.pt", flush=True)


if __name__ == "__main__":
    main()
