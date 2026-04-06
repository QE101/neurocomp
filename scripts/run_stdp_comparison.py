"""Phase 1A.2: STDP-PC Interaction Experiment.

Four conditions, same graph seed, same patterns, 3000 A-B cycles each:
  1. STDP on driving + PC-native on modulatory (current setup)
  2. Fixed driving weights + PC-native on modulatory (STDP off)
  3. Three-factor STDP on driving + PC-native on modulatory
  4. PC-native on BOTH driving and modulatory (no STDP anywhere)

Compare: suppression %, mismatch ratio, convergence speed.
"""

import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

sys.path.insert(0, ".")

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.dynamics.recorder import StateRecorder
from graph_brain.dynamics.simulator import Simulator
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.edges.stdp import STDP, ThreeFactorSTDP
from graph_brain.edges.structural import StructuralPlasticity
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.predictive_coding import PredictiveCodingModel, PCWeightUpdate
from graph_brain.types import EdgeType, HierarchyLevel, NodeRole
from graph_brain.utils.profiling import StepTimer


BASE_CONFIG = {
    "nodes": {"n_excitatory": 2000, "n_pv": 175, "n_sst": 175, "n_vip": 150, "noise_std": 0.005},
    "edges": {"structural": {"enabled": False}},
    "simulation": {"device": "cuda", "seed": 42, "record_interval": 50},
    "hierarchy": {"enabled": True, "error_ratio": 0.4, "pc_learning_rate": 0.1,
                  "inter_level_p": 0.3, "inter_level_sigma": 0.5,
                  "pattern_duration": 50, "input_strength": 2.0},
}

N_CYCLES = 3000
PD = 50  # pattern duration
STRENGTH = 2.0


class CustomSimulator:
    """Simulator with configurable driving edge learning rule."""

    def __init__(self, graph, config, driving_rule="stdp"):
        self.graph = graph
        self.config = config
        cfg = config

        self.message_passer = TypedMessagePasser(cfg, graph.n_nodes, graph.device)
        self.node_model = PredictiveCodingModel(cfg)
        self.pc_weight_update = PCWeightUpdate(cfg)
        self.homeostatic = HomeostaticScaling(cfg.edges.homeostatic)
        self.stp = ShortTermPlasticity(cfg.edges.stp)
        self.intrinsic = IntrinsicPlasticity(cfg.nodes)
        self.timer = StepTimer()

        # Configure driving edge rule
        self.driving_rule = driving_rule
        if driving_rule == "stdp":
            self.driving_updater = STDP(cfg.edges.stdp)
        elif driving_rule == "three_factor":
            self.driving_updater = ThreeFactorSTDP(cfg.edges.stdp)
        elif driving_rule == "pc_native":
            self.driving_updater = PCWeightUpdate(cfg)  # PC-native on driving too
        elif driving_rule == "fixed":
            self.driving_updater = None  # no updates
        else:
            raise ValueError(f"Unknown driving rule: {driving_rule}")

        self._ext_input = None
        self._ext_indices = None

    def step(self):
        graph = self.graph
        dt = self.config.nodes.dt
        step = graph.step_count

        # Message passing
        self.message_passer.send_messages(graph, step)
        inputs = self.message_passer.read_inputs(step)

        if self._ext_input is not None:
            inputs.basal[self._ext_indices.long()] += self._ext_input
            self._ext_input = None
            self._ext_indices = None

        # Node model
        self.node_model.step(graph.node_state, inputs, step * dt)

        # STP
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                self.stp.update(graph.edge_store(et), graph.node_state, dt)

        # Driving edge learning rule
        if self.driving_updater is not None and graph.has_edge_type(EdgeType.DRIVING):
            if self.driving_rule in ("stdp", "three_factor"):
                self.driving_updater.update(graph.edge_store(EdgeType.DRIVING), graph.node_state, dt)
            elif self.driving_rule == "pc_native":
                self.driving_updater.update(graph.edge_store(EdgeType.DRIVING), graph.node_state)

        # PC weight update on modulatory (always)
        if graph.has_edge_type(EdgeType.MODULATORY):
            self.pc_weight_update.update(graph.edge_store(EdgeType.MODULATORY), graph.node_state)

        # Homeostatic + intrinsic (every 100 steps)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    self.homeostatic.update(graph.edge_store(et), graph.node_state, dt)
            self.intrinsic.update(graph.node_state)

        graph.increment_step()

    def inject_input(self, indices, values):
        self._ext_indices = indices.to(self.graph.device)
        self._ext_input = values.to(self.graph.device)


def run_condition(name, driving_rule):
    """Run one experimental condition. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"  CONDITION: {name} (driving_rule={driving_rule})")
    print(f"{'='*60}")

    config = GraphBrainConfig.from_dict(BASE_CONFIG)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    builder = HierarchyBuilder(config)
    builder.build(graph)

    ns = graph.node_state
    l1_error = torch.where(ns.role_level_mask(NodeRole.ERROR, HierarchyLevel.LEVEL_1))[0]
    pattern_a = l1_error[:l1_error.shape[0] // 2]
    pattern_b = l1_error[l1_error.shape[0] // 2:]

    sim = CustomSimulator(graph, config, driving_rule=driving_rule)
    t0 = time.perf_counter()

    cycle_errors = []
    for cycle in range(N_CYCLES):
        err_sum, n = 0.0, 0
        for pat in [pattern_a, pattern_b]:
            for s in range(PD):
                vals = torch.full((pat.shape[0],), STRENGTH, device="cuda")
                sim.inject_input(pat, vals)
                sim.step()
                err_sum += ns.output[l1_error].mean().item()
                n += 1
        cycle_errors.append(err_sum / n)

        if (cycle + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            sup = (1 - cycle_errors[-1] / cycle_errors[0]) * 100
            print(f"  Cycle {cycle+1:4d}: err={cycle_errors[-1]:.4f} sup={sup:.1f}% ({elapsed:.0f}s)")

    # Baseline + violation
    for s in range(PD):
        vals = torch.full((pattern_a.shape[0],), STRENGTH, device="cuda")
        sim.inject_input(pattern_a, vals)
        sim.step()
    baseline = []
    for s in range(PD):
        vals = torch.full((pattern_b.shape[0],), STRENGTH, device="cuda")
        sim.inject_input(pattern_b, vals)
        sim.step()
        baseline.append(ns.output[l1_error].mean().item())
    for s in range(PD):
        vals = torch.full((pattern_a.shape[0],), STRENGTH, device="cuda")
        sim.inject_input(pattern_a, vals)
        sim.step()
    violation = []
    for s in range(PD):
        vals = torch.full((pattern_a.shape[0],), STRENGTH, device="cuda")
        sim.inject_input(pattern_a, vals)
        sim.step()
        violation.append(ns.output[l1_error].mean().item())

    bl = np.mean(baseline)
    vl = np.mean(violation)
    total = time.perf_counter() - t0

    sup_final = (1 - cycle_errors[-1] / cycle_errors[0]) * 100
    ratio = vl / max(bl, 1e-8)

    print(f"\n  Result: sup={sup_final:.1f}% mismatch={ratio:.3f}x time={total:.0f}s")

    return {
        "name": name,
        "driving_rule": driving_rule,
        "cycle_errors": cycle_errors,
        "baseline": bl,
        "violation": vl,
        "ratio": ratio,
        "suppression": sup_final,
        "time": total,
        "error_first": cycle_errors[0],
        "error_500": cycle_errors[499],
        "error_1000": cycle_errors[999],
        "error_3000": cycle_errors[-1],
    }


def main():
    print("Phase 1A.2: STDP-PC Interaction Experiment")
    print(f"{N_CYCLES} A-B cycles per condition, 4 conditions\n")

    conditions = [
        ("1. STDP + PC-native (current)", "stdp"),
        ("2. Fixed driving + PC-native", "fixed"),
        ("3. Three-factor STDP + PC-native", "three_factor"),
        ("4. PC-native everywhere", "pc_native"),
    ]

    results = []
    for name, rule in conditions:
        r = run_condition(name, rule)
        results.append(r)

    # Summary table
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Condition':<40} {'Sup%':>6} {'Mismatch':>9} {'Err@500':>8} {'Err@3000':>9} {'Time':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<40} {r['suppression']:>5.1f}% {r['ratio']:>8.3f}x {r['error_500']:>8.3f} {r['error_3000']:>8.3f} {r['time']:>5.0f}s")

    # Winner
    best = max(results, key=lambda r: r["suppression"])
    print(f"\nWinner (suppression): {best['name']} at {best['suppression']:.1f}%")
    best_mm = max(results, key=lambda r: r["ratio"])
    print(f"Winner (mismatch):    {best_mm['name']} at {best_mm['ratio']:.3f}x")

    torch.save(results, "stdp_comparison_results.pt")
    print("\nSaved to stdp_comparison_results.pt")


if __name__ == "__main__":
    main()
