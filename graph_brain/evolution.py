"""Evolutionary search for optimal energy functional parameters.

Population of graphs, each with a different EnergyGenome (lambda values).
Each individual is evaluated by running for K steps and measuring fitness.
Selection, crossover, mutation produce the next generation.

Fitness = multi-objective: prediction quality + reconstruction quality
          + information content - energy cost.

The lambda values that produce the best self-organisation are discovered
by evolution, not hand-tuned.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph, NodeState
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.energy import EnergyFunctional, EnergyGenome, TemporalHebbianState, apply_energy_gradient
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.nodes.model import TwoCompartmentModel
from graph_brain.types import EdgeType, NodeType


@dataclass
class FitnessResult:
    """Fitness evaluation of one individual."""
    genome: EnergyGenome
    prediction_loss: float
    reconstruction_loss: float
    mi_proxy: float
    activity: float
    edge_count: int
    # Emergent structure metrics
    output_differentiation: float  # std of mean output across spatial regions
    weight_asymmetry: float        # asymmetry between upward vs downward edges
    suppression_ratio: float       # how much apical suppresses output (PC signature)
    fitness: float = 0.0

    def compute_fitness(self) -> float:
        """Multi-objective fitness. Higher = better."""
        # Good prediction (low loss = high fitness)
        pred_score = 1.0 / (self.prediction_loss + 0.1)
        # Good reconstruction
        recon_score = 1.0 / (self.reconstruction_loss + 0.1)
        # High information content
        info_score = 1.0 / (self.mi_proxy + 0.1)
        # Low energy
        energy_score = 1.0 / (self.activity + 0.01)
        # Emergent structure bonuses
        diff_bonus = self.output_differentiation  # more differentiation = more structure
        asym_bonus = self.weight_asymmetry  # weight asymmetry suggests hierarchy
        supp_bonus = self.suppression_ratio  # apical suppression = PC signature

        self.fitness = (
            pred_score * 2.0
            + recon_score * 1.0
            + info_score * 1.0
            + energy_score * 0.5
            + diff_bonus * 3.0
            + asym_bonus * 2.0
            + supp_bonus * 5.0  # heavily reward PC-like suppression
        )
        return self.fitness


def evaluate_individual(
    genome: EnergyGenome,
    base_config: GraphBrainConfig,
    n_steps: int = 2000,
    pattern_duration: int = 50,
    input_strength: float = 2.0,
) -> FitnessResult:
    """Evaluate one genome by running a graph for n_steps.

    Presents alternating A-B patterns (same as Phase 1A test)
    and measures how well the graph self-organises.
    """
    config = base_config
    graph = NeuromorphicGraph(config)
    graph.initialize()

    ns = graph.node_state
    device = graph.device
    N = graph.n_nodes

    # Components (no hierarchy — this is self-organisation)
    mp = TypedMessagePasser(config, N, device)
    nm = TwoCompartmentModel(config.nodes)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    energy_fn = EnergyFunctional(genome)
    temporal_state = TemporalHebbianState(N, device)

    # Input nodes: bottom 20% of excitatory nodes (by z position)
    exc_mask = ns.type_mask(NodeType.EXCITATORY)
    exc_indices = torch.where(exc_mask)[0]
    exc_z = ns.position[exc_indices, 2]
    z_threshold = exc_z.quantile(0.2)
    input_mask = exc_z <= z_threshold
    input_nodes = exc_indices[input_mask]

    # Define patterns A and B
    n_input = input_nodes.shape[0]
    pattern_a_nodes = input_nodes[:n_input // 2]
    pattern_b_nodes = input_nodes[n_input // 2:]

    # Run simulation with energy-driven updates
    pattern_idx = 0  # 0 = A, 1 = B
    step_in_pattern = 0

    energy_history = []

    for step in range(n_steps):
        # Determine current pattern
        if step_in_pattern >= pattern_duration:
            pattern_idx = 1 - pattern_idx
            step_in_pattern = 0

        current_pat = pattern_a_nodes if pattern_idx == 0 else pattern_b_nodes
        next_pat = pattern_b_nodes if pattern_idx == 0 else pattern_a_nodes

        # Message passing
        mp.send_messages(graph, step)
        inputs = mp.read_inputs(step)

        # Inject input
        input_vals = torch.full((current_pat.shape[0],), input_strength, device=device)
        inputs.basal[current_pat.long()] += input_vals

        # Node update
        nm.step(ns, inputs, float(step))

        # STP
        for et in EdgeType:
            if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                stp.update(graph.edge_store(et), ns, config.nodes.dt)

        # Energy-driven weight update (every step)
        apply_energy_gradient(graph, genome, temporal_state, config.nodes.dt)

        # Homeostatic + intrinsic (every 100 steps)
        if step % 100 == 0:
            for et in EdgeType:
                if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
                    hom.update(graph.edge_store(et), ns, config.nodes.dt)
            ip.update(ns)

        # Structural plasticity via energy (every 500 steps)
        if step > 0 and step % 500 == 0 and config.edges.structural.enabled:
            from graph_brain.edges.structural import StructuralPlasticity
            sp = StructuralPlasticity(config)
            sp.update(graph)

        # Record energy periodically
        if step % 200 == 0:
            target_vals = torch.full((next_pat.shape[0],), input_strength, device=device)
            target_full = torch.zeros(current_pat.shape[0], device=device)
            e = energy_fn.compute(graph, input_vals, target_full, current_pat)
            energy_history.append(e)

        graph.increment_step()
        step_in_pattern += 1

    # --- Measure emergent structure ---

    # Output differentiation: do different spatial regions have different activity?
    exc_output = ns.output[exc_indices]
    exc_positions = ns.position[exc_indices, 2]  # z-axis
    n_regions = 5
    region_means = []
    for i in range(n_regions):
        low = i / n_regions
        high = (i + 1) / n_regions
        region_mask = (exc_positions >= low) & (exc_positions < high)
        if region_mask.any():
            region_means.append(float(exc_output[region_mask].mean()))
    output_differentiation = float(np.std(region_means)) if len(region_means) > 1 else 0.0

    # Weight asymmetry: are upward edges (low z → high z) different from downward?
    upward_weight_sum = 0.0
    downward_weight_sum = 0.0
    n_up, n_down = 0, 0
    for et in (EdgeType.DRIVING, EdgeType.MODULATORY):
        if graph.has_edge_type(et):
            store = graph.edge_store(et)
            src_z = ns.position[store.src.long(), 2]
            dst_z = ns.position[store.dst.long(), 2]
            up_mask = dst_z > src_z
            down_mask = dst_z < src_z
            if up_mask.any():
                upward_weight_sum += float(store.weight[up_mask].sum())
                n_up += int(up_mask.sum())
            if down_mask.any():
                downward_weight_sum += float(store.weight[down_mask].sum())
                n_down += int(down_mask.sum())

    up_mean = upward_weight_sum / max(n_up, 1)
    down_mean = downward_weight_sum / max(n_down, 1)
    weight_asymmetry = abs(up_mean - down_mean) / (max(up_mean, down_mean) + 1e-6)

    # Suppression ratio: how much does apical input reduce output?
    # Compare nodes with high apical to nodes with low apical
    exc_apical = ns.apical[exc_indices].abs()
    high_apical = exc_apical > exc_apical.median()
    low_apical = ~high_apical
    if high_apical.any() and low_apical.any():
        high_ap_output = float(exc_output[high_apical].mean())
        low_ap_output = float(exc_output[low_apical].mean())
        suppression_ratio = max(0.0, (low_ap_output - high_ap_output) / (low_ap_output + 1e-6))
    else:
        suppression_ratio = 0.0

    # Final energy
    last_energy = energy_history[-1] if energy_history else {"prediction": 10.0, "reconstruction": 10.0, "mi_proxy": 10.0, "activity": 10.0, "edge_count": 0}

    result = FitnessResult(
        genome=genome,
        prediction_loss=last_energy["prediction"],
        reconstruction_loss=last_energy["reconstruction"],
        mi_proxy=last_energy["mi_proxy"],
        activity=last_energy["activity"],
        edge_count=last_energy["edge_count"],
        output_differentiation=output_differentiation,
        weight_asymmetry=weight_asymmetry,
        suppression_ratio=suppression_ratio,
    )
    result.compute_fitness()
    return result


class EvolutionarySearch:
    """Evolutionary search for optimal energy genome."""

    def __init__(
        self,
        base_config: GraphBrainConfig,
        population_size: int = 20,
        n_steps_per_eval: int = 2000,
        seed: int = 42,
    ):
        self.base_config = base_config
        self.pop_size = population_size
        self.n_steps = n_steps_per_eval
        self.rng = random.Random(seed)

        # Initialize population with diverse genomes
        self.population: list[EnergyGenome] = []
        for _ in range(population_size):
            genome = EnergyGenome(
                lambda_activity=10 ** self.rng.uniform(-4, 0),
                lambda_weight=10 ** self.rng.uniform(-5, -1),
                lambda_edge=10 ** self.rng.uniform(-6, -2),
                lambda_prediction=10 ** self.rng.uniform(-1, 1),
                lambda_reconstruction=10 ** self.rng.uniform(-1, 1),
                lambda_mi=10 ** self.rng.uniform(-1, 1),
            )
            self.population.append(genome)

    def run_generation(self, generation: int) -> list[FitnessResult]:
        """Evaluate all individuals in the current population."""
        results = []
        for i, genome in enumerate(self.population):
            result = evaluate_individual(genome, self.base_config, self.n_steps)
            results.append(result)
            print(f"  Gen {generation} [{i+1}/{self.pop_size}]: "
                  f"fit={result.fitness:.3f} pred={result.prediction_loss:.3f} "
                  f"diff={result.output_differentiation:.4f} "
                  f"asym={result.weight_asymmetry:.4f} "
                  f"supp={result.suppression_ratio:.4f} | {genome}", flush=True)
        return results

    def select_and_breed(self, results: list[FitnessResult]) -> list[EnergyGenome]:
        """Tournament selection + crossover + mutation."""
        # Sort by fitness
        results.sort(key=lambda r: r.fitness, reverse=True)

        # Keep top 25% as elites
        n_elite = max(2, self.pop_size // 4)
        elites = [r.genome for r in results[:n_elite]]

        # Fill rest with crossover + mutation of top 50%
        top_half = [r.genome for r in results[:self.pop_size // 2]]
        new_pop = list(elites)  # elites pass through unchanged

        while len(new_pop) < self.pop_size:
            p1 = self.rng.choice(top_half)
            p2 = self.rng.choice(top_half)
            child = p1.crossover(p2, self.rng)
            child = child.mutate(mutation_rate=0.4, rng=self.rng)
            new_pop.append(child)

        return new_pop

    def run(self, n_generations: int = 10) -> list[list[FitnessResult]]:
        """Run the full evolutionary search."""
        all_results = []

        for gen in range(n_generations):
            print(f"\n{'='*70}", flush=True)
            print(f"  GENERATION {gen + 1}/{n_generations}", flush=True)
            print(f"{'='*70}", flush=True)

            t0 = time.perf_counter()
            results = self.run_generation(gen + 1)
            elapsed = time.perf_counter() - t0
            all_results.append(results)

            # Stats
            fitnesses = [r.fitness for r in results]
            best = max(results, key=lambda r: r.fitness)
            print(f"\n  Gen {gen+1} summary: best={max(fitnesses):.3f} "
                  f"mean={np.mean(fitnesses):.3f} std={np.std(fitnesses):.3f} ({elapsed:.0f}s)", flush=True)
            print(f"  Best genome: {best.genome}", flush=True)
            print(f"  Best metrics: pred={best.prediction_loss:.3f} "
                  f"diff={best.output_differentiation:.4f} "
                  f"asym={best.weight_asymmetry:.4f} "
                  f"supp={best.suppression_ratio:.4f}", flush=True)

            # Breed next generation (except last)
            if gen < n_generations - 1:
                self.population = self.select_and_breed(results)

        return all_results
