"""Phase 1B: Evolutionary search for energy functional parameters.

Runs generations of graphs with different energy genomes in parallel.
Discovers the lambda values that produce self-organising predictive coding.
"""

import sys
import time
from multiprocessing import Process, Queue

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

from graph_brain.config import GraphBrainConfig
from graph_brain.energy import EnergyGenome
from graph_brain.evolution import evaluate_individual, FitnessResult

BASE_CONFIG = GraphBrainConfig.from_dict({
    "nodes": {"n_excitatory": 1000, "n_pv": 90, "n_sst": 90, "n_vip": 70,
              "noise_std": 0.005},
    "edges": {"structural": {"enabled": True, "update_interval": 500,
                               "growth_rate": 0.1, "prune_threshold": 0.005,
                               "edge_cost": 0.00001, "max_degree": 2000}},
    "simulation": {"device": "cuda", "seed": 42},
    "hierarchy": {"enabled": False},  # NO hierarchy — self-organisation only
})

POP_SIZE = 16
N_GENERATIONS = 10
N_STEPS = 3000
N_PARALLEL = 4  # processes at a time


def eval_worker(genome_dict, config_dict, n_steps, result_queue, idx):
    """Worker process for evaluating one genome."""
    from graph_brain.config import GraphBrainConfig
    from graph_brain.energy import EnergyGenome
    from graph_brain.evolution import evaluate_individual

    genome = EnergyGenome(**genome_dict)
    config = GraphBrainConfig.from_dict(config_dict)

    result = evaluate_individual(genome, config, n_steps)
    result_queue.put((idx, result))


def evaluate_population_parallel(genomes, config, n_steps, n_parallel=4):
    """Evaluate a population of genomes in parallel batches."""
    config_dict = config.model_dump()
    results = [None] * len(genomes)

    for batch_start in range(0, len(genomes), n_parallel):
        batch_end = min(batch_start + n_parallel, len(genomes))
        queue = Queue()
        procs = []

        for i in range(batch_start, batch_end):
            p = Process(target=eval_worker,
                        args=(genomes[i].to_dict(), config_dict, n_steps, queue, i))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        while not queue.empty():
            idx, result = queue.get()
            results[idx] = result

    return results


def main():
    import random
    rng = random.Random(42)

    print(f"Phase 1B: Evolutionary Energy Functional Search", flush=True)
    print(f"Population: {POP_SIZE}, Generations: {N_GENERATIONS}, "
          f"Steps/eval: {N_STEPS}, Parallel: {N_PARALLEL}", flush=True)

    # Initialize population with diverse genomes
    population = []
    for _ in range(POP_SIZE):
        genome = EnergyGenome(
            lambda_activity=10 ** rng.uniform(-4, 0),
            lambda_weight=10 ** rng.uniform(-5, -1),
            lambda_edge=10 ** rng.uniform(-6, -2),
            lambda_prediction=10 ** rng.uniform(-1, 1),
            lambda_reconstruction=10 ** rng.uniform(-1, 1),
            lambda_mi=10 ** rng.uniform(-1, 1),
            lambda_compartment=10 ** rng.uniform(-2, 1),
        )
        population.append(genome)

    all_gen_results = []

    for gen in range(N_GENERATIONS):
        print(f"\n{'='*70}", flush=True)
        print(f"  GENERATION {gen+1}/{N_GENERATIONS}", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.perf_counter()
        results = evaluate_population_parallel(population, BASE_CONFIG, N_STEPS, N_PARALLEL)
        elapsed = time.perf_counter() - t0

        # Print results
        for i, r in enumerate(results):
            if r is not None:
                print(f"  [{i+1:2d}] fit={r.fitness:7.3f} pred={r.prediction_loss:.3f} "
                      f"diff={r.output_differentiation:.4f} asym={r.weight_asymmetry:.4f} "
                      f"supp={r.suppression_ratio:.4f}", flush=True)

        valid = [r for r in results if r is not None]
        fitnesses = [r.fitness for r in valid]
        best = max(valid, key=lambda r: r.fitness)

        print(f"\n  Gen {gen+1}: best={max(fitnesses):.3f} mean={np.mean(fitnesses):.3f} "
              f"std={np.std(fitnesses):.3f} ({elapsed:.0f}s)", flush=True)
        print(f"  Best: {best.genome}", flush=True)
        print(f"  Metrics: pred={best.prediction_loss:.3f} recon={best.reconstruction_loss:.3f} "
              f"diff={best.output_differentiation:.4f} asym={best.weight_asymmetry:.4f} "
              f"supp={best.suppression_ratio:.4f}", flush=True)

        all_gen_results.append(valid)

        # Selection + breeding (except last generation)
        if gen < N_GENERATIONS - 1:
            valid.sort(key=lambda r: r.fitness, reverse=True)
            n_elite = max(2, POP_SIZE // 4)
            elites = [r.genome for r in valid[:n_elite]]
            top_half = [r.genome for r in valid[:POP_SIZE // 2]]

            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                p1 = rng.choice(top_half)
                p2 = rng.choice(top_half)
                child = p1.crossover(p2, rng)
                child = child.mutate(mutation_rate=0.4, rng=rng)
                new_pop.append(child)
            population = new_pop

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print("EVOLUTION COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)

    # Best per generation
    for gen, gen_results in enumerate(all_gen_results):
        best = max(gen_results, key=lambda r: r.fitness)
        print(f"  Gen {gen+1}: best_fit={best.fitness:.3f} supp={best.suppression_ratio:.4f} "
              f"diff={best.output_differentiation:.4f}", flush=True)

    # Overall best
    all_flat = [r for gen in all_gen_results for r in gen]
    overall_best = max(all_flat, key=lambda r: r.fitness)
    print(f"\n  OVERALL BEST: fitness={overall_best.fitness:.3f}", flush=True)
    print(f"  {overall_best.genome}", flush=True)
    print(f"  suppression={overall_best.suppression_ratio:.4f} "
          f"differentiation={overall_best.output_differentiation:.4f} "
          f"asymmetry={overall_best.weight_asymmetry:.4f}", flush=True)

    torch.save({
        "all_results": [[{"fitness": r.fitness, "genome": r.genome.to_dict(),
                          "suppression": r.suppression_ratio,
                          "differentiation": r.output_differentiation,
                          "asymmetry": r.weight_asymmetry,
                          "prediction": r.prediction_loss}
                         for r in gen] for gen in all_gen_results],
        "best_genome": overall_best.genome.to_dict(),
    }, "evolution_results.pt")
    print("\nSaved to evolution_results.pt", flush=True)


if __name__ == "__main__":
    main()
