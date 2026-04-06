"""CLI entry point: parameter sweep for stability analysis."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

from graph_brain.config import GraphBrainConfig
from graph_brain.sweep.runner import SweepRunner
from graph_brain.sweep.analysis import results_to_dataframe, print_summary, plot_stability_heatmap, plot_param_importance


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep")
    parser.add_argument("--config", default="configs/phase0_validation.yaml")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of random samples")
    parser.add_argument("--n-steps", type=int, default=2000, help="Sim steps per config")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--strategy", default="random", choices=["random", "grid"])
    parser.add_argument("--output-dir", default="sweep_results")
    args = parser.parse_args()

    base_config = GraphBrainConfig.from_yaml(args.config)

    sweep_spec = {
        "edges.stdp.learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05],
        "edges.stdp.a_minus": [0.005, 0.00525, 0.0055, 0.006],
        "nodes.noise_std": [0.001, 0.005, 0.01, 0.05],
        "edges.connectivity.inhib_perisomatic.p_max": [0.3, 0.5, 0.7],
        "nodes.ip_learning_rate": [0.0, 0.0001, 0.001],
    }

    runner = SweepRunner(base_config, sweep_spec)
    results = runner.run(
        n_samples=args.n_samples,
        strategy=args.strategy,
        n_steps=args.n_steps,
        n_workers=args.workers,
    )

    df = results_to_dataframe(results)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sweep_results.csv", index=False)

    print_summary(df)

    # Plots
    if df["status"].nunique() > 1:
        # Find two params with most unique values for heatmap
        param_cols = [c for c in sweep_spec.keys()]
        if len(param_cols) >= 2:
            plot_stability_heatmap(
                df, param_cols[0], param_cols[1],
                save_path=str(out_dir / "stability_heatmap.png"),
            )
        plot_param_importance(df, save_path=str(out_dir / "param_importance.png"))
        print(f"\nPlots saved to {out_dir}/")

    # Print best stable configs
    stable = df[df["status"] == "STABLE"]
    if len(stable) > 0:
        print(f"\nTop 5 stable configs (by lowest output std):")
        top = stable.nsmallest(5, "output_std")
        for _, row in top.iterrows():
            params = {k: row[k] for k in sweep_spec.keys() if k in row}
            print(f"  output={row['final_output_mean']:.4f} std={row['output_std']:.4f} | {params}")


if __name__ == "__main__":
    main()
