"""Sweep result analysis and visualization."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_brain.sweep.runner import SweepResult


def results_to_dataframe(results: list[SweepResult]) -> pd.DataFrame:
    """Convert sweep results to a DataFrame for analysis."""
    rows = []
    for r in results:
        row = {**r.params}
        row["config_id"] = r.config_id
        row["status"] = r.status
        row["final_output_mean"] = r.final_output_mean
        row["final_activity_ema"] = r.final_activity_ema
        row["final_weight_mean"] = r.final_weight_mean
        row["output_std"] = r.output_std
        row["max_output"] = r.max_output
        row["run_time_sec"] = r.run_time_sec
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print sweep summary statistics."""
    total = len(df)
    stable = (df["status"] == "STABLE").sum()
    exploded = (df["status"] == "EXPLODED").sum()
    died = (df["status"] == "DIED").sum()
    errors = (df["status"] == "ERROR").sum()

    print(f"\n{'='*60}")
    print(f"Sweep Summary: {total} configurations")
    print(f"  STABLE:   {stable} ({stable/total*100:.0f}%)")
    print(f"  EXPLODED: {exploded} ({exploded/total*100:.0f}%)")
    print(f"  DIED:     {died} ({died/total*100:.0f}%)")
    if errors:
        print(f"  ERROR:    {errors} ({errors/total*100:.0f}%)")
    print(f"{'='*60}")

    if stable > 0:
        stable_df = df[df["status"] == "STABLE"]
        print(f"\nStable configs — output mean: "
              f"{stable_df['final_output_mean'].mean():.4f} "
              f"± {stable_df['final_output_mean'].std():.4f}")
        print(f"Stable configs — weight mean: "
              f"{stable_df['final_weight_mean'].mean():.4f} "
              f"± {stable_df['final_weight_mean'].std():.4f}")


def plot_stability_heatmap(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2D heatmap showing stability region in parameter space."""
    # Encode status as numeric: STABLE=1, DIED=0, EXPLODED=-1
    status_map = {"STABLE": 1, "DIED": 0, "EXPLODED": -1, "ERROR": -2}
    df = df.copy()
    df["status_num"] = df["status"].map(status_map)

    pivot = df.pivot_table(
        values="status_num",
        index=y_param,
        columns=x_param,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot.values,
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.4g}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.4g}" for v in pivot.index])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title("Stability Region (green=stable, red=exploded, yellow=died)")
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], format=plt.FuncFormatter(
        lambda x, _: {-1: "EXPLODED", 0: "DIED", 1: "STABLE"}.get(int(x), "?")))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_param_importance(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of parameter importance for stability."""
    param_cols = [c for c in df.columns if c not in {
        "config_id", "status", "final_output_mean", "final_activity_ema",
        "final_weight_mean", "output_std", "max_output", "run_time_sec", "status_num"
    }]

    importances = {}
    stable_mask = df["status"] == "STABLE"

    for col in param_cols:
        if df[col].nunique() <= 1:
            continue
        # Mutual information proxy: correlation between param and stability
        try:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            if numeric_col.notna().all():
                corr = abs(numeric_col.corr(stable_mask.astype(float)))
                importances[col] = corr
        except Exception:
            pass

    if not importances:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameter importance data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(importances.keys())
    vals = list(importances.values())
    sort_idx = np.argsort(vals)[::-1]
    ax.barh([names[i] for i in sort_idx], [vals[i] for i in sort_idx], color="steelblue")
    ax.set_xlabel("|Correlation with Stability|")
    ax.set_title("Parameter Importance for Stability")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
