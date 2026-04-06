"""Analyze Full Diagnostic Data: crunch the multi-GB snapshot data into insights.

Loads the per-epoch snapshots and computes every metric we need to understand
the oscillation mechanism, representation emergence, sleep effects, hierarchy
function, attention circuit, and more.

Outputs:
- CSV of all summary metrics per epoch (for custom plotting)
- Console report of key findings
- Correlation matrix of all metrics (surfaces non-obvious relationships)

Usage: python analyze_diagnostic.py [path_to_snapshots.pkl]
If no path given, uses the latest checkpoint in D:/neurocomp/
"""

import sys, os, glob
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

DATA_DIR = 'D:/neurocomp'


def load_snapshots(path=None):
    """Load the latest snapshot file."""
    if path is None:
        files = sorted(glob.glob(f'{DATA_DIR}/full_diagnostic_snapshots_ep*.pkl'))
        if not files:
            complete = f'{DATA_DIR}/full_diagnostic_complete.pkl'
            if os.path.exists(complete):
                files = [complete]
            else:
                print('No snapshot files found!')
                return None
        path = files[-1]
    print(f'Loading {path}...', flush=True)
    with open(path, 'rb') as f:
        snapshots = pickle.load(f)
    print(f'Loaded {len(snapshots)} epoch snapshots', flush=True)
    return snapshots


def extract_summary_metrics(snapshots):
    """Extract all scalar metrics into a DataFrame (one row per epoch)."""
    rows = []
    for snap in snapshots:
        row = {}
        for key, val in snap.items():
            if isinstance(val, (int, float, np.floating, np.integer)):
                row[key] = float(val)
            elif isinstance(val, bool):
                row[key] = float(val)
            # Skip tensors/arrays (output vectors etc)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index('epoch')
    return df


def analyze_oscillation(df, snapshots):
    """Analyze the oscillation mechanism from critical edge data."""
    print('\n' + '='*60, flush=True)
    print('  1. OSCILLATION MECHANISM', flush=True)
    print('='*60, flush=True)

    # Find critical edge columns
    crit_cols = [c for c in df.columns if c.startswith('crit_') and c.endswith('_w_mean')]
    oja_cols = [c for c in df.columns if c.startswith('crit_') and c.endswith('_oja_mean')]

    if not crit_cols:
        print('  No critical edge data found', flush=True)
        return

    # Similarity columns (proxy for discrimination)
    sim_cols = [c for c in df.columns if c.startswith('sim_')]

    # For each critical edge set, correlate weight with output similarity
    print('\n  Critical edge weight vs representation similarity:', flush=True)
    for crit_col in crit_cols:
        pair = crit_col.replace('crit_', '').replace('_w_mean', '')
        # Find the corresponding similarity
        parts = pair.split('->')
        if len(parts) == 2:
            src, dst_et = parts
            dst = dst_et.split('_')[0]
            sim_key = f'sim_{src}_{dst}'
            if sim_key not in df.columns:
                sim_key = f'sim_{dst}_{src}'
            if sim_key in df.columns:
                valid = df[[crit_col, sim_key]].dropna()
                if len(valid) > 10:
                    corr = valid[crit_col].corr(valid[sim_key])
                    print(f'    {pair}: weight-similarity r={corr:.3f} ({len(valid)} epochs)', flush=True)

    # Cross-sequence competition
    day_crits = [c for c in crit_cols if 'Mon' in c or 'Tue' in c or 'Wed' in c]
    digit_crits = [c for c in crit_cols if c.startswith('crit_D')]
    if day_crits and digit_crits:
        day_mean = df[day_crits].mean(axis=1)
        digit_mean = df[digit_crits].mean(axis=1)
        valid = pd.concat([day_mean, digit_mean], axis=1).dropna()
        if len(valid) > 10:
            comp_corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            print(f'\n  Cross-sequence competition (days vs digits weights): r={comp_corr:.3f}', flush=True)
            if comp_corr < -0.3:
                print('  >> ANTI-CORRELATED: sequences compete for edge strength', flush=True)
            elif comp_corr > 0.3:
                print('  >> CORRELATED: sequences rise/fall together (systemic)', flush=True)
            else:
                print('  >> INDEPENDENT: sequences don\'t interfere', flush=True)

    # Oja force analysis
    if oja_cols:
        print('\n  Oja stabilizer force on critical edges:', flush=True)
        for col in oja_cols[:5]:
            pair = col.replace('crit_', '').replace('_oja_mean', '')
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f'    {pair}: mean={vals.mean():.4f}, trend={vals.iloc[-5:].mean() - vals.iloc[:5].mean():+.4f}', flush=True)


def analyze_representations(df, snapshots):
    """Analyze representation emergence and stability."""
    print('\n' + '='*60, flush=True)
    print('  2. REPRESENTATION EMERGENCE', flush=True)
    print('='*60, flush=True)

    # Sparsity trajectory
    active_cols = [c for c in df.columns if c.startswith('active_pct_')]
    if active_cols:
        print('\n  Sparsity (active % per symbol):', flush=True)
        for col in active_cols[:5]:
            sym = col.replace('active_pct_', '')
            early = df[col].iloc[:10].mean() * 100
            late = df[col].iloc[-10:].mean() * 100
            print(f'    {sym}: early={early:.1f}% -> late={late:.1f}% (bio target: 17%)', flush=True)

    # Representational stability (cosine sim of output vectors between consecutive epochs)
    print('\n  Representational stability (output vector consistency):', flush=True)
    symbols_to_check = ['Mon', 'Tue', 'D1', 'D2']
    for sym in symbols_to_check:
        key = f'output_{sym}'
        stabilities = []
        for i in range(1, min(len(snapshots), 50)):
            if key in snapshots[i] and key in snapshots[i-1]:
                a = snapshots[i][key].float()
                b = snapshots[i-1][key].float()
                if a.norm() > 0 and b.norm() > 0:
                    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
                    stabilities.append(cos)
        if stabilities:
            early_stab = np.mean(stabilities[:10])
            late_idx = min(len(stabilities), 40)
            late_stab = np.mean(stabilities[max(0,late_idx-10):late_idx])
            print(f'    {sym}: early_stability={early_stab:.3f} -> late={late_stab:.3f} '
                  f'(1.0 = perfectly stable)', flush=True)

    # Symbol similarity evolution
    sim_cols = [c for c in df.columns if c.startswith('sim_')]
    if sim_cols:
        # Within-sequence similarity (should stay high or increase)
        within_day = [c for c in sim_cols if all(d in c for d in ['Mon']) and any(d in c for d in ['Tue', 'Wed'])]
        # Cross-sequence similarity (should decrease)
        cross = [c for c in sim_cols if ('Mon' in c or 'Tue' in c) and ('D1' in c or 'D2' in c)]

        if within_day:
            early_within = df[within_day].iloc[:10].mean().mean()
            late_within = df[within_day].iloc[-10:].mean().mean()
            print(f'\n  Within-sequence similarity (days): {early_within:.3f} -> {late_within:.3f}', flush=True)
        if cross:
            early_cross = df[cross].iloc[:10].mean().mean()
            late_cross = df[cross].iloc[-10:].mean().mean()
            print(f'  Cross-sequence similarity (days vs digits): {early_cross:.3f} -> {late_cross:.3f}', flush=True)
            if late_cross < early_cross:
                print('  >> Representations DIVERGING between sequences (good)', flush=True)


def analyze_sleep(df, snapshots):
    """Analyze sleep effects."""
    print('\n' + '='*60, flush=True)
    print('  3. SLEEP EFFECTS', flush=True)
    print('='*60, flush=True)

    sleep_cols = [c for c in df.columns if c.startswith('sleep_')]
    if not sleep_cols:
        print('  No sleep data yet (sleep happens every 20 epochs)', flush=True)
        return

    # Weight change during sleep
    change_cols = [c for c in sleep_cols if '_change_mean' in c]
    ratio_cols = [c for c in sleep_cols if '_ratio' in c]

    if change_cols:
        print('\n  Weight change during sleep (per edge type):', flush=True)
        for col in change_cols:
            et = col.replace('sleep_', '').replace('_change_mean', '')
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f'    {et}: mean_change={vals.mean():.6f} (negative = downscaling)', flush=True)

    if ratio_cols:
        print('\n  Post/pre sleep weight ratio:', flush=True)
        for col in ratio_cols:
            et = col.replace('sleep_', '').replace('_ratio', '')
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f'    {et}: ratio={vals.mean():.4f} (0.95 = pure homeostasis)', flush=True)


def analyze_consolidation(df):
    """Analyze consolidation dynamics."""
    print('\n' + '='*60, flush=True)
    print('  4. CONSOLIDATION', flush=True)
    print('='*60, flush=True)

    if 'driving_n_frozen' in df.columns:
        frozen = df['driving_n_frozen'].dropna()
        if len(frozen) > 0:
            first_freeze = frozen[frozen > 0].index[0] if (frozen > 0).any() else 'never'
            print(f'  First frozen edges at epoch: {first_freeze}', flush=True)
            print(f'  Final frozen count: {frozen.iloc[-1]:.0f}', flush=True)
            print(f'  Frozen growth: {frozen.iloc[:10].mean():.0f} -> {frozen.iloc[-10:].mean():.0f}', flush=True)

    if 'driving_replay_count_mean' in df.columns:
        rc = df['driving_replay_count_mean'].dropna()
        if len(rc) > 0:
            print(f'  Replay count: early={rc.iloc[:10].mean():.2f} -> late={rc.iloc[-10:].mean():.2f}', flush=True)


def analyze_hierarchy(df):
    """Analyze hierarchy function."""
    print('\n' + '='*60, flush=True)
    print('  5. HIERARCHY', flush=True)
    print('='*60, flush=True)

    if 'l1_mean_out' in df.columns and 'l2_mean_out' in df.columns:
        l1 = df['l1_mean_out'].dropna()
        l2 = df['l2_mean_out'].dropna()
        print(f'  Level 1 output: early={l1.iloc[:10].mean():.4f} -> late={l1.iloc[-10:].mean():.4f}', flush=True)
        print(f'  Level 2 output: early={l2.iloc[:10].mean():.4f} -> late={l2.iloc[-10:].mean():.4f}', flush=True)
        print(f'  L2/L1 ratio: early={l2.iloc[:10].mean()/max(l1.iloc[:10].mean(),1e-8):.3f} -> '
              f'late={l2.iloc[-10:].mean()/max(l1.iloc[-10:].mean(),1e-8):.3f}', flush=True)

    if 'l1_mean_ema' in df.columns and 'l2_mean_ema' in df.columns:
        print(f'  L1 activity EMA: {df["l1_mean_ema"].iloc[-10:].mean():.4f}', flush=True)
        print(f'  L2 activity EMA: {df["l2_mean_ema"].iloc[-10:].mean():.4f}', flush=True)


def analyze_attention(df):
    """Analyze VIP/SST/PV dynamics."""
    print('\n' + '='*60, flush=True)
    print('  6. ATTENTION CIRCUIT (VIP/SST/PV)', flush=True)
    print('='*60, flush=True)

    for nt in ['pv', 'sst', 'vip']:
        mean_col = f'{nt}_mean_out'
        std_col = f'{nt}_std_out'
        if mean_col in df.columns:
            vals = df[mean_col].dropna()
            early = vals.iloc[:10].mean()
            late = vals.iloc[-10:].mean()
            trend = 'UP' if late > early * 1.05 else 'DOWN' if late < early * 0.95 else 'STABLE'
            print(f'  {nt.upper()}: early={early:.4f} -> late={late:.4f} ({trend})', flush=True)


def analyze_structural(df):
    """Analyze structural plasticity."""
    print('\n' + '='*60, flush=True)
    print('  7. STRUCTURAL PLASTICITY', flush=True)
    print('='*60, flush=True)

    if 'total_edges' in df.columns:
        edges = df['total_edges'].dropna()
        print(f'  Edges: start={edges.iloc[0]:.0f} -> end={edges.iloc[-1]:.0f} '
              f'(net {edges.iloc[-1]-edges.iloc[0]:+.0f})', flush=True)

    # Edge count per type
    for et_name in ['driving', 'modulatory', 'inhib_perisomatic', 'inhib_dendritic', 'disinhibition']:
        col = f'{et_name}_n_edges'
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f'  {et_name}: {vals.iloc[0]:.0f} -> {vals.iloc[-1]:.0f}', flush=True)


def analyze_timing(df):
    """Analyze effects of timing graduation."""
    print('\n' + '='*60, flush=True)
    print('  8. TIMING ADAPTATION', flush=True)
    print('='*60, flush=True)

    if 'pd_steps' in df.columns:
        # Find transition points
        steps = df['pd_steps']
        transitions = steps.diff().fillna(0) != 0
        trans_epochs = df.index[transitions].tolist()
        print(f'  Timing transitions at epochs: {trans_epochs}', flush=True)

        for epoch in trans_epochs:
            before = df.loc[max(1, epoch-5):epoch-1]
            after = df.loc[epoch:epoch+5]
            act_col = [c for c in df.columns if c.startswith('active_pct_Mon')]
            if act_col:
                b = before[act_col[0]].mean() * 100
                a = after[act_col[0]].mean() * 100
                print(f'    Epoch {epoch}: Mon active {b:.1f}% -> {a:.1f}%', flush=True)


def compute_correlation_matrix(df):
    """Compute correlation matrix of all scalar metrics."""
    print('\n' + '='*60, flush=True)
    print('  9. CORRELATION ANALYSIS', flush=True)
    print('='*60, flush=True)

    # Select numeric columns with enough variance
    numeric = df.select_dtypes(include=[np.number])
    # Drop columns with no variance
    std = numeric.std()
    numeric = numeric.loc[:, std > 1e-8]

    if numeric.shape[1] < 5:
        print('  Not enough numeric columns for correlation', flush=True)
        return None

    corr = numeric.corr()

    # Find strongest non-trivial correlations (exclude obvious pairs)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            c = corr.iloc[i, j]
            if abs(c) > 0.5 and not np.isnan(c):
                col_i, col_j = corr.columns[i], corr.columns[j]
                # Skip trivial: same metric different aggregation (mean vs sum, etc)
                base_i = col_i.rsplit('_', 1)[0] if '_' in col_i else col_i
                base_j = col_j.rsplit('_', 1)[0] if '_' in col_j else col_j
                if base_i == base_j:
                    continue
                # Skip: same symbol different edge type
                if col_i.split('_')[0] == col_j.split('_')[0] and len(col_i.split('_')[0]) <= 3:
                    continue
                pairs.append((col_i, col_j, c))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f'\n  Top 20 strongest correlations (|r| > 0.5):', flush=True)
    for col1, col2, r in pairs[:20]:
        print(f'    {col1} <-> {col2}: r={r:.3f}', flush=True)

    return corr


def analyze_gain(df):
    """Analyze intrinsic plasticity effects."""
    print('\n' + '='*60, flush=True)
    print('  10. INTRINSIC PLASTICITY (GAIN)', flush=True)
    print('='*60, flush=True)

    if 'gain_mean' in df.columns:
        g = df['gain_mean'].dropna()
        print(f'  Gain: early={g.iloc[:10].mean():.4f} -> late={g.iloc[-10:].mean():.4f}', flush=True)
    if 'gain_std' in df.columns:
        gs = df['gain_std'].dropna()
        print(f'  Gain spread: early={gs.iloc[:10].mean():.4f} -> late={gs.iloc[-10:].mean():.4f}', flush=True)
        if gs.iloc[-10:].mean() > gs.iloc[:10].mean() * 1.5:
            print('  -> Nodes DIFFERENTIATING gain (some sensitive, some not)', flush=True)


def main():
    print('='*60, flush=True)
    print('  DIAGNOSTIC ANALYSIS', flush=True)
    print('='*60, flush=True)

    path = sys.argv[1] if len(sys.argv) > 1 else None
    snapshots = load_snapshots(path)
    if snapshots is None:
        return

    # Need torch for representation analysis
    global torch
    import torch

    # Extract summary metrics
    print('\nExtracting summary metrics...', flush=True)
    df = extract_summary_metrics(snapshots)
    print(f'  {len(df)} epochs, {len(df.columns)} metrics per epoch', flush=True)

    # Save CSV
    csv_path = f'{DATA_DIR}/diagnostic_metrics.csv'
    df.to_csv(csv_path)
    print(f'  Saved metrics CSV: {csv_path}', flush=True)

    # Run all analyses
    analyze_oscillation(df, snapshots)
    analyze_representations(df, snapshots)
    analyze_sleep(df, snapshots)
    analyze_consolidation(df)
    analyze_hierarchy(df)
    analyze_attention(df)
    analyze_structural(df)
    analyze_timing(df)
    analyze_gain(df)
    corr = compute_correlation_matrix(df)

    # Save correlation matrix
    if corr is not None:
        corr_path = f'{DATA_DIR}/diagnostic_correlations.csv'
        corr.to_csv(corr_path)
        print(f'\n  Saved correlation matrix: {corr_path}', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'  ANALYSIS COMPLETE', flush=True)
    print(f'  Metrics CSV: {csv_path}', flush=True)
    print(f'  {len(df)} epochs analyzed, {len(df.columns)} metrics', flush=True)
    print(f'{"="*60}', flush=True)


if __name__ == '__main__':
    main()
