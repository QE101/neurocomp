"""Context disambiguation diagnostic.

Tests whether the graph's INTERNAL STATE differs across contexts:
  Present a26 -> a30, freeze, snapshot all activations
  Present a28 -> a30, freeze, snapshot all activations
  Present a31 -> a30, freeze, snapshot all activations
  Compare snapshots per level.

If patterns are identical: no level encodes context. Architecture problem.
If patterns differ but readout fails: probe is wrong.
If patterns differ significantly: substrate works, learning rule needs work.

Loads from latest checkpoint or trains briefly if none exists.

Usage:
    python3 scripts/diagnose_context.py [checkpoint_path]
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import everything from the main script — same config, same setup
from scripts import run_adaptive_consolidation as rac
from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.core.message_passing import TypedMessagePasser
from graph_brain.core.delay_buffer import Channel
from graph_brain.engine.fused_plasticity import FusedPlasticity
from graph_brain.edges.short_term import ShortTermPlasticity
from graph_brain.edges.homeostatic import HomeostaticScaling
from graph_brain.nodes.intrinsic import IntrinsicPlasticity
from graph_brain.oscillations import ThetaDrive
from graph_brain.hierarchy import HierarchyBuilder
from graph_brain.types import EdgeType, NodeType


def run_context_test(ns, graph, mp, theta, tau_mult, fused, exc_mask,
                     symbols, ctx_name, hub_name='a30', steps=100):
    """Present ctx -> hub, return snapshots of basal/apical/output at end."""
    ctx_pattern = symbols[ctx_name]
    hub_pattern = symbols[hub_name]

    # Present context predecessor
    for s in range(steps):
        step = graph.step_count
        rac.dual_channel_send(ns, graph, mp, ns.device)
        inputs = mp.read_inputs(step)
        inputs.basal[ctx_pattern.long()] += 2.0
        rac.error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        fused.stp(ns.output)
        graph.increment_step()

    # Present hub
    for s in range(steps):
        step = graph.step_count
        rac.dual_channel_send(ns, graph, mp, ns.device)
        inputs = mp.read_inputs(step)
        inputs.basal[hub_pattern.long()] += 2.0
        rac.error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        fused.stp(ns.output)
        graph.increment_step()

    # Snapshot
    return {
        'basal': ns.basal.clone(),
        'apical': ns.apical.clone(),
        'output': ns.output.clone(),
        'pred_err': ns.prediction_error.clone(),
    }


def cooldown(ns, graph, mp, theta, tau_mult, steps=30):
    for s in range(steps):
        step = graph.step_count
        rac.dual_channel_send(ns, graph, mp, ns.device)
        inputs = mp.read_inputs(step)
        rac.error_node_update(ns, inputs, theta_mod=theta.get_modulation(step), tau_mult=tau_mult)
        graph.increment_step()


def main():
    print("=" * 60)
    print("  CONTEXT DISAMBIGUATION DIAGNOSTIC")
    print("=" * 60)

    # Build graph (same config as training script)
    config = GraphBrainConfig.from_dict(rac.CONFIG_50K)
    graph = NeuromorphicGraph(config)
    graph.initialize()
    ns = graph.node_state
    device = graph.device
    rac.cache_type_masks(ns)

    # Recurrent + small world (same as training)
    rac.add_recurrent_edges(graph, k_recurrent=10)
    rac.add_small_world_edges(graph, fraction=0.2)

    # Hierarchy
    builder = HierarchyBuilder(config)
    h_stats, tau_mult = builder.build(graph)
    print(f"\nHierarchy: {config.hierarchy.n_levels} levels")
    for lv in range(1, config.hierarchy.n_levels + 1):
        n_lv = (ns.hierarchy_level == lv).sum().item()
        print(f"  L{lv}: {n_lv:,} nodes")

    # Components
    mp = TypedMessagePasser(config, ns.n_nodes, device)
    stp = ShortTermPlasticity(config.edges.stp)
    hom = HomeostaticScaling(config.edges.homeostatic)
    ip = IntrinsicPlasticity(config.nodes)
    theta = ThetaDrive(frequency_hz=6.0, amplitude=0.5)
    rac.precompute_edge_data(graph, mp)
    rac.init_noise_buffer(ns.n_nodes, device)

    # Build sensory surface (same seed)
    exc_idx = torch.where(rac._exc_mask)[0]
    exc_z = ns.position[exc_idx, 2]
    input_region = exc_idx[exc_z <= exc_z.quantile(rac.INPUT_FRACTION)]
    symbols, sequences, novel_sequences, n_on = rac.build_sensory_symbols(input_region, device)

    # Fused plasticity
    rac._fused = FusedPlasticity(graph, config.edges.stp, mp.delay_buffer)
    fused = rac._fused

    # Try to load checkpoint
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt_path is None:
        # Find latest in default dir
        ckpts = sorted(Path('checkpoints/adaptive_consolidation').glob('epoch_*.pt'))
        if ckpts:
            ckpt_path = str(ckpts[-1])

    if ckpt_path and Path(ckpt_path).exists():
        print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
        rac.load_checkpoint(ckpt_path, graph)
        # Reload fused buffer copies after weight restore
        rac._fused = FusedPlasticity(graph, config.edges.stp, mp.delay_buffer)
        fused = rac._fused
    else:
        print("\nNo checkpoint found. Diagnostic on UNTRAINED graph (sanity check only).", flush=True)
        print("For meaningful results, run training first or pass a checkpoint path.", flush=True)

    # Build per-level masks (excitatory only)
    level_masks = {}
    for lv in range(1, config.hierarchy.n_levels + 1):
        m = ns.type_mask(NodeType.EXCITATORY) & (ns.hierarchy_level == lv)
        level_masks[lv] = m

    # ================================================================
    # Run all 3 contexts
    # ================================================================
    print("\nRunning context test (3 predecessors -> a30):", flush=True)

    # Reset state
    ns.basal.zero_(); ns.apical.zero_(); ns.output.zero_()
    ns.prediction_error.zero_()
    mp.delay_buffer.reset()

    contexts = ['a26', 'a28', 'a31']

    # Test BOTH timings: slow (100 steps) and fast (5 steps)
    for label, steps in [('SLOW (100 steps)', 100), ('FAST (5 steps)', 5)]:
        print(f"\n--- {label} ---", flush=True)
        snapshots = {}
        for ctx in contexts:
            cooldown(ns, graph, mp, theta, tau_mult, steps=50)
            ns.basal.zero_(); ns.apical.zero_()
            snap = run_context_test(ns, graph, mp, theta, tau_mult, fused,
                                     rac._exc_mask, symbols, ctx, steps=steps)
            snapshots[ctx] = snap
            print(f"  {ctx} -> a30: done", flush=True)

        compare_snapshots(snapshots, level_masks, contexts, config.hierarchy.n_levels)

    print("\n" + "=" * 60)
    print("  INTERPRETATION")
    print("=" * 60)
    print("""
  diff = 0.00 -> patterns identical (no context encoding)
  diff = 0.01-0.05 -> tiny differences (substrate sees, readout can't)
  diff = 0.05-0.20 -> moderate (substrate works, learning rule weak)
  diff > 0.20 -> strong differentiation
""")


def compare_snapshots(snapshots, level_masks, contexts, n_levels):
    print(f"\n  {'Level':<8} {'metric':<12} {'a26 vs a28':>12} {'a26 vs a31':>12} {'a28 vs a31':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    def cos_sim(a, b):
        if a.norm() < 1e-6 or b.norm() < 1e-6:
            return 0.0
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    for lv in range(1, n_levels + 1):
        mask = level_masks[lv]
        if mask.sum().item() == 0:
            continue
        for field in ['basal', 'apical', 'output']:
            patterns = {ctx: snapshots[ctx][field][mask] for ctx in contexts}
            diff_ab = 1 - cos_sim(patterns['a26'], patterns['a28'])
            diff_ac = 1 - cos_sim(patterns['a26'], patterns['a31'])
            diff_bc = 1 - cos_sim(patterns['a28'], patterns['a31'])
            print(f"  L{lv:<7} {field:<12} {diff_ab:>12.4f} {diff_ac:>12.4f} {diff_bc:>12.4f}")
        print()


if __name__ == '__main__':
    main()
