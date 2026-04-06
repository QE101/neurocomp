"""Contrastive learning with predict/observe phases.

Each pattern presentation has two phases:

PREDICT phase (input OFF):
    The network runs from internal state alone. Whatever activates is the
    system's prediction of what's coming. No external input — activity is
    purely internally generated.

OBSERVE phase (input ON):
    The actual input arrives. The system sees reality.

Learning rule:
    For each edge, compare the activity of its destination node between
    the two phases. If predict_activity ≈ observe_activity, the prediction
    was correct — strengthen. If they differ, the prediction was wrong —
    weaken. This is contrastive Hebbian learning (Hinton, Movellan).

    Δw ∝ (src_predict × dst_predict) - (src_observe × dst_observe)
    when prediction matches observation, both terms are similar → small update
    when prediction is wrong, the terms differ → edge adjusts

Combined with sparsity pressure:
    - Silence during predict phase = no prediction = no reward
    - Correct prediction during predict phase + silence during observe phase
      = maximum efficiency (predicted correctly, saved energy by suppressing)
    - Wrong prediction during predict phase = activity during both phases = expensive

This creates the incentive: "predict correctly and suppress" > "be silent" > "predict wrong"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.energy import EnergyGenome
from graph_brain.types import EdgeType


@dataclass
class PhaseSnapshot:
    """Captured node state during one phase."""
    output: Tensor     # [N]
    basal: Tensor      # [N]
    apical: Tensor     # [N]


class ContrastiveLearning:
    """Contrastive predict/observe weight updates."""

    def __init__(self, genome: EnergyGenome):
        self.genome = genome

    def capture_snapshot(self, graph: NeuromorphicGraph) -> PhaseSnapshot:
        """Capture current node state as a phase snapshot."""
        ns = graph.node_state
        return PhaseSnapshot(
            output=ns.output.detach().clone(),
            basal=ns.basal.detach().clone(),
            apical=ns.apical.detach().clone(),
        )

    def update_weights(
        self,
        graph: NeuromorphicGraph,
        predict_snap: PhaseSnapshot,
        observe_snap: PhaseSnapshot,
    ) -> dict[str, float]:
        """Apply contrastive weight update after both phases complete.

        Strengthen edges where predict and observe matched.
        Weaken edges where they differed.

        Returns metrics dict.
        """
        g = self.genome
        ns = graph.node_state

        total_match = 0.0
        total_mismatch = 0.0
        n_edges_updated = 0

        for et in EdgeType:
            if not graph.has_edge_type(et) or et == EdgeType.ELECTRICAL:
                continue

            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            src_idx = store.src.long()
            dst_idx = store.dst.long()

            # Predict phase: what the system generated internally
            src_pred = predict_snap.output[src_idx]
            dst_pred = predict_snap.output[dst_idx]

            # Observe phase: what actually happened
            src_obs = observe_snap.output[src_idx]
            dst_obs = observe_snap.output[dst_idx]

            # Contrastive Hebbian:
            # predict_term: edges active during prediction (internal model)
            # observe_term: edges active during observation (reality)
            predict_term = src_pred * dst_pred
            observe_term = src_obs * dst_obs

            # Match signal: how similar were the two phases at each edge?
            # High match = prediction was correct for this edge's endpoints
            match = predict_term * observe_term / (predict_term.abs() + observe_term.abs() + 1e-6)

            # Contrastive update: strengthen when phases agree, weaken when they disagree
            # When predict matches observe: both terms similar → dw ≈ 0 (already correct)
            # When predict differs from observe: large discrepancy → adjust toward observe
            dw_contrastive = observe_term - predict_term

            # For modulatory edges: extra incentive to match predictions to observations
            if et in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
                # Apical tracking: did the apical prediction match the basal observation?
                apical_pred = predict_snap.apical[dst_idx]
                basal_obs = observe_snap.basal[dst_idx]
                tracking_quality = 1.0 - (apical_pred - basal_obs).abs() / (basal_obs.abs() + 1e-6)
                tracking_quality = tracking_quality.clamp(-1.0, 1.0)

                # Reward modulatory edges that carried accurate predictions
                dw_tracking = src_pred * tracking_quality * 0.5
            else:
                dw_tracking = torch.zeros_like(dw_contrastive)

            # Metabolic penalty (same as energy gradient)
            weight_decay = g.lambda_weight * 2.0 * store.weight
            activity_cost = g.lambda_activity * (src_obs + dst_obs) * store.weight * 0.1

            # Combined update
            dw = 0.001 * (
                g.lambda_prediction * dw_contrastive
                + g.lambda_compartment * dw_tracking
                - weight_decay
                - activity_cost
            )

            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

            # Metrics
            total_match += float(match.sum())
            total_mismatch += float((predict_term - observe_term).abs().sum())
            n_edges_updated += store.n_edges

        return {
            "match": total_match / max(n_edges_updated, 1),
            "mismatch": total_mismatch / max(n_edges_updated, 1),
        }
