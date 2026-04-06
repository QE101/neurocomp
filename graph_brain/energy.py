"""Energy functional for self-organisation experiments.

Combined objective:
    E = prediction_loss + reconstruction_loss + mi_loss
        + lambda_activity * activity_cost
        + lambda_weight * weight_cost
        + lambda_edge * edge_cost

The energy functional drives self-organisation. Nodes and edges adapt to
minimise E, which means they must simultaneously:
    - Predict the next input (prediction_loss)
    - Reconstruct the current input from internal state (reconstruction_loss)
    - Maximise information captured about the input (mi_loss = negative MI)
    - Minimise metabolic cost (activity, weights, edges)

The balance between these objectives (the lambdas) is found by evolution,
not hand-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph
from graph_brain.types import EdgeType, NodeType


@dataclass
class EnergyGenome:
    """The evolvable parameters of the energy functional."""
    lambda_activity: float = 0.01    # penalty on total node activity (sparsity)
    lambda_weight: float = 0.001     # penalty on total edge weight (regularisation)
    lambda_edge: float = 0.0001      # penalty per edge (structural cost)
    lambda_prediction: float = 1.0   # weight on prediction loss
    lambda_reconstruction: float = 0.5  # weight on reconstruction loss
    lambda_mi: float = 0.3          # weight on mutual information loss
    lambda_compartment: float = 0.1  # penalty on |basal - apical| (Zhang et al. 2025)

    # All evolvable field names
    _fields = ["lambda_activity", "lambda_weight", "lambda_edge",
               "lambda_prediction", "lambda_reconstruction", "lambda_mi",
               "lambda_compartment"]

    def mutate(self, mutation_rate: float = 0.3, rng=None) -> EnergyGenome:
        """Return a mutated copy. Log-normal mutations to stay positive."""
        import random
        rng = rng or random
        fields = {}
        for name in self._fields:
            val = getattr(self, name)
            if rng.random() < mutation_rate:
                factor = 2.0 ** (rng.gauss(0, 0.5))
                val = max(val * factor, 1e-8)
            fields[name] = val
        return EnergyGenome(**fields)

    def crossover(self, other: EnergyGenome, rng=None) -> EnergyGenome:
        """Uniform crossover between two genomes."""
        import random
        rng = rng or random
        fields = {}
        for name in self._fields:
            if rng.random() < 0.5:
                fields[name] = getattr(self, name)
            else:
                fields[name] = getattr(other, name)
        return EnergyGenome(**fields)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._fields}

    def __repr__(self):
        return (f"Genome(act={self.lambda_activity:.4f} w={self.lambda_weight:.4f} "
                f"edge={self.lambda_edge:.5f} pred={self.lambda_prediction:.3f} "
                f"recon={self.lambda_reconstruction:.3f} mi={self.lambda_mi:.3f} "
                f"comp={self.lambda_compartment:.3f})")


class EnergyFunctional:
    """Computes the energy of a graph given a genome and input/target."""

    def __init__(self, genome: EnergyGenome):
        self.genome = genome

    def compute(
        self,
        graph: NeuromorphicGraph,
        input_pattern: Tensor,
        target_pattern: Tensor,
        input_nodes: Tensor,
    ) -> dict[str, float]:
        """Compute all energy components. Returns dict of named losses.

        Args:
            graph: the graph to evaluate
            input_pattern: [K] current input values at input_nodes
            target_pattern: [K] next-step target values (for prediction)
            input_nodes: [K] indices of input nodes
        """
        ns = graph.node_state
        device = ns.device
        g = self.genome

        # --- Prediction loss: can the representation predict the next input? ---
        # Use output of excitatory nodes near the input region as prediction
        exc_mask = ns.type_mask(NodeType.EXCITATORY)
        exc_output = ns.output[exc_mask]

        # Prediction = mean output of excitatory nodes at input locations
        # Compare to target pattern
        pred_at_input = ns.output[input_nodes.long()]
        prediction_loss = F.mse_loss(pred_at_input, target_pattern)

        # --- Reconstruction loss: can the representation recreate the current input? ---
        # Use basal compartment at input nodes as reconstruction
        recon_at_input = ns.basal[input_nodes.long()]
        reconstruction_loss = F.mse_loss(
            recon_at_input / (recon_at_input.abs().max() + 1e-6),
            input_pattern / (input_pattern.abs().max() + 1e-6),
        )

        # --- Mutual information proxy: correlation between input and representation ---
        # Use negative correlation as loss (maximise MI = minimise negative correlation)
        if exc_output.numel() > 1 and exc_output.std() > 1e-8:
            # Subsample excitatory outputs for MI estimation
            n_sample = min(500, exc_output.shape[0])
            rep_sample = exc_output[:n_sample]
            # Correlation between representation variance and input variance
            rep_var = rep_sample.var()
            mi_proxy_loss = 1.0 / (rep_var + 0.1)  # low variance = low info = high loss
        else:
            mi_proxy_loss = torch.tensor(10.0, device=device)

        # --- Metabolic costs ---
        # Activity cost: L1 on all outputs (sparsity)
        activity_cost = ns.output.abs().mean()

        # Weight cost: L2 on all edge weights
        total_weight_sq = torch.tensor(0.0, device=device)
        total_edges = 0
        for et in EdgeType:
            if graph.has_edge_type(et):
                store = graph.edge_store(et)
                total_weight_sq += (store.weight ** 2).sum()
                total_edges += store.n_edges
        weight_cost = total_weight_sq / max(total_edges, 1)

        # Edge cost: total number of edges
        edge_cost = torch.tensor(float(total_edges), device=device)

        # --- Total energy ---
        total = (
            g.lambda_prediction * prediction_loss
            + g.lambda_reconstruction * reconstruction_loss
            + g.lambda_mi * mi_proxy_loss
            + g.lambda_activity * activity_cost
            + g.lambda_weight * weight_cost
            + g.lambda_edge * edge_cost
        )

        return {
            "total": float(total),
            "prediction": float(prediction_loss),
            "reconstruction": float(reconstruction_loss),
            "mi_proxy": float(mi_proxy_loss),
            "activity": float(activity_cost),
            "weight": float(weight_cost),
            "edge_count": total_edges,
        }


class TemporalState:
    """Tracks previous-step node states for temporal learning.

    Stores previous output, basal, and apical for computing:
    - Temporal Hebbian: pre_{t-1} × post_t (causal prediction)
    - Confusion signal: rate of change in basal vs apical
    """

    def __init__(self, n_nodes: int, device: str = "cpu"):
        self.prev_output = torch.zeros(n_nodes, device=device)
        self.prev_basal = torch.zeros(n_nodes, device=device)
        self.prev_apical = torch.zeros(n_nodes, device=device)

    def update(self, ns) -> None:
        """Snapshot current state for next step's comparison."""
        self.prev_output = ns.output.detach().clone()
        self.prev_basal = ns.basal.detach().clone()
        self.prev_apical = ns.apical.detach().clone()


# Keep old name as alias for backward compatibility
TemporalHebbianState = TemporalState


def apply_energy_gradient(
    graph: NeuromorphicGraph,
    genome: EnergyGenome,
    temporal_state: TemporalState,
    dt: float = 1.0,
) -> None:
    """Apply energy-driven weight updates with precision-gated local sparsity.

    Attempt 7: Per-node adaptive activity cost.

    The fundamental tension (attempts 1-6): global λ_activity builds structure
    (topology, hierarchy) but kills function (prediction activity). You can't
    simultaneously penalise activity and require prediction activity on every
    node equally.

    Solution: λ_activity is PER-NODE, scaled by precision.

        effective_λ[node] = λ_base × precision[node]

    - High precision (good predictions) → high activity cost → be quiet, save energy
    - Low precision (bad predictions) → low activity cost → be active, learn

    This creates simultaneous structure AND function:
    - Confident nodes: high sparsity → quiet → efficient (structure mode)
    - Uncertain nodes: low sparsity → active → learning (function mode)
    - Both coexist on the same graph at the same time
    - No global mode switching, no two-stage protocol

    Precision is already computed per-node from error statistics. We just
    connect it to the activity cost. Precision naturally increases as
    predictions improve (self-reinforcing: better predictions → higher
    precision → higher sparsity → more efficient). And precision drops
    when the environment changes (novel input → bad predictions → low
    precision → low sparsity → active learning).

    Combined with temporal Hebbian (causal prediction) and accuracy reward
    (apical tracking basal), this gives the system three simultaneous forces:
    1. Learn causal structure (temporal Hebbian)
    2. Reward correct predictions (accuracy)
    3. Be efficient where confident, active where uncertain (precision-gated sparsity)
    """
    ns = graph.node_state
    ts = temporal_state

    # Per-node precision (already tracked in NodeState)
    # Update precision from current prediction error
    pred_error = (ns.basal - ns.apical).abs()
    alpha_p = dt / 200.0  # precision EMA time constant
    ns.error_mean_ema = ns.error_mean_ema * (1 - alpha_p) + pred_error * alpha_p
    ns.precision = (1.0 / (ns.error_mean_ema + 0.1)).clamp(0.1, 50.0)

    # Per-node effective activity cost: confident nodes pay more
    # Scale: precision ranges ~0.1 (very uncertain) to ~50 (very confident)
    # Normalize so mean effective_lambda ≈ genome.lambda_activity
    precision_normalized = ns.precision / (ns.precision.mean() + 1e-6)
    effective_lambda = genome.lambda_activity * precision_normalized

    # Rate of change for accuracy signal
    delta_basal = ns.basal - ts.prev_basal
    delta_apical = ns.apical - ts.prev_apical
    basal_change_mag = delta_basal.abs() + 1e-6
    accuracy = (delta_basal * delta_apical) / (basal_change_mag * (delta_apical.abs() + 1e-6) + 1e-6)
    accuracy = accuracy.clamp(-1.0, 1.0)

    for et in EdgeType:
        if graph.has_edge_type(et) and et != EdgeType.ELECTRICAL:
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue

            src_idx = store.src.long()
            dst_idx = store.dst.long()
            src_prev = ts.prev_output[src_idx]
            dst_curr = ns.output[dst_idx]
            src_curr = ns.output[src_idx]
            dst_prev = ts.prev_output[dst_idx]

            # 1. Temporal Hebbian: causal prediction
            causal = src_prev * dst_curr
            anti_causal = dst_prev * src_curr
            predictive_drive = causal - anti_causal

            # 2. Accuracy reward
            dst_accuracy = accuracy[dst_idx]
            if et in (EdgeType.MODULATORY, EdgeType.INHIB_DENDRITIC):
                accuracy_drive = src_curr * dst_accuracy
            elif et in (EdgeType.DRIVING, EdgeType.INHIB_PERISOMATIC):
                accuracy_drive = src_curr * delta_basal[dst_idx].abs() * 0.01
            else:
                accuracy_drive = torch.zeros_like(src_curr)

            # 3. Precision-gated metabolic penalty
            # Source and destination pay their OWN effective lambda
            weight_decay = genome.lambda_weight * 2.0 * store.weight
            src_lambda = effective_lambda[src_idx]
            dst_lambda = effective_lambda[dst_idx]
            activity_penalty = (src_lambda * src_curr + dst_lambda * dst_curr) * store.weight

            # Net update
            dw = dt * 0.001 * (
                genome.lambda_prediction * predictive_drive
                + genome.lambda_compartment * accuracy_drive
                - weight_decay
                - activity_penalty
            )
            store.weight += dw
            store.weight.clamp_(0.0, 1.0)

    # Update temporal state
    temporal_state.update(ns)
