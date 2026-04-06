"""Predictive coding node dynamics with adaptive precision.

Error nodes:
    prediction_error = basal (evidence) - apical (prediction)
    output = precision * f(|prediction_error|)

    Precision is estimated locally per node from error statistics:
        error_mean_ema = EMA of prediction_error
        error_var_ema  = EMA of (prediction_error - error_mean)^2
        precision = 1 / (error_var_ema + eps), clamped to [min, max]

    This creates the self-balancing loop:
        - Predictions too weak → error consistent → variance low → precision HIGH
          → predictions amplified → error decreases
        - Predictions too strong → error oscillates → variance high → precision LOW
          → predictions dampened → error stabilizes

Representation nodes:
    Integrate precision-weighted error signals from below.
    Their output IS the prediction sent downward via modulatory edges.
    Slow decay — they hold state across pattern presentations.

    Update: Δbasal = lr × precision_of_source × error_signal

Modulatory edge weight update (PC-native, replaces STDP on these edges):
    Δw = lr × precision_at_target × error_at_target × output_of_source
    Self-limiting: as predictions improve, error drops, updates stop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_brain.config import GraphBrainConfig
from graph_brain.core.graph import EdgeStore, NodeState, NeuromorphicGraph
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.types import EdgeType, HierarchyLevel, NodeRole, NodeType


class PredictiveCodingModel:
    """PC-aware node dynamics with adaptive precision."""

    def __init__(self, config: GraphBrainConfig):
        self.node_cfg = config.nodes
        self.h_cfg = config.hierarchy
        self.dt = config.nodes.dt

        # Precision EMA time constant (how fast precision adapts)
        self.precision_tau = 200.0  # ms — adapts over ~200 steps
        self.precision_min = 0.5
        self.precision_max = 100.0

    def step(
        self,
        node_state: NodeState,
        inputs: CompartmentInputs,
        current_time: float,
    ) -> None:
        """Update all node states in-place with role-specific PC dynamics."""
        dt = self.dt
        device = node_state.device

        # Masks
        error_mask = node_state.role_mask(NodeRole.ERROR)
        repr_mask = node_state.role_mask(NodeRole.REPRESENTATION)
        none_mask = node_state.role_mask(NodeRole.NONE)
        exc_mask = node_state.type_mask(NodeType.EXCITATORY)
        pv_mask = node_state.type_mask(NodeType.PV)
        sst_mask = node_state.type_mask(NodeType.SST)
        vip_mask = node_state.type_mask(NodeType.VIP)

        # =====================
        # ERROR NODES
        # =====================
        if error_mask.any():
            err_f = error_mask.float()

            # Leaky integration of evidence (basal) and prediction (apical)
            node_state.basal += dt * (
                -node_state.basal / self.node_cfg.basal_tau + inputs.basal
            ) * err_f

            sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
            node_state.apical += dt * (
                -node_state.apical / self.node_cfg.apical_tau
                + inputs.apical * (1.0 - sst_gate)
            ) * err_f

            # Prediction error
            pred_error = node_state.basal - node_state.apical
            node_state.prediction_error = torch.where(error_mask, pred_error, node_state.prediction_error)

            # Update precision statistics
            # Track absolute error magnitude — this decreases as predictions improve,
            # regardless of sign oscillation from temporal patterns (A/B alternation)
            alpha_p = dt / self.precision_tau
            abs_error = pred_error.abs()
            node_state.error_mean_ema = torch.where(
                error_mask,
                node_state.error_mean_ema * (1 - alpha_p) + abs_error * alpha_p,
                node_state.error_mean_ema,
            )

            # Precision = inverse of mean absolute error
            # High error → low precision → weak predictions (let evidence through)
            # Low error → high precision → strong predictions (suppress expected input)
            new_precision = 1.0 / (node_state.error_mean_ema + 0.1)
            new_precision = new_precision.clamp(self.precision_min, self.precision_max)
            node_state.precision = torch.where(error_mask, new_precision, node_state.precision)

            # Output = prediction error with non-linear suppression gate.
            #
            # The key: when prediction (apical) closely matches evidence (basal),
            # the error node should go nearly SILENT, not just output a smaller number.
            #
            # suppression = apical / (|basal| + eps)  — how much of the evidence is explained
            # gate = 1 - sigmoid(k * (suppression - 0.5))  — sharp transition around 50% explained
            #   suppression < 0.3 → gate ≈ 1.0 (poor prediction, full error)
            #   suppression > 0.7 → gate ≈ 0.0 (good prediction, silenced)
            #
            # output = |error| * gate * pv_gain
            basal_mag = node_state.basal.abs() + 1e-6
            suppression = (node_state.apical / basal_mag).clamp(0.0, 2.0)
            gate = 1.0 - torch.sigmoid(8.0 * (suppression - 0.5))

            pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
            error_output = F.relu(pred_error.abs()) * gate * pv_gain * node_state.gain

            node_state.output = torch.where(error_mask, error_output, node_state.output)

        # =====================
        # REPRESENTATION NODES
        # =====================
        if repr_mask.any():
            repr_f = repr_mask.float()

            # Precision-weighted error input arrives via driving edges (basal)
            # The error nodes' output already includes precision weighting,
            # so the driving input here carries precision × error
            repr_error_input = inputs.basal * repr_f

            # Slow integration: representation persists (5x slower decay)
            repr_tau = self.node_cfg.basal_tau * 5.0
            repr_decay = -node_state.basal / repr_tau

            # Update representation with error signal
            lr = self.h_cfg.pc_learning_rate
            node_state.basal += dt * (repr_decay + lr * repr_error_input) * repr_f

            # Apical = context from higher levels (mostly 0 in 2-level system)
            node_state.apical += dt * (
                -node_state.apical / self.node_cfg.apical_tau + inputs.apical
            ) * repr_f

            # Output = current prediction state
            pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
            repr_output = F.softplus(node_state.basal) * pv_gain * node_state.gain

            node_state.output = torch.where(repr_mask, repr_output, node_state.output)

        # =====================
        # UNASSIGNED EXCITATORY (standard two-compartment)
        # =====================
        unassigned_exc = none_mask & exc_mask
        if unassigned_exc.any():
            ue_f = unassigned_exc.float()
            node_state.basal += dt * (
                -node_state.basal / self.node_cfg.basal_tau + inputs.basal
            ) * ue_f
            sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)
            node_state.apical += dt * (
                -node_state.apical / self.node_cfg.apical_tau
                + inputs.apical * (1.0 - sst_gate)
            ) * ue_f
            pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)
            std_out = F.softplus(node_state.basal) * self._g(node_state.apical) * pv_gain * node_state.gain
            node_state.output = torch.where(unassigned_exc, std_out, node_state.output)

        # =====================
        # INHIBITORY INTERNEURONS (PV, SST, VIP)
        # =====================
        for inh_type, inh_mask in [(NodeType.PV, pv_mask), (NodeType.SST, sst_mask), (NodeType.VIP, vip_mask)]:
            if not inh_mask.any():
                continue
            inh_f = inh_mask.float()
            inh_input = inputs.basal
            if inh_type == NodeType.PV:
                inh_input = inh_input + inputs.electrical

            node_state.basal += dt * (
                -node_state.basal / self.node_cfg.basal_tau + inh_input
            ) * inh_f

            inh_output = F.softplus(node_state.basal) * node_state.gain * inh_f
            if inh_type == NodeType.SST:
                vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
                inh_output = inh_output * vip_inhib

            node_state.output = torch.where(inh_mask, inh_output, node_state.output)

        # =====================
        # NOISE + CLAMP + TRACKING
        # =====================
        noise = torch.randn(node_state.n_nodes, device=device) * self.node_cfg.noise_std
        node_state.output += noise
        node_state.output.clamp_(min=0.0)

        spiking = node_state.output > node_state.threshold
        node_state.last_spike_time[spiking] = current_time

        alpha = dt / self.node_cfg.ip_tau
        node_state.activity_ema.lerp_(node_state.output, alpha)

    def _g(self, x: Tensor) -> Tensor:
        """Apical gating: sigmoid centered so g(0) ≈ 1."""
        slope = self.node_cfg.apical_slope
        base = torch.sigmoid(torch.tensor(0.0))
        return torch.sigmoid(slope * x + slope * self.node_cfg.apical_center) / base


class PCWeightUpdate:
    """PC-native weight update for modulatory (prediction) edges.

    Replaces STDP on inter-level modulatory connections.
    Δw = lr × precision_at_target × error_at_target × output_of_source

    Self-limiting: as predictions improve → error drops → updates stop.
    """

    def __init__(self, config: GraphBrainConfig):
        self.lr = config.hierarchy.pc_learning_rate
        self.w_min = config.edges.stdp.w_min
        self.w_max = config.edges.stdp.w_max

    def update(self, store: EdgeStore, node_state: NodeState) -> None:
        """Update modulatory edge weights using PC learning rule."""
        if store.n_edges == 0:
            return

        # Source = representation node output (the prediction signal)
        src_output = node_state.output[store.src.long()]

        # Target = error node's precision-weighted error
        dst_error = node_state.prediction_error[store.dst.long()]
        dst_precision = node_state.precision[store.dst.long()]

        # Weight update: strengthen edges that carry predictions to high-precision error nodes
        # Sign: if error > 0 (evidence > prediction), strengthen prediction
        #        if error < 0 (prediction > evidence), weaken prediction
        dw = self.lr * dst_precision * dst_error * src_output

        # Normalize by precision magnitude to prevent explosive updates
        dw = dw / (dst_precision.mean() + 1e-6)

        store.weight += dw * 0.01  # weight adaptation
        store.weight.clamp_(self.w_min, self.w_max)
