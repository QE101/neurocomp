"""Two-compartment node model with typed inhibition circuits.

Excitatory nodes: two compartments (basal + apical) with gating.
Inhibitory nodes (PV, SST, VIP): single compartment.

The core computation per excitatory node:
    basal += dt * (-basal/tau_b + driving_input)
    apical += dt * (-apical/tau_a + modulatory_input * (1 - sst_gating))
    output = f(basal) * g(apical) * (1 - pv_inhibition) * gain + noise

Where:
    f() = softplus (basal activation)
    g() = sigmoid centered at 1.0 (apical gating — ungated when no top-down)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_brain.config import NodeConfig
from graph_brain.core.graph import NodeState
from graph_brain.core.message_passing import CompartmentInputs
from graph_brain.types import NodeType


class TwoCompartmentModel:
    """Vectorized two-compartment node dynamics for all node types."""

    def __init__(self, config: NodeConfig):
        self.cfg = config
        self.dt = config.dt

    def step(
        self,
        node_state: NodeState,
        inputs: CompartmentInputs,
        current_time: float,
    ) -> None:
        """Update all node states in-place for one timestep.

        Args:
            node_state: mutable NodeState (modified in-place)
            inputs: compartment inputs from message passing
            current_time: current simulation time (ms) for STDP timing
        """
        dt = self.dt
        device = node_state.device

        # --- Masks for node types ---
        exc_mask = node_state.type_mask(NodeType.EXCITATORY)
        pv_mask = node_state.type_mask(NodeType.PV)
        sst_mask = node_state.type_mask(NodeType.SST)
        vip_mask = node_state.type_mask(NodeType.VIP)

        # =====================
        # EXCITATORY NODES (two-compartment)
        # =====================

        # Basal compartment: leaky integration of driving input
        basal_decay = -node_state.basal / self.cfg.basal_tau
        basal_drive = inputs.basal
        node_state.basal += dt * (basal_decay + basal_drive) * exc_mask

        # Apical compartment: modulated by SST inhibition (dendritic gating)
        sst_gate = torch.sigmoid(inputs.sst_inhibition * 5.0)  # smooth 0-1 gate
        apical_decay = -node_state.apical / self.cfg.apical_tau
        apical_drive = inputs.apical * (1.0 - sst_gate)
        node_state.apical += dt * (apical_decay + apical_drive) * exc_mask

        # Output: f(basal) * g(apical) * pv_gain * gain + noise
        basal_act = self._f(node_state.basal)
        apical_gate = self._g(node_state.apical)
        pv_gain = torch.clamp(1.0 - inputs.pv_inhibition, min=0.0, max=1.0)

        exc_output = basal_act * apical_gate * pv_gain * node_state.gain

        # =====================
        # PV INTERNEURONS (single compartment + electrical coupling)
        # =====================
        pv_input = inputs.basal + inputs.electrical  # PV receives driving + gap junctions
        pv_decay = -node_state.basal / self.cfg.basal_tau
        node_state.basal += dt * (pv_decay + pv_input) * pv_mask
        pv_output = self._f(node_state.basal) * node_state.gain * pv_mask

        # =====================
        # SST INTERNEURONS (single compartment)
        # =====================
        # SST receives driving input but is inhibited by VIP
        sst_input = inputs.basal
        sst_vip_inhib = torch.clamp(1.0 - inputs.sst_inhibition, min=0.0, max=1.0)
        sst_decay = -node_state.basal / self.cfg.basal_tau
        node_state.basal += dt * (sst_decay + sst_input) * sst_mask
        sst_output = self._f(node_state.basal) * node_state.gain * sst_vip_inhib * sst_mask

        # =====================
        # VIP INTERNEURONS (single compartment)
        # =====================
        vip_input = inputs.basal
        vip_decay = -node_state.basal / self.cfg.basal_tau
        node_state.basal += dt * (vip_decay + vip_input) * vip_mask
        vip_output = self._f(node_state.basal) * node_state.gain * vip_mask

        # =====================
        # COMBINE OUTPUTS + NOISE
        # =====================
        noise = torch.randn(node_state.n_nodes, device=device) * self.cfg.noise_std
        node_state.output = exc_output + pv_output + sst_output + vip_output + noise

        # Clamp output to non-negative (firing rates can't be negative)
        node_state.output.clamp_(min=0.0)

        # Update spike timing for STDP (threshold crossing)
        spiking = node_state.output > node_state.threshold
        node_state.last_spike_time[spiking] = current_time

        # Update activity EMA for homeostasis
        alpha = self.dt / self.cfg.ip_tau
        node_state.activity_ema.lerp_(node_state.output, alpha)

    def _f(self, x: Tensor) -> Tensor:
        """Basal activation function."""
        if self.cfg.basal_activation == "softplus":
            return F.softplus(x)
        else:  # relu
            return F.relu(x)

    def _g(self, x: Tensor) -> Tensor:
        """Apical gating function: sigmoid centered at apical_center.

        g(0) ≈ 1.0 when apical_center = 1.0, meaning ungated by default.
        Positive apical input amplifies (g > 1), negative suppresses (g < 1).

        Specifically: g(x) = 2 * sigmoid(slope * (x + center)) so that
        g(0) = 2 * sigmoid(slope * center) ≈ 1.0 for center ≈ 0.55/slope.

        Simplified: g(x) = sigmoid(slope * x) / sigmoid(0) = normalized sigmoid
        where g(0) = 1 exactly.
        """
        slope = self.cfg.apical_slope
        base = torch.sigmoid(torch.tensor(0.0))  # = 0.5
        return torch.sigmoid(slope * x + slope * self.cfg.apical_center) / base
