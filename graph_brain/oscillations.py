"""Oscillatory dynamics: endogenous PING gamma + theta modulation.

PING (Pyramidal-Interneuron Network Gamma):
    EXC active → drives PV (via local field coupling) → PV inhibits EXC →
    EXC recovers → cycle repeats. Produces gamma-band (~40-70Hz) oscillation.

    Critical discovery: no EXC→PV edge pathway exists in the type constraints.
    PV receives excitatory drive via LOCAL FIELD COUPLING — each PV node senses
    the mean output of nearby excitatory nodes as ambient drive.

Theta (4-8Hz):
    External sinusoidal modulation of excitatory basal inputs.
    Creates cross-frequency coupling: high theta → strong EXC → strong PING.

GABA decay:
    Leaky integration of PV inhibition with fast time constant (3ms).
    Replaces instantaneous pv_gain with temporal dynamics.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.types import EdgeType, NodeType


class PINGMechanism:
    """Endogenous gamma via Pyramidal-Interneuron Network Gamma."""

    def __init__(
        self,
        graph: NeuromorphicGraph,
        coupling_strength: float = 0.3,
        radius: float = 0.3,
        pv_tau: float = 5.0,
        gaba_tau: float = 3.0,
        inhib_boost: float = 3.0,
        field_radius: float = 0.3,
    ):
        self.pv_tau = pv_tau
        self.gaba_tau = gaba_tau
        self.device = graph.device
        ns = graph.node_state
        N = ns.n_nodes

        self.pv_idx = torch.where(ns.type_mask(NodeType.PV))[0]
        self.exc_idx = torch.where(ns.type_mask(NodeType.EXCITATORY))[0]

        # Per-node basal tau (PV gets fast tau, others keep 10.0)
        self.node_tau = torch.full((N,), 10.0, device=self.device)
        self.node_tau[self.pv_idx] = pv_tau

        # GABA inhibition state (leaky integration, not instantaneous)
        self.pv_inhib_state = torch.zeros(N, device=self.device)

        # Setup dense PV-PV gap junctions
        self._setup_gap_junctions(graph, coupling_strength, radius)

        # Boost PV→EXC inhibition strength
        self._boost_inhibition(graph, inhib_boost)

        # Pre-compute local field coupling: PV→nearby EXC mapping
        self._build_field_coupling(graph, field_radius)

    def _setup_gap_junctions(self, graph, coupling, radius):
        """Dense PV-PV gap junctions for gamma synchrony."""
        ns = graph.node_state
        pv_pos = ns.position[self.pv_idx]
        n_pv = self.pv_idx.shape[0]

        if n_pv < 2:
            return

        diff = pv_pos.unsqueeze(1) - pv_pos.unsqueeze(0)
        dist = (diff * diff).sum(dim=2).sqrt()
        connected = (dist < radius) & (dist > 0)

        # Ensure each PV has at least 5 neighbors
        for i in range(n_pv):
            if connected[i].sum() < 5:
                _, nearest = dist[i].topk(min(6, n_pv), largest=False)
                nearest = nearest[nearest != i][:5]
                connected[i, nearest] = True
                connected[nearest, i] = True

        # Remove existing electrical edges
        if graph.has_edge_type(EdgeType.ELECTRICAL):
            old = graph.edge_store(EdgeType.ELECTRICAL)
            graph.remove_edges(EdgeType.ELECTRICAL,
                               torch.ones(old.n_edges, dtype=torch.bool, device=self.device))

        si, di = torch.where(connected)
        sg = self.pv_idx[si].to(torch.int32)
        dg = self.pv_idx[di].to(torch.int32)
        if sg.shape[0] > 0:
            weights = torch.full((sg.shape[0],), coupling, device=self.device)
            graph.add_edges(EdgeType.ELECTRICAL, sg, dg, weights=weights)

    def _boost_inhibition(self, graph, boost_factor):
        """Strengthen PV→EXC inhibition for PING."""
        if graph.has_edge_type(EdgeType.INHIB_PERISOMATIC):
            store = graph.edge_store(EdgeType.INHIB_PERISOMATIC)
            store.weight *= boost_factor
            store.weight.clamp_(0.0, 1.0)

    def _build_field_coupling(self, graph, field_radius):
        """Pre-compute EXC→PV local field coupling.

        Each PV node senses the mean output of nearby excitatory nodes.
        This bridges the missing EXC→PV edge pathway.
        """
        ns = graph.node_state
        positions = ns.position
        n_pv = self.pv_idx.shape[0]

        # For each PV node, find nearby EXC nodes
        pv_pos = positions[self.pv_idx]
        exc_pos = positions[self.exc_idx]

        # All-pairs distance PV × EXC
        diff = pv_pos.unsqueeze(1) - exc_pos.unsqueeze(0)  # [n_pv, n_exc, 3]
        dist = (diff * diff).sum(dim=2).sqrt()  # [n_pv, n_exc]

        # Mask: which EXC nodes are within field_radius of each PV node
        self.field_mask = dist < field_radius  # [n_pv, n_exc] bool

        # Count neighbors for normalization
        self.field_counts = self.field_mask.float().sum(dim=1).clamp(min=1.0)  # [n_pv]

    def compute_field_drive(self, graph: NeuromorphicGraph) -> Tensor:
        """Compute local excitatory field at each PV node.

        Returns [N] tensor with field drive at PV positions, zero elsewhere.
        """
        ns = graph.node_state
        exc_output = ns.output[self.exc_idx]  # [n_exc]

        # For each PV node: mean output of nearby EXC nodes
        # field_mask is [n_pv, n_exc], exc_output is [n_exc]
        masked_output = exc_output.unsqueeze(0) * self.field_mask.float()  # [n_pv, n_exc]
        field_sum = masked_output.sum(dim=1)  # [n_pv]
        field_mean = field_sum / self.field_counts  # [n_pv]

        # Place into full [N] tensor
        drive = torch.zeros(ns.n_nodes, device=self.device)
        drive[self.pv_idx] = field_mean
        return drive

    def update_gaba(self, pv_inhibition: Tensor, dt: float = 1.0) -> Tensor:
        """Leaky integration of GABA (PV inhibition) with fast decay.

        Returns the smoothed pv_gain factor.
        """
        self.pv_inhib_state += dt * (
            -self.pv_inhib_state / self.gaba_tau + pv_inhibition
        )
        self.pv_inhib_state.clamp_(min=0.0)
        return torch.clamp(1.0 - self.pv_inhib_state, min=0.0, max=1.0)


class ThetaDrive:
    """External theta-band modulation of global excitability."""

    def __init__(self, frequency_hz: float = 6.0, amplitude: float = 0.5):
        self.frequency = frequency_hz
        self.amplitude = amplitude

    def get_modulation(self, step: int, dt: float = 1.0) -> float:
        """Scalar modulation factor for excitatory basal inputs."""
        phase = 2.0 * math.pi * self.frequency * step * dt / 1000.0
        return 1.0 + self.amplitude * math.sin(phase)

    def get_phase(self, step: int, dt: float = 1.0) -> float:
        """Current theta phase in [0, 2*pi]."""
        return (2.0 * math.pi * self.frequency * step * dt / 1000.0) % (2.0 * math.pi)


class OscillationAnalyzer:
    """Power spectral analysis and cross-frequency coupling."""

    def __init__(self, buffer_size: int = 2000, dt: float = 1.0):
        self.buffer_size = buffer_size
        self.dt = dt
        self.pv_history: list[float] = []
        self.exc_history: list[float] = []

    def record(self, pv_mean: float, exc_mean: float) -> None:
        self.pv_history.append(pv_mean)
        self.exc_history.append(exc_mean)
        if len(self.pv_history) > self.buffer_size:
            self.pv_history = self.pv_history[-self.buffer_size:]
            self.exc_history = self.exc_history[-self.buffer_size:]

    def compute_power_spectrum(self, signal: list[float]) -> tuple[Tensor, Tensor]:
        """Compute power spectrum via FFT. Returns (frequencies_hz, power)."""
        x = torch.tensor(signal, dtype=torch.float32)
        x = x - x.mean()  # remove DC

        n = len(x)
        fft_vals = torch.fft.rfft(x)
        power = (fft_vals.abs() ** 2) / n
        freqs = torch.fft.rfftfreq(n, d=self.dt / 1000.0)  # Hz

        return freqs, power

    def detect_gamma(self, signal: Optional[list[float]] = None) -> dict:
        """Detect gamma peak in PV output power spectrum."""
        sig = signal or self.pv_history
        if len(sig) < 100:
            return {"found": False, "frequency": 0, "snr": 0}

        freqs, power = self.compute_power_spectrum(sig)

        # Gamma band: 30-100 Hz
        gamma_mask = (freqs >= 30) & (freqs <= 100)
        if not gamma_mask.any():
            return {"found": False, "frequency": 0, "snr": 0}

        gamma_power = power[gamma_mask]
        gamma_freqs = freqs[gamma_mask]

        peak_idx = gamma_power.argmax()
        peak_freq = float(gamma_freqs[peak_idx])
        peak_power = float(gamma_power[peak_idx])
        mean_power = float(gamma_power.mean())
        snr = peak_power / (mean_power + 1e-10)
        snr_db = 10 * math.log10(snr + 1e-10)

        return {
            "found": snr_db > 3.0,
            "frequency": peak_freq,
            "snr_db": snr_db,
            "peak_power": peak_power,
        }

    def compute_phase_amplitude_coupling(
        self,
        slow_signal: list[float],
        fast_signal: list[float],
        n_bins: int = 18,
    ) -> float:
        """Modulation Index (Tort et al. 2010) for theta-gamma coupling.

        Returns MI value. Higher = stronger coupling.
        """
        if len(slow_signal) < 200 or len(fast_signal) < 200:
            return 0.0

        slow = torch.tensor(slow_signal, dtype=torch.float32)
        fast = torch.tensor(fast_signal, dtype=torch.float32)

        # Bandpass theta (4-8Hz) via FFT
        n = len(slow)
        freqs = torch.fft.rfftfreq(n, d=self.dt / 1000.0)

        # Theta phase
        slow_fft = torch.fft.rfft(slow - slow.mean())
        theta_mask = (freqs >= 4) & (freqs <= 8)
        slow_fft_filtered = slow_fft * theta_mask.float()
        theta_analytic = torch.fft.irfft(slow_fft_filtered, n=n)
        # Hilbert-like phase extraction
        theta_fft_h = slow_fft_filtered.clone()
        theta_fft_h[1:] *= 2  # analytic signal approximation
        theta_complex = torch.fft.irfft(theta_fft_h, n=n)
        theta_phase = torch.atan2(theta_complex, theta_analytic)

        # Gamma amplitude envelope
        fast_fft = torch.fft.rfft(fast - fast.mean())
        gamma_mask = (freqs >= 30) & (freqs <= 100)
        fast_fft_filtered = fast_fft * gamma_mask.float()
        gamma_filtered = torch.fft.irfft(fast_fft_filtered, n=n)
        gamma_amplitude = gamma_filtered.abs()

        # Phase-amplitude distribution
        bin_edges = torch.linspace(-math.pi, math.pi, n_bins + 1)
        mean_amp = torch.zeros(n_bins)
        for i in range(n_bins):
            mask = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
            if mask.any():
                mean_amp[i] = gamma_amplitude[mask].mean()

        # Modulation Index = KL divergence from uniform
        if mean_amp.sum() <= 0:
            return 0.0
        p = mean_amp / mean_amp.sum()
        uniform = torch.ones(n_bins) / n_bins
        kl = (p * torch.log((p + 1e-10) / uniform)).sum()
        mi = float(kl) / math.log(n_bins)

        return mi
