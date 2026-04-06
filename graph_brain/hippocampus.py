"""Hippocampal fast encoding + sleep replay consolidation.

Three components mirroring biological hippocampal circuitry:
- DentateGyrus: pattern separation via fixed sparse random projection + k-WTA
- CA3Memory: one-shot auto-associative storage via outer-product Hebbian
- HippocampalSystem: facade orchestrating encode() and replay()

The hippocampus is a SEPARATE tensor buffer, not part of the cortical graph.
It receives cortical patterns as input and injects replay patterns as output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from graph_brain.config import HippocampalConfig


class DentateGyrus:
    """Pattern separation: sparse random projection + k-winners-take-all.

    Transforms dense cortical activation into sparse, non-overlapping DG codes.
    Projection is fixed (not learned) — biological mossy fibers are non-plastic.
    With 2% sparsity and 2000 DG nodes, two random cortical patterns produce
    near-orthogonal DG codes (collision probability negligible).
    """

    def __init__(self, n_dg: int, n_cortical: int, sparsity: float,
                 fan_in: int, device: str, generator: torch.Generator):
        self.n_dg = n_dg
        self.n_cortical = n_cortical
        self.sparsity = sparsity
        self.k = max(1, int(n_dg * sparsity))  # number of winners

        # Sparse random projection: each DG node samples fan_in cortical inputs
        fan_in = min(fan_in, n_cortical)
        # Indices: [n_dg, fan_in] — which cortical nodes each DG node reads from
        self.proj_indices = torch.stack([
            torch.randperm(n_cortical, generator=generator, device='cpu')[:fan_in]
            for _ in range(n_dg)
        ]).to(device)  # [n_dg, fan_in]

        # Weights: fixed random, normalized per DG node
        self.proj_weights = torch.randn(n_dg, fan_in, device=device, generator=None)
        self.proj_weights /= self.proj_weights.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def separate(self, cortical_pattern: Tensor) -> Tensor:
        """Transform dense cortical activation to sparse DG code.

        Args:
            cortical_pattern: [n_cortical] dense activation

        Returns:
            [n_dg] sparse activation (only k entries nonzero)
        """
        # Gather cortical activations at projection indices
        gathered = cortical_pattern[self.proj_indices]  # [n_dg, fan_in]
        # Dot product with projection weights
        raw = (gathered * self.proj_weights).sum(dim=1)  # [n_dg]
        # k-winners-take-all
        return self._kwta(raw)

    def _kwta(self, x: Tensor) -> Tensor:
        """Keep top-k values, zero the rest."""
        topk_vals, topk_idx = x.topk(self.k)
        out = torch.zeros_like(x)
        out[topk_idx] = topk_vals.clamp(min=0.0)
        return out


class CA3Memory:
    """Auto-associative memory via recurrent Hebbian weights.

    One-shot encoding: a single outer-product update stores a sparse pattern.
    Pattern completion: partial cue → recurrent dynamics → full pattern.
    Capacity: ~0.14 * n_ca3 / (sparsity * log(1/sparsity)) patterns.
    """

    def __init__(self, n_ca3: int, sparsity: float, encoding_lr: float, device: str):
        self.n_ca3 = n_ca3
        self.sparsity = sparsity
        self.encoding_lr = encoding_lr
        self.k = max(1, int(n_ca3 * sparsity))

        # Recurrent weight matrix (the memory)
        self.weights = torch.zeros(n_ca3, n_ca3, device=device)

        # DG-to-CA3 projection (fixed random, like mossy fibers)
        self.dg_to_ca3 = None  # initialized when DG size is known

    def init_dg_projection(self, n_dg: int, device: str, generator: torch.Generator):
        """Initialize the DG → CA3 random projection."""
        self.dg_to_ca3 = torch.randn(self.n_ca3, n_dg, device=device)
        self.dg_to_ca3 /= self.dg_to_ca3.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def encode(self, dg_code: Tensor) -> Tensor:
        """One-shot encoding of a DG code into CA3.

        Args:
            dg_code: [n_dg] sparse DG activation

        Returns:
            ca3_code: [n_ca3] sparse CA3 activation
        """
        # Project DG to CA3
        raw = self.dg_to_ca3 @ dg_code  # [n_ca3]
        ca3_code = self._kwta(raw)

        # One-shot Hebbian: outer product with Oja-like stabilization
        outer = ca3_code.unsqueeze(1) * ca3_code.unsqueeze(0)  # [n_ca3, n_ca3]
        dw = self.encoding_lr * (outer - self.weights * outer)
        self.weights += dw
        self.weights.clamp_(0.0, 1.0)
        # Zero diagonal (no self-connections)
        self.weights.fill_diagonal_(0.0)

        return ca3_code

    def complete(self, partial_ca3: Tensor, steps: int = 3) -> Tensor:
        """Pattern completion from partial cue via recurrent dynamics."""
        x = partial_ca3.clone()
        for _ in range(steps):
            raw = self.weights @ x
            x = self._kwta(raw)
        return x

    def _kwta(self, x: Tensor) -> Tensor:
        topk_vals, topk_idx = x.topk(self.k)
        out = torch.zeros_like(x)
        out[topk_idx] = topk_vals.clamp(min=0.0)
        return out


@dataclass
class StoredPattern:
    """A pattern stored in hippocampal memory."""
    cortical_pattern: Tensor   # [n_cortical] — the original cortical snapshot
    dg_code: Tensor            # [n_dg] — sparse DG representation
    ca3_code: Tensor           # [n_ca3] — sparse CA3 representation
    encode_step: int           # when it was encoded
    replay_count: int = 0      # times replayed


class HippocampalSystem:
    """Facade orchestrating DG + CA3 for encode/replay.

    Usage:
        hipp = HippocampalSystem(config, cortical_input_indices, device)
        hipp.encode(cortical_output)   # snapshot → DG → CA3 → store
        hipp.replay(inject_fn, lr_scale)  # retrieve → project → inject to cortex
    """

    def __init__(self, config: HippocampalConfig, cortical_input_indices: Tensor,
                 n_cortical: int, device: str, seed: int = 42):
        self.config = config
        self.cortical_input_indices = cortical_input_indices  # which cortical nodes are inputs
        self.n_cortical_input = cortical_input_indices.shape[0]
        self.n_cortical = n_cortical
        self.device = device

        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed + 1000)  # different seed from cortical graph

        # Build DG
        self.dg = DentateGyrus(
            n_dg=config.n_dg,
            n_cortical=self.n_cortical_input,
            sparsity=config.dg_sparsity,
            fan_in=min(config.dg_fan_in, self.n_cortical_input),
            device=device,
            generator=gen,
        )

        # Build CA3
        self.ca3 = CA3Memory(
            n_ca3=config.n_ca3,
            sparsity=config.ca3_sparsity,
            encoding_lr=config.encoding_lr,
            device=device,
        )
        self.ca3.init_dg_projection(config.n_dg, device, gen)

        # Readout: CA3 → cortical space (fixed random projection)
        self.ca3_to_cortical = torch.randn(self.n_cortical_input, config.n_ca3, device=device)
        self.ca3_to_cortical /= self.ca3_to_cortical.norm(dim=1, keepdim=True).clamp(min=1e-6)

        # Pattern buffer
        self.patterns: list[StoredPattern] = []

    def encode(self, cortical_output: Tensor, step: int) -> StoredPattern:
        """Encode current cortical activation into hippocampal memory.

        Args:
            cortical_output: [N] full cortical output tensor
            step: current simulation step

        Returns:
            The stored pattern
        """
        # Snapshot cortical input region
        cortical_snapshot = cortical_output[self.cortical_input_indices].detach().clone()

        # DG pattern separation
        dg_code = self.dg.separate(cortical_snapshot)

        # CA3 one-shot encoding
        ca3_code = self.ca3.encode(dg_code)

        # Store
        pattern = StoredPattern(
            cortical_pattern=cortical_snapshot,
            dg_code=dg_code,
            ca3_code=ca3_code,
            encode_step=step,
        )

        if len(self.patterns) >= self.config.max_patterns:
            # Overwrite least-replayed pattern
            min_idx = min(range(len(self.patterns)),
                          key=lambda i: self.patterns[i].replay_count)
            self.patterns[min_idx] = pattern
        else:
            self.patterns.append(pattern)

        return pattern

    def get_replay_pattern(self, idx: int) -> Tensor:
        """Get a cortical-space replay pattern for injection.

        Uses the stored cortical snapshot directly (not CA3 reconstruction)
        for maximum fidelity. CA3 serves as the addressing system.

        Args:
            idx: index into pattern buffer

        Returns:
            [n_cortical_input] replay activation for injection into cortex
        """
        pattern = self.patterns[idx]
        pattern.replay_count += 1
        return pattern.cortical_pattern * self.config.replay_strength

    def n_stored(self) -> int:
        return len(self.patterns)

    def replay_schedule(self, n_cycles: int) -> list[int]:
        """Generate interleaved replay schedule.

        Returns list of pattern indices, interleaved so no pattern repeats
        consecutively. Prioritizes recently encoded and under-replayed patterns.
        """
        if not self.patterns:
            return []

        n = len(self.patterns)
        schedule = []
        for _ in range(n_cycles):
            # Shuffle pattern order each cycle for interleaving
            order = torch.randperm(n).tolist()
            schedule.extend(order)
        return schedule
