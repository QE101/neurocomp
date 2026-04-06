"""Circular delay buffer for conduction delays.

Messages don't arrive instantly — they enter the buffer at
current_step + delay_steps and are read when that step arrives.

Each compartment has its own buffer channel. The buffer is a dense
tensor of shape [max_delay + 1, N, n_channels] that wraps around
via modular indexing.

Memory cost: max_delay * N * n_channels * 4 bytes
At N=5000, max_delay=6, 6 channels: 5000 * 6 * 6 * 4 = 720 KB. Negligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class Channel(IntEnum):
    """Delay buffer channels — one per compartment input type."""
    BASAL = 0
    APICAL = 1
    PV_INHIBITION = 2
    SST_INHIBITION = 3
    VIP_INHIBITION = 4   # VIP→SST disinhibition channel
    ELECTRICAL = 5
    RETROGRADE = 6


N_CHANNELS = len(Channel)


class DelayBuffer:
    """Circular buffer for delayed message delivery.

    Usage:
        buf = DelayBuffer(n_nodes=5000, max_delay_steps=6, device="cuda")

        # At each step, the message passer writes delayed messages:
        buf.write(channel=Channel.BASAL, dst_indices, values, delay_steps, current_step)

        # Then the simulator reads what's arrived:
        inputs = buf.read(current_step)  # returns [N, n_channels]
    """

    def __init__(self, n_nodes: int, max_delay_steps: int, device: str = "cpu"):
        self.n_nodes = n_nodes
        self.max_delay_steps = max_delay_steps
        self.device = device
        self.buf_len = max_delay_steps + 1  # +1 so we can write to step+max while reading step

        # Buffer: [n_channels, buf_len, N] — channel-first for contiguous channel slicing
        self._buffer = torch.zeros(
            N_CHANNELS, self.buf_len, n_nodes,
            device=device,
        )

    def write(
        self,
        channel: int,
        dst_indices: Tensor,
        values: Tensor,
        delay_steps: Tensor,
        current_step: int,
    ) -> None:
        """Write delayed messages into future buffer slots.

        Uses a single flat scatter operation — no Python loop over delay values.

        Args:
            channel: which compartment channel (Channel enum)
            dst_indices: [E] destination node indices
            values: [E] message values to deliver
            delay_steps: [E] delay in timesteps per edge (int)
            current_step: current simulation step
        """
        if dst_indices.numel() == 0:
            return

        # Target timestep for each message
        target_steps = current_step + delay_steps.long()
        # Modular index into circular buffer
        buf_indices = target_steps % self.buf_len

        # Flatten to 1D scatter: flat_idx = buf_idx * n_nodes + dst_node
        # Then scatter into a flattened view of buffer[:, :, channel]
        flat_idx = buf_indices * self.n_nodes + dst_indices.long()
        # Channel-first layout: _buffer[channel, :, :] is contiguous
        # So _buffer[channel].reshape(-1) is always a zero-copy view
        flat_buf = self._buffer[channel].reshape(-1)
        flat_buf.index_add_(0, flat_idx, values)

    def write_immediate(
        self,
        channel: int,
        dst_indices: Tensor,
        values: Tensor,
        current_step: int,
    ) -> None:
        """Write messages that arrive THIS step (delay=0, for gap junctions etc)."""
        if dst_indices.numel() == 0:
            return
        buf_idx = current_step % self.buf_len
        self._buffer[channel, buf_idx].index_add_(0, dst_indices.long(), values)

    def read(self, current_step: int) -> Tensor:
        """Read all messages that have arrived at current_step.

        Returns: [N, n_channels] tensor of accumulated inputs.
        Clears the slot after reading.
        """
        buf_idx = current_step % self.buf_len
        # Gather from channel-first layout: [n_channels, buf_len, N] → [N, n_channels]
        result = self._buffer[:, buf_idx, :].T.clone()  # [N, n_channels]
        # Clear the slot for reuse
        self._buffer[:, buf_idx, :].zero_()
        return result

    def reset(self) -> None:
        """Clear all buffered messages."""
        self._buffer.zero_()

    def to(self, device: str) -> DelayBuffer:
        """Move buffer to device."""
        self._buffer = self._buffer.to(device)
        self.device = device
        return self
