"""Asynchronous weight updates on separate CUDA streams.

Learning rules (STDP, PC weight update, homeostatic scaling) read node state
but don't affect node dynamics within the same step. They can run concurrently
with the NEXT step's message passing and node updates.

Pipeline:
    Step N:  [message_pass + node_update]  [weight_updates from step N-1 complete]
    Step N+1: [message_pass + node_update]  [weight_updates from step N launch]

The weight updates read from a snapshot of node state taken at the end of
the node update. They write to edge weights, which are read by message
passing. Since message passing for step N+1 reads weights AFTER the step N
weight update completes (guaranteed by stream synchronization on the weight
stream before message passing reads), consistency is maintained.

At worst, weight updates use node state that's 1 step stale. For slow
learning rules (homeostatic: every 100 steps, structural: every 500 steps),
1-step staleness is irrelevant. Even for per-step rules (STP, PC weight),
the weight change per step is tiny, so staleness doesn't matter.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


class AsyncWeightUpdater:
    """Manages async weight updates on a separate CUDA stream.

    Usage:
        updater = AsyncWeightUpdater(device="cuda")

        # Each step:
        updater.sync()  # wait for previous weight updates to finish
        # ... do message passing + node update on default stream ...
        updater.launch(update_fn)  # launch weight updates on async stream
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._enabled = str(device) != "cpu" and torch.cuda.is_available()

        if self._enabled:
            self._stream = torch.cuda.Stream()
            self._event = torch.cuda.Event()
        else:
            self._stream = None
            self._event = None

        self._pending = False

    def sync(self) -> None:
        """Wait for any pending async weight updates to complete.

        Call this BEFORE message passing reads edge weights.
        """
        if self._enabled and self._pending:
            self._event.synchronize()
            self._pending = False

    def launch(self, update_fn: Callable[[], None]) -> None:
        """Launch weight updates on the async stream.

        Args:
            update_fn: callable that performs all weight updates.
                        Will execute on the async CUDA stream.
        """
        if self._enabled:
            # Record event on default stream so async stream waits
            # for node updates to finish
            default_event = torch.cuda.Event()
            default_event.record()

            with torch.cuda.stream(self._stream):
                # Wait for default stream to finish node updates
                self._stream.wait_event(default_event)
                # Run weight updates
                update_fn()
                # Record completion event
                self._event.record()
            self._pending = True
        else:
            # CPU: just run synchronously
            update_fn()

    @property
    def is_async(self) -> bool:
        return self._enabled
