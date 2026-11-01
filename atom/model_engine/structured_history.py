# SPDX-License-Identifier: MIT
"""A bounded view of the scheduler's append-only committed output prefix."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CommittedTokens(Sequence):
    # Placeholder slots beyond length may be changed by deferred delivery.
    # Earlier committed tokens are immutable until a prefill/replay, which
    # explicitly invalidates the worker's incremental grammar state.
    source: Sequence
    length: int

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.source[i] for i in range(*key.indices(self.length))]
        if key < 0:
            key += self.length
        if not 0 <= key < self.length:
            raise IndexError(key)
        return self.source[key]
