# SPDX-License-Identifier: MIT
"""Compact, lossless transport of a replay frontier from TP row owners."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailOwnership:
    local_rows: np.ndarray
    gathered_rows: np.ndarray
    owner_capacity: int
    packet_rows: int


def plan_tail_ownership(rows, original_rows, tp_size):
    """Map the replay plan's global rows to equally sized owner packets.

    Only selected rows occupy packets. An owner with no selected row sends
    zero padding; the reconstruction never selects that padding. The order of
    the replay frontier is preserved, including requests crossing owners.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if original_rows <= 0 or tp_size <= 0 or rows.ndim != 1:
        raise ValueError("Invalid owner geometry or replay frontier")
    if np.any(rows < 0) or np.any(rows >= original_rows):
        raise ValueError("Replay frontier contains a padding or out-of-range row")
    capacity = (original_rows + tp_size - 1) // tp_size
    owners = rows // capacity
    counts = np.bincount(owners, minlength=tp_size)
    packet_rows = int(counts.max())
    local = np.full((tp_size, packet_rows), -1, dtype=np.int32)
    gathered = np.empty(len(rows), dtype=np.int32)
    for rank in range(tp_size):
        selected = np.flatnonzero(owners == rank)
        n = len(selected)
        local[rank, :n] = rows[selected] - rank * capacity
        gathered[selected] = rank * packet_rows + np.arange(n)
    return TailOwnership(local, gathered, capacity, packet_rows)
