# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A kernel's mailbox regions laid out in its scratch or peer buffer."""

PAIR_BYTES = 8  # a (value, tag) pair
# a debug build's wait record, one a CTA: flag, region id, pair index, tag,
# seen tag, CTA, clock, pad (``atom.mono.device.sync._spin_bounded``)
DIAG_WORDS = 8


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


def byte_layout(items, align: int = 16, start: int = 0) -> dict:
    """``items`` (name, bytes) of plain words laid out in order: name -> (byte
    offset, bytes), each region from an ``align``-byte boundary at or past
    ``start``.

    Plain words never share a buffer with tagged pairs: one launch's stale
    plain word in a later launch's tag position can equal that launch's tag,
    and its poll takes the pair as landed."""
    out, off = {}, start
    for name, nbytes in items:
        off = (off + align - 1) // align * align
        out[name] = (off, nbytes)
        off += nbytes
    return out
