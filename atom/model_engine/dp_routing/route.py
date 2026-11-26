# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DP placement's accounting and stable snapshot counter names."""

from typing import NamedTuple

PREFIX_OWNER = "prefix_owner_total"
PREFIX_MOVED = "prefix_moved_total"
PREFIX_COLD = "prefix_cold_total"
PREFIX_DECISIONS = (PREFIX_OWNER, PREFIX_MOVED, PREFIX_COLD)
PREFIX_MOVED_TOKENS = "prefix_moved_tokens"
PREFIX_LABELS = (
    (PREFIX_OWNER, "owner"),
    (PREFIX_MOVED, "moved"),
    (PREFIX_COLD, "cold"),
)


class Route(NamedTuple):
    rank: int
    decision: str
    credit: int = 0
    moved_tokens: int = 0
