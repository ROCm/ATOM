# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Serving profiles admitted by the DeepSeek-V4 Mono backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V4MonoProfile:
    """One serving contract; request width and token rows stay distinct."""

    name: str
    tokens_per_seq: int
    auto_request_widths: tuple[int, ...]
    max_requests: int
    greedy_terminal: bool
    continuation: bool

    def accepts_graph(self, running_bs: int, max_seqlen_q: int, *, force: bool) -> bool:
        if max_seqlen_q != self.tokens_per_seq:
            return False
        if not 1 <= running_bs <= self.max_requests:
            return False
        return force or running_bs in self.auto_request_widths


ORDINARY_PROFILE = V4MonoProfile(
    "ordinary",
    tokens_per_seq=1,
    auto_request_widths=(1, 4, 8),
    max_requests=16,
    greedy_terminal=True,
    continuation=True,
)

MTP3_TARGET_PROFILE = V4MonoProfile(
    "mtp3_target",
    tokens_per_seq=4,
    auto_request_widths=(1,),
    # C2 is exposed only by force for the S8-vs-dual-S4 structure probe.
    # Auto remains C1 until that probe has positive end-to-end evidence.
    max_requests=2,
    greedy_terminal=False,
    continuation=True,
)
