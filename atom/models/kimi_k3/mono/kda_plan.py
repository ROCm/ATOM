# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Where K1 (``kernels.attention.kda``) reads ``f_b_proj`` when it is a
per-token FP8 plane: E4M3 in AITER's (16, 16) preshuffle
(``atom.mono.plan.preshuffle``), ``HEAD_DIM`` columns a row.

Kept out of ``kda``, which imports FlyDSL, so that a CPU test can check these
against the preshuffle written out. As ``atom.mono.plan.arith``: the kernel
calls them on traced values, the test on ints.
"""

from atom.models.kimi_k3.mono.config import HEAD_DIM
from atom.mono.plan.preshuffle import w8_offset


def fb_w8_offset(h, wave, lane):
    """The dword of the lane's first f_b operand in the FP8 plane: wave w's rows
    16 w .. of head h, row lane % 16 of them, K columns 32 g .. (g = lane / 16)."""
    return (
        w8_offset(h * HEAD_DIM + wave * 16, lane % 16, lane // 16 * 32, HEAD_DIM) // 4
    )


def fb_w8_dword(at, q):
    """The dword of f_b operand q (K columns 32 g + 8 q of the lane's row) in
    the FP8 plane, ``at`` the lane's first (``fb_w8_offset``): 16 B half q / 2
    of its 32-column K block, 8 B q % 2 into it."""
    return at + w8_offset(0, 0, 16 * (q // 2), HEAD_DIM) // 4 + q % 2 * 2
