# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""RED, a CTA's GEMV accumulators in LDS (``attn_pre.gemv_fp8_mfmas``): each
wave's 16x16 MFMA C, lane 16 g + col holding rows 4 g .. 4 g + 3 of token
column col, and the reduction over the waves that reads them back a (row,
token) a thread.

A reduction's lanes walk the rows down a lane group and the tokens across
the groups; in the C layout those rows sit a lane group (a multiple of the
banks' words) apart, so they would read one bank. The column is swizzled
within its group by the group: the same values, spread over the banks."""

import flydsl.expr as fx

from atom.mono.plan.execution import WAVES


def acc_word(w, g, col):
    """RED's word of wave ``w``'s lane (g, col) C quad."""
    return (w * 64 + g * 16 + (col ^ (4 * g))) * 4


def acc_sum(red, r, t, waves=None):
    """Row r, token t of the waves' accumulators at ``red``, summed in wave
    order from 0 (``waves``: which, in order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + (acc_word(w, r // 4, t) + r % 4))
    return tot
