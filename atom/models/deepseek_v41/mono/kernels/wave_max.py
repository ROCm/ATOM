# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The wave's max of an i32 order key (``atom.mono.device.select``'s keys) in
every lane, by DPP row rotations and the permlane swaps."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

from atom.mono.device.ops import xred


def wave_max_key(v):
    """The wave's max of the i32 key v, in every lane: the row's by rotations
    8, 4, 2, 1 (an integer max takes a DPP operand in its one instruction, and
    a rotation reads no lane outside the row: no lane keeps an old value),
    then the rows' by the permlane swaps. An exact max: the value
    ``butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)`` gives, in fewer
    instructions."""
    for r in (8, 4, 2, 1):
        v = fx.max(
            v, fx.Int32(update_dpp_i32(fx.Int32(0), v, 0x120 + r, 0xF, 0xF, True))
        )
    for off in (16, 32):
        v = xred(v, off, fx.max)
    return v
