# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/device/stamps.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.).
"""Timeline stamps: ``s_memrealtime`` (100 MHz) per CTA per point, kept in LDS
until the kernel ends (a global store per stamp would sit on every later
``s_waitcnt vmcnt(0)``). ``on`` is a compile-time switch: off, calls trace to nothing.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr

from .ops import traced


@traced
def stamp_begin(on, tls, tid, points):
    stamp(on, tls, tid, 0)


@traced
def stamp(on, tls, tid, k):
    pass


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    """The CTA's record -> ``tl`` [CTA][points] int64."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    tl + fx.Int64((bid * points + tid) * 8),
                ),
                fx.ptr_load(tls + tid),
            )
