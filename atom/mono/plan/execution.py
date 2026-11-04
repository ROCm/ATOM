# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx950 execution model the existing mono kernels are written for.

One CTA per CU of an MI355X, all of them co-resident -- a spin wait on another
CTA can only make progress if that CTA is running, so the runner refuses a GPU
with another CU count -- and THREADS threads each.

Architecture topology lives separately in ``plan.architecture``.  Keeping this
module explicitly pinned to ``GFX950`` is deliberate: a wave32 primitive alone
does not validate the grid residency, LDS allocation or stage placement needed
to turn an existing persistent kernel into a gfx1250 kernel.
"""

from atom.mono.plan.architecture import GFX950

ARCHITECTURE = GFX950
BLOCKS = 256
THREADS = 512
WAVE_SIZE = ARCHITECTURE.wave_size
WAVES = THREADS // WAVE_SIZE
LDS_BYTES = 160 * 1024  # a CU's LDS: one CTA's whole


def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base`` (any count of
    CTAs past 0: placements that add task counts run past the grid), then every
    BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS
