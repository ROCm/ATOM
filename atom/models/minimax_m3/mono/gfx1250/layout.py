# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``layout`` on gfx1250: the same scratch / symmetric regions, mailbox table
and stage placement, with the counts that follow from 16 waves a CTA.

A router task is gfx950's 8 experts, each on two waves (a K half each), so the
task count and placement are gfx950's. The split's PV partials are two key
halves (``kernels.stages.attention``).
"""

from atom.models.minimax_m3.mono.config import MOE_SLOTS, N_ROUTED
from atom.models.minimax_m3.mono.layout import (  # noqa: F401
    AG_RS_FROM,
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_O,
    N_ROUTER,
    N_SPLIT,
    O_K,
    O_ROW,
    O_ROWS,
    PAGES_PER_BLOCK,
    SCORE_COPIES,
    SCRATCH,
    SCRATCH_BYTES,
    SH_ROWS,
    SH_TASKS,
    SPARSE_TABLE_WORDS,
    SPLIT_KEYS,
    UG_PER_SLOT,
    UG_ROWS,
    WIDE_FROM,
    XN8_ROW,
    XSC_ROW,
    H,
    diag_region_names,
    mailbox_regions,
    shared_after_routing,
    stage_bases,
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)
from atom.mono.device.gfx1250 import WAVE, WAVES

PV_HALVES = 2  # the split's PV: a wave's 16 dims over one half of the keys
ACC_WORDS = WAVES * WAVE * 8  # every wave's 16x16 f32 accumulator
assert N_ROUTED // N_ROUTER * 2 == WAVES  # a router task: two waves an expert


def pool_words(tokens: int) -> int:
    """``layout.pool_words`` with gfx1250's partial sizes."""
    from atom.models.minimax_m3.mono.config import HIDDEN

    if not wide_moe(tokens):
        return HIDDEN // 2 * tokens
    rows = MOE_SLOTS * tokens + 1
    return max(
        PV_HALVES * O_K,
        tokens * (XN8_ROW + XSC_ROW + N_ROUTED) + ACC_WORDS,
        rows * (MID_ROW + MIDSC_ROW),
    )
