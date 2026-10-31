# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``layout`` on gfx1250: the same scratch / symmetric regions, mailbox table
and stage placement, with the counts that follow from 16 waves a CTA.

A router task holds one expert a wave (16), so a token takes N_ROUTER = 8 tasks;
the gfx950 scratch regions sized by its 16 tasks a token hold them. The split's
PV partials are two key halves (``kernels.stages.attention``).
"""

from atom.models.minimax_m3.mono.config import MOE_SLOTS, N_ROUTED
from atom.models.minimax_m3.mono.layout import (  # noqa: F401
    AG_RS_FROM,
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_O,
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
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)
from atom.mono.device.gfx1250 import BLOCKS, WAVE, WAVES

N_ROUTER = N_ROUTED // WAVES  # one expert a wave
PV_HALVES = 2  # the split's PV: a wave's 16 dims over one half of the keys
ACC_WORDS = WAVES * WAVE * 8  # every wave's 16x16 f32 accumulator


def stage_bases(tokens: int):
    """``layout.stage_bases`` with this N_ROUTER."""
    n_split = N_SPLIT * tokens
    n_merge = tokens if tokens > 1 else 0
    router = (n_split + n_merge + N_O) % BLOCKS
    return {
        "split": 0,
        "merge": n_split,
        "o": n_split + n_merge,
        "router": router,
        "shared": _shared_base(tokens, router),
        "ug": 0,
        "down": n_split,
    }


def _shared_base(tokens: int, router: int) -> int:
    n_router = N_ROUTER * tokens
    n_split = N_SPLIT * tokens
    most = max(ug_tasks_of(c, tokens) for c in range(BLOCKS))

    def cost(base):
        ctas = [(base + j) % BLOCKS for j in range(SH_TASKS)]
        busiest = max(ug_tasks_of(c, tokens) for c in ctas)
        in_router = sum((c - router) % BLOCKS < n_router for c in ctas)
        in_down = sum(n_split <= c < n_split + N_DN for c in ctas)
        return busiest, in_router if busiest == most else 0, in_down, in_router

    return min(range(BLOCKS), key=cost)


def shared_after_routing(tokens: int) -> bool:
    loads = [ug_tasks_of(c, tokens) for c in range(BLOCKS)]
    return min(loads) == max(loads)


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
