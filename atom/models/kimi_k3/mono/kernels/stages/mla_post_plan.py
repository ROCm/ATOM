# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MLA post stage's geometry in plain integers (``stages.mla_post`` is the
kernel): which core columns, weight rows and x slabs each task takes, and where
the tasks sit. Free of FlyDSL so it is testable on a CPU runner."""

from atom.mono.plan.execution import WAVES

ROWS = 16  # weight rows a task (gemv.ROWS)
KCH = 128  # K a chunk (gemv.KCH)
HIDDEN = 7168  # g_proj K
PROJ = 1536  # core width a rank: 12 heads x 128
HEADS = 12
LORA = 512  # kv_lora_rank: v-up K
VH = 128  # v_head_dim
PT = PROJ // ROWS  # 96 post tasks
GROUP_COLS = WAVES * KCH  # 1024: one chunk a wave (gemv.KGROUP)
G_GROUPS = HIDDEN // GROUP_COLS  # 7
V_CH = LORA // KCH  # 4 v-up chunks: waves 0 .. 3
XPAD = 16  # bytes a padded fp8 x row
MAX_ROWS = 16  # the widest step built with the stage (wider: the unfused path)

# Every K2 CTA runs o_proj, which reads every core column, so the 96 tasks gate
# all of it; they sit where the least other work is queued behind them: the
# 64 CTAs with one o_proj task (the AttnRes CTAs 0 .. 55 and 248 .. 255), the
# two-task CTAs with no early GEMV (152 .. 159, 244 .. 247) and the last 20
# latent parts (224 .. 243), whose weights are issued after their o_proj.
# ``(first CTA, first task, count)`` a range.
CTA_RANGES = ((0, 0, 56), (152, 56, 8), (224, 64, 32))


def scratch_pairs() -> dict:
    return {"mpdone": PT}


def task_cta(j: int) -> int:
    """The CTA task ``j`` runs on."""
    for cta0, j0, n in CTA_RANGES:
        if j0 <= j < j0 + n:
            return cta0 + j - j0
    raise ValueError(f"MLA post: no task {j}")


def cta_task(bid: int) -> int:
    """The task CTA ``bid`` runs, or -1 (``mla_post.task_of`` traced)."""
    for cta0, j0, n in CTA_RANGES:
        if cta0 <= bid < cta0 + n:
            return j0 + bid - cta0
    return -1


def task_rows(j: int) -> tuple[int, int, int]:
    """Task ``j``'s ``(g_proj / core row0, head, first v row of that head)``:
    core columns ``row0 .. row0 + 15`` are g_proj rows ``row0 ..`` and rows
    ``vr0 ..`` of head ``h``'s W_V."""
    return j * ROWS, j // (VH // ROWS), (j % (VH // ROWS)) * ROWS


def slabs(s: int, budget: int) -> list:
    """``[(g0, g1)]`` 1024-column groups of x staged together (fp8, ``XPAD`` a
    row) in ``budget`` bytes of LDS: one slab whenever S x 7168 fits."""
    for n in range(1, G_GROUPS + 1):
        per = -(-G_GROUPS // n)
        if s * (per * GROUP_COLS + XPAD) <= budget:
            out, g = [], 0
            for i in range(n):
                take = -(-(G_GROUPS - g) // (n - i))
                out.append((g, g + take))
                g += take
            return out
    raise AssertionError(f"MLA post: {s} rows x {GROUP_COLS} B exceed {budget} B")


def lds_words(s: int, budget: int) -> int:
    """The post stage's LDS (int32 words): the widest x slab, and at least the
    per-task av tile (16 x 16 words)."""
    w = max(g1 - g0 for g0, g1 in slabs(s, budget)) * GROUP_COLS
    return max(s * (w + XPAD) // 4, ROWS * ROWS)


def w8_dword(row0, lane, ldk, kcol):
    """The dword where lane ``lane``'s 8 fp8 of weight rows ``row0`` .. + 16
    (``row0`` a multiple of 16) start, at K column ``kcol`` (a multiple of 32) +
    its lane group's 8: AITER's (16, 16) preshuffle of a ``[N, ldk]`` E4M3
    plane. The next q step (32 columns on) is 128 dwords further. Plain
    integer arithmetic: the kernel calls it on traced values."""
    g = lane // ROWS
    byte = (
        ((row0 // ROWS * (ldk // 32) + kcol // 32) * 2 + g // 2) * 256
        + lane % ROWS * 16
        + g % 2 * 8
    )
    return byte // 4


def shuffle16(w):
    """A ``[N, K]`` (N % 16 == 0, K % 32 == 0) byte plane in AITER's (16, 16)
    preshuffle, as ``w8_dword`` reads it: a torch-only reference."""
    n, k = w.shape
    return (
        w.reshape(n // 16, 16, k // 32, 2, 2, 8).permute(0, 2, 3, 1, 4, 5).reshape(n, k)
    )
