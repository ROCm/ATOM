# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/attention/back.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K2a: one Kimi-K3 layer (KDA, or MLA past its attention) from o_proj's
input to the MoE's input row -- o_proj, its TP all-reduce inside the kernel,
and the MLP AttnRes seam.

Stages ``layer.build`` places in K2's launch (its CTA map: ``layer``):

    gemv              o_proj (448 x 16 rows, K = 1536 local): GEMV (``w8``: FP8) of the
                      core rows (plain loads: the previous launch wrote them) ->
                      bf16 partial -> pushed to its owner rank's AR region (system
                      scope, this rank's slot)
    attn              per 128 columns: the 8 ranks' partials summed in rank
                      order (fp32 -> bf16: the custom all-reduce's result) ->
                      LDS delta; then K1's AttnRes stages with the MLP's weights:
                      prefix + delta -> prefix out, the sources' softmax mix,
                      post_attention_layernorm -> the MoE input x

No rank can overwrite a peer's unread partials: to push layer L's partials
a rank must have passed layer L - 1's reduce, which needed every peer's L - 1
push, and the mailbox tag tells the two layers apart. Between steps the region
is zeroed and peer-fenced (``StepMailboxes.begin_step``).
"""

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels import token_groups
from atom.models.kimi_k3.mono.kernels.attention.kda import (
    COLS,
    HIDDEN,
    LDS_PAD,
    NA,
    PROJ,
    ROWS,
    TG,
    groups,
    seam_moves,
)
from atom.mono.device import gemv
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    mfma_bf16,
    rsrc,
    traced,
)
from atom.mono.device.ranks import sum_partials
from atom.mono.plan.execution import THREADS, WAVES
from atom.mono.plan.layout import PAIR_BYTES, pair_layout

TP = 8
OROWS = HIDDEN  # o_proj rows (full hidden, a rank's K slice)
OTASKS = OROWS // ROWS  # 448
OK = PROJ  # K = 1536 a rank
OKCH = OK // 128  # 12 chunks


@dataclass(frozen=True)
class AttnPostBuild:
    """The step's shape ``scratch_layout`` sizes K2a's regions for."""

    tokens: int
    # the MLP AttnRes block sources
    nblocks: int


# o_proj's all-reduce in two halves: a column's partials go to the rank owning
# its OWN_COLS (OWN_TASKS of the AttnRes tasks), whose sum, rank order, goes to
# every rank (``stage_reduce``). A rank pushes each peer an eighth of what
# pushing every partial to every rank took.
OWN_COLS = HIDDEN // TP  # 896
OWN_TASKS = OWN_COLS // COLS  # 7


def ar_pairs(s):
    """The AR region, the partials of a rank's own columns: [token group][src
    rank][token][column pair of OWN_COLS]."""
    return TP * s * OWN_COLS // 2


def dg_pairs(s):
    """The DG region, the owners' reduced deltas: [token][column pair]."""
    return s * HIDDEN // 2


def group_view(c, g: int):
    """``c`` seen from token group g (``token_groups.group_view``): the AttnRes
    seam's moves (``kda.seam_moves``) plus o_proj's, the group's core rows,
    its block of AR and its rows of DG (every peer's copy and this rank's)."""
    rows, regions = seam_moves(c)
    own = tuple(
        (region + side, pairs * PAIR_BYTES)
        for region, pairs in (("ar", ar_pairs(TG)), ("dg", dg_pairs(TG)))
        for side in ("_peer", "_own")
    )
    return token_groups.group_view(
        c, g, rows + ((("args", "core"), OK * 2),), regions + own
    )


def scratch_layout(key: AttnPostBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA * groups(s),
    }
    return pair_layout(pairs.items())


def o_task_or(own, task):
    """o_proj task ``task`` while it exists, else the CTA's ``own`` task: the
    round a CTA skips loads its own slab again (cache hits), not the last
    task's slab on every such CTA at once."""
    return (task < OTASKS).select(task, own)


def o_loads(c, rg, paired=False):
    """Rows 16 rg .. of o_proj, this wave's K chunks (wave, wave + 8 < 12),
    ``gemv.chunk_ops``' each."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_o"])
    return [
        gemv.chunk_ops(
            r_w,
            rg * ROWS,
            OK,
            fx.min(wave + WAVES * i, OKCH - 1) * 128,
            lane,
            paired,
            w8=c["w8"],
        )
        for i in range(2)
    ]


@traced
def load_core(c):
    """The core rows (S x 1536 bf16) -> LDS, rows padded."""
    s, tid = c["S"], c["tid"]
    words = s * OK // 8
    row = OK + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["core"]), e * 4, vec_width=4, dtype=T.i32)
        )
        t = e // (OK // 8)
        k = e % (OK // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV -> bf16 partials -> the AR region of the rank owning
    these columns, this rank's slot (thread: token, row pair)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    row = OK + LDS_PAD
    scales = None
    if const_expr(c["w8"]):
        scales = gemv.pair_scales(c["args"]["w_o_s"], rg * ROWS, tid)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(2):
        ch = wave + WAVES * i
        live = ch < OKCH
        k = fx.min(ch, OKCH - 1) * 128 + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            xb = fx.Vector.from_elements(
                [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
            acc = mfma_bf16(
                gemv.operand(ops[i][q], c["w8"]), xb.bitcast(fx.BFloat16), acc
            )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0, v1 = gemv.row_pair(red, rp, tt, scales)
        groups_own = OWN_COLS // ROWS
        for p in range_constexpr(TP):
            if rg // groups_own == p:
                at = (c["rank"] * s + tt) * OWN_COLS + (rg - p * groups_own) * ROWS
                c["put_bf"](c["ar_peer"][p], at + 2 * rp, [v0, v1])
    gpu.barrier()


@traced
def stage_reduce(c, task):
    """Columns 128 task .. -> bf16 delta in LDS (thread: token, column pair).
    Their owner (``OWN_COLS``) sums the 8 ranks' partials (AR, rank order)
    and sends the sum to every rank's DG; every other rank reads it there.

    There are ``s * COLS / 2`` work items and ``THREADS`` threads, so the pass
    is strided past 8 rows.
    """
    s, tid = c["S"], c["tid"]
    n = s * (COLS // 2)
    owner = task // OWN_TASKS == c["rank"]
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < n:
            t = e // (COLS // 2)
            cp = e % (COLS // 2)
            col = task * COLS + 2 * cp
            # defined before the traced ``if``: a name its branches bind is
            # not seen past it
            lo, hi = fx.Float32(0.0), fx.Float32(0.0)
            if owner:
                lo, hi = sum_partials(
                    c["poll"],
                    c["ar_own"],
                    lambda src, t=t, col=col: (
                        (src * s + t) * OWN_COLS + col - c["rank"] * OWN_COLS
                    )
                    // 2,
                    TP,
                )
                lo, hi = bf16_round(lo), bf16_round(hi)
                for p in range_constexpr(TP):
                    c["put_bf"](c["dg_peer"][p], t * HIDDEN + col, [lo, hi])
            else:
                w = c["poll"]([(c["dg_own"], (t * HIDDEN + col) // 2, 1)])[0][0]
                lo, hi = bf2_f32(w)
            fx.ptr_store(lo, c["delta_lds"] + (t * COLS + 2 * cp))
            fx.ptr_store(hi, c["delta_lds"] + (t * COLS + 2 * cp + 1))
    gpu.barrier()
