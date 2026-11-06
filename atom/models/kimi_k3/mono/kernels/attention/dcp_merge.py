# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's DCP merge (``K2Build.dcp``): a full-attention layer's decode output
merged across the KV shards and gated inside K2, in place of the production
path's pack -> all-to-all -> combine -> sigmoid(g) * o.

Under DCP = TP every rank decodes all 96 heads over its own KV shard
(``MLAAttention._dcp_defer_merge``: the partial ``o`` after project-before-merge
and its LSE). Rank d owns heads 12 d .. 12 d + 11, the slice its ``o_proj``
reads. First every CTA, then a merge task (token group g, local head h) on
CTA ``g NH + h``:

    push    (every CTA, ``stage_push``) this rank's partial of head 12 d + h to
            rank d's DCP region, this rank's slot (system scope), for every d
    merge   the 8 sources' (o, lse) polled, merged in rank order (fp32) ->
            bf16 -> x sigmoid(gate) -> bf16 core rows (plain, device scope),
            then the group's CRDY flag h

Every ``o_proj`` CTA polls a group's 12 flags before it loads that group's core
rows (``wait_core``). The arithmetic is
``dcp_ops._dcp_a2a_unpack_combine_rank_order_kernel`` (the combine these layers'
decode steps take off mono, ``cp_lse_a2a(rank_order=True)``) then aiter's
``_fused_sigmoid_mul_kernel``, op for op, so a token's core is the same bits
whichever path its step took.

No rank can overwrite a peer's unread partials: to push layer L's a rank must
have finished layer L - 1's K2, whose o_proj all-reduce needed every peer's
o_proj, which waited for that peer's merge.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.config import HEAD_DIM, MLA_HEADS, TOKEN_GROUP, groups
from atom.models.kimi_k3.mono.kernels.attention.back import TP
from atom.models.kimi_k3.mono.kernels.attention.kda import HD, NH, PROJ, sigmoid
from atom.mono.device import dcp_merge as merge
from atom.mono.device.ops import CM_DEV, bf16_round, bf_hi, bf_lo, rsrc, traced
from atom.mono.device.sync import publish
from atom.mono.plan.execution import BLOCKS, THREADS

HEADS = MLA_HEADS  # a rank's decode heads under DCP = TP
assert HEADS == TP * NH and HD == HEAD_DIM
WORDS = HD // 2  # bf16 pair words a head row
SLOT = WORDS + 1  # pairs a (source, token, head): o, then the LSE
QUADS = HD // 4  # threads a merged (token, head) row: 4 columns each
MERGE_BATCH = 4  # sources whose rows a merge thread polls at once


def dcp_pairs(s):
    """The DCP region: [source rank][token][local head][SLOT]."""
    return TP * s * NH * SLOT


def crdy_pairs(s):
    """The CRDY flags: [token group][local head]."""
    return groups(s) * NH


def merge_tasks(s):
    return groups(s) * NH


def _slot(c, src, t, h):
    """The first pair of (source ``src``, token ``t``, local head ``h``)."""
    return ((src * c["S"] + t) * NH + h) * SLOT


@traced
def stage_push(c, bid):
    """This rank's partials of every token and head -> each owner rank's DCP
    region, this rank's slot (system scope), spread over the whole grid
    (thread: a token, local head and word pair -- or a token and local head's
    LSE -- for all 8 owners). Every owner's load of a kind is issued before its
    first store, so the push costs two load latencies, not one per owner."""
    a, tid = c["args"], c["tid"]
    s = c["S"]
    gid = bid * THREADS + tid
    grid = BLOCKS * THREADS
    pairs = WORDS // 2
    n_w, n_l = s * NH * pairs, s * NH
    words = []
    for i in range_constexpr(-(-n_w // grid)):
        e = gid + grid * i
        # clamped: every load is unconditional, only the store is gated
        es = fx.min(e, n_w - 1)
        t, h, k = es // (NH * pairs), es // pairs % NH, es % pairs
        for d in range_constexpr(TP):
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(a["part_o"]),
                    (t * HEADS + d * NH + h) * WORDS + 2 * k,
                    vec_width=2,
                    dtype=T.i32,
                )
            )
            words.append((e, d, _slot(c, c["rank"], t, h) + 2 * k, w))
    # the words out before the LSEs' loads: both sets live at once spilled
    # the 16-row build
    for e, d, at, w in words:
        if e < n_w:
            c["put_words"](c["dcp_peer"][d], at, [w[0], w[1]])
    lses = []
    for i in range_constexpr(-(-n_l // grid)):
        e = gid + grid * i
        es = fx.min(e, n_l - 1)
        t, h = es // NH, es % NH
        # an empty sparse row carries one dummy KV slot (the production pack
        # kernel's HAS_OWNED_COUNTS); a dense layer is handed counts of 1
        cnt = fx.Int32(bo.buffer_load(rsrc(a["owned"]), t, vec_width=1, dtype=T.i32))
        for d in range_constexpr(TP):
            lse = fx.Float32(
                bo.buffer_load(
                    rsrc(a["part_lse"]),
                    t * HEADS + d * NH + h,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            lse = (cnt == 0).select(fx.Float32(merge.NEG_INF), lse)
            lses.append((e, d, _slot(c, c["rank"], t, h) + WORDS, lse))
    for e, d, at, lse in lses:
        if e < n_l:
            c["put"](c["dcp_peer"][d], at, lse)


@traced
def stage_merge(c, g, h, sg):
    """Token group g (``sg`` rows), local head h: the 8 sources merged and
    gated -> core rows (thread: token, 4 columns); then CRDY flag (g, h)."""
    a, tid = c["args"], c["tid"]
    if tid < sg * QUADS:
        tl = tid // QUADS
        q = tid % QUADS
        t = g * TOKEN_GROUP + tl
        # the LSEs first, then the rows MERGE_BATCH sources a poll: all 8
        # sources' rows live at once spilled the 16-row build
        got = c["poll"](
            [(c["dcp_own"], _slot(c, src, t, h) + WORDS, 1) for src in range(TP)]
        )
        lses = [merge.clean_lse(got[src][0].bitcast(fx.Float32)) for src in range(TP)]
        total = merge.lse_total(lses)
        acc = [fx.Float32(0.0)] * 4
        rows = []
        for src in range_constexpr(TP):
            if src % MERGE_BATCH == 0:
                rows = c["poll"](
                    [
                        (c["dcp_own"], _slot(c, b, t, h) + 2 * q, 2)
                        for b in range(src, src + MERGE_BATCH)
                    ]
                )
            f = merge.lse_factor(lses[src], total)
            w0, w1 = rows[src % MERGE_BATCH]
            vals = [bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1)]
            acc = merge.lse_accumulate(acc, vals, f)
        # aiter's ``_fused_sigmoid_mul_kernel`` on the bf16 merged row
        at = t * (PROJ // 2) + h * WORDS + 2 * q
        gw = fx.Vector(bo.buffer_load(rsrc(a["gate"]), at, vec_width=2, dtype=T.i32))
        gates = [bf_lo(gw[0]), bf_hi(gw[0]), bf_lo(gw[1]), bf_hi(gw[1])]
        outs = []
        for j in range_constexpr(4):
            # aiter's ``_sigmoid_exp2``: its 1.44269504089 is 1 / ln 2 in fp32
            outs.append(bf16_round(acc[j]) * sigmoid(gates[j]))
        packed = (
            fx.Vector.from_elements(outs, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        )
        bo.buffer_store(packed, rsrc(a["core"]), at, cache_modifier=CM_DEV)
    publish(c["put"], c["crdy"], g * NH + h, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def stage_dcp_merge(c, task):
    """Merge task ``task`` (< ``merge_tasks``), after the grid's push: one body
    for the whole groups, one for a last group of fewer rows."""
    full, rest = divmod(c["S"], TOKEN_GROUP)
    g, h = task // NH, task % NH
    if const_expr(full > 0):  # noqa: SIM102 (a build constant, then the task)
        if g < full:
            stage_merge(c, g, h, TOKEN_GROUP)
    if const_expr(rest > 0):  # noqa: SIM102
        if g == full:
            stage_merge(c, g, h, rest)


@traced
def wait_core(c, g):
    """Token group g's core rows are in: its 12 CRDY flags."""
    if c["tid"] < NH:
        c["poll"]([(c["crdy"], g * NH + c["tid"], 1)])
    gpu.barrier()
