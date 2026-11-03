# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/attention/kda.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K1 ``kda_pre``: one Kimi-K3 KDA layer from the AttnRes seam to the gated core
output, for a uniform step of R requests x L tokens, S = R L rows (a spec
verify, or a plain decode at L = 1).

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads, in three roles so that
no CTA that polls a mailbox has weight loads in flight (a poll's load retires
behind every earlier load):

    attn CTAs 0 .. 55   slice (128 hidden columns each): prefix (+ delta) ->
                        bf16 prefix out, per-source partial sum of squares and
                        qk dot (PART); gate (token t on CTA t): softmax weights
                        (WGT); mix: the softmax mix, its partial sum of squares
                        (MSQ), every slice's -> output RMSNorm -> bf16 x rows
                        (plain device-scope stores, an XRDY flag a slice)
    gemv CTAs 56 ..     in_proj, two 16-row groups each: the first group's
                        weights issued at kernel start, the second's as x lands
                        -> q|k|v|g|f_a|beta (PROJ)
    kda tasks           (R x 12 heads x 8 v-chunks) on CTAs 0 ..: inputs that
                        need no hand-off loaded first (metadata, conv window,
                        state rows, f_b rows), then conv update of the owned
                        channels (q / k out through QK), f_b MFMA + lower-bound
                        gate, l2norm, the recurrence on 16 state rows (a state a
                        token written), partial norms (ONRM), gated RMSNorm ->
                        core_out

Rounding follows the original path where it is elementwise (the AttnRes prefix
add, every bf16 hand-off between the original kernels); split-K reductions
differ only in their order.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import TOKEN_GROUP, group_rows, groups
from atom.models.kimi_k3.mono.kernels import token_groups
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device import gemv
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterflies,
    butterfly,
    hw_exp2,
    hw_rsq,
    kernel_symbol,
    lane_all_gather,
    lane_reduce_scatter,
    ld_f32,
    ld_i32,
    load_ptr64,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.layout import PAIR_BYTES, pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi

HIDDEN = 7168
NA = 56  # attn CTAs
COLS = HIDDEN // NA  # 128 columns a slice
QC = COLS // 4  # threads a (token, source) row: 4 columns each
GSL = 7  # slices a gate / mix-norm thread sums (NA = 8 x 7)
HD = 128
# a state row's 128 columns over 32 lanes, 4 a lane: its sums' butterfly
ROW_LANES = (16, 8, 4, 2, 1)
NH = 12  # local heads at TP8
ROWS = 16
PROJ = NH * HD  # 1536
QKV = 3 * PROJ
G2_0 = QKV  # the output gate g
# ATOM fuses the hidden projections as [q | k | v | g | beta | f_a]
# (``KimiKDAAttention.process_weights_after_loading`` appends b_proj then
# f_a_proj); vLLM's kernels were written for the opposite tail order
BETA_0 = G2_0 + PROJ  # 6144
FA_0 = BETA_0 + NH  # 6156
FUSED = FA_0 + HD  # 6284: the rows ATOM's weight actually holds
NPROJ = -(-FUSED // ROWS) * ROWS  # 6288: the GEMV's last task's tile
GEMV_TASKS = NPROJ // ROWS  # 393
GEMV_CTAS = (GEMV_TASKS + 1) // 2  # 197
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
# A wave takes one K chunk from each of these groups, so ``i`` in the GEMV's
# ``range_constexpr(PER_WAVE)`` loop indexes a 1024-column group of the x row
# and a contiguous range of ``i`` is a contiguous column slab.
KGROUP = WAVES * KCH  # 1024
assert HIDDEN == PER_WAVE * KGROUP
# gfx950's LDS a workgroup, and what ``Smem`` takes before any role does. The
# x staging is the only part of the union that grows with the row count, so
# these two decide how many rows a build can hold at once.
LDS_LIMIT = 163840
LDS_FIXED = 8192
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 15

assert NA + GEMV_CTAS <= BLOCKS and NA == 8 * GSL

# Token groups. A step wider than ``TG`` rows runs its per-token stages (the
# AttnRes slice / gate / mix) a group at a time on the 16-row code, and its
# GEMV streams each weight slab once for every group (``group_view``); a
# token's arithmetic is the 16-row build's, so its outputs are too.
TG = TOKEN_GROUP


def seam_moves(c):
    """The AttnRes seam's ``token_groups.group_view`` moves: its per-token
    arguments and buffer (bytes a row), its per-group regions (bytes a group).
    K1 and K2's seams share them: what one of them lacks (K2 has no ``delta``
    argument and no PROJ) is not there to move."""
    ns = c["NS"]
    rows = (
        (("args", "prefix"), HIDDEN * 2),
        (("args", "delta"), HIDDEN * 2),
        (("args", "blocks"), fx.Int64(c["args"]["blk_sm"]) * fx.Int64(2)),
        ("xbuf", HIDDEN * 2),
        # check mode's taps: bf16 rows of x and of PROJ
        (("args", "x_dbg"), HIDDEN * 2),
        (("args", "proj_dbg"), NPROJ * 2),
    )
    regions = tuple(
        (region, pairs * PAIR_BYTES)
        for region, pairs in (
            ("part", NA * TG * ns * 2),
            ("wgt", TG * (ns + 1)),
            ("msq", NA * TG),
            ("xrdy", NA),
            ("proj", TG * NPROJ),
        )
    )
    return rows, regions


def group_view(c, g: int):
    """``c`` seen from token group g (``token_groups.group_view``): the seam's
    arguments and regions moved to it."""
    return token_groups.group_view(c, g, *seam_moves(c))


def head_tasks(s: int, qlen: int) -> bool:
    """Whether a step's KDA runs a task a (request, head) rather than a
    (request, head, 16-row chunk): on a plain decode, or once the chunk tasks
    take more than one round of CTAs.

    A chunk task's latency is mostly hand-offs -- the head's q / k from the
    other chunks' conv, their partial norms -- and the gates, which every
    chunk of a head recomputes. Past a round those repeat a round, so a head
    in one task (no hand-off, the gates once; ``head_parts`` splits its rows
    over two or four CTAs while they fit a round) wins. Within a round the
    chunk tasks run in parallel and load their state eight times faster,
    except on a plain decode, whose head task has one token to walk. Bits
    equal either way."""
    rounds = -(-(s // qlen) * NH * NVCH // BLOCKS)
    return qlen == 1 or rounds > 1


def head_parts(s: int, qlen: int) -> int:
    """CTAs a head task's chunks are split over (``head_tasks``): from 8 rows
    on, four, else two, while the parts still take one round. A head's rows
    are independent; only its output norm sums all of them, so a part
    publishes its chunks' partials (ONRM) and takes the others', as the chunk
    tasks do -- the same sums, in chunk order. Its conv and gates every part
    computes (no q / k hand-off); the first writes the conv state back once
    the others have read it (``kda_head_conv_state``). Below 8 rows the
    norm's hand-off outweighs the halved recurrence: one part."""
    if s < 8:
        return 1
    for p in (4, 2):
        if (s // qlen) * NH * p <= BLOCKS:
            return p
    return 1


def head_chunks(c) -> int:
    """A head task's chunks (``head_parts``)."""
    return NVCH // c["hparts"]


def slab_plan(s: int) -> list[tuple[int, int]]:
    """``[(first group, groups)]`` covering the x row's ``PER_WAVE`` column
    groups, in the fewest slabs whose widest fits ``s`` rows in LDS.

    One slab means the whole row is staged at once, which is what every build up
    to S=8 does and is byte-for-byte the behaviour this had before slabbing
    existed. Past that the row is staged a slab at a time: the GEMV reduces over
    K and its accumulator already lives across the whole ``i`` loop, so walking
    the groups in the same increasing order through several slabs leaves each
    wave's accumulation order **exactly** as it was. The result is bit-identical,
    not merely close.
    """
    for nslab in range(1, PER_WAVE + 1):
        widest = -(-PER_WAVE // nslab)
        if s * (widest * KGROUP + LDS_PAD) * 2 + LDS_FIXED <= LDS_LIMIT:
            plan, start = [], 0
            for n in range(nslab):
                # the remaining groups spread over the remaining slabs, so the
                # widths differ by at most one and none exceeds ``widest``
                take = -(-(PER_WAVE - start) // (nslab - n))
                plan.append((start, take))
                start += take
            return plan
    raise AssertionError(
        f"{s} rows: even one column group needs "
        f"{s * (KGROUP + LDS_PAD) * 2 + LDS_FIXED} B of LDS, over {LDS_LIMIT}"
    )


def slab_cols(s: int) -> int:
    """The widest slab's columns, which is what ``xl`` is sized for."""
    return max(n for _, n in slab_plan(s)) * KGROUP


_STREAM = fx.Stream(None)
# this model's kernels and the shared mono framework: in every build's JIT
# cache key (``mono.sources``)
_SOURCES = SOURCES


@dataclass(frozen=True)
class KdaPreBuild:
    # ``sym``: the field's short name in the kernel symbol (``symbol_params``)
    tokens: int = field(metadata={"sym": "s"})  # S
    qlen: int = field(metadata={"sym": "l"})  # L, tokens a request
    # AttnRes block sources (the prefix is one more)
    nblocks: int = field(metadata={"sym": "nb"})
    delta: bool = field(metadata={"sym": "d"})
    # conv state entries: CONV_W - 1 + num_spec
    state_len: int = field(metadata={"sym": "sl"})
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    debug: bool = field(default=False, metadata={"sym": "g"})
    timeline: bool = field(default=False, metadata={"sym": "t"})
    # a block-write layer's block: the updated prefix stored there
    write_idx: int = field(default=-1, metadata={"sym": "wi"})
    # ReplaySSM (``fla_ops.replayssm``): the record buffer's length (its
    # ``cache_len``), 0 without. A request's state is its checkpoint plus the
    # (k, u, g) records committed since (``write_pos``), rebuilt here; each
    # token writes a record instead of a state (``kda_replay``)
    replay_cap: int = field(default=0, metadata={"sym": "rp"})
    # its verify window (``replayssm_max_query_len``), whatever this step's
    # query length: the flush rule (``replay_slot``) is the commit's
    replay_window: int = field(default=0, metadata={"sym": "w"})
    # column groups a ``gemv_groups`` unit (a staging: a load round trip and a
    # barrier). Two take that latency half as often; one holds half the weights
    # in flight, leaving the VGPRs x loaded ahead needs (``gemv_staging``), or
    # a build that spilled with two (``runner._lean_width``)
    gemv_unit: int = field(default=2, metadata={"sym": "u"})
    # units a ``gemv_groups`` unit's x is loaded ahead of its MFMAs (held in
    # VGPRs meanwhile): x was just written by other CTAs, so one unit ahead
    # left its L2 / MALL latency exposed every unit
    x_ahead: int = field(default=1, metadata={"sym": "a"})
    kda0: int = field(default=0, metadata={"sym": "k0"})  # the first KDA task's CTA


ABI = KernelAbi(
    (
        "prefix",
        "delta",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_in",
        "w_fb",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "on_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "st_stride",
        "num_acc",
        # ReplaySSM (``replay_cap``; 0 / unread without): a table of i64
        # pointers (``REPLAY_TABLE``) the KDA tasks load: kernel arguments
        # are live in SGPRs across every role, and five more spilled the
        # 56-row GEMV's VGPRs
        "replay",
        "core_out",
        "x_dbg",
        "proj_dbg",
        "scratch",
        "plain",
        "layer",
        "tl",
    )
)


def scratch_layout(key: KdaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    assert s % key.qlen == 0
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA * groups(s),
        "proj": s * NPROJ,
        "qk": s * NH * 2 * HD,
        "onrm": s * NH * NVCH,
        # a head's readers done with what its writer overwrites (``readers_done``)
        "rdone": s * NH * NVCH,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def plain_layout(key: KdaPreBuild) -> dict:
    """The plain (untagged) words, in the step's ``plain`` buffer: the bf16 x
    rows. Never in the scratch: a later launch's tagged region over them would
    take a stale word that equals its tag for a landed pair."""
    return {"x": (0, key.tokens * HIDDEN * 2)}


def plain_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in plain_layout(key).values())


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


# ---------------------------------------------------------------- attn CTAs
def slice_loads(c, task):
    """``stage_slice``'s loads that no hand-off decides (the sources, the prefix,
    the norm and query weights), a list a round: a caller that waits before
    the slice (K2's all-reduce) issues them ahead of the wait."""
    s, ns, nb, tid = c["S"], c["NS"], c["NB"], c["tid"]
    a = c["args"]
    out = []
    for i in range(-(-s * ns * QC // THREADS)):
        e = fx.min(tid + THREADS * i, s * ns * QC - 1)
        row = e // QC  # (token, source)
        t = row // ns
        src = row % ns
        col = task * COLS + (e % QC) * 4
        out.append(
            {
                "blk": ld_bf4(
                    a["blocks"],
                    t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col,
                ),
                "pre": ld_bf4(a["prefix"], t * HIDDEN + col),
                "w": ld_bf4(a["ares_nw"], col),
                "q": ld_bf4(a["ares_qk"], col),
            }
        )
    return out


@traced
def stage_slice(c, task):
    """``slice_compute`` on loads issued right here."""
    slice_compute(c, task, slice_loads(c, task))


@traced
def slice_compute(c, task, ld):
    """Columns 128 task ..: every token's sources into LDS ``vals`` (4 columns a
    thread), the prefix (+ delta) rounded to bf16 and stored back, per-(token,
    source) partial sum of squares and qk dot -> PART. ``ld``: ``slice_loads``."""
    s, ns, nb, tid, lane = c["S"], c["NS"], c["NB"], c["tid"], c["lane"]
    a = c["args"]
    c0 = task * COLS
    n = s * ns * QC  # threads' work items
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        live = (tid + THREADS * i) < n
        row = e // QC  # (token, source)
        t = row // ns
        src = row % ns
        col = c0 + (e % QC) * 4
        is_pre = src == nb
        blk = ld[i]["blk"]
        pre = ld[i]["pre"]
        if const_expr(c["delta"]):
            if const_expr(c.get("delta_lds") is not None):
                # the delta reduced into LDS first (K2a: the all-reduced o_proj)
                dl = [
                    fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
                    for j in range(4)
                ]
            else:
                dl = ld_bf4(a["delta"], t * HIDDEN + col)
            # fp32, like production. `apply_attn_res` accumulates
            # `prefix + add (+ add2)` in fp32, uses that fp32 value as the
            # last softmax candidate and rounds only where it STORES the
            # prefix -- which `bf4_words` below already does. Rounding here
            # as well put an extra bf16 step into the candidate score, the
            # softmax weights and the mixed row of every layer that takes a
            # delta, compounding over ~60 layers.
            pre = [pre[j] + dl[j] for j in range(4)]
            if const_expr(c.get("reset") is not None):
                # K2 on a block-write layer (``reset`` 1, a runtime argument):
                # the prefix restarts at the attention's output
                restart = c["reset"] != 0
                pre = [restart.select(dl[j], pre[j]) for j in range(4)]
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre), rsrc(a["prefix"]), (t * HIDDEN + col) // 2
                )
        # the block-write layer's attention seam: the updated prefix is block
        # write_idx. The compile-time gate stays outside the traced condition:
        # the two cannot be one `and`.
        if const_expr(c.get("write_idx", -1) >= 0):  # noqa: SIM102
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col) // 2,
                )
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        w = ld[i]["w"]
        q = ld[i]["q"]
        # a thread past the end holds a clamped copy of the last element: it
        # must not store (it raced the owner with zeros)
        if live:
            for j in range_constexpr(4):
                fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


@traced
def stage_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s and their sum."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"](
            [(c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)]
        )
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            c["put"](c["wgt"], t * (ns + 1) + lane, ex)
        if lane == 0:
            c["put"](c["wgt"], t * (ns + 1) + ns, den)
    gpu.barrier()


@traced
def stage_mix(c, task):
    """Columns 128 task ..: the softmax mix of ``vals`` (4 columns a thread), its
    sum of squares (MSQ), every slice's -> rstd, then bf16 x = mixed rstd w ->
    the plain x rows, and the slice's XRDY flag."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    c0 = task * COLS
    nw = s * (ns + 1)
    if tid < nw:
        v = c["poll"]([(c["wgt"], tid, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + tid)
    gpu.barrier()
    mine = tid < s * QC
    t = fx.min(tid // QC, s - 1)
    cc = (tid % QC) * 4
    mixed = []
    for j in range_constexpr(4):
        acc = fx.Float32(0.0)
        for src in range_constexpr(ns):
            acc = acc + fx.ptr_load(red + (t * (ns + 1) + src)) * fx.ptr_load(
                c["vals"] + ((t * ns + src) * COLS + cc + j)
            )
        mixed.append(acc / fx.ptr_load(red + (t * (ns + 1) + ns)))
    sq = (
        mixed[0] * mixed[0]
        + mixed[1] * mixed[1]
        + mixed[2] * mixed[2]
        + mixed[3] * mixed[3]
    )
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    if mine & (lane % 32 == 0):
        c["put"](c["msq"], task * s + t, sq)
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (128 + g * s + tt))
    gpu.barrier()
    tot = fx.ptr_load(red + (128 + t))
    for g in range_constexpr(1, 8):
        tot = tot + fx.ptr_load(red + (128 + g * s + t))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
    wn = ld_bf4(a["in_nw"], c0 + cc)
    xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
    if mine:
        at = (t * HIDDEN + c0 + cc) // 2
        bo.buffer_store(xw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
        if const_expr(c["debug"]):
            bo.buffer_store(xw, rsrc(a["x_dbg"]), at)
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


# ---------------------------------------------------------------- gemv CTAs
def gemv_loads(c, rg, chunks, paired: bool = False):
    """This wave's weight operands of rows 16 rg .. for its K chunks ``chunks``
    (indices into wave, wave + 8, ..), ``gemv.chunk_ops``' each. ``rg`` past
    the last group loads the last group. ``paired`` (whole 128 B lines a load)
    from ``gemv_groups`` only: the one-group path, at the VGPR edge, spills
    with it."""
    lane, wave = c["lane"], c["wave"]
    # bounded: the last task's rows past ``FUSED`` are not in the weight,
    # and a buffer load past num_records returns zero
    r_w = rsrc(c["args"]["w_in"], FUSED * HIDDEN * 2)
    row0 = fx.min(rg, GEMV_TASKS - 1) * ROWS
    return [
        gemv.chunk_ops(r_w, row0, HIDDEN, (wave + WAVES * i) * KCH, lane, paired)
        for i in chunks
    ]


@traced
def await_x(c):
    """Every slice's x ready. Hoisted out of the staging so that slabbing it
    does not re-poll the same mailboxes once a slab."""
    if c["tid"] < NA:
        c["poll"]([(c["xrdy"], c["tid"], 1)])
    gpu.barrier()


@traced
def load_x(c, i0: int, ni: int):
    """Column groups ``i0 .. i0 + ni`` of every token's bf16 x row into LDS.

    The slab lands at local column 0, so a consumer's column is its global one
    less ``i0 * KGROUP``. ``ni == PER_WAVE`` stages the whole row, which is the
    single-slab case and is what every build up to S=8 does.
    """
    tid = c["tid"]
    words = x_words(c["S"], ni)
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        v, at = x_chunk(c, i0, ni, fx.min(tid + THREADS * i, words - 1))
        if tid + THREADS * i < words:
            fx.ptr_store(v, at)
    gpu.barrier()


def x_words(s: int, ni: int) -> int:
    """The 16 B chunks of every token's column groups ``ni`` (``load_x``)."""
    return s * ni * KGROUP // 8


def x_chunk(c, i0: int, ni: int, e):
    """Chunk ``e`` of the x slab of column groups ``i0 .. i0 + ni``: its 16 B
    from XBUF and its LDS address (token rows ``xrow`` apart, the slab from
    local column 0)."""
    cols = ni * KGROUP
    t, k = e // (cols // 8), e % (cols // 8) * 8
    v = fx.Vector(
        bo.buffer_load(
            rsrc(c["xbuf"]),
            (t * HIDDEN + i0 * KGROUP + k) // 8 * 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    return v, c["xl"] + (t * c["xrow"] + k) // 2


X_BATCH = 16  # the x wave's 16 B loads in flight (64 VGPRs: no weights yet)


def x_wave(s: int) -> bool:
    """Whether a one-group step stages its first slab's x on the last wave
    (``x_wave_stage``): at 1-2 rows and past one slab (on one slab at 3..8
    rows that wave's own weights, issued after the rows, end late)."""
    return s <= 2 or len(slab_plan(s)) > 1


@traced
def x_wave_stage(c, i0: int, ni: int):
    """The last wave's: XRDY polled and the first slab's x rows into LDS by its
    64 lanes alone, before its weights are issued, so these loads queue behind
    none (vmcnt counts a wave's loads in order); the other waves' weights are
    already in flight. No barrier here: the caller's, past every wave's
    weights, publishes the rows."""
    s, lane = c["S"], c["lane"]
    if lane < NA:
        c["poll"]([(c["xrdy"], lane, 1)])
    words = x_words(s, ni)
    n = (words + 63) // 64
    # X_BATCH loads in flight, then their stores: a store after each load
    # waits a round trip a chunk
    for b0 in range_constexpr(0, n, X_BATCH):
        got = [
            (i, *x_chunk(c, i0, ni, fx.min(lane + 64 * i, words - 1)))
            for i in range(b0, min(b0 + X_BATCH, n))
        ]
        for i, v, at in got:
            if lane + 64 * i < words:
                fx.ptr_store(v, at)


@traced
def gemv_slab(c, ops, ni: int, acc):
    """``acc`` advanced by this slab's ``ni`` column groups of rows 16 rg ...

    ``ops[j]`` is the weight of the slab's group ``j``, already in registers.
    The x columns are read at their slab-local offset; the accumulation order
    within a wave is the group order, which the slab loop preserves.
    """
    s, lane, wave = c["S"], c["lane"], c["wave"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    row = c["xrow"]
    for j in range_constexpr(ni):
        k = wave * KCH + j * KGROUP + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                ops[j][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


@traced
def gemv_finish(c, rg, acc):
    """A finished accumulator reduced across waves -> bf16 PROJ (and proj_dbg)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (tid < ROWS * s) & (rg < GEMV_TASKS):
        r = tid % ROWS
        tt = tid // ROWS
        v = bf16r(row_sum(red, r, tt))
        c["put"](c["proj"], tt * NPROJ + rg * ROWS + r, v)
        if const_expr(c["debug"]):
            bo.buffer_store(
                v.to(fx.BFloat16),
                rsrc(c["args"]["proj_dbg"]),
                tt * NPROJ + rg * ROWS + r,
            )
    gpu.barrier()


@traced
def attnres(c, task, sg, on, tls):
    """AttnRes task ``task`` (< NA) of one token group (``group_view``):
    slice -> gate (tasks < sg) -> mix -> x rows, XRDY."""
    enter_stage("k1.slice")
    stage_slice(c, task)
    stamp(on, tls, c["tid"], 1)
    enter_stage("k1.gate")
    if task < sg:
        stage_gate(c, task)
    stamp(on, tls, c["tid"], 2)
    enter_stage("k1.mix")
    stage_mix(c, task)
    stamp(on, tls, c["tid"], 3)


def gemv_staging(s: int) -> dict:
    """``gemv_groups``' staging for an s-row step (``KdaPreBuild.gemv_unit`` /
    ``x_ahead``): from three token groups on, a column group a unit with x
    four units ahead; else two groups a unit with x one ahead."""
    return {"gemv_unit": 1, "x_ahead": 4} if groups(s) >= 3 else {}


def xu_row(unit):
    """A staged token row of a ``gemv_groups`` unit of ``unit`` column groups (bf16)."""
    return unit * KGROUP + LDS_PAD


def units_kg(unit):
    """``gemv_groups``' column-group units: (first group, groups)."""
    return [(i, min(unit, PER_WAVE - i)) for i in range(0, PER_WAVE, unit)]


def x_unit_loads(c, i0, n):
    """A thread's chunks of column groups i0 .. i0 + n of the (group's) x rows."""
    return gemv.rows_loads(c["tid"], c["xbuf"], c["S"], n * KGROUP, HIDDEN, i0 * KGROUP)


def x_unit_store(c, xs, n, buf):
    """``x_unit_loads(c, ..)``' chunks (n column groups) into LDS rows of
    ``xu_row`` at ``buf``: ``c`` the same group's view."""
    gemv.rows_store(c["tid"], xs, c["S"], n * KGROUP, buf, xu_row(c["unit"]))


@traced
def gemv_groups(c, gi):
    """A step of several token groups: this GEMV CTA's two row groups (16 rg ..,
    rg = 2 gi, 2 gi + 1) ``KdaPreBuild.gemv_unit`` column groups (KGROUP) at a time, their
    weights loaded once and multiplied against every token group's x, an
    accumulator a (row group, token group). The units (column groups, token
    group) go through ``gemv.staged_units`` with x ``KdaPreBuild.x_ahead``
    units ahead, and a column-group unit's first token group issues the next
    column groups' weights under its MFMAs. Each wave walks its chunks in the
    16-row build's order, so every token's sum is the same."""
    rg0, rg1 = 2 * gi, 2 * gi + 1
    G = c["G"]
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    acc0, acc1 = [zero] * G, [zero] * G
    row = xu_row(c["unit"])
    kunits = units_kg(c["unit"])
    units = [(k, g) for k in range(len(kunits)) for g in range(G)]

    def unit_ops(k):
        i0, n = kunits[k]
        return [
            gemv_loads(c, rg0, range(i0, i0 + n), paired=True),
            gemv_loads(c, rg1, range(i0, i0 + n), paired=True),
        ]

    ops = {0: unit_ops(0)}
    # every group's slices ready (XRDY, NA a group)
    if c["tid"] < NA * G:
        c["poll"]([(c["xrdy"], c["tid"], 1)])
    gpu.barrier()

    def x_of(u):
        k, g = units[u]
        return x_unit_loads(group_view(c, g), *kunits[k])

    def x_put(u, xs, buf):
        k, g = units[u]
        x_unit_store(group_view(c, g), xs, kunits[k][1], buf)

    def multiply(u, buf):
        k, g = units[u]
        if const_expr(g == 0 and k + 1 < len(kunits)):
            ops[k + 1] = unit_ops(k + 1)
        cu = {**group_view(c, g), "xl": buf, "xrow": row}
        acc0[g] = gemv_slab(cu, ops[k][0], kunits[k][1], acc0[g])
        acc1[g] = gemv_slab(cu, ops[k][1], kunits[k][1], acc1[g])

    bufs = [c["xl"], c["xl"] + TG * row // 2]
    gemv.staged_units(len(units), c["x_ahead"], bufs, x_of, x_put, multiply)
    stamp(c["on"], c["tls"], c["tid"], 5)
    for g in range_constexpr(G):
        gemv_finish(group_view(c, g), rg0, acc0[g])
        gemv_finish(group_view(c, g), rg1, acc1[g])


# ---------------------------------------------------------------- kda, ReplaySSM
# the ``replay`` table's i64 entries: the requests' checkpoint slots, the slots'
# record cursors, the layer's k / u / g record buffers (``runner._replay_table``)
REPLAY_TABLE = ("slot_idx", "write_pos", "rbuf_k", "rbuf_u", "rbuf_g")


def replay_ptrs(c):
    """The ``replay`` table's pointers by name, loaded by a KDA task: only its
    branch holds them."""
    return {
        name: load_ptr64(c["args"]["replay"], i) for i, name in enumerate(REPLAY_TABLE)
    }


def replay_slot(c, rp, req):
    """ReplaySSM (``replay_cap``): request req's checkpoint slot, the records
    committed to it (``write_pos``; the prefill sentinel -1 is none), whether
    this step folds them into the checkpoint first (a flush: the buffer could
    not take two more windows), and where its own records start -- 0 past a
    flush, else after the committed ones (``_replayssm_kda_fwd_kernel``)."""
    slot = uniform(ld_i32(rp["slot_idx"], req))
    h_rec = uniform(fx.max(ld_i32(rp["write_pos"], fx.max(slot, 0)), fx.Int32(0)))
    flush = h_rec + 2 * c["rwin"] > c["cap"]
    return slot, h_rec, flush, flush.select(fx.Int32(0), h_rec)


def rec_at(c, slot, h, pos):
    """The first element of record pos of head h in a slot's [head][cap][HD]
    record buffers."""
    return ((fx.max(slot, 0) * NH + h) * c["cap"] + pos) * HD


def replay_max(cap: int, window: int) -> int:
    """The most records a step finds committed: a step flushes once
    h + 2 window > cap and then writes from 0, so h <= max(window, cap - window)."""
    return max(window, cap - window)


class KdaLdsWords(NamedTuple):
    """The KDA stage's LDS (``build``'s ``KdaLds``), f32 words a field."""

    fa: int
    qkl: int
    decay: int
    glog: int  # the log gates (ReplaySSM's g records): ``kda_gates``
    rec: int  # a task's ReplaySSM records u / k / g, staged (``kda_replay``)
    vloc: int
    ol: int
    onl: int
    beta: int


def kda_lds_words(key: KdaPreBuild) -> KdaLdsWords:
    L, cap = key.qlen, key.replay_cap
    # a head task holds the whole head's v and outputs (``head_tasks``)
    vrows = HD if head_tasks(key.tokens, L) else VCH
    return KdaLdsWords(
        fa=L * HD,
        qkl=L * 2 * HD,
        decay=L * HD,
        glog=L * HD if cap else 16,
        rec=3 * replay_max(cap, key.replay_window) * HD if cap else 16,
        vloc=L * vrows,
        ol=L * vrows,
        onl=max(L * NVCH, 16),
        beta=16,
    )


def lds_refusal(key: KdaPreBuild) -> str:
    """Why ``key``'s KDA stage does not fit the LDS, "" when it does. Its
    ReplaySSM records grow with the record buffer (``cache_len``)."""
    words = kda_lds_words(key)
    need = sum(words) * 4 + LDS_FIXED
    if need <= LDS_LIMIT:
        return ""
    return (
        f"K1's KDA stage needs {need} B of LDS, over {LDS_LIMIT}: ReplaySSM's"
        f" records {words.rec * 4} B (cache_len {key.replay_cap}, window"
        f" {key.replay_window})"
    )


def replay_stage(c, rp, slot, h):
    """The CTA's records (``replay_max`` of them) into LDS ``rec``, u / k / g
    planes of f32, one round trip for all of them."""
    tid = c["tid"]
    n = replay_max(c["cap"], c["rwin"]) * HD
    for i in range_constexpr(-(-n // THREADS)):
        e = fx.min(tid + THREADS * i, n - 1)
        at = rec_at(c, slot, h, 0) + e
        fx.ptr_store(ld_bf(rp["rbuf_u"], at), c["rec"] + e)
        fx.ptr_store(ld_bf(rp["rbuf_k"], at), c["rec"] + (n + e))
        fx.ptr_store(ld_f32(rp["rbuf_g"], at), c["rec"] + (2 * n + e))
    gpu.barrier()


# A head's tasks share memory one of them overwrites: its conv window (a head
# split over parts, ``head_parts``: every part reads all 3 HD channels, the
# first writes them back) and, on a ReplaySSM flush, its k / g records (one
# task a head rewrites positions 0 .. that every task has just staged). The
# writer is task 0 of the head; reader r > 0 publishes once it has read both.
def readers_done(c, req, h, r, who):
    """Reader ``r`` of head (req, h) done reading (its loads consumed: called
    after the conv and the replay staging)."""
    publish(c["put"], c["rdone"], (req * NH + h) * NVCH + r, fx.Int32(1), who)


@traced
def await_readers(c, req, h, n):
    """The writer: readers 1 .. n of head (req, h) done."""
    if c["tid"] < n:
        c["poll"]([(c["rdone"], (req * NH + h) * NVCH + 1 + c["tid"], 1)])
    gpu.barrier()


@traced
def kda_replay(c, k, h, rows, k0, st):
    """A thread's state rows ``rows`` of head h (columns k0 .. k0 + 3) after
    the slot's h_rec committed records, from the checkpoint's ``st``: the
    records' own recurrence S <- diag(exp g_j) S + u_j k_j, in place, from the
    records in LDS (``replay_stage``): the state ``_replay_tiles``' closed
    form gives (exp C S + sum_j exp(C - C_j) u_j k_j, a GEMM there), each
    thread its own elements. Only the h_rec live records are read (a stale
    one need not be finite)."""
    n = len(rows)
    hmax = replay_max(c["cap"], c["rwin"])
    h_rec = k["h_rec"]
    replay_stage(c, k["rp"], k["slot0"], h)
    lu, lk, lg = c["rec"], c["rec"] + hmax * HD, c["rec"] + 2 * hmax * HD
    out = [st[r][e] for r in range(n) for e in range(4)]  # row r col e: 4 r + e
    for j in range(h_rec):
        dk = [
            hw_exp2(fx.ptr_load(lg + (j * HD + k0 + e)) * fx.Float32(LOG2E))
            for e in range(4)
        ]
        kj = [fx.ptr_load(lk + (j * HD + k0 + e)) for e in range(4)]
        for r in range_constexpr(n):
            u = fx.ptr_load(lu + (j * HD + rows[r]))
            for e in range_constexpr(4):
                out[4 * r + e] = out[4 * r + e] * dk[e] + u * kj[e]
    return [[out[4 * r + e] for e in range(4)] for r in range(n)]


@traced
def replay_records(c, k, h, j, row, k0, vn, kv, writes_k):
    """Token j's ReplaySSM record at the slot's ``rbase + j``: u (bf16, its
    delta-corrected value) for state row ``row`` by the row's first lane, and
    -- by ``writes_k``, one thread group a head -- its k (bf16, l2-normed) and
    log gate (fp32) for columns k0 .. k0 + 3."""
    rp = k["rp"]
    at = rec_at(c, k["slot0"], h, k["rbase"] + j)
    if c["lane"] % 32 == 0:
        bo.buffer_store(fx.Float32(vn).to(fx.BFloat16), rsrc(rp["rbuf_u"]), at + row)
    if writes_k:
        bo.buffer_store(bf4_words(kv), rsrc(rp["rbuf_k"]), (at + k0) // 2)
        gs = [fx.ptr_load(c["glog"] + (j * HD + k0 + e)) for e in range(4)]
        bo.buffer_store(
            fx.Vector.from_elements(gs, fx.Float32), rsrc(rp["rbuf_g"]), at + k0
        )


# ---------------------------------------------------------------- kda
def kda_stage(c, req, h, ch):
    """Chunk ch's inputs (``kda_task_inputs``): a thread < 3 VCH owns conv
    channel (part, 16 ch + d), every thread state row 16 ch + tid / 32."""
    return kda_task_inputs(c, req, h, ch, (ch * VCH, VCH), [ch * VCH + c["tid"] // 32])


def state_late(s: int) -> bool:
    """Whether a KDA task issues its initial state rows after its last poll
    (``kda_gates``) rather than with its other inputs: vmcnt counts loads in
    issue order, so polls issued behind 64 KB of state wait for all of it.
    Below 8 rows the gates are too short to cover the state's latency: it
    goes with the other inputs."""
    return s >= 8


def state_loads(k):
    """A task's initial state rows ``k["rows"]`` into ``k["st"]``."""
    k["st"] = [
        fx.Vector(bo.buffer_load(rsrc(k["st_base"]), off, vec_width=4, dtype=T.f32))
        for off in k["head_offs"]
    ]


def fb_loads(c, k):
    """This wave's f_b rows (``kda_gates``' MFMA A operands) into ``k["fb"]``:
    with a task's other inputs on a plain decode, after ``kda_gates``' polls
    on a verify (its f_a poll is L tokens' worth, and in vmcnt order it would
    wait for the 32 KB)."""
    k["fb"] = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["w_fb"]), k["fb_at"] + 4 * q, vec_width=4, dtype=T.i32
            )
        )
        for q in range(4)
    ]


def kda_task_inputs(c, req, h, m0, channels, rows):
    """A KDA task's inputs that need no hand-off, issued before its first poll:
    metadata, the owned conv channels' window / weights (``channels`` (c0, n):
    a thread < 3 n owns channel (part, c0 + d)), the initial state rows
    ``rows`` (a chunk's each, from chunk m0), the f_b rows of this wave and the
    per-head constants."""
    c0, width = channels
    L, sl, tid, lane, wave = c["L"], c["SL"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    k = {"m0": m0}
    # every value a buffer resource is built on wave-uniform (else a waterfall)
    # a slot is live at ``>= 0``, not ``> 0``: ATOM's state pool numbers slots
    # from zero and marks a padded row with PAD_SLOT_ID (-1), so the first
    # request of a step owns slot 0. Testing ``> 0`` reads that request as
    # stateless, which zeroes its whole recurrence output -- silently, because
    # a zeroed core is a legal answer for a request that really has no state.
    k["acc_idx"] = uniform(ld_i32(a["num_acc"], req)) - 1
    k["slot_c"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"]))
    k["slot0"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"] + k["acc_idx"]))
    k["slots"] = [
        uniform(ld_i32(a["st_idx"], req * a["st_stride"] + j)) for j in range(L)
    ]
    if const_expr(c["cap"]):
        # ReplaySSM: the state starts at the checkpoint (``kda_replay``)
        k["rp"] = replay_ptrs(c)
        k["slot0"], k["h_rec"], k["flush"], k["rbase"] = replay_slot(c, k["rp"], req)
    ct = fx.min(tid, 3 * width - 1)
    part = ct // width
    chan = part * PROJ + h * HD + c0 + ct % width
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot_c"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]  # the channel's first entry (bf16 elements)
    k["cs"], k["co"], k["chan"], k["part"] = cs, co, chan, part
    k["cw"] = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
    )
    k["win"] = [
        ld_bf(cs, co + (k["acc_idx"] + j) * a["cs_tok"]) for j in range(CONV_W - 1)
    ]
    k["old"] = [
        ld_bf(cs, co + (k["acc_idx"] + 1 + j) * a["cs_tok"]) for j in range(sl - L)
    ]
    # recurrence: a thread's state rows, k columns (tid % 32) 4 ..
    k0 = (tid % 32) * 4
    k["rows"] = rows
    k["head_offs"] = [(h * HD + r) * HD + k0 for r in rows]
    k["st_base"] = (
        a["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(a["rs_seq"]) * 4
    )
    if const_expr(not state_late(c["S"])):
        state_loads(k)
    # f_b: wave w's rows 16 w .. of head h, a lane 64 B of its row
    fr = h * HD + wave * 16 + lane % 16
    fb = (fr * HD + (lane // 16) * 32) // 2
    k["fb_at"] = fb
    if const_expr(L == 1):
        fb_loads(c, k)
    k["aa"] = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    k["dtb"] = [
        ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i)
        for i in range(4)
    ]
    return k


@traced
def kda_conv(c, k, req, h, ch):
    """Owned channels: the spec conv update of request ``req``'s L tokens -> new
    conv state; q / k out through QK, v to LDS."""
    L, tid = c["L"], c["tid"]
    stamp(c["on"], c["tls"], tid, 12)
    xs = conv_inputs(c, k, req, h)
    if tid < 3 * VCH:
        xv = [f32_of(x[0]) for x in xs]
        stamp(c["on"], c["tls"], tid, 13)
        conv_state_store(c, k, xv)
        stamp(c["on"], c["tls"], tid, 14)
        outs = conv_tokens(k, xv)
        for j in range_constexpr(L):
            o = outs[j]
            if k["part"] < 2:
                c["put"](
                    c["qk"],
                    ((req * NH + h) * L + j) * 2 * HD
                    + k["part"] * HD
                    + ch * VCH
                    + tid % VCH,
                    o,
                )
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * VCH + tid % VCH))
    gpu.barrier()


@traced
def kda_gates(c, k, req, h):
    """f_a, beta and the head's q / k into LDS (a chunk task polls them here,
    a head task's conv already did); g1 = f_b f_a by MFMA (a wave 16 rows) ->
    lower-bound gate -> decay in LDS; q / k l2-normed (q scaled)."""
    L = c["L"]
    if const_expr(not c["kda_head"]):
        chunk_gate_inputs(c, req, h)
    if const_expr(L > 1):
        fb_loads(c, k)
    if const_expr(state_late(c["S"])):
        state_loads(k)
    if const_expr(not c["kda_head"]):
        gpu.barrier()
    gates_compute(c, k, h)


@traced
def chunk_gate_inputs(c, req, h):
    """A chunk task's q / k (the head's other chunk tasks' conv), polled into
    LDS; its f_a and beta came with its own conv's poll."""
    L, tid = c["L"], c["tid"]
    n = L * 2 * HD
    es = [fx.min(tid + THREADS * i, n - 1) for i in range(-(-n // THREADS))]
    got = c["poll"]([(c["qk"], (req * NH + h) * L * 2 * HD + e, 1) for e in es])
    for i in range_constexpr(len(es)):
        if tid + THREADS * i < n:
            fx.ptr_store(f32_of(got[i][0]), c["qkl"] + es[i])


def gate_input_specs(c, req, h):
    """The gates' f_a and beta of request req's head h as poll specs, to ride
    a task's conv poll (one round trip to PROJ for both stages, not three in
    series), and the f_a elements this thread stores (``store_gate_inputs``)."""
    L, tid = c["L"], c["tid"]
    n_fa = L * HD
    fa_e = [fx.min(tid + THREADS * i, n_fa - 1) for i in range(-(-n_fa // THREADS))]
    specs = [
        (c["proj"], (req * L + e // HD) * NPROJ + FA_0 + e % HD, 1) for e in fa_e
    ] + [(c["proj"], (req * L + fx.min(tid, L - 1)) * NPROJ + BETA_0 + h, 1)]
    return specs, fa_e


@traced
def store_gate_inputs(c, got, fa_e):
    """``gate_input_specs``' polled words -> f_a and sigmoid(beta) in LDS."""
    L, tid = c["L"], c["tid"]
    for i in range_constexpr(len(fa_e)):
        if tid + THREADS * i < L * HD:
            fx.ptr_store(f32_of(got[i][0]), c["fa"] + fa_e[i])
    if tid < L:
        fx.ptr_store(sigmoid(f32_of(got[-1][0])), c["beta"] + tid)


@traced
def gates_compute(c, k, h):
    """``kda_gates`` past its inputs in LDS (f_a, beta, q / k)."""
    L, lane, wave = c["L"], c["lane"], c["wave"]
    # g1 rows 16 wave .. x tokens: A = f_b rows (staged), B = f_a [token][K]
    t = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xs = [fx.ptr_load(c["fa"] + (t * HD + kb + e)) for e in range(8)]
        xb = fx.Vector.from_elements(xs, fx.Float32).to(fx.BFloat16)
        acc = mfma_bf16(k["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(acc[i]) + k["dtb"][i]
        gate = fx.Float32(c["lb"]) * sigmoid(k["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), c["decay"] + (t * HD + kk))
            if const_expr(c["cap"]):
                fx.ptr_store(gate, c["glog"] + (t * HD + kk))
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + (row * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, c["qkl"] + (row * HD + lane))
            fx.ptr_store(x1 * sc, c["qkl"] + (row * HD + 64 + lane))
    gpu.barrier()


@traced
def kda_recur(c, k, req, h, ch):
    """State rows 16 ch .. of head h: the recurrence over the L tokens
    (``recur_tokens``), bf16 outputs to LDS ``ol`` [token][16] and their
    partial norms (ONRM)."""
    tid = c["tid"]
    live0 = k["slot0"] >= 0
    stv = [[live0.select(k["st"][0][e], fx.Float32(0.0)) for e in range(4)]]
    if const_expr(c["cap"]):
        stv = kda_replay(c, k, h, k["rows"], (tid % 32) * 4, stv)
        # chunk 0 rewrites the k / g records from 0 on a flush: the head's
        # other chunks must have staged them first
        readers_done(c, req, h, ch, (tid == 0) & (ch != 0))
        if (ch == 0) & k["flush"]:
            await_readers(c, req, h, NVCH - 1)
        if k["flush"] & live0:
            # the records folded: the checkpoint moves to here
            store_states(c, k, k["slot0"], stv)
    recur_tokens(c, k, h, stv, VCH, [tid // 32])
    norm_partials(c, k, req, h, VCH, 0, to_lds=False)


def store_states(c, k, slot, stv):
    """The task's state rows ``stv`` (a chunk's each) into ``slot``."""
    a = c["args"]
    dst = rsrc(a["rstate"] + fx.Int64(slot) * fx.Int64(a["rs_seq"]) * 4)
    for m in range_constexpr(len(stv)):
        bo.buffer_store(
            fx.Vector.from_elements(stv[m], fx.Float32), dst, k["head_offs"][m]
        )


@traced
def recur_tokens(c, k, h, stv, lstride, lrows):
    """The recurrence over the L tokens for the task's state rows (``stv``: a
    chunk's each, ``k["rows"]``): a token's row reductions side by side
    (``butterflies``), its state written to its slot (ReplaySSM: its records),
    its bf16 outputs to LDS ``ol`` at ``j lstride + lrows[m]``, where ``vloc``
    holds its v. Every store after the arithmetic, which is branch-free."""
    L, tid, lane = c["L"], c["tid"], c["lane"]
    k0 = (tid % 32) * 4
    live0 = k["slot0"] >= 0
    nm = len(stv)
    for j in range_constexpr(L):
        qv = [fx.ptr_load(c["qkl"] + ((2 * j) * HD + k0 + e)) for e in range(4)]
        kv = [fx.ptr_load(c["qkl"] + ((2 * j + 1) * HD + k0 + e)) for e in range(4)]
        dk = [fx.ptr_load(c["decay"] + (j * HD + k0 + e)) for e in range(4)]
        beta = fx.ptr_load(c["beta"] + j)
        slot = k["slots"][j]
        dec = [[stv[m][e] * dk[e] for e in range(4)] for m in range(nm)]
        kdot = [d[0] * kv[0] + d[1] * kv[1] + d[2] * kv[2] + d[3] * kv[3] for d in dec]
        if const_expr(nm > 1):
            # a row's sum on a lane, then every lane all of them: same bits
            p1, _ = lane_reduce_scatter(kdot, ROW_LANES, lane)
            ps = lane_all_gather(p1, nm, ROW_LANES, lane)
        else:
            ps = butterflies(kdot, ROW_LANES)
        vns = [
            (fx.ptr_load(c["vloc"] + (j * lstride + lrows[m])) - ps[m]) * beta
            for m in range(nm)
        ]
        stv = [[dec[m][e] + vns[m] * kv[e] for e in range(4)] for m in range(nm)]
        qdot = [s[0] * qv[0] + s[1] * qv[1] + s[2] * qv[2] + s[3] * qv[3] for s in stv]
        if const_expr(nm > 1):
            # only one lane stores a row's output: no gather
            o1, orow = lane_reduce_scatter(qdot, ROW_LANES, lane)
            os_ = [live0.select(bf16r(o1), fx.Float32(0.0))]
        else:
            os_ = butterflies(qdot, ROW_LANES)
            os_ = [live0.select(bf16r(o), fx.Float32(0.0)) for o in os_]
        if const_expr(c["cap"]):
            if live0:
                for m in range_constexpr(nm):
                    # k / g once a head: its first chunk's first thread group
                    replay_records(
                        c,
                        k,
                        h,
                        j,
                        k["rows"][m],
                        k0,
                        vns[m],
                        kv,
                        (tid // 32 == 0) & (k["m0"] + m == 0),
                    )
        else:
            if (slot >= 0) & live0:
                store_states(c, k, slot, stv)
        if const_expr(nm > 1):
            # the lanes whose bits below the halving offsets are zero, one
            # a row: row ``orow``
            if lane % (32 // nm) == 0:
                fx.ptr_store(os_[0], c["ol"] + (j * lstride + pick(lrows, orow)))
        else:
            if lane % 32 == 0:
                fx.ptr_store(os_[0], c["ol"] + (j * lstride + lrows[0]))


@traced
def norm_partials(c, k, req, h, lstride, lcol0, to_lds):
    """Each token's partial norm a chunk m = m0 .. (``ol``'s 16 outputs from
    column ``lcol0 + 16 (m - m0)`` of a ``lstride``-wide token row): to LDS
    ``onl`` (``to_lds``, one part a head) or ONRM. Both behind a barrier."""
    L, tid = c["L"], c["tid"]
    nm = len(k["rows"])
    gpu.barrier()
    if tid < L * nm:
        j = tid // nm
        i = tid % nm
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (j * lstride + lcol0 + i * VCH + e))
            ss = ss + x * x
        if const_expr(to_lds):
            fx.ptr_store(ss, c["onl"] + tid)
        else:
            c["put"](c["onrm"], ((req * NH + h) * L + j) * NVCH + k["m0"] + i, ss)
    gpu.barrier()


@traced
def kda_norm(c, req, h, ch):
    """Rows 16 ch ..: gated RMSNorm over the head's 128 (every chunk's partial)
    -> core_out."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < L * VCH:
        j = tid // VCH
        e = tid % VCH
        vr = ch * VCH + e
        got = c["poll"](
            [(c["onrm"], ((req * NH + h) * L + j) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)]
        )
        ss = sum_polled(got[:NVCH])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * VCH + e))
            * r
            * ld_f32(a["on_w"], vr)
            * sigmoid(f32_of(got[NVCH][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )
    gpu.barrier()


# ---------------------------------------------------------------- kda, a head a task
# ``head_tasks``: the chunk path's arithmetic, in its order, for all NVCH chunks
# of a head in one CTA -- conv a thread a channel (q / k straight to LDS), the
# recurrence a thread a column quad of NVCH rows, the norm's chunk partials
# summed in LDS in chunk order -- so the two paths' outputs are the same bits.
def kda_head_stage(c, req, h, m0):
    """``kda_stage`` for a head (``kda_task_inputs``): a thread < 3 HD owns
    conv channel (part, d), and every thread the state rows tid / 32 + 16 m of
    each of the task's chunks m = m0 .. (``head_chunks``)."""
    rows = [(m0 + i) * VCH + c["tid"] // 32 for i in range(head_chunks(c))]
    return kda_task_inputs(c, req, h, m0, (0, HD), rows)


@traced
def kda_head_conv(c, k, req, h):
    """``kda_conv`` for a head's 3 HD channels: q / k into ``qkl`` (the layout
    ``kda_gates`` polls them into), v into ``vloc`` [token][128]; the gates'
    f_a and beta in the same poll batch (``gate_input_specs``)."""
    tid = c["tid"]
    xs = conv_inputs(c, k, req, h)
    if tid < 3 * HD:
        xv = [f32_of(x[0]) for x in xs]
        # one part: the conv state here; several: by the first, once the others
        # have read their windows (``kda_head_conv_state``)
        if const_expr(c["hparts"] == 1):
            conv_state_store(c, k, xv)
        d = tid % HD
        outs = conv_tokens(k, xv)
        for j in range_constexpr(c["L"]):
            o = outs[j]
            if k["part"] < 2:
                fx.ptr_store(o, c["qkl"] + ((2 * j + k["part"]) * HD + d))
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * HD + d))
    gpu.barrier()


def conv_inputs(c, k, req, h):
    """A conv thread's channel of request ``req``'s L tokens (PROJ), polled
    with the gates' f_a and beta (``gate_input_specs``), which go to LDS: the
    tokens' words."""
    L = c["L"]
    gspecs, fa_e = gate_input_specs(c, req, h)
    got = c["poll"](
        [(c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)] + gspecs
    )
    store_gate_inputs(c, got[L:], fa_e)
    return got[:L]


def conv_tokens(k, xv):
    """The causal conv of a channel's window and its new tokens ``xv``, then
    SiLU: each token's bf16 output."""
    seq = k["win"] + xv
    out = []
    for j in range(len(xv)):
        acc = fx.Float32(0.0)
        for q in range(CONV_W):
            acc = acc + k["cw"][q] * seq[j + q]
        out.append(bf16r(acc * sigmoid(acc)))
    return out


@traced
def conv_state_store(c, k, xv):
    """A head-task thread's conv channel: the request's new conv state (the
    window's last old entries, then the step's L tokens ``xv``)."""
    sl, L = c["SL"], c["L"]
    a = c["args"]
    if k["slot_c"] >= 0:
        rs = rsrc(k["cs"])
        for j in range_constexpr(sl):
            nv = k["old"][j] if j < sl - L else xv[j - (sl - L)]
            bo.buffer_store(
                fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
            )


@traced
def kda_head_conv_state(c, k, req, h):
    """A head split over parts (``head_parts``): every part reads the whole
    conv window, the first writes it back -- after the others have read it
    (their ``readers_done``) and, on a ReplaySSM flush, staged the head's
    records the first then rewrites. The step's tokens re-read from PROJ
    (landed)."""
    L, tid = c["L"], c["tid"]
    readers_done(c, req, h, k["m0"] // head_chunks(c), (tid == 0) & (k["m0"] != 0))
    if k["m0"] == 0:
        await_readers(c, req, h, c["hparts"] - 1)
        if tid < 3 * HD:
            xs = c["poll"](
                [(c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)]
            )
            conv_state_store(c, k, [f32_of(xs[j][0]) for j in range(L)])


@traced
def kda_head_recur(c, k, req, h):
    """``kda_recur`` for the task's chunks m0 .. (``head_chunks``): rows
    tid / 32 + 16 m, the outputs to ``ol`` [token][128], each chunk's partial
    norm to ``onl`` (one part a head) or ONRM (``head_parts``)."""
    tid = c["tid"]
    live0 = k["slot0"] >= 0
    stv = [[live0.select(st[e], fx.Float32(0.0)) for e in range(4)] for st in k["st"]]
    if const_expr(c["cap"]):
        stv = kda_replay(c, k, h, k["rows"], (tid % 32) * 4, stv)
    if const_expr(c["hparts"] > 1):
        kda_head_conv_state(c, k, req, h)
    if const_expr(c["cap"]):  # noqa: SIM102
        if k["flush"] & live0:
            # the records folded: the checkpoint moves to here
            store_states(c, k, k["slot0"], stv)
    stamp(c["on"], c["tls"], tid, 12)
    recur_tokens(c, k, h, stv, HD, k["rows"])
    stamp(c["on"], c["tls"], tid, 13)
    norm_partials(c, k, req, h, HD, k["m0"] * VCH, to_lds=c["hparts"] == 1)


@traced
def kda_head_norm(c, k, req, h):
    """``kda_norm`` for the task's rows: the chunk partials summed in chunk
    order, as ``sum_polled`` sums the polled ones. A thread a (token, row),
    ``THREADS`` of them a pass."""
    L, tid = c["L"], c["tid"]
    n = L * head_chunks(c) * VCH
    for p in range_constexpr(-(-n // THREADS)):
        kda_head_norm_row(c, k, req, h, tid + THREADS * p)
    gpu.barrier()


@traced
def kda_head_norm_row(c, k, req, h, e):
    """``kda_head_norm``'s (token, task row) ``e`` (none past the task's L x
    rows)."""
    L = c["L"]
    a = c["args"]
    nr = head_chunks(c) * VCH
    if e < L * nr:
        j = e // nr
        vr = k["m0"] * VCH + e % nr
        g = c["poll"]([(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)])
        if const_expr(c["hparts"] == 1):
            ss = fx.ptr_load(c["onl"] + j * NVCH)
            for m in range_constexpr(1, NVCH):
                ss = ss + fx.ptr_load(c["onl"] + (j * NVCH + m))
        else:
            ss = sum_polled(
                c["poll"](
                    [
                        (c["onrm"], ((req * NH + h) * L + j) * NVCH + m, 1)
                        for m in range(NVCH)
                    ]
                )
            )
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * HD + vr))
            * r
            * ld_f32(a["on_w"], vr)
            * sigmoid(f32_of(g[0][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )


@traced
def kda_head_tasks(c, on, tls):
    s, L, tid, bid = c["S"], c["L"], c["tid"], c["bid"]
    parts = c["hparts"]
    # a head's parts on adjacent CTAs
    for task in range(first_task(bid, c["kda0"]), (s // L) * NH * parts, BLOCKS):
        req = task // (NH * parts)
        h = (task // parts) % NH
        enter_stage("k1.stage")
        k = kda_head_stage(c, req, h, task % parts * head_chunks(c))
        stamp(on, tls, tid, 7)
        enter_stage("k1.conv")
        kda_head_conv(c, k, req, h)
        stamp(on, tls, tid, 8)
        enter_stage("k1.gates")
        kda_gates(c, k, req, h)
        stamp(on, tls, tid, 9)
        enter_stage("k1.recur")
        kda_head_recur(c, k, req, h)
        stamp(on, tls, tid, 10)
        enter_stage("k1.onorm")
        kda_head_norm(c, k, req, h)
        stamp(on, tls, tid, 11)


@traced
def kda_tasks(c, on, tls):
    if const_expr(c["kda_head"]):
        kda_head_tasks(c, on, tls)
        return
    s, L, tid, bid = c["S"], c["L"], c["tid"], c["bid"]
    tasks = (s // L) * NH * NVCH
    for task in range(first_task(bid, c["kda0"]), tasks, BLOCKS):
        req = task // (NH * NVCH)
        h = (task // NVCH) % NH
        ch = task % NVCH
        enter_stage("k1.stage")
        k = kda_stage(c, req, h, ch)
        stamp(on, tls, tid, 7)
        enter_stage("k1.conv")
        kda_conv(c, k, req, h, ch)
        stamp(on, tls, tid, 8)
        enter_stage("k1.gates")
        kda_gates(c, k, req, h)
        stamp(on, tls, tid, 9)
        enter_stage("k1.recur")
        kda_recur(c, k, req, h, ch)
        stamp(on, tls, tid, 10)
        enter_stage("k1.onorm")
        kda_norm(c, req, h, ch)
        stamp(on, tls, tid, 11)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: KdaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    # per-token stages and the x staging hold one token group (``TG``)
    G = groups(s)
    sg = min(s, TG)
    assert NA * G <= BLOCKS, "a token group's AttnRes a 56-CTA block"
    # The x staging holds one slab of the row at a time; at S <= 8 that is the
    # whole row and the single-slab path is what this kernel has always run.
    plan = slab_plan(sg)
    xrow = slab_cols(sg) + LDS_PAD
    assert sg * xrow * 2 + LDS_FIXED <= LDS_LIMIT, "x slab exceeds LDS"
    assert L <= 16 and key.nblocks >= 1

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, sg * ns * COLS, 16]

    # past one token group the GEMV stages a unit at a time in two buffers
    # (``gemv_groups``)
    xl_words = max(sg * xrow, 2 * TG * xu_row(key.gemv_unit) if G > 1 else 0) // 2
    assert xl_words * 4 + LDS_FIXED <= LDS_LIMIT, "x staging exceeds LDS"

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, xl_words, 16]

    head = head_tasks(s, L)
    assert not lds_refusal(key), lds_refusal(key)
    kw = kda_lds_words(key)

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, kw.fa, 16]
        qkl: fx.Array[fx.Float32, kw.qkl, 16]
        decay: fx.Array[fx.Float32, kw.decay, 16]
        glog: fx.Array[fx.Float32, kw.glog, 16]
        rec: fx.Array[fx.Float32, kw.rec, 16]
        vloc: fx.Array[fx.Float32, kw.vloc, 16]
        ol: fx.Array[fx.Float32, kw.ol, 16]
        onl: fx.Array[fx.Float32, kw.onl, 16]
        beta: fx.Array[fx.Float32, kw.beta, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        kda: KdaLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = kernel_symbol("k3_mono_kda_pre", **symbol_params(key))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        replay: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        plain: Int64,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl, kl = role.attn.peek(), role.gemv.peek(), role.kda.peek()
        # ``layer`` is this launch's slot of the step; every mailbox pair is
        # zeroed and peer-fenced before the step's first kernel
        # (``StepMailboxes.begin_step``), so no earlier step's pair carries a
        # live tag
        mb = Mailbox(layer)
        c = {
            "S": s,
            "G": G,
            "L": L,
            # the LDS stride of one staged x row: the widest slab plus its pad
            "xrow": xrow,
            "NS": ns,
            "NB": key.nblocks,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "debug": key.debug,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "onorm_eps": key.onorm_eps,
            "lb": key.lower_bound,
            "kda0": key.kda0,
            "write_idx": key.write_idx,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": al.vals.ptr,
            "xl": gl.xl.ptr,
            "fa": kl.fa.ptr,
            "qkl": kl.qkl.ptr,
            "decay": kl.decay.ptr,
            "glog": kl.glog.ptr,
            "rec": kl.rec.ptr,
            "cap": key.replay_cap,
            "rwin": key.replay_window,
            "unit": key.gemv_unit,
            "x_ahead": key.x_ahead,
            "vloc": kl.vloc.ptr,
            "ol": kl.ol.ptr,
            "onl": kl.onl.ptr,
            "beta": kl.beta.ptr,
            "kda_head": head,
            "hparts": head_parts(s, L) if head else 1,
            "xbuf": plain + fx.Int64(plain_layout(key)["x"][0]),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "replay": replay,
                "core_out": core_out,
                "x_dbg": x_dbg,
                "proj_dbg": proj_dbg,
            },
        }
        for region in ("part", "wgt", "msq", "xrdy", "proj", "qk", "onrm", "rdone"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        # token group g's AttnRes on CTAs 56 g ..: the groups side by side; the
        # GEMV waits for every group's x (XRDY, NA a group) before its first
        # unit (``gemv_groups``)
        for g in range_constexpr(G):
            if (bid >= NA * g) & (bid < NA * (g + 1)):
                attnres(group_view(c, g), bid - NA * g, group_rows(s, g), on, tls)
        if const_expr(G > 1):  # noqa: SIM102
            if bid >= NA:
                enter_stage("k1.gemv")
                gemv_groups(c, bid - NA)
                stamp(on, tls, tid, 6)
        if const_expr(G == 1):  # noqa: SIM102
            if bid >= NA:
                enter_stage("k1.gemv")
                gi = bid - NA
                rg0 = 2 * gi
                rg1 = 2 * gi + 1
                # the first slab's first group, and the second group's first half,
                # at kernel start, so those loads are in flight across the wait for
                # x -- the role split exists for this
                xw = x_wave(s)
                if const_expr(xw):  # noqa: SIM102
                    if c["wave"] == WAVES - 1:
                        x_wave_stage(c, plan[0][0], plan[0][1])
                ops0 = gemv_loads(c, rg0, range(plan[0][0], plan[0][0] + plan[0][1]))
                half0 = min(4, plan[0][1])
                ops1_0 = gemv_loads(c, rg1, range(plan[0][0], plan[0][0] + half0))
                if const_expr(xw):
                    gpu.barrier()
                else:
                    await_x(c)
                acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
                acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
                # ``range_constexpr`` + indexing, not ``enumerate``: this is inside
                # the kernel body, which flydsl AST-rewrites, and a ``for`` over an
                # arbitrary Python iterable with tuple unpacking is not a form the
                # rewriter is meant to see.
                for sl in range_constexpr(len(plan)):
                    i0 = plan[sl][0]
                    ni = plan[sl][1]
                    if const_expr(sl):
                        # the slab's own weights, then the previous slab's readers
                        # before its LDS is overwritten
                        ops0 = gemv_loads(c, rg0, range(i0, i0 + ni))
                        gpu.barrier()
                    if const_expr(sl or not xw):
                        load_x(c, i0, ni)
                    if const_expr(sl == 0):
                        stamp(on, tls, tid, 4)
                    # the second group's first half in flight under the first's
                    # MFMAs. 4 is what the single-slab path has always issued, so a
                    # build that still takes one slab keeps its exact schedule.
                    half = min(4, ni)
                    if const_expr(sl == 0):
                        ops1 = ops1_0
                    else:
                        ops1 = gemv_loads(c, rg1, range(i0, i0 + half))
                    acc0 = gemv_slab(c, ops0, ni, acc0)
                    ops1 = ops1 + gemv_loads(c, rg1, range(i0 + half, i0 + ni))
                    acc1 = gemv_slab(c, ops1, ni, acc1)
                gemv_finish(c, rg0, acc0)
                stamp(on, tls, tid, 5)
                gemv_finish(c, rg1, acc1)
                stamp(on, tls, tid, 6)
        kda_tasks(c, on, tls)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        replay: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        plain: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            replay,
            core_out,
            x_dbg,
            proj_dbg,
            scratch,
            plain,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kda_pre, launch)
    _BUILDS[key] = launch
    return launch
