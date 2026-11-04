# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/attention/back.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K2a ``attn_post``: one Kimi-K3 KDA layer from the gated core output to the
MoE's input row -- o_proj, its TP all-reduce inside the kernel, and the MLP
AttnRes seam.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    gemv CTAs 56 ..   o_proj (448 x 16 rows, K = 1536 local): bf16 GEMV of the
                      core rows (plain loads: the previous launch wrote them) ->
                      bf16 partial -> pushed to every rank's AR region (system
                      scope, this rank's slot)
    attn CTAs 0 .. 55 per 128 columns: the 8 ranks' partials summed in rank
                      order (fp32 -> bf16: the custom all-reduce's result) ->
                      LDS delta; then K1's AttnRes stages with the MLP's weights:
                      prefix + delta -> prefix out, the sources' softmax mix,
                      post_attention_layernorm -> the MoE input x

No rank can overwrite a peer's unread partials: to push layer L's partials
a rank must have passed layer L - 1's reduce, which needed every peer's L - 1
push, and the mailbox tag tells the two layers apart. Between steps the region
is zeroed and peer-fenced (``StepMailboxes.begin_step``).
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.kernels import token_groups
from atom.models.kimi_k3.mono.kernels.attention.kda import (
    _SOURCES,
    COLS,
    HIDDEN,
    LDS_PAD,
    NA,
    PROJ,
    ROWS,
    TG,
    groups,
    seam_moves,
    stage_gate,
    stage_mix,
    stage_slice,
)
from atom.mono.device import gemv
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    kernel_symbol,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.ranks import peer_bases, sum_partials
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import PAIR_BYTES, pair_layout
from atom.mono.runtime.abi import KernelAbi

TP = 8
OROWS = HIDDEN  # o_proj rows (full hidden, a rank's K slice)
OTASKS = OROWS // ROWS  # 448
OK = PROJ  # K = 1536 a rank
OKCH = OK // 128  # 12 chunks
GEMV0 = NA  # the first gemv CTA
GEMV_N = BLOCKS - NA  # 200
TL_POINTS = 6  # start, gemv, pushed, reduced, slice/gate, mix

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPostBuild:
    # ``sym``: the field's short name in the kernel symbol (``symbol_params``)
    tokens: int = field(metadata={"sym": "s"})
    # the MLP AttnRes block sources
    nblocks: int = field(metadata={"sym": "nb"})
    eps: float = 1e-5
    out_eps: float = 1e-5
    timeline: bool = field(default=False, metadata={"sym": "t"})


ABI = KernelAbi(
    (
        "core",
        "w_o",
        "prefix",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "x_out",
        "scratch",
        "peers",
        "rank",
        "layer",
        "tl",
    )
)


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


def peer_bytes(max_tokens):
    """The standalone launch's peer regions: AR, then DG."""
    return (ar_pairs(max_tokens) + dg_pairs(max_tokens)) * 8


def scratch_layout(key: AttnPostBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA * groups(s),
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: AttnPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


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
            r_w, rg * ROWS, OK, fx.min(wave + WAVES * i, OKCH - 1) * 128, lane, paired
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
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
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


_BUILDS: dict = {}


def build(key: AttnPostBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    # ``stage_reduce`` strides, so only ``stage_o``'s single pass binds here
    assert s * (ROWS // 2) <= THREADS
    lay = scratch_layout(key)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, s * (OK + LDS_PAD) // 2, 16]
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = kernel_symbol("k3_mono_attn_post", **symbol_params(key))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_post(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        x_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(layer)
        bases = peer_bases(peers, TP)
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "xl": lds.xl.ptr,
            "vals": lds.vals.ptr,
            "delta_lds": lds.dl.ptr,
            "xbuf": x_out,
            "ar_peer": [preg(b, 0, "ar") for b in bases],
            "ar_own": preg(load_own(peers, rank), 0, "ar"),
            "dg_peer": [preg(b, ar_pairs(s) * 8, "dg") for b in bases],
            "dg_own": preg(load_own(peers, rank), ar_pairs(s) * 8, "dg"),
            "args": {
                "core": core,
                "w_o": w_o,
                "prefix": prefix,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "x_dbg": 0,
            },
        }
        for region in ("part", "wgt", "msq", "xrdy"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid >= GEMV0:
            gi = bid - GEMV0
            # every task's weights in flight first: nothing here polls
            ops = [o_loads(c, o_task_or(gi, gi + GEMV_N * j)) for j in range(3)]
            load_core(c)
            stamp(on, tls, tid, 1)
            for j in range_constexpr(3):
                if gi + GEMV_N * j < OTASKS:
                    stage_o(c, gi + GEMV_N * j, ops[j])
            stamp(on, tls, tid, 2)
        if bid < NA:
            stage_reduce(c, bid)
            stamp(on, tls, tid, 3)
            stage_slice(c, bid)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 4)
            stage_mix(c, bid)
            stamp(on, tls, tid, 5)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        x_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        attn_post(
            core,
            w_o,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            x_out,
            scratch,
            peers,
            rank,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(attn_post, launch)
    _BUILDS[key] = launch
    return launch


def load_own(peers, rank):
    """This rank's own peer buffer base (its entry of the table)."""
    from atom.mono.device.ops import load_ptr64

    return load_ptr64(peers, rank)
