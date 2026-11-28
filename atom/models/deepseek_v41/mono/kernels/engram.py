# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""An Engram layer's memory gate in one launch: what the runner's column gather
(``col_gather`` SETTLE, a sharded step's) and ``EngramOp.forward`` (MXFP8 quant,
the FP8 ``wkv`` GEMM, ``engram_post_wkv``) do. ``split`` (a column-sharded
step): this rank's own columns of every row, the gate's sums exchanged across
the ranks; unsplit (a token-parallel step's own rows, or TP 1): every column of
the rows given, the sums local. "Own columns" below: the rank's, or all.

1. ``wkv`` (rows: HC keys of ``hidden`` then the shared value, each
   ``hidden`` wide; K the embedding width) on the rank's own columns of the
   key and value rows (``own`` a branch), K split in ``parts`` of ``PART``:
   CTA (j, part) quantizes the part's columns of every row's embedding into
   LDS (group 32, E8M0 ceil: ``quantize_fp8``), then for its row groups j,
   j + gctas / parts, .. the scaled MFMA (a K step a wave) -> the part's
   partial of every row to PARTS [part][row][column] (device scope), then its
   flag (EPF).
2. Row t, branch b a CTA (``gate``): the owed post settled into the
   residual's stream b on the own columns (``mhc_post``'s rounding), the key
   and the value (the parts added in part order, rounded to bf16: the GEMM's
   output), the three sums ``engram_post_wkv`` reduces (r^2, k^2, (r w) k)
   over the own columns pushed to every rank (ENG), every rank's added in
   rank order, the gate, and r + gate v (bf16) into ``out``'s own columns.

The sums run in another order than the original kernels'; every rounding
point is the original's. The other columns of ``out`` are not written (the
next seam reads its own columns only).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    gemv_fp8_loads,
    gemv_fp8_mfmas,
    lds_row,
)
from atom.models.deepseek_v41.mono.kernels.dims import HC, HIDDEN, Dims
from atom.models.deepseek_v41.mono.kernels.mailbox import Mailbox, epoch_of, preg
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.kernels.red import acc_sum
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    butterfly,
    exp_f32,
    fp8_pack4,
    hw_rsq,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import publish
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

WIDTH = 6144  # the embedding's (``engram_hidden_size``)
PART = 1024  # K a part: a scaled MFMA step (128) a wave
RG = 16  # weight rows a row group, rows a token tile
PREFETCH = 3  # row groups' weights in flight a CTA past the one it multiplies

ABI = KernelAbi(
    (
        "res", "sub", "post", "comb", "emb", "w", "ws", "gate_w", "out", "parts",
        "sym", "peers", "rank", "layer", "epoch",
    )
)  # fmt: skip


@dataclass(frozen=True)
class EngramBuild:
    # the rows this launch gates (a token-parallel step: the rank's own)
    rows: int
    tp: int
    eps: float
    # the residual column-sharded (every rank its columns of every row, the
    # gate's sums exchanged); else every column of ``rows`` on this rank
    split: bool = True
    # 0: both stages, the gate waiting on the projection's EPF flags; 1: the
    # projection alone, no flag (``host_project``: in a host layer's launch);
    # 2: the gate alone, on PARTS an earlier launch left (no wait: the stream
    # orders them)
    mode: int = 0
    # the CTAs the projection is placed on (a host layer's idle CTAs)
    ctas: int = BLOCKS

    def __post_init__(self):
        assert HIDDEN % (self.ranks * RG) == 0
        assert self.parts <= self.ctas <= BLOCKS
        assert WIDTH % PART == 0 and PART // 128 == WAVES, "a K step a wave"
        assert self.rows * HC <= BLOCKS, "a (row, stream) a CTA"
        assert self.lds_words * 4 <= 160 * 1024

    @property
    def step_rows(self) -> int:
        """The step's rows (the peer layout's): a token-parallel step's every
        rank's ``rows``."""
        return self.rows if self.split else self.rows * self.tp

    @property
    def ranks(self) -> int:
        """The ranks a row's columns are split over."""
        return self.tp if self.split else 1

    @property
    def own(self) -> int:
        """A branch's own columns (the rank's)."""
        return HIDDEN // self.ranks

    @property
    def rgs(self) -> int:
        return self.own // RG

    @property
    def groups(self) -> int:
        """Row groups: HC keys' and the value's own rows."""
        return (HC + 1) * self.rgs

    @property
    def parts(self) -> int:
        return WIDTH // PART

    @property
    def gctas(self) -> int:
        return self.ctas // self.parts * self.parts

    @property
    def stride(self) -> int:
        """Row groups apart a CTA's (the CTAs of a part)."""
        return self.gctas // self.parts

    @property
    def per_cta(self) -> int:
        """A CTA's row groups at most."""
        return -(-self.groups // self.stride)

    @property
    def tiles(self) -> int:
        return -(-self.rows // RG)

    @property
    def x_row(self) -> int:
        return lds_row(PART // 4)

    @property
    def s_row(self) -> int:
        return lds_row(PART // 32)

    # LDS words: x (every tile's rows), their codes, the waves' partials,
    # the gate's sums
    @property
    def a_xs(self) -> int:
        return self.tiles * RG * self.x_row

    @property
    def a_red(self) -> int:
        return self.a_xs + self.tiles * RG * self.s_row

    @property
    def a_sum(self) -> int:
        return self.a_red + WAVES * 64 * 4

    @property
    def lds_words(self) -> int:
        return self.a_sum + 3 * WAVES + 3 * self.ranks


def parts_bytes(rows: int, tp: int) -> int:
    return WIDTH // PART * rows * (HC + 1) * (HIDDEN // tp) * 4


@traced
def quantize_part(c, part):
    """The part's columns of every row's embedding -> MXFP8 in LDS (a thread a
    word of 4 values, a group of 32 = 8 threads; ``quantize_fp8``'s rule)."""
    p, tid = c["key"], c["tid"]
    a = c["args"]
    words = PART // 4
    for k in range_constexpr(-(-p.rows * words // THREADS)):
        wi = tid + k * THREADS
        live = wi < p.rows * words
        wi = fx.min(wi, fx.Int32(p.rows * words - 1))
        t, w = wi // words, wi % words
        v = fx.Vector(
            bo.buffer_load(
                rsrc(a["emb"]), (t * WIDTH + part * PART + 4 * w) // 2,
                vec_width=2, dtype=T.i32,
            )
        )  # fmt: skip
        x = [bf2_f32(v[e // 2])[e % 2] for e in range(4)]
        amax = fx.max(fx.max(abs(x[0]), abs(x[1])), fx.max(abs(x[2]), abs(x[3])))
        amax = butterfly(amax, (1, 2, 4), fx.max)
        code = code_ceil(
            fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX)
        )
        q = [clamp_fp8(e * rcp_pow2(code)) for e in x]
        if live:
            fx.ptr_store(fp8_pack4(*q), c["lds"] + (t * p.x_row + w))
            if w % 8 == 0:
                fx.ptr_store(code, c["lds"] + (p.a_xs + t * p.s_row + w // 8))


@traced
def project(c):
    """CTA (j, part) < ``gctas``: its row groups' partials (module docstring,
    1.), then its EPF flag."""
    enter_stage("eng.project")
    p, t, tid = c["key"], c["t"], c["tid"]
    a = c["args"]
    if t < p.gctas:
        part, j = t % p.parts, t // p.parts

        def loads(i):
            """Row group j + i stride's weights (clamped past the last)."""
            g = fx.min(j + i * p.stride, fx.Int32(p.groups - 1))
            branch, local = g // p.rgs, g % p.rgs
            rg = (branch * HIDDEN + _first_col(c)) // RG + local
            return gemv_fp8_loads(
                c, a["w"], a["ws"], PART, rg, k_row=WIDTH, k0=part * PART
            )

        # the first groups' weights stream while the part is quantized; then
        # each group's MFMAs with the next PREFETCH groups' loads in flight
        q = [loads(i) for i in range(min(PREFETCH, p.per_cta))]
        quantize_part(c, part)
        gpu.barrier()
        for i in range_constexpr(p.per_cta):
            if const_expr(i + PREFETCH < p.per_cta):
                q.append(loads(i + PREFETCH))
            g = j + i * p.stride
            if g < p.groups:
                _group(c, part, g, q[i])
        if const_expr(p.mode == 0):
            publish(c["mb"].put, c["epf"], t, 1, tid == 0)


@traced
def _group(c, part, g, ops):
    """Row group g's MFMAs (weights ``ops``) against every token tile, its
    part's partial of every row to PARTS."""
    p, tid = c["key"], c["tid"]
    red = c["ldsf"] + p.a_red
    r_parts = rsrc(c["args"]["parts"])
    for tile in range_constexpr(p.tiles):
        n = min(RG, p.rows - RG * tile)
        gemv_fp8_mfmas(
            c, PART, c["lds"] + tile * RG * p.x_row,
            c["lds"] + (p.a_xs + tile * RG * p.s_row), red, ops, rows=n,
        )  # fmt: skip
        gpu.barrier()
        if tid < RG * n:
            r, tk = tid % RG, tid // RG
            v = acc_sum(red, r, tk)
            row = RG * tile + tk
            at = (part * p.rows + row) * (HC + 1) * p.own + g * RG + r
            bo.buffer_store(v, r_parts, at, cache_modifier=CM_DEV)
        gpu.barrier()


@traced
def await_project(c):
    """Every projection CTA's EPF flag, wave 0 a flag a lane, then the CTA."""
    p, lane = c["key"], c["lane"]
    if c["wave"] == 0:
        for k in range_constexpr(-(-p.gctas // 64)):
            c["mb"].poll([(c["epf"], fx.min(lane + 64 * k, fx.Int32(p.gctas - 1)), 1)])
    gpu.barrier()


def _summed(c, t, col):
    """Projected column ``col`` of row t: its parts added in part order,
    rounded to bf16 (the GEMM's output)."""
    p = c["key"]
    s = None
    for part in range_constexpr(p.parts):
        at = (part * p.rows + t) * (HC + 1) * p.own + col
        v = _ld_f32(c["args"]["parts"], at, CM_DEV)
        s = v if s is None else s + v
    return bf16_round(s)


def _first_col(c):
    """The first of this rank's columns (0 unsplit: every column)."""
    p = c["key"]
    return c["rank"] * p.own if const_expr(p.split) else 0


def _ld_f32(ptr, i, cm=0):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32, cache_modifier=cm)
    )


def _ld_bf(ptr, i):
    w = fx.Int32(bo.buffer_load(rsrc(ptr), i // 2, vec_width=1, dtype=T.i32))
    lo, hi = bf2_f32(w)
    return (i % 2 == 0).select(lo, hi)


@traced
def gate(c, i):
    """Row i // HC, stream i % HC (module docstring, 2.)."""
    enter_stage("eng.gate")
    p, tid, lane, wave = c["key"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    t, b = i // HC, i % HC
    rounds = -(-p.own // THREADS)
    c0 = _first_col(c)
    # the residual's streams, the owed post, the mixes: no wait decides them
    post_b = _ld_f32(a["post"], t * HC + b)
    comb_b = [_ld_f32(a["comb"], (t * HC + h) * HC + b) for h in range(HC)]
    rs, ws, cols, lives = [], [], [], []
    for k in range_constexpr(rounds):
        u = tid + k * THREADS
        lives.append(u < p.own)
        col = c0 + fx.min(u, fx.Int32(p.own - 1))
        cols.append(col)
        x = _ld_bf(a["sub"], t * HIDDEN + col)
        rr = [_ld_bf(a["res"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
        v = fx.Float32(fmath.fma(x, post_b, rr[0] * comb_b[0]))
        for h in range_constexpr(1, HC):
            v = fx.Float32(fmath.fma(rr[h], comb_b[h], v))
        rs.append(bf16_round(v))
        ws.append(_ld_f32(a["gate_w"], b * HIDDEN + col))
    if const_expr(p.mode == 0):
        await_project(c)
    ks, vs = [], []
    zero = fx.Float32(0.0)
    s1, s2, s3 = zero, zero, zero
    for k in range_constexpr(rounds):
        loc = cols[k] - c0
        key = _summed(c, t, b * p.own + loc)
        ks.append(key)
        vs.append(_summed(c, t, HC * p.own + loc))
        r = rs[k]
        s1 = s1 + lives[k].select(r * r, zero)
        s2 = s2 + lives[k].select(key * key, zero)
        s3 = s3 + lives[k].select((r * ws[k]) * key, zero)
    sums = [butterfly(s, (1, 2, 4, 8, 16, 32)) for s in (s1, s2, s3)]
    red = c["ldsf"] + p.a_sum
    if lane == 0:
        for e in range_constexpr(3):
            fx.ptr_store(sums[e], red + (e * WAVES + wave))
    gpu.barrier()
    tot = []
    for e in range_constexpr(3):
        s = fx.ptr_load(red + e * WAVES)
        for w in range_constexpr(1, WAVES):
            s = s + fx.ptr_load(red + (e * WAVES + w))
        tot.append(s)
    total = tot
    if const_expr(p.split):
        total = _exchanged(c, t, b, tot)
    dim = fx.Float32(float(HIDDEN))
    rstd = hw_rsq(total[0] / dim + fx.Float32(p.eps))
    rstd = rstd * hw_rsq(total[1] / dim + fx.Float32(p.eps))
    dot = (total[2] * rstd) * fx.Float32(HIDDEN**-0.5)
    mag = fx.Float32(fmath.sqrt(fx.max(abs(dot), fx.Float32(1e-6))))
    signed = (
        (mag.bitcast(fx.Int32) & 0x7FFFFFFF)
        | (dot.bitcast(fx.Int32) & fx.Int32(-(1 << 31)))
    ).bitcast(fx.Float32)
    g = fx.Float32(1.0) / (fx.Float32(1.0) + exp_f32(-signed))
    for k in range_constexpr(rounds):
        if lives[k]:
            o = bf16_round(rs[k] + g * vs[k])
            bo.buffer_store(
                o.to(fx.BFloat16), rsrc(a["out"]), (t * HC + b) * HIDDEN + cols[k]
            )


@traced
def _exchanged(c, t, b, tot):
    """Row t, stream b's three sums over this rank's columns (``tot``) to
    every rank (ENG), every rank's back added in rank order."""
    p, tid, lane, wave = c["key"], c["tid"], c["lane"], c["wave"]
    red = c["ldsf"] + p.a_sum
    if tid < 3 * p.ranks:
        dst, e = tid // 3, tid % 3
        at = ((t * HC + b) * p.ranks + c["rank"]) * 3 + e
        mine = (e == 0).select(tot[0], (e == 1).select(tot[1], tot[2]))
        for r in range_constexpr(p.ranks):
            if dst == r:
                c["mb"].put(c["eng_fan"][r], at, mine)
    if wave == 0:
        k = fx.min(lane, fx.Int32(3 * p.ranks - 1))
        (got,) = c["mb"].poll([(c["eng"], (t * HC + b) * p.ranks * 3 + k, 1)])
        if lane < 3 * p.ranks:
            fx.ptr_store(got[0].bitcast(fx.Float32), red + (3 * WAVES + lane))
    gpu.barrier()
    total = []
    for e in range_constexpr(3):
        s = fx.ptr_load(red + (3 * WAVES + e))
        for r in range_constexpr(1, p.ranks):
            s = s + fx.ptr_load(red + (3 * WAVES + 3 * r + e))
        total.append(s)
    return total


@traced
def gates(c):
    """CTA i < rows HC: row i // HC, stream i % HC."""
    if c["t"] < c["key"].rows * HC:
        gate(c, c["t"])


def host_project(key: EngramBuild, t, tid, lds, rank, emb, w, ws, parts):
    """The projection (module docstring, 1.) of ``key`` (mode 1) inside
    another launch: its CTA ``t`` (< ``key.gctas``, the host's dense index of
    this CTA) with the host's ``lds`` (an Int32 pointer to ``key.lds_words``
    words free until the call returns): the same tasks, quantization, MFMAs and
    PARTS layout as the projection launch's, no EPF flag."""
    assert key.mode == 1
    c = {
        "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64, "t": t,
        "rank": rank, "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
        "args": {"emb": emb, "w": w, "ws": ws, "parts": parts},
    }  # fmt: skip
    project(c)
    gpu.barrier()


def build_engram(key: EngramBuild):
    keyed = key_tuple(key, SOURCES)
    peer = peer_offsets(key.step_rows, Dims(key.tp))
    name = kernel_symbol(
        "v41_engram", s=key.rows, tp=key.tp, sp=int(key.split),
        **({"m": key.mode} if key.mode else {}),
    )  # fmt: skip

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def engram(
        res: Int64, sub: Int64, post: Int64, comb: Int64, emb: Int64, w: Int64,
        ws: Int64, gate_w: Int64, out: Int64, parts: Int64, sym: Int64,
        peers: Int64, rank: Int32, layer: Int32, epoch: Int64,
    ):  # fmt: skip
        # the step's epoch: every hand-off of the launch tagged with it (``kernels.mailbox``)
        ep = epoch_of(epoch)
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek().words.ptr
        bases = peer_bases(peers, key.tp)
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "t": fx.block_idx.x, "rank": rank, "S": key.rows,
            "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
            "args": {
                "res": res, "sub": sub, "post": post, "comb": comb, "emb": emb,
                "w": w, "ws": ws, "gate_w": gate_w, "out": out, "parts": parts,
            },
            "mb": Mailbox(layer, epoch=ep),
            "epf": preg(sym, peer["epf"], "epf"),
            "eng": preg(sym, peer["eng"], "eng"),
            "eng_fan": [preg(bases[r], peer["eng"], "eng") for r in range(key.ranks)],
        }  # fmt: skip
        if const_expr(key.mode != 2):
            project(c)
        if const_expr(key.mode != 1):
            gates(c)
        _ = keyed

    @launcher(name)
    def launch(
        res: Int64, sub: Int64, post: Int64, comb: Int64, emb: Int64, w: Int64,
        ws: Int64, gate_w: Int64, out: Int64, parts: Int64, sym: Int64,
        peers: Int64, rank: Int32, layer: Int32, epoch: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        engram(
            res, sub, post, comb, emb, w, ws, gate_w, out, parts, sym, peers, rank,
            layer, epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(engram, launch)
    return launch
