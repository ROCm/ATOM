# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A compressing layer's compressor in one launch: what
``DeepseekV41Attention._compress_batch`` runs between K1 and K2 (the
projection, ``fused_compress_attn`` with its ring update, ``wk``,
``write_index_rows_fp4``).

1. The projection ``normed . wkv_gate^T`` (ratio 2: kv | score, 1024 columns;
   ratio 1: kv, 512), K split: CTA (row group g, quarter q) takes 16 weight
   rows x a quarter of K, its 8 waves an eighth of that each (bf16 MFMA, the
   step's rows the B columns); the waves' sums added in wave order in LDS,
   then the CTA's partial of every row to PARTS [row][column][quarter] at
   device scope and its flag (CPF). A projected value is its 4 partials added
   in quarter order (ratio 1: rounded to bf16, the original GEMM's output).
2. A compress plan row is ``parts`` CTAs (``pool``), each a slice of the
   channels, a channel pair a thread: the K = ratio entries of its group (ring
   rows before the step's first row, projected rows past it), the online
   softmax pool; the slice's sum of squares to every slice of the row (CSS),
   the norm over their total (in slice order), the latent (bf16), RoPE at the
   group's first position, the main pool's row (bf16, or planar FP8: group 32,
   E8M0); then the slice's share of the index key ``wk . latent`` (a wave 16
   keys, MFMA, the slice of ``wk`` loaded before any wait) to the row's first
   slice (CKP), which adds the shares in slice order, rounds the key to bf16,
   norms, rotates and codes it (E2M1, E8M0 a 32) into the FP4_MFMA plane.
3. A write plan row a CTA (``ring_row``): the row's kv and score + ape into
   the ring.

A slice reads its rows' partials and its share of ``wk`` alone: a CTA streams
only a few KB, the hand-offs between the slices carry a word a slice and a key.
The pool reads the ring before the step's rows, the ring update writes the
step's rows: the runner admits a build only where those never share a slot
(``rows a request + ratio - 1 <= ring``). The sums run in another order than
the original GEMMs'; every rounding point and quantization is the original's.
``latent`` and ``keys`` (the latent and the index key before its norm, a plan
row each) are outputs too: what a check compares.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import HEAD_DIM, HIDDEN, kvrow_scale0
from atom.models.deepseek_v41.mono.kernels.index_query import FP4_MIN_AMAX, _e2m1
from atom.models.deepseek_v41.mono.kernels.mailbox import Mailbox, sreg
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_pair,
    bf16_round,
    butterfly,
    exp_f32,
    fp8_pack4,
    hw_rsq,
    kernel_symbol,
    launcher,
    ld_f32,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import publish
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

KSTEP = 32  # K an MFMA step
RG = 16  # weight rows a row group (the MFMA's M), step rows a chunk (its N)
QUARTERS = 4  # K slices a row group at most: PARTS' partials a value
ROPE = 64
NOPE_PAIRS = (HEAD_DIM - ROPE) // 2  # the latent's first rotated pair
KEY_NOPE_PAIRS = (ip.DIM - ROPE) // 2  # the index key's
FP4_PAGE = 64  # rows a 64-row group of the FP4_MFMA plane
FP4_GROUP_BYTES = FP4_PAGE * ip.DIM // 2
FP4_GROUP_SCALES = FP4_PAGE * ip.DIM // 32
CP_SLICES = 8  # the most slices a compress plan row (``CompressBuild.parts``)

ABI = KernelAbi(
    (
        "x", "w", "parts", "cplan", "ncap", "wplan", "nwcap", "slots", "kv_state",
        "score_state", "state_stride", "ring", "ape", "norm_w", "cos", "sin",
        "pages", "page_stride", "page_scale", "scale_stride", "tables",
        "table_stride", "per_page", "wk", "k_norm_w", "ivals", "iscales",
        "latent", "keys", "scratch", "layer", "tl",
    )
)  # fmt: skip


def scratch_pairs(rows: int) -> list[tuple[str, int]]:
    """The compressor's mailboxes, region -> tagged pairs: CPF a projection
    CTA's flag (its partials in PARTS, ``parts_bytes``); CSS a plan row's
    slices' sums of squares, [row][slice]; CKP their shares of its index key,
    [row][slice][key]. A plan row a step row at most."""
    return [
        ("cpf", BLOCKS),
        ("css", rows * CP_SLICES),
        ("ckp", rows * CP_SLICES * ip.DIM),
    ]


def scratch_layout(rows: int) -> dict[str, tuple[int, int]]:
    """The launch's mailboxes in its scratch (device scope, zeroed at the
    step's begin), region -> (byte offset, bytes) (``scratch_pairs``)."""
    return pair_layout(scratch_pairs(rows))


def scratch_bytes(rows: int) -> int:
    return max(off + n for off, n in scratch_layout(rows).values())


def parts_bytes(rows: int) -> int:
    """PARTS: the projection's partials, plain f32 [row][column][quarter]
    (uncached: written and read across CTAs at device scope)."""
    return rows * 2 * HEAD_DIM * QUARTERS * 4


# a timeline build's stamps past the start: the projection's flag put, a
# slice's flags in, its sum of squares, its latent, its main pool row, its
# key share put, the key in the plane (a row's first slice)
TL_STAGES = ("proj", "await", "sum", "latent", "main", "share", "key")


@dataclass(frozen=True)
class CompressBuild:
    # the step's rows (the plan's ragged ids index them): the compress and
    # write plans' rows at most (``ncap`` / ``nwcap``, the launch's)
    rows: int
    ratio: int
    # the compressor's norm and the index key's
    eps: float
    k_eps: float
    kv_fp8: bool = False
    timeline: bool = False
    # K slices a row group (``QUARTERS`` at most): the projection's CTAs are
    # its row groups x this
    quarters: int = QUARTERS

    def __post_init__(self):
        assert self.ratio in (1, 2)
        assert self.quarters <= QUARTERS
        assert HIDDEN % (self.quarters * WAVES * KSTEP) == 0
        assert self.proj_ctas <= BLOCKS
        assert self.cap + self.wcap <= BLOCKS, "a plan row a CTA at least"

    @property
    def cap(self) -> int:
        return self.rows

    @property
    def wcap(self) -> int:
        return self.rows

    @property
    def cols(self) -> int:
        """The projection's columns: kv, then (ratio 2) the score."""
        return HEAD_DIM * (2 if self.ratio == 2 else 1)

    @property
    def proj_ctas(self) -> int:
        return self.cols // RG * self.quarters

    @property
    def wsteps(self) -> int:
        """K steps a wave."""
        return HIDDEN // (self.quarters * WAVES * KSTEP)

    @property
    def chunks(self) -> int:
        return -(-self.rows // RG)

    @property
    def parts(self) -> int:
        """Slices a compress plan row: the most (to CP_SLICES) that leave a CTA
        a task."""
        n = CP_SLICES
        while n > 1 and self.cap * n + self.wcap > BLOCKS:
            n //= 2
        return n

    @property
    def pairs(self) -> int:
        """Channel pairs a slice."""
        return HEAD_DIM // 2 // self.parts

    @property
    def tasks(self) -> int:
        return self.cap * self.parts + self.wcap

    @property
    def task0(self) -> int:
        """The plan rows' first CTA: past the projection's where there are
        free ones (the tasks wrap around to CTA 0)."""
        return self.proj_ctas % BLOCKS

    @property
    def ctas(self) -> int:
        return max(self.proj_ctas, min(BLOCKS, self.task0 + self.tasks))

    # LDS words: the projection's wave sums; a slice's latent (bf16 pairs)
    # and its waves' sums of squares
    @property
    def a_red(self) -> int:
        return self.pairs

    @property
    def lds_words(self) -> int:
        return max(WAVES * self.chunks * 64 * 4, self.a_red + WAVES + CP_SLICES)


def _row4(ptr, i):
    """Plan row ``i`` (4 int32), wave-uniform."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i * 4, vec_width=4, dtype=T.i32))
    return [uniform(v[e]) for e in range(4)]


def _word(ptr, i):
    return uniform(fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32)))


def _f32x2(ptr, i):
    """f32 elements i, i + 1 of the buffer at the wave-uniform ``ptr``."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=2, dtype=T.f32))
    return [v[0], v[1]]


def _rope_at(a, comp, rp):
    """cos, sin (bf16 tables, as f32) of RoPE pair ``rp`` at position ``comp``."""
    at = comp * (ROPE // 2) + rp
    return [ld_f32(a[t], at, T.bf16) for t in ("cos", "sin")]


def _state_row(a, slot, pos, ring):
    """Ring entry ``pos`` of ``slot``: its byte offset in either ring (64
    bits: a slot's stride runs the length of the pool), wave-uniform."""
    return (fx.Int64(slot) * a["state_stride"] + fx.Int64(pos % ring) * HEAD_DIM) * 4


@traced
def project(c):
    """CTA (row group, quarter) < ``proj_ctas``: its partial of every row
    (module docstring, 1.), then its CPF flag."""
    enter_stage("cp.project")
    p, t, tid = c["key"], c["t"], c["tid"]
    lane, wave = c["lane"], c["wave"]
    # read outside the traced region (a ``c`` method call in it makes the
    # context the region's state)
    x_wait, x_cm = c.get("x_wait"), c.get("x_cm", 0)
    if t < p.proj_ctas:
        g, q = t // p.quarters, t % p.quarters
        k0 = (q * WAVES + wave) * (p.wsteps * KSTEP) + 8 * (lane // 16)
        wrow = g * RG + lane % 16
        a = [
            fx.Vector(
                bo.buffer_load(
                    c["r_w"], (wrow * HIDDEN + k0 + s * KSTEP) // 2,
                    vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
                )
            )
            for s in range(p.wsteps)
        ]  # fmt: skip
        if const_expr(x_wait is not None):
            # the layer kernel: the rows' norm (K1) in this launch, awaited
            # with the weights in flight
            x_wait()
        xs = []
        for ch in range_constexpr(p.chunks):
            row = fx.min(lane % 16 + RG * ch, fx.Int32(p.rows - 1))
            xs.append(
                [
                    fx.Vector(
                        bo.buffer_load(
                            c["r_x"], (row * HIDDEN + k0 + s * KSTEP) // 2,
                            vec_width=4, dtype=T.i32,
                            cache_modifier=x_cm,
                        )
                    )
                    for s in range(p.wsteps)
                ]
            )  # fmt: skip
        for ch in range_constexpr(p.chunks):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for s in range_constexpr(p.wsteps):
                acc = mfma_bf16(
                    a[s].bitcast(fx.BFloat16), xs[ch][s].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, c["ldsf"] + ((wave * p.chunks + ch) * 64 + lane) * 4)
        gpu.barrier()
        for k in range_constexpr(-(-RG * p.rows // THREADS)):
            i = tid + k * THREADS
            if i < RG * p.rows:
                r, tok = i % RG, i // RG
                at = ((tok // RG) * 64 + tok % RG + RG * (r // 4)) * 4 + r % 4
                s = fx.ptr_load(c["ldsf"] + at)
                for w in range_constexpr(1, WAVES):
                    s = s + fx.ptr_load(c["ldsf"] + (at + w * p.chunks * 256))
                col = g * RG + r
                bo.buffer_store(
                    s, c["r_parts"], (tok * p.cols + col) * p.quarters + q,
                    cache_modifier=CM_DEV,
                )  # fmt: skip
        publish(c["mb"].put, c["cpf"], t, 1, tid == 0)


@traced
def await_groups(c, ranges):
    """The CPF flags of the projection CTAs of the row groups in ``ranges``
    ((first, count) each, the first a traced or Python int), polled by wave 0
    a flag a lane (the CTA's other waves would only add traffic to the polled
    lines), then the CTA past."""
    lane, quarters = c["lane"], c["key"].quarters
    total = sum(n for _, n in ranges) * quarters
    if c["wave"] == 0:
        for k in range_constexpr(-(-total // 64)):
            f = fx.min(lane + 64 * k, fx.Int32(total - 1))
            at, below = f, 0
            for first, n in ranges:
                past = f >= below
                at = past.select(first * quarters + (f - below), at)
                below += n * quarters
            c["mb"].poll([(c["cpf"], at, 1)])
    gpu.barrier()


def proj_pairs(c, cols):
    """Projected columns (col, col + 1) of each (row, col) of ``cols`` (their
    CTAs' flags awaited), one load each: its ``quarters`` partials added in
    quarter order (ratio 1: rounded to bf16)."""
    p = c["key"]
    out = []
    for tok, col in cols:
        pair = []
        for j in range(2):
            v = fx.Vector(
                bo.buffer_load(
                    c["r_parts"], (tok * p.cols + col + j) * p.quarters,
                    vec_width=p.quarters, dtype=T.f32, cache_modifier=CM_DEV,
                )
            )  # fmt: skip
            s = v[0]
            for q in range(1, p.quarters):
                s = s + v[q]
            pair.append(bf16_round(s) if const_expr(p.ratio == 1) else s)
        out.append(pair)
    return out


def wk_share(c, h):
    """Slice h's share of ``wk``: the wave's 16 key rows over the slice's
    channels, as MFMA A operands, a K step each."""
    p, lane, wave = c["key"], c["lane"], c["wave"]
    r_wk = rsrc(c["args"]["wk"])
    k0 = (wave * RG + lane % 16) * HEAD_DIM + h * (2 * p.pairs) + 8 * (lane // 16)
    return [
        fx.Vector(bo.buffer_load(r_wk, (k0 + s * KSTEP) // 2, vec_width=4, dtype=T.i32))
        for s in range(2 * p.pairs // KSTEP)
    ]


@traced
def pool(c, i, h):
    """Slice h of compress plan row ``i`` (module docstring, 2.)."""
    enter_stage("cp.pool")
    p, tid, wave = c["key"], c["tid"], c["wave"]
    a = c["args"]
    crdy = c.get("crdy")  # outside the traced region (``project``)
    rid, bid, pos, wl = _row4(a["cplan"], i)
    if pos >= 0:
        mine = tid < p.pairs
        tt = h * p.pairs + fx.min(tid, fx.Int32(p.pairs - 1))  # the channel pair
        slot = _word(a["slots"], bid)
        ci = pos // p.ratio
        comp = ci * p.ratio
        block = _word(a["tables"], bid * a["table_stride"] + ci // a["per_page"])
        nw = _f32x2(a["norm_w"], 2 * tt)
        cs, sn = _rope_at(a, comp, fx.max(tt - NOPE_PAIRS, fx.Int32(0)))
        rings = []
        for k in range_constexpr(p.ratio):
            s = pos - p.ratio + 1 + k
            off = _state_row(a, slot, fx.max(s, fx.Int32(0)), a["ring"])
            rings.append(
                (
                    s,
                    _f32x2(a["kv_state"] + off, 2 * tt),
                    _f32x2(a["score_state"] + off, 2 * tt),
                )
            )
        apes = [_f32x2(a["ape"], k * HEAD_DIM + 2 * tt) for k in range(p.ratio)]
        ws = wk_share(c, h)
        # the row groups of the slice's kv (and score) columns
        span = 2 * p.pairs // RG
        groups = [(h * span, span)]
        if const_expr(p.ratio == 2):
            groups.append((HEAD_DIM // RG + h * span, span))
        await_groups(c, groups)
        c["stamp"](2)
        v = _pooled(c, rid, wl, tt, rings, apes)
        rstd = _row_rstd(c, i, h, mine, v[0] * v[0] + v[1] * v[1])
        nv = [(v[j] * rstd) * nw[j] for j in range(2)]
        word = bf16_pair(nv[0], nv[1]).bitcast(fx.Int32)
        if mine:
            fx.ptr_store(word, c["lds"] + (tt - h * p.pairs))
            bo.buffer_store(word, rsrc(a["latent"]), i * (HEAD_DIM // 2) + tt)
        c["stamp"](4)
        rot = tt >= NOPE_PAIRS
        e = rot.select(nv[0] * cs - nv[1] * sn, nv[0])
        o = rot.select(nv[1] * cs + nv[0] * sn, nv[1])
        _main_row(c, block, ci % a["per_page"], tt, mine, e, o, rid)
        if const_expr(crdy is not None):
            # the layer kernel: the slice's main row stores drained before its
            # key share (row i's CRDY follows the shares)
            rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()  # the slice's latent in LDS
        c["stamp"](5)
        _key_share(c, i, h, ws)
        c["stamp"](6)
        if h == 0:
            flat = fx.Int64(block) * a["per_page"] + fx.Int64(ci % a["per_page"])
            if wave == 0:
                _key_out(c, i, comp, flat)
            if const_expr(crdy is not None):
                # plan row i's entry (every slice's main row, drained before
                # its share, and its key) readable by this launch's attention
                # and scorer, which await every plan row's (``ncap``)
                publish(c["mb"].put, crdy, i, 1, tid == 0)
            c["stamp"](7)
    if const_expr(crdy is not None):  # noqa: SIM102
        # a padding row (pos -1) has no entry: its CRDY at once, so a waiter
        # on every plan row up to ``ncap`` never waits on it
        if (pos < 0) & (h == 0):
            publish(c["mb"].put, crdy, i, 1, tid == 0)


def _pooled(c, rid, wl, tt, rings, apes):
    """Channel pair tt's pooled values: ``fused_compress_attn``'s online
    softmax over the group's K = ratio entries."""
    p = c["key"]
    if const_expr(p.ratio == 1):
        (kv,) = proj_pairs(c, [(rid, 2 * tt)])
        return kv
    neg = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    m = [neg, neg]
    acc = [zero, zero]
    ws = [zero, zero]
    # every entry's projected row, the ring's too (a row of the step either
    # way): the loads issued together
    rows = [fx.max(rid - (p.ratio - 1 - k), fx.Int32(0)) for k in range(p.ratio)]
    got = proj_pairs(c, [(r, col) for r in rows for col in (2 * tt, HEAD_DIM + 2 * tt)])
    for k in range_constexpr(p.ratio):
        s, rkv, rsc = rings[k]
        kv_in, sc_in = got[2 * k], got[2 * k + 1]
        from_ring = k < wl
        live = s >= 0
        for j in range_constexpr(2):
            kv = from_ring.select(live.select(rkv[j], zero), kv_in[j])
            sc = from_ring.select(live.select(rsc[j], neg), sc_in[j] + apes[k][j])
            m_new = fx.max(m[j], sc)
            scale = (m[j] == neg).select(zero, exp_f32(m[j] - m_new))
            w = (sc == neg).select(zero, exp_f32(sc - m_new))
            acc[j] = acc[j] * scale + w * kv
            ws[j] = ws[j] * scale + w
            m[j] = m_new
    return [acc[j] / ws[j] for j in range(2)]


@traced
def _row_rstd(c, i, h, mine, sq):
    """The row's RMSNorm factor: the slice's sum of squares (its pair threads'
    in wave order) to CSS, every slice's (in slice order) back."""
    p, lane, wave = c["key"], c["lane"], c["wave"]
    s = butterfly(mine.select(sq, fx.Float32(0.0)), (1, 2, 4, 8, 16, 32))
    red = c["ldsf"] + p.a_red
    if lane == 0:
        fx.ptr_store(s, red + wave)
    gpu.barrier()
    s = fx.ptr_load(red)
    for w in range_constexpr(1, -(-p.pairs // 64)):
        s = s + fx.ptr_load(red + w)
    c["stamp"](3)
    if c["tid"] == 0:
        c["mb"].put(c["css"], i * CP_SLICES + h, s)
    # wave 0 polls, a lane a slice; the CTA reads them from LDS
    if wave == 0:
        g = fx.min(lane, fx.Int32(p.parts - 1))
        (got,) = c["mb"].poll([(c["css"], i * CP_SLICES + g, 1)])
        if lane < p.parts:
            fx.ptr_store(got[0].bitcast(fx.Float32), red + (WAVES + lane))
    gpu.barrier()
    tot = fx.ptr_load(red + WAVES)
    for g in range_constexpr(1, p.parts):
        tot = tot + fx.ptr_load(red + (WAVES + g))
    return hw_rsq(tot * fx.Float32(1.0 / HEAD_DIM) + fx.Float32(p.eps))


@traced
def _main_row(c, block, row, tt, mine, e, o, rid):
    """The rotated pair (e, o) of the latent into the main pool's row: bf16,
    or (``kv_fp8``) the bf16 values' E4M3 codes and their group's E8M0; in the
    layer kernel also into ``kvrow`` row S + ``rid`` (``dims.kvrow_bytes``: the
    entry step row ``rid`` completes), where this launch's attention reads it
    (another XCD's L2 may hold the pool row's line)."""
    p = c["key"]
    a = c["args"]
    kvrow = c.get("kvrow")
    step_row = fx.Int64(p.rows + rid)
    if const_expr(not p.kv_fp8):
        dst = (
            a["pages"]
            + fx.Int64(block) * a["page_stride"]
            + fx.Int64(row) * (HEAD_DIM * 2)
        )
        word = bf16_pair(e, o).bitcast(fx.Int32)
        if mine:
            bo.buffer_store(word, rsrc(dst), tt)
        if const_expr(kvrow is not None):
            cdst = kvrow + step_row * (HEAD_DIM * 2)
            if mine:
                bo.buffer_store(word, rsrc(cdst), tt)
    else:
        x = [bf16_round(e), bf16_round(o)]
        amax = butterfly(fx.max(abs(x[0]), abs(x[1])), (1, 2, 4, 8), fx.max)
        code = code_ceil(
            fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX)
        )
        q = [clamp_fp8(v * rcp_pow2(code)) for v in x]
        nb = [xshfl(v, 1) for v in q]
        word = fp8_pack4(q[0], q[1], nb[0], nb[1])
        dst = a["pages"] + fx.Int64(block) * a["page_stride"] + fx.Int64(row) * HEAD_DIM
        if mine & (tt % 2 == 0):
            bo.buffer_store(word, rsrc(dst), tt // 2)
        sdst = (
            a["page_scale"]
            + fx.Int64(block) * a["scale_stride"]
            + fx.Int64(row) * (HEAD_DIM // 32)
        )
        if mine & (tt % 16 == 0):
            bo.buffer_store(fx.Int8(code), rsrc(sdst), tt // 16)
        if const_expr(kvrow is not None):
            cdst = kvrow + step_row * HEAD_DIM
            sdst = kvrow + kvrow_scale0(p.rows) + step_row * (HEAD_DIM // 32)
            if mine & (tt % 2 == 0):
                bo.buffer_store(word, rsrc(cdst), tt // 2)
            if mine & (tt % 16 == 0):
                bo.buffer_store(fx.Int8(code), rsrc(sdst), tt // 16)


@traced
def _key_share(c, i, h, ws):
    """Slice h's share of row i's index key: ``wk``'s slice (``ws``) times
    the slice's latent (LDS, every B column), a wave 16 keys, to CKP."""
    p, lane, wave = c["key"], c["lane"], c["wave"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for s in range_constexpr(2 * p.pairs // KSTEP):
        x = fx.Vector(
            fx.ptr_load(
                c["lds"] + (s * KSTEP + 8 * (lane // 16)) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        acc = mfma_bf16(ws[s].bitcast(fx.BFloat16), x.bitcast(fx.BFloat16), acc)
    if lane % 16 == 0:
        for e in range_constexpr(4):
            k = wave * RG + 4 * (lane // 16) + e
            c["mb"].put(c["ckp"], (i * CP_SLICES + h) * ip.DIM + k, acc[e])


@traced
def _key_out(c, i, comp, flat):
    """Wave 0 of row i's first slice: dims 2 lane, 2 lane + 1 of the key, the
    slices' shares added in slice order and rounded to bf16, normed (bf16),
    rotated (bf16), coded (E2M1, a 32-dim group's E8M0) into the FP4_MFMA
    plane."""
    enter_stage("cp.key")
    p, lane = c["key"], c["lane"]
    a = c["args"]
    # the layer kernel: written through, for the scorer of this launch on
    # another XCD (it reads the plane past the entry's CRDY)
    plane_cm = CM_DEV if c.get("crdy") is not None else 0
    got = c["mb"].poll(
        [(c["ckp"], (i * CP_SLICES + g) * ip.DIM + 2 * lane, 2) for g in range(p.parts)]
    )
    k = []
    for j in range_constexpr(2):
        s = got[0][j].bitcast(fx.Float32)
        for g in range_constexpr(1, p.parts):
            s = s + got[g][j].bitcast(fx.Float32)
        k.append(bf16_round(s))
    bo.buffer_store(
        bf16_pair(k[0], k[1]).bitcast(fx.Int32),
        rsrc(a["keys"]),
        i * (ip.DIM // 2) + lane,
    )
    w = _f32x2(a["k_norm_w"], 2 * lane)
    ss = butterfly(k[0] * k[0] + k[1] * k[1], (1, 2, 4, 8, 16, 32))
    rstd = hw_rsq(ss * fx.Float32(1.0 / ip.DIM) + fx.Float32(p.k_eps))
    kn = [bf16_round((k[j] * rstd) * w[j]) for j in range(2)]
    cs, sn = _rope_at(a, comp, fx.max(lane - KEY_NOPE_PAIRS, fx.Int32(0)))
    rot = lane >= KEY_NOPE_PAIRS
    e = rot.select(bf16_round(kn[0] * cs - kn[1] * sn), kn[0])
    o = rot.select(bf16_round(kn[1] * cs + kn[0] * sn), kn[1])
    amax = butterfly(fx.max(abs(e), abs(o)), (1, 2, 4, 8), fx.max)
    bits = (fx.max(amax, fx.Float32(FP4_MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(
        fx.Int32
    )
    code = (bits >> 23) + fx.Int32(
        ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    )
    inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
    byte = _e2m1(e * inv) | (_e2m1(o * inv) << 4)
    group, t = flat // FP4_PAGE, fx.Int32(flat % FP4_PAGE)
    chunk = lane // 16
    vdst = a["ivals"] + group * FP4_GROUP_BYTES
    bo.buffer_store(
        fx.Int8(byte), rsrc(vdst), (chunk * FP4_PAGE + t) * 16 + lane % 16,
        cache_modifier=plane_cm,
    )  # fmt: skip
    if lane % 16 == 0:
        sdst = a["iscales"] + group * FP4_GROUP_SCALES
        at = (chunk % 2) * 128 + (t % 32) * 4 + (chunk // 2) * 2 + t // 32
        bo.buffer_store(fx.Int8(code), rsrc(sdst), at, cache_modifier=plane_cm)


@traced
def ring_row(c, i):
    """Write plan row ``i``: its row's kv and score + ape into the ring."""
    enter_stage("cp.ring")
    p, tid = c["key"], c["tid"]
    a = c["args"]
    rid, bid, pos, _ = _row4(a["wplan"], i)
    if pos >= 0:
        tt = fx.min(tid, fx.Int32(HEAD_DIM // 2 - 1))
        slot = _word(a["slots"], bid)
        ape = _f32x2(a["ape"], (pos % p.ratio) * HEAD_DIM + 2 * tt)
        await_groups(c, [(0, p.cols // RG)])
        (kv,) = proj_pairs(c, [(rid, 2 * tt)])
        # ratio 1 pools its kv alone: the kv is the gate input too
        sc = kv
        if const_expr(p.ratio == 2):
            (sc,) = proj_pairs(c, [(rid, HEAD_DIM + 2 * tt)])
        off = _state_row(a, slot, pos, a["ring"])
        if tid < HEAD_DIM // 2:
            bo.buffer_store(
                fx.Vector.from_elements(kv, fx.Float32),
                rsrc(a["kv_state"] + off),
                2 * tt,
            )
            bo.buffer_store(
                fx.Vector.from_elements([sc[j] + ape[j] for j in range(2)], fx.Float32),
                rsrc(a["score_state"] + off),
                2 * tt,
            )


@traced
def plan_rows(c):
    """Task i on CTA task0 + i (wrapping): slice i % parts of compress plan
    row i // parts, then the write plan's rows; the rows past the plans'
    (``ncap`` / ``nwcap``) idle."""
    p, a = c["key"], c["args"]
    i = (c["t"] + (BLOCKS - p.task0)) % BLOCKS
    n = p.cap * p.parts
    if (i < n) & (i // p.parts < a["ncap"]):
        pool(c, i // p.parts, i % p.parts)
    if (i >= n) & (i - n < a["nwcap"]):
        ring_row(c, i - n)


def build_compress(key: CompressBuild):
    keyed = key_tuple(key, SOURCES)
    regions = {r: off for r, (off, _) in scratch_layout(key.rows).items()}
    name = kernel_symbol(
        "v41_compress", s=key.rows, r=key.ratio, fp8=int(key.kv_fp8),
        tl=int(key.timeline),
    )  # fmt: skip
    points = 1 + len(TL_STAGES)

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]
        tls: fx.Array[fx.Int64, points if key.timeline else 1, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def compress(
        x: Int64, w: Int64, parts: Int64, cplan: Int64, ncap: Int32, wplan: Int64,
        nwcap: Int32, slots: Int64, kv_state: Int64, score_state: Int64,
        state_stride: Int64, ring: Int32, ape: Int64, norm_w: Int64, cos: Int64,
        sin: Int64, pages: Int64, page_stride: Int64, page_scale: Int64,
        scale_stride: Int64, tables: Int64, table_stride: Int32, per_page: Int32,
        wk: Int64, k_norm_w: Int64, ivals: Int64, iscales: Int64, latent: Int64,
        keys: Int64, scratch: Int64, layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        smem = fx.SharedAllocator().allocate(Smem).peek()
        lds, tls = smem.words.ptr, smem.tls.ptr
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "t": fx.block_idx.x, "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
            "args": {
                "cplan": cplan, "ncap": ncap, "wplan": wplan, "nwcap": nwcap,
                "slots": slots, "kv_state": kv_state, "score_state": score_state,
                "state_stride": state_stride, "ring": ring, "ape": ape,
                "norm_w": norm_w, "cos": cos, "sin": sin, "pages": pages,
                "page_stride": page_stride, "page_scale": page_scale,
                "scale_stride": scale_stride, "tables": tables,
                "table_stride": table_stride, "per_page": per_page, "wk": wk,
                "k_norm_w": k_norm_w, "ivals": ivals, "iscales": iscales,
                "latent": latent, "keys": keys,
            },
            "r_w": rsrc(w, key.cols * HIDDEN * 2),
            "r_x": rsrc(x, key.rows * HIDDEN * 2),
            "r_parts": rsrc(parts, parts_bytes(key.rows)),
            "mb": Mailbox(layer),
            "cpf": sreg(scratch, regions["cpf"], "cpf"),
            "css": sreg(scratch, regions["css"], "css"),
            "ckp": sreg(scratch, regions["ckp"], "ckp"),
        }  # fmt: skip
        c["stamp"] = lambda k: stamp(key.timeline, tls, tid, k)
        stamp_begin(key.timeline, tls, tid, points)
        project(c)
        c["stamp"](1)
        plan_rows(c)
        stamp_flush(key.timeline, tls, tl, tid, fx.block_idx.x, points)
        _ = keyed

    @launcher(name)
    def launch(
        x: Int64, w: Int64, parts: Int64, cplan: Int64, ncap: Int32, wplan: Int64,
        nwcap: Int32, slots: Int64, kv_state: Int64, score_state: Int64,
        state_stride: Int64, ring: Int32, ape: Int64, norm_w: Int64, cos: Int64,
        sin: Int64, pages: Int64, page_stride: Int64, page_scale: Int64,
        scale_stride: Int64, tables: Int64, table_stride: Int32, per_page: Int32,
        wk: Int64, k_norm_w: Int64, ivals: Int64, iscales: Int64, latent: Int64,
        keys: Int64, scratch: Int64, layer: Int32, tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        compress(
            x, w, parts, cplan, ncap, wplan, nwcap, slots, kv_state, score_state,
            state_stride, ring, ape, norm_w, cos, sin, pages, page_stride,
            page_scale, scale_stride, tables, table_stride, per_page, wk, k_norm_w,
            ivals, iscales, latent, keys, scratch, layer, tl,
        ).launch(grid=(key.ctas,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(compress, launch)
    return launch
