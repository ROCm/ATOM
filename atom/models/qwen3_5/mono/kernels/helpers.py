# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device helpers of the Qwen3.5 mono layer kernel.

Every helper is a plain function whose first argument is the kernel's ``LaunchState``: the values of
one launch (``mbox``, ``lane`` / ``wave`` / ``tid``, scratch layout, LDS pointers, task bases, ...).
``LaunchState`` also exposes each helper bound to it (``st.put``, ``st.poll``, ...), and helpers call
one another through it, so a kernel that installs its own variant (``st.poll = partial(poll, st,
batch=...)``, or a function of its own) changes it for every helper. Variants are
keyword arguments (``poll``, ``peer_reduce``) or separate functions, each a build-time choice.
"""

from functools import partial

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5.mono.kernels.ops import (
    ROW_TILE,
    SCALE_BM,
    ld_f32,
    lds_ld,
    lds_st,
)
from atom.models.qwen3_5.mono.sources import SOURCES
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    mxfp4_to_bf16x8,
    traced,
    wave_sum,
    xshfl,
)
from atom.mono.device.stamps import stamp
from atom.mono.device.sync import preg, sreg
from atom.mono.plan.execution import THREADS, WAVES
from atom.mono.plan.trace import enter_stage

# a kernel captures this digest of the model's kernel sources and the mono framework
# (``sources.SOURCES``): the JIT cache keys a helper's source only when its walk reaches it
SHARED_SOURCE_KEY = SOURCES

# --------------------------------------------------------------------------- tagged-pair mailboxes


def mb(st, name):
    """Scratch mailbox region ``name`` (an ``Addr``: the mailbox puts and polls it at device scope)."""
    return sreg(st.scratch, st.SC[name], name)


def put2(st, a, i, v0, v1):
    """Pairs i, i+1 (i even) := (v0, tag), (v1, tag) in one 16-byte store."""
    st.mbox.put_words(
        a, i, [fx.Float32(v0).bitcast(fx.Int32), fx.Float32(v1).bitcast(fx.Int32)]
    )


def get(st, base_addr, i):
    poll = st.poll
    return poll([(base_addr, i, 1)])[0][0]


def getf(st, base_addr, i):
    get = st.get
    return get(base_addr, i).bitcast(fx.Float32)


def getf_many(st, specs):
    """[(base, i)] single pairs -> list of f32."""
    poll = st.poll
    return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]


def get2_many(st, specs):
    """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
    poll = st.poll
    return [
        (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
        for v in poll([(b, i, 2) for b, i in specs])
    ]


# --------------------------------------------------------------------- task placement


def start(st, name):
    """This CTA's first task of stage ``name`` (every stage's task loop starts here, in stage order: the
    traced hand-off contract records the stage)."""
    enter_stage(name)
    stamp(st.timeline, st.tls, st.tid, 1 + list(st.base).index(name))
    G = st.G
    base = st.base
    bid = st.bid
    return (bid + (G - base[name])) & (G - 1)


# -------------------------------------------------------------------------------- block reductions


@traced
def block_sums(st, vs):
    """Block-wide sums of several per-thread values with one LDS exchange."""
    lane = st.lane
    red = st.red
    wave = st.wave
    ws = [wave_sum(v) for v in vs]
    if lane == 0:
        for i in range_constexpr(len(vs)):
            lds_st(red, i * WAVES + wave, ws[i])
    gpu.barrier()
    tots = []
    for i in range_constexpr(len(vs)):
        t = lds_ld(red, i * WAVES)
        for w in range_constexpr(1, WAVES):
            t = t + lds_ld(red, i * WAVES + w)
        tots.append(t)
    gpu.barrier()
    return tots


@traced
def block_sum(st, v):
    lane = st.lane
    red = st.red
    wave = st.wave
    w = wave_sum(v)
    if lane == 0:
        lds_st(red, wave, w)
    gpu.barrier()
    t = lds_ld(red, 0)
    for i in range_constexpr(1, WAVES):
        t = t + lds_ld(red, i)
    gpu.barrier()
    return t


# --------------------------------------------------------------------------------- MFMA GEMV units


def unit_f8f8(st, w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
    """Issue one 128-k chunk (64-k chunks kc, kc + 1) of row group ``rg`` against the
    FP8 activation at LDS word ``b_word``; ``coef()`` = activation scale (x route weight).
    """
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(
            bo.buffer_load(
                w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32
            )
        )
        for h in range(2)
    ]
    s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
    return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)


def unit_bf16_rows(st, w_rsrc, rg, kc, K, b_word, ln=None):
    """Issue one 64-k chunk of row group ``rg`` of a row-major BF16 ``[N, K]`` matrix (ATOM's unshuffled
    layout): lane ``16 * g + r`` reads K 64 kc + 16 g .. + 15 of row r (32 contiguous bytes), the activation
    order of the ``fp8s`` units."""
    lane = st.lane
    ln = lane if ln is None else ln
    base = ((rg * 16 + ln % 16) * K + kc * 64 + (ln // 16) * 16) // 2
    wv = [
        fx.Vector(bo.buffer_load(w_rsrc, base + 4 * h, vec_width=4, dtype=T.i32))
        for h in range(2)
    ]
    return ("bf16s", wv, None, b_word + (lane // 16) * 8)


def unit_bf16(st, w_rsrc, rg, kc, NKC, b_word, ln=None):
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(
            bo.buffer_load(
                w_rsrc,
                (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for sp in range(2)
    ]
    return ("bf16", wv, None, b_word + (lane // 16) * 4)


def unit_mxfp4_atom(st, w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
    """Issue one 128-K tile of an MXFP4 bank in ATOM's gfx950 layout and its E8M0 scale (lane
    ``16 * kl + r`` holds one 32-K scale block of row ``r``; the scale dword's four bytes cover
    tiles kc (even, odd) x row groups (even, odd))."""
    lane = st.lane
    ln = lane if ln is None else ln
    k1n = -(-(K // 32) // 8)  # scale K blocks, padded to 8, in groups of 8
    raw = fx.Vector(
        bo.buffer_load(
            w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32
        )
    )
    word = fx.Int32(
        bo.buffer_load(
            s_rsrc, ((rg // 2) * k1n + kc // 2) * 64 + ln, vec_width=1, dtype=T.i32
        )
    )
    sc_byte = word.shrui(((kc % 2) * 2 + rg % 2) * 8) & fx.Int32(0xFF)
    return ("mxfp4_atom", (raw, sc_byte), coef, b_word + (lane // 16) * 16)


def mma_units(st, acc, units):
    """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
    xs = st.xs
    v4f = fx.Vector.make_type(4, fx.Float32)
    for unit_format, wv, coef, bw in units:
        if const_expr(callable(coef) and unit_format != "mxfp4_atom"):
            coef = coef()
        if const_expr(unit_format == "mxfp4_atom"):
            # step sp takes K 32 * kl + 8 * sp .. of the lane's block (B words 16 * kl + 4 * sp)
            raw, sc_byte = wv
            c = (
                fx.Vector.from_elements(acc, fx.Float32)
                if coef is None
                else fx.Vector.filled(4, 0.0, fx.Float32)
            )
            sc = (sc_byte << fx.Int32(23)).bitcast(fx.Float32)
            for sp in range_constexpr(4):
                a = mxfp4_to_bf16x8(raw[sp], sc)
                b = fx.ptr_load(xs + (bw + sp * 4), result_type=v4f).bitcast(
                    fx.BFloat16
                )
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(coef is None):
                acc = [c[e] for e in range(4)]
            else:
                f = coef() if const_expr(callable(coef)) else coef
                acc = [acc[e] + c[e] * f for e in range(4)]
            continue
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
            a = fx.Vector.from_elements(
                [wv[h][e] for h in range(2) for e in range(4)], fx.Int32
            )
            bv = [
                fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(
                    fx.Int32
                )
                for h in range(2)
            ]
            b = fx.Vector.from_elements(
                [bv[h][e] for h in range(2) for e in range(4)], fx.Int32
            )
            one = fx.Int32(127)
            c = fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]
                )
            )
        else:  # bf16 / bf16s: two K32 MFMAs
            for sp in range_constexpr(2):
                a = wv[sp].bitcast(fx.BFloat16)
                b = fx.ptr_load(
                    xs + (bw + sp * (4 if unit_format == "bf16s" else 16)),
                    result_type=v4f,
                ).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
        if const_expr(coef is None):
            acc = [acc[e] + c[e] for e in range(4)]
        elif const_expr(unit_format == "f8f8"):
            # aiter's blockscale GEMMs: c = fma(block partial, a_scale * w_scale, c)
            acc = [fmath.fma(c[e], coef, acc[e]) for e in range(4)]
        else:
            acc = [acc[e] + c[e] * coef for e in range(4)]
    return acc


def run_units(st, make_unit, cpw, batch, pre=None, each=None):
    """Software pipelined: issue batch b+1's loads before computing batch b.
    ``pre`` = the already-issued first batch (prefetched before a wait). With ``each``, every unit is
    computed on its own and handed over, ``each(c, tile)``, instead of summed."""
    mma_units = st.mma_units
    acc = [fx.Float32(0.0) for _ in range(4)]
    starts = list(range(0, cpw, batch))
    cur = pre if pre is not None else [make_unit(c) for c in range(min(batch, cpw))]
    for bi in range_constexpr(len(starts)):
        nxt = None
        if const_expr(bi + 1 < len(starts)):
            n0 = starts[bi + 1]
            nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
        if const_expr(each is not None):
            for i in range_constexpr(len(cur)):
                each(
                    starts[bi] + i,
                    mma_units([fx.Float32(0.0) for _ in range(4)], [cur[i]]),
                )
        else:
            acc = mma_units(acc, cur)
        cur = nxt
    return acc


def bf16_combine(rule, parts):
    """The sum of a BF16 GEMM's per-wave fp32 partials ``parts`` (each a contiguous K share, in K order)
    as baseline ATOM's GEMM rounds it (``layout.BF16_GEMM_RULES``): ``("single",)`` one fp32 sum;
    ``("splitk", n)`` n equal K ranges, each rounded to bf16 and added into a bf16 sum in range order;
    ``("halves",)`` the even and the odd partials (each 256-wide K tile's first / second 128) summed,
    rounded to bf16 and added, bf16."""
    W = len(parts)
    if const_expr(rule[0] == "single"):
        tot = parts[0]
        for j in range_constexpr(1, W):
            tot = tot + parts[j]
        return tot
    if const_expr(rule[0] == "halves"):
        assert W % 2 == 0, W
        h0 = parts[0]
        h1 = parts[1]
        for j in range_constexpr(2, W, 2):
            h0 = h0 + parts[j]
            h1 = h1 + parts[j + 1]
        return bf16_round(bf16_round(h0) + bf16_round(h1))
    assert rule[0] == "splitk" and W % rule[1] == 0, (rule, W)
    per = W // rule[1]
    acc = fx.Float32(0.0)
    for i in range_constexpr(rule[1]):
        t = parts[i * per]
        for j in range_constexpr(1, per):
            t = t + parts[i * per + j]
        # the zeroed bf16 output plus the first range is that range's rounding
        acc = bf16_round(acc + bf16_round(t))
    return acc


def _wave_parts(red, wpr, rl, n, r):
    """The wpr waves' partials of output (row rl, column n) in ``red``, in wave (K) order."""
    parts = []
    for j in range(wpr):
        ww = (rl // 16) * wpr + j
        parts.append(lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4))
    return parts


def _wave_total(rule, parts):
    if rule is None:
        tot = fx.Float32(0.0)
        for p in parts:
            tot = tot + p
        return tot
    return bf16_combine(rule, parts)


@traced
def reduce_rows(st, R, acc, emit, count=None, rule=None):
    """Sum per-wave MFMA tiles; emit(row_local, local sample column, value). With ``rule`` (a BF16 GEMM's,
    see ``bf16_combine``) the waves' partials are combined as baseline ATOM's GEMM rounds them.
    """
    S = st.S
    lane = st.lane
    red = st.red
    tid = st.tid
    wave = st.wave
    count = S if count is None else count
    wpr = WAVES // R
    fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
    gpu.barrier()
    n_out = R * 16 * count
    for i in range_constexpr((n_out + THREADS - 1) // THREADS):
        t = tid + i * THREADS
        if t < n_out:
            rl = t % (R * 16)
            n = t // (R * 16)
            r = rl % 16
            emit(rl, n, _wave_total(rule, _wave_parts(red, wpr, rl, n, r)))


def emit_out(st, stride):
    outs = st.outs

    def f(rl, n, v):
        lds_st(outs, n * stride + rl, v)

    return f


def n_sel(st, count=None):
    """This lane's local MFMA B column; inactive columns duplicate the last one."""
    S = st.S
    lane = st.lane
    count = S if count is None else count
    return fx.min(lane % 16, count - 1)


# ------------------------------------------------------------------------------ activation staging


def rmsnorm_tail_ks(st, n):
    """This thread's group-of-4 starting indices of n elements. A ragged tail is
    clamped to the last group (idempotent rewrites); ``active`` masks its
    contribution to sums (None when there is no tail)."""
    tid = st.tid
    nq = n // 4
    full = nq // THREADS
    ks = [(tid + i * THREADS) * 4 for i in range(full)]
    active = None
    if const_expr(nq % THREADS):
        w = tid + full * THREADS
        active = w < nq
        ks.append(fx.min(w, nq - 1) * 4)
    return ks, active


@traced
def stage_x_pairs(st, name, n_total, src_of):
    """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
    (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
    tid = st.tid
    xs = st.xs
    mb = st.mb
    poll = st.poll
    nq = n_total // 4
    full = nq // THREADS
    vals = poll(
        [(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)]
    )
    for i in range_constexpr(full):
        for j in range_constexpr(2):
            lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
    if const_expr(nq % THREADS):
        w = tid + full * THREADS
        if w < nq:
            v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
            for j in range_constexpr(2):
                lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))


@traced
def st_f8(st, k, q0, q1):
    """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
    k ^ 2) in K order (the order aiter's ``shuffle_weight`` (16, 16) matrices take).
    Call from the whole wave."""
    lane = st.lane
    xs = st.xs
    w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
    nb = xshfl(w, 1)
    if lane % 2 == 0:
        lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))


# ---------------------------------------------------------------------------------- TP peer reduce


@traced
def peer_reduce(
    st,
    region,
    t,
    residual,
    out_fn,
    tile=ROW_TILE,
    *,
    residual_f32=False,
):
    """Push BF16 partials in tagged pairs to every peer, then sum all ranks' pairs from the own
    symmetric buffer in rank order (W = 1: no exchange). As aiter's fused all-reduce + RMSNorm, the
    rank sum is rounded to bf16 before the residual joins in f32: out_fn(s, row, v0, v1) gets
    ``residual + bf16(sum)``, unrounded. ``residual`` is fn(s, row) -> (r0, r1) (plain loads, issued
    first), a mailbox base (pairs s * HIDDEN + row, polled with the peers; with ``residual_f32`` one f32 a
    pair, the residual its bf16 rounding), or None when the caller adds its own.
    """
    HIDDEN = st.HIDDEN
    S = st.S
    SY = st.SY
    W = st.W
    lane = st.lane
    outs = st.outs
    peer_dst = st.peer_dst
    rank = st.rank
    sym = st.sym
    tid = st.tid
    wave = st.wave
    poll = st.poll
    put_bf = st.put_bf
    if const_expr(W > 1):
        # one wave per destination peer
        if wave < W:
            pair_count = S * tile // 2
            for batch in range_constexpr((pair_count + 63) // 64):
                pair = lane + batch * 64
                if pair < pair_count:
                    si = pair // (tile // 2)
                    ri = (pair % (tile // 2)) * 2
                    put_bf(
                        preg(peer_dst, SY[region], region),
                        (rank * S + si) * HIDDEN + t * tile + ri,
                        [
                            lds_ld(outs, si * tile + ri),
                            lds_ld(outs, si * tile + ri + 1),
                        ],
                    )
        gpu.barrier()
    if tid < S * tile // 2:
        s = tid // (tile // 2)
        r = (tid % (tile // 2)) * 2
        row = t * tile + r
        r0 = fx.Float32(0.0)
        r1 = fx.Float32(0.0)
        if const_expr(callable(residual)):
            r0, r1 = residual(s, row)
        v0 = lds_ld(outs, s * tile + r)
        v1 = lds_ld(outs, s * tile + r + 1)
        res_spec = []
        if const_expr(residual is not None and not callable(residual)):
            res_spec = [
                (
                    (residual, s * HIDDEN + row, 2)
                    if residual_f32
                    else (residual, (s * HIDDEN + row) // 2, 1)
                )
            ]
        if const_expr(W == 1):
            parts = [(v0, v1)]
            got = poll(res_spec) if res_spec else []
        else:
            own = preg(sym, SY[region], region)
            specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
            got = poll(specs + res_spec)
            parts = [bf2_f32(v[0]) for v in got[:W]]
            got = got[W:]
        if const_expr(res_spec and residual_f32):
            r0 = bf16_round(got[0][0].bitcast(fx.Float32))
            r1 = bf16_round(got[0][1].bitcast(fx.Float32))
        elif const_expr(res_spec):
            r0, r1 = bf2_f32(got[0][0])
        t0 = fx.Float32(0.0)
        t1 = fx.Float32(0.0)
        for src in range_constexpr(W):
            t0 = t0 + parts[src][0]
            t1 = t1 + parts[src][1]
        out_fn(s, row, r0 + bf16_round(t0), r1 + bf16_round(t1))


HELPERS = (
    mb,
    put2,
    get,
    getf,
    getf_many,
    get2_many,
    start,
    block_sums,
    block_sum,
    unit_f8f8,
    unit_bf16,
    unit_bf16_rows,
    unit_mxfp4_atom,
    mma_units,
    run_units,
    reduce_rows,
    emit_out,
    n_sel,
    rmsnorm_tail_ks,
    stage_x_pairs,
    st_f8,
    peer_reduce,
)


class LaunchState:
    """One launch's values for the shared helpers, and the helpers bound to it (``st.put`` ...).

    ``source_key`` is the kernel's captured ``SHARED_SOURCE_KEY`` (see there)."""

    def __init__(
        self,
        *,
        source_key,
        S,
        tid,
        lane,
        wave,
        bid,
        G,
        base,
        scratch,
        SC,
        mbox,
        red,
        outs,
        xs,
        rank=None,
        W=1,
        HIDDEN=None,
        peer_dst=None,
        sym=None,
        SY=None,
        timeline=False,
        tls=None,
    ):
        assert (
            source_key == SHARED_SOURCE_KEY
        ), "the kernel was built against older shared helper sources"
        self.S = S
        self.tid = tid
        self.lane = lane
        self.wave = wave
        self.bid = bid
        self.G = G
        self.base = base
        self.scratch = scratch
        self.SC = SC
        self.mbox = mbox
        self.red = red
        self.outs = outs
        self.xs = xs
        self.rank = rank
        self.W = W
        self.HIDDEN = HIDDEN
        self.peer_dst = peer_dst
        self.sym = sym
        self.SY = SY
        self.timeline = timeline
        self.tls = tls
        for fn in HELPERS:
            setattr(self, fn.__name__, partial(fn, self))
        self.put, self.put_bf, self.poll = mbox.put, mbox.put_bf, mbox.poll
