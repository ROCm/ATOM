"""Hyper-connection stages: A1/M1 (norm, down, injection) and A2/M2 (up, read,
mixed pack) of the attention and MLP HCs.

The HC state ``h`` is the layer input (4 streams x 2560 bf16); ``w'`` the
folded ``hc_norm.weight`` (``bf16(w + 1)``).

A1, task ``g < 10``: low-rank columns ``32 g .. 32 g + 31``. Every A1 task
normalizes all 4 streams and packs them, then wave ``w`` runs the down chain of
n-block ``2 g + w // 4`` over ``kit = p, p + 4, ..., p + 76`` with
``p = w % 4``. Wave 0 reduces the 4 partials in order, applies ``silu(y / 4)``
and publishes the block's MXFP8 pack. Tasks 10-13 compute injection weight
``t - 10`` from the bf16 normalized streams: the weight row is staged through
LDS so wave 0's 32 dot-product chains read it lane-contiguous.

A2, task ``t`` (80): columns ``32 t .. 32 t + 31`` of every stream. Wave ``w``
runs the up chain of n-block ``2 t + 160 (w // 2) + w % 2`` over k_iters 0-2,
then the read epilogue; wave 0 averages the streams and publishes the mixed
row's fp8 words, bf16 words and exponent. A2 only needs the norm's
statistics, so it reads its own 128 values of ``h`` and ``w'``. From
``INV_ROWS`` request rows A1's task 0 of each row hands A2 the 4 streams'
inverse RMS (``inv`` mailbox) instead: rebuilding every row's statistics costs
A2 more than the hand-off there (and less up to 9 rows).

Both HCs run this code on a descriptor (:func:`attn_hc`, :func:`mlp_hc`): stage
names, weights, mailboxes, and where the streams come from. The attention HC
reads the layer input from memory. The MLP HC's streams are C1's output: M1
polls them from the ``w`` mailbox, and M2 builds the chunk square sums from
C1's 16-column sums (``whs``, the same tree as the norm's chunk sums) instead
of polling all the streams. From ``INV_ROWS`` rows M2 polls M1's inverse RMS
(``minv``) and its own words of the streams.

Batched builds (``c.rows`` request rows) run every row's arithmetic exactly as
one request's. A1 runs every request row's one-request tasks, on that row's
streams and mailboxes: down group ``g`` of row ``r`` is task ``10 r + g``,
injection weight ``i`` task ``10 rows + 4 r + i``. A2 builds every row's
statistics, low-rank pack and products; wave ``r`` publishes row r's mixed
row.
"""

from types import SimpleNamespace

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import (
    bpermute_i32,
    read_lane_i32,
    xred,
)
from atom.models.qwen4_exp.mono.kernels.layout import (
    INJ_LANES,
    INJ_SEG,
    TASKS,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    add,
    bf16r,
    bf16x8,
    lds_row,
    mb_ix,
    mfma_operands,
    payload_wait,
    row_ix,
    task_index,
)

VECS = C.hc_width // 8  # 1280 vectors of 8 bf16 over the 4 streams
STREAM_VECS = C.hidden // 8  # 320
ROUNDS = 3
assert (ROUNDS - 1) * THREADS < VECS <= ROUNDS * THREADS
STREAM_CHUNKS = C.hidden // 32  # 80
INV_HIDDEN = 0x39CCCCCD  # fl(1/2560)
EPS = 0x358637BD  # fl(1e-6)

DOWN_KI = p2.k_iters(C.hc_width)  # 80
DOWN_CHAIN = DOWN_KI // 4  # 20 k_iters per partial
DOWN_SBLOCKS = (
    DOWN_CHAIN + 15
) // 16  # 16-step scale blocks per chain (``p2.scales_chain``)
DOWN_GROUPS = C.hc_lowrank // 32  # 10
INJ_TASK = DOWN_GROUPS  # first injection task
INJ_ROW_VECS = C.hc_width // 8  # 1280 vectors of 8 bf16 per weight row
INJ_VECS = INJ_ROW_VECS // INJ_LANES  # 40 per dot-product lane
INJ_CHUNK = 8  # vectors per scheduling chunk of the chain
assert DOWN_SBLOCKS == 2 and DOWN_CHAIN % 4 == 0  # a1 loads 4 + 1 scale dwords

# Byte sizes of the down weights (both HCs): their resources carry these, so
# a load at ``OOB`` (int32 elements) returns zeros without touching memory.
RECORDS = {
    "w_down": C.hc_lowrank * C.hc_width // 2,
    "s_down": C.hc_lowrank // 16 * 4 * DOWN_SBLOCKS * 64 * 16,
}
OOB = 1 << 28
UP_KI = 3  # of 4 stored k_iters; the 4th is all zero
UP_NB = 2 * C.hidden // 32  # n-blocks per stream: 160
INV_ROWS = 12  # A1 -> A2 and M1 -> M2 hand the inverse RMS over from this many rows
INV_TID = 256  # A2 from INV_ROWS rows: first of the threads polling the inverse RMS
C1_TILES = TASKS["c1"]  # producers of the MLP HC's streams


def attn_hc(c, p):
    """The attention HC. Its streams are the layer input, or in the PLE
    layer P2's ``h'`` (polled; 2 sums per P2 task and stream)."""
    if c.kind == "delta_ple":
        src = dict(
            polled=True, h=c.mb_ph, hs=c.mb_phs, producers=TASKS["p2"], hint_stride=2
        )
    else:
        src = dict(polled=False, h=p.h_in, hs=None, producers=0, hint_stride=0)
    return SimpleNamespace(
        down="a1", up="a2", **src, wide_inv=c.rows >= INV_ROWS, inv=c.mb_inv,
        norm_w=p.hc_norm_w, w_down=p.w_down, s_down=p.s_down, w_up=p.w_up, s_up=p.s_up, w_inj=p.w_inj,
        lr8=c.mb_lr8, lre=c.mb_lre, inj=c.mb_inj, x8=c.mb_x8, xe=c.mb_xe, xbf=c.mb_xbf,
    )  # fmt: skip


def mlp_hc(c, p):
    return SimpleNamespace(
        down="m1", up="m2", polled=True, h=c.mb_w, hs=c.mb_whs, producers=C1_TILES, hint_stride=1,
        wide_inv=c.rows >= INV_ROWS, inv=c.mb_minv,
        norm_w=p.m_norm_w, w_down=p.m_w_down, s_down=p.m_s_down, w_up=p.m_w_up, s_up=p.m_s_up, w_inj=p.m_w_inj,
        lr8=c.mb_mlr8, lre=c.mb_mlre, inj=c.mb_minj, x8=c.mb_mx8, xe=c.mb_mxe, xbf=c.mb_mxbf,
    )  # fmt: skip


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def fma(a, b, c):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32,
            "llvm.fma.f32",
            [
                fx.Float32(a).ir_value(),
                fx.Float32(b).ir_value(),
                fx.Float32(c).ir_value(),
            ],
            [],
            [],
        )
    )


def sigmoid_acc(x):
    """``1 / (1 + expf(-x))``: correctly rounded divide, accurate exp."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - x))


def natural_sum(v):
    """Wave sum in a ``shfl_down`` tree (offsets 1, 2, ..., 32)."""
    for off in (1, 2, 4, 8, 16, 32):
        v = xred(v, off, add)
    return v


def lane_gather(v, first, n):
    """Lane ``l`` gets ``v`` of lanes ``first(l) .. first(l) + n - 1``."""
    bits = v.bitcast(fx.Int32)
    return [bpermute_i32((first + e) * 4, bits).bitcast(fx.Float32) for e in range(n)]


def norm_load(r_h, tid, row=0, keep=None):
    """This thread's ``ROUNDS`` vectors of 8 bf16 (vector ``tid + 512 r``, clamped)
    of request row ``row``; zeros (out of range) where ``keep`` is false."""
    out = []
    for r in range(ROUNDS):
        i = row_ix(fx.min(tid + THREADS * r, VECS - 1) * 4, row, HC_WORDS)
        if keep is not None:
            i = keep.select(i, OOB)
        out.append(fx.Vector(bo.buffer_load(r_h, i, vec_width=4, dtype=T.i32)))
    return out


def poll_streams(c, hc, rows):
    """:func:`norm_load` from the ``w`` mailbox (one pair per bf16 word), of
    each request row of ``rows``, in one poll."""
    specs = []
    for row in rows:
        for r in range(ROUNDS):
            v = fx.min(c.tid + THREADS * r, VECS - 1)
            specs += [(hc.h, mb_ix(v * 4, row), 2), (hc.h, mb_ix(v * 4 + 2, row), 2)]
    w = mbx.poll(specs, c.tag)
    return [
        [
            fx.Vector.from_elements(
                w[2 * (q * ROUNDS + r)] + w[2 * (q * ROUNDS + r) + 1], fx.Int32
            )
            for r in range(ROUNDS)
        ]
        for q in range(len(rows))
    ]


def _chunk_sum(words):
    """Square sum of one 32-chunk as a natural-pairing tree; the
    chunk's 4 vectors sit on 4 adjacent lanes."""
    q = [x * x for x in bf16x8(words)]
    s = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]))
    return xred(xred(s, 1, add), 2, add)


def norm_sums(c, hc, hs, row=0):
    """Square sums of the 32-chunks into LDS ``hsum`` of request row
    ``row``."""
    tid, lane = c.tid, c.lane
    hsum = lds_row(c, c.lds_f, "hsum", row)
    sums = [_chunk_sum([h[e] for e in range(4)]) for h in hs]
    if lane % 4 == 0:
        fx.ptr_store(sums[0], hsum + tid // 4)
        fx.ptr_store(sums[1], hsum + (tid + THREADS) // 4)
        if tid < VECS - 2 * THREADS:
            fx.ptr_store(sums[2], hsum + (tid + 2 * THREADS) // 4)


norm_sums = ASTRewriter.transform(norm_sums)


def halves_sums(c, hc):
    """:func:`norm_sums` from C1's 16-column sums: chunk ``i`` (stream
    ``i // 80``) is the sum of halves ``2 i`` and ``2 i + 1``."""
    tid = c.tid
    if tid < VECS // 4:
        vs = mbx.poll([(hc.hs, mb_ix(tid * 2, r), 2) for r in range(c.rows)], c.tag)
        for r in range_constexpr(c.rows):
            v = vs[r]
            fx.ptr_store(
                _f32(v[0]) + _f32(v[1]),
                c.lds_f + row_ix(c.L.lds["hsum"] + tid, r, c.L.stride["hsum"]),
            )


halves_sums = ASTRewriter.transform(halves_sums)


def norm_inv(c, hc):
    """Inverse RMS of the 4 streams from ``hsum`` into LDS ``inv``. Every
    thread must call this; it starts and ends with a barrier.
    Batched builds: wave ``w`` handles (request row, stream) ``w, w + 8, ...``.
    """
    lane, wave = c.lane, c.wave
    rows = c.rows
    hsum = c.lds_f + c.L.lds["hsum"]
    gpu.barrier()
    for k in range_constexpr(-(-rows * C.hc_count // WAVES)):
        if const_expr(rows == 1):
            b, row, hs = wave, 0, hsum
        else:
            q = wave + WAVES * k
            b, row = q % C.hc_count, q // C.hc_count
            hs = lds_row(c, c.lds_f, "hsum", row)
        if b + row * C.hc_count < rows * C.hc_count:
            v = fx.ptr_load(hs + (b * STREAM_CHUNKS + lane))
            hi = fx.ptr_load(hs + (b * STREAM_CHUNKS + 64 + fx.min(lane, 15)))
            v = (lane < 16).select(v + hi, v)
            ms = natural_sum(v) * _f32(INV_HIDDEN)
            stab = ms + _f32(EPS)
            inv = (fx.Float64(1.0) / fx.Float64(fmath.sqrt(stab.to(fx.Float64)))).to(
                fx.Float32
            )
            if lane == 0:
                fx.ptr_store(
                    inv, c.lds_f + row_ix(c.L.lds["inv"] + b, row, c.L.stride["inv"])
                )
    gpu.barrier()


norm_inv = ASTRewriter.transform(norm_inv)


def put_inv(c, hc, t, rr):
    """From ``INV_ROWS`` rows, A1's task 0 of request row ``rr``: the row's
    inverse RMS (LDS ``inv``) for A2, which then reads only its own words."""
    tid = c.tid
    if (t == 0) & (tid < C.hc_count):
        mbx.put(
            hc.inv, mb_ix(tid, rr), fx.ptr_load(c.lds_f + (c.L.lds["inv"] + tid)), c.tag
        )


put_inv = ASTRewriter.transform(put_inv)


def poll_inv(c, hc):
    """From ``INV_ROWS`` rows, A2: every row's inverse RMS (:func:`put_inv`) into
    LDS ``inv``, one value per thread from ``INV_TID``."""
    tid = c.tid
    if (tid >= INV_TID) & (tid < INV_TID + C.hc_count * c.rows):
        b, r = (tid - INV_TID) % C.hc_count, (tid - INV_TID) // C.hc_count
        (v,) = mbx.poll([(hc.inv, mb_ix(b, r), 1)], c.tag)
        fx.ptr_store(v[0], c.lds_i + row_ix(c.L.lds["inv"] + b, r, c.L.stride["inv"]))


poll_inv = ASTRewriter.transform(poll_inv)


def norm_values(c, h, w, r, row=0):
    """The 8 normalized values (bf16, as f32) of round ``r``'s vector of
    request row ``row``."""
    v = fx.min(c.tid + THREADS * r, VECS - 1)
    inv = fx.ptr_load(
        c.lds_f + row_ix(c.L.lds["inv"] + v // STREAM_VECS, row, c.L.stride["inv"])
    )
    xs = bf16x8([h[e] for e in range(4)])
    ws = bf16x8([w[e] for e in range(4)])
    return [bf16r((xs[e] * inv) * ws[e]) for e in range(8)]


def norm_pack(c, hs_rows, ws):
    """MXFP8 pack of the normalized streams into LDS ``n8`` / ``ne``, of every
    request row of ``hs_rows``. Every thread must call this; it ends with a
    barrier."""
    tid, lane = c.tid, c.lane
    n8s = [lds_row(c, c.lds_i, "n8", q) for q in range(len(hs_rows))]
    nes = [lds_row(c, c.lds_i, "ne", q) for q in range(len(hs_rows))]
    for q in range_constexpr(len(hs_rows)):
        hs, n8, ne = hs_rows[q], n8s[q], nes[q]
        for r in range_constexpr(ROUNDS):
            y = norm_values(c, hs[r], ws[r], r, q)
            amax = fmath.absf(y[0])
            for e in range_constexpr(1, 8):
                amax = fx.max(amax, fmath.absf(y[e]))
            amax = xred(xred(amax, 1, fx.max), 2, fx.max)
            ex, scale = p2.e8m0_fp8(amax)
            words = fx.Vector.from_elements(
                [p2.fp8_word(*y[0:4], scale), p2.fp8_word(*y[4:8], scale)], fx.Int32
            )
            v = tid + THREADS * r
            if v < VECS:
                fx.ptr_store(words, n8 + v * 2)
                if lane % 4 == 0:
                    fx.ptr_store(ex, ne + v // 4)
    gpu.barrier()


norm_pack = ASTRewriter.transform(norm_pack)


def inj_word(v):
    """LDS word of vector ``v`` (8 bf16) in the injection's padded lane layout."""
    return v // INJ_VECS * INJ_SEG + v % INJ_VECS * 4


def norm_bf16(c, hs, ws):
    """The normalized streams as bf16 words into LDS ``nbf`` (injection
    layout). Every thread must call this; it ends with a barrier."""
    tid = c.tid
    nbf = c.lds_i + c.L.lds["nbf"]
    for r in range_constexpr(ROUNDS):
        y = norm_values(c, hs[r], ws[r], r)
        words = fx.Vector.from_elements(
            [mbx.bf16_pair(y[2 * e], y[2 * e + 1]) for e in range(4)], fx.Int32
        )
        v = tid + THREADS * r
        if v < VECS:
            fx.ptr_store(words, nbf + inj_word(v))
    gpu.barrier()


norm_bf16 = ASTRewriter.transform(norm_bf16)


def _a1_down(c, hc, g, hs, ws, units, rr):
    lane, wave = c.lane, c.wave
    n8 = c.lds_i + c.L.lds["n8"]
    ne = c.lds_i + c.L.lds["ne"]
    red = c.lds_f + c.L.lds["red"]
    v4i = fx.Vector.make_type(4, fx.Int32)
    part = wave % 4

    norm_pack(c, [hs], ws)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(DOWN_CHAIN):
        kit = part + 4 * i
        off = kit * 32 + (lane // 16) * 4
        a0 = fx.Vector(fx.ptr_load(n8 + off, result_type=v4i))
        a1 = fx.Vector(fx.ptr_load(n8 + (off + 16), result_type=v4i))
        a = fx.Vector.from_elements(
            [a0[e] for e in range(4)] + [a1[e] for e in range(4)], fx.Int32
        )
        ea = fx.ptr_load(ne + (kit * 4 + lane // 16))
        acc = p2.mfma_p2(acc, a, units[0][i], ea, units[1][i // 4], 0, i % 4)
    if lane < 16:
        fx.ptr_store(acc[0], red + (wave * 16 + lane))
    gpu.barrier()

    # Lane L (and L + 32): column 32 g + L, n-block half L // 16.
    if wave == 0:
        col = (lane % 32) // 16 * 64 + lane % 16
        low = fx.ptr_load(red + col)
        for q in range_constexpr(1, 4):
            low = low + fx.ptr_load(red + (col + 16 * q))
        s = bf16r(bf16r(low) * 0.25)
        act = bf16r(s / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - s)))
        amax = fmath.absf(act)
        for off in (1, 2, 4, 8, 16):
            amax = xred(amax, off, fx.max)
        ex, scale = p2.e8m0_fp8(amax)
        v = lane_gather(act, (lane % 4) * 8, 8)
        lo = p2.fp8_word(v[0], v[1], v[2], v[3], scale)
        hi = p2.fp8_word(v[4], v[5], v[6], v[7], scale)
        if lane < 4:
            mbx.put2(hc.lr8, mb_ix(g * 8 + lane * 2, rr), lo, hi, c.tag)
        if lane == 0:
            mbx.put(hc.lre, mb_ix(g, rr), ex, c.tag)


_a1_down = ASTRewriter.transform(_a1_down)


def _a1_inject(c, hc, row, hs, ws, rr):
    tid, lane, wave = c.tid, c.lane, c.wave
    nbf = c.lds_i + c.L.lds["nbf"]
    injw = c.lds_i + c.L.lds["injw"]
    v4i = fx.Vector.make_type(4, fx.Int32)

    # Off the critical path (C1 consumes it): the row loads after the norm so
    # that down tasks issue no injection loads at all.
    wl = [
        fx.Vector(
            bo.buffer_load(
                hc.w_inj,
                (row * INJ_ROW_VECS + fx.min(tid + THREADS * r, INJ_ROW_VECS - 1)) * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for r in range(ROUNDS)
    ]
    for r in range_constexpr(ROUNDS):
        v = tid + THREADS * r
        if v < INJ_ROW_VECS:
            fx.ptr_store(wl[r], injw + inj_word(v))
    norm_bf16(c, hs, ws)
    # Lane L < 32 (lanes 32-63 repeat): the chain over k = 320 L + 8 v + e,
    # then the row sum over lanes in order.
    if wave == 0:
        base = (lane % INJ_LANES) * INJ_SEG
        part = fx.Float32(0.0)
        for ch in range_constexpr(INJ_VECS // INJ_CHUNK):
            rocdl.sched_barrier(0)
            for i in range_constexpr(INJ_CHUNK):
                off = base + 4 * (ch * INJ_CHUNK + i)
                wv = fx.Vector(fx.ptr_load(injw + off, result_type=v4i))
                yv = fx.Vector(fx.ptr_load(nbf + off, result_type=v4i))
                wf = bf16x8([wv[e] for e in range(4)])
                yf = bf16x8([yv[e] for e in range(4)])
                for e in range_constexpr(8):
                    part = fma(wf[e], yf[e], part)
        rocdl.sched_barrier(0)
        bits = part.bitcast(fx.Int32)
        s = read_lane_i32(bits, 0).bitcast(fx.Float32)
        for k in range_constexpr(1, INJ_LANES):
            s = s + read_lane_i32(bits, k).bitcast(fx.Float32)
        val = bf16r(bf16r(s) * 0.25)
        val = bf16r(sigmoid_acc(val))
        val = bf16r(val * 2.0)
        if lane == 0:
            mbx.put(hc.inj, mb_ix(row, rr), val, c.tag)


_a1_inject = ASTRewriter.transform(_a1_inject)


def _a1_units(hc, down, nb, part, lane):
    """This wave's down chain: 20 k_iters and their 20 scale bytes (one
    dwordx4 + one dword, ``p2.scales_chain``), out of range unless ``down``."""
    weights = [
        fx.Vector(
            bo.buffer_load(
                hc.w_down,
                down.select(((nb * DOWN_KI + part + 4 * i) * 64 + lane) * 4, OOB),
                vec_width=4,
                dtype=T.i32,
            )
        )
        for i in range(DOWN_CHAIN)
    ]
    sw = ((nb * 4 + part) * DOWN_SBLOCKS * 64 + lane) * 4
    s0 = fx.Vector(
        bo.buffer_load(hc.s_down, down.select(sw, OOB), vec_width=4, dtype=T.i32)
    )
    s1 = fx.Int32(
        bo.buffer_load(
            hc.s_down, down.select(sw + 64 * 4, OOB), vec_width=1, dtype=T.i32
        )
    )
    return weights, [s0[0], s0[1], s0[2], s0[3], s1]


def _a1_task(c, hc, t, rr, live):
    """One request's A1 task ``t`` (``< 14``) on request row ``rr``: ``c`` is a
    one-request view of the context, and the CTA has a task where ``live``."""
    tid, lane, wave = c.tid, c.lane, c.wave
    g = fx.min(t, DOWN_GROUPS - 1)
    nb = g * 2 + wave // 4
    part = wave % 4
    row = t - INJ_TASK  # injection weight ``row``

    def hint(m):
        return hc.hs, mb_ix(m * hc.hint_stride, rr)

    if live:
        # Down tasks load their chain unconditionally; injection tasks read it
        # out of range (zeros, no memory traffic), so no load sits in a branch
        # that LLVM would hoist the norm above. The stage is issue bound:
        # every load instruction counts, out of range or not.
        down = t < INJ_TASK
        if const_expr(hc.polled):
            # The streams come late (C1): prefetch everything, then poll.
            ws = norm_load(hc.norm_w, tid)
            units = _a1_units(hc, down, nb, part, lane)
            payload_wait(c, hint, hc.producers)
            (hs,) = poll_streams(c, hc, [rr])
        else:
            # h and w' first: the barrier keeps every wave's norm loads ahead
            # of any wave's weight stream in the CU's load pipe (oldest-first
            # issue would otherwise queue wave 7's h behind ~140 KiB of down
            # weights).
            if const_expr(c.pre is not None and hc.down == "a1"):
                hs, ws = c.pre  # a1_preload's, issued at the kernel's entry
            else:
                hs = norm_load(hc.h, tid, rr)
                ws = norm_load(hc.norm_w, tid)
            rocdl.sched_barrier(0)
            gpu.barrier()
            units = _a1_units(hc, down, nb, part, lane)
        norm_sums(c, hc, hs)
        norm_inv(c, hc)
        if const_expr(hc.wide_inv):
            put_inv(c, hc, t, rr)
        if down:
            _a1_down(c, hc, g, hs, ws, units, rr)
        else:
            _a1_inject(c, hc, row, hs, ws, rr)


_a1_task = ASTRewriter.transform(_a1_task)


def a1_preload(c, p):
    """A1's ``h`` and ``w'`` loads of this CTA's task (``a1``'s row), issued
    at the kernel's entry ahead of the step counter's use (the launch tag);
    out of range (no memory traffic) on CTAs without an A1 task."""
    t = task_index(c, "a1")
    live = t < c.L.tasks["a1"]
    if const_expr(c.rows == 1):
        rr = 0
    else:
        n = c.rows * DOWN_GROUPS
        rr = (t < n).select(t // DOWN_GROUPS, fx.max(t - n, 0) // C.hc_count)
    return norm_load(p.h_in, c.tid, rr, keep=live), norm_load(
        p.hc_norm_w, c.tid, keep=live
    )


def a1(c, hc):
    t = task_index(c, hc.down)
    if const_expr(c.rows == 1):
        _a1_task(c, hc, t, 0, t < TASKS[hc.down])
    else:
        # Batched builds: every request row's own tasks (down groups of all
        # rows, then injection weights), each one request's code on its row.
        one = SimpleNamespace(**vars(c))
        one.rows = 1
        n = c.rows * DOWN_GROUPS
        inj = fx.max(t - n, 0)
        down = t < n
        rr = down.select(t // DOWN_GROUPS, inj // C.hc_count)
        lt = down.select(t % DOWN_GROUPS, INJ_TASK + inj % C.hc_count)
        _a1_task(one, hc, lt, rr, t < c.L.tasks[hc.down])


a1 = ASTRewriter.transform(a1)


def a2(c, hc):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, hc.up)
    l8 = c.lds_i + c.L.lds["l8"]
    le = c.lds_i + c.L.lds["le"]
    prod = c.lds_f + c.L.lds["prod"]
    sl8, sle, sp = c.L.stride["l8"], c.L.stride["le"], c.L.stride["prod"]
    v4i = fx.Vector.make_type(4, fx.Int32)
    stream = wave // 2
    atom = wave % 2
    nb = t * 2 + stream * UP_NB + atom
    j = lane % 16
    idx = stream * C.hidden + t * 32 + atom * 16 + j

    # Batched builds: hint ``m`` is producer ``m % n`` of request row ``m // n``.
    def hint(m):
        k, r = (m, 0) if rows == 1 else (m % DOWN_GROUPS, m // DOWN_GROUPS)
        return hc.lre, mb_ix(k, r)

    def c1_hint(m):
        k, r = (m, 0) if rows == 1 else (m % hc.producers, m // hc.producers)
        return hc.hs, mb_ix(k * hc.hint_stride, r)

    if t < c.L.tasks[hc.up]:
        wu = [
            fx.Vector(
                bo.buffer_load(
                    hc.w_up, ((nb * 4 + kit) * 64 + lane) * 4, vec_width=4, dtype=T.i32
                )
            )
            for kit in range(UP_KI)
        ]
        su = fx.Int32(bo.buffer_load(hc.s_up, nb * 64 + lane, vec_width=1, dtype=T.i32))
        ww = fx.Int32(bo.buffer_load(hc.norm_w, idx // 2, vec_width=1, dtype=T.i32))
        if const_expr(hc.polled):
            if const_expr(not hc.wide_inv):
                payload_wait(c, c1_hint, hc.producers * rows)
                hws = [
                    v[0]
                    for v in mbx.poll(
                        [(hc.h, mb_ix(idx // 2, r), 1) for r in range(rows)], c.tag
                    )
                ]
                halves_sums(c, hc)
        else:
            hws = [
                fx.Int32(
                    bo.buffer_load(
                        hc.h, row_ix(idx // 2, r, HC_WORDS), vec_width=1, dtype=T.i32
                    )
                )
                for r in range(rows)
            ]
            if const_expr(not hc.wide_inv):
                for r in range_constexpr(rows):
                    norm_sums(c, hc, norm_load(hc.h, tid, r), r)
        if const_expr(not hc.wide_inv):
            norm_inv(c, hc)

        # The low-rank pack: 80 fp8 words then zeros to 96, 10 exponents then
        # 127; from INV_ROWS rows the streams' inverse RMS (A1's task 0 of each row).
        payload_wait(c, hint, DOWN_GROUPS * rows)
        if const_expr(hc.polled and hc.wide_inv):
            # A1's tasks of a row polled all its streams before publishing.
            hws = [
                v[0]
                for v in mbx.poll(
                    [(hc.h, mb_ix(idx // 2, r), 1) for r in range(rows)], c.tag
                )
            ]
        if tid < C.hc_lowrank // 8:
            vs = mbx.poll([(hc.lr8, mb_ix(tid * 2, r), 2) for r in range(rows)], c.tag)
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Vector.from_elements(vs[r], fx.Int32),
                    l8 + row_ix(tid * 2, r, sl8),
                )
        if (tid >= 64) & (tid < 64 + DOWN_GROUPS):
            es = mbx.poll([(hc.lre, mb_ix(tid - 64, r), 1) for r in range(rows)], c.tag)
            for r in range_constexpr(rows):
                fx.ptr_store(es[r][0], le + row_ix(tid - 64, r, sle))
        if (tid >= 128) & (tid < 128 + 16):
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Int32(0), l8 + row_ix(tid - 128 + C.hc_lowrank // 4, r, sl8)
                )
        if (tid >= 192) & (tid < 192 + 2):
            for r in range_constexpr(rows):
                fx.ptr_store(
                    fx.Int32(p2.SCALE_ONE), le + row_ix(tid - 192 + DOWN_GROUPS, r, sle)
                )
        if const_expr(hc.wide_inv):
            poll_inv(c, hc)
        gpu.barrier()

        xl8, xle = mfma_operands(c, l8, le, "l8", "le")
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for kit in range_constexpr(UP_KI):
            off = kit * 32 + (lane // 16) * 4
            a0 = fx.Vector(fx.ptr_load(xl8 + off, result_type=v4i))
            a1_ = fx.Vector(fx.ptr_load(xl8 + (off + 16), result_type=v4i))
            a = fx.Vector.from_elements(
                [a0[e] for e in range(4)] + [a1_[e] for e in range(4)], fx.Int32
            )
            ea = fx.ptr_load(xle + (kit * 4 + lane // 16))
            acc = p2.mfma_p2(acc, a, wu[kit], ea, su, 0, kit)
        for r in range_constexpr(rows):
            hw = hws[r]
            rw = bf16r(sigmoid_acc(bf16r(acc[r])))
            x = (j % 2 == 0).select(hw << 16, hw & fx.Int32(-65536)).bitcast(fx.Float32)
            w = (j % 2 == 0).select(ww << 16, ww & fx.Int32(-65536)).bitcast(fx.Float32)
            inv = fx.ptr_load(
                c.lds_f + row_ix(c.L.lds["inv"] + stream, r, c.L.stride["inv"])
            )
            y = bf16r((x * inv) * w)
            if lane < 16:
                fx.ptr_store(
                    bf16r(rw * y), prod + row_ix(stream * 32 + atom * 16 + j, r, sp)
                )
        gpu.barrier()

        # Lane L (and L + 32): mixed[32 t + L]. Batched builds: wave r, of
        # request row r.
        rq = 0 if rows == 1 else fx.min(wave, rows - 1)
        if (wave == 0) if rows == 1 else (wave < rows):
            col = lane % 32
            s = fx.Float32(0.0)
            for b in range_constexpr(C.hc_count):
                s = s + fx.ptr_load(prod + row_ix(b * 32 + col, rq, sp))
            mixed = bf16r(s * 0.25)
            amax = fmath.absf(mixed)
            for off in (1, 2, 4, 8, 16):
                amax = xred(amax, off, fx.max)
            ex, scale = p2.e8m0_fp8(amax)
            v8 = lane_gather(mixed, (lane % 4) * 8, 8)
            lo = p2.fp8_word(v8[0], v8[1], v8[2], v8[3], scale)
            hi = p2.fp8_word(v8[4], v8[5], v8[6], v8[7], scale)
            v4 = lane_gather(mixed, (lane % 8) * 4, 4)
            if lane < 4:
                mbx.put2(hc.x8, mb_ix(t * 8 + lane * 2, rq), lo, hi, c.tag)
            if lane < 8:
                mbx.put2(
                    hc.xbf,
                    mb_ix(t * 16 + lane * 2, rq),
                    mbx.bf16_pair(v4[0], v4[1]),
                    mbx.bf16_pair(v4[2], v4[3]),
                    c.tag,
                )
            if lane == 0:
                mbx.put(hc.xe, mb_ix(t, rq), ex, c.tag)


a2 = ASTRewriter.transform(a2)
