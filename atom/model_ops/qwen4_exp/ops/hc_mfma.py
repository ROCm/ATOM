# SPDX-License-Identifier: MIT

"""Two-launch MFMA hyper-connection mix for Qwen3.8 Flash-Next decode.

The served HC step is a deferred combine, a per-stream Gemma norm, the
``[down | inject]`` projection, ``scaled_silu``, the up projection and the
gated mean over the four streams. The native path takes five launches; this
one takes two:

``down``: one CTA per (output tile, K chunk of 640 within one stream). It
applies the pending combine to its chunk (the tile-0 CTAs store the new
streams), and feeds ``bf16(x * (1 + w))`` to 16x16x32 bf16 MFMAs. The stream's
rstd is factored out of the dot product, so a chunk needs no row norm: each
CTA also stores its chunk's sum of squares. The last CTA of a tile to finish
(a device-scope counter, reset by that CTA) sums the 16 chunk partials in
fixed order, scales each stream's sum by its rstd, and stores the bf16 gate
(``scaled_silu``) and the injection logits.

``up``: one CTA per 16 hidden channels, one wave per stream. Each wave
projects its 16 up rows for all tokens on the MFMA, rounds to bf16 and
multiplies the sigmoid by the normed stream; the CTA averages the four
streams in stream order.

Every sum has a fixed order, so all TP ranks (which mix redundantly) produce
the same bits. Rounding matches the native path except that the down
projection's operand is ``bf16(x * (1 + w))`` instead of
``bf16(x * rstd * (1 + w))``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, Stream, T

from atom.mono.device.ops import (
    CM_DEV,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    launcher,
    mfma_bf16,
    rsrc,
    xred,
)

HIDDEN = 2560
HC = 4
WIDTH = HIDDEN * HC
RANK = 320
THREADS = 256
WAVES = THREADS // 64
KSTEP = 32  # the MFMA's K
CHUNK = 640  # K a down CTA covers, within one stream
CHUNKS = WIDTH // CHUNK
STREAM_CHUNKS = HIDDEN // CHUNK
DOWN_STEPS = CHUNK // KSTEP // WAVES  # K steps a wave of a down CTA takes
UP_STEPS = RANK // KSTEP
MAX_ROWS = 48
MAX_TILES = (MAX_ROWS + 15) // 16
O_TILE = 32  # down outputs a CTA covers
OT = O_TILE // 16
_CURRENT_STREAM = Stream(None)


def _sigmoid(x):
    return hw_rcp(fx.Float32(1.0) + hw_exp2(-fx.Float32(x) * LOG2E))


def _bf(x):
    return fx.Float32(fx.Float32(x).to(fx.BFloat16))


def _unpack(word):
    word = fx.Int32(word)
    return (word << 16).bitcast(fx.Float32), (word & fx.Int32(-65536)).bitcast(
        fx.Float32
    )


def _pack(lo, hi):
    return (
        fx.Vector.from_elements([fx.Float32(lo), fx.Float32(hi)], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _words(vec, n):
    return [fx.Int32(vec[i]) for i in range(n)]


def _operand(words):
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def down_tiles(outputs: int, o_tile: int = O_TILE) -> int:
    return -(-outputs // o_tile)


@functools.cache
def build_down(
    rows: int,
    pending: bool,
    outputs: int,
    eps: float,
    chunk: int = 640,
    threads: int = 256,
    o_tile: int = 32,
):
    """Kernel ``down`` for ``rows`` tokens; ``outputs`` is 324 (down + inject)
    or 320. Scratch: ``part`` f32 [chunks, tiles, OT, tiles_t, 64, 4], ``sq``
    f32 [tiles, chunks, 16 tiles_t], ``count`` i32 [tiles] (zero between
    launches)."""
    assert 1 <= rows <= MAX_ROWS and outputs in (RANK, RANK + HC)
    THREADS = threads
    WAVES = threads // 64
    CHUNK = chunk
    CHUNKS = WIDTH // chunk
    STREAM_CHUNKS = HIDDEN // chunk
    DOWN_STEPS = chunk // KSTEP // WAVES
    assert STREAM_CHUNKS * chunk == HIDDEN and DOWN_STEPS * KSTEP * WAVES == chunk
    O_TILE = o_tile
    OT = o_tile // 16
    tt_n = (rows + 15) // 16
    rp = tt_n * 16
    tiles = down_tiles(outputs, o_tile)
    part_tile = OT * tt_n * 64 * 4  # floats of one (chunk, tile) partial
    groups = OT * tt_n * 64  # vec4 groups of a partial

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * OT * MAX_TILES * 64 * 4, 16]
        sqr: fx.Array[fx.Float32, WAVES * MAX_TILES * 16, 16]
        flag: fx.Array[fx.Int32, 4, 16]

    name = kernel_symbol(
        "qwen38_hc_down",
        r=rows,
        p=int(pending),
        o=outputs,
        e=eps,
        c=chunk,
        t=threads,
        n=o_tile,
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def down(
        h: Int64,
        block: Int64,
        raw_prev: Int64,
        norm_w: Int64,
        w_cat: Int64,
        h_out: Int64,
        part: Int64,
        sq: Int64,
        count: Int64,
        gate: Int64,
        raw_out: Int64,
    ):
        bid = gpu.block_idx.x
        tile = bid // CHUNKS
        kc = bid % CHUNKS
        stream = kc // STREAM_CHUNKS
        tid = gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        col, kq = lane % 16, lane // 16
        smem = fx.SharedAllocator().allocate(Smem).peek()
        red, sqr, flag = smem.red.ptr, smem.sqr.ptr, smem.flag.ptr
        h_rsrc, b_rsrc, w_rsrc, nw_rsrc = (
            rsrc(h),
            rsrc(block),
            rsrc(w_cat),
            rsrc(norm_w),
        )
        tokens = [fx.min(tt * 16 + col, rows - 1) for tt in range(tt_n)]
        factors = []
        for tt in range_constexpr(tt_n):
            if pending:
                raw = fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            rsrc(raw_prev),
                            tokens[tt] * HC + stream,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                )
                factors.append(fx.Float32(2.0) * _sigmoid(_bf(raw * (1.0 / HC))))
        acc = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tt_n)]
            for _ in range(OT)
        ]
        sums = [fx.Float32(0.0) for _ in range(tt_n)]
        # every step's loads go out before any use: a use stores (h_out),
        # which would hold the next step's loads behind it
        loaded = []
        for i in range_constexpr(DOWN_STEPS):
            k0 = kc * CHUNK + (wave + WAVES * i) * KSTEP + 8 * kq
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (
                            fx.min(tile * O_TILE + ot * 16 + col, outputs - 1) * WIDTH
                            + k0
                        )
                        // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for ot in range(OT)
            ]
            nw = [
                fx.Vector(bo.buffer_load(nw_rsrc, k0 + 4 * j, vec_width=4, dtype=T.f32))
                for j in range(2)
            ]
            xv = [
                fx.Vector(
                    bo.buffer_load(
                        h_rsrc,
                        (tokens[tt] * WIDTH + k0) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for tt in range(tt_n)
            ]
            bv = []
            if pending:
                bv = [
                    fx.Vector(
                        bo.buffer_load(
                            b_rsrc,
                            (tokens[tt] * HIDDEN + k0 - stream * HIDDEN) // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for tt in range(tt_n)
                ]
            loaded.append((k0, wv, nw, xv, bv))
        for i in range_constexpr(DOWN_STEPS):
            k0, wv, nw, xv, bv = loaded[i]
            scale = [fx.Float32(1.0) + fx.Float32(nw[j // 4][j % 4]) for j in range(8)]
            ops = []
            for tt in range_constexpr(tt_n):
                xw = _words(xv[tt], 4)
                if pending:
                    bw = _words(bv[tt], 4)
                    combined = []
                    for j in range_constexpr(4):
                        lo, hi = _unpack(xw[j])
                        blo, bhi = _unpack(bw[j])
                        combined.append(
                            _pack(lo + blo * factors[tt], hi + bhi * factors[tt])
                        )
                    xw = combined
                    if (tile == 0) & (tt * 16 + col < rows):
                        bo.buffer_store(
                            fx.Vector.from_elements(xw, fx.Int32),
                            rsrc(h_out),
                            (tokens[tt] * WIDTH + k0) // 2,
                        )
                scaled = []
                for j in range_constexpr(4):
                    lo, hi = _unpack(xw[j])
                    sums[tt] = sums[tt] + lo * lo + hi * hi
                    scaled.append(_pack(lo * scale[2 * j], hi * scale[2 * j + 1]))
                ops.append(_operand(scaled))
            for ot in range_constexpr(OT):
                wop = wv[ot].bitcast(fx.BFloat16)
                for tt in range_constexpr(tt_n):
                    acc[ot][tt] = mfma_bf16(wop, ops[tt], acc[ot][tt])
        for ot in range_constexpr(OT):
            for tt in range_constexpr(tt_n):
                fx.ptr_store(
                    acc[ot][tt], red + (((wave * OT + ot) * tt_n + tt) * 64 + lane) * 4
                )
        for tt in range_constexpr(tt_n):
            total = xred(xred(sums[tt], 16, lambda a, b: a + b), 32, lambda a, b: a + b)
            if kq == 0:
                fx.ptr_store(total, sqr + (wave * tt_n + tt) * 16 + col)
        gpu.barrier()
        part_rsrc, sq_rsrc = rsrc(part), rsrc(sq)
        for g0 in range_constexpr(0, groups, THREADS):
            g = g0 + tid
            if g < groups:
                total = [fx.Float32(0.0) for _ in range(4)]
                for w in range_constexpr(WAVES):
                    for e in range_constexpr(4):
                        total[e] = total[e] + fx.ptr_load(
                            red + (w * groups + g) * 4 + e
                        )
                bo.buffer_store(
                    fx.Vector.from_elements(total, fx.Float32),
                    part_rsrc,
                    (kc * tiles + tile) * part_tile + g * 4,
                    cache_modifier=CM_DEV,
                )
        if tid < rp:
            total = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                total = total + fx.ptr_load(sqr + w * rp + tid)
            bo.buffer_store(
                total, sq_rsrc, (tile * CHUNKS + kc) * rp + tid, cache_modifier=CM_DEV
            )
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            ticket = fx.atomic_add(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                    fx.Int64(count) + fx.Int64(tile * 4),
                ),
                fx.Int32(1),
                syncscope=rocdl.SyncScope.Agent,
            )
            fx.ptr_store(fx.Int32(ticket), flag)
        gpu.barrier()
        if fx.ptr_load(flag) == CHUNKS - 1:
            # the tile's last CTA: each group loads its chunk partials and its
            # token's sums of squares together, one round trip
            gate_rsrc, raw_rsrc = rsrc(gate), rsrc(raw_out)
            for g0 in range_constexpr(0, groups, THREADS):
                g = g0 + tid
                if g < groups:
                    ot, tt, glane = g // (tt_n * 64), (g // 64) % tt_n, g % 64
                    token = tt * 16 + glane % 16
                    out0 = tile * O_TILE + ot * 16 + 4 * (glane // 16)
                    parts = [
                        fx.Vector(
                            bo.buffer_load(
                                part_rsrc,
                                (c * tiles + tile) * part_tile + g * 4,
                                vec_width=4,
                                dtype=T.f32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        for c in range(CHUNKS)
                    ]
                    sq_parts = [
                        fx.Float32(
                            bo.buffer_load(
                                sq_rsrc,
                                (tile * CHUNKS + c) * rp + token,
                                vec_width=1,
                                dtype=T.f32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        for c in range(CHUNKS)
                    ]
                    d = [fx.Float32(0.0) for _ in range(4)]
                    for s in range_constexpr(HC):
                        ps = [fx.Float32(0.0) for _ in range(4)]
                        total = fx.Float32(0.0)
                        for q in range_constexpr(STREAM_CHUNKS):
                            v = parts[s * STREAM_CHUNKS + q]
                            for e in range_constexpr(4):
                                ps[e] = ps[e] + fx.Float32(v[e])
                            total = total + sq_parts[s * STREAM_CHUNKS + q]
                        r = hw_rsq(total * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps))
                        for e in range_constexpr(4):
                            d[e] = d[e] + r * ps[e]
                    d = [_bf(v) for v in d]
                    if token < rows:
                        if out0 < RANK:
                            act = []
                            for e in range_constexpr(4):
                                x = _bf(d[e] * (1.0 / HC))
                                act.append(x * _sigmoid(x))
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [_pack(act[0], act[1]), _pack(act[2], act[3])],
                                    fx.Int32,
                                ),
                                gate_rsrc,
                                (token * RANK + out0) // 2,
                            )
                        elif out0 < outputs:
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [_pack(d[0], d[1]), _pack(d[2], d[3])], fx.Int32
                                ),
                                raw_rsrc,
                                token * (HC // 2),
                            )
            if tid == 0:
                bo.buffer_store(fx.Int32(0), rsrc(count), tile, cache_modifier=CM_DEV)

    @launcher(name)
    def launch(
        h: Int64,
        block: Int64,
        raw_prev: Int64,
        norm_w: Int64,
        w_cat: Int64,
        h_out: Int64,
        part: Int64,
        sq: Int64,
        count: Int64,
        gate: Int64,
        raw_out: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        down(
            h, block, raw_prev, norm_w, w_cat, h_out, part, sq, count, gate, raw_out
        ).launch(grid=(tiles * CHUNKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


@functools.cache
def build_up(rows: int, eps: float, chunk: int = 640, threads: int = 256):
    """Kernel ``up``: the gated mean, from the streams ``h``, ``down``'s
    sums of squares (tile 0's) and gate. With 512 threads two waves split
    each stream's K, summed in order (low half first)."""
    assert 1 <= rows <= MAX_ROWS and threads in (256, 512)
    STREAM_CHUNKS = HIDDEN // chunk
    halves = threads // 256
    steps = UP_STEPS // halves
    tt_n = (rows + 15) // 16
    rp = tt_n * 16

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 2 * HC * MAX_TILES * 64 * 4, 16]

    name = kernel_symbol("qwen38_hc_up", r=rows, e=eps, c=chunk, t=threads)

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def up(h: Int64, norm_w: Int64, sq: Int64, gate: Int64, w_up: Int64, mixed: Int64):
        bid = gpu.block_idx.x
        h0 = bid * 16
        tid = gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        s, half = wave % HC, wave // HC
        col, kq = lane % 16, lane // 16
        red = fx.SharedAllocator().allocate(Smem).peek().red.ptr
        g_rsrc, w_rsrc = rsrc(gate), rsrc(w_up)
        tokens = [fx.min(tt * 16 + col, rows - 1) for tt in range(tt_n)]
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tt_n)]
        # the epilogue's operands first: none depends on the projection
        nw = fx.Vector(
            bo.buffer_load(
                rsrc(norm_w), s * HIDDEN + h0 + 4 * kq, vec_width=4, dtype=T.f32
            )
        )
        sq_rsrc, h_rsrc = rsrc(sq), rsrc(h)
        sq_parts = [
            [
                fx.Float32(
                    bo.buffer_load(
                        sq_rsrc,
                        (s * STREAM_CHUNKS + q) * rp + tt * 16 + col,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                for q in range(STREAM_CHUNKS)
            ]
            for tt in range(tt_n)
        ]
        x_words = [
            fx.Vector(
                bo.buffer_load(
                    h_rsrc,
                    (tokens[tt] * WIDTH + s * HIDDEN + h0 + 4 * kq) // 2,
                    vec_width=2,
                    dtype=T.i32,
                )
            )
            for tt in range(tt_n)
        ]
        operands = []
        for i in range_constexpr(steps):
            k0 = (half * steps + i) * KSTEP + 8 * kq
            wv = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((s * HIDDEN + h0 + col) * RANK + k0) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
            gv = [
                fx.Vector(
                    bo.buffer_load(
                        g_rsrc, (tokens[tt] * RANK + k0) // 2, vec_width=4, dtype=T.i32
                    )
                ).bitcast(fx.BFloat16)
                for tt in range(tt_n)
            ]
            operands.append((wv, gv))
        for i in range_constexpr(steps):
            wv, gv = operands[i]
            for tt in range_constexpr(tt_n):
                acc[tt] = mfma_bf16(wv, gv[tt], acc[tt])
        if halves == 2:
            for tt in range_constexpr(tt_n):
                fx.ptr_store(acc[tt], red + ((wave * tt_n + tt) * 64 + lane) * 4)
            gpu.barrier()
        if half == 0:
            for tt in range_constexpr(tt_n):
                up_sum = acc[tt]
                if halves == 2:
                    lo_sum, hi_sum = [], []
                    for e in range_constexpr(4):
                        lo_sum.append(
                            fx.ptr_load(red + ((s * tt_n + tt) * 64 + lane) * 4 + e)
                        )
                        hi_sum.append(
                            fx.ptr_load(
                                red + (((s + HC) * tt_n + tt) * 64 + lane) * 4 + e
                            )
                        )
                    up_sum = [lo_sum[e] + hi_sum[e] for e in range(4)]
                total = fx.Float32(0.0)
                for q in range_constexpr(STREAM_CHUNKS):
                    total = total + sq_parts[tt][q]
                rstd = hw_rsq(total * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps))
                xw = x_words[tt]
                x = []
                for j in range_constexpr(2):
                    lo, hi = _unpack(xw[j])
                    x += [lo, hi]
                vals = []
                for e in range_constexpr(4):
                    normed = _bf(x[e] * rstd * (fx.Float32(1.0) + fx.Float32(nw[e])))
                    vals.append(_sigmoid(_bf(up_sum[e])) * normed)
                fx.ptr_store(
                    fx.Vector.from_elements(vals, fx.Float32),
                    # slot HC + s: wave s + HC's partial, which only wave s read
                    red + (((HC + s) * tt_n + tt) * 64 + lane) * 4,
                )
        gpu.barrier()
        if tid < tt_n * 64:
            tt, glane = tid // 64, tid % 64
            token = tt * 16 + glane % 16
            total = [fx.Float32(0.0) for _ in range(4)]
            for w in range_constexpr(HC):
                for e in range_constexpr(4):
                    total[e] = total[e] + fx.ptr_load(
                        red + (((HC + w) * tt_n + tt) * 64 + glane) * 4 + e
                    )
            if token < rows:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [
                            _pack(total[0] * (1.0 / HC), total[1] * (1.0 / HC)),
                            _pack(total[2] * (1.0 / HC), total[3] * (1.0 / HC)),
                        ],
                        fx.Int32,
                    ),
                    rsrc(mixed),
                    (token * HIDDEN + h0 + 4 * (glane // 16)) // 2,
                )

    @launcher(name)
    def launch(
        h: Int64,
        norm_w: Int64,
        sq: Int64,
        gate: Int64,
        w_up: Int64,
        mixed: Int64,
        stream: Stream = _CURRENT_STREAM,
    ):
        up(h, norm_w, sq, gate, w_up, mixed).launch(
            grid=(HIDDEN // 16, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


DOWN_CHUNK = 512
DOWN_THREADS = 512
DOWN_O_TILE = 32
UP_THREADS = 256


class _Scratch:
    def __init__(self, device: torch.device):
        most = (WIDTH // 256) * down_tiles(RANK + HC, 16)
        self.part = torch.empty(
            most * MAX_TILES * 64 * 4, dtype=torch.float32, device=device
        )
        self.sq = torch.empty(most * MAX_TILES * 16, dtype=torch.float32, device=device)
        self.count = torch.zeros(
            down_tiles(RANK + HC, 16), dtype=torch.int32, device=device
        )


@functools.cache
def _scratch(device: torch.device) -> _Scratch:
    return _Scratch(device)


def mix_fused(hc, hyper_input: torch.Tensor, pending=None):
    """``Qwen4ExpHyperConnection.mix_fused`` in two launches (shapes checked
    by the caller)."""
    rows, device = hyper_input.shape[0], hyper_input.device
    has_inject = hc.block_inject_weight is not None
    outputs = RANK + HC if has_inject else RANK
    eps = float(hc.hc_norm.variance_epsilon)
    scratch = _scratch(device)
    has_pending = pending is not None
    if has_pending:
        block, raw_prev = pending
        h_new = torch.empty_like(hyper_input)
    else:
        block = raw_prev = h_new = hyper_input
    gate = torch.empty((rows, RANK), dtype=torch.bfloat16, device=device)
    mixed = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    raw_out = (
        torch.empty((rows, HC), dtype=torch.bfloat16, device=device)
        if has_inject
        else gate
    )
    stream = torch.cuda.current_stream(device)
    build_down(rows, has_pending, outputs, eps, DOWN_CHUNK, DOWN_THREADS, DOWN_O_TILE)(
        hyper_input.data_ptr(),
        block.data_ptr(),
        raw_prev.data_ptr(),
        hc.hc_norm.weight.data_ptr(),
        hc.fused_w_cat.data_ptr(),
        h_new.data_ptr(),
        scratch.part.data_ptr(),
        scratch.sq.data_ptr(),
        scratch.count.data_ptr(),
        gate.data_ptr(),
        raw_out.data_ptr(),
        stream=stream,
    )
    build_up(rows, eps, DOWN_CHUNK, UP_THREADS)(
        h_new.data_ptr(),
        hc.hc_norm.weight.data_ptr(),
        scratch.sq.data_ptr(),
        gate.data_ptr(),
        hc.input_mix_weight_up.weight.data_ptr(),
        mixed.data_ptr(),
        stream=stream,
    )
    return mixed, (h_new, raw_out if has_inject else None)
