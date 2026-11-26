# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Engram``: DeepSeek-V4.1's n-gram memory (checkpoint ``inference/engram.py`` + ``model.py`` ``Engram``), applied
to the residual streams before layers 1 and 14, per row of the step.

What one decode row at position ``P`` does in the reference (``x`` = the 4 HC streams [4, 5120] bf16, the Block's
input before ``hc_mixes``):

1. **Hash** (``NgramHashState``): the row's token and the 3 before it, each through the compressed-token map
   (``token_map``, 129280 -> 99092 ids); a slot before position 0 (or after a dead / image token) is the compressed
   pad id. ``rolling_i = z_0 m_0 ^ z_1 m_1 ^ .. ^ z_i m_i`` (int64; ``m`` the layer's 4 odd multipliers, ~2^46),
   i = 1, 2, 3 (the 2-, 3-, 4-gram); hash column ``8 (i - 1) + h`` (h < 8 heads) is
   ``rolling_i % prime[i - 1][h] + offset[col]``: 24 row ids into the layer's table, each column its own disjoint
   prime-sized (~16M) row range.
2. **Lookup** (``ParallelEngramEmbedding``): 24 rows of the table (384M rows x 256 E4M3 + 8 E8M0 scales a row, one
   scale per 32: 101 GB a layer), dequantized to bf16 (exact) -> 6144 values.
3. **Projection** ``kv = wkv(rows)``: an FP8 ``Linear`` (32 x 32 block scales) [25600, 6144]: the rows through
   ``act_quant`` (MXFP8, E8M0 per 32; the rows are FP8 values times powers of two, so exact), the FP8 GEMM, bf16 out.
   157 MB of weights a layer, replicated in the official TP model.
4. **Gate**, per HC copy c (normalized over 5120, not jointly): ``key_c = kv[c 5120 ..]``, ``value = kv[20480 ..]``;
   ``dot = sum(h_c * (q_w k_w)_c * key_c) * rsqrt(mean h_c^2 + eps) * rsqrt(mean key_c^2 + eps) * 5120^-1/2``;
   ``gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))``; ``out_c = bf16(h_c + gate value)``. All 4 streams are
   updated with the one shared value, each with its own gate.

Steps 1-3 depend only on the token ids, not on the streams: they can run anywhere before the gate (beside the
previous layers), which is where this block's cost hides in a composed step.

**TP** (``Scenario.tp`` = 4; any tp dividing the 24 columns and the width in the split modes, e.g. 8: 3 columns and 640
dims of each vector a rank, the gate's 4 slices unchanged on the gathered kv, so TP8 has TP4's bits): the table is
row-sharded by hash column: rank r holds columns 6 r .. 6 r + 5, i.e. the contiguous row range ``[offset[6 r], offset[6
r + 6])`` (25 GB of each layer's table; 51 GB a rank for both). wkv is column-sharded by dim slice: rank r computes
key_c / value dims ``1280 r .. 1280 (r + 1)`` (6400 columns, 39 MB), so every rank needs all 24 rows (gathered once) and
the gate needs a cross-rank sum of 2 partials a copy plus the value slices (gathered once). TP1 runs the same 4 dim
slices, so TP1 and TP4 give the same bits.

Stages:

- ``<name>.look`` (rows x 3 tasks at TP1, rows at TP4; a wave per hash column): hash, the row's 64 code words and
  its scales (64-bit addresses: the table is past a buffer's 4 GB), ``act_quant``'s MXFP8 encoding of the row
  (scale ``ceil_pow2(max(amax, 1e-4) / 448)``; the codes shifted, exact), put into the MXFP8 channel ``<name>.e``
  (TP1) or every rank's peer regions ``<name>.pq`` / ``<name>.ps`` (TP > 1); stores the row's compressed id into
  ``hist`` (later launches' history);
- ``<name>.gath`` (TP > 1, a task a row): the 4 ranks' columns -> the local channel ``<name>.e``;
- ``<name>.wkv``: ``Linear`` MXFP8 -> BF16 channel ``<name>.kv`` (6400 x tp columns: slice-major per vector);
- ``<name>.part`` (a task per (row, dim slice): rows x 4 at TP1, rows at TP4): the slice's partial dot and key
  square sums of the 4 copies, the value slice -> region ``<name>.px`` (scratch at TP1, every rank's peer copy at
  TP > 1);
- ``<name>.gate`` (a task per (row, copy)): the 4 slices' partials summed in slice order, sum h_c^2 (fixed order),
  the gate, ``out`` (a tagged BF16 channel [rows, 4 x 5120]) and / or ``out_arg`` (plain bf16 [rows, 4, 5120]).

**Split** (``mode``; the wiring this block is meant for): ``"kv"`` = look (+ gath) + wkv, then (TP > 1) the
ranks' kv shares all-gathered (``<name>.kvx``, a destination a task) and stored, full width in the natural column
order, to the plain ``kv_arg`` [rows, 25600] bf16 (``<name>.store``): it depends only on the token ids, so it runs in
an earlier launch (beside the previous layer). ``"gate"`` = one stage a (row, copy) reading ``kv_arg``: ``part``'s
slice partials in the same mapping and order, then ``gate``'s arithmetic (the same bits as ``"full"``), no peer hop
and no in-launch hand-off: the only Engram work on the step's critical path.

Args: ``ids`` int32 [rows] (the step's token ids), ``pos`` int32 [batch] (the step's first position), ``hist`` int32
[batch, hist_len] (compressed ids by position: the prefill's, then each launch's own rows; -1 = dead), ``tmap`` int32
[vocab] (the compressed-token map, shared by both layers), ``<name>.hash`` int64 (``pack_hash``), ``<name>.tq`` uint8
[shard rows, 256] / ``<name>.ts`` uint8 [shard rows, 8] (this rank's table shard), ``<name>.gw`` f32 [4, 5120]
(``q_weight * k_weight``), the wkv args (``pack``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl.expr import gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    bf2_f32,
    bf16_pair,
    butterfly,
    fp8_pack4,
    global_load,
    hw_rsq,
    ld_i32,
    load_ptr64,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from ...device.gfx950.peer import from_rank, tile_index, to_rank
from ...plan.program import Region, Scenario, Stage
from ...plan.trace import Space
from ...target import MI355X
from ...weights.mx import fp8_block_to_mxfp8
from ..base import Block, Channel, Fmt
from .linear import BF16Out, Linear
from .norm import e8m0_code

THREADS, WAVES = MI355X.threads, MI355X.waves
HIDDEN, HC, HEAD_DIM, NGRAM, NHEADS = 5120, 4, 256, 4, 8
NCOLS = (NGRAM - 1) * NHEADS  # 24 hash columns
KIN = NCOLS * HEAD_DIM  # 6144: wkv's K
NOUT = (HC + 1) * HIDDEN  # 25600: wkv's N (4 keys, the value)
SLICES = 4  # the gate's dim slices (TP4's column split; TP1 runs the same slices: the same bits)
SL = HIDDEN // SLICES  # 1280
SL_T = SL // 4  # 320 threads a slice in ``part`` (4 dims each)
PXV = HIDDEN // 2  # value pairs a row of ``<name>.px``
PX = (
    PXV + SLICES * 2 * HC
)  # + (dot, ksq) of the 4 copies per slice: pair PXV + 8 s + c (dot), + 4 + c (ksq)
QW, SW = KIN // 4, KIN // 32  # MXFP8 words / scale codes a row
CODE_W = HEAD_DIM // 4  # 64 code words a table row: a wave
# ``<name>.hash`` (int64): multipliers [NGRAM], primes [NCOLS], offsets [NCOLS + 1] (the last: the table's rows), pad
H_MULT, H_PRIME, H_OFF = 0, NGRAM, NGRAM + NCOLS
H_PAD = H_OFF + NCOLS + 1
H_WORDS = H_PAD + 1
DEAD = -1
ACT_MIN = 1e-4  # act_quant's amax clamp (``_pow2_scale(amax, 448, 1e-4)``)


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    spec: int
    tp: int
    hist_len: int
    eps: float
    out: str  # "" without the channel
    out_arg: str  # "" without the plain output
    ids: str
    pos: str
    hist: str
    tmap: str
    n: int  # wkv columns on this rank
    x: str  # the streams channel
    kv_arg: str = (
        ""  # mode "kv": the full kv stored there; "gate": read from there (an earlier launch's)
    )
    look_dst: bool = (
        False  # TP > 1: a look task puts to one rank (tasks x tp; a CTA's remote puts are the limit)
    )
    store_chunks: int = (
        1  # mode "kv", TP > 1: store tasks a row (each a 1 / n of the row's words from the 4 ranks)
    )
    staged: str = (
        ""  # mode "kv": the rows already looked up (plain bf16 [rows, 6144]: ATOM's staged rows), no table
    )
    xg: int = (
        0  # mode "gate": x is a sharded residual's gathered R (``HCSeam(res_gather=)``) over these ranks; 0: a channel
    )

    @property
    def cpr(self) -> int:  # hash columns a rank
        return NCOLS // self.tp

    @property
    def per_task(
        self,
    ) -> (
        int
    ):  # look: columns a task (a wave each): the most up to WAVES that divide a rank's
        # columns (TP2: 12 columns a rank -> 6 a task, 2 tasks; 8 a task left 4 columns unlooked-up, their pairs never
        # put: the gather waited forever)
        return max(d for d in range(1, min(WAVES, self.cpr) + 1) if self.cpr % d == 0)

    @property
    def chunks(self) -> int:  # look: tasks a row
        return self.cpr // self.per_task

    @property
    def slices(self) -> int:  # part: slices a rank
        return SLICES // self.tp


def _a(p: _Plan, s: str) -> str:
    return f"{p.name}.{s}"


# ------------------------------------------------------------------ device: look


@traced
def _put_word(a, i, w, tag, ok):
    if ok:
        mb.put_words(a, i, [w], tag)


@traced
def _store_i32(ptr, i, v, ok):
    if ok:
        bo.buffer_store(fx.Int32(v), rsrc(ptr), i)


def _compressed(k, p: _Plan, row, b, j, pos0, back):
    """The compressed id at position ``pos0 + j - back`` (python ``back``) and whether the n-gram is cut there
    (before position 0 or a dead token): a step row's own token through ``tmap``, an older one from ``hist``.
    """
    q = pos0 + j - back
    in_step = back <= j
    tok = ld_i32(k.args[p.ids], fx.max(row - back, fx.Int32(0)))
    from_step = ld_i32(k.args[p.tmap], fx.max(tok, fx.Int32(0)))
    from_hist = ld_i32(k.args[p.hist], b * p.hist_len + fx.max(q, fx.Int32(0)))
    z = in_step.select(from_step, from_hist)
    return z, (q < 0) | (z == DEAD)


def _look(k, t, p: _Plan):
    """Task t: row t // chunks, hash columns ``per_task`` from (chunk); wave w = one column (one table row).
    ``look_dst``: task t = (that task, destination rank t % tp): the same rows, put to that rank only.
    """
    dst_t = None
    if p.look_dst and p.tp > 1:
        t, dst_t = t // p.tp, t % p.tp
    row, chunk = t // p.chunks, t % p.chunks
    b, j = row // p.spec, row % p.spec
    lane = k.lane
    live = k.wave < p.per_task
    col0 = fx.Int32(0) if p.tp == 1 else k.rank * p.cpr
    col = col0 + chunk * p.per_task + fx.min(k.wave, fx.Int32(p.per_task - 1))
    hash_arg = k.args[_a(p, "hash")]
    pos0 = uniform(ld_i32(k.args[p.pos], b))
    pad = fx.Int64(load_ptr64(hash_arg, H_PAD))
    prime = load_ptr64(hash_arg, H_PRIME + col)
    off = load_ptr64(hash_arg, H_OFF + col)
    row_lo = load_ptr64(hash_arg, H_OFF + col0)
    # the 4 tokens (current first), then the rolling hash of the column's n-gram
    blocked = fx.Boolean(False)
    roll = [None] * NGRAM
    z0 = None
    for back in range(NGRAM):
        z, cut = _compressed(k, p, row, b, j, pos0, back)
        if back == 0:
            z0 = z
        blocked = blocked | cut
        zv = blocked.select(pad, fx.Int64(z))
        prod = zv * load_ptr64(hash_arg, H_MULT + back)
        roll[back] = prod if back == 0 else roll[back - 1] ^ prod
    i = col // NHEADS + 1  # the n-gram: 2 .. 4 tokens
    rolling = (i == 1).select(roll[1], (i == 2).select(roll[2], roll[3]))
    local = rolling % prime + off - row_lo
    # the row: 64 code words (a word a lane), its 8 scale codes (lane's group g = lane // 8)
    g = lane // 8
    qa = k.args[_a(p, "tq")] + local * HEAD_DIM + fx.Int64(lane * 4)
    sa = k.args[_a(p, "ts")] + local * (HEAD_DIM // 32) + fx.Int64((g // 4) * 4)
    word = fx.Int32(global_load(qa))
    sw = fx.Int32(global_load(sa))
    s = (sw >> ((g % 4) * 8)) & 255
    scale = (s == 0).select(fx.Int32(0x00400000), s << 23).bitcast(fx.Float32)
    vals = [
        fx.Float32(rocdl.cvt_f32_fp8(T.f32, word.ir_value(), e)) * scale
        for e in range(4)
    ]
    # act_quant (E8M0 per 32 = 8 lanes): amax, the power-of-two scale, the codes (exact shifts of the stored ones)
    amax = fx.max(fx.max(fx.max(vals[0], -vals[0]), fx.max(vals[1], -vals[1])),
                  fx.max(fx.max(vals[2], -vals[2]), fx.max(vals[3], -vals[3])))  # fmt: skip
    amax = butterfly(amax, (1, 2, 4), fx.max)
    code = e8m0_code(fx.max(amax, fx.Float32(ACT_MIN)))
    inv = ((254 - code) << 23).bitcast(fx.Float32)
    q = fp8_pack4(vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv)
    wi = row * QW + col * CODE_W + lane
    gi = row * SW + col * (HEAD_DIM // 32) + g
    first = (lane % 8) == 0
    if dst_t is not None:
        _put_word(to_rank(k, _a(p, "pq"), dst_t), wi, q, k.tag, live)
        _put_word(to_rank(k, _a(p, "ps"), dst_t), gi, code, k.tag, live & first)
    # this row's compressed id, for later launches' n-grams
    once = (chunk == 0) & (k.wave == 0) & (lane == 0)
    if dst_t is not None:
        once = once & (dst_t == 0)
    _store_i32(k.args[p.hist], b * p.hist_len + pos0 + j, z0, once)


def _gath(k, t, p: _Plan):
    """TP > 1, row t: every rank's columns (its slot of the peer regions) -> the local MXFP8 channel."""
    specs, dst = [], []
    for r in range(-(-QW // THREADS)):
        w = k.tid + r * THREADS
        wc = fx.min(w, fx.Int32(QW - 1))
        src = (wc // CODE_W) // p.cpr
        specs.append((from_rank(k, _a(p, "pq"), src), t * QW + wc, 1))
        dst.append((k.mb[_a(p, "e.q")], t * QW + wc, w < QW))
    gc = fx.min(k.tid, fx.Int32(SW - 1))
    sspec = [
        (from_rank(k, _a(p, "ps"), (gc // (HEAD_DIM // 32)) // p.cpr), t * SW + gc, 1)
    ]
    got = mb.poll(specs + sspec, k.tag)
    for (a, i, ok), v in zip(dst, got[: len(dst)], strict=True):
        _put_word(a, i, v[0], k.tag, ok)
    _put_word(k.mb[_a(p, "e.s")], t * SW + gc, got[-1][0], k.tag, k.tid < SW)


# ------------------------------------------------------------------ device: part, gate


KVX_CHUNK = (
    4 * THREADS
)  # mode "kv", TP > 1: kv words a ``kvx`` task pushes (to one rank)


def _slice(k, p: _Plan, t):
    """``part`` task t -> (row, the rank's slice ls, the slice s)."""
    row, ls = t // p.slices, t % p.slices
    return row, ls, ls if p.tp == 1 else k.rank * p.slices + ls


def _kv_col(p: _Plan, vec: int, ls, d):
    """Column of ``<name>.kv`` holding vector ``vec`` (0-3 keys, 4 value), dim d of the rank's slice ls."""
    return vec * (p.slices * SL) + ls * SL + d


@traced
def _put_f(a, i, v, tag, ok):
    if ok:
        mb.put(a, i, v, tag)


@traced
def _put_w2(a, i, w0, w1, tag, ok):
    if ok:
        mb.put_words(a, i, [w0, w1], tag)


def _px_dsts(k, p: _Plan):
    """Where ``part`` puts: the scratch region (TP1) or every rank's peer copy (its own slot)."""
    return (
        [k.mb[_a(p, "px")]]
        if p.tp == 1
        else [to_rank(k, _a(p, "px"), dst) for dst in range(p.tp)]
    )


def _part(k, t, p: _Plan):
    """Task t: (row, dim slice s). Thread i < 320: dims s * 1280 + 4 i .. + 3 of the 4 copies."""
    row, ls, s = _slice(k, p, t)
    i = fx.min(k.tid, fx.Int32(SL_T - 1))
    act = k.tid < SL_T
    d = 4 * i
    x = k.mb[p.x]
    kvw = [
        (row * p.n + _kv_col(p, c, ls, d)) // 2 for c in range(HC + 1)
    ]  # word index of each vector's 4 dims
    gw = rsrc(k.args[_a(p, "gw")])
    w4s = [
        fx.Vector(bo.buffer_load(gw, c * HIDDEN + s * SL + d, vec_width=4, dtype=T.f32))
        for c in range(HC)
    ]
    hsp = [
        (x, (row * HC * HIDDEN + c * HIDDEN + s * SL + d) // 2, 2) for c in range(HC)
    ]
    kv = k.mb[_a(p, "kv")]
    specs = []
    for c in range(HC):
        specs += [(kv, kvw[c], 2), hsp[c]]
    got = mb.poll(specs + [(kv, kvw[HC], 2)], k.tag)
    sums = []
    zero = fx.Float32(0.0)
    for c in range(HC):
        key = [v for word in got[2 * c] for v in bf2_f32(word)]
        h = [v for word in got[2 * c + 1] for v in bf2_f32(word)]
        dot, ksq = fx.Float32(0.0), fx.Float32(0.0)
        for e in range(4):
            dot = dot + (h[e] * fx.Float32(w4s[c][e])) * key[e]
            ksq = ksq + key[e] * key[e]
        sums += [act.select(dot, zero), act.select(ksq, zero)]
    dsts = _px_dsts(k, p)
    # the value slice, passed through as raw bf16 words
    vw = got[2 * HC]
    for a in dsts:
        _put_w2(a, row * PX + (s * SL + d) // 2, vw[0], vw[1], k.tag, act)
    # the 8 sums over the CTA: each wave's in butterfly order, then the waves in order (the same bits on any rank)
    red = k.ldsf
    wv = [wave_sum(v) for v in sums]
    gpu.barrier()
    for q in range(2 * HC):
        _lds_put(red + (k.wave * (2 * HC) + q), wv[q], k.lane == 0)
    gpu.barrier()
    q = fx.min(k.tid, fx.Int32(2 * HC - 1))
    tot = fx.Float32(fx.ptr_load(red + q))
    for w in range(1, WAVES):
        tot = tot + fx.Float32(fx.ptr_load(red + (w * (2 * HC) + q)))
    # q = 2 c (dot) / 2 c + 1 (ksq) -> pair PXV + 8 s + c / + 4 + c
    c, kind = q // 2, q % 2
    for a in dsts:
        _put_f(
            a, row * PX + PXV + 2 * HC * s + kind * HC + c, tot, k.tag, k.tid < 2 * HC
        )
    gpu.barrier()  # the next task reuses the LDS


def _kvx(k, t, p: _Plan):
    """Mode "kv", TP > 1: task t = (row, chunk, destination rank): this rank's kv words -> that rank's peer copy
    (one destination a task: a CTA's remote puts are the limit)."""
    words = p.n // 2
    chunks = -(-words // KVX_CHUNK)
    row, rest = t // (chunks * p.tp), t % (chunks * p.tp)
    ch, dst = rest // p.tp, rest % p.tp
    specs, idx = [], []
    for r in range(KVX_CHUNK // THREADS):
        w = ch * KVX_CHUNK + k.tid + r * THREADS
        wc = fx.min(w, fx.Int32(words - 1))
        specs.append((k.mb[_a(p, "kv")], row * words + wc, 1))
        idx.append((row * words + wc, w < words))
    got = mb.poll(specs, k.tag)
    a = to_rank(k, _a(p, "kvp"), dst)
    for (i, ok), v in zip(idx, got, strict=True):
        _put_word(a, i, v[0], k.tag, ok)


def kv_share_word(w, tp: int):
    """Word ``w`` of a rank's kv share (``rank_columns``' order, 2 bf16 a word) -> (its vector, its word in the
    rank's part of the vector): a vector's words a rank 5120 / tp / 2 (TP4: 640). Python ints or traced ones.
    """
    vw = HIDDEN // tp // 2
    return w // vw, w % vw


def _store_full(k, t, p: _Plan):
    """Mode "kv", TP > 1: row t's full kv (every rank's share, from its slot) -> the plain arg in the natural
    column order [4 keys, value] x 5120 (``kv_share_word``). ``store_chunks`` > 1: task t = (row, chunk), the chunk's
    1 / n of the share's words."""
    words = p.n // 2  # a rank's share
    cw = -(-words // p.store_chunks)
    row, w0 = t, 0
    if p.store_chunks > 1:
        row, w0 = t // p.store_chunks, (t % p.store_chunks) * cw
    for r in range(-(-cw // THREADS)):
        w = w0 + k.tid + r * THREADS
        wc = fx.min(w, fx.Int32(words - 1))
        got = mb.poll(
            [(from_rank(k, _a(p, "kvp"), s), row * words + wc, 1) for s in range(p.tp)],
            k.tag,
        )
        vec, d2 = kv_share_word(wc, p.tp)
        ok = (w < words) & (w < w0 + cw) if p.store_chunks > 1 else w < words
        for s in range(p.tp):
            nat = (
                row * (NOUT // 2) + vec * (HIDDEN // 2) + s * (HIDDEN // p.tp // 2) + d2
            )
            _store_i32(k.args[p.kv_arg], nat, got[s][0], ok)


@traced
def _lds_put(ptr, v, ok):
    if ok:
        fx.ptr_store(v, ptr)


def _px_addr(k, p: _Plan, s):
    """Where ``gate`` polls slice ``s``'s part: the scratch region, or the slot of the rank that computed it."""
    return k.mb[_a(p, "px")] if p.tp == 1 else from_rank(k, _a(p, "px"), s // p.slices)


PER_T = PXV // THREADS  # 5 pairs a thread in ``gate``


def _gate(k, t, p: _Plan):
    """Task t: (row, copy c). Thread i: pairs i + 512 m (m < 5) of the row's 5120 dims."""
    row, c = t // HC, t % HC
    x = k.mb[p.x]
    pairs = [k.tid + THREADS * m for m in range(PER_T)]
    hspec = [(x, (row * HC * HIDDEN + c * HIDDEN) // 2 + pi, 1) for pi in pairs]
    vspec = [(_px_addr(k, p, pi // (SL // 2)), row * PX + pi, 1) for pi in pairs]
    sspec = []
    for s in range(SLICES):
        a = _px_addr(k, p, s)
        sspec += [
            (a, row * PX + PXV + 2 * HC * s + c, 1),
            (a, row * PX + PXV + 2 * HC * s + HC + c, 1),
        ]
    hw = mb.poll(hspec, k.tag)
    vw = mb.poll(vspec + sspec, k.tag)
    h = [v for w in hw for v in bf2_f32(w[0])]
    hsq = fx.Float32(0.0)
    for v in h:
        hsq = hsq + v * v
    red = k.ldsf
    wv = wave_sum(hsq)
    gpu.barrier()
    _lds_put(red + k.wave, wv, k.lane == 0)
    gpu.barrier()
    hsq = fx.Float32(fx.ptr_load(red + 0))
    for w in range(1, WAVES):
        hsq = hsq + fx.Float32(fx.ptr_load(red + w))
    sv = [g[0].bitcast(fx.Float32) for g in vw[PER_T:]]
    dot, ksq = sv[0], sv[1]
    for s in range(1, SLICES):
        dot, ksq = dot + sv[2 * s], ksq + sv[2 * s + 1]
    _finish(
        k, p, row, c, pairs, h, [bf2_f32(vw[m][0]) for m in range(PER_T)], dot, ksq, hsq
    )
    gpu.barrier()  # the next task reuses the LDS


def _gate_fused(k, t, p: _Plan):
    """Mode "gate": task t = (row, copy c), the full kv in the plain arg ``kv_arg`` (natural columns). ``part``'s
    4 slice partials of copy c with ``part``'s exact mapping and order (thread i < 320: dims s 1280 + 4 i .. + 3,
    summed in order, each wave by butterfly, the waves in order), then ``gate``'s arithmetic: the same bits as the
    two-stage path, without its hand-off."""
    row, c = t // HC, t % HC
    x = k.mb[p.x] if not p.xg else from_rank(k, p.x, 0)
    kva = rsrc(k.args[p.kv_arg])
    gw = rsrc(k.args[_a(p, "gw")])
    i = fx.min(k.tid, fx.Int32(SL_T - 1))
    act = k.tid < SL_T
    d = 4 * i
    pairs = [k.tid + THREADS * m for m in range(PER_T)]
    # loads: the slices' keys / weights (part's mapping), the value (gate's mapping), then the polls of h (both)
    keys = [fx.Vector(bo.buffer_load(kva, (row * NOUT + c * HIDDEN + s * SL + d) // 2, vec_width=2, dtype=T.i32))
            for s in range(SLICES)]  # fmt: skip
    w4s = [
        fx.Vector(bo.buffer_load(gw, c * HIDDEN + s * SL + d, vec_width=4, dtype=T.f32))
        for s in range(SLICES)
    ]
    vals = [fx.Int32(bo.buffer_load(kva, (row * NOUT + HC * HIDDEN) // 2 + pi, vec_width=1, dtype=T.i32))
            for pi in pairs]  # fmt: skip
    hs = mb.poll(
        [
            (x, _xw(k, p, (row * HC * HIDDEN + c * HIDDEN + s * SL + d) // 2), 2)
            for s in range(SLICES)
        ],
        k.tag,
    )
    hw = mb.poll(
        [(x, _xw(k, p, (row * HC * HIDDEN + c * HIDDEN) // 2 + pi), 1) for pi in pairs],
        k.tag,
    )
    zero = fx.Float32(0.0)
    sums = []
    for s in range(SLICES):
        key = [v for j in range(2) for v in bf2_f32(fx.Int32(keys[s][j]))]
        h = [v for word in hs[s] for v in bf2_f32(word)]
        dot, ksq = fx.Float32(0.0), fx.Float32(0.0)
        for e in range(4):
            dot = dot + (h[e] * fx.Float32(w4s[s][e])) * key[e]
            ksq = ksq + key[e] * key[e]
        sums += [act.select(dot, zero), act.select(ksq, zero)]
    hg = [v for w in hw for v in bf2_f32(w[0])]
    hsq = fx.Float32(0.0)
    for v in hg:
        hsq = hsq + v * v
    sums.append(hsq)
    red = k.ldsf
    nv = len(sums)  # 2 x 4 slice partials + sum h^2
    wv = [wave_sum(v) for v in sums]
    gpu.barrier()
    for q in range(nv):
        _lds_put(red + (k.wave * nv + q), wv[q], k.lane == 0)
    gpu.barrier()
    tot = []
    for q in range(nv):
        v = fx.Float32(fx.ptr_load(red + q))
        for w in range(1, WAVES):
            v = v + fx.Float32(fx.ptr_load(red + (w * nv + q)))
        tot.append(v)
    dot, ksq, hsq = tot[0], tot[1], tot[2 * SLICES]
    for s in range(1, SLICES):
        dot, ksq = dot + tot[2 * s], ksq + tot[2 * s + 1]
    _finish(k, p, row, c, pairs, hg, [bf2_f32(v) for v in vals], dot, ksq, hsq)
    gpu.barrier()  # the next task reuses the LDS


def _xw(k, p: _Plan, w):
    """x's pair ``w`` (row-major [rows x HC, HIDDEN / 2] pairs): itself, or (``xg``) its index from the gathered
    region's slot (parity, 0) (``peer.tile_index``, the sharded seam's tiles)."""
    if not p.xg:
        return w
    from .mhc import GATHER_TILE

    return tile_index(w, HIDDEN // 2, p.xg, p.rows * HC, GATHER_TILE)


def _finish(k, p: _Plan, row, c, pairs, h, vals, dot, ksq, hsq):
    """The gate from the summed partials, ``out = bf16(h + gate value)`` for the thread's pairs."""
    eps = fx.Float32(p.eps)
    rstd = hw_rsq(hsq / fx.Float32(float(HIDDEN)) + eps) * hw_rsq(
        ksq / fx.Float32(float(HIDDEN)) + eps
    )
    dot = dot * rstd * fx.Float32(HIDDEN**-0.5)
    mag = fx.Float32(fmath.sqrt(fx.max(fx.max(dot, -dot), fx.Float32(1e-6))))
    sign = dot.bitcast(fx.Int32) & fx.Int32(-0x80000000)
    signed = ((mag.bitcast(fx.Int32) & 0x7FFFFFFF) | sign).bitcast(fx.Float32)
    gate = fx.Float32(1.0) / (fx.Float32(1.0) + fx.Float32(fmath.exp(-signed)))
    for m, pi in enumerate(pairs):
        v0, v1 = vals[m]
        y0, y1 = h[2 * m] + gate * v0, h[2 * m + 1] + gate * v1
        e = row * HC * HIDDEN + c * HIDDEN + 2 * pi
        word = bf16_pair(y0, y1).bitcast(fx.Int32)
        if p.out:
            mb.put_words(k.mb[p.out], e // 2, [word], k.tag)


# ------------------------------------------------------------------ plan


class Engram(Block):
    """One layer's Engram (module docstring). ``x``: the streams, a tagged BF16 channel of width 4 x 5120 (row r
    = the 4 copies of row r); ``out`` (tagged BF16, the same shape) and / or ``out_arg`` (plain bf16 [rows, 4, 5120]).
    ``hist_len``: ``hist``'s positions per request."""

    def __init__(self, name: str, x: Channel, ids: str, pos: str, hist: str, tmap: str, hist_len: int,
                 out: Channel | None = None, out_arg: str = "", eps: float = 1e-20,
                 wkv_knobs: dict | None = None, mode: str = "full", kv_arg: str = "", look_dst: bool = False,
                 store_chunks: int = 1, staged: str = "", x_gathered: bool = False):  # fmt: skip
        if x.fmt is not Fmt.BF16 or x.flagged or x.width != HC * HIDDEN:
            raise ValueError(
                f"Engram {name}: x is a tagged BF16 channel of width {HC * HIDDEN}"
            )
        if out is not None and (
            out.fmt is not Fmt.BF16 or out.flagged or out.width != HC * HIDDEN
        ):
            raise ValueError(
                f"Engram {name}: out is a tagged BF16 channel of width {HC * HIDDEN}"
            )
        if mode not in ("full", "kv", "gate") or (mode != "full" and not kv_arg):
            raise ValueError(
                f"Engram {name}: mode full / kv / gate (kv, gate: with kv_arg), got {mode!r}"
            )
        if staged and mode != "kv":
            raise ValueError(f"Engram {name}: staged rows feed the kv half (mode 'kv')")
        if out is None and not out_arg and mode != "kv":
            raise ValueError(f"Engram {name}: give an output channel or arg")
        self.mode, self.kv_arg, self.look_dst, self.store_chunks = (
            mode,
            kv_arg,
            look_dst,
            store_chunks,
        )
        self.staged = staged
        # TP, a sharded residual: x is the gathered R a sharded seam put (``HCSeam(res_gather=)``, mode "gate")
        if x_gathered and mode != "gate":
            raise ValueError(
                f"Engram {name}: x_gathered reads a sharded seam's gathered R in mode 'gate'"
            )
        self.x_gathered = x_gathered
        self.name, self.x, self.out, self.out_arg = name, x, out, out_arg
        self.ids, self.pos, self.hist, self.tmap, self.hist_len, self.eps = (
            ids,
            pos,
            hist,
            tmap,
            hist_len,
            eps,
        )
        self.e = Channel(f"{name}.e", KIN, Fmt.MXFP8)
        self.wkv_knobs = dict(
            wkv_knobs or {}
        )  # Linear knobs (k_split, group_rows, wave_rows) for wkv

    def channels(self, tp: int, rows: int = 1) -> tuple[Channel, Linear]:
        """wkv's output channel and its ``Linear``. Default knobs: the plan's, except TP1 above 8 rows (the 25600
        columns' derived shape does not fit there: K in 2 slices, 248 tasks)."""
        kv = Channel(f"{self.name}.kv", NOUT // tp, Fmt.BF16)
        knobs = self.wkv_knobs or ({"k_split": 2} if tp == 1 and rows > 8 else {})
        return kv, Linear(
            f"{self.name}.wkv",
            ("wkv",),
            self.e,
            NOUT // tp,
            BF16Out(kv),
            weight_format="mxfp8",
            **knobs,
        )

    def plan(self, sc: Scenario) -> _Plan:
        if NCOLS % sc.tp or HIDDEN % sc.tp or (self.mode == "full" and SLICES % sc.tp):
            raise ValueError(f"Engram {self.name}: TP dividing the {NCOLS} hash columns and the width (mode 'full': "
                             f"the {SLICES} gate slices too), got {sc.tp}")  # fmt: skip
        return _Plan(self.name, sc.rows, sc.spec, sc.tp, self.hist_len, self.eps, self.out.name if self.out else "",
                     self.out_arg, self.ids, self.pos, self.hist, self.tmap, NOUT // sc.tp, self.x.name,
                     self.kv_arg if self.mode == "gate" else "", self.look_dst and sc.tp > 1,
                     self.store_chunks if sc.tp > 1 else 1, self.staged,
                     sc.tp if self.x_gathered else 0)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        key = ("engram",) + (p,)
        n = lambda s: f"{self.name}.{s}"
        _, lin = self.channels(sc.tp, sc.rows)
        st = []
        if self.mode != "gate":
            look_args = (p.ids, p.pos, p.hist, p.tmap, n("hash"), n("tq"), n("ts"))
            look_w = (Region(n("pq"), sc.rows * QW, (n("look"),), space=Space.PEER),
                      Region(n("ps"), sc.rows * SW, (n("look"),), space=Space.PEER))  # fmt: skip
            # hist: the step rows' ids are stored for later launches only (a look reads positions before ``pos``)
            st.append(Stage(n("look"), sc.rows * p.chunks * (sc.tp if p.look_dst else 1), partial(_look, p=p),
                            writes=tuple(look_w), args=look_args, key=key, later_args=(p.hist,)))  # fmt: skip
            if sc.tp > 1:
                st.append(Stage(n("gath"), sc.rows, partial(_gath, p=p),
                                writes=self.e.regions(sc.rows, (n("gath"),)), key=key))  # fmt: skip
            st += lin.stages(sc)
        if self.mode == "kv":
            sp = replace(p, kv_arg=self.kv_arg)
            # TP > 1: the shares all-gathered, so the gate's launch has the full kv locally (no peer hop there)
            chunks = -(-(p.n // 2) // KVX_CHUNK)
            st.append(Stage(n("kvx"), sc.rows * chunks * sc.tp, partial(_kvx, p=sp),
                            writes=(Region(n("kvp"), sc.rows * p.n // 2, (n("kvx"),), space=Space.PEER),),
                            key=key + ("kvx",)))  # fmt: skip
            st.append(Stage(n("store"), sc.rows * p.store_chunks, partial(_store_full, p=sp), args=(self.kv_arg,),
                            key=key + ("store",)))  # fmt: skip
            return st
        if (
            self.mode == "gate"
        ):  # the full kv is local (mode "kv" all-gathered it): one stage, no hand-off
            p = replace(p, tp=1, n=NOUT)
            out_w = (
                self.out.regions(sc.rows, (n("gate"),)) if self.out is not None else ()
            )
            args = (n("gw"), p.kv_arg) + ((p.out_arg,) if p.out_arg else ())
            return [Stage(n("gate"), sc.rows * HC, partial(_gate_fused, p=p), lds_words=WAVES * (2 * SLICES + 1),
                          writes=tuple(out_w), args=args, key=("engram",) + (p,) + ("fused",))]  # fmt: skip
        px = Region(
            n("px"),
            sc.rows * PX,
            (n("part"),),
            space=Space.PEER if p.tp > 1 else Space.SCRATCH,
        )
        st.append(Stage(n("part"), sc.rows * p.slices, partial(_part, p=p), lds_words=WAVES * 2 * HC, writes=(px,),
                        args=(n("gw"),), key=key))  # fmt: skip
        out_w = self.out.regions(sc.rows, (n("gate"),)) if self.out is not None else ()
        st.append(Stage(n("gate"), sc.rows * HC, partial(_gate, p=p), lds_words=WAVES,
                        writes=tuple(out_w), args=(p.out_arg,) if p.out_arg else (), key=key))  # fmt: skip
        return st

    def pack(
        self, weights: dict, tp: int = 1, rank: int = 0, device="cuda"
    ) -> dict[str, torch.Tensor]:
        """``weights``: ``wkv`` (float8_e4m3fn [25600, 6144]), ``wkv_s`` (E8M0 / uint8 [800, 192]), ``q_weight`` /
        ``k_weight`` [4, 5120]; plus ``hash`` (``pack_hash``). Returns this rank's args (not the table: ``shard``).
        """
        out = {}
        if self.mode != "gate":
            wkv = weights["wkv"].view(torch.float8_e4m3fn)
            m = fp8_block_to_mxfp8(wkv, weights["wkv_s"].view(torch.uint8))
            cols = rank_columns(tp, rank)
            m.packed, m.codes = m.packed[cols].contiguous().to(device), m.codes[
                cols
            ].contiguous().to(device)
            _, lin = self.channels(tp)
            out.update(
                {
                    k: v.to(device)
                    for k, v in lin.pack(SimpleNamespace(linears={"wkv": m})).items()
                }
            )
            out[f"{self.name}.hash"] = weights["hash"].to(device)
        if self.mode != "kv":
            gw = weights["q_weight"].float() * weights["k_weight"].float()
            out[f"{self.name}.gw"] = gw.contiguous().to(device)
        return out


# ------------------------------------------------------------------ host helpers


def rank_columns(tp: int, rank: int) -> torch.Tensor:
    """wkv's output rows rank ``rank`` computes, in ``<name>.kv``'s column order: for each vector (4 keys, the
    value) the dims ``5120 / tp`` rank .. (at TP 1 / 2 / 4 the rank's gate slices; any tp dividing the width: the
    split modes, whose gate keeps its 4 slices on the gathered kv)."""
    lo, hi = rank * HIDDEN // tp, (rank + 1) * HIDDEN // tp
    return torch.cat(
        [torch.arange(v * HIDDEN + lo, v * HIDDEN + hi) for v in range(HC + 1)]
    )


def pack_hash(multipliers, primes, offsets, num_rows: int, pad: int) -> torch.Tensor:
    """``<name>.hash`` (int64 [H_WORDS]) of one layer from ``NgramHashState``'s buffers: multipliers [4], primes
    [3, 8] (column order: n-gram major), offsets [24], the table's rows, the compressed pad id.
    """
    out = torch.zeros(H_WORDS, dtype=torch.int64)
    out[H_MULT : H_MULT + NGRAM] = torch.as_tensor(multipliers).flatten()
    out[H_PRIME : H_PRIME + NCOLS] = torch.as_tensor(primes).flatten()
    out[H_OFF : H_OFF + NCOLS] = torch.as_tensor(offsets).flatten()
    out[H_OFF + NCOLS] = num_rows
    out[H_PAD] = pad
    return out


def shard_rows(hash_words: torch.Tensor, tp: int, rank: int) -> tuple[int, int]:
    """The table rows [lo, hi) rank ``rank`` holds: its hash columns' ranges, contiguous."""
    cpr = NCOLS // tp
    off = hash_words[H_OFF : H_OFF + NCOLS + 1]
    return int(off[rank * cpr]), int(off[(rank + 1) * cpr])
