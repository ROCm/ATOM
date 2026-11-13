# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Compressor``: DeepSeek-V4.1's KV compressor at a KV-source layer (checkpoint ``inference/model.py``
``Compressor.forward`` and the latent's path in ``Attention._compress_kv``), per row of the step.

What one decode row at position ``P`` does in the reference (``x`` = ``attn_norm``'s BF16 output, 5120):

- ratio 1 (layer 20): ``kv = bf16(x . wkv^T)`` (BF16 ``F.linear``, 512 out) and the row always emits:
  ``latent = norm(kv)``;
- ratio 2 (layers 2, 8, 14): ``kv = x . wkv^T``, ``score = x . wgate^T`` (fp32: the BF16 weights promoted, so
  exact products and fp32 sums); both go into state slot ``P % 2``. Only when ``(P + 1) % 2 == 0`` (odd ``P``)
  the group completes: per channel ``w = softmax(score[slot 0], score[slot 1])`` and
  ``latent = norm(bf16(kv[0] w0 + kv[1] w1))``; an even row emits nothing (the reference returns ``None``);
- ``norm``: RMSNorm over the 512 channels, eps 1e-20, one rounding: ``bf16(x rsqrt(mean x^2 + eps) w)``;
- then the KV-cache entry: RoPE on the last 64 channels at position ``P + 1 - ratio`` (the group's first token;
  the layer's compressed YaRN table) -> FP4 QAT (``fp4_act_quant(inplace)``: groups of 16, E4M3 scale
  ``e4m3(max(amax, 6 / 512) / 6)``, E2M1 round-to-nearest-even, dequantized) -> ``compress_kv[P // ratio]``.
  ``qat=False`` (an engine whose cache holds the bf16 latent unquantized, e.g. ATOM's bf16 pool: its arithmetic, not
  the official model's) stores the rotated bf16 latent as is.

So at ratio 2 a row's work is the two 512 x 5120 GEMVs and the state write; the pooling, norm, RoPE, QAT and
the cache write happen on every other row only. At ratio 1 every row emits.

Stages: the projections are ``GatheredDot``'s dense K-split MFMA mode (``kslices`` x ``tiles`` knobs: one dot
for ``wkv``, one for ``wgate``; bf16 x and W on ``mfma_f32_16x16x32_bf16``: exact products, fp32 sums, the
reference's arithmetic up to the sum order) with ``sum_stage=False``; ``<name>.pool`` (a task a row, a thread a
channel) adds the dot's slice partials in slice order and does the rest.

Ratio-2 state (ATOM's compress rings in place, plain fp32 args ``kv_state`` / ``score_state``: request slot s's
ring at ``s * stride`` (``paged.ring_stride``), position P at ring slot ``P % slots``, ``slots`` = ATOM's 7 = 2 + a verify's
drafts): ``slots`` = 2 is the reference's layout. A row whose group completes takes its partner (position P - 1) from the
step when the partner is a row of the same step (its partials polled again: the same sum, the same bits), else
from the state (written by an earlier launch). Only the last row of the step per slot writes the state; a row
that would overwrite the slot the step's first row reads waits for that row's ``<name>.rd`` flag (an in-launch
hand-off through a tagged region, not plain memory). With ``slots`` >= spec + 1 no row of a step overwrites a slot an
earlier position still needs: a verify step whose later rows are rejected leaves the accepted rows' state intact (the reference's 2 slots assume every row is accepted).

Outputs:

- ``cache`` (ATOM's pool in place: the owner's main rows, ``device.gfx950.paged``): entry ``P // ratio`` := the
  QAT'd (``qat``), rotated latent, for every emitting row of a live request (later launches read it); ``kv8``: ATOM's
  fp8 pool, the entry quantized to its MXFP8 row (``paged.KV8_ROW_BYTES``) as ATOM's writer does;
- ``out`` (a tagged BF16 channel, 512): the same entry, row r (zeros on a row that does not emit): the in-launch
  hand-off for this layer's attention / indexer (plain stores are not visible to another XCD within a launch);
- ``latent`` (optional tagged BF16 channel, 512): the normed latent before RoPE (what the indexer's ``wk``
  projects; zeros on a row that does not emit).

Args: ``pos`` int32 [batch] (the step's first position per request), ``rope`` / ``rope.sin`` (ATOM's bf16 RoPE
caches, ``rope_cs``), ``norm`` bf16 [512], the weights bf16 [512, 5120] (``pack``), the request words and the page
geometry (``paged.REQ``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import mailbox as mb
from ...device.gfx950 import paged
from ...device.gfx950.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rsq,
    ld_i32,
    rope_cs,
    rope_sin,
    rsrc,
    traced,
    uniform,
    wave_sum,
    xshfl,
)
from ...plan.program import Region, Scenario, Stage
from ...target import MI355X
from ..base import Block, Channel, Fmt
from .attention.mla_window import _rope_pair
from .gathered import GatheredDot
from .norm import dot2_bf16

THREADS = MI355X.threads
HD, ROPE, HIDDEN, IHD = 512, 64, 5120, 128  # IHD: the indexer's key width
# the bf16 ``aux`` arg (one pointer: SGPRs): the compressor's norm [512], then (index keys) wk [128, 512] in the
# pool's lane order (``pack_wk``), k_norm [128]
AUX_WK = HD // 2  # words
AUX_KNORM = AUX_WK + IHD * HD // 2
FP4_GROUP = 16  # the compressed cache's QAT group (E4M3 scales)
FP4_MIN_AMAX = 6 * 2**-9  # fp4_act_quant's clamp: the smallest scale is E4M3's 2^-9
NORM_LANES = 64  # the norm's sum map: lane i holds channels 8 i .. 8 i + 7 (NormSpec(512, threads 64, wps 4))
assert HD == THREADS


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    spec: int
    batch: int
    ratio: int
    slots: int
    kslices: int
    kv_part: str  # the dots' slice partial channels (width kslices x 512)
    score_part: str  # "" at ratio 1
    pos: str
    rope: str
    aux: str
    cache: str
    cache_len: int
    kv_state: str
    score_state: str
    out: str
    latent: str  # "" without the pre-RoPE latent channel
    eps: float
    poll: int  # the pool's pair specs a poll batch (0: one batch a (channel, row))
    index_out: str  # the index keys' channel ("" without the index-key path)
    flag: bool  # some row overwrites the slot the step's first row reads: the .rd hand-off
    qat: bool = (
        True  # the cache entry's FP4 QAT (False: the rotated bf16 latent, unquantized)
    )
    kv8: bool = False  # the cache is ATOM's fp8 pool: the entry stored as its MXFP8 row (``paged.KV8_ROW_BYTES``)

    @property
    def rd(self) -> str:
        return f"{self.name}.rd"


# ------------------------------------------------------------------ device


def _slices(k, groups, c, s, batch):
    """Channel c of each (part channel, row) in ``groups``: the dot's ``s`` slice partials, added in slice order
    (``GatheredDot``'s ``.sum``). ``batch``: pair specs per poll batch (0: one batch a group).
    """
    specs = [
        (k.mb[part], r * (s * HD) + i * HD + c, 1)
        for part, r in groups
        for i in range(s)
    ]
    got = [
        g
        for q in range(len(groups))
        for g in mb.poll(specs[q * s : (q + 1) * s], k.tag)
    ]
    out = []
    for q in range(len(groups)):
        v = fx.Float32(0.0)
        for g in got[q * s : (q + 1) * s]:
            v = v + g[0].bitcast(fx.Float32)
        out.append(v)
    return out


def _e2m1(a):
    """|value| / scale (>= 0) -> the E2M1 level, round to nearest, ties to the even code, saturating at 6
    (``fp4_act_quant``'s argmin with its tie rule)."""
    lv = fx.Float32(6.0)
    for bound, level, tie_up in ((5.0, 4.0, False), (3.5, 3.0, True), (2.5, 2.0, False), (1.75, 1.5, True),
                                 (1.25, 1.0, False), (0.75, 0.5, True), (0.25, 0.0, False)):  # fmt: skip
        below = (a < fx.Float32(bound)) if tie_up else (a <= fx.Float32(bound))
        lv = below.select(fx.Float32(level), lv)
    return lv


def _rope_lanes(v, c, first, cs):
    """RoPE of the bf16 value ``v`` held one a lane, adjacent lanes a pair, on lanes ``c >= first`` (cos, sin
    ``cs`` of the lane's pair): the rotated value, bf16-rounded (lanes below ``first`` keep ``v``).
    """
    other = xshfl(v, 1)
    odd = (c & 1) == 1
    re, im = odd.select(other, v), odd.select(v, other)
    rr, ri = _rope_pair(re, im, fx.Float32(cs[0]), fx.Float32(cs[1]), False)
    return (c >= first).select(bf16_round(odd.select(ri, rr)), v)


def _fp4_qat_pow2(v):
    """``fp4_act_quant(inplace, block 32, E8M0 scales)`` (the index keys) of the bf16 value ``v``, its group the 32
    lanes of a half wave: scale = 2^ceil(log2(max(amax, 6 2^-126) / 6)) (torch's log2 / ceil / exp2), the
    dequantized value."""
    amax = butterfly(fx.Float32(fx.max(v, -v)), (1, 2, 4, 8, 16), fx.max)
    amax = fx.max(amax, fx.Float32(6 * 2**-126))
    scale = hw_exp2(fx.Float32(fmath.ceil(fmath.log2(amax / fx.Float32(6.0)))))
    q = v / scale
    lv = _e2m1(fx.Float32(fx.max(q, -q)))
    neg = q.bitcast(fx.Int32) < fx.Int32(0)
    return neg.select(-lv, lv) * scale


def _index_loads(k, p: _Plan, c, epos):
    """The index-key path's plain loads, issued at task start (their latency hides under the dots' wait): lane c's
    16 wk steps (``pack_wk``), its RoPE pair and its k_norm weight."""
    r_aux = rsrc(k.args[p.aux])
    wk = [fx.Vector(bo.buffer_load(r_aux, AUX_WK + (i * THREADS + c) * 4, vec_width=4, dtype=T.i32))
          for i in range(16)]  # fmt: skip
    ci = fx.min(c, fx.Int32(IHD - 1))
    kpair = fx.max(ci - (IHD - ROPE), fx.Int32(0)) // 2
    cs_k = rope_cs(k, p.rope, epos, kpair)
    g_lo, g_hi = bf2_f32(ld_i32(k.args[p.aux], AUX_KNORM + ci // 2))
    return wk, cs_k, ((ci & 1) == 1).select(g_hi, g_lo)


def _index_key(k, t, p: _Plan, c, b, pos, loads, lat, emits):
    """The indexer's key of the latent (``Indexer.forward`` of a KV source): k_norm(wk latent) [128], RoPE on its
    last 64 at the group's first position, the FP4 QAT (block 32, E8M0) -> ``index_out`` (the Indexer's own pack
    stage keeps the FP4 cache).
    Lane c: wk row c // 4, K quarter c % 4 (``_index_loads``), summed with dot2 into 4 accumulators (the order of
    the reference's BF16 ``F.linear`` is not known; the bf16 key absorbs it)."""
    wk, cs_k, gamma = loads
    lw, kb = (
        HD,
        HD + HD // 2,
    )  # LDS (words): the latent as packed bf16 at lw, the 128 keys (f32) at kb
    ci = fx.min(c, fx.Int32(IHD - 1))
    _lds_put(
        k.lds + (lw + c // 2),
        bf16_pair(lat, xshfl(lat, 1)).bitcast(fx.Int32),
        (c & 1) == 0,
    )
    gpu.barrier()
    o, kq = c // 4, c % 4
    acc = [fx.Float32(0.0)] * 4
    for i in range(16):
        xv = fx.Vector(
            fx.ptr_load(
                k.lds + (lw + kq * 64 + 4 * i),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        for e in range(4):
            acc[e] = dot2_bf16(wk[i][e], xv[e], acc[e])
    tot = butterfly((acc[0] + acc[1]) + (acc[2] + acc[3]), (1, 2))
    _lds_put(k.ldsf + (kb + o), bf16_round(tot), kq == 0)
    gpu.barrier()
    kk = fx.Float32(fx.ptr_load(k.ldsf + (kb + ci)))
    # k_norm over 128 (lane i < 16 sums keys 8 i .. 8 i + 7 with dot2, the 512-wide norm's map)
    lane = c % NORM_LANES
    ss = fx.Float32(0.0)
    for w in range(4):
        vals = [
            fx.Float32(fx.ptr_load(k.ldsf + (kb + 8 * (lane % 16) + 2 * w + e)))
            for e in range(2)
        ]
        word = bf16_pair(vals[0], vals[1]).bitcast(fx.Int32)
        word = (lane < 16).select(word, fx.Int32(0))
        ss = dot2_bf16(word, word, ss)
    rstd = hw_rsq(wave_sum(ss) * (1.0 / IHD) + fx.Float32(p.eps))
    key = bf16_round(kk * rstd * gamma)
    key = _fp4_qat_pow2(_rope_lanes(key, ci, IHD - ROPE, cs_k))
    if emits is not None:
        key = emits.select(key, fx.Float32(0.0))
    nxt = xshfl(key, 1)
    ok = ((c & 1) == 0) & (c < IHD)
    _put_pair(k, k.mb[p.index_out], t * IHD + ci, key, nxt, ok)


@traced
def _lds_put(ptr, v, ok):
    if ok:
        fx.ptr_store(v, ptr)


@traced
def _put_pair(k, a, e, v0, v1, ok):
    if ok:
        chan.put_bf16(a, e, [v0, v1], k.tag)


@traced
def _store_entry(addr, w, ok, nbytes=4):
    if ok:
        paged.global_store(addr, w, nbytes)


def _kv8_entry(k, p: _Plan, b, pos, c, entry, ok):
    """Channel c's ``entry`` (bf16) into entry pos // ratio's MXFP8 row of ATOM's fp8 pool where ``ok``, as ATOM's
    writer stores it (``paged.kv8_code``: its group the 32 channels of a half wave); -> the value the row holds
    (QAT: what ATOM's readers take). A lane c % 4 == 0 stores channels c .. c + 3 (a word), lane c % 32 == 0 the
    group's code."""
    amax = butterfly(fx.Float32(fx.max(entry, -entry)), (1, 2, 4, 8, 16), fx.max)
    code = paged.kv8_code(amax)
    scaled = paged.kv8_scaled(entry, code)
    n1 = xshfl(scaled, 1)
    word = fp8_pack4(scaled, n1, xshfl(scaled, 2), xshfl(n1, 2))  # (lane c ^ 2's neighbour: c + 3)
    row = paged.row_addr(k, p.cache, b, pos // p.ratio, p.ratio, row_bytes=paged.KV8_ROW_BYTES)
    _store_entry(row + fx.Int64(c), word, ok & ((c % 4) == 0))
    _store_entry(row + fx.Int64(HD + c // paged.KV8_GROUP), code, ok & ((c % paged.KV8_GROUP) == 0), 1)
    return paged.kv8_value(scaled, code)


def _state_at(k, p: _Plan, b, slot):
    """Word index of channel 0 of request b's ring slot ``slot`` in ATOM's compress rings."""
    return paged.req_slot(k, b) * paged.ring_stride(k) + slot * HD


@traced
def _store_state(k, p: _Plan, b, slot, c, kv, sc, ok, wait):
    if ok:
        if const_expr(p.flag):  # noqa: SIM102 (traced)
            if wait:  # the step's first row has read its state: the slot may change now
                mb.poll([(k.mb[p.rd], b, 1)], k.tag)
        i = _state_at(k, p, b, slot) + c
        bo.buffer_store(kv.bitcast(fx.Int32), rsrc(k.args[p.kv_state]), i)
        bo.buffer_store(sc.bitcast(fx.Int32), rsrc(k.args[p.score_state]), i)


@traced
def _publish_read(k, p: _Plan, b, first):
    if first:
        mb.publish(k.mb[p.rd], b, fx.Int32(1), k.tag, k.tid == 0)


def _pool(k, t, p: _Plan):
    """Row t; thread c = channel c."""
    c = k.tid
    b, j = t // p.spec, t % p.spec
    pos = uniform(ld_i32(k.args[p.pos], b)) + j
    live = paged.row_live(
        k, t
    )  # (a padded or dead request's row writes none of ATOM's storage)
    # plain loads first (independent of the dots): gamma, the RoPE pair, the state partner
    gw = ld_i32(k.args[p.aux], c // 2)
    g_lo, g_hi = bf2_f32(gw)
    gamma = ((c & 1) == 1).select(g_hi, g_lo)
    epos = fx.max(pos + (1 - p.ratio), fx.Int32(0))
    pair = fx.max(c - (HD - ROPE), fx.Int32(0)) // 2
    cs = rope_cs(k, p.rope, epos, pair)
    iload = _index_loads(k, p, c, epos) if p.index_out else None
    emits = None  # ratio 1: every row
    if p.ratio == 1:
        y = bf16_round(_slices(k, [(p.kv_part, t)], c, p.kslices, p.poll)[0])
    else:
        pslot = (pos + (p.slots - 1)) % p.slots
        si = _state_at(k, p, b, pslot) + c
        st_kv = fx.Float32(
            bo.buffer_load(rsrc(k.args[p.kv_state]), si, vec_width=1, dtype=T.f32)
        )
        st_sc = fx.Float32(
            bo.buffer_load(rsrc(k.args[p.score_state]), si, vec_width=1, dtype=T.f32)
        )
        emits = (pos & 1) == 1
        # the partner (position P - 1): the step's previous row (its partials polled again), or the state (the
        # step's first row)
        groups = [(p.kv_part, t), (p.score_part, t)]
        if p.spec > 1:
            in_step = j > 0
            pr = in_step.select(t - 1, t)
            groups += [(p.kv_part, pr), (p.score_part, pr)]
        sums = _slices(k, groups, c, p.kslices, p.poll)
        kv, sc = sums[0], sums[1]
        kv0, s0 = st_kv, st_sc
        if p.spec > 1:
            kv0, s0 = in_step.select(sums[2], st_kv), in_step.select(sums[3], st_sc)
        # torch's softmax over the two slots (slot 0 = P - 1, slot 1 = P): exp(s - max) / sum, then the products
        m = fx.max(s0, sc)
        e0, e1 = fx.Float32(fmath.exp(s0 - m)), fx.Float32(fmath.exp(sc - m))
        tot = e0 + e1
        y = bf16_round(kv0 * (e0 / tot) + kv * (e1 / tot))
        # state: the step's last row of each slot writes it (after the first row has read its partner)
        slot = pos % p.slots
        last = j + p.slots >= p.spec
        wait = (j > 0) & (((j + 1) % p.slots) == 0)
        if p.flag:
            _publish_read(k, p, b, j == 0)
        _store_state(k, p, b, slot, c, kv, sc, last & live, wait)
    # RMSNorm over the 512 channels (the kv_norm map: lane i of wave 0 sums channels 8 i .. 8 i + 7 with dot2)
    fx.ptr_store(y, k.ldsf + c)
    gpu.barrier()
    lane = c % NORM_LANES
    ss = fx.Float32(0.0)
    for w in range(4):
        vals = [
            fx.Float32(fx.ptr_load(k.ldsf + (8 * lane + 2 * w + e))) for e in range(2)
        ]
        word = bf16_pair(vals[0], vals[1]).bitcast(fx.Int32)
        ss = dot2_bf16(word, word, ss)
    tot = wave_sum(
        ss
    )  # every wave computes it (wave 0's is the one), no second barrier
    rstd = hw_rsq(tot * (1.0 / HD) + fx.Float32(p.eps))
    lat = bf16_round(y * rstd * gamma)
    # RoPE on the last 64 channels (adjacent pairs), at the group's first position
    entry = _rope_lanes(lat, c, HD - ROPE, cs)
    if emits is not None:
        zero = fx.Float32(0.0)
        entry = emits.select(entry, zero)
        lat = emits.select(lat, zero)
    if const_expr(p.kv8):
        # ATOM's fp8 pool: the entry as its writer quantizes the bf16 row (``fused_compress``'s ``fp8_groups``), a
        # group of 32 channels a half wave; this launch's readers (``out``) take the values the row holds
        entry = _kv8_entry(k, p, b, pos, c, entry, live if emits is None else live & emits)
    # outputs: a pair a even thread
    nxt = xshfl(entry, 1)
    even = (c & 1) == 0
    _put_pair(k, k.mb[p.out], t * HD + c, entry, nxt, even)
    if const_expr(not p.kv8):
        # the entry into ATOM's pool, in place: the owner's main row of entry pos // ratio (``paged``)
        addr = paged.row_addr(k, p.cache, b, pos // p.ratio, p.ratio) + fx.Int64(c * 2)
        ok = even & live
        _store_entry(
            addr,
            bf16_pair(entry, nxt).bitcast(fx.Int32),
            ok if emits is None else ok & emits,
        )
    if p.index_out:
        _index_key(k, t, p, c, b, pos, iload, lat, emits)
    gpu.barrier()  # the next task reuses the LDS


# ------------------------------------------------------------------ plan


class Compressor(Block):
    """The KV compressor of one KV-source layer (module docstring). ``x``: ``attn_norm``'s BF16 output channel
    (5120); ``out`` (and optionally ``latent``): tagged BF16 channels of width 512."""

    def __init__(self, name: str, x: Channel, ratio: int, out: Channel, pos: str, rope: str, cache: str,
                 cache_len: int, latent: Channel | None = None, kv_state: str = "", score_state: str = "",
                 slots: int = 0, index_out: Channel | None = None, kslices: int = 4,
                 tiles: int = 1, poll_batch: int = 0, eps: float = 1e-20, qat: bool = True,
                 kv8: bool = False):  # fmt: skip
        if ratio not in (1, 2):
            raise ValueError(
                f"Compressor {name}: ratio 1 or 2 (DeepSeek-V4.1's), got {ratio}"
            )
        if x.fmt is not Fmt.BF16 or x.flagged or x.width != HIDDEN:
            raise ValueError(
                f"Compressor {name}: x is a tagged BF16 channel of width {HIDDEN}"
            )
        for ch in (out, latent):
            if ch is not None and (
                ch.fmt is not Fmt.BF16 or ch.flagged or ch.width != HD
            ):
                raise ValueError(
                    f"Compressor {name}: out / latent are tagged BF16 channels of width {HD}"
                )
        if index_out is not None and (
            index_out.fmt is not Fmt.BF16 or index_out.flagged or index_out.width != IHD
        ):
            raise ValueError(
                f"Compressor {name}: index_out is a tagged BF16 channel of width {IHD}"
            )
        slots = slots or ratio
        if ratio == 2 and (not kv_state or not score_state or slots < 2):
            raise ValueError(
                f"Compressor {name}: ratio 2 needs kv_state / score_state and 2 slots at least"
            )
        self.name, self.x, self.ratio, self.out, self.latent = (
            name,
            x,
            ratio,
            out,
            latent,
        )
        self.pos, self.rope, self.cache, self.cache_len = pos, rope, cache, cache_len
        self.kv_state, self.score_state, self.slots, self.eps = (
            kv_state,
            score_state,
            slots,
            eps,
        )
        self.wkv, self.wgate, self.aux = f"{name}.wkv", f"{name}.wgate", f"{name}.aux"
        self.index_out = index_out
        mk = lambda n, w: GatheredDot(f"{name}.{n}", x, w, Channel(f"{name}.{n}", HD, Fmt.F32),
                                      kslices=kslices, sum_stage=False, tiles=tiles)  # fmt: skip
        self.dots = [mk("kv", self.wkv)] + (
            [mk("score", self.wgate)] if ratio == 2 else []
        )
        self.kslices, self.tiles, self.poll_batch = kslices, tiles, poll_batch
        # the cache entry's FP4 QAT (the official model's); False: the rotated bf16 latent as is (an engine's bf16
        # cache without the QAT, ATOM's). A numerics choice, not a schedule knob
        self.qat = qat
        # the cache's rows: ATOM's fp8 pool (MXFP8, ``_Plan.kv8``) or its bf16 one; the deployment's
        self.kv8 = kv8

    def plan(self, sc: Scenario) -> _Plan:
        spec = sc.spec
        flag = self.ratio == 2 and any(
            (j + 1) % self.slots == 0 for j in range(max(1, spec - self.slots), spec)
        )
        return _Plan(self.name, sc.rows, spec, sc.batch, self.ratio, self.slots, self.kslices,
                     self.dots[0].part.name, self.dots[1].part.name if self.ratio == 2 else "", self.pos, self.rope,
                     self.aux, self.cache, self.cache_len, self.kv_state, self.score_state, self.out.name,
                     self.latent.name if self.latent else "", self.eps, self.poll_batch,
                     self.index_out.name if self.index_out else "", flag, self.qat, kv8=self.kv8)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        pname = f"{self.name}.pool"
        writes = list(self.out.regions(sc.rows, (pname,)))
        if self.index_out is not None:
            writes += self.index_out.regions(sc.rows, (pname,))
        if p.flag:
            writes.append(Region(p.rd, sc.batch, (pname,), exchange=True))
        args = (self.pos, self.rope, rope_sin(self.rope), self.aux, self.cache, *paged.REQ) + (
            (self.kv_state, self.score_state) if self.ratio == 2 else ())  # fmt: skip
        lds = HD + (HD // 2 + IHD if self.index_out is not None else 0)
        # the caches' entries are for later launches: this launch's readers take them from ``out`` / ``index_out``
        later = (self.cache,)
        pool = Stage(pname, sc.rows, partial(_pool, p=p), lds_words=lds, writes=tuple(writes), args=args,
                     key=("compressor_pool",) + (p,), later_args=later)  # fmt: skip
        return [st for d in self.dots for st in d.stages(sc)] + [pool]

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """``layer``: a dict of the layer's tensors as the checkpoint names them
        (``attn.compressor.{wkv,wgate,norm}.weight``), any device; returns bf16 tensors on that device.
        """
        bf = lambda n: layer[n].to(torch.bfloat16).contiguous()
        out = {self.wkv: bf("attn.compressor.wkv.weight")}
        if self.ratio == 2:
            out[self.wgate] = bf("attn.compressor.wgate.weight")
        aux = [bf("attn.compressor.norm.weight").flatten()]
        if self.index_out is not None:
            aux += [
                pack_wk(bf("attn.indexer.wk.weight")).flatten(),
                bf("attn.indexer.k_norm.weight").flatten(),
            ]
        out[self.aux] = torch.cat(aux)
        return out


def pack_wk(wk: torch.Tensor) -> torch.Tensor:
    """wk [128, 512] -> [16 steps, 512 lanes, 8] (lane c: row c // 4, K (c % 4) x 128 + 8 i .. + 8 at step i): every
    load instruction of the pool's GEMV reads 8 KB contiguous across the CTA."""
    return (
        wk.view(IHD, 4, 16, 8).permute(2, 0, 1, 3).reshape(16, THREADS, 8).contiguous()
    )


# ------------------------------------------------------------------ torch reference (the official arithmetic)
