# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The block indexer of block-sparse attention (Qwen3.8-Flash-Next's QSA), any
``IndexerGeometry`` with ``head_dim`` 128 (``check_geometry``).

Per step row (request ``b`` at position ``L = seq_lens[b] + s``) the indexer scores every
complete block of ``compress`` tokens against the row's index queries and publishes the
best ``top_blocks`` blocks:

    iqk [rows, (heads + 1) * head_dim] bf16 (raw index queries | raw index key)
        ->  sel [rows, top_blocks] int32: block ids in rank order, -1 past min(visible, top)

A row sees blocks ``0 .. cb - 1``, ``cb = (L + 1) // compress``. With ``cb <= top_blocks`` it
attends all of them in block order (no scores needed); otherwise the ``top_blocks`` best by
score, equal scores to the lower block, in rank order. The incomplete block's tokens
``compress * cb .. L`` are implied by the position (the consumer appends them).

The indexer also owns its two caches (Atom's, paged on the KV block table, ``kv_block`` tokens a
page): it appends each step row's raw key to ``index_raw`` [pages, kv_block, 1, head_dim] and the
key of a block a row completes (``4 b + 3 == L``) to ``index_cmp`` [pages, kv_block / compress, 1,
head_dim], as Atom's ``Qwen4ExpAttention._select_tokens`` does. Nothing in the launch reads
those rows back: the step's raw keys come from the channel.

Stages (one source for every (B, S)):

``<prefix>.score``: only rows with ``cb > top_blocks`` score (``heads <= 4``: a row's heads in
one lane). Span tasks per (request, row tile) score the request's blocks complete before the step
(``b < nc = seq_lens // compress``, visible to every row) in tiles of 16 blocks with f64 MFMAs
(4 x 4 x 4 for a tile of ``tile_rows x heads <= 4`` query vectors, else 16 x 16 x 4: at ``spec >=
4`` from ``WIDE_FROM`` cached blocks, chosen at run time), a tile's keys read once for its rows;
the products of bf16 values are exact in f64 and the sums are f64 sums, the contract's
arithmetic. Then one task a row keys and scores the blocks the step completes and appends the
raw key: waves ``< heads`` build index query head ``w`` (zero-centred RMSNorm, RoPE at ``L``) in
LDS, every lane holds its 8 dims of every head in f64; a **group** of 16 lanes scores one block
(lane ``j``: dims ``8 j .. 8 j + 7`` of the key, the head dots in f64, reduced over the group by
DPP).

The blocks the step completes (``nc <= b < cb``, a few) are scored on the row's task,
one group each, keyed from their raw rows (``index_raw`` below ``seq_lens``, the channel from
there): the pooled mean, the key norm, RoPE at ``compress b``. The row that completes the
block writes the key to ``index_cmp`` (whether or not it scores).

A score goes out as an f32 pair (``<prefix>.isc``, ``rows x max_blocks``).

``<prefix>.chunk`` (contexts past ``CHUNK`` blocks; rows x ``chunks`` tasks): the best
``top_blocks`` of each ``CHUNK`` blocks (``device.topk.select``) to
``<prefix>.cand``.

``<prefix>.final`` (rows tasks): the best ``top_blocks`` of the candidates (or of the scores,
up to ``CHUNK`` blocks) ranked (``topk.rank``) to ``sel``; rows with ``cb <= top_blocks``
publish ``0 .. cb - 1``.

Addressing: the caches are read and written at 64-bit addresses (any pool size); a negative
block-table entry is no page (a padded request: ``seq_lens`` 0, table row -1): never read or
written. Args: ``iqk`` (channel), ``index_raw``, ``index_cmp``, ``block_table`` [batch,
width] int32, ``seq_lens`` [batch] int32 (tokens cached before the step), ``<prefix>_w``
(``pack``: bf16 q norm | k norm, ``head_dim`` each) and ``rope`` ([positions, rotary_dim]
bf16: cos then sin of the ``rotary_dim / 2`` frequencies, ``rope_table``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.polls import mpoll_finish, mpoll_start
from atom.models.qwen38.mono.kernels.blocks.qknorm import rope_pos
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import topk
from atom.models.qwen38.mono.kernels.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    fresh,
    fresh_s,
    global_load,
    global_store,
    lane_gather,
    ld_i32,
    mfma_f64_4x4x4,
    mfma_f64_16x16x4,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.models.qwen38.mono.kernels.geometry import IndexerGeometry
from atom.models.qwen38.mono.kernels.program import Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X
from atom.mono.plan.trace import note

THREADS, WAVES, WAVE = MI355X.threads, MI355X.waves, MI355X.wave
# an i32 word offset past any region: a buffer store there is dropped
OOB_WORD = 0x20000000
DIM = 128  # head_dim the lane maps support
GROUP = 16  # lanes scoring one block (8 dims each)
KEYS = 16  # keys per thread in a select over scores
CHUNK = KEYS * THREADS  # blocks a chunk task (or the final task) selects from
# the final task's keys a thread for rows of up to SMALL_KEYS * THREADS blocks
SMALL_KEYS = 5
TILE = 16  # blocks an f64 MFMA tile scores
CHAINS = 4  # independent accumulation chains a tile (the MFMA's latency)
WIDE_FROM = CHUNK  # cached blocks from which a spec >= 4 request is scored on 4-row (16 x 16 x 4) tiles
MFMA_HEADS = 4  # a row's heads in one lane (16 x 16 x 4: D rows l // 16 + 4 e)


def check_geometry(geo: IndexerGeometry) -> None:
    """Raise unless the device code supports ``geo``."""
    half = geo.rotary_dim // 2
    ok = (
        geo.head_dim == DIM
        and 1 <= geo.heads <= MFMA_HEADS
        and half % 8 == 0
        and (half // 8) & (half // 8 - 1) == 0
        and geo.rotary_dim <= geo.head_dim
        and 1 <= geo.top_blocks <= THREADS
    )
    if not ok:
        raise ValueError(
            f"Indexer supports head_dim {DIM}, <= {MFMA_HEADS} heads, rotary_dim/16 a power of two; got {geo}"
        )


def rope_table(positions: int, geo: IndexerGeometry, device="cuda") -> torch.Tensor:
    """[positions, rotary_dim] bf16: cos then sin of the rotary_dim / 2 frequencies (built on
    the GPU, bit-exact with Atom's cache) in aiter's ``cos_sin_cache`` layout, so Atom's own
    cache (its first ``positions`` rows) can be bound in place instead."""
    from atom.models.qwen38.mono.kernels.rope import rope_cos_sin

    half = geo.rotary_dim // 2
    cos, sin = rope_cos_sin(
        torch.arange(positions), geo.rotary_dim, geo.rope_theta, device="cuda"
    )
    return torch.cat((cos[:, :half], sin[:, :half]), -1).contiguous().to(device)


@dataclass(frozen=True)
class _Cfg:
    """Every Python constant the emit functions close over (part of the build key)."""

    batch: int
    spec: int
    width: int  # block-table entries per request
    blk: int  # tokens a page of the caches (ATOM Qwen at --block-size 64: 64)
    heads: int
    compress: int
    top: int
    rot: int
    eps: float
    src: str  # the iqk channel
    src_width: int  # its row width (elements)
    col: int  # the index queries' first element in a row
    sel: str
    prefix: str
    spans: int = 1  # span tasks per (request, row tile)
    row_tile: int = 0  # rows a tile's query vectors cover (0: derived, ``tile_rows``)
    # the tiling of requests with at least ``WIDE_FROM`` cached blocks
    wide: _Cfg | None = None
    raw: str = "index_raw"
    cmp: str = "index_cmp"
    block_table: str = "block_table"
    seq_lens: str = "seq_lens"
    weights: str = "idx_w"
    rope: str = "rope"
    # int32 [batch]: the request's RoPE position minus its cache row (ATOM's MRoPE offset, 0 for text)
    rope_off: str = "rope_off"

    @property
    def rows(self) -> int:
        return self.batch * self.spec

    # ---- span tasks per (request, row tile), then one task a row

    @property
    def tile_rows(self) -> int:
        """Rows of a request a tile's query vectors cover (vector ``h * tile_rows + s``): the
        knob, else as many as fill a 4 x 4 x 4 (``4 // heads``, at most ``spec``)."""
        tr = min(self.row_tile or max(1, 4 // self.heads), self.spec)
        return tr if tr * self.heads <= 4 else min(tr, 16 // MFMA_HEADS)

    @property
    def shape(self) -> int:
        """The f64 MFMA: 4 (4 x 4 x 4, four groups: ``tile_rows x heads <= 4`` query vectors
        against 16 blocks) or 16 (16 x 16 x 4: ``tile_rows x heads <= 16``, vector ``m = 4 h +
        s``: ``tile_rows`` 4 then)."""
        return 4 if self.tile_rows * self.heads <= 4 else 16

    @property
    def vstride(self) -> int:
        """Query vector ``h * vstride + s`` (head ``h``, tile row ``s``): the MFMA's row order."""
        return self.tile_rows if self.shape == 4 else MFMA_HEADS

    @property
    def pages_per_tile(self) -> int:
        return TILE // self.per_page

    @property
    def tiles(self) -> int:
        return -(-self.spec // self.tile_rows)

    @property
    def span_total(self) -> int:
        return self.batch * self.tiles * self.spans

    @property
    def stage_tasks(self) -> int:
        """Tasks of the score stage."""
        return self.span_total + self.rows

    @property
    def vec_slots(self) -> int:
        """Query vectors a task holds in LDS."""
        mine = max(self.heads, 16 if self.shape == 16 else 4)
        return max(mine, self.wide.vec_slots) if self.wide is not None else mine

    @property
    def per_page(self) -> int:
        """Blocks a page."""
        return self.blk // self.compress

    @property
    def max_blocks(self) -> int:
        return self.width * self.per_page

    @property
    def chunks(self) -> int:
        """Chunk tasks per row (1: no chunk stage, the final task selects from the scores)."""
        return -(-self.max_blocks // self.chunk)

    @property
    def chunk(self) -> int:
        """Blocks a chunk task (or the final task, directly from the scores) selects from."""
        return KEYS * THREADS

    @property
    def direct_keys(self) -> int:
        """Keys a thread holds when the final task selects from the scores."""
        return min(KEYS, -(-self.max_blocks // THREADS))

    @property
    def cand_keys(self) -> int:
        """Candidates a thread holds in the final task (chunks x top, thread-major)."""
        return -(-self.chunks * self.top // THREADS)

    @property
    def index_bits(self) -> int:
        """Bits of a block id (the TopK tie-break index)."""
        return max(1, (self.max_blocks - 1).bit_length())

    @property
    def isc(self) -> str:
        return f"{self.prefix}.isc"

    @property
    def cand(self) -> str:
        return f"{self.prefix}.cand"

    @property
    def rope_words(self) -> int:
        """i32 words of a RoPE table row (cos half, then sin half)."""
        return self.rot // 2


# ------------------------------------------------------------------ small helpers


def _fresh_ctx(k):
    """The stage context with lane / tid / wave behind opaque moves (nothing derived from them
    is hoisted out of the task loop into registers held for the stage)."""
    return SimpleNamespace(
        **dict(vars(k), lane=fresh(k.lane), tid=fresh(k.tid), wave=fresh_s(k.wave))
    )


def _f64(x):
    return fx.Float32(x).to(fx.Float64)


def _fma64(a, b, c):
    return fx.Float64(
        llvm.call_intrinsic(
            T.f64, "llvm.fma.f64", [fx.Float64(v).ir_value() for v in (a, b, c)], [], []
        )
    )


def _xshfl64(v, off):
    w = fx.Vector.from_elements([v], fx.Float64).bitcast(fx.Int32)
    got = [xshfl(w[0], off), xshfl(w[1], off)]
    return fx.Vector.from_elements(got, fx.Int32).bitcast(fx.Float64)[0]


def _unpack(words):
    """bf16 pair words -> f32 values, low half first."""
    out = []
    for w in words:
        lo, hi = bf2_f32(w)
        out += [lo, hi]
    return out


def _rsqrt_rn(x):
    """``1 / sqrt(x)`` correctly rounded to f32 (through f64), as torch.rsqrt."""
    return (fx.Float64(1.0) / fx.Float64(fmath.sqrt(_f64(x)))).to(fx.Float32)


# The sum of squares of a head (DIM = 128) in torch's GPU order for
# ``x.float().square().mean(-1)``: 32 threads sum 4 consecutive elements each in order, then an
# xor butterfly over the threads, offsets 1, 2, 4, 8, 16 (adjacent first).
# (bf16 squares are exact in fp32: only the order matters.)


def _head_ss2(x, lane):
    """A head held 2 dims a lane (lane l: dims 2l, 2l + 1): lanes 2t, 2t + 1 are torch's thread
    t; every lane ends with the head's sum."""
    p = [xshfl(v, 1) for v in x]  # the other lane of the pair
    odd = (lane & 1) != 0
    a = [odd.select(p[e], x[e]) for e in range(2)]  # thread t's dims 4t, 4t + 1
    b = [odd.select(x[e], p[e]) for e in range(2)]  # its dims 4t + 2, 4t + 3
    ss = ((a[0] * a[0] + a[1] * a[1]) + b[0] * b[0]) + b[1] * b[1]
    for off in (2, 4, 8, 16, 32):
        ss = ss + xshfl(ss, off)
    return ss


def _head_ss8(x):
    """A head held 8 dims a lane by a 16-lane group (lane j: dims 8j ..): lane j is torch's
    threads 2j, 2j + 1 (the butterfly's first step within the lane); every lane of the group
    ends with the sum."""
    t = []
    for h in range(2):
        v = x[4 * h] * x[4 * h]
        for e in range(1, 4):
            v = v + x[4 * h + e] * x[4 * h + e]
        t.append(v)
    ss = t[0] + t[1]
    for off in (1, 2, 4, 8):
        ss = ss + xshfl(ss, off)
    return ss


def _norm(xs, ws, ss, eps):
    """``bf16((x * rsqrt(ss / DIM + eps)) * (1 + w))`` (fp32; ``ss`` the head's sum of squares)."""
    inv = _rsqrt_rn(ss * fx.Float32(1.0 / DIM) + fx.Float32(eps))
    return [bf16_round((x * inv) * (fx.Float32(1.0) + w)) for x, w in zip(xs, ws)]


def _rope(ys, cs, sn, partner_off, neg, active):
    """``bf16(bf16(y cos) + bf16(pair sin))``, ``pair`` the partner lane's value (negated where
    ``neg``); lanes not ``active`` keep ``y``."""
    out = []
    for y, c_, s_ in zip(ys, cs, sn):
        p = xshfl(y, partner_off)
        p = neg.select(-p, p)
        out.append(
            active.select(bf16_round(bf16_round(y * c_) + bf16_round(p * s_)), y)
        )
    return out


def _page_of(k, c: _Cfg, req, block):
    """Block-table entry of the page holding ``block`` (per lane)."""
    jj = fx.max(fx.min(block // c.per_page, c.width - 1), 0)
    return ld_i32(k.args[c.block_table], req * c.width + jj)


def _cmp_addr(k, c: _Cfg, page, block, j):
    """64-bit address of lane ``j``'s 16 bytes of ``block``'s compressed key."""
    row = fx.max(page, 0) * c.per_page + block % c.per_page
    return fx.Int64(k.args[c.cmp]) + fx.Int64(row) * (DIM * 2) + fx.Int64(j * 16)


def _raw_addr(k, c: _Cfg, page, token, word):
    """64-bit address of i32 ``word`` of ``token``'s raw key."""
    row = fx.max(page, 0) * c.blk + token % c.blk
    return fx.Int64(k.args[c.raw]) + fx.Int64(row) * (DIM * 2) + fx.Int64(word * 4)


# ------------------------------------------------------------------ scores


def _dots(q64, kf):
    """The group's head dots of one block (f64, every lane of the group): lane ``j`` holds
    key dims ``8 j ..`` (``kf``, f32) and ``q64[h]`` the same dims of each head."""
    out = []
    for qh in q64:
        acc = qh[0] * _f64(kf[0])
        for e in range(1, 8):
            acc = _fma64(qh[e], _f64(kf[e]), acc)
        out.append(acc)
    for off in (8, 4, 2, 1):
        out = [a + _xshfl64(a, off) for a in out]
    return out


def _score(dots, head_dim):
    """relu per head, the heads in order, one division by fl32(sqrt(head_dim)), -0 -> +0."""
    tot = None
    for d in dots:
        r = fx.max(d.to(fx.Float32), fx.Float32(0.0))
        tot = r if tot is None else tot + r
    return tot / fx.Float32(math.sqrt(head_dim)) + fx.Float32(0.0)


def _q_head(k, c: _Cfg, row, rp, head, slot):
    """(One wave) index query ``head`` of ``row`` normed and rotated at RoPE table row ``rp`` into LDS
    vector ``slot`` (f32, bf16 values; lane ``l`` dims ``2 l, 2 l + 1``)."""
    lane = k.lane
    e0 = row * c.src_width + c.col + head * DIM + 2 * lane
    x = chan.poll_bf16([(k.mb[c.src], e0, 2)], k.tag)[0]
    w = _unpack([ld_i32(k.args[c.weights], lane)])
    ss = _head_ss2(x, lane)
    y = _norm(x, w, ss, c.eps)
    half_words = c.rope_words // 2  # words of the cos half
    rw = fx.Int64(rp) * c.rope_words + fx.Int64(lane % half_words)
    cs = _unpack([fx.Int32(_ld_rope(k, c, rw))])
    sn = _unpack([fx.Int32(_ld_rope(k, c, rw + half_words))])
    # lane l holds dims 2l, 2l+1: the partner of dim d < rot / 2 is d + rot / 2 (lane l + rot / 4)
    y = _rope(y, cs, sn, c.rot // 4, lane < c.rot // 4, lane < c.rot // 2)
    for e in range_constexpr(2):
        fx.ptr_store(y[e], k.ldsf + (slot * DIM + 2 * lane + e))


@traced
def _q_prologue(k, c: _Cfg, row, rp):
    """Waves < heads: index query head ``wave`` normed and rotated at RoPE table row ``rp`` into
    LDS (f32, bf16 values); then a barrier."""
    if k.wave < c.heads:
        _q_head(k, c, row, rp, k.wave, k.wave)
    gpu.barrier()


def _rope_row(k, c: _Cfg, req, pos):
    """The RoPE table row of request ``req``'s cache row ``pos``: ATOM's position (``rope_off``)."""
    return rope_pos(pos + uniform(ld_i32(k.args[c.rope_off], req)), c.width * c.blk)


def _ld_rope(k, c: _Cfg, word):
    """One i32 word of the RoPE table at a 64-bit word index."""
    return global_load(fx.Int64(k.args[c.rope]) + word * 4, 1)


def _q_lanes(k, c: _Cfg):
    """This lane's 8 dims (``8 (lane % 16) ..``) of every head, f64, from LDS."""
    j = k.lane % GROUP
    out = []
    for h in range(c.heads):
        vals = []
        for half in range(2):
            v = fx.Vector(
                fx.ptr_load(
                    k.ldsf + (h * DIM + 8 * j + 4 * half),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            )
            vals += [_f64(v[e]) for e in range(4)]
        out.append(vals)
    return out


def _put_score(k, c: _Cfg, pair, score, ok):
    """Score pair ``pair`` := (score, tag) where ``ok``; elsewhere the store goes out of the
    region's buffer range and is dropped (no branch: a DPP reduction under a partial exec mask
    reads stale lanes)."""
    a = k.mb[c.isc]
    note(a.region, a.space, "put")
    off = ok.select(fx.Int32(pair) * 2, fx.Int32(OOB_WORD))
    words = fx.Vector.from_elements(
        [score.bitcast(fx.Int32), fx.Int32(k.tag)], fx.Int32
    )
    bo.buffer_store(
        words, rsrc(a.value, c.rows * c.max_blocks * 8), off, cache_modifier=CM_DEV
    )


@traced
def _scored_new(k, c: _Cfg, row, req, l0, pos, nc, cb):
    b, key = _new_key(k, c, req, l0, nc, cb)
    _q_prologue(k, c, row, _rope_row(k, c, req, pos))
    _new_out(k, c, row, req, pos, _q_lanes(k, c), b, key, cb)


def _raw_rows(k, c: _Cfg, req, l0, block, j):
    """The block's raw keys, lane ``j``'s 8 dims of each token (f32): tokens below ``l0`` from
    ``index_raw``, the step's from the channel."""
    page = _page_of(k, c, req, block)
    out = []
    for r in range(c.compress):
        tok = block * c.compress + r
        mem = fx.Vector(global_load(_raw_addr(k, c, page, tok, 4 * j), 4))
        srow = req * c.spec + fx.min(fx.max(tok - l0, 0), c.spec - 1)
        e0 = srow * c.src_width + c.col + c.heads * DIM + 8 * j
        got = chan.poll_bf16([(k.mb[c.src], e0, 4), (k.mb[c.src], e0 + 4, 4)], k.tag)
        step = got[0] + got[1]
        memf = _unpack([mem[e] for e in range(4)])
        out.append([(tok >= l0).select(step[e], memf[e]) for e in range(8)])
    return out


def _block_key(k, c: _Cfg, req, l0, block, j):
    """The key of ``block`` (lane ``j``'s 8 dims, bf16 values as f32): pooled mean of its raw
    rows in token order, the key norm, RoPE at its first token."""
    raw = _raw_rows(k, c, req, l0, block, j)
    pooled = []
    for e in range(8):
        tot = raw[0][e]
        for r in range(1, c.compress):
            tot = tot + raw[r][e]
        pooled.append(bf16_round(tot / fx.Float32(c.compress)))
    ss = _head_ss8(pooled)
    wk = _unpack(
        [
            fx.Int32(v)
            for v in fx.Vector(
                bo.buffer_load(
                    rsrc(k.args[c.weights]), DIM // 2 + 4 * j, vec_width=4, dtype=T.i32
                )
            )
        ]
    )
    y = _norm(pooled, wk, ss, c.eps)
    # lane j holds dims 8 j ..: rotated for j < rot / 8, partner lane j ^ (rot / 16)
    half = c.rot // 2
    lanes = half // 8
    rw = fx.Int64(_rope_row(k, c, req, block * c.compress)) * c.rope_words + fx.Int64(
        4 * (j % lanes)
    )
    cs = _unpack([fx.Int32(_ld_rope(k, c, rw + e)) for e in range(4)])
    sn = _unpack(
        [fx.Int32(_ld_rope(k, c, rw + c.rope_words // 2 + e)) for e in range(4)]
    )
    return _rope(y, cs, sn, lanes, (j & lanes) == 0, j < 2 * lanes)


def _new_key(k, c: _Cfg, req, l0, nc, cb):
    """(block, key) of this group's new block ``nc + g WAVES + wave`` of ``nc .. cb`` (the
    step completes them, one group each); the key of a clamped block where there is none.
    """
    g, j = k.lane // GROUP, k.lane % GROUP
    b = nc + g * WAVES + k.wave
    return b, _block_key(k, c, req, l0, fx.max(fx.min(b, cb - 1), 0), j)


@traced
def _new_out(k, c: _Cfg, row, req, pos, q64, b, key, cb):
    """A new block's score (``q64`` given: the row scores) and, by the row that completes it,
    its key to ``index_cmp``."""
    j = k.lane % GROUP
    if const_expr(q64 is not None):  # no branch (see _put_score)
        score = _score(_dots(q64, key), DIM)
        _put_score(k, c, row * c.max_blocks + b, score, (j == 0) & (b < cb))
    if b < cb:
        page = _page_of(k, c, req, b)
        if (b * c.compress + c.compress - 1 == pos) & (page >= 0):
            words = [mbx_pair(key[2 * e], key[2 * e + 1]) for e in range(4)]
            global_store(
                _cmp_addr(k, c, page, b, j), fx.Vector.from_elements(words, fx.Int32)
            )


@traced
def _unscored_last(k, c: _Cfg, row, req, l0, pos, nc, cb):
    """A row that does not score: only if it completes a block, that block's key (one group
    of wave 0 writes it)."""
    if (pos + 1) % c.compress == 0:
        g, j = k.lane // GROUP, k.lane % GROUP
        key = _block_key(k, c, req, l0, cb - 1, j)
        page = _page_of(k, c, req, cb - 1)
        if (k.wave == 0) & (g == 0) & (page >= 0):
            words = [mbx_pair(key[2 * e], key[2 * e + 1]) for e in range(4)]
            global_store(
                _cmp_addr(k, c, page, cb - 1, j),
                fx.Vector.from_elements(words, fx.Int32),
            )


def mbx_pair(a, b):
    """Two f32 (bf16 values) -> one packed bf16 pair word (Int32)."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


@traced
def _append_raw(k, c: _Cfg, row, req, pos):
    """The last wave: row ``pos``'s raw key (from the channel) to ``index_raw``."""
    if k.wave == WAVES - 1:
        e0 = row * c.src_width + c.col + c.heads * DIM + 2 * k.lane
        w = chan.poll_words([(k.mb[c.src], e0 // 2, 1)], k.tag)[0][0]
        page = _page_of(k, c, req, pos // c.compress)
        if page >= 0:
            global_store(_raw_addr(k, c, page, pos, k.lane), fx.Int32(w))


# ------------------------------------------------------------------ scores on the matrix cores
#
# A request's cached blocks (``0 .. nc - 1``, visible to every row of the step) are scored in
# tiles of 16 blocks against a row tile's query vectors with f64 MFMAs: products of bf16 values
# are exact in f64 and the sums are f64 sums, the contract's arithmetic (the head dots in f64,
# rounded once to fp32), so the scores are bit-identical to the 16-lane group dots of the new
# blocks (``_dots``). The tile's keys are read once for all the rows of the tile.
#
# Lane ``l`` loads 64 bytes of block ``l % 16``: 16-byte pieces ``i = 0 .. 3`` at dims ``32 i +
# 8 c ..`` (``c = l // 16``; a piece of 16 lanes is one 64-byte line). MFMA ``t = 0 .. 31``
# takes from lane ``l`` the key dim ``d(c, t) = 32 (t // 8) + 8 c + t % 8`` (its K row ``c``);
# the query operand holds the same dims (K order is free: the sum is exact). Query vector ``v
# = h * vstride + s`` (head ``h`` of the tile's row ``s``) is ``A`` row ``v``; 16 x 16 x 4
# (``vstride`` 4) then leaves row ``s = l // 16``'s heads ``h = e`` in lane ``l``, 4 x 4 x 4
# leaves vector ``l // 16`` of block ``l % 16`` in lane ``l`` (the heads gathered by
# ``ds_bpermute``).
#
# Tiles go to the request tile's ``spans x WAVES`` waves round-robin, wave-major (tile ``i`` to
# wave slot ``i % slots``): at short contexts the few tiles land one a CTA; at long ones every
# wave takes ``tiles / slots`` of them, two in flight.


def _tile_slot(k, c: _Cfg, tq):
    """This wave's slot among the request tile's ``spans x WAVES`` waves, wave-major (tile ``i``
    goes to slot ``i % slots``: a few tiles land on different CTAs, one wave each)."""
    return k.wave * c.spans + tq


def _tile_pages(k, c: _Cfg, req, slot, j0):
    """The page vector of tiles ``j0 ..`` of this wave's list (tile ``slot + j (spans x
    WAVES)``): lane ``l`` holds the block-table entry of page ``l % ppt`` of list tile ``j0 + l
    // ppt`` (one load a wave for ``WAVE / ppt`` tiles)."""
    ppt = c.pages_per_tile
    i = slot + (j0 + k.lane // ppt) * (c.spans * WAVES)
    e = fx.max(fx.min(i * (TILE // c.per_page) + k.lane % ppt, c.width - 1), 0)
    return ld_i32(k.args[c.block_table], req * c.width + e)


def _q_tile_issue(k, c: _Cfg, req, tile, l0):
    """The loads of the tile's query vectors (a half-wave a vector: job ``2 wave + lane // 32``
    = head ``j % heads`` of tile row ``j // heads``, lane ``t = lane % 32`` dims ``4 t .. 4 t +
    3``, torch's thread ``t`` of the norm's sum): norm weights, RoPE words and the first
    attempt of the query poll, in flight (a CU returns loads in order: the key loads issued
    after them do not delay them).
    Jobs past the tile's rows recompute a valid row into an unused slot (no branch)."""
    t = k.lane % 32
    j = 2 * k.wave + k.lane // 32
    jobs = c.tile_rows * c.heads
    jj = fx.min(j, jobs - 1)
    s, h = jj // c.heads, jj % c.heads
    srow = fx.min(tile * c.tile_rows + s, c.spec - 1)
    w = [ld_i32(k.args[c.weights], 2 * t + e) for e in range(2)]
    # rotated dims below rot: 4 t .. lies in frequencies (4 t) % half ..
    half = c.rot // 2
    rw = fx.Int64(_rope_row(k, c, req, l0 + srow)) * c.rope_words + fx.Int64(
        (4 * t) % half // 2
    )
    cs = [fx.Int32(_ld_rope(k, c, rw + e)) for e in range(2)]
    sn = [fx.Int32(_ld_rope(k, c, rw + c.rope_words // 2 + e)) for e in range(2)]
    row = req * c.spec + srow
    spec = [(k.mb[c.src], (row * c.src_width + c.col + h * DIM + 4 * t) // 2, 2, None)]
    return SimpleNamespace(
        slot=h * c.vstride + s, w=w, cs=cs, sn=sn, spec=spec, first=mpoll_start(spec)
    )


def _q_tile_finish(k, c: _Cfg, q):
    """Poll the half-wave's query head, norm (torch's sum order: thread ``t`` sums its 4
    squares in order, then the xor butterfly 1 .. 16), rotate, store to LDS slot ``q.slot``.
    """
    t = k.lane % 32
    x = _unpack(mpoll_finish(q.spec, q.first, k.tag)[0])
    ss = ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3]
    for off in (1, 2, 4, 8, 16):
        ss = ss + xshfl(ss, off)
    y = _norm(x, _unpack(q.w), ss, c.eps)
    # the partner of dim d < rot / 2 is d + rot / 2: lane t ^ (rot / 8)
    lanes = c.rot // 8
    y = _rope(y, _unpack(q.cs), _unpack(q.sn), lanes, t < lanes, t < 2 * lanes)
    for e in range(4):
        fx.ptr_store(y[e], k.ldsf + (q.slot * DIM + 4 * t + e))


def _a_operand(k, c: _Cfg, tile):
    """The query operand of the 32 MFMAs (f64, 0 for vectors the tile does not have)."""
    cc = k.lane // 16
    v = k.lane % 16 if c.shape == 16 else k.lane % 4
    h, s = v // c.vstride, v % c.vstride
    ok = (h < c.heads) & (s < c.tile_rows) & (tile * c.tile_rows + s < c.spec)
    out = []
    for i in range(4):
        for half in range(2):
            vec = fx.Vector(
                fx.ptr_load(
                    k.ldsf + (v * DIM + 32 * i + 8 * cc + 4 * half),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            )
            out += [_f64(ok.select(vec[e], fx.Float32(0.0))) for e in range(4)]
    return out


def _issue_tile(k, c: _Cfg, pg, jj, i, nc):
    """Tile ``i``'s key loads (in flight): lane ``l``'s four 16-byte pieces of block ``16 i + l
    % 16`` (clamped below ``nc``), its page from the page vector ``pg`` (list tile ``jj`` of it).
    """
    b = fx.max(fx.min(i * TILE + k.lane % 16, nc - 1), 0)
    p = fx.max(
        fx.min(b // c.per_page - i * (TILE // c.per_page), c.pages_per_tile - 1), 0
    )
    page = lane_gather(pg, jj * c.pages_per_tile + p)
    cc = k.lane // 16
    return [
        fx.Vector(global_load(_cmp_addr(k, c, page, b, 4 * e + cc), 4))
        for e in range(4)
    ]


def _score32(vals, head_dim):
    """``_score`` of head dots already rounded to fp32."""
    tot = None
    for v in vals:
        r = fx.max(v, fx.Float32(0.0))
        tot = r if tot is None else tot + r
    return tot / fx.Float32(math.sqrt(head_dim)) + fx.Float32(0.0)


def _consume_tile(k, c: _Cfg, req, tile, a, b0, keys, end):
    """The tile's dots (``CHAINS`` accumulation chains, then their f64 sum) and its scores,
    put with no branch (``_put_score``)."""
    if c.shape == 16:
        zero = fx.Vector.from_elements([fx.Float64(0.0)] * 4, fx.Float64)
    else:
        zero = fx.Float64(0.0)
    accs = [zero] * CHAINS
    for t in range(32):
        w = fx.Int32(keys[t // 8][(t % 8) // 2])
        kv = _f64(bf_lo(w) if t % 2 == 0 else bf_hi(w))
        j = t % CHAINS
        if c.shape == 16:
            accs[j] = mfma_f64_16x16x4(a[t], kv, accs[j])
        else:
            accs[j] = mfma_f64_4x4x4(a[t], kv, accs[j])
    n, s = k.lane % 16, k.lane // 16
    b = b0 + n
    if c.shape == 16:  # lane l: row s = l // 16, its heads in elements e = h
        dots = []
        for h in range(c.heads):
            d = accs[0][h]
            for j in range(1, CHAINS):
                d = d + accs[j][h]
            dots.append(d.to(fx.Float32))
    else:  # lane l: vector l // 16 of block l % 16; row s's heads from lanes 16 (h tile_rows + s) + n
        d = accs[0]
        for j in range(1, CHAINS):
            d = d + accs[j]
        mine = d.to(fx.Float32).bitcast(fx.Int32)
        dots = []
        for h in range(c.heads):
            src = fx.min(16 * (h * c.tile_rows + s) + n, WAVE - 1)
            dots.append(lane_gather(mine, src).bitcast(fx.Float32))
    srow = tile * c.tile_rows + s
    ok = (s < c.tile_rows) & (srow < c.spec) & (b < end)
    row = req * c.spec + fx.min(srow, c.spec - 1)
    _put_score(k, c, row * c.max_blocks + b, _score32(dots, DIM), ok)


@traced
def _tile_span(k, c: _Cfg, req, tile, tq, l0, nc):
    """This wave's tiles ``slot, slot + spans x WAVES, ..`` of the request's cached blocks:
    the page vector, the query loads and the first two tiles' key loads issued, then the query
    polls and vectors (every wave: the task's vectors); then the tiles, a page vector a
    ``WAVE / ppt`` of them (past that from ~1M tokens at B1S1)."""
    slot = _tile_slot(k, c, tq)
    tiles = (nc + TILE - 1) // TILE
    nw = c.spans * WAVES
    n = fx.max(tiles - slot + nw - 1, 0) // nw  # this wave's tiles
    tpv = WAVE // c.pages_per_tile
    pg = _tile_pages(k, c, req, slot, 0)
    q = _q_tile_issue(k, c, req, tile, l0)
    k0 = _issue_tile(k, c, pg, 0, slot, nc)
    k1 = _issue_tile(k, c, pg, 1, slot + nw, nc)
    _q_tile_finish(k, c, q)
    gpu.barrier()
    a = _a_operand(k, c, tile)
    _segment(k, c, req, tile, a, pg, slot, 0, fx.min(n, tpv), nc, k0, k1)
    for j0 in range(tpv, n, tpv):
        pg2 = _tile_pages(k, c, req, slot, j0)
        i0 = slot + j0 * nw
        ka = _issue_tile(k, c, pg2, 0, i0, nc)
        kb = _issue_tile(k, c, pg2, 1, i0 + nw, nc)
        _segment(k, c, req, tile, a, pg2, i0, 0, fx.min(n - j0, tpv), nc, ka, kb)


@traced
def _segment(k, c: _Cfg, req, tile, a, pg, i0, j0, n, nc, ka, kb):
    """List tiles ``j0 .. n`` (tile ``i0 + j (spans x WAVES)``, one page vector ``pg``) in
    pairs, double-buffered: ``ka`` / ``kb`` hold tiles ``j``, ``j + 1`` (issued); tile ``j +
    2``'s loads go into ``ka`` once tile ``j`` is computed, before tile ``j + 1`` is (no
    buffer rotates: a copy of a load in flight would wait for it)."""
    nw = c.spans * WAVES
    for j in range(j0, n, 2):
        i = i0 + j * nw
        _consume_tile(k, c, req, tile, a, i * TILE, ka, nc)
        if j + 2 < n:
            ka = _issue_tile(k, c, pg, j + 2, i + 2 * nw, nc)
        if j + 1 < n:
            _consume_tile(k, c, req, tile, a, (i + nw) * TILE, kb, nc)
        if j + 3 < n:
            kb = _issue_tile(k, c, pg, j + 3, i + 3 * nw, nc)


@traced
def _span_task(k, c: _Cfg, req, r, l0, nc):
    """Span task ``r`` of request ``req`` under the tiling of ``c`` (row tile ``r // spans``):
    it works when a row of its tile scores and its wave 0 has a tile (slot ``tq``)."""
    tile, tq = r // c.spans, r % c.spans
    last = fx.min((tile + 1) * c.tile_rows, c.spec) - 1  # the tile's last row
    if ((l0 + last + 1) // c.compress > c.top) & (tq * TILE < nc):
        _tile_span(k, c, req, tile, tq, l0, nc)


@traced
def _score_body_mfma(k, t, c: _Cfg):
    """Span tasks (request, row tile, span) first, then one task a row: the blocks the step
    completes (scored by 16-lane groups, ``_dots``), the raw key append."""
    if t < c.span_total:
        per_req = c.tiles * c.spans
        req, r = t // per_req, t % per_req
        l0 = uniform(ld_i32(k.args[c.seq_lens], req))
        nc = l0 // c.compress
        if const_expr(c.wide is not None):
            if nc >= WIDE_FROM:
                _span_task(k, c.wide, req, r, l0, nc)
            else:
                _span_task(k, c, req, r, l0, nc)
        else:
            _span_task(k, c, req, r, l0, nc)
        gpu.barrier()  # the next task's prologue rewrites the LDS queries
    else:
        if t < c.stage_tasks:
            row = t - c.span_total
            req = row // c.spec
            l0 = uniform(ld_i32(k.args[c.seq_lens], req))
            pos = l0 + row % c.spec
            cb = (pos + 1) // c.compress
            nc = l0 // c.compress
            if cb > c.top:
                if cb > nc:
                    _scored_new(k, c, row, req, l0, pos, nc, cb)
            else:
                _unscored_last(k, c, row, req, l0, pos, nc, cb)
            _append_raw(k, c, row, req, pos)
            gpu.barrier()


def _score_stage(k, t, c: _Cfg):
    _score_body_mfma(_fresh_ctx(k), t, c)


# ------------------------------------------------------------------ selection


def _poll_scores(k, c: _Cfg, row, base, cb, per):
    """This thread's ``per`` scores ``base + tid + THREADS e`` (strided: a wave's polls
    coalesce) as order keys, 0 from ``cb`` on."""
    first = base + k.tid
    specs = [
        (k.mb[c.isc], row * c.max_blocks + fx.min(first + THREADS * e, cb - 1))
        for e in range(per)
    ]
    got = chan.poll_f32(specs, k.tag)
    us, idxs = [], []
    for e in range(per):
        i = first + THREADS * e
        us.append((i < cb).select(topk.fold_f32(got[e]), fx.Int32(0)))
        idxs.append(fx.Int32(i))
    return us, idxs


def _chunk_stage(k, t, c: _Cfg):
    _chunk_body(_fresh_ctx(k), t, c)


@traced
def _chunk_body(k, t, c: _Cfg):
    if t < c.rows * c.chunks:
        row, ch = t // c.chunks, t % c.chunks
        pos = uniform(ld_i32(k.args[c.seq_lens], row // c.spec)) + row % c.spec
        cb = (pos + 1) // c.compress
        if (cb > c.chunk) & (ch * c.chunk < cb):
            base = ch * c.chunk
            us, idxs = _poll_scores(k, c, row, base, fx.min(cb, base + c.chunk), KEYS)
            n_out = topk.select(
                k.lds, k.tid, k.lane, k.wave, us, idxs, c.top, c.index_bits
            )
            u, idx = topk.slot(k.lds, k.tid, n_out, c.top, c.index_bits)
            if k.tid < c.top:
                chan.put_words(
                    k.mb[c.cand],
                    2 * ((row * c.chunks + ch) * c.top + k.tid),
                    [u, idx],
                    k.tag,
                )
            gpu.barrier()  # the slots are read before the next task's select clears the LDS


def _final_stage(k, t, c: _Cfg):
    _final_body(_fresh_ctx(k), t, c)


@traced
def _final_body(k, t, c: _Cfg):
    if t < c.rows:
        pos = uniform(ld_i32(k.args[c.seq_lens], t // c.spec)) + t % c.spec
        cb = (pos + 1) // c.compress
        if cb > c.top:
            idx, r = _ranked(k, c, t, cb)
            if r >= 0:
                chan.put_i32(k.mb[c.sel], t * c.top + r, idx, k.tag)
        else:
            if k.tid < c.top:
                chan.put_i32(
                    k.mb[c.sel],
                    t * c.top + k.tid,
                    (k.tid < cb).select(fx.Int32(k.tid), fx.Int32(-1)),
                    k.tag,
                )


@traced
def _ranked(k, c: _Cfg, row, cb):
    """(a block among the row's best ``top``, its rank) per thread (rows with more than
    ``top`` blocks; every rank 0 .. top - 1 is held by one thread)."""
    idx, r = fx.Int32(-1), fx.Int32(-1)
    if const_expr(c.chunks > 1):
        if cb > c.chunk:
            idx, r = _from_candidates(k, c, row, cb)
        else:
            idx, r = _from_scores_sized(k, c, row, cb)
    else:
        idx, r = _from_scores_sized(k, c, row, cb)
    return idx, r


@traced
def _from_scores_sized(k, c: _Cfg, row, cb):
    """The direct selection with as few keys a thread as the row's blocks allow."""
    idx, r = fx.Int32(-1), fx.Int32(-1)
    if const_expr(c.direct_keys > SMALL_KEYS):
        if cb <= SMALL_KEYS * THREADS:
            idx, r = _from_scores(k, c, row, cb, SMALL_KEYS)
        else:
            idx, r = _from_scores(k, c, row, cb, c.direct_keys)
    else:
        idx, r = _from_scores(k, c, row, cb, c.direct_keys)
    return idx, r


def _from_scores(k, c: _Cfg, row, cb, per):
    us, idxs = _poll_scores(k, c, row, 0, cb, per)
    n_out = topk.select(k.lds, k.tid, k.lane, k.wave, us, idxs, c.top, c.index_bits)
    _, idx, r = topk.rank(k.lds, k.tid, k.lane, k.wave, n_out, c.top, c.index_bits)
    return idx, r


def _from_candidates(k, c: _Cfg, row, cb):
    """The chunks' candidates (strided over (chunk, slot)); only the ``ceil(cb / CHUNK)``
    chunks the row has are polled."""
    have = (cb + c.chunk - 1) // c.chunk * c.top
    base = row * c.chunks * c.top
    first = k.tid
    specs = [
        (k.mb[c.cand], 2 * (base + fx.min(first + THREADS * e, have - 1)), 2)
        for e in range(c.cand_keys)
    ]
    got = chan.poll_words(specs, k.tag)
    us = [
        (first + THREADS * e < have).select(got[e][0], fx.Int32(0))
        for e in range(c.cand_keys)
    ]
    idxs = [got[e][1] for e in range(c.cand_keys)]
    n_out = topk.select(k.lds, k.tid, k.lane, k.wave, us, idxs, c.top, c.index_bits)
    _, idx, r = topk.rank(k.lds, k.tid, k.lane, k.wave, n_out, c.top, c.index_bits)
    return idx, r


# ------------------------------------------------------------------ the block


def default_spans(sc: Scenario, tiles: int) -> int:
    """Span tasks per (request, row tile): the grid less one task a row."""
    return max(1, (MI355X.cus - sc.rows) // (sc.batch * tiles))


class Indexer(Block):
    """The block indexer (see the module docstring). ``iqk``: the channel holding each row's raw
    index queries (``heads x head_dim``) then its raw index key, from element ``col`` of a row
    of ``iqk_width``; ``sel``: the output channel name (``Fmt.I32``, ``top_blocks`` wide);
    ``table_width``: block-table entries per request (any width; Atom passes ``max_model_len /
    kv_block``); ``kv_block``: tokens a page of the caches."""

    name = "indexer"

    def __init__(
        self,
        geo: IndexerGeometry,
        iqk: str = "iqk",
        iqk_width: int | None = None,
        col: int = 0,
        sel: str = "sel",
        table_width: int = 1024,
        prefix: str = "idx",
        index_raw: str = "index_raw",
        index_cmp: str = "index_cmp",
        block_table: str = "block_table",
        seq_lens: str = "seq_lens",
        weights: str | None = None,
        rope: str = "rope",
        rope_off: str = "rope_off",
        q_norm: str = "self_attn.indexer.q_layernorm",
        k_norm: str = "self_attn.indexer.k_layernorm",
        kv_block: int = 0,
    ):
        check_geometry(geo)
        self.geo = geo
        self.blk = kv_block  # tokens a page of the caches
        if not kv_block or self.blk % geo.compress:
            raise ValueError(
                f"Indexer: kv_block {kv_block} must be a positive multiple of compress {geo.compress}"
            )
        width = (geo.heads + 1) * geo.head_dim
        self.iqk = Channel(iqk, iqk_width or width, Fmt.BF16)
        if col + width > self.iqk.width or col % 8:
            raise ValueError(
                f"Indexer: iqk columns {col}..{col + width} outside a row of {self.iqk.width}"
            )
        self.col = col
        self.sel = Channel(sel, geo.top_blocks, Fmt.I32)
        self.table_width, self.prefix = table_width, prefix
        self.q_norm, self.k_norm = q_norm, k_norm
        self.names = {
            "raw": index_raw,
            "cmp": index_cmp,
            "block_table": block_table,
            "seq_lens": seq_lens,
            "weights": weights or f"{prefix}_w",
            "rope": rope,
            "rope_off": rope_off,
        }

    def cfg(self, sc: Scenario) -> _Cfg:
        g = self.geo
        c = _Cfg(
            batch=sc.batch,
            spec=sc.spec,
            width=self.table_width,
            blk=self.blk,
            heads=g.heads,
            compress=g.compress,
            top=g.top_blocks,
            rot=g.rotary_dim,
            eps=g.eps,
            src=self.iqk.name,
            src_width=self.iqk.width,
            col=self.col,
            sel=self.sel.name,
            prefix=self.prefix,
            **self.names,
        )
        c = replace(c, spans=default_spans(sc, c.tiles))
        rows4 = 16 // MFMA_HEADS  # the rows of a 16 x 16 x 4 tile
        if sc.spec >= rows4 and c.tile_rows < rows4:
            # long contexts (``WIDE_FROM`` cached blocks on): 4-row tiles (16 x 16 x 4), the keys read once for 4
            # rows; the same span tasks a request
            per_req = c.tiles * c.spans
            w = replace(c, row_tile=rows4)
            c = replace(c, wide=replace(w, spans=per_req // w.tiles))
        return c

    def stages(self, sc: Scenario) -> list[Stage]:
        c = self.cfg(sc)
        score, chunk, final = (
            f"{self.prefix}.{s}" for s in ("score", "chunk", "final")
        )
        lds_sel = topk.lds_words(c.top, c.index_bits)
        stages = [
            Stage(
                score,
                c.stage_tasks,
                partial(_score_stage, c=c),
                lds_words=c.vec_slots * DIM,
                # _put_score's own resource
                writes=(Region(c.isc, c.rows * c.max_blocks, (score,), single=True),),
                args=(
                    c.seq_lens,
                    c.block_table,
                    c.raw,
                    c.cmp,
                    c.weights,
                    c.rope,
                    c.rope_off,
                ),
                key=(c,),
            )
        ]
        if c.chunks > 1:
            stages.append(
                Stage(
                    chunk,
                    c.rows * c.chunks,
                    partial(_chunk_stage, c=c),
                    lds_words=lds_sel,
                    writes=(Region(c.cand, c.rows * c.chunks * c.top * 2, (chunk,)),),
                    args=(c.seq_lens,),
                    key=(c,),
                )
            )
        stages.append(
            Stage(
                final,
                c.rows,
                partial(_final_stage, c=c),
                lds_words=lds_sel,
                writes=self.sel.regions(c.rows, (final,)),
                args=(c.seq_lens,),
                key=(c,),
            )
        )
        return stages

    @property
    def max_positions(self) -> int:
        return self.table_width * self.blk

    def weight_specs(self) -> tuple:
        """The norm weights (bf16 q norm | k norm) and the RoPE table ([positions, rotary_dim]
        bf16, cos half then sin half: aiter's ``cos_sin_cache`` layout, so Atom's own cache can
        be bound in place)."""
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        g = self.geo
        return (
            WeightSpec(
                self.names["weights"], "bf16", (2 * g.head_dim,), (torch.bfloat16,)
            ),
            WeightSpec(
                self.names["rope"],
                "bf16",
                (self.max_positions, g.rotary_dim),
                (torch.bfloat16,),
            ),
        )

    def pack(self, layer) -> dict[str, torch.Tensor]:
        qn, kn = layer.retained[self.q_norm], layer.retained[self.k_norm]
        w = torch.cat(
            (qn.to(torch.bfloat16).flatten(), kn.to(torch.bfloat16).flatten())
        ).contiguous()
        return {
            self.names["weights"]: w,
            self.names["rope"]: rope_table(self.max_positions, self.geo, qn.device),
        }
