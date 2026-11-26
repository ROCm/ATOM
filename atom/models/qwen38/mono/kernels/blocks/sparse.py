# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``SparseAttnCore``: gated attention over each row's selected key list (block-sparse attention,
Qwen3.8-Flash-Next's QSA), any ``AttnGeometry`` with head_dim 256 and at most 16 query heads
per KV head, any ``IndexerGeometry`` (``check_geometry``).

    qn [rows, q_width] bf16 (q normed, rotated: ``QKNormRope``), kv [rows, 2 kv_width] bf16
    (the step rows' k | v), sel [rows, top_blocks] int32 (the indexer's blocks in rank order),
    the paged cache, proj's gate columns  ->  attn [rows, q_width] MXFP8 (the o_proj input;
    a BF16 ``out`` channel: bf16, the MTP drafter's)

The numerics: ``q' = bf16(q D^-0.5 log2 e)``, ``s_j = q' . k_j`` (fp32), **one global max**
``m`` per (row, head) over the whole list, ``e_j = exp2(s_j - m)``, ``out = bf16(sum_j bf16(e_j)
v_j / sum_j e_j)``, ``bf16(out bf16(sigmoid(gate)))``, MXFP8 (group 32). The probabilities are rounded
to bf16 against the global max exactly as the contract does (an online / blocked softmax
would round them against partial maxima, which o_proj's re-quantization amplifies); only
fp32 summation orders differ from torch's.

**The list** of row ``r`` (request ``b``, offset ``s``, position ``p = seq_lens[b] + s``):
``vis = (p + 1) // c`` complete blocks (``c`` = compress), ``nsel = min(vis, top_blocks)``;
list position ``j < c nsel`` is token ``c sel[r, j // c] + j % c``, then the incomplete block's
tokens ``c vis .. p``: ``ak = c nsel + (p + 1) - c vis`` keys in all. A token below
``seq_lens[b]`` is read from the paged cache (64-bit addresses); a step token
(``>= seq_lens[b]``, possibly anywhere in the list once a step completes a block) comes from
``kv`` (staged in LDS), never from the cache (no read-after-write within the launch).

Two stages, one source for every (B, S):

``<prefix>.attn``: task = (row, KV head, split ``i``); the list's 16-key tiles ``i, i +
splits, ...`` (at most ``m`` of them) are the task's. Splits without a tile (short lists)
return at once.

  1. Polls, issued with the length load and before any K / V load (a CU returns loads in
     order): ``sel`` for the task's tiles, the row's ``rep`` q heads, the request's step rows
     of KV head ``h`` (every polled pair is published each launch: no masks). Then the
     tokens' pages (block table), then K (wave ``w``: tile ``w``, addressed from its own
     ``sel`` entry) and V (wave ``w``: dims ``32 w ..`` of every tile) at per-lane 64-bit
     addresses; keys past the list, step tokens and padded requests read a zero row, never
     the pool.
  2. ``q'`` into LDS (rows past ``rep`` zero), the step rows into LDS; QK on ``mfma
     16x16x32`` (16 keys x 16 heads, K from registers patched with LDS for step tokens).
  3. The task's max per head goes to the exchange ``<prefix>.ml``; V (patched) is transposed
     into wave-private LDS meanwhile; then the group's (row, KV head) global max: every task
     polls every live split's maxima (one round trip).
  4. ``P = bf16(exp2(s - m))`` into LDS, the fp32 sums of ``e`` per head; P @ V on ``mfma
     16x16x16`` (wave ``w``: dims ``32 w ..``, every key of the task).
  5. The partial numerator (fp32, device-scope bulk stores, ``<prefix>.o``), drained, then
     the partial denominator as a tagged pair (``<prefix>.dl``): the merge's flag.

``<prefix>.merge`` (rows x q_heads x 256 / merge_dims tasks): polls the denominators of the
row's live splits and the gate, sums the partials in split order, divides, gates and
quantizes to MXFP8 (``out``).

The caches (``kv_block`` tokens a block-table entry): token-major bf16 ``k_cache`` / ``v_cache``
[pages, kv_block, kv_heads, d] (the drafter's), or with ``fp8`` ATOM Qwen's fp8 row layout: E4M3
``k_cache`` / ``v_cache`` [pages, kv_block, kv_heads, d] and fp32 ``k_scale`` / ``v_scale`` [pages,
kv_block, kv_heads].

Decomposition: ``splits`` (tasks per (row, KV head)) the power of two filling the grid up to 64, at
least 8 tiles' worth (``default_splits``: an exchange group never straddles a round of tasks; a task
holds at most 8 tiles), ``merge_dims`` (``default_merge_dims``); ``m`` (tiles a task holds) from the
indexer budget and ``splits``.

Padded requests (Atom's graph batches: ``seq_lens`` 0, block-table row -1) read no page;
their rows attend to their own step keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.polls import mpoll_finish, mpoll_start
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_pack4,
    fresh,
    fresh_s,
    global_load,
    hw_exp2,
    ld_i32,
    rsrc,
    traced,
    uniform,
    wave_sum,
    xred,
    xshfl,
)
from atom.models.qwen38.mono.kernels.geometry import AttnGeometry, IndexerGeometry
from atom.models.qwen38.mono.kernels.program import Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X
from atom.mono.plan.trace import note

D = 256  # Flash's head_dim (the lane mappings)
TILE = 16  # keys per tile (an MFMA's M)
WAVES = MI355X.waves
GRID = MI355X.cus
NEG = -1.0e30  # score of a masked key
LOG2E = 1.4426950408889634
FP8_MAX = 448.0
MAX_M = WAVES  # tiles a task holds at most (one per wave for QK; more spill SGPRs)


def check_geometry(geo: AttnGeometry, index: IndexerGeometry) -> None:
    if geo.head_dim != D or geo.rep > 16:
        raise ValueError(
            f"SparseAttnCore supports head_dim {D} and <= 16 q heads per KV head; got {geo}"
        )
    if index.compress < 1 or index.top_blocks < 1:
        raise ValueError(f"unsupported indexer geometry {index}")


def _pow2_floor(x: int) -> int:
    return 1 << (max(1, x).bit_length() - 1)


def _pow2_ceil(x: int) -> int:
    return 1 << max(0, x - 1).bit_length()


@dataclass(frozen=True)
class _Cfg:
    """Every Python constant of the two stages (their build key)."""

    batch: int
    spec: int
    qh: int
    kvh: int
    compress: int
    topk: int
    width: int  # block-table entries per request
    splits: int
    merge_dims: int
    proj_w: int
    col: (
        int  # first column of q_proj in proj (the gate of head j is at col + 2 D j + D)
    )
    kv_block: int  # tokens a block-table entry
    qn: str
    kv: str
    sel: str
    proj: str
    out: str
    k_cache: str
    v_cache: str
    block_table: str
    seq_lens: str
    zero: str
    ml: str
    o: str
    dl: str
    d: int = D  # head_dim
    # the caches: ATOM Qwen's fp8 row layout (module docstring), else token-major bf16
    fp8: bool = False
    k_scale: str = ""
    v_scale: str = ""
    # attn as a BF16 channel (a bf16 o_proj's input: the MTP drafter's), not MXFP8
    out_bf16: bool = False

    @property
    def page(self) -> int:  # tokens a block-table entry
        return self.kv_block

    @property
    def rows(self) -> int:
        return self.batch * self.spec

    @property
    def qd(self) -> int:  # dims of a q / k / v row a lane holds
        return self.d // 64

    @property
    def qp(self) -> int:  # q' row pitch in LDS words
        return self.d // 2 + 4

    @property
    def vd(self) -> int:  # V dims a wave owns (its P @ V columns)
        return self.d // WAVES

    @property
    def vg(self) -> int:  # V dims a lane loads a tile (a quarter of the wave's)
        return self.vd // 4

    @property
    def rep(self) -> int:
        return self.qh // self.kvh

    @property
    def groups(self) -> int:
        return self.rows * self.kvh

    @property
    def tasks(self) -> int:
        return self.groups * self.splits

    @property
    def max_keys(self) -> int:
        return self.topk * self.compress + self.compress - 1

    @property
    def tiles_max(self) -> int:
        return -(-self.max_keys // TILE)

    @property
    def m(self) -> int:  # tiles a task holds
        return -(-self.tiles_max // self.splits)

    @property
    def tw(self) -> int:  # QK tiles a wave holds
        return -(-self.m // WAVES)

    @property
    def kv_width(self) -> int:
        return self.kvh * self.d

    @property
    def out_w(self) -> int:
        return self.qh * self.d

    # ---- attention LDS (int32 words)
    @property
    def q_off(self) -> int:
        return 0

    @property
    def kn_off(self) -> int:
        return self.q_off + 16 * self.qp

    @property
    def vn_off(self) -> int:
        return self.kn_off + self.spec * self.d // 2

    @property
    def pp(self) -> int:  # P row pitch (words): 16 m bf16 + pad
        return 8 * self.m + 2

    @property
    def p_off(self) -> int:
        return self.vn_off + self.spec * self.d // 2

    @property
    def vp(self) -> int:  # transposed-V row pitch (words)
        return 8 * self.m + 2

    @property
    def vt_off(self) -> int:
        return -(-(self.p_off + 16 * self.pp) // 4) * 4

    @property
    def vt_words(self) -> int:  # one wave's transposed V: vd dims x 16 m keys
        return self.vd * self.vp

    @property
    def wmax_off(self) -> int:
        return self.vt_off + WAVES * self.vt_words

    @property
    def wden_off(self) -> int:
        return self.wmax_off + WAVES * 16

    @property
    def gmax_off(self) -> int:
        return self.wden_off + WAVES * 16

    @property
    def dummy_off(self) -> int:
        return self.gmax_off + 16

    @property
    def lds_words(self) -> int:
        return self.dummy_off + 4 * 64

    # ---- merge
    @property
    def mchunks(self) -> int:
        return self.d // self.merge_dims

    @property
    def merge_tasks(self) -> int:
        return self.rows * self.qh * self.mchunks


# ------------------------------------------------------------------ helpers


def _v2f():
    return fx.Vector.make_type(2, fx.Float32)


def _v4f():
    return fx.Vector.make_type(4, fx.Float32)


def _lds_bf8(ldsf, word):
    return fx.ptr_load(ldsf + word, result_type=_v4f()).bitcast(fx.BFloat16)


def _lds_bf4(ldsf, word):
    return fx.ptr_load(ldsf + word, result_type=_v2f()).bitcast(fx.BFloat16)


def _lds_i4(ldsf, word):
    v = fx.Vector(fx.ptr_load(ldsf + word, result_type=_v4f()).bitcast(fx.Int32))
    return [v[e] for e in range(4)]


def _mfma16(a, b, c):
    """16x16x32 bf16: a[i = lane % 16][k = 8 (lane / 16) ..], b[k][j = lane % 16]."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def _mfma16k16(a, b, c):
    """16x16x16 bf16: a[i = lane % 16][k = 4 (lane / 16) ..], b[k][j = lane % 16]."""
    i16 = [x.bitcast(fx.Int16) for x in (a, b)]
    return fx.Vector(
        rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [i16[0], i16[1], c])
    )


def _lds_put(ldsf, word, value, cond, dummy):
    """LDS store at ``word`` where ``cond``, else at the lane's ``dummy`` words (no branch)."""
    fx.ptr_store(value, ldsf + cond.select(fx.Int32(word), fx.Int32(dummy)))


def _words_f32(words):
    return fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32).bitcast(
        fx.Float32
    )


def _f32_words(words):
    """f32-typed words (``bf16_pair``'s) as one LDS store's value."""
    return words[0] if len(words) == 1 else fx.Vector.from_elements(words, fx.Float32)


def _add(a, b):
    return a + b


def _fresh_ctx(k):
    """lane / tid / wave behind opaque moves: derived math stays in the task."""
    return SimpleNamespace(
        **dict(vars(k), lane=fresh(k.lane), tid=fresh(k.tid), wave=fresh_s(k.wave))
    )


def list_len(c: _Cfg, n, s):
    """(vis, nsel, ak, ntiles) of the row at offset ``s`` of a request with ``n`` cached tokens (module
    docstring)."""
    p1 = n + s + 1
    vis = p1 // c.compress
    nsel = fx.min(vis, c.topk)
    ak = c.compress * nsel + p1 - c.compress * vis
    return vis, nsel, ak, (ak + TILE - 1) // TILE


# ------------------------------------------------------------------ attention stage


def _attn(k, t, c: _Cfg):
    _attn_body(_fresh_ctx(k), t, c)


@traced
def _attn_body(k, t, c: _Cfg):
    r = t // c.splits // c.kvh
    n_raw = ld_i32(k.args[c.seq_lens], r // c.spec)
    # every poll's first attempt goes out with the length load (validity needs the length)
    sp = _specs(k, c, t)
    firsts = [mpoll_start(sp[0]), mpoll_start(sp[1]), mpoll_start(sp[3])]
    n = uniform(n_raw)
    _, _, _, ntiles = list_len(c, n, r % c.spec)
    if t % c.splits < ntiles:
        _task(k, c, t, n, firsts)


def _specs(k, c: _Cfg, t):
    """The task's input polls: (sel specs, q specs, q rows, kv specs, kv rows). Every polled
    pair is published each launch (``sel``: all ``top_blocks`` entries a row, -1 past the
    list), so indices are clamped and no spec is masked (a mask is an SGPR pair held across
    the spin)."""
    grp, i = t // c.splits, t % c.splits
    r, h = grp // c.kvh, grp % c.kvh
    b = r // c.spec
    lane, wave = k.lane, k.wave
    kk = lane % 16
    cc = c.compress
    sel_specs = []
    for j in _key_positions(c, i, wave, kk):
        sel_specs.append(
            (k.mb[c.sel], r * c.topk + fx.min(j // cc, c.topk - 1), 1, None)
        )
    q_specs, q_rows = [], []
    for qq in range(2):
        jq = wave + WAVES * qq
        e = r * c.out_w + (h * c.rep + fx.min(jq, c.rep - 1)) * c.d + c.qd * lane
        q_specs.append((k.mb[c.qn], e // 2, c.qd // 2, None))
        q_rows.append((jq, jq < c.rep))
    kv_specs, kv_rows = [], []
    for it in range(-(-2 * c.spec // WAVES)):
        rh = wave + WAVES * it
        # None: every wave has a row
        ok = None if WAVES * (it + 1) <= 2 * c.spec else rh < 2 * c.spec
        rhc = fx.min(rh, 2 * c.spec - 1)
        sq, kind = rhc // 2, rhc % 2
        e = (
            (b * c.spec + sq) * 2 * c.kv_width
            + kind * c.kv_width
            + h * c.d
            + c.qd * lane
        )
        kv_specs.append((k.mb[c.kv], e // 2, c.qd // 2, None))
        kv_rows.append((sq, kind, ok))
    return sel_specs, q_specs, q_rows, kv_specs, kv_rows


def _key_positions(c: _Cfg, i, wave, kk):
    """List positions of this lane's key (``lane % 16`` of a tile) in the task's V tiles
    (0 .. m - 1) and then in the wave's QK tiles (``wave + 8 q``, clamped)."""
    out = [16 * (i + c.splits * mt) + kk for mt in range(c.m)]
    for q in range(c.tw):
        out.append(16 * (i + c.splits * fx.min(wave + WAVES * q, c.m - 1)) + kk)
    return out


def _task(k, c: _Cfg, t, n, firsts):
    """One (row, KV head, split) task with at least one tile (straight-line code)."""
    grp = t // c.splits
    i = t % c.splits
    r, h = grp // c.kvh, grp % c.kvh
    b, s = r // c.spec, r % c.spec
    vis, nsel, ak, ntiles = list_len(c, n, s)
    lane, wave, tid = k.lane, k.wave, k.tid
    g4, kk = lane // 16, lane % 16
    cc = c.compress
    ldsf = k.ldsf
    dummy = c.dummy_off + 4 * lane
    js = _key_positions(c, i, wave, kk)
    sel_specs, q_specs, q_rows, kv_specs, kv_rows = _specs(k, c, t)
    sel_first, q_first, kv_first = firsts

    # ---- tokens and pages (the V tiles', then the wave's QK tiles')
    sel_got = mpoll_finish(sel_specs, sel_first, k.tag)
    toks, nows, addrs = [], [], []
    zero = fx.Int64(k.args[c.zero])
    row_bytes = c.kvh * c.d * 2
    for e, j in enumerate(js):
        inb = j < cc * nsel
        tok = inb.select(cc * sel_got[e][0] + j % cc, cc * vis + (j - cc * nsel))
        live = j < ak
        cached = live & (tok >= 0) & (tok < n)
        bt = b * c.width + fx.max(fx.min(tok // c.page, c.width - 1), 0)
        page = ld_i32(k.args[c.block_table], bt)
        toks.append(tok)
        nows.append(live & (tok >= n))
        addrs.append((cached, page, tok))
    kaddr, vaddr = [], []
    if const_expr(c.fp8):
        kraw, vraw = _fp8row_loads(k, c, h, addrs, zero)
    for e, (cached, page, tok) in enumerate(addrs if not c.fp8 else []):
        ok = cached & (page >= 0)
        off = (
            fx.Int64(fx.max(page, 0)) * c.page + fx.Int64(tok % c.page)
        ) * row_bytes + fx.Int64(h * c.d * 2)
        if e < c.m:
            vaddr.append(ok.select(fx.Int64(k.args[c.v_cache]) + off, zero))
        else:
            kaddr.append(ok.select(fx.Int64(k.args[c.k_cache]) + off, zero))

    # ---- K (the wave's tiles) and V (the wave's dims of every tile)
    if const_expr(not c.fp8):
        kraw, vraw = _bf16_loads(k, c, kaddr, vaddr, zero)

    # ---- 2. q' and the step rows into LDS
    scale = 1.0 / math.sqrt(c.d)
    q_got = mpoll_finish(q_specs, q_first, k.tag)
    for qq in range(2):
        jq, ok = q_rows[qq]
        vals = []
        for w in q_got[qq]:
            vals += list(bf2_f32(w))
        vals = [
            ok.select(bf16_round((x * scale) * LOG2E), fx.Float32(0.0)) for x in vals
        ]
        words = [bf16_pair(vals[2 * e], vals[2 * e + 1]) for e in range(c.qd // 2)]
        fx.ptr_store(
            _f32_words(words), ldsf + (c.q_off + jq * c.qp + (c.qd // 2) * lane)
        )
    kv_got = mpoll_finish(kv_specs, kv_first, k.tag)
    for it in range(len(kv_specs)):
        sq, kind, ok = kv_rows[it]
        word = (
            (kind == 0).select(fx.Int32(c.kn_off), fx.Int32(c.vn_off))
            + sq * (c.d // 2)
            + (c.qd // 2) * lane
        )
        if ok is None:
            fx.ptr_store(_words_f32(kv_got[it]), ldsf + word)
        else:
            _lds_put(ldsf, word, _words_f32(kv_got[it]), ok, dummy)
    gpu.barrier()

    # ---- QK: scores [key 4 g4 + v][head kk] per wave tile
    scores, tiles = [], []
    for q in range(c.tw):
        mtw = wave + WAVES * q
        tile_ok = (mtw < c.m).select(fx.Int32(1), fx.Int32(0))
        tok_w = toks[c.m + q]
        now_w = (mtw < c.m) & nows[c.m + q]
        krow = fx.max(fx.min(tok_w - n, c.spec - 1), 0)
        kops, qops = [], []
        for st in range(c.d // 32):
            new = _lds_i4(ldsf, c.kn_off + krow * (c.d // 2) + 16 * st + 4 * g4)
            raw = kraw[q][st]
            kw = [now_w.select(new[e], raw[e]) for e in range(4)]
            kops.append(fx.Vector.from_elements(kw, fx.Int32).bitcast(fx.BFloat16))
            qops.append(_lds_bf8(ldsf, c.q_off + kk * c.qp + 16 * st + 4 * g4))
        a2 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for st in range(c.d // 32):
            a2[st % 2] = _mfma16(kops[st], qops[st], a2[st % 2])
        acc = a2[0] + a2[1]
        tpos = 16 * (i + c.splits * mtw) + 4 * g4
        sv = []
        for v in range(4):
            ok = (tile_ok != 0) & (tpos + v < ak) & (kk < c.rep)
            x = acc[v]
            sv.append(ok.select(x, fx.Float32(NEG)))
        scores.append(sv)
        tiles.append((mtw, tile_ok))
    mloc = fx.Float32(NEG)
    for sv in scores:
        mloc = fx.max(mloc, fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])))
    mloc = xred(xred(mloc, 16, fx.max), 32, fx.max)
    fx.ptr_store(mloc, ldsf + (c.wmax_off + wave * 16 + kk))  # the 4 lane groups agree
    gpu.barrier()

    # ---- 3. the task's max per head to the exchange, then the group's global max
    gbase = (r * c.kvh + h) * c.rep
    _put_max(k, c, gbase, i, tid)
    # V (patched) transposed into the wave's LDS while the maxima travel (wave-private: no barrier)
    ldsb = fx.recast_iter(fx.BFloat16, k.lds)
    vt = c.vt_off + wave * c.vt_words
    for mt in range(c.m):
        vrow = fx.max(fx.min(toks[mt] - n, c.spec - 1), 0)
        new = _lds_i4(
            ldsf, c.vn_off + vrow * (c.d // 2) + c.vd * wave // 2 + (c.vg // 2) * g4
        )
        nw = nows[mt]
        words = [nw.select(new[e], vraw[mt][e]) for e in range(c.vg // 2)]
        for e in range(c.vg):
            lo, hi = bf2_f32(words[e // 2])
            val = (lo if e % 2 == 0 else hi).to(fx.BFloat16)
            fx.ptr_store(val, ldsb + (2 * (vt + (c.vg * g4 + e) * c.vp) + 16 * mt + kk))
    nvalid = fx.min(ntiles, c.splits)
    _xchg_all(k, c, gbase, nvalid, dummy)
    gpu.barrier()

    # ---- 4. P = bf16(exp2(s - m)) into LDS, the sums of e
    gm = fx.ptr_load(ldsf + (c.gmax_off + kk))
    den = fx.Float32(0.0)
    for q in range(c.tw):
        mtw, tile_ok = tiles[q]
        ev = [
            hw_exp2(x - gm) for x in scores[q]
        ]  # masked scores: NEG - m underflows to 0
        den = den + ((ev[0] + ev[1]) + (ev[2] + ev[3]))
        words = [bf16_pair(ev[0], ev[1]), bf16_pair(ev[2], ev[3])]
        pword = c.p_off + kk * c.pp + 8 * fx.min(mtw, c.m - 1) + 2 * g4
        _lds_put(
            ldsf, pword, fx.Vector.from_elements(words, fx.Float32), tile_ok != 0, dummy
        )
    den = xred(xred(den, 16, _add), 32, _add)
    fx.ptr_store(den, ldsf + (c.wden_off + wave * 16 + kk))
    gpu.barrier()

    # ---- P @ V: [dim 4 g4 + v][head kk] for the wave's 2 x 16 dims
    accs = []
    for dt in range(c.vd // 16):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for mt in range(c.m):
            a = _lds_bf4(ldsf, vt + (16 * dt + kk) * c.vp + 8 * mt + 2 * g4)
            bm = _lds_bf4(ldsf, c.p_off + kk * c.pp + 8 * mt + 2 * g4)
            acc = _mfma16k16(a, bm, acc)
        accs.append(acc)

    # ---- 5. the partial numerator (dense fp32 at device scope), drained before the denominator
    # pair that flags it
    o_addr = k.mb[c.o]
    note(o_addr.region, o_addr.space, "put")
    ro = rsrc(o_addr.value, c.rows * c.qh * c.splits * c.d * 4)
    prow = (r * c.qh + h * c.rep + fx.min(kk, c.rep - 1)) * c.splits + i
    for dt in range(c.vd // 16):
        d0 = c.vd * wave + 16 * dt + 4 * g4
        # padded heads: out of range
        off = (kk < c.rep).select(prow * c.d + d0, fx.Int32(0x20000000))
        bo.buffer_store(accs[dt], ro, off, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    _put_den(k, c, r, h, i, tid)
    gpu.barrier()  # the task's last LDS reads before any wave moves on


def _bf16_loads(k, c: _Cfg, kaddr, vaddr, zero):
    """The bf16 caches: K (the wave's tiles, the lane's key, dims ``32 st + 8 g4 ..``) and V (the wave's dims of
    every tile)."""
    wave, g4 = k.wave, k.lane // 16
    kraw = []
    for q in range(c.tw):
        ka = (wave + WAVES * q < c.m).select(kaddr[q], zero)
        kraw.append(
            [
                fx.Vector(global_load(ka + fx.Int64((32 * st + 8 * g4) * 2), 4))
                for st in range(c.d // 32)
            ]
        )
    vraw = [
        fx.Vector(
            global_load(vaddr[mt] + fx.Int64((c.vd * wave + c.vg * g4) * 2), c.vg // 2)
        )
        for mt in range(c.m)
    ]
    return kraw, vraw


def _e4m3(word, byte):
    """Byte ``byte`` of an i32 word as E4M3 -> f32."""
    return fx.Float32(rocdl.cvt_f32_fp8(T.f32, fx.Int32(word).ir_value(), byte))


def _fp8_pairs(vals):
    return [
        bf16_pair(vals[2 * e], vals[2 * e + 1]).bitcast(fx.Int32)
        for e in range(len(vals) // 2)
    ]


def _scaled_bf16(words, s):
    """i32 words of 4 E4M3 bytes each -> their bf16 values times ``s`` as i32 words (two a word): the f32 product
    rounded to bf16 (RNE), right for any per-token scale (ATOM's insert writes amax / 448; exact for a power of two).
    Not ``v_cvt_scalef32_pk_bf16_fp8``: its scale is an exponent (E8M0), the mantissa of a non-power-of-two scale is
    dropped."""
    return fx.Vector.from_elements(
        _fp8_pairs([_e4m3(w, b) * s for w in words for b in range(4)]), fx.Int32
    )


def _fp8row_loads(k, c: _Cfg, h, addrs, zero):
    """ATOM Qwen's fp8 caches (row layout: ``k_cache`` / ``v_cache`` [pages, block, kv_heads, d] E4M3, ``k_scale`` /
    ``v_scale`` [pages, block, kv_heads] fp32): the operands ``_bf16_loads`` gives (bf16 words), each code times its
    (token, head)'s scale in f32 rounded to bf16 (``_scaled_bf16``). A key past the list (or no page) reads the zero
    row: 0 codes, scale 0."""
    wave, g4 = k.wave, k.lane // 16
    rows = []
    for cached, page, tok in addrs:
        ok = cached & (page >= 0)
        rows.append(
            (
                ok,
                (fx.Int64(fx.max(page, 0)) * c.page + fx.Int64(tok % c.page)) * c.kvh
                + fx.Int64(h),
            )
        )

    def scale(arg, ok, row):
        a = ok.select(fx.Int64(k.args[arg]) + row * 4, zero)
        return fx.Int32(global_load(a, 1)).bitcast(fx.Float32)

    kraw = []
    for q in range(c.tw):
        ok, row = rows[c.m + q]
        ok = ok & (wave + WAVES * q < c.m)
        ks = scale(c.k_scale, ok, row)
        ka = ok.select(fx.Int64(k.args[c.k_cache]) + row * c.d, zero)
        words = []
        for st in range(c.d // 32):
            v = fx.Vector(
                global_load(ka + ok.select(fx.Int64(32 * st + 8 * g4), fx.Int64(0)), 2)
            )
            words.append(_scaled_bf16([v[0], v[1]], ks))
        kraw.append(words)
    vraw = []
    nw = c.vg // 4  # words of the lane's vg codes
    for mt in range(c.m):
        ok, row = rows[mt]
        vs = scale(c.v_scale, ok, row)
        va = ok.select(fx.Int64(k.args[c.v_cache]) + row * c.d, zero)
        v = fx.Vector(
            global_load(
                va + ok.select(fx.Int64(c.vd * wave + c.vg * g4), fx.Int64(0)), nw
            )
        )
        vraw.append(_scaled_bf16([v[e] for e in range(nw)], vs))
    return kraw, vraw


@traced
def _put_max(k, c: _Cfg, gbase, i, tid):
    if tid < c.rep:
        m = fx.ptr_load(k.ldsf + (c.wmax_off + fx.min(tid, 15)))
        for w in range_constexpr(1, WAVES):
            m = fx.max(m, fx.ptr_load(k.ldsf + (c.wmax_off + w * 16 + fx.min(tid, 15))))
        mb.put(k.mb[c.ml], (gbase + tid) * c.splits + i, m, k.tag)


def _xchg_all(k, c: _Cfg, gbase, nvalid, dummy):
    """Every task polls the group's maxima, two splits a 16-byte load spread over all threads
    (the group's [rep][splits] pairs are contiguous), then reduces each head over its
    ``splits / 2`` consecutive threads. One split: no exchange (its own
    maxima are the group's)."""
    tid, ldsf = k.tid, k.ldsf
    total = c.rep * c.splits
    g = c.splits // 2  # threads per head
    specs, meta = [], []
    for q in range(-(-total // (2 * MI355X.threads))):
        f = 2 * (tid + MI355X.threads * q)
        live = f < total
        fc = fx.min(f, total - 2)
        sp = fc % c.splits
        specs.append(
            (
                k.mb[c.ml],
                gbase * c.splits + fc,
                2,
                (live & (sp < nvalid), live & (sp + 1 < nvalid)),
            )
        )
        meta.append((fc // c.splits, live))
    got = mpoll_finish(specs, mpoll_start(specs), k.tag)
    for q in range(len(specs)):
        ok0, ok1 = specs[q][3]
        head, live = meta[q]
        m = fx.max(
            ok0.select(got[q][0].bitcast(fx.Float32), fx.Float32(NEG)),
            ok1.select(got[q][1].bitcast(fx.Float32), fx.Float32(NEG)),
        )
        for off in [o for o in (1, 2, 4, 8, 16, 32) if o < g]:
            m = xred(m, off, fx.max)
        _lds_put(ldsf, c.gmax_off + fx.min(head, 15), m, live & (tid % g == 0), dummy)
    _lds_put(
        ldsf,
        c.gmax_off + fx.min(tid, 15),
        fx.Float32(0.0),
        (tid >= c.rep) & (tid < 16),
        dummy,
    )


@traced
def _put_den(k, c: _Cfg, r, h, i, tid):
    if tid < c.rep:
        d = fx.ptr_load(k.ldsf + (c.wden_off + fx.min(tid, 15)))
        for w in range_constexpr(1, WAVES):
            d = d + fx.ptr_load(k.ldsf + (c.wden_off + w * 16 + fx.min(tid, 15)))
        mb.put(k.mb[c.dl], (r * c.qh + h * c.rep + tid) * c.splits + i, d, k.tag)


# ------------------------------------------------------------------ merge stage


def _merge(k, t, c: _Cfg):
    _merge_body(_fresh_ctx(k), t, c)


@traced
def _merge_body(k, t, c: _Cfg):
    md = c.merge_dims
    r = t // (c.qh * c.mchunks)
    hq = (t // c.mchunks) % c.qh
    dc = t % c.mchunks
    lane, wave, tid = k.lane, k.wave, k.tid
    ldsf = k.ldsf
    n = uniform(ld_i32(k.args[c.seq_lens], r // c.spec))
    _, _, _, ntiles = list_len(c, n, r % c.spec)
    nvalid = fx.min(ntiles, c.splits)
    base = (r * c.qh + hq) * c.splits
    dg = fx.min(tid, md - 1)
    ge = r * c.proj_w + c.col + hq * 2 * c.d + c.d + dc * md + dg
    nq = -(-c.splits // 64)
    specs = [
        (
            k.mb[c.dl],
            base + fx.min(lane + 64 * q, c.splits - 1),
            1,
            lane + 64 * q < nvalid,
        )
        for q in range_constexpr(nq)
    ]
    specs.append((k.mb[c.proj], ge // 2, 1, None))
    per = md // 4  # lanes per split (4 dims each)
    spw = 64 // per  # splits per wave per load
    sub, gl = lane % per, lane // per
    slots = [
        (q * WAVES + wave) * spw + gl
        for q in range_constexpr(-(-c.splits // (spw * WAVES)))
    ]
    got = mpoll_finish(specs, mpoll_start(specs), k.tag)
    den = fx.Float32(0.0)
    for q in range_constexpr(nq):
        den = den + specs[q][3].select(got[q][0].bitcast(fx.Float32), fx.Float32(0.0))
    tot_den = wave_sum(den)
    gw = got[nq][0]
    gate = ((dg % 2) == 0).select(bf2_f32(gw)[0], bf2_f32(gw)[1])
    a4 = [fx.Float32(0.0) for _ in range_constexpr(4)]
    o_addr = k.mb[c.o]
    note(o_addr.region, o_addr.space, "poll")
    ro = rsrc(o_addr.value)
    for split in slots:
        ok = split < nvalid
        v = fx.Vector(
            bo.buffer_load(
                ro,
                (base + fx.min(split, c.splits - 1)) * c.d + dc * md + 4 * sub,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        a4 = [
            a4[e] + ok.select(v[e].bitcast(fx.Float32), fx.Float32(0.0))
            for e in range_constexpr(4)
        ]
    for off in [o for o in (1, 2, 4, 8, 16, 32) if o >= per]:
        a4 = [xred(a4[e], off, _add) for e in range_constexpr(4)]
    for e in range_constexpr(4):
        fx.ptr_store(a4[e], ldsf + (wave * md + 4 * sub + e))  # the lane groups agree
    gpu.barrier()
    if tid < md:
        tot = fx.ptr_load(ldsf + tid)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(ldsf + (w * md + tid))
        _gate_mxfp8(
            k, c, tot / tot_den, gate, r * c.out_w + hq * c.d + dc * md + tid, tid
        )
    gpu.barrier()


@traced
def _gate_mxfp8(k, c: _Cfg, attn, gate, e0, tid):
    """``bf16(bf16(attn) bf16(sigmoid(gate)))`` quantized to MXFP8 (group 32 = 32 threads), or (``out_bf16``)
    put as bf16."""
    a = bf16_round(attn)
    sig = bf16_round(fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(-gate)))
    x = bf16_round(a * sig)
    amax = fx.Float32(fmath.absf(x))
    for o in range_constexpr(5):
        amax = xred(amax, 16 >> o, fx.max)
    bits = amax.bitcast(fx.Int32)
    ex = (bits >> 23) & 0xFF
    code = ex - 8 + ((bits & 0x7FFFFF) > 0x600000).select(fx.Int32(1), fx.Int32(0))
    code = fx.max(fx.min(code, 254), 1)
    code = (amax == 0.0).select(fx.Int32(127), code)
    if const_expr(
        c.out_bf16
    ):  # the gated row itself, two elements a put (even threads)
        x1 = xshfl(x, 1)
        if tid % 2 == 0:
            chan.put_bf16(k.mb[c.out], e0, [x, x1], k.tag)
        return
    scale_inv = ((254 - code) << 23).bitcast(fx.Float32)
    y = fx.max(fx.min(x * scale_inv, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))
    y1 = xshfl(y, 1)
    y2 = xshfl(y, 2)
    y3 = xshfl(y1, 2)
    word = fp8_pack4(y, y1, y2, y3)
    if tid % 4 == 0:
        chan.put_mxfp8_word(k.mb[c.out + ".q"], e0 // 4, word, k.tag)
    if tid % 32 == 0:
        chan.put_mxfp8_scale(k.mb[c.out + ".s"], e0 // 32, code, k.tag)


# ------------------------------------------------------------------ the block


def default_splits(sc: Scenario, geo: AttnGeometry, index: IndexerGeometry) -> int:
    """Tasks per (row, KV head): a power of two filling the grid up to 64 (128 splits make the
    exchange's polls contend), at least enough that a task holds at most
    ``MAX_M`` tiles (beyond the grid, tasks run in rounds)."""
    groups = sc.rows * geo.kv_heads
    keys = index.top_blocks * index.compress + index.compress - 1
    tiles = -(-keys // TILE)
    fill = min(64, _pow2_floor(max(1, GRID // groups)))
    return max(fill, _pow2_ceil(-(-tiles // MAX_M)))


def default_merge_dims(sc: Scenario, q_heads: int, d: int = D) -> int:
    for md in (64, 128, 256):
        if md <= d and sc.rows * q_heads * (d // md) <= GRID:
            return md
    return d


class SparseAttnCore(Block):
    """qn, kv, sel (+ proj's gate) -> attn (MXFP8 or BF16 [rows, q_width]).

    ``geo`` / ``index``: the attention and indexer geometries (``check_geometry``);
    ``proj_width`` / ``col``: the proj channel's row width and q_proj's column in it (the
    gate is read from there). Args (plain memory): ``k_cache``, ``v_cache`` (module docstring; with
    ``fp8`` also ``k_scale`` / ``v_scale``), ``block_table`` int32 [batch, table_width], ``seq_lens`` int32
    [batch] (cached tokens before the step; a padded request: 0 over a row of -1) and ``<prefix>_zero``
    (``pack``: a zero row read for keys past the list). ``fp8_row`` must equal ``fp8`` (the row layout is the
    only fp8 one); ``kv_block``: tokens a block-table entry."""

    name = "sparse_attn"

    def __init__(
        self,
        geo: AttnGeometry,
        index: IndexerGeometry,
        qn: str = "qn",
        kv: str = "kv",
        sel: str = "sel",
        proj: str = "proj",
        out: str | Channel = "attn",
        proj_width: int | None = None,
        col: int = 0,
        table_width: int = 1024,
        prefix: str = "sa",
        k_cache: str = "k_cache",
        v_cache: str = "v_cache",
        block_table: str = "block_table",
        seq_lens: str = "seq_lens",
        zero: str | None = None,
        fp8: bool = False,
        k_scale: str = "k_scale",
        v_scale: str = "v_scale",
        kv_block: int = 0,
        fp8_row: bool = False,
    ):
        check_geometry(geo, index)
        if fp8_row != fp8 or not kv_block:
            raise ValueError(f"SparseAttnCore: fp8 caches are the row layout (fp8_row) and kv_block is given; got "
                             f"fp8={fp8}, fp8_row={fp8_row}, kv_block={kv_block}")  # fmt: skip
        self.kv_block, self.fp8 = kv_block, fp8
        self.fp8_names = {"k_scale": k_scale, "v_scale": v_scale} if fp8 else {}
        self.geo, self.index, self.prefix = geo, index, prefix
        self.table_width = table_width
        self.proj_width = proj_width or geo.proj_width
        self.col = col
        self.names = {
            "qn": qn,
            "kv": kv,
            "sel": sel,
            "proj": proj,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "seq_lens": seq_lens,
            "zero": zero or f"{prefix}_zero",
        }
        self.out_ch = (
            out if isinstance(out, Channel) else Channel(out, geo.q_width, Fmt.MXFP8)
        )
        self.out = self.out_ch.name
        self.sel_ch = Channel(sel, index.top_blocks, Fmt.I32)

    def cfg(self, sc: Scenario) -> _Cfg:
        c = _Cfg(
            batch=sc.batch,
            spec=sc.spec,
            qh=self.geo.q_heads,
            kvh=self.geo.kv_heads,
            compress=self.index.compress,
            topk=self.index.top_blocks,
            width=self.table_width,
            splits=default_splits(sc, self.geo, self.index),
            merge_dims=default_merge_dims(sc, self.geo.q_heads, self.geo.head_dim),
            proj_w=self.proj_width,
            col=self.col,
            kv_block=self.kv_block,
            out=self.out,
            out_bf16=self.out_ch.fmt is Fmt.BF16,
            ml=f"{self.prefix}.ml",
            o=f"{self.prefix}.o",
            dl=f"{self.prefix}.dl",
            **self.names,
            d=self.geo.head_dim,
            fp8=self.fp8,
            **self.fp8_names,
        )
        if c.m > MAX_M:
            raise ValueError(
                f"{c}: {c.m} tiles a task (at most {MAX_M}: one per wave); use more splits"
            )
        if c.tasks > GRID and GRID % c.splits:
            raise ValueError(
                f"{c}: an exchange group would straddle a round (splits must divide {GRID})"
            )
        if c.lds_words * 4 > MI355X.lds_bytes:
            raise ValueError(f"{c}: LDS {c.lds_words * 4} B; use more splits")
        if c.splits < 2:
            raise ValueError(
                f"{c}: one split (the global-max exchange needs at least two)"
            )
        return c

    def stages(self, sc: Scenario) -> list[Stage]:
        c = self.cfg(sc)
        attn, merge = f"{self.prefix}.attn", f"{self.prefix}.merge"
        n = self.names
        return [
            Stage(
                attn,
                c.tasks,
                partial(_attn, c=c),
                lds_words=c.lds_words,
                writes=(
                    Region(c.ml, c.groups * c.rep * c.splits, (attn,), exchange=True),
                    # bulk, own resource
                    Region(
                        c.o, sc.rows * c.qh * c.splits * c.d // 2, (attn,), single=True
                    ),
                    Region(c.dl, sc.rows * c.qh * c.splits, (attn,)),
                ),
                args=(
                    n["seq_lens"],
                    n["block_table"],
                    n["k_cache"],
                    n["v_cache"],
                    n["zero"],
                )
                + tuple(self.fp8_names.values()),
                key=(c,),
            ),
            Stage(
                merge,
                c.merge_tasks,
                partial(_merge, c=c),
                lds_words=WAVES * c.merge_dims,
                writes=self.out_ch.regions(sc.rows, (merge,)),
                args=(n["seq_lens"],),
                key=(c,),
            ),
        ]

    def pack(self, layer=None) -> dict[str, torch.Tensor]:
        return {
            self.names["zero"]: torch.zeros(
                self.geo.head_dim, dtype=torch.bfloat16, device="cuda"
            )
        }

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        return (
            WeightSpec(
                self.names["zero"], "bf16", (self.geo.head_dim,), (torch.bfloat16,)
            ),
        )
