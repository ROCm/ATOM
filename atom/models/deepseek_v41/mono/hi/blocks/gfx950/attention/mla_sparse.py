# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's sparse MLA decode of a compressed layer: the sliding window plus the top-512 compressed keys
(checkpoint ``inference/model.py`` ``Attention.forward`` with ``compress_ratio`` > 0, ``sparse_attn`` as the
reference). A consumer layer's attention core; the window part is ``MLAWindowAttn``'s.

The reference walks the key list (the window's 128 positions, oldest first, then the 512 selected compressed
positions, -1 padded) in chunks of 64 with a running max: per chunk c, m_c = max(m_{c-1}, the chunk's max score),
P_c = exp(s - m_c) (bf16 for the numerator, fp32 for the denominator), num_c = num_{c-1} corr_c + P_c V_c,
den_c = den_{c-1} corr_c + sum(P_c), corr_c = exp(m_{c-1} - m_c); o = bf16(num / (den + exp(sink - m))). Ten chunks of
640 KB of keys a row is too much for one CU, so the chunks run as parallel tasks and only the recurrence is serial:

- ``<name>`` (a task per (row, 16-head tile)): the window's two chunks, ``MLAWindowAttn``'s code up to its P V,
  then num_1 [16 heads][512], m_1 and den_1 published (``<name>.num``, ``<name>.w``). Puts the ring-read flag
  ``<name>.done`` (``RingWrite``) as ``MLAWindowAttn`` does.
- ``<name>.c`` (a task per (row, tile, compressed chunk)): the chunk's 64 keys gathered from the plain compressed
  cache by the selection (both from earlier launches: issued at task start, before the q poll), its scores and their
  max per head published (``<name>.mx``); then the prefix max m_c from m_1 and the earlier chunks' maxima (max is
  exact: the same m_c as the serial walk), P_c, the chunk's sum(P_c) (``<name>.ls``) and P_c V_c (``<name>.t``).
  An all-masked chunk gives max -inf, P 0, corr 1: the serial walk's bits too.
- ``<name>.m`` (a task per (row, tile, 64-dim slice)): the recurrence in chunk order (the reference's mul and add,
  not fused), the sink, the division, bf16, the inverse RoPE on the last 64 dims and the QAT (MXFP8 group 32) into
  ``out``.

Arithmetic: each chunk's scores and P V are ``MLAWindowAttn``'s MFMA chains, the recurrence the reference's
operations in its order, so the result is ``MLAWindowAttn``'s bits extended to 10 chunks.

``smerge`` (the merge takes the raw scores): the chain above is three hand-offs in series after
the scores (m_1 and the sibling maxima to every chunk, then 64 KB of P V partials a task to the merge), ~1.1-1.4 µs
round trips under load each, plus the puts. With ``smerge`` the window task (or, ``wsplit``, a task per window chunk)
and the chunk tasks stop after their scores and put them (8 KB a chunk, ``<name>.s``); a merge task per (row, tile,
``mheads`` heads, 64 dims) lands V^T of its slice for all 640 keys while they come (the ring, ``kvn``, the compressed
cache by the selection, a source's new entries), then polls its heads' scores of all ten chunks, runs the softmax
with the producers' thread mapping and operations (each chunk's max exact, its prefix max the running max), P V with
the producers' MFMA operands, the recurrence and the output: one hand-off after the scores. The same bits.
Optionally the merge's window V comes from an exact copy the caller keeps beside the bf16 ring: ``ring8`` the ring's
MXFP8 copy.

Args: ``pos``, ``ring``, ``rope`` (the layer's table: a compressed layer's YaRN one), ``sink`` as ``MLAWindowAttn``;
``sel`` int32 [batch x spec, 512] the step rows' compressed positions (raw: no window offset; -1 pads), ``ckv`` the
compressed cache: the owner's main rows in ATOM's pool, read in place (``device.gfx950.paged``: a selected position's
row through the request's block table; the request words and the page geometry, ``paged.REQ``, beside it). Both are
written by earlier launches (the KV / index source's): a consumer layer. Not yet: a source layer's own step entries
and selection from its Compressor / Indexer channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ....device.gfx950 import buffer_ops as bo
from ....device.gfx950 import channels as chan
from ....device.gfx950 import gemv, paged
from ....device.gfx950 import mailbox as mb
from ....device.gfx950.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    butterfly,
    global_load,
    lds_read_tr16,
    mfma_bf16,
    rope_cs,
    rope_sin,
    rsrc,
    traced,
    uniform,
    xred,
    xshfl,
)
from ....plan.program import Region, Scenario, Stage
from ....plan.trace import note
from ....target import MI355X
from ...base import Block, Channel, Fmt
from . import mla_window as mw
from .mla_window import (
    CH,
    HD,
    HEADS,
    PITCH,
    ROPE,
    TILE,
    WINDOW,
    _APlan,
    _done,
    _key_land,
    _lds_bf8,
    _own_keys,
    _pv_all,
    _q_tile,
    _qat_word,
    _ring_issue,
    _row_pos,
    _scores_all,
    _softmax_all,
)

THREADS, WAVES = MI355X.threads, MI355X.waves
SEL = 512  # the selected compressed positions a row
NCH = SEL // CH  # compressed chunks
CPITCH = CH + 4  # LDS words a chunk score row (f32)
CPPITCH = CH // 2 + 4  # LDS words a chunk P row (bf16 pairs)
SLICE = 64  # dims a merge task
# ``smerge`` (the merge takes the raw scores): the window's two chunks and the compressed ones, NCT in all. A merge
# task (row, its ``mheads`` heads, 64 dims) holds V^T of its slice (a dim's 640 keys as bf16 pairs a row: the MFMA's B
# operand is 4 consecutive words), P of its heads (16 rows: the MFMA's A), per-head maxima and sums, then t
NCT = 2 + NCH
VTP = NCT * CH // 2 + 4  # LDS words a V^T row (one dim, 640 keys)
PP = NCT * CH // 2 + 4  # LDS words a P row (one head, 640 keys)
OP = SLICE + 4  # LDS words a t row (f32: a chunk's head over the slice)


@dataclass(frozen=True)
class _SPlan:
    name: str
    rows: int
    spec: int
    q: str
    kvn: str
    ring: str
    pos: str
    rope: str
    sink: str
    sel: str
    ckv: str
    cache_len: int
    out: str
    heads: int = HEADS
    sel_ch: bool = (
        False  # sel is a tagged F32 channel (int32 bits) of this launch (an index source's Indexer)
    )
    new: str = (
        ""  # a KV source: this launch's entries from this tagged BF16 channel (the Compressor's, row-indexed)
    )
    ratio: int = (
        1  # ... entries from base // ratio (row j emits (base + j) // ratio at (base + j + 1) % ratio == 0)
    )
    ring_slots: int = WINDOW  # the window ring's slots (MLAWindowAttn's ``ring_slots``)
    ring_first: bool = (
        False  # the window task lands the ring before its q poll (MLAWindowAttn's ``ring_first``)
    )
    new_early: bool = (
        False  # a KV source: every entry the step emits polled into LDS at chunk task start
    )
    dsplit: int = (
        1  # window / chunk tasks a (row, tile[, chunk]) by P V dims (MLAWindowAttn's ``dsplit``)
    )
    smerge: bool = (
        False  # the window / chunk tasks put their raw scores; the merge runs the softmax and the P V
    )
    mheads: int = TILE  # smerge: heads a merge task (1, 2, 4, 8 or 16 of a tile)
    wsplit: bool = (
        False  # smerge: the window's two chunks as two tasks (64 keys each, as a compressed chunk's)
    )
    ckv4: bool = (
        False  # smerge: the merge's compressed V from the cache's FP4 copy (``<ckv>.q4`` / ``.s4``: exact)
    )
    ring8: bool = (
        False  # smerge: the merge's window V from the ring's MXFP8 copy (``<ring>.q8`` / ``.s8``: exact)
    )
    cratio: int = (
        1  # the compressed cache's ratio (its rows a page in ATOM's pool, ``paged``)
    )
    # the chunk / merge tasks' index math as unsigned shifts and one modulo (keys, rows and slots are never negative;
    # a window key's position is at least -(WINDOW - 1)) and the merge's V^T pairs packed by v_perm_b32: fewer VALU
    # instructions, the same addresses and bits
    lean: bool = False
    # the merge's V as loaded (``vtr``): key-major rows [640 keys][32 words] (one 16 B LDS store a lane and load, the
    # 4-word pieces of a row XOR-swizzled by key bits 1 and 3: conflict-free stores and reads), the P V's B operand by
    # gfx950's LDS transpose reads (``lds_read_tr16``: the same operand values, so the same bits) instead of V^T built
    # by lane-pair exchanges and 4 B stores
    vtr: bool = False
    # chunk tasks without this launch's entries (no ``new``): the gathered keys stored into LDS before the q poll (they
    # land long before q), so only the scores follow q; the same bits
    kfirst: bool = False
    # the output's inverse RoPE quantized from f32 (``MLAWindowAttn(unrope_f32=)``): no bf16 rounding of the 64 dims
    unrope_f32: bool = False

    @property
    def tiles(
        self,
    ):  # 16-head tiles a row, a ragged last one included (``MLAWindowAttn``'s ``_APlan.tiles``)
        return -(-self.heads // TILE)

    def head(self, ht, *hh):
        """The live head a tile's head (the sum of ``hh``, added in order: a 16-head program's arithmetic as before)
        computes (a ragged tile's dead heads: the last live head's)."""
        h = ht * TILE
        for x in hh:
            h = h + x
        return fx.min(h, fx.Int32(self.heads - 1)) if self.heads % TILE else h

    def win(
        self,
    ) -> _APlan:  # the window part's plan (MLAWindowAttn's LDS layout and names)
        return _APlan(self.name, self.rows, self.spec, self.q, self.kvn, self.ring, self.pos, self.rope, self.sink,
                      self.out, f"{self.name}.done", self.heads, ring_slots=self.ring_slots,
                      ring_first=self.ring_first, dsplit=self.dsplit, unrope_f32=self.unrope_f32)  # fmt: skip

    # chunk task LDS (words): q tile at 0, the chunk's keys, S, P, the keys' valid flags
    @property
    def c_k(self):
        return TILE * PITCH

    @property
    def c_s(self):
        return self.c_k + CH * PITCH

    @property
    def c_p(self):
        return self.c_s + TILE * CPITCH

    @property
    def c_v(self):
        return self.c_p + TILE * CPPITCH

    @property
    def c_i(self):  # the chunk's selected positions (a channel selection), int32
        return self.c_v + CH

    @property
    def c_n(
        self,
    ):  # new_early: the step's emitted entries, row j's (if it emits) at j x HD / 2 (bf16 pairs)
        return -(-(self.c_i + CH) // 4) * 4

    @property
    def c_lds(self):
        return self.c_n + (self.spec * (HD // 2) if self.new_early and self.new else 0)

    # smerge merge task LDS (words): V^T [SLICE][VTP] at 0 (then t [NCT][mheads][OP] over it), P [TILE][PP], the
    # selection [SEL] (int32), a KV source's emitted entries' slices [spec][SLICE / 2], maxima and sums [mheads][NCT]
    @property
    def mgroups(
        self,
    ):  # head groups a tile (a single tile of fewer than 16 heads, e.g. 8 at TP8: its live ones)
        return (
            -(-min(self.heads, TILE) // self.mheads)
            if self.tiles == 1
            else TILE // self.mheads
        )

    @property
    def m_vt(self):
        return 0

    @property
    def m_p(self):
        return SLICE * VTP

    @property
    def m_i(self):
        return self.m_p + TILE * PP

    @property
    def m_n(self):
        return self.m_i + SEL

    @property
    def m_x(self):
        return self.m_n + (self.spec * (SLICE // 2) if self.new else 0)

    @property
    def m_l(self):
        return self.m_x + self.mheads * NCT

    @property
    def m_lds(self):
        return self.m_l + self.mheads * NCT


# ------------------------------------------------------------------ the window part


def _win(k, t, p: _SPlan):
    """Row r, tile ht: ``MLAWindowAttn``'s task up to its P V (chunks 0 and 1), then num_1, m_1 and den_1 out."""
    if p.wsplit:
        _wchunk(k, t, p)
        return
    ap = p.win()
    rt, j = (
        (t // p.dsplit, t % p.dsplit) if p.dsplit > 1 else (t, 0)
    )  # (row, tile), the task's dim part
    r, ht = rt // p.tiles, rt % p.tiles
    b, base, pos = _row_pos(k, p.pos, p.spec, r)  # 6: P V, 7: num out
    if ap.ring_first:
        mw._ring_then_q(k, ap, r, ht, b, base, pos)
    else:
        _q_tile(k, ap, r, ht, pos)
        _key_land(k, ap, pos, _ring_issue(k, ap, b, pos), base)
    gpu.barrier()  # every ring row (zeros at the step's own slots) is in before the own rows overwrite them
    _own_keys(k, ap, b, base, pos)
    gpu.barrier()
    _done(k, ap, t)  # the ring is read
    _scores_all(k, ap)
    gpu.barrier()
    if p.smerge:  # the raw scores out; the merge does the rest
        _win_s(k, ap, p, rt)
        gpu.barrier()  # the next task reuses the LDS
        return
    _softmax_all(k, ap, pos)
    gpu.barrier()
    if (
        p.dsplit > 1
    ):  # m_1 (and den_1) out before the P V: the chunk tasks' prefix max waits on it; part 0's
        _win_scalars_if(k, ap, p, rt, j == 0)
    else:
        _win_scalars(k, ap, p, rt)
    acc = _pv_all(k, ap, j * (HD // p.dsplit) if p.dsplit > 1 else 0)
    ld = k.lane // 16
    a = k.mb[f"{p.name}.num"]
    for i in range_constexpr(
        0, len(acc), 2
    ):  # (dim 2 m, dim 2 m + 1) of the lane's 4 heads
        (dim, v0), (_, v1) = acc[i], acc[i + 1]
        for e in range_constexpr(4):
            mb.put2(a, (rt * TILE + ld * 4 + e) * HD + dim, v0[e], v1[e], k.tag)
    gpu.barrier()  # the next task reuses the LDS


def _win_s(k, ap, p, rt):
    """smerge: the window's raw scores S[head][key] (f32, unscaled) as pairs ``<name>.s`` [(row, tile)][NCT][head][CH]
    (chunks 0, 1; the compressed chunks' at 2 ..): thread -> head ``tid // 32``, keys ``(tid % 32) * 2 .. + 2`` of each
    chunk (one 16-byte put)."""
    hh, kk = k.tid // 32, (k.tid % 32) * 2
    a = k.mb[f"{p.name}.s"]
    for c in range_constexpr(WINDOW // CH):
        at = ap.a_s + hh * mw.SPITCH + c * CH + kk
        mb.put2(a, ((rt * NCT + c) * TILE + hh) * CH + kk, fx.ptr_load(k.ldsf + at), fx.ptr_load(k.ldsf + (at + 1)),
                k.tag)  # fmt: skip


def _wchunk(k, t, p: _SPlan):
    """smerge + wsplit: (row, tile, window chunk c): its 64 ring rows (the step's own over theirs), the q tile, the
    scores (``_chunk_scores``: the window task's MFMA chain per key) and their put as chunk c of ``<name>.s``.
    """
    ap = p.win()
    rt, c = t // 2, t % 2
    r, ht = rt // p.tiles, rt % p.tiles
    b, base, pos = _row_pos(k, p.pos, p.spec, r)
    r_ring = rsrc(k.args[p.ring])

    def ring():
        vs = []
        for i in range(CH // WAVES):  # one key row (1 KB) a wave instruction
            kp = pos - (WINDOW - 1) + c * CH + k.wave * (CH // WAVES) + i
            slot = _ring_slot(p, kp)
            vs.append(fx.Vector(bo.buffer_load(r_ring, _row_words(p, b * p.ring_slots + slot) + k.lane * 4,
                                               vec_width=4, dtype=T.i32)))  # fmt: skip
        for i in range(CH // WAVES):
            key = k.wave * (CH // WAVES) + i
            kp = pos - (WINDOW - 1) + c * CH + key
            ok = (kp >= 0) & (kp < base)
            words = [ok.select(vs[i][e], fx.Int32(0)) for e in range(4)]
            fx.ptr_store(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float32),
                         k.ldsf + (p.c_k + key * PITCH + k.lane * 4))  # fmt: skip

    if ap.ring_first:
        ring()
        rocdl.sched_barrier(0)  # keep the q poll after the ring's landing
        _q_tile(k, ap, r, ht, pos)
    else:
        _q_tile(k, ap, r, ht, pos)
        ring()
    gpu.barrier()  # every ring row (zeros at the step's own slots) is in before the own rows overwrite them
    _wown_keys(k, p, b, base, pos, c)
    gpu.barrier()
    _done(k, ap, t)  # the ring is read
    _chunk_scores(k, p)
    gpu.barrier()
    hh, kk = k.tid // 32, (k.tid % 32) * 2
    at = p.c_s + hh * CPITCH + kk
    mb.put2(k.mb[f"{p.name}.s"], ((rt * NCT + c) * TILE + hh) * CH + kk, fx.ptr_load(k.ldsf + at),
            fx.ptr_load(k.ldsf + (at + 1)), k.tag)  # fmt: skip
    gpu.barrier()  # the next task reuses the LDS


@traced
def _wown_keys(k, p: _SPlan, b, base, pos, c):
    """``_own_keys`` for window chunk c: the step's own rows whose window index lies in it, over their LDS rows."""
    for jj in range_constexpr(-(-p.spec * (HD // 2) // (THREADS * 2))):
        w = jj * (THREADS * 2) + k.tid * 2
        row, dw = w // (HD // 2), w % (HD // 2)
        g = base + row - (pos - (WINDOW - 1)) - c * CH  # the key's index in the chunk
        if (base + row <= pos) & (g >= 0) & (g < CH):
            got = mb.poll(
                [(k.mb[p.kvn], (b * p.spec + row) * (HD // 2) + dw, 2)], k.tag
            )[0]
            fx.ptr_store(
                fx.Vector.from_elements(got, fx.Int32).bitcast(fx.Float32),
                k.ldsf + (p.c_k + g * PITCH + dw),
            )


# ------------------------------------------------------------------ a compressed chunk


@traced
def _win_scalars_if(k, ap, p, t, ok):
    if (k.tid < TILE) & ok:
        mb.put2(k.mb[f"{p.name}.w"], 2 * (t * TILE + k.tid), fx.ptr_load(k.ldsf + (ap.a_ml + k.tid)),
                fx.ptr_load(k.ldsf + (ap.a_ml + TILE + k.tid)), k.tag)  # fmt: skip


@traced
def _win_scalars(k, ap, p, t):
    if k.tid < TILE:
        mb.put2(k.mb[f"{p.name}.w"], 2 * (t * TILE + k.tid), fx.ptr_load(k.ldsf + (ap.a_ml + k.tid)),
                fx.ptr_load(k.ldsf + (ap.a_ml + TILE + k.tid)), k.tag)  # fmt: skip


@traced
def _sel_land(k, p: _SPlan, r, c):
    """A channel selection: the chunk's 64 positions into LDS (thread < 64 polls one)."""
    if k.tid < CH:
        w = mb.poll([(k.mb[p.sel], r * SEL + c * CH + k.tid, 1)], k.tag)[0][0]
        fx.ptr_store(fx.Int32(w), k.lds + (p.c_i + k.tid))


def _ckv_addr(k, p: _SPlan, b, idx, uni: bool = True):
    """The 64-bit address of request b's compressed entry ``idx`` (clamped into [0, cache_len): a pad's read lands on
    a valid row) in ATOM's pool, through its block table (``paged.row_addr``; ``uni``: a wave-uniform entry).
    """
    e = fx.min(fx.max(idx, 0), fx.Int32(p.cache_len - 1))
    return paged.row_addr(k, p.ckv, b, e, p.cratio, uni)


def _gather_issue(k, p: _SPlan, b, r, c):
    """The chunk's 64 key rows from the compressed cache by the selection, one key row (1 KB) a wave instruction:
    wave w keys 8 w .. + 8, lane l bytes 16 l .. + 16. -> ([valid], [position], [words]). A KV source's entries of
    this launch are loaded too (clamped) and replaced in ``_gather_land``."""
    oks, idxs, vs = [], [], []
    for i in range(CH // WAVES):
        key = k.wave * (CH // WAVES) + i
        if p.sel_ch:
            idx = uniform(fx.Int32(fx.ptr_load(k.lds + (p.c_i + key))))
        else:
            idx = uniform(fx.Int32(bo.buffer_load(rsrc(k.args[p.sel]), r * SEL + c * CH + key, vec_width=1,
                                                  dtype=T.i32)))  # fmt: skip
        oks.append(idx >= 0)
        idxs.append(idx)
        # (the row's page from the block table: a scalar load, then the 1 KiB row from its own base)
        vs.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(_ckv_addr(k, p, b, idx)), k.lane * 4, vec_width=4, dtype=T.i32
                )
            )
        )
    return oks, idxs, vs


def _gather_land(k, p: _SPlan, oks, idxs, vs, b, base):
    for i in range(CH // WAVES):
        key = k.wave * (CH // WAVES) + i
        words = [oks[i].select(vs[i][e], fx.Int32(0)) for e in range(4)]
        if (
            p.new and p.new_early
        ):  # an entry this launch emits: from the LDS copy of the Compressor's channel
            words = _new_entry_lds(k, p, base, idxs[i], words)
        fx.ptr_store(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float32),
                     k.ldsf + (p.c_k + key * PITCH + k.lane * 4))  # fmt: skip
        _valid_put(k, p, key, oks[i])


@traced
def _new_entry_lds(k, p, base, idx, words):
    """An entry this launch emits, from the LDS copy ``_new_land`` made (the same words)."""
    w0, w1, w2, w3 = words
    if idx >= base // p.ratio:
        j = fx.min(fx.max((idx + 1) * p.ratio - 1 - base, 0), fx.Int32(p.spec - 1))
        at = p.c_n + j * (HD // 2) + k.lane * 4
        w0, w1, w2, w3 = (fx.Int32(fx.ptr_load(k.lds + (at + e))) for e in range(4))
    return [w0, w1, w2, w3]


@traced
def _new_land(k, p: _SPlan, b, base):
    """new_early: the step's emitted entries (request b's rows j with (base + j + 1) % ratio == 0) from the
    Compressor's channel into LDS (thread: row tid / 32, 8 words); they land long before the selection does.
    """
    j = fx.min(k.tid // 32, fx.Int32(p.spec - 1))
    w = (k.tid % 32) * 8
    if (k.tid // 32 < p.spec) & ((base + j + 1) % p.ratio == 0):
        row = (b * p.spec + j) * (HD // 2) + w
        got = mb.poll([(k.mb[p.new], row + 2 * i, 2) for i in range(4)], k.tag)
        at = p.c_n + j * (HD // 2) + w
        for i in range_constexpr(4):
            fx.ptr_store(fx.Int32(got[i][0]), k.lds + (at + 2 * i))
            fx.ptr_store(fx.Int32(got[i][1]), k.lds + (at + 2 * i + 1))


@traced
def _valid_put(k, p, key, ok):
    if k.lane == 0:
        fx.ptr_store(
            ok.select(fx.Float32(1.0), fx.Float32(0.0)), k.ldsf + (p.c_v + key)
        )


@traced
def _chunk_scores(k, p: _SPlan):
    """S[head][key] (unscaled f32) of the chunk's 64 keys: waves 0 .. 3 take keys 16 w .. + 16 (``_scores_all``'s
    MFMA chain)."""
    lm, ld = k.lane % 16, k.lane // 16
    if k.wave < CH // 16:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for kst in range_constexpr(HD // 32):
            a = _lds_bf8(k, p.c_k + (k.wave * 16 + lm) * PITCH + kst * 16 + ld * 4)
            bq = _lds_bf8(k, lm * PITCH + kst * 16 + ld * 4)
            acc = mfma_bf16(a, bq, acc)
        for e in range_constexpr(4):
            fx.ptr_store(
                acc[e], k.ldsf + (p.c_s + lm * CPITCH + k.wave * 16 + ld * 4 + e)
            )


def _chunk_softmax(k, p: _SPlan, t, c, lead=True):
    """Thread -> head ``tid // 32``, keys ``(tid % 32) * 2 .. + 2`` (``_softmax_all``'s mapping and operations for one
    chunk): the chunk max out, the prefix max in, P (bf16) to LDS, sum(P) out."""
    hh, kk = k.tid // 32, (k.tid % 32) * 2
    scale = fx.Float32(1.0 / math.sqrt(HD))
    sv = []
    for i in range_constexpr(2):
        raw = fx.ptr_load(k.ldsf + (p.c_s + hh * CPITCH + kk + i)) * scale
        sv.append(
            (fx.ptr_load(k.ldsf + (p.c_v + kk + i)) != fx.Float32(0.0)).select(
                raw, fx.Float32(-math.inf)
            )
        )
    mx = butterfly(fx.max(sv[0], sv[1]), (1, 2, 4, 8, 16), fx.max)
    tt = t // NCH  # the (row, tile)
    _lead_put(k, k.mb[f"{p.name}.mx"], (tt * NCH + c) * TILE + hh, mx, lead)
    # the prefix max: m_1 (the window's) and the earlier chunks' maxima (max in any order: exact). Every chunk's max
    # is polled in one batch (each task puts its own before it waits: no cycle), the later ones dropped
    specs = [(k.mb[f"{p.name}.w"], 2 * (tt * TILE + hh), 1)]
    specs += [
        (k.mb[f"{p.name}.mx"], (tt * NCH + jc) * TILE + hh, 1) for jc in range(NCH)
    ]
    got = mb.poll(specs, k.tag)
    m_prev = got[0][0].bitcast(fx.Float32)
    for jc in range_constexpr(NCH):
        m_prev = (fx.Int32(jc) < c).select(
            fx.max(m_prev, got[1 + jc][0].bitcast(fx.Float32)), m_prev
        )
    m_new = fx.max(m_prev, mx)
    pv = [fx.Float32(fmath.exp(v - m_new)) for v in sv]
    lsum = butterfly(pv[0] + pv[1], (1, 2, 4, 8, 16))
    fx.ptr_store(bf16_pair(pv[0], pv[1]), k.ldsf + (p.c_p + hh * CPPITCH + kk // 2))
    _lead_put(k, k.mb[f"{p.name}.ls"], (tt * NCH + c) * TILE + hh, lsum, lead)


def _lead_put(k, a, i, v, ok=True):
    """The head's lead thread puts (``ok``: a traced condition too, e.g. a dim part's index)."""
    if ok is True:
        _lead_put1(k, a, i, v)
    else:
        _lead_put_if(k, a, i, v, ok)


@traced
def _lead_put_if(k, a, i, v, ok):
    if (k.tid % 32 == 0) & ok:
        mb.put(a, i, v, k.tag)


@traced
def _lead_put1(k, a, i, v):
    if k.tid % 32 == 0:
        mb.put(a, i, v, k.tag)


@traced
def _chunk_pv(k, p: _SPlan, t, j=0):
    """P_c V_c: wave w holds dims 64 w .. + 64 of the 16 heads (``_pv_all``'s operands and order for one chunk; with
    ``dsplit`` 2, part j's 256: 32 a wave), put as (dim 2 m, dim 2 m + 1) pairs of each of the lane's 4 heads.
    """
    lm, ld = k.lane % 16, k.lane // 16
    a_t = k.mb[f"{p.name}.t"]
    dpw = 64 // p.dsplit
    for blk in range_constexpr(dpw // 32):
        d_blk = j * (HD // p.dsplit) + k.wave * dpw + blk * 32
        dword = d_blk // 2 + lm
        ts = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for kst in range_constexpr(CH // 32):
            a = _lds_bf8(k, p.c_p + lm * CPPITCH + kst * 16 + ld * 4)
            ws = [fx.ptr_load(k.ldsf + (p.c_k + (kst * 32 + ld * 8 + i) * PITCH + dword)).bitcast(fx.Int32)
                  for i in range(8)]  # fmt: skip
            halves = (
                [w & fx.Int32(0xFFFF) for w in ws],
                [fx.Int32(fx.Uint32(w) >> 16) for w in ws],
            )
            for par in range_constexpr(2):
                hv = halves[par]
                bw = [hv[2 * i] | (hv[2 * i + 1] << fx.Int32(16)) for i in range(4)]
                ts[par] = mfma_bf16(
                    a,
                    fx.Vector.from_elements(bw, fx.Int32).bitcast(fx.BFloat16),
                    ts[par],
                )
        dim = d_blk + 2 * lm
        for e in range_constexpr(4):
            mb.put2(a_t, (t * TILE + ld * 4 + e) * HD + dim, ts[0][e], ts[1][e], k.tag)


def _chunk(k, t, p: _SPlan):
    """(row, tile, compressed chunk c[, dim part j])."""
    t, j = (t // p.dsplit, t % p.dsplit) if p.dsplit > 1 else (t, 0)
    tt, c = t // NCH, t % NCH
    r, ht = tt // p.tiles, tt % p.tiles
    b, base, pos = _row_pos(k, p.pos, p.spec, r)
    if p.new_early and p.new:  # before the selection's poll: the entries land first
        _new_land(k, p, b, base)
    if p.sel_ch:  # this launch's selection (an index source)
        _sel_land(k, p, r, c)
    if p.sel_ch or (p.new_early and p.new):
        gpu.barrier()
    oks, idxs, vs = _gather_issue(
        k, p, b, r, c
    )  # a consumer: plain, from earlier launches, before the q poll
    if p.kfirst and not p.new:  # the keys into LDS while q comes
        _gather_land(k, p, oks, idxs, vs, b, base)
    _q_tile(k, p.win(), r, ht, pos)
    if not (p.kfirst and not p.new):
        _gather_land(k, p, oks, idxs, vs, b, base)
    gpu.barrier()
    _chunk_scores(k, p)
    gpu.barrier()
    if (
        p.smerge
    ):  # the raw scores out (``<name>.s``, chunk 2 + c); the merge does the rest
        hh, kk = k.tid // 32, (k.tid % 32) * 2
        at = p.c_s + hh * CPITCH + kk
        mb.put2(k.mb[f"{p.name}.s"], ((tt * NCT + 2 + c) * TILE + hh) * CH + kk, fx.ptr_load(k.ldsf + at),
                fx.ptr_load(k.ldsf + (at + 1)), k.tag)  # fmt: skip
        gpu.barrier()  # the next task reuses the LDS
        return
    _chunk_softmax(
        k, p, t, c, True if p.dsplit == 1 else j == 0
    )  # part 0 puts the chunk's max and sum
    gpu.barrier()
    _chunk_pv(k, p, t, j)
    gpu.barrier()  # the next task reuses the LDS


# ------------------------------------------------------------------ the merge


# ------------------------------------------------------------------ the merge from the raw scores (smerge)


def _bfly(vs, offsets, op=None):
    """``butterfly`` of each of ``vs`` (the same operations each), level by level across them: independent chains
    interleaved instead of one after another."""
    op = op or (lambda x, y: x + y)
    for off in offsets:
        vs = [xred(v, off, op) for v in vs]
    return vs


def _ring_slot(p: _SPlan, kp):
    """A window key's ring slot: ``kp mod ring_slots`` (``kp >= -(WINDOW - 1)``; ``lean``: one unsigned modulo)."""
    if p.lean:
        return fx.Int32(fx.Uint32(kp + p.ring_slots) % p.ring_slots)
    return ((kp % p.ring_slots) + p.ring_slots) % p.ring_slots


def _row_words(p: _SPlan, row):
    """Word offset of a cache / ring row (``row >= 0``) of HD bf16."""
    return row * (HD // 2) if p.lean else (row * HD) // 2


def _perm(a, b, sel):
    """v_perm_b32: bytes of {a (4-7), b (0-3)} picked by ``sel``'s bytes."""
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.perm", [fx.Int32(a).ir_value(), fx.Int32(b).ir_value(),
                                                                     fx.Int32(sel).ir_value()], [], []))  # fmt: skip


@traced
def _msel_land(k, p: _SPlan, r):
    """The row's 512 selected positions into LDS (a thread each: the channel's pair, or the plain arg)."""
    if const_expr(p.sel_ch):
        w = mb.poll([(k.mb[p.sel], r * SEL + k.tid, 1)], k.tag)[0][0]
    else:
        w = bo.buffer_load(
            rsrc(k.args[p.sel]), r * SEL + k.tid, vec_width=1, dtype=T.i32
        )
    fx.ptr_store(fx.Int32(w), k.lds + (p.m_i + k.tid))


@traced
def _mnew_land(k, p: _SPlan, b, base, s):
    """A KV source: the slice's words of the entries the step emits (``_new_land`` for 64 dims): thread -> row
    ``tid // 16``, words ``2 (tid % 16) .. + 2``."""
    j = fx.min(k.tid // 16, fx.Int32(p.spec - 1))
    w = (k.tid % 16) * 2
    if (k.tid // 16 < p.spec) & ((base + j + 1) % p.ratio == 0):
        got = mb.poll(
            [(k.mb[p.new], (b * p.spec + j) * (HD // 2) + s * (SLICE // 2) + w, 2)],
            k.tag,
        )[0]
        at = p.m_n + j * (SLICE // 2) + w
        fx.ptr_store(fx.Int32(got[0]), k.lds + at)
        fx.ptr_store(fx.Int32(got[1]), k.lds + (at + 1))


@traced
def _mnew_entry(k, p, base, idx, piece, words):
    w0, w1, w2, w3 = words
    if idx >= base // p.ratio:
        j = fx.min(fx.max((idx + 1) * p.ratio - 1 - base, 0), fx.Int32(p.spec - 1))
        at = p.m_n + j * (SLICE // 2) + piece * 4
        w0, w1, w2, w3 = (fx.Int32(fx.ptr_load(k.lds + (at + e))) for e in range(4))
    return [w0, w1, w2, w3]


def _vk_word(p: _SPlan, key, word):
    """``vtr``: the LDS word of a key's row word ``word`` (dims 2 word, + 1): row ``key`` of 32 words, its 4-word
    pieces XOR-swizzled by ``(key & 2) | ((key >> 1) & 4)``."""
    sw = (key & 2) | ((key >> 1) & 4)
    return p.m_vt + key * (SLICE // 2) + ((sw ^ (word >> 2)) << 2) + (word & 3)


def _vt_store(k, p: _SPlan, key0, words, par, piece):
    """A lane pair (lane ^ 1: keys key0, key0 + 1 by ``par``) of 4 words each (dims 8 piece .. + 8 of a key):
    halves exchanged so that each lane holds 4 words (one dim, the key pair: the even key in the low half), dims
    8 piece + 4 par .. + 4, stored into V^T. ``vtr``: each lane's 4 words as they are into its key's row.
    """
    if p.vtr:
        fx.ptr_store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32),
                     k.lds + _vk_word(p, key0 + par, piece * 4))  # fmt: skip
        return
    even = par == 0
    sent = [even.select(words[2], words[0]), even.select(words[3], words[1])]
    got = [fx.Int32(xshfl(v, 1)) for v in sent]
    own = [even.select(words[0], words[2]), even.select(words[1], words[3])]
    out = []
    if (
        p.lean
    ):  # the same words: (key0's half, key0 + 1's half) by byte selects of (own, got)
        lo = even.select(fx.Int32(0x01000504), fx.Int32(0x05040100))
        hi = even.select(fx.Int32(0x03020706), fx.Int32(0x07060302))
        for o, g in zip(own, got, strict=True):
            out += [_perm(o, g, lo), _perm(o, g, hi)]
        half = fx.Int32(fx.Uint32(key0) >> 1)
    d0 = piece * 8 + par * 4
    for m in range(4):
        fx.ptr_store(
            out[m].bitcast(fx.Float32), k.ldsf + (p.m_vt + (d0 + m) * VTP + half)
        )


def _mown_spec(k, p: _SPlan, b, base, pos, s):
    """The step's own window keys' slices (``kvn``): thread -> row ``tid // 16`` (clamped), words ``2 (tid % 16) ..
    + 2`` (4 dims) -> (the poll spec, whether the thread has a row)."""
    j = fx.min(k.tid // 16, fx.Int32(p.spec - 1))
    w = (k.tid % 16) * 2
    spec = (k.mb[p.kvn], (b * p.spec + j) * (HD // 2) + s * (SLICE // 2) + w, 2)
    return spec, (k.tid // 16 < p.spec) & (base + j <= pos)


def _mown_try(spec):
    """One attempt of the own rows' poll, issued at task start, not waited for (``linear._try_batch``)."""
    a, i, n = spec
    note(a.region, a.space, "poll")
    v = fx.Vector(
        bo.buffer_load(
            rsrc(a.value),
            fx.Int32(i) * 2,
            vec_width=2 * n,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    return [v[e] for e in range(2 * n)]


@traced
def _mown_land(k, p: _SPlan, base, pos, spec, ok, words, ldsb):
    """The attempt's words (polled again by a thread whose tags were stale) over their V^T columns (bf16 stores, after
    the ring rows' zeros)."""
    if ok:
        w0, w1 = fx.Int32(words[0]), fx.Int32(words[2])
        if (words[1] != k.tag) | (words[3] != k.tag):
            got = mb.poll([spec], k.tag)[0]
            w0, w1 = fx.Int32(got[0]), fx.Int32(got[1])
        j = fx.min(k.tid // 16, fx.Int32(p.spec - 1))
        w = (k.tid % 16) * 2
        g = base + j - (pos - (WINDOW - 1))  # the key's index in the window
        if const_expr(p.vtr):  # the two words into the key's row
            fx.ptr_store(
                fx.Vector.from_elements([w0, w1], fx.Int32), k.lds + _vk_word(p, g, w)
            )
        for i, wd in enumerate((w0, w1) if not p.vtr else ()):
            v = fx.Vector.from_elements([wd], fx.Int32).bitcast(fx.BFloat16)
            for e in range_constexpr(2):
                fx.ptr_store(v[e], ldsb + ((p.m_vt + (2 * (w + i) + e) * VTP) * 2 + g))


@traced
def _mdone(k, p, t):
    if k.tid == 0:
        mb.put(k.mb[f"{p.name}.mdone"], t, fx.Int32(1), k.tag)


@traced
def _lds_put_if(k, ok, word, v):
    if ok:
        fx.ptr_store(v, k.ldsf + word)


def _pick(vals, i):
    """``vals[i]`` for a traced index (a select chain)."""
    out = vals[0]
    for c in range(1, len(vals)):
        out = (i == c).select(vals[c], out)
    return out


def _ms_specs(k, p: _SPlan, tt, hg):
    """The task's scores poll (``_msoftmax``'s thread mapping) -> (specs, chunks, valid, h, j)."""
    G = THREADS // (p.mheads * 32)
    g, h, j = k.tid // (p.mheads * 32), (k.tid // 32) % p.mheads, k.tid % 32
    hh = hg * p.mheads + h
    ccs = [g + G * i for i in range(-(-NCT // G))]
    oks = [cc < NCT for cc in ccs]
    ccs = [fx.min(cc, fx.Int32(NCT - 1)) for cc in ccs]
    specs = [
        (k.mb[f"{p.name}.s"], ((tt * NCT + cc) * TILE + hh) * CH + 2 * j, 2)
        for cc in ccs
    ]
    return specs, ccs, oks, h, j


def _msoftmax(k, p: _SPlan, tt, hg, pos):
    """The softmax of the task's heads over all ten chunks with the producers' thread mapping per chunk (32 threads
    a head, keys 2 j, 2 j + 1: the same butterflies and operations as ``_softmax_all`` / ``_chunk_softmax``): thread
    -> chunk group g (chunks g, g + G, ..), head h, j. Each chunk's max into LDS, then each head's running max over the
    chunks in order (a compressed chunk's prefix max: max is exact), P (bf16) into LDS [head][640 keys] and each
    chunk's sum(P) into LDS."""
    specs, ccs, oks, h, j = _ms_specs(k, p, tt, hg)
    kk = 2 * j
    got = mb.poll(specs, k.tag)
    scale = fx.Float32(1.0 / math.sqrt(HD))
    svs = []
    for i, cc in enumerate(ccs):
        sv = []
        for e in range(2):
            raw = got[i][e].bitcast(fx.Float32) * scale
            ci = fx.min(fx.max(cc - 2, 0), fx.Int32(NCH - 1))
            win = (cc < 2) & (pos - (WINDOW - 1) + cc * CH + kk + e >= 0)
            cmp = (cc >= 2) & (
                fx.Int32(fx.ptr_load(k.lds + (p.m_i + ci * CH + kk + e))) >= 0
            )
            sv.append((win | cmp).select(raw, fx.Float32(-math.inf)))
        svs.append(sv)
    mxs = _bfly([fx.max(sv[0], sv[1]) for sv in svs], (1, 2, 4, 8, 16), fx.max)
    for i, cc in enumerate(ccs):
        _lds_put_if(k, oks[i] & (j == 0), p.m_x + h * NCT + cc, mxs[i])
    gpu.barrier()
    m_prev, ms = fx.Float32(mw.NEG_INIT), []
    for c in range(NCT):
        m_prev = fx.max(m_prev, fx.ptr_load(k.ldsf + (p.m_x + h * NCT + c)))
        ms.append(m_prev)
    pvs = [
        [fx.Float32(fmath.exp(v - _pick(ms, cc))) for v in sv]
        for sv, cc in zip(svs, ccs, strict=True)
    ]
    lss = _bfly([pv[0] + pv[1] for pv in pvs], (1, 2, 4, 8, 16))
    for i, cc in enumerate(ccs):
        _lds_put_if(
            k,
            oks[i],
            p.m_p + h * PP + cc * (CH // 2) + j,
            bf16_pair(pvs[i][0], pvs[i][1]),
        )
        _lds_put_if(k, oks[i] & (j == 0), p.m_l + h * NCT + cc, lss[i])


@traced
def _mt_store(k, p, cc, acc, dim, ld):
    """t[cc][head][dim] of the lane's heads ld * 4 + e below ``mheads``."""
    for e in range_constexpr(4):
        if ld * 4 + e < p.mheads:
            fx.ptr_store(
                acc[e], k.ldsf + (p.m_vt + (cc * p.mheads + ld * 4 + e) * OP + dim)
            )


def _mpv(k, p: _SPlan):
    """P V of every chunk for the slice: wave w the 16 dims ``16 (w % 4) ..`` (an MFMA column a dim, B from V^T: the
    producers' operand values, so their bits), the chunks of half w // 4; held until every wave's V^T reads are done,
    then t [NCT][mheads][OP] over V^T."""
    lm, ld = k.lane % 16, k.lane // 16
    cb, half = k.wave % 4, k.wave // 4
    dim = cb * 16 + lm
    if (
        p.vtr
    ):  # the lane's row of each transpose read: key 8 ld + lm // 4 (+ 4, 32 kst, 64 cc), dims 16 cb + 4 (lm % 4)
        vk = (
            fx.recast_iter(fx.BFloat16, k.lds)
            + _vk_word(p, ld * 8 + (lm >> 2), cb * 8 + (lm & 3) * 2) * 2
        )
    ts = []
    for i in range(NCT // 2):
        cc = half * (NCT // 2) + i
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for kst in range(CH // 32):
            a = _lds_bf8(k, p.m_p + lm * PP + cc * (CH // 2) + kst * 16 + ld * 4)
            if (
                p.vtr
            ):  # keys 8 ld .. + 4 and + 4 .. + 8 of the 32 (the swizzle depends on key bits 1 and 3 only)
                off = (cc * CH + kst * 32) * SLICE
                w = lds_read_tr16(vk + off) + lds_read_tr16(vk + (off + 4 * SLICE))
                bq = fx.Vector.from_elements(w, fx.Int32).bitcast(fx.BFloat16)
            else:
                bq = _lds_bf8(
                    k, p.m_vt + dim * VTP + cc * (CH // 2) + kst * 16 + ld * 4
                )
            acc = mfma_bf16(a, bq, acc)
        ts.append((cc, acc))
    gpu.barrier()  # every wave's V^T reads are done: t overlays it
    for cc, acc in ts:
        _mt_store(k, p, cc, acc, dim, ld)


def _smerge(k, t, p: _SPlan):
    """smerge (row, tile, head group, 64-dim slice): V^T of the slice for every key (the window's from the ring and
    ``kvn``, the compressed ones by the selection) landed while the scores come, then the raw scores of the task's
    heads in all ten chunks polled (one round trip after the last chunk's scores), the softmax, P V, the recurrence
    and the output."""
    nsl = HD // SLICE
    tt, rem = t // (p.mgroups * nsl), t % (p.mgroups * nsl)
    hg, s = (
        rem // nsl,
        rem % nsl,
    )  # head-group siblings 8 tasks apart: the same XCD (their V^T reads alike)
    r, ht = tt // p.tiles, tt % p.tiles
    b, base, pos = _row_pos(k, p.pos, p.spec, r)
    # V loads: key kq of each 64 (lane pairs: consecutive keys), words 4 piece .. + 4 of the slice
    par, piece = k.lane % 2, (k.lane // 2) % 8
    kq = k.wave * 8 + (k.lane // 16) * 2 + par
    r_ring = rsrc(k.args[p.ring])
    wv = []
    for i in range(
        0 if p.ring8 else WINDOW // CH
    ):  # the window's ring rows (before the step's first position)
        kp = pos - (WINDOW - 1) + i * CH + kq
        slot = _ring_slot(p, kp)
        wv.append(fx.Vector(bo.buffer_load(r_ring, _row_words(p, b * p.ring_slots + slot) + s * (SLICE // 2)
                                           + piece * 4, vec_width=4, dtype=T.i32)))  # fmt: skip
    own_spec, own_ok = _mown_spec(k, p, b, base, pos, s)
    own = _mown_try(own_spec)
    _msel_land(k, p, r)
    if p.new:
        _mnew_land(k, p, b, base, s)
    gpu.barrier()
    oks, idxs, cv = [], [], []
    for c in range(0 if p.ckv4 else NCH):
        idx = fx.Int32(fx.ptr_load(k.lds + (p.m_i + c * CH + kq)))
        oks.append(idx >= 0)
        idxs.append(idx)
        # (a lane's own key: its page by a vector table load, the 16 B by a 64-bit global load)
        off = fx.Int64((s * (SLICE // 2) + piece * 4) * 4)
        cv.append(fx.Vector(global_load(_ckv_addr(k, p, b, idx, uni=False) + off, 4)))
    for i in range(0 if p.ring8 else WINDOW // CH):
        kp = pos - (WINDOW - 1) + i * CH + kq
        ok = (kp >= 0) & (kp < base)
        _vt_store(
            k,
            p,
            i * CH + kq - par,
            [ok.select(wv[i][e], fx.Int32(0)) for e in range(4)],
            par,
            piece,
        )
    for c in range(0 if p.ckv4 else NCH):
        words = [oks[c].select(cv[c][e], fx.Int32(0)) for e in range(4)]
        if p.new:
            words = _mnew_entry(k, p, base, idxs[c], piece, words)
        _vt_store(k, p, (2 + c) * CH + kq - par, words, par, piece)
    gpu.barrier()  # every ring row (zeros at the step's own slots) is in before the own rows overwrite them
    _mown_land(
        k, p, base, pos, own_spec, own_ok, own, fx.recast_iter(fx.BFloat16, k.lds)
    )
    if not p.ring8:  # (the copy is the import's: RingWrite writes only the bf16 ring)
        _mdone(k, p, t)  # the ring is read (RingWrite)
    _msoftmax(k, p, tt, hg, pos)
    _smerge_tail(k, p, t, tt, hg, s, r, ht, pos)


def _smerge_tail(k, p: _SPlan, t, tt, hg, s, r, ht, pos):
    """After the softmax: P V, the recurrence, the output."""
    hh, j = (
        fx.min(k.tid // 32, fx.Int32(p.mheads - 1)),
        k.tid % 32,
    )  # the output's head (tid < mheads x 32) and dims
    head = p.head(ht, hg * p.mheads, hh)
    d0 = s * SLICE + 2 * j
    sink = fx.Float32(
        bo.buffer_load(rsrc(k.args[p.sink]), head, vec_width=1, dtype=T.f32)
    )
    first = fx.max(d0 - (HD - ROPE), 0) // 2  # the rotary pair (d0, d0 + 1)
    cs = rope_cs(k, p.rope, pos, first)
    gpu.barrier()
    _mpv(k, p)
    gpu.barrier()
    # the window's state (``_softmax_all``: m, den and the corrections from -1e30), then ``_merge``'s recurrence
    mx = [fx.ptr_load(k.ldsf + (p.m_x + hh * NCT + c)) for c in range(NCT)]
    ls = [fx.ptr_load(k.ldsf + (p.m_l + hh * NCT + c)) for c in range(NCT)]
    tv = [[fx.ptr_load(k.ldsf + (p.m_vt + (cc * p.mheads + hh) * OP + 2 * j + e)) for e in range(2)]
          for cc in range(NCT)]  # fmt: skip
    m_prev, den, corrs = fx.Float32(mw.NEG_INIT), fx.Float32(0.0), []
    for c in range(2):
        m_new = fx.max(m_prev, mx[c])
        corr = fx.Float32(fmath.exp(m_prev - m_new))
        den = den * corr + ls[c]
        corrs.append(corr)
        m_prev = m_new
    num = [
        (fx.Float32(0.0) * corrs[0] + tv[0][e]) * corrs[1] + tv[1][e] for e in range(2)
    ]
    m = m_prev
    for c in range(NCH):  # the reference's recurrence in chunk order
        m_new = fx.max(m, mx[2 + c])
        corr = fx.Float32(fmath.exp(m - m_new))
        num = [num[e] * corr + tv[2 + c][e] for e in range(2)]
        den = den * corr + ls[2 + c]
        m = m_new
    den = den + fx.Float32(fmath.exp(sink - m))
    o = [bf16_round(num[e] / den) for e in range(2)]
    is_rope = d0 >= HD - ROPE
    a_, b_ = mw._rope_pair(o[0], o[1], fx.Float32(cs[0]), fx.Float32(cs[1]), True)
    o = [is_rope.select(a_, o[0]), is_rope.select(b_, o[1])]
    am = butterfly(
        fx.max(mw._ab(o[0]), mw._ab(o[1])), (1, 2, 4, 8), fx.max
    )  # the 32-dim group: 16 lanes
    code = gemv.activation_code(am)
    nb = [
        xshfl(o[e], 1) for e in range(2)
    ]  # the odd lane's two values: the even lane's word
    word = _qat_word([o[0], o[1], nb[0], nb[1]], code)[0]
    e0 = r * p.heads * HD + head * HD + d0
    _out_put_if(k, p, j, e0, word, code, k.tid < p.mheads * 32)
    gpu.barrier()  # the next task reuses the LDS


@traced
def _out_put_if(k, p, j, e0, word, code, ok):
    if (j % 2 == 0) & ok:
        chan.put_mxfp8_word(k.mb[f"{p.out}.q"], e0 // 4, word, k.tag)
    if (j % 16 == 0) & ok:
        chan.put_mxfp8_scale(k.mb[f"{p.out}.s"], e0 // 32, code, k.tag)


def _from_rows(knob, sc: Scenario) -> bool:
    """A copy knob: False, True, or an int n = from n rows."""
    return bool(knob) and sc.rows >= (1 if knob is True else knob)


class MLASparseAttn(Block):
    """A compressed consumer layer's attention (module docstring); ``heads``: this rank's: 16-head tiles, the last one
    ragged when heads is not a multiple of 16 (8 at TP8: its dead heads compute and write the last live head's).
    """

    def __init__(self, name: str, q: Channel, kvn: Channel, out: Channel, ring: str, pos: str, rope: str, sink: str,
                 sel: str | Channel, ckv: str, cache_len: int, heads: int = HEADS, new: Channel | None = None,
                 ratio: int = 1, ring_slots: int = WINDOW, ring_first: bool = False,
                 new_early: bool = False, dsplit: int = 1, smerge: bool = False, mheads: int = TILE,
                 wsplit: bool = False, ckv4: bool | int = False, ring8: bool | int = False,
                 cache_ratio: int = 0, lean: bool = False, vtr: bool = False, kfirst: bool = False,
                 unrope_f32: bool = False):  # fmt: skip
        if heads < 1:
            raise ValueError(f"MLASparseAttn {name}: heads {heads}")
        if q.fmt is not Fmt.BF16 or q.width != heads * HD or q.flagged:
            raise ValueError(
                f"MLASparseAttn {name}: q is a tagged BF16 channel of width {heads * HD}"
            )
        if kvn.fmt is not Fmt.BF16 or kvn.width != HD or kvn.flagged:
            raise ValueError(
                f"MLASparseAttn {name}: kvn is a tagged BF16 channel of width {HD}"
            )
        if out.fmt is not Fmt.MXFP8 or out.width != heads * HD or out.flagged:
            raise ValueError(
                f"MLASparseAttn {name}: out is a tagged MXFP8 channel of width {heads * HD}"
            )
        self.name, self.q, self.kvn, self.out, self.heads = name, q, kvn, out, heads
        self.ring, self.pos, self.rope, self.sink = ring, pos, rope, sink
        if isinstance(sel, Channel) and (
            sel.fmt is not Fmt.F32 or sel.width != SEL or sel.flagged
        ):
            raise ValueError(
                f"MLASparseAttn {name}: a channel selection is a tagged F32 channel of width {SEL}"
            )
        if new is not None and (
            new.fmt is not Fmt.BF16 or new.width != HD or new.flagged
        ):
            raise ValueError(
                f"MLASparseAttn {name}: new entries are a tagged BF16 channel of width {HD}"
            )
        self.sel, self.ckv, self.cache_len, self.new, self.ratio = (
            sel,
            ckv,
            cache_len,
            new,
            ratio,
        )
        if ring_slots < WINDOW:
            raise ValueError(
                f"MLASparseAttn {name}: ring_slots {ring_slots} < the window {WINDOW}"
            )
        self.ring_slots, self.ring_first = ring_slots, ring_first
        # a KV source: the step's emitted entries polled into LDS at chunk task start instead of after the gather (one
        # round trip fewer on the chain after the selection; the same words)
        self.new_early = new_early
        # window / chunk tasks a (row, tile[, chunk]) by P V dims: each lands its keys, scores and softmax in full (the
        # same operations), then the P V and the puts of HD / dsplit dims (half the 64 KB of partials a task at 2); the
        # window's m_1 / den_1 and a chunk's max / sum from part 0. RingWrite counts the window's parts (``dsplit``)
        if dsplit not in (1, 2):
            raise ValueError(f"MLASparseAttn {name}: dsplit 1 or 2")
        self.dsplit = dsplit
        # the window / chunk tasks put their raw scores (8 KB a chunk) and stop; the merge tasks land their 64 dims
        # of every key's V beforehand, then run the softmax of all ten chunks (the producers' operations: the same
        # P, maxima and sums), the P V and the recurrence: one hand-off after the scores instead of the prefix max's
        # and the 64 KB partials'; the same bits. RingWrite also waits for the merge's ring
        # reads (``ring_mdone``)
        if smerge and dsplit != 1:
            raise ValueError(
                f"MLASparseAttn {name}: smerge has no P V in the window / chunk tasks (dsplit 1)"
            )
        if mheads not in (
            0,
            1,
            2,
            4,
            8,
            16,
        ):  # 1 / 2: fewer heads' softmax a task, more tasks (few rows)
            raise ValueError(f"MLASparseAttn {name}: mheads 1, 2, 4, 8, 16 or 0 (auto)")
        if wsplit and not smerge:
            raise ValueError(f"MLASparseAttn {name}: wsplit needs smerge")
        self.smerge, self.mheads, self.wsplit = smerge, mheads, wsplit
        # ckv4 (the merge's compressed V from an FP4 copy of the cache): refused, the cache is ATOM's pool
        self.ckv4 = ckv4 if smerge else False
        if self.ckv4:
            raise ValueError(
                f"MLASparseAttn {name}: ckv4 (the cache's FP4 copy): the cache is ATOM's pool, in place"
            )
        if cache_ratio < 1:
            raise ValueError(
                f"MLASparseAttn {name}: cache_ratio (the compressed cache's ratio: its page geometry)"
            )
        self.cache_ratio = cache_ratio
        # ring8: the same for the window's V from the ring's MXFP8 copy (True or from n rows).
        # The copy is the caller's (the hook's import, a step): RingWrite writes the bf16 ring only and no longer
        # waits for the merge
        self.ring8 = ring8 if smerge else False
        # the chunk / merge tasks' index math lean and the merge's V^T pairs by v_perm (``_SPlan.lean``): the same bits
        self.lean = lean
        # the merge's V key-major and its P V's B operand by LDS transpose reads (``_SPlan.vtr``): the same bits
        self.vtr = vtr and smerge
        # chunk tasks: the gathered keys into LDS before the q poll (``_SPlan.kfirst``): the same bits
        self.kfirst = kfirst
        # the output's inverse RoPE quantized from f32 (``MLAWindowAttn(unrope_f32=)``): a numerics choice
        self.unrope_f32 = unrope_f32

    @property
    def ring_parts(self) -> int:
        """RingWrite: the window stage's tasks a (row, tile) (each puts its ring-read flag)."""
        return 2 if self.wsplit else self.dsplit

    def ring_mdone(self, sc: Scenario) -> tuple[str, int] | None:
        """RingWrite: the merge's ring-read flags (region, tasks a row), with ``smerge``."""
        p = self.plan(sc)
        return f"{self.name}.mdone", HD // SLICE * p.tiles * p.mgroups

    def _mheads(self, sc: Scenario) -> int:
        """``mheads`` 0 ("auto"): 4 up to 6 rows (192 merge tasks), else 16 (layer 3's program, A / B at 16 rows:
        16 −1.6 µs, 8 −0.9, 4 +5.2 against smerge off)."""
        return self.mheads or (4 if sc.rows <= 6 else 16)

    def plan(self, sc: Scenario) -> _SPlan:
        sel_ch = isinstance(self.sel, Channel)
        return _SPlan(self.name, sc.rows, sc.spec, self.q.name, self.kvn.name, self.ring, self.pos, self.rope,
                      self.sink, self.sel.name if sel_ch else self.sel, self.ckv, self.cache_len, self.out.name,
                      self.heads, sel_ch, self.new.name if self.new is not None else "", self.ratio,
                      self.ring_slots, self.ring_first,
                      self.new_early, self.dsplit,
                      self.smerge, self._mheads(sc), self.wsplit,
                      _from_rows(self.ckv4, sc), _from_rows(self.ring8, sc), max(self.cache_ratio, 1),
                      lean=self.lean, vtr=self.vtr, kfirst=self.kfirst, unrope_f32=self.unrope_f32)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        rt = sc.rows * p.tiles  # (row, tile) pairs
        n = self.name
        if p.smerge and p.spec * 16 > THREADS:
            raise ValueError(
                f"MLASparseAttn {n}: smerge lands the own keys a thread per 2 words (spec <= 32)"
            )
        if p.smerge:
            # one scores region for both producers: the merge's poll indexes it by chunk
            w_writes = (Region(f"{n}.done", rt * self.ring_parts, (n,)),
                        Region(f"{n}.s", rt * NCT * TILE * CH, (n, f"{n}.c")))  # fmt: skip
            c_writes = ()
            m_writes = tuple(self.out.regions(sc.rows, (f"{n}.m",))) + (
                ()
                if p.ring8
                else (
                    Region(f"{n}.mdone", rt * p.mgroups * (HD // SLICE), (f"{n}.m",)),
                )
            )
            m_args = (
                (
                    self.pos,
                    self.rope,
                    rope_sin(self.rope),
                    self.sink,
                    self.ring,
                    self.ckv,
                    *paged.REQ,
                )
                + (() if p.sel_ch else (self.sel,))
                + ((f"{self.ckv}.q4", f"{self.ckv}.s4") if p.ckv4 else ())
                + ((f"{self.ring}.q8", f"{self.ring}.s8") if p.ring8 else ())
            )
            m_emit, m_lds = partial(_smerge, p=p), p.m_lds
        ap = p.win()
        return [
            Stage(n, rt * self.ring_parts, partial(_win, p=p), lds_words=p.c_lds if p.wsplit else ap.lds_words,
                  writes=w_writes,
                  args=(self.pos, self.ring, self.rope, rope_sin(self.rope)),
                  key=(p,) + ("win",), row_tasks=p.tiles * self.ring_parts),
            Stage(f"{n}.c", rt * NCH * p.dsplit, partial(_chunk, p=p), lds_words=p.c_lds, writes=c_writes,
                  args=(self.pos, self.rope, rope_sin(self.rope), self.ckv, *paged.REQ)
                  + (() if p.sel_ch else (self.sel,)),
                  key=(p,) + ("chunk",), row_tasks=p.tiles * NCH * p.dsplit),
            Stage(f"{n}.m", rt * (p.mgroups if p.smerge else 1) * (HD // SLICE), m_emit, lds_words=m_lds,
                  writes=m_writes,
                  args=m_args,
                  key=(p,) + ("merge",), row_tasks=p.tiles * (p.mgroups if p.smerge else 1) * (HD // SLICE)),
        ]  # fmt: skip
