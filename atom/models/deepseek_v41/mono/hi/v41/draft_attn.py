# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark draft stages' attention: ``MLABlockAttn``, every row of a draft block attends the same span.

A draft block is ``spec`` rows (the anchor's next positions ``base = anchor + 1 .. anchor + spec``); every one of its
rows attends its request's window up to the anchor (positions ``anchor - 127 .. anchor``, from the engine's draft ring:
the rows the context write left there) and all ``spec`` block keys (from ``kvn``, both ways: not causal). The engine's
mono draft (ATOM ``step_meta.write_draft_step_meta``) lists the keys window first (position order), then the block;
the arithmetic here is the window attention's (``mla_window``: 64-key chunks, the reference's online softmax, bf16
probabilities in the numerator, fp32 in the denominator, the sink at the end, then the inverse RoPE and the QAT) with
a third chunk of the ``spec`` block keys: scores, the chunk's max / correction / probabilities and its P V by plain
fp32 FMAs (5 keys), the same operations as the reference's chunk step.

Per task (row, 16-head tile): q (RoPE at the row's position) and the window's 128 ring rows into LDS as the window
attention lands them, the block's keys into their own LDS rows, the block scores (before the window's scores overlay
q), then the window's two chunks (``mla_window``'s helpers), the block chunk, the finish and the output groups at the
row's position. No ring write: block keys are never read back (the next step's context write owns those positions).

Args: ``pos`` int32 [batch] (each request's first block position, anchor + 1), ``rope``, ``sink``, ``ring`` (the
stage's ring base: the engine's pool at the layer's window start), ``geo`` int32 [8] (num_slots, slot_rows,
ring_slots, ring_stride, run_rows: ``row = slot * slot_rows + (q // ring_stride) * run_rows + q % ring_stride``, q =
position % ring_slots) and ``slot`` int32 [batch] (each request's slot; one outside the pool reads slot 0's rows, the
engine's padding rule: such a request's drafts are discarded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..blocks.base import Block, Channel, Fmt
from ..blocks.gfx950.attention import mla_window as mw
from ..device.gfx950 import buffer_ops as bo
from ..device.gfx950 import mailbox as mb
from ..device.gfx950.ops import (
    bf2_f32,
    bf16_round,
    ld_i32,
    rope_sin,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from ..plan.program import Scenario, Stage

THREADS, WAVES = mw.THREADS, mw.WAVES
HD, TILE, PITCH, WINDOW, CH = mw.HD, mw.TILE, mw.PITCH, mw.WINDOW, mw.CH
BKEYS = 8  # LDS slots a head's block scores / probabilities take (>= spec)


@dataclass(frozen=True)
class _BPlan(mw._APlan):
    ring_arg: str = ""
    geo: str = ""
    slot: str = ""

    @property
    def a_b(self):  # the block's key rows [spec][PITCH]
        return -(-(self.a_rope + mw.ROPE) // 4) * 4

    @property
    def a_s2(self):  # the block scores [TILE][BKEYS] f32 (raw q . k)
        return self.a_b + self.spec * PITCH

    @property
    def a_p2(
        self,
    ):  # the block probabilities [TILE][BKEYS] (f32 of the bf16-rounded values)
        return self.a_s2 + TILE * BKEYS

    @property
    def a_c2(self):  # the block chunk's correction [TILE]
        return self.a_p2 + TILE * BKEYS

    @property
    def lds_words(self):
        return -(-(self.a_c2 + TILE) // 4) * 4


def _ring_issue(k, p: _BPlan, slot, pos_w):
    """``mla_window._ring_issue`` from the engine's ring: the window's keys ``pos_w - 127 .. pos_w`` at their rows
    (a position below 0 reads a clamped row, dropped by ``_key_land``)."""
    g = k.args[p.geo]
    slot_rows, ring_slots = uniform(ld_i32(g, 1)), uniform(ld_i32(g, 2))
    stride, run = uniform(ld_i32(g, 3)), uniform(ld_i32(g, 4))
    r_ring = rsrc(k.args[p.ring_arg])
    out = []
    for i in range(WINDOW // WAVES):
        kp = pos_w - (WINDOW - 1) + k.wave * (WINDOW // WAVES) + i
        q = ((kp % ring_slots) + ring_slots) % ring_slots
        row = slot * slot_rows + (q // stride) * run + q % stride
        out.append(
            fx.Vector(
                bo.buffer_load(
                    r_ring, row * (HD // 2) + k.lane * 4, vec_width=4, dtype=T.i32
                )
            )
        )
    return out


@traced
def _block_keys(k, p: _BPlan, b):
    """The request's ``spec`` block keys (``kvn`` rows ``b * spec ..``) into LDS rows ``a_b``."""
    for j in range_constexpr(-(-p.spec * (HD // 2) // (THREADS * 2))):
        w = j * (THREADS * 2) + k.tid * 2
        row, dw = w // (HD // 2), w % (HD // 2)
        if row < p.spec:
            got = mb.poll(
                [(k.mb[p.kvn], (b * p.spec + row) * (HD // 2) + dw, 2)], k.tag
            )[0]
            fx.ptr_store(
                fx.Vector.from_elements(got, fx.Int32).bitcast(fx.Float32),
                k.ldsf + (p.a_b + row * PITCH + dw),
            )


def _scores_block(k, p: _BPlan):
    """S2[head][key] = q . k (raw, f32) of the 16 heads and the block's keys: 4 lanes a (head, key), 128 dims each
    (fp32 products of the bf16 pairs in dim order), the 4 parts summed ((0 + 1) + (2 + 3)); before the window's scores
    overlay q."""
    pair, part = k.tid // 4, k.tid % 4
    pc = fx.min(pair, fx.Int32(TILE * p.spec - 1))
    hh, key = pc // p.spec, pc % p.spec
    qa, ka = hh * PITCH + part * (HD // 8), p.a_b + key * PITCH + part * (HD // 8)
    acc = fx.Float32(0.0)
    for w in range_constexpr(HD // 8):
        qw = fx.ptr_load(k.ldsf + (qa + w)).bitcast(fx.Int32)
        kw = fx.ptr_load(k.ldsf + (ka + w)).bitcast(fx.Int32)
        q0, q1 = bf2_f32(qw)
        k0, k1 = bf2_f32(kw)
        acc = acc + q0 * k0
        acc = acc + q1 * k1
    acc = acc + xshfl(acc, 1)
    acc = acc + xshfl(acc, 2)
    _s2_store(k, p, pair, part, hh, key, acc)


@traced
def _s2_store(k, p: _BPlan, pair, part, hh, key, acc):
    if (part == 0) & (pair < TILE * p.spec):
        fx.ptr_store(acc, k.ldsf + (p.a_s2 + hh * BKEYS + key))


@traced
def _softmax_block(k, p: _BPlan):
    """Thread h < 16: the block chunk's step of the online softmax after the window's two (``mla_window._ml_store``'s
    max / denominator): the chunk's max, the correction, the probabilities (bf16 into ``a_p2``), the denominator; the
    final max and denominator back where ``_finish`` reads them, the correction into ``a_c2``.
    """
    if k.tid < TILE:
        hh = k.tid
        scale = fx.Float32(1.0 / math.sqrt(HD))
        m_prev = fx.ptr_load(k.ldsf + (p.a_ml + hh))
        den = fx.ptr_load(k.ldsf + (p.a_ml + TILE + hh))
        s = [
            fx.ptr_load(k.ldsf + (p.a_s2 + hh * BKEYS + j)) * scale
            for j in range(p.spec)
        ]
        mx = s[0]
        for v in s[1:]:
            mx = fx.max(mx, v)
        m_new = fx.max(m_prev, mx)
        corr = fx.Float32(fmath.exp(m_prev - m_new))
        pv = [fx.Float32(fmath.exp(v - m_new)) for v in s]
        lsum = pv[0]
        for v in pv[1:]:
            lsum = lsum + v
        for j in range_constexpr(p.spec):
            fx.ptr_store(bf16_round(pv[j]), k.ldsf + (p.a_p2 + hh * BKEYS + j))
        fx.ptr_store(m_new, k.ldsf + (p.a_ml + hh))
        fx.ptr_store(den * corr + lsum, k.ldsf + (p.a_ml + TILE + hh))
        fx.ptr_store(corr, k.ldsf + (p.a_c2 + hh))


def _pv_block(k, p: _BPlan, acc):
    """``mla_window._pv_all``'s outputs (dim, the 4 heads' values) with the block chunk's step: v corr2 + sum_j
    bf16(P2[h][j]) V2[j][dim] (fp32, keys in order)."""
    ld = k.lane // 16
    heads = [ld * 4 + e for e in range(4)]
    corr = [fx.ptr_load(k.ldsf + (p.a_c2 + h)) for h in heads]
    probs = [
        [fx.ptr_load(k.ldsf + (p.a_p2 + h * BKEYS + j)) for j in range(p.spec)]
        for h in heads
    ]
    out = []
    for dim, vals in acc:
        vw = [
            fx.ptr_load(k.ldsf + (p.a_b + j * PITCH + dim // 2)).bitcast(fx.Int32)
            for j in range(p.spec)
        ]
        lo = (dim % 2) == 0
        vv = []
        for w in vw:
            a, b = bf2_f32(w)
            vv.append(
                lo.select(a, b)
                if not isinstance(dim, int)
                else (a if dim % 2 == 0 else b)
            )
        new = []
        for e in range(4):
            t = vals[e] * corr[e]
            for j in range(p.spec):
                t = t + probs[e][j] * vv[j]
            new.append(t)
        out.append((dim, new))
    return out


def _attn(k, t, p: _BPlan):
    r, ht = t // p.tiles, t % p.tiles
    b, base, pos = mw._row_pos(k, p.pos, p.spec, r)
    pos_w = base - 1  # the window ends at the anchor for every row of the block
    slot = uniform(ld_i32(k.args[p.slot], b))
    nslots = uniform(ld_i32(k.args[p.geo], 0))
    slot = ((slot >= 0) & (slot < nslots)).select(slot, fx.Int32(0))
    sinks = mw._sink_issue(k, p, ht)
    rcs = mw._rope_issue(k, p, pos)
    mw._q_tile(k, p, r, ht, pos)
    mw._rope_land(k, p, rcs)
    mw._key_land(k, p, pos_w, _ring_issue(k, p, slot, pos_w), base)
    _block_keys(k, p, b)
    gpu.barrier()
    _scores_block(k, p)
    gpu.barrier()
    mw._scores_all(k, p)
    gpu.barrier()
    mw._softmax_all(k, p, pos_w)
    gpu.barrier()
    _softmax_block(k, p)
    gpu.barrier()
    acc = _pv_block(k, p, mw._pv_all(k, p, 0))
    gpu.barrier()  # o overwrites the keys
    mw._finish(k, p, acc, sinks)
    gpu.barrier()
    mw._out_groups(k, p, r, ht, pos, 0)
    gpu.barrier()  # the next task reuses the LDS


class MLABlockAttn(Block):
    """The draft block's attention (module docstring); ``heads``: this rank's (16-head tiles, a ragged last one as
    ``MLAWindowAttn``'s: 8 a rank at TP8)."""

    def __init__(self, name: str, q: Channel, kvn: Channel, out: Channel, pos: str, rope: str, sink: str, ring: str,
                 geo: str, slot: str, heads: int = mw.HEADS, unrope_f32: bool = False):  # fmt: skip
        if heads < 1:
            raise ValueError(f"MLABlockAttn {name}: heads {heads}")
        if q.fmt is not Fmt.BF16 or q.width != heads * HD or q.flagged:
            raise ValueError(
                f"MLABlockAttn {name}: q is a tagged BF16 channel of width {heads * HD}"
            )
        if kvn.fmt is not Fmt.BF16 or kvn.width != HD or kvn.flagged:
            raise ValueError(
                f"MLABlockAttn {name}: kvn is a tagged BF16 channel of width {HD}"
            )
        if out.fmt is not Fmt.MXFP8 or out.width != heads * HD or out.flagged:
            raise ValueError(
                f"MLABlockAttn {name}: out is a tagged MXFP8 channel of width {heads * HD}"
            )
        self.name, self.q, self.kvn, self.out, self.heads = name, q, kvn, out, heads
        self.pos, self.rope, self.sink, self.ring, self.geo, self.slot = (
            pos,
            rope,
            sink,
            ring,
            geo,
            slot,
        )
        # the output's inverse RoPE quantized from f32 (``MLAWindowAttn(unrope_f32=)``: ATOM's order, the draft's K2a
        # irq path); False: bf16-rounded first (the official order). A numerics choice, not a schedule knob
        self.unrope_f32 = unrope_f32

    def plan(self, sc: Scenario) -> _BPlan:
        if not 1 <= sc.spec <= BKEYS:
            raise ValueError(
                f"MLABlockAttn {self.name}: a block of 1 .. {BKEYS} rows, got {sc.spec}"
            )
        return _BPlan(self.name, sc.rows, sc.spec, self.q.name, self.kvn.name, self.ring, self.pos, self.rope,
                      self.sink, self.out.name, f"{self.name}.done", self.heads, WINDOW, False, 1,
                      unrope_f32=self.unrope_f32, ring_arg=self.ring, geo=self.geo, slot=self.slot)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        tasks = sc.rows * p.tiles
        writes = tuple(self.out.regions(sc.rows, (self.name,)))
        args = (
            self.pos,
            self.ring,
            self.rope,
            rope_sin(self.rope),
            self.sink,
            self.geo,
            self.slot,
        )
        return [Stage(self.name, tasks, partial(_attn, p=p), lds_words=p.lds_words, writes=writes, args=args,
                      key=(p,))]  # fmt: skip
