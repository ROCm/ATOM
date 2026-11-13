# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's sliding-window MLA decode (a window-only layer's attention core; checkpoint
``inference/model.py`` ``Attention.forward`` with ``compress_ratio`` 0, ``sparse_attn`` as the reference).

Three blocks, between the projections and ``wo_a``:

``KVPrep``: the step rows' new window keys: ``kv_norm`` output (a BF16 channel, 512) -> RoPE on the last 64 dims
(adjacent pairs as complex numbers, at the row's position) -> the FP8 QAT (``act_quant(inplace)``: MXFP8 group 32
quantize and dequantize) -> a BF16 channel ``kvn``.

``MLAWindowAttn``: a task per (row, 16-head tile). The row at position P attends to the 128 positions P - 127 .. P
(oldest first, as the reference lists the ring): those before the step's first position from the pre-step
ring (plain bf16 [batch, 128, 512], slot = position % 128), the step's own from ``kvn``, positions below 0
masked. MQA latent attention: K = V = the 512-dim key rows, 64 query heads. The reference's arithmetic: two
chunks of 64 keys in order, a running max (initially -1e30), p = exp(s - max) with s = q . k / sqrt(512), the
numerator rescaled and accumulated with bf16(p), the denominator with fp32 p, and the attention sink added at the
end (den + exp(sink - max)); o = bf16(num / den). Then the inverse RoPE on o's last 64 dims and the QAT before
``wo_a`` (MXFP8 group 32): the output is an MXFP8 channel of width 64 x 512. Scores and P V on the bf16 MFMA
(D = 512): q . k sums in another order than torch's, nothing else.

``RingWrite``: after every attention task of the request has read the ring, the step rows' keys go into it
(slot = position % 128). A row of a multi-row step sees the old contents of the slots its later rows overwrite,
so the write cannot come first.

Args: ``pos`` int32 [batch] (the step's first position per request), ``rope`` f32 [max_pos, 64] (cos, sin of
pair i at 2 i, 2 i + 1: the reference's ``freqs_cis`` as real pairs, from the CPU table), ``sink`` f32 [64].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ....device.gfx950 import buffer_ops as bo
from ....device.gfx950 import channels as chan
from ....device.gfx950 import gemv
from ....device.gfx950 import mailbox as mb
from ....device.gfx950.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    div_rn,
    ld_i32,
    mfma_bf16,
    rope_cs,
    rope_sin,
    rsrc,
    traced,
    uniform,
)
from ....plan.program import Region, Scenario, Stage
from ....target import MI355X
from ...base import Block, Channel, Fmt

THREADS, WAVES = MI355X.threads, MI355X.waves
HEADS, HD, ROPE, WINDOW = 64, 512, 64, 128
CH = 64  # keys a chunk (the reference's sparse_attn block)
TILE = 16  # heads a task (the MFMA's N)
PITCH = HD // 2 + 4  # LDS words a q / key row (bf16 pairs, +4: bank spread)
SPITCH = WINDOW + 4  # LDS words a score row (f32, both chunks)
PPITCH = WINDOW // 2 + 4  # LDS words a P row (bf16 pairs, both chunks)
NEG_INIT = -1.0e30  # the reference's initial running max
TINY_BITS = 37 << 23  # |x| below 2^-90 (fp32 bits): _finish divides exactly
HUGE_BITS = (
    187 << 23
)  # a denominator from 2^60 (fp32 bits, positive): _finish divides exactly
OGP = 33  # LDS words a 32-dim group of o (+1: the output threads' 32-word strides on distinct banks)
OPITCH = (HD // 32) * OGP  # LDS words an o row
assert (
    TILE * (HD // 32) * 2 == THREADS
)  # the out phase: two threads a (head, 32-dim group)


def _ab(x):
    return fx.max(x, -x)


# ------------------------------------------------------------------ shared device helpers


def _fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def _rope_pair(re, im, c, s, inverse):
    """(re + i im) times (c + i s), or its conjugate, as PyTorch's GPU complex multiply rounds it (measured on 1M
    samples): real = fma(re, c, -(im s')), imag = fma(im, c, re s'), s' = s (or -s for the conjugate). The CPU
    product rounds each term and differs in the last bit: the hw reference runs on the GPU.
    """
    s2 = -s if inverse else s
    return _fma(re, c, -(im * s2)), _fma(im, c, re * s2)


def _qat_word(vals, code):
    """Four values of one MX group -> (the E4M3 word, the four dequantized values: the QAT's)."""
    word = gemv.quant_word(vals, code)
    scale = (code << 23).bitcast(fx.Float32)
    back = [
        fx.Float32(rocdl.cvt_f32_fp8(T.f32, fx.Int32(word).ir_value(), i)) * scale
        for i in range(4)
    ]
    return word, back


def _group_code(vals):
    am = _ab(vals[0])
    for v in vals[1:]:
        am = fx.max(am, _ab(v))
    return gemv.activation_code(am)


def _rope_given(vals, cs):
    """RoPE of ``vals`` (adjacent pairs) with their pairs' (cos, sin) ``cs`` already loaded ."""
    out = []
    for i in range(len(vals) // 2):
        a, b = _rope_pair(
            vals[2 * i],
            vals[2 * i + 1],
            fx.Float32(cs[i][0]),
            fx.Float32(cs[i][1]),
            False,
        )
        out += [bf16_round(a), bf16_round(b)]
    return out


def _row_pos(k, pos_arg, spec, r):
    """(request, its first position, this row's position)."""
    b = r // spec
    base = uniform(ld_i32(k.args[pos_arg], b))
    return b, base, base + r % spec


# ------------------------------------------------------------------ KVPrep


@dataclass(frozen=True)
class _KVPlan:
    name: str
    rows: int
    spec: int
    kv: str  # BF16 channel (512): kv_norm's output
    kvn: str  # BF16 channel (512): out
    pos: str
    rope: str


@traced
def _kvprep(k, r, p: _KVPlan):
    """Thread g < 16: the row's 32-dim group g."""
    if k.tid < HD // 32:
        g = k.tid
        _, _, pos = _row_pos(k, p.pos, p.spec, r)
        e0 = r * HD + g * 32
        # the RoPE pairs of the rotary groups issued before the poll (every thread: clamped, unused below them)
        first = fx.max(g * 32 - (HD - ROPE), 0) // 2
        cs = [rope_cs(k, p.rope, pos, first + i) for i in range(16)]
        got = mb.poll([(k.mb[p.kv], e0 // 2 + 2 * j, 2) for j in range(8)], k.tag)
        vals = [v for w in got for word in w for v in bf2_f32(word)]
        rot = vals  # (assigned before the dynamic if: its branches only rebind it)
        if g >= (HD - ROPE) // 32:
            rot = _rope_given(vals, cs)
        code = _group_code(rot)
        back = []
        for w in range_constexpr(8):
            back += _qat_word(rot[4 * w : 4 * w + 4], code)[1]
        for j in range_constexpr(8):
            chan.put_bf16(
                k.mb[p.kvn],
                e0 + 4 * j,
                [bf16_round(v) for v in back[4 * j : 4 * j + 4]],
                k.tag,
            )


class KVPrep(Block):
    def __init__(self, name: str, kv: Channel, kvn: Channel, pos: str, rope: str):
        for ch in (kv, kvn):
            if ch.fmt is not Fmt.BF16 or ch.width != HD or ch.flagged:
                raise ValueError(
                    f"KVPrep {name}: kv / kvn are tagged BF16 channels of width {HD}"
                )
        self.name, self.kv, self.kvn, self.pos, self.rope = name, kv, kvn, pos, rope

    def stages(self, sc: Scenario) -> list[Stage]:
        p = _KVPlan(
            self.name,
            sc.rows,
            sc.spec,
            self.kv.name,
            self.kvn.name,
            self.pos,
            self.rope,
        )
        return [Stage(self.name, sc.rows, partial(_kvprep, p=p), writes=tuple(self.kvn.regions(sc.rows, (self.name,))),
                      args=(self.pos, self.rope, rope_sin(self.rope)), key=(p,), row_tasks=1)]  # fmt: skip


# ------------------------------------------------------------------ MLAWindowAttn


@dataclass(frozen=True)
class _APlan:
    name: str
    rows: int
    spec: int
    q: str  # BF16 channel (64 x 512): wq_b's output, before RoPE
    kvn: str  # BF16 channel (512)
    ring: str  # bf16 [batch, ring_slots, 512], position p at slot p % ring_slots
    pos: str
    rope: str
    sink: str
    out: str  # MXFP8 channel (64 x 512)
    done: str  # region: a pair a task, put once the task has read the ring
    heads: int = HEADS  # this rank's query heads
    ring_slots: int = (
        WINDOW  # the ring's slots (WINDOW + drafts: a rejected draft row overwrites no live key)
    )
    ring_first: bool = (
        False  # the ring rows into LDS before the q poll (they need no input: a task that starts before
    )
    #                           its q lands has them in by then; module docstring of ``MLAWindowAttn``)
    dsplit: int = (
        1  # tasks a (row, tile): each the scores and softmax of all 512 dims, then the P V, finish and
    )
    #                  output of its 512 / dsplit dims (the same bits: those are per output element)
    unrope_f32: bool = (
        False  # the output's inverse RoPE quantized from f32 (no bf16 rounding of the 64 rotary dims)
    )

    @property
    def tiles(self):
        """16-head tiles a row. A ragged last tile (heads not a multiple of 16, e.g. 8 at TP8) runs its dead heads on
        the last live head's q and sink and writes them there again: the same values, the same bits.
        """
        return -(-self.heads // TILE)

    @property
    def ragged(self):
        return self.heads % TILE != 0

    def head(self, ht, *hh):
        """The live head a tile's head (the sum of ``hh``, added in order: a 16-head program's arithmetic as before)
        computes: itself, or (ragged) at most the last head."""
        h = ht * TILE
        for x in hh:
            h = h + x
        return fx.min(h, fx.Int32(self.heads - 1)) if self.ragged else h

    @property
    def a_k(
        self,
    ):  # both chunks' keys [WINDOW][PITCH], after the q tile [TILE][PITCH] at 0
        return TILE * PITCH

    @property
    def a_s(
        self,
    ):  # S [TILE][SPITCH] f32, over the q tile (stored once every wave has read q)
        return 0

    @property
    def a_p(self):
        return self.a_k + WINDOW * PITCH

    @property
    def a_ml(self):  # final m, den, then corr0 / corr1, a head each
        return self.a_p + TILE * PPITCH

    @property
    def a_rope(
        self,
    ):  # (cos, sin) of the row's 32 rotary pairs (the output's inverse RoPE), loaded at task start
        return self.a_ml + 4 * TILE

    @property
    def a_o(self):  # o f32 [TILE][OPITCH] after the P V, over the keys (``o_word``)
        return self.a_k

    def o_word(self, hh, dim):
        return self.a_o + hh * OPITCH + (dim // 32) * OGP + dim % 32

    @property
    def lds_words(self):
        return -(-(self.a_rope + ROPE) // 4) * 4


def _q_tile(k, p: _APlan, r, ht, pos):
    """q of the tile's 16 heads into LDS [head][PITCH], RoPE'd. Load j of a thread: words ``1024 j + 2 tid ..
    + 2`` of the tile (4 dims; consecutive lanes, consecutive 16 B: a lane's 64 B of one head an instruction made 4x
    the L2 requests); the RoPE table's (cos, sin) of those dims issued before the poll.
    """
    e0w = (r * p.heads * HD + ht * TILE * HD) // 2
    ws = [j * (THREADS * 2) + k.tid * 2 for j in range(TILE * HD // 2 // (THREADS * 2))]
    css = []
    for w in ws:
        first = fx.max(
            (w % (HD // 2)) - (HD - ROPE) // 2, 0
        )  # the rotary pair of the word's first two dims
        css.append([rope_cs(k, p.rope, pos, first + i) for i in range(2)])
    if (
        not p.ragged
    ):  # the tile's words (in this order: the 16-head programs' code as before)
        got = mb.poll([(k.mb[p.q], e0w + w, 2) for w in ws], k.tag)
    else:  # a dead head's words are its tile's last live head's (``_APlan.head``)
        r0w = (r * p.heads * HD) // 2
        got = mb.poll(
            [
                (k.mb[p.q], r0w + p.head(ht, w // (HD // 2)) * (HD // 2) + w % (HD // 2), 2)
                for w in ws
            ],
            k.tag,
        )
    for j, w in enumerate(ws):
        hh, dw = w // (HD // 2), w % (HD // 2)
        vals = [v for word in got[j] for v in bf2_f32(word)]
        is_rope = dw * 2 >= HD - ROPE
        out = []
        for i in range(2):
            a, b = _rope_pair(
                vals[2 * i],
                vals[2 * i + 1],
                fx.Float32(css[j][i][0]),
                fx.Float32(css[j][i][1]),
                False,
            )
            out += [
                is_rope.select(bf16_round(a), vals[2 * i]),
                is_rope.select(bf16_round(b), vals[2 * i + 1]),
            ]
        for i in range(2):
            fx.ptr_store(
                bf16_pair(out[2 * i], out[2 * i + 1]), k.ldsf + (hh * PITCH + dw + i)
            )


def _ring_issue(k, p: _APlan, b, pos):
    """The window's ring rows, one key row (1 KB) a wave instruction: wave w loads keys 16 w .. + 16, lane l bytes
    16 l .. + 16 of each (a lane's 16 B pieces of 8 rows an instruction made 8x the L2 requests: ~4 µs). Rows outside
    the ring part (negative or the step's own positions) are loaded from a clamped slot and dropped in
    ``_key_land``."""
    r_ring = rsrc(k.args[p.ring])
    out = []
    for i in range(WINDOW // WAVES):
        kp = pos - (WINDOW - 1) + k.wave * (WINDOW // WAVES) + i
        slot = ((kp % p.ring_slots) + p.ring_slots) % p.ring_slots
        out.append(fx.Vector(bo.buffer_load(r_ring, ((b * p.ring_slots + slot) * HD) // 2 + k.lane * 4, vec_width=4,
                                            dtype=T.i32)))  # fmt: skip
    return out


def _key_land(k, p: _APlan, pos, vs, base):
    """The ring rows of ``_ring_issue`` into LDS [key][PITCH] (zero below position 0 and at the step's own
    positions, which ``_own_keys`` writes)."""
    for i in range(WINDOW // WAVES):
        key = k.wave * (WINDOW // WAVES) + i
        kp = pos - (WINDOW - 1) + key
        ok = (kp >= 0) & (kp < base)
        words = [ok.select(vs[i][e], fx.Int32(0)) for e in range(4)]
        fx.ptr_store(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float32),
                     k.ldsf + (p.a_k + key * PITCH + k.lane * 4))  # fmt: skip


@traced
def _own_keys(k, p: _APlan, b, base, pos):
    """The step's own key rows (positions base .. pos, from ``kvn``) over their ring slots in LDS. Load j of a
    thread: words ``1024 j + 2 tid .. + 2`` of the request's rows (consecutive lanes, consecutive 16 B).
    """
    for j in range_constexpr(-(-p.spec * (HD // 2) // (THREADS * 2))):
        w = j * (THREADS * 2) + k.tid * 2
        row, dw = w // (HD // 2), w % (HD // 2)
        if base + row <= pos:
            got = mb.poll(
                [(k.mb[p.kvn], (b * p.spec + row) * (HD // 2) + dw, 2)], k.tag
            )[0]
            g = base + row - (pos - (WINDOW - 1))  # the key's index in the window
            fx.ptr_store(
                fx.Vector.from_elements(got, fx.Int32).bitcast(fx.Float32),
                k.ldsf + (p.a_k + g * PITCH + dw),
            )


def _lds_bf8(k, word):
    return fx.ptr_load(
        k.ldsf + word, result_type=fx.Vector.make_type(4, fx.Float32)
    ).bitcast(fx.BFloat16)


def _scores_all(k, p: _APlan):
    """S[head][key] (unscaled f32) of all 128 keys: wave w takes keys 16 w .. + 16 (each key's MFMA chain as in
    one chunk). S overlays the q tile, so it is stored after every wave's last q read.
    """
    lm, ld = k.lane % 16, k.lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for kst in range_constexpr(HD // 32):
        a = _lds_bf8(k, p.a_k + (k.wave * 16 + lm) * PITCH + kst * 16 + ld * 4)
        bq = _lds_bf8(k, lm * PITCH + kst * 16 + ld * 4)
        acc = mfma_bf16(a, bq, acc)
    gpu.barrier()
    for e in range_constexpr(4):
        fx.ptr_store(acc[e], k.ldsf + (p.a_s + lm * SPITCH + k.wave * 16 + ld * 4 + e))


def _softmax_all(k, p: _APlan, pos):
    """Thread -> head ``tid // 32``, keys ``(tid % 32) * 2 .. + 2`` of each chunk: the reference's two chunk steps
    in order (running max from -1e30, P (bf16), the denominator), chunk 1 from chunk 0's max and denominator in
    registers: the same operations, in the same order, as one chunk at a time."""
    hh, kk = k.tid // 32, (k.tid % 32) * 2
    scale = fx.Float32(1.0 / math.sqrt(HD))
    m_prev, den, corrs = fx.Float32(NEG_INIT), fx.Float32(0.0), []
    for c in range(WINDOW // CH):
        sv = []
        for i in range_constexpr(2):
            kp = pos - (WINDOW - 1) + c * CH + kk + i
            raw = fx.ptr_load(k.ldsf + (p.a_s + hh * SPITCH + c * CH + kk + i)) * scale
            sv.append((kp >= 0).select(raw, fx.Float32(-math.inf)))
        mx = butterfly(fx.max(sv[0], sv[1]), (1, 2, 4, 8, 16), fx.max)
        m_new = fx.max(m_prev, mx)
        corr = fx.Float32(fmath.exp(m_prev - m_new))
        pv = [fx.Float32(fmath.exp(v - m_new)) for v in sv]
        lsum = butterfly(pv[0] + pv[1], (1, 2, 4, 8, 16))
        fx.ptr_store(
            bf16_pair(pv[0], pv[1]), k.ldsf + (p.a_p + hh * PPITCH + (c * CH + kk) // 2)
        )
        den = den * corr + lsum
        corrs.append(corr)
        m_prev = m_new
    _ml_store(k, p, hh, m_prev, den, corrs[0], corrs[1])


@traced
def _ml_store(k, p: _APlan, hh, m, den, corr0, corr1):
    if k.tid % 32 == 0:
        fx.ptr_store(m, k.ldsf + (p.a_ml + hh))
        fx.ptr_store(den, k.ldsf + (p.a_ml + TILE + hh))
        fx.ptr_store(corr0, k.ldsf + (p.a_ml + 2 * TILE + hh))
        fx.ptr_store(corr1, k.ldsf + (p.a_ml + 3 * TILE + hh))


def _pv_all(k, p: _APlan, d0=0):
    """num = (0 corr0 + bf16(P0) V0) corr1 + bf16(P1) V1 (the reference's two chunk steps), dims ``d0`` .. ``d0`` +
    512 / dsplit: wave w holds 64 / dsplit of them for the 16 heads, both chunks' MFMAs in one pass. Each 32-dim block
    runs two MFMAs, lane m's column being dim 2 m in one and 2 m + 1 in the other, so the 8 LDS words a lane loads (a
    word: two dims of a key) feed both (16 dims a wave: one block, the wave's parity only). -> [(dim, the 4 values of
    heads 4 (lane // 16) .. + 4)]."""
    lm, ld = k.lane % 16, k.lane // 16
    corr = [
        [fx.ptr_load(k.ldsf + (p.a_ml + (2 + c) * TILE + ld * 4 + e)) for e in range(4)]
        for c in range(2)
    ]
    dpw = HD // p.dsplit // WAVES  # dims a wave: 64, 32 or 16
    if dpw >= 32:
        blocks, pars = [k.wave * dpw + b * 32 + d0 for b in range(dpw // 32)], [0, 1]
    out = []
    for base in blocks:
        dword = base // 2 + lm
        ts = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(WINDOW // CH)]
            for _ in pars
        ]  # [parity][chunk]
        for c in range(WINDOW // CH):
            for kc in range(CH // 32):
                kst = c * (CH // 32) + kc
                a = _lds_bf8(k, p.a_p + lm * PPITCH + kst * 16 + ld * 4)
                ws = [
                    fx.ptr_load(
                        k.ldsf + (p.a_k + (kst * 32 + ld * 8 + i) * PITCH + dword)
                    ).bitcast(fx.Int32)
                    for i in range(8)
                ]
                halves = (
                    [w & fx.Int32(0xFFFF) for w in ws],
                    [fx.Int32(fx.Uint32(w) >> 16) for w in ws],
                )
                for j, par in enumerate(pars):
                    hv = halves[par]
                    bw = [hv[2 * i] | (hv[2 * i + 1] << fx.Int32(16)) for i in range(4)]
                    ts[j][c] = mfma_bf16(
                        a,
                        fx.Vector.from_elements(bw, fx.Int32).bitcast(fx.BFloat16),
                        ts[j][c],
                    )
        for j, par in enumerate(pars):
            t0, t1 = ts[j]
            vals = [
                (fx.Float32(0.0) * corr[0][e] + t0[e]) * corr[1][e] + t1[e]
                for e in range(4)
            ]
            out.append((base + 2 * lm + (k.wave % 2 if par is None else par), vals))
    return out


def _rope_issue(k, p: _APlan, pos):
    """(cos, sin) of rotary pair ``min(tid, 31)`` at the row's position (``_rope_land`` stores threads < 32's)."""
    return rope_cs(k, p.rope, pos, fx.min(k.tid, fx.Int32(ROPE // 2 - 1)))


@traced
def _rope_land(k, p: _APlan, cs):
    if k.tid < ROPE // 2:
        fx.ptr_store(fx.Float32(cs[0]), k.ldsf + (p.a_rope + 2 * k.tid))
        fx.ptr_store(fx.Float32(cs[1]), k.ldsf + (p.a_rope + 2 * k.tid + 1))


def _sink_issue(k, p: _APlan, ht):
    """This lane's four heads' sink values (``_finish``), loaded at task start."""
    r_sink = rsrc(k.args[p.sink])
    return [fx.Float32(bo.buffer_load(r_sink, p.head(ht, (k.lane // 16) * 4, e), vec_width=1, dtype=T.f32))
            for e in range(4)]  # fmt: skip


def _finish(k, p: _APlan, acc, sinks):
    """o = bf16(num / (den + exp(sink - max))) into LDS f32 [head][dim] (``acc``: ``_pv_all``'s). One true divide a
    head (r = 1 / den), then each quotient by ``div_rn`` (q = x r and one fma correction: the correctly rounded
    quotient, so the bits of ``x / den``, while the remainder cannot underflow); a lane with a numerator below 2^-90
    (or zero: the sign of -0) or a denominator from 2^60 divides every value exactly."""
    ld = k.lane // 16
    dens, rs = [], []
    for e in range(4):
        hh = ld * 4 + e
        m = fx.ptr_load(k.ldsf + (p.a_ml + hh))
        den = fx.ptr_load(k.ldsf + (p.a_ml + TILE + hh))
        den = den + fx.Float32(fmath.exp(sinks[e] - m))
        dens.append(den)
        rs.append(fx.Float32(1.0) / den)
    small = None
    for _, vals in acc:
        for e in range(4):
            b = vals[e].bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
            small = b if small is None else fx.min(small, b)
    big = None  # den >= 2^60 (an exp(sink - max) near overflow): exact as well
    for den in dens:
        d = den.bitcast(fx.Int32)
        big = d if big is None else fx.max(big, d)
    fast = (small >= fx.Int32(TINY_BITS)) & (big < fx.Int32(HUGE_BITS))
    words = [[p.o_word(ld * 4 + e, dim) for e in range(4)] for dim, _ in acc]
    _finish_store(
        k,
        fast,
        [[div_rn(v[e], dens[e], rs[e]) for e in range(4)] for _, v in acc],
        words,
    )
    _finish_exact(
        k,
        (small < fx.Int32(TINY_BITS)) | (big >= fx.Int32(HUGE_BITS)),
        acc,
        dens,
        words,
    )


@traced
def _finish_store(k, ok, qs, words):
    if ok:
        for q, ws in zip(qs, words):
            for e in range_constexpr(4):
                fx.ptr_store(bf16_round(q[e]), k.ldsf + ws[e])


@traced
def _finish_exact(k, ok, acc, dens, words):
    if ok:
        for (_, vals), ws in zip(acc, words):
            for e in range_constexpr(4):
                fx.ptr_store(bf16_round(vals[e] / dens[e]), k.ldsf + ws[e])


def _unrope_lds(k, p: _APlan, first, vals):
    """The inverse RoPE of ``vals`` (adjacent pairs from rotary pair ``first``) from the LDS table (``_rope_land``)."""
    out = []
    for i in range(len(vals) // 2):
        c = fx.ptr_load(k.ldsf + (p.a_rope + 2 * (first + i)))
        sn = fx.ptr_load(k.ldsf + (p.a_rope + 2 * (first + i) + 1))
        a, b = _rope_pair(vals[2 * i], vals[2 * i + 1], c, sn, True)
        out += [a, b] if p.unrope_f32 else [bf16_round(a), bf16_round(b)]
    return out


@traced
def _out_groups(k, p: _APlan, r, ht, pos, g0=0):
    """Per (head, 32-dim group of the task's 16 / dsplit from ``g0``) 2 dsplit threads, 16 / dsplit dims each (part
    ``tid % (2 dsplit)``): the inverse RoPE on the last 64 dims, the group's code from the parts' |max| (the lanes'
    max: exact), then the QAT: E4M3 words and (part 0) the E8M0 code into the MXFP8 channel.
    """
    tpg = 2 * p.dsplit  # threads a group
    dpt = 32 // tpg  # dims a thread
    per_head = (HD // 32 // p.dsplit) * tpg
    hh, g, part = k.tid // per_head, (k.tid % per_head) // tpg, k.tid % tpg
    if const_expr(p.dsplit > 1):
        g = g + g0
    o0 = (
        p.a_o + hh * OPITCH + g * OGP + part * dpt
    )  # outside the dynamic if (no method calls on local objects there)
    vals = [fx.ptr_load(k.ldsf + (o0 + i)) for i in range(dpt)]
    rot = vals
    if g >= (HD - ROPE) // 32:
        rot = _unrope_lds(k, p, (g * 32 - (HD - ROPE)) // 2 + part * (dpt // 2), vals)
    am = _ab(rot[0])
    for v in rot[1:]:
        am = fx.max(am, _ab(v))
    # every lane: a group's threads are adjacent lanes (the butterfly stays within them)
    code = gemv.activation_code(
        butterfly(am, tuple(1 << i for i in range(tpg.bit_length() - 1)), fx.max)
    )
    e0 = r * p.heads * HD + p.head(ht, hh) * HD + g * 32
    for w in range_constexpr(dpt // 4):
        chan.put_mxfp8_word(k.mb[f"{p.out}.q"], e0 // 4 + part * (dpt // 4) + w,
                            _qat_word(rot[4 * w : 4 * w + 4], code)[0], k.tag)  # fmt: skip
    _scale_put(k, p, part == 0, e0, code)


@traced
def _scale_put(k, p: _APlan, ok, e0, code):
    if ok:
        chan.put_mxfp8_scale(k.mb[f"{p.out}.s"], e0 // 32, code, k.tag)


@traced
def _done(k, p: _APlan, t):
    if k.tid == 0:
        mb.put(k.mb[p.done], t, fx.Int32(1), k.tag)


def _ring_then_q(k, p: _APlan, r, ht, b, base, pos):
    """``ring_first``: the ring rows landed in LDS, then the q tile polled (vector memory returns in order, so a q
    poll issued with the ring in flight would wait behind it: the ring goes first, whole, while q is not there yet).
    """
    _key_land(k, p, pos, _ring_issue(k, p, b, pos), base)
    rocdl.sched_barrier(0)  # keep the q poll after the ring's landing
    _q_tile(k, p, r, ht, pos)


def _attn(k, t, p: _APlan):
    rt, j = (
        (t // p.dsplit, t % p.dsplit) if p.dsplit > 1 else (t, 0)
    )  # (row, tile), the task's dim part
    r, ht = rt // p.tiles, rt % p.tiles
    b, base, pos = _row_pos(k, p.pos, p.spec, r)
    sinks = _sink_issue(k, p, ht)
    rcs = _rope_issue(k, p, pos)
    if p.ring_first:
        _ring_then_q(k, p, r, ht, b, base, pos)
        _rope_land(k, p, rcs)
        gpu.barrier()  # every ring row (zeros at the step's own slots) is in before the own rows overwrite them
    _own_keys(k, p, b, base, pos)
    gpu.barrier()
    _done(k, p, t)  # the ring is read
    _scores_all(k, p)
    gpu.barrier()
    _softmax_all(k, p, pos)
    gpu.barrier()
    acc = _pv_all(k, p, j * (HD // p.dsplit) if p.dsplit > 1 else 0)
    gpu.barrier()  # o overwrites the keys
    _finish(k, p, acc, sinks)
    gpu.barrier()
    _out_groups(k, p, r, ht, pos, j * (HD // 32 // p.dsplit) if p.dsplit > 1 else 0)
    gpu.barrier()  # the next task reuses the LDS


class MLAWindowAttn(Block):
    """``heads``: this rank's query heads (64 at TP1; TP shards them, 16 a rank at TP4, 8 at TP8): 16-head tiles, the
    last one ragged when heads is not a multiple of 16 (``_APlan.tiles``)."""

    def __init__(self, name: str, q: Channel, kvn: Channel, out: Channel, ring: str, pos: str, rope: str, sink: str,
                 heads: int = HEADS, ring_slots: int = WINDOW, ring_first: bool = False, dsplit: int = 1,
                 unrope_f32: bool = False):  # fmt: skip
        if heads < 1:
            raise ValueError(f"MLAWindowAttn {name}: heads {heads}")
        if dsplit not in (1, 2, 4):
            raise ValueError(f"MLAWindowAttn {name}: dsplit 1, 2 or 4")
        if q.fmt is not Fmt.BF16 or q.width != heads * HD or q.flagged:
            raise ValueError(
                f"MLAWindowAttn {name}: q is a tagged BF16 channel of width {heads * HD}"
            )
        if kvn.fmt is not Fmt.BF16 or kvn.width != HD or kvn.flagged:
            raise ValueError(
                f"MLAWindowAttn {name}: kvn is a tagged BF16 channel of width {HD}"
            )
        if out.fmt is not Fmt.MXFP8 or out.width != heads * HD or out.flagged:
            raise ValueError(
                f"MLAWindowAttn {name}: out is a tagged MXFP8 channel of width {heads * HD}"
            )
        self.name, self.q, self.kvn, self.out, self.heads = name, q, kvn, out, heads
        self.ring, self.pos, self.rope, self.sink = ring, pos, rope, sink
        if ring_slots < WINDOW:
            raise ValueError(
                f"MLAWindowAttn {name}: ring_slots {ring_slots} < the window {WINDOW}"
            )
        self.ring_slots, self.ring_first, self.dsplit = ring_slots, ring_first, dsplit
        # the output's inverse RoPE: o is bf16, the rotation bf16-rounded before the MXFP8 QAT (the official model's
        # order); ``unrope_f32``: the f32 rotation quantized directly (an engine's arithmetic, ATOM's). A numerics
        # choice, not a schedule knob
        self.unrope_f32 = unrope_f32

    def plan(self, sc: Scenario) -> _APlan:
        return _APlan(self.name, sc.rows, sc.spec, self.q.name, self.kvn.name, self.ring, self.pos, self.rope,
                      self.sink, self.out.name, f"{self.name}.done", self.heads,
                      self.ring_slots, self.ring_first,
                      self.dsplit, unrope_f32=self.unrope_f32)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        tasks = sc.rows * p.tiles * self.dsplit
        writes = tuple(self.out.regions(sc.rows, (self.name,))) + (
            Region(p.done, tasks, (self.name,)),
        )
        args = (self.pos, self.ring, self.rope, rope_sin(self.rope), self.sink)
        return [Stage(self.name, tasks, partial(_attn, p=p), lds_words=p.lds_words, writes=writes, args=args,
                      key=(p,))]  # fmt: skip


# ------------------------------------------------------------------ RingWrite


@dataclass(frozen=True)
class _RPlan:
    name: str
    rows: int
    spec: int
    kvn: str
    ring: str
    pos: str
    done: str  # MLAWindowAttn's
    heads: int = HEADS  # MLAWindowAttn's (its tasks a row: ceil(heads / 16) x dsplit)
    ring_slots: int = WINDOW  # MLAWindowAttn's
    dsplit: int = 1  # MLAWindowAttn's
    mdone: str = (
        ""  # another stage's ring-read flags (``MLASparseAttn(smerge=True)``'s merge), ``mper`` a row
    )
    mper: int = 0


@traced
def _ring_write(k, r, p: _RPlan):
    """Row r: once every attention task of its request has read the ring, its key into slot position % ring_slots."""
    b, _, pos = _row_pos(k, p.pos, p.spec, r)
    tiles = (
        -(-p.heads // TILE) * p.dsplit
    )  # (a ragged last tile counts: ``_APlan.tiles``)
    if k.tid < p.spec * tiles:
        mb.poll([(k.mb[p.done], b * p.spec * tiles + k.tid, 1)], k.tag)
    if const_expr(p.mper > 0):  # noqa: SIM102 (traced)
        if k.tid < p.spec * p.mper:
            mb.poll([(k.mb[p.mdone], b * p.spec * p.mper + k.tid, 1)], k.tag)
    gpu.barrier()
    if k.tid < HD // 2:
        w = mb.poll([(k.mb[p.kvn], (r * HD) // 2 + k.tid, 1)], k.tag)[0][0]
        bo.buffer_store(
            w,
            rsrc(k.args[p.ring]),
            ((b * p.ring_slots + pos % p.ring_slots) * HD) // 2 + k.tid,
        )


class RingWrite(Block):
    def __init__(self, name: str, attn: MLAWindowAttn):
        self.name, self.attn = name, attn

    def stages(self, sc: Scenario) -> list[Stage]:
        a = self.attn
        md = a.ring_mdone(sc) if hasattr(a, "ring_mdone") else None
        p = _RPlan(self.name, sc.rows, sc.spec, a.kvn.name, a.ring, a.pos, f"{a.name}.done", a.heads, a.ring_slots,
                   getattr(a, "ring_parts", getattr(a, "dsplit", 1)), *(md or ("", 0)))  # fmt: skip
        return [
            Stage(
                self.name,
                sc.rows,
                partial(_ring_write, p=p),
                args=(a.pos, a.ring),
                key=(p,),
                row_tasks=1,
            )
        ]
