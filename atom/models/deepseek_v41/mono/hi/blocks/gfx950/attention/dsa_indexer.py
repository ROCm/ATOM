# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Indexer``: DeepSeek-V4.1's sparse-attention indexer at an index-source layer (checkpoint ``inference/model.py``
``Indexer.forward``, decode), per row of the step: the top-512 compressed positions the row's attention reads.

What one decode row at position ``P`` does in the reference (``qr`` = ``q_norm``'s output, the attention's q
low-rank, 1280; ``x`` = ``attn_norm``'s BF16 output, 5120; ``ratio`` = the layer's compress ratio):

- ``q = wq_b(qr)``: FP8 32x32 block-scaled weights [32 heads x 128, 1280], the activation MXFP8 (the attention's
  own ``wq_b`` arithmetic), BF16 out [32, 128];
- RoPE on each head's last 64 dims at ``P`` (the layer's compressed YaRN table), BF16; then the FP4 QAT
  ``fp4_act_quant(q, 32, inplace)`` (groups of 32, E8M0 power-of-two scales);
- ``w = bf16(x . weights_proj^T) * 128^-1/2 * 32^-1/2`` (a plain BF16 ``F.linear`` [32, 5120]; the scale is 2^-6:
  exact);
- the row sees ``n = (P + 1) // ratio`` index keys (``index_k``: the KV source's FP4-QAT'd keys, BF16 [n, 128],
  the one this step emitted included); ``s[h, t] = bf16(q[h] . k[t])`` (a BF16 einsum: the FP4 products and their
  fp32 sums are exact, one rounding);
- ``score[t] = bf16(sum_h bf16(relu(s[h, t]) * w[h]))`` (fp32 sum over the 32 heads);
- (ratio 1, compressed length past 16384 only: layer 20's candidate blocks mask the other index sources'
  scores to -inf: ``candidates``, the bitmap; ``compact``: only the candidate blocks' positions are scored);
- the top ``min(512, n)`` by score, ties to the lower position (the stack's ``stable_topk``), ascending; the
  reference adds the window offset (128) when it hands them to the attention: here they are raw positions.

``scores_f32`` (an engine's arithmetic, not the official model's: ATOM's FP4-plane scorer, the aiter row-group kernel):
``s[h, t]`` stays the MFMA's f32 sum, ``relu(s) * w`` f32, the head sum f32, the scores an F32 channel, TopK in its
f32 mode; layer 20's block maxima are then the f32 scores' (an F32 channel; ``bmax_bf16`` and ``topk_packed``, which
need bf16 scores, are refused). The weights ``w`` stay bf16 (both engines round them so).

Stages:

- ``<name>.wq_b``: ``Linear`` MXFP8 -> BF16 channel ``<name>.q`` (4096; the 32 heads; replicated at TP: every rank
  scores every head, so no rank exchanges partial scores);
- ``<name>.w``: ``GatheredDot`` dense K-split (``mfma_f32_16x16x32_bf16``: exact products, fp32 sums) of ``x`` with
  ``weights_proj``, slice partials only (``sum_stage=False``);
- ``<name>.qprep`` (``qfmt="fp4"``, default): a task a row: q's RoPE and FP4 QAT once -> channel ``<name>.q4``
  (per row 512 words of 8 E2M1 codes, 128 E8M0 scale words: 2.5 KB tagged instead of 16 KB of BF16 q);
- ``<name>.score``: a task per (request, group of ``rows_per_task`` rows, tile of 128 ``subtiles`` positions). It
  polls its rows' FP4 q (decoded to bf16; ``qfmt="bf16"``: wq_b's q, RoPE + QAT here) and the weights' slice
  partials (summed in slice order, rounded, x 2^-6) into LDS; wave w scores ``subtiles`` x 16
  positions: the keys straight from the cache into the MFMA's B operand (64 contiguous bytes a lane), q (A) from
  LDS, 8 MFMAs a (row, 16 positions); relu, the two bf16 roundings, the head sum (in registers, then lanes 16 / 32)
  -> F32 channel ``<name>.scores`` [rows, n_max] (a BF16 value a pair). Keys the step itself emitted (a KV
  source: ``new_keys``, the Compressor's in-launch ``index_out`` channel) are polled from that channel;
- ``TopK`` (``blocks.gfx950.topk``, exact, ties to the lowest index) over the scores with the per-row bound
  ``bound`` (plain int32 [rows] = ``(P + j + 1) // ratio``, written by the host with ``pos``) -> ``out``.

A candidate reader (``candidates``) built with ``compact`` (an n_max from which it applies) past ``CTOPK`` blocks
(n_max > 16384) scores only its
row's candidate blocks: ``<name>.clist`` (a task a row) turns the bitmap into the row's block ids, ascending, into
channel ``<name>.clist`` (``CTOPK`` ids, -1 padded); the score tasks (a row each: the rows' blocks differ) score
compact column c = position ``clist[c // 8] x 8 + c % 8`` (-inf at or past the row's bound) into a scores channel of
``CW`` = 16384 columns; ``TopK`` over those (one level) lifts its winners back to positions (``TopK(lift=)``). The
same ids as the masked full width: the compact columns are the candidate positions in ascending order (ties still
go to the lower position), every masked position is -inf there, and a row past 16384 positions keeps 2048 blocks
(>= 16377 finite scores), so no -inf is among its 512 winners; up to 2048 blocks the list is every reachable block
and the compact columns are the positions.

Reduction order: the q . k sum and the head sum are exact in fp32 for FP4 operands unless a row's group scales
span > ~11 (q . k) / ~16 (heads) binades; then the order is the MFMA's, then lanes (16, 32) and head tiles. The
weights' fp32 sum order (the dot's) is not hipBLASLt's: a weight whose exact value sits at a bf16 rounding midpoint
can come out an ulp apart (none in the fixtures' 128 rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ....device.gfx950 import buffer_ops as bo
from ....device.gfx950 import channels as chan
from ....device.gfx950 import mailbox as mb
from ....device.gfx950 import paged
from ....device.gfx950.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    block_excl_scan,
    butterfly,
    global_load,
    hw_exp2,
    ld_i32,
    popcount,
    rope_cs,
    rope_sin,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from ....plan.program import Scenario, Stage
from ....plan.trace import note
from ....target import MI355X
from ...base import Block, Channel, Fmt
from ..gathered import GatheredDot
from ..linear import BF16Out, Linear
from ..topk import SEG, TopK
from .mla_window import _rope_pair

HEADS, IHD, ROPE, QR, HIDDEN, TOPK_K = 32, 128, 64, 1280, 5120, 512
OOB_WORD = (
    0x20000000  # an i32 word offset past any region: a buffer store there is dropped
)
QW = HEADS * IHD  # 4096: q per row
W_SCALE = 2.0**-6  # 128^-1/2 x 32^-1/2: float32(the official product) is exactly 2^-6
THREADS, WAVES = MI355X.threads, MI355X.waves
CBLOCK, CTOPK = (
    8,
    2048,
)  # the candidate blocks: 8 positions, the top 2048 (config candidate_*)
CW = (
    CTOPK * CBLOCK
)  # 16384: a compact reader's columns (the candidate blocks' positions)
CSEG = 4096  # a compact reader's TopK segment
RAW_MAX = 2048  # qfmt auto: no qprep stage up to this n_max
ONE_LEVEL = 16384  # TopK in one level up to this many positions (seg = n_max)
WIDE = 49152  # from this n_max a score task takes 4 subtiles a wave (256 positions)
Q4W = QW // 8 + QW // 32  # 640 words a row of FP4 q
HPITCH = (
    IHD // 2 + 4
)  # LDS words a head of q (+4: the 16 heads of an A load on distinct banks)
HP4 = (
    IHD // 8 + 4
)  # LDS words a head of FP4 q codes (20: an A load's 16 heads on distinct banks)
KW = 20  # an FP4 key in the ``knew`` channel: 16 code words (element e at nibble e % 8 of word e // 8), the 4 E8M0 group
#          scales in word 16 (group j at byte j), 3 words padding (80 B rows: 16 B aligned)
E2M1, FP4_FMT = 4, 4  # the f8f6f4 MFMA's E2M1 operand format (cbsz / blgp)
assert QW // 2 == 4 * THREADS  # q prologue: 4 words (8 elements) a thread a row


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    spec: int
    batch: int
    ratio: int
    n_max: int  # the scores channel's width (positions a row may see, at most)
    cache_len: int
    g: int  # rows a task
    nt: int  # 16-position subtiles a wave
    kslices: int  # weights_proj's K slices
    q: str
    q4: str  # "" : the score tasks poll wq_b's BF16 q and do RoPE / QAT themselves; else the qprep stage's FP4 q
    wpart: str  # the weights' dense dot's slice partials (summed by the score tasks)
    scores: str
    keys: str
    new: str  # "" without in-launch keys
    pos: str
    rope: str
    bmax: str = ""  # candidate source: the block maxima channel (width n_max / 8)
    cand_in: str = ""  # candidate reader: layer 20's bitmap arg (int32 [rows, cwords])
    knew: str = (
        ""  # FP4 keys at a KV source: kpack's channel of the step's keys (KW words a row)
    )
    kfp4: bool = (
        True  # the keys are ATOM's FP4 plane (``paged``) and the scores run on the scaled FP4 MFMA
    )
    kpack_lanes: bool = (
        False  # kpack: a key's FP4 conversion over wave 0's 64 lanes (2 values a lane), not 4 threads
    )
    clist: str = (
        ""  # compact reader: the channel of the row's candidate block ids (the score tasks' columns)
    )
    # ``short_tiles``: 1 the wide score stage of a pair (its tasks run for rows past ``cap`` positions), 2 the narrow
    # one (one subtile a wave, ``ntl`` tiles a group: rows up to ``cap``); 0 the one score stage; a length-chosen
    # pair: 3 the persistent stage (requests past ``cap``), 4 the tile stage (up to it), 5 the ``rows_long`` tile stage
    # (past it)
    mode: int = 0
    cap: int = 0
    ntl: int = 0
    # ``Indexer(live=)``: the score stage runs the tiles the requests' longest row reaches (``Stage.live_tasks``),
    # task t = (request, row group, live tile t % live tiles); the one score stage only (``mode`` 0)
    live: bool = False
    # ``Indexer(persist=)``: ``per`` score tasks a (request, row group), task c taking tiles c, c + per, .. up to the
    # tiles the request's last row reaches, the rows' q and weights into LDS once a task; 0: a task a tile
    per: int = 0
    # ``Indexer(split=)``: ``sp`` TP ranks score positions [r Q, r Q + Q) of a request (Q = ceil(n / (sp tile)) tile)
    # into ``wloc`` columns (column = position - r Q); the TopKs merge the ranks' answers
    sp: int = 0
    wloc: int = 0
    su: int = (
        0  # (sp) the shares' unit: the persistent stage's tile, whatever this stage's tile
    )
    scap: int = (
        0  # (sp) requests up to this many positions scored whole on every rank (``pcap``: the tile stage's)
    )
    lean: bool = (
        False  # ``Indexer(lean_epi=)``: the score epilogue's bf16 roundings in pairs
    )
    pu: int = (
        1  # ``Indexer(punroll=)``: tiles whose keys a persistent task loads together (in flight at once)
    )
    csparse: bool = (
        False  # ``Indexer(clist_sparse=)``: ``_clist`` reads the row's live words and stores its set bits
    )
    bbf: bool = (
        False  # ``Indexer(bmax_bf16=)``: the block maxima a BF16 channel (two blocks a pair; bf16-exact values)
    )
    f32: bool = (
        False  # ``Indexer(scores_f32=)``: q . k, relu(s) w and the head sum unrounded, an F32 scores channel
    )
    # ``Indexer(items=)`` (with ``per``): the persistent tasks shared among the requests in proportion to their tiles
    # (request b's per[b] tasks a row group, task c taking tiles c, c + per[b], ..), instead
    # of ``per`` a (request, row group) whatever its length; a ``Program.dyn_rows`` build's dead requests get none
    items: bool = False
    # ``Indexer(kmfma=)``: ATOM's FP4 plane in its ``mfma`` layout (gfx950's OPUS scorer; ``paged``), not
    # ``preshuffle``: the keys' and scales' addresses only
    kmfma: bool = False

    @property
    def width(self) -> int:  # the scores channel's columns
        return CW if self.clist else self.wloc if self.sp else self.n_max

    @property
    def nblocks(self) -> int:
        return self.n_max // CBLOCK

    @property
    def cwords(self) -> int:
        return -(-self.nblocks // 32)

    @property
    def groups(self) -> int:
        return -(-self.spec // self.g)

    @property
    def tile(self) -> int:  # positions a task
        return WAVES * self.nt * 16

    @property
    def tiles(self) -> int:
        return self.ntl or -(-self.width // self.tile)

    @property
    def tasks(self) -> int:
        return self.batch * self.groups * (self.per or self.tiles)

    @property
    def lds_q(
        self,
    ) -> (
        int
    ):  # the weights' offset: after q (FP4: the codes, then a scale code a (row, head, group))
        return self.g * HEADS * (HP4 + 4) if self.kfp4 else self.g * HEADS * HPITCH

    @property
    def lds_words(self) -> int:
        return self.lds_q + self.g * HEADS


# ------------------------------------------------------------------ device


_E2M1 = (
    (5.0, False),
    (3.5, True),
    (2.5, False),
    (1.75, True),
    (1.25, False),
    (0.75, True),
    (0.25, False),
)


def _e2m1_code(a):
    """|value| / scale (>= 0) -> the E2M1 code 0 .. 7 (``_e2m1``'s rounding: nearest, ties to the even code)."""
    code = fx.Int32(7)
    for i, (bound, tie_up) in enumerate(_E2M1):
        below = (a < fx.Float32(bound)) if tie_up else (a <= fx.Float32(bound))
        code = below.select(fx.Int32(6 - i), code)
    return code


def _q_rows(k, p: _Plan, rows_pos):
    """Poll + RoPE of this thread's 8 elements (words 4 tid ..) of wq_b's q for each (row, position)."""
    tid = k.tid
    rot = tid % 16 >= 8
    pair0 = (
        fx.max(tid % 16 - 8, fx.Int32(0)) * 4
    )  # this thread's first RoPE pair (4 pairs: 8 elements)
    specs, css = [], []
    for r, ps in rows_pos:
        specs += [
            (k.mb[p.q], r * (QW // 2) + 4 * tid, 2),
            (k.mb[p.q], r * (QW // 2) + 4 * tid + 2, 2),
        ]
        css.append([rope_cs(k, p.rope, ps, pair0 + 2 * h, n=2) for h in range(2)])
    got = mb.poll(specs, k.tag)
    out = []
    for g in range(len(rows_pos)):
        words = got[2 * g] + got[2 * g + 1]
        cs = css[g]
        vals = []
        for w in range(4):
            lo, hi = bf2_f32(words[w])
            c, s = fx.Float32(cs[w // 2][2 * (w % 2)]), fx.Float32(
                cs[w // 2][2 * (w % 2) + 1]
            )
            rr, ri = _rope_pair(lo, hi, c, s, False)
            vals += [rot.select(bf16_round(rr), lo), rot.select(bf16_round(ri), hi)]
        out.append(vals)
    return out


def _qat_codes(vals):
    """q's FP4 QAT (``fp4_act_quant(32, E8M0)``) of this thread's 8 values, the group the 4 lanes of a quad: (the
    word of 8 E2M1 codes, element i at bits 4 i, the E8M0 code of the group's scale)."""
    am = fx.Float32(fx.max(vals[0], -vals[0]))
    for v in vals[1:]:
        am = fx.max(am, fx.Float32(fx.max(v, -v)))
    am = butterfly(am, (1, 2), fx.max)
    am = fx.max(am, fx.Float32(6 * 2**-126))
    scale = hw_exp2(fx.Float32(fmath.ceil(fmath.log2(am / fx.Float32(6.0)))))
    e8 = (
        scale.bitcast(fx.Int32) >> 23
    ) & 255  # the E8M0 code (the scale is a normal power of two)
    word = fx.Int32(0)
    for i, v in enumerate(vals):
        q = v / scale
        code = _e2m1_code(fx.Float32(fx.max(q, -q)))
        code = (q.bitcast(fx.Int32) < fx.Int32(0)).select(
            code + 8, code
        )  # the sign bit (-0 keeps it)
        word = word | (code << (4 * i))
    return word, e8


def _qprep(k, t, p: _Plan):
    """Row t: q's RoPE and FP4 QAT once -> channel ``q4`` (codes, E8M0 scales)."""
    b, j = t // p.spec, t % p.spec
    pos = uniform(ld_i32(k.args[p.pos], b)) + j
    word, e8 = _qat_codes(_q_rows(k, p, [(t, pos)])[0])
    mb.put_words(k.mb[p.q4], t * Q4W + k.tid, [word], k.tag)
    _put_scale(k, k.mb[p.q4], t * Q4W + QW // 8 + k.tid // 4, e8, k.tid % 4 == 0)


def _fp4_pack(vals):
    """32 FP4-QAT'd values (f32, each a code times one power of two) -> (4 code words, the E8M0 code). The scale is
    2^(floor(log2 max |v|) - 2): every value / scale is then an E2M1 level, so the codes are exact (zeros: 2^0).
    """
    am = fx.Float32(fx.max(vals[0], -vals[0]))
    for v in vals[1:]:
        am = fx.max(am, fx.Float32(fx.max(v, -v)))
    eb = (am.bitcast(fx.Int32) >> 23) & 255
    e8 = (eb == 0).select(fx.Int32(127), fx.max(eb - 2, fx.Int32(1)))
    inv = ((254 - e8) << 23).bitcast(fx.Float32)  # 2^-(e8 - 127): exact
    words = []
    for w in range(4):
        word = fx.Int32(0)
        for i in range(8):
            q = vals[8 * w + i] * inv
            code = _e2m1_code(fx.Float32(fx.max(q, -q)))
            code = (q.bitcast(fx.Int32) < fx.Int32(0)).select(code + 8, code)
            word = word | (code << (4 * i))
        words.append(word)
    return words, e8


@traced
def _new_fp4(k, p, b, pos, pp, n_old, n_hi, kg, w0, w1, w2, w3, ksc, any_new):
    """A subtile holding one of the step's keys (wave-uniform ``any_new``): the lane's group of it from the channel,
    bf16 -> FP4 codes; the cache's words otherwise (the conversion costs ~800 VALU ops a lane: only where needed).
    """
    if any_new:
        live = (pp >= n_old) & (pp < n_hi)
        je = fx.min(
            fx.max((pp + 1) * p.ratio - 1 - pos, fx.Int32(0)), fx.Int32(p.spec - 1)
        )
        n0, n1, n2, n3, sw = _poll_knew(
            k, k.mb[p.knew], (b * p.spec + je) * KW, kg, k.tag, live
        )
        w0, w1, w2, w3 = (
            live.select(n0, w0),
            live.select(n1, w1),
            live.select(n2, w2),
            live.select(n3, w3),
        )
        ksc = live.select((sw >> (8 * kg)) & 255, ksc)
    return [w0, w1, w2, w3], ksc


@traced
def _poll_knew(k, a, base, kg, tag, live):
    """The lane's 4 code words and the scale word of an FP4 key in the ``knew`` channel (``kpack``'s)."""
    n0, n1, n2, n3, sw = fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    if live:
        got = mb.poll(
            [(a, base + 4 * kg, 2), (a, base + 4 * kg + 2, 2), (a, base + 16, 1)], tag
        )
        n0, n1, n2, n3, sw = got[0][0], got[0][1], got[1][0], got[1][1], got[2][0]
    return n0, n1, n2, n3, sw


def _kpack(k, t, p: _Plan):
    """Row t (a KV source): the key it emitted (channel ``new``, bf16) -> entry (P + 1) // ratio - 1 of ATOM's FP4
    plane, in place (a live row's; ``paged``), for later launches, and the ``knew`` channel (this launch's). Threads 0 .. 3: one 32-group each (``kpack_lanes``: wave 0, 16 lanes a group, 2 values a
    lane: the same operations on each value, the group's max and the code words by lane butterflies).
    """
    b, j = t // p.spec, t % p.spec
    pos = uniform(ld_i32(k.args[p.pos], b)) + j
    emits = (pos + 1) % p.ratio == 0
    rl = paged.row_live(
        k, t
    )  # (a padded or dead request's row writes none of ATOM's plane)
    if p.kpack_lanes:
        _kpack_lanes_if(k, t, p, b, (pos + 1) // p.ratio - 1, emits & (k.tid < 64), rl)
    else:
        _kpack_if(k, t, p, b, (pos + 1) // p.ratio - 1, emits & (k.tid < 4), rl)


def key_scales(keys: str) -> str:
    """The pointer-table entry of the E8M0 scales plane beside the FP4 index plane's values ``keys``."""
    return keys + ".s"


@traced
def _key_store(k, p, b, e, g, words, e8, rl):
    """Entry ``e``'s group ``g`` into ATOM's FP4 plane, in place, on a live row (``rl``): its 16 B of codes
    (``words``: the 4 words; (w, word): word w alone; None: none) and its E8M0 byte (``e8``; None: none).
    """
    if rl:
        unit, kt = paged.key_unit(k, b, e, p.ratio)
        addr = paged.key_value_addr(k, p.keys, unit, kt, g, mfma=p.kmfma)
        if const_expr(isinstance(words, tuple)):
            paged.global_store(addr + fx.Int64(4 * words[0]), words[1])
        if const_expr(isinstance(words, list)):
            paged.global_store(addr, fx.Vector.from_elements(words, fx.Int32), 16)
        if const_expr(e8 is not None):
            paged.global_store(
                paged.key_scale_addr(k, key_scales(p.keys), unit, kt, g, mfma=p.kmfma),
                e8,
                1,
            )


@traced
def _kpack_if(k, t, p, b, e, live, rl):
    if live:
        g = k.tid
        got = mb.poll(
            [(k.mb[p.new], t * (IHD // 2) + 16 * g + 2 * i, 2) for i in range(8)], k.tag
        )
        vals = []
        for pr in got:
            for w in pr:
                lo, hi = bf2_f32(w)
                vals += [lo, hi]
        words, e8 = _fp4_pack(vals)
        _key_store(k, p, b, e, g, words, e8, rl)
        mb.put_words(
            k.mb[p.knew], t * KW + 4 * g, words[:2], k.tag
        )  # the in-launch copy for the score tasks
        mb.put_words(k.mb[p.knew], t * KW + 4 * g + 2, words[2:], k.tag)
        sw = butterfly(e8 << (8 * g), (1, 2), _or)
        if g == 0:
            mb.put_words(k.mb[p.knew], t * KW + 16, [sw], k.tag)


def _or(a, b):
    return a | b


@traced
def _kpack_lanes_if(k, t, p, b, e, live, rl):
    """``_kpack_if``'s outputs from wave 0: lane l holds group g = l / 16's pair word j = l % 16 (elements 2 j,
    2 j + 1), ``_fp4_pack``'s max (16-lane butterfly: max is exact in any order), scale and per-element codes; code
    word w of the group from lanes 4 w .. + 4 (OR butterfly), the scale word from the 4 groups' lane 0s.
    """
    if live:
        g, j = k.lane // 16, k.lane % 16
        w = mb.poll([(k.mb[p.new], t * (IHD // 2) + 16 * g + j, 1)], k.tag)[0][0]
        lo, hi = bf2_f32(w)
        am = butterfly(
            fx.max(fx.Float32(fx.max(lo, -lo)), fx.Float32(fx.max(hi, -hi))),
            (1, 2, 4, 8),
            fx.max,
        )
        eb = (am.bitcast(fx.Int32) >> 23) & 255
        e8 = (eb == 0).select(fx.Int32(127), fx.max(eb - 2, fx.Int32(1)))
        inv = ((254 - e8) << 23).bitcast(fx.Float32)  # 2^-(e8 - 127): exact
        nib = fx.Int32(0)
        for i, v in enumerate((lo, hi)):
            q = v * inv
            code = _e2m1_code(fx.Float32(fx.max(q, -q)))
            code = (q.bitcast(fx.Int32) < fx.Int32(0)).select(code + 8, code)
            nib = nib | (code << (4 * ((2 * j + i) % 8)))
        word = butterfly(nib, (1, 2), _or)
        sw = butterfly(e8 << (8 * g), (16, 32), _or)
        _kpack_lane_out(k, t, p, b, e, g, j, word, sw, e8, rl)


@traced
def _kpack_lane_out(k, t, p, b, e, g, j, word, sw, e8, rl):
    if j % 4 == 0:  # (lane j = 0 of group g also its scale byte)
        _key_store(k, p, b, e, g, (j // 4, word), None, rl)
        mb.put_words(k.mb[p.knew], t * KW + 4 * g + j // 4, [word], k.tag)
    if j == 0:
        _key_store(k, p, b, e, g, None, e8, rl)
    if k.lane == 0:
        mb.put_words(k.mb[p.knew], t * KW + 16, [sw], k.tag)


def _prologue4k(k, p: _Plan, b, pos, j0):
    """(FP4 keys) The task's rows' FP4 q codes -> LDS [g][head][HP4 words], their scale codes after
    ([g][head][group]); the weights after those."""
    tid = k.tid
    specs = []
    for g in range(p.g):
        r = b * p.spec + fx.min(j0 + g, fx.Int32(p.spec - 1))
        specs += [
            (k.mb[p.q4], r * Q4W + tid, 1),
            (k.mb[p.q4], r * Q4W + QW // 8 + tid // 4, 1),
        ]
    wt, wspecs = _wts_specs(k, p, b, j0)
    got = mb.poll(specs, k.tag)
    for g in range(p.g):
        fx.ptr_store(got[2 * g][0], k.lds + ((g * HEADS + tid // 16) * HP4 + tid % 16))
        _lds_put(
            k.lds + (p.g * HEADS * HP4 + g * HEADS * 4 + tid // 4),
            got[2 * g + 1][0],
            tid % 4 == 0,
        )
    _wts_finish(k, p, wt, wspecs)


@traced
def _put_scale(k, a, i, v, ok):
    if ok:
        mb.put_words(a, i, [v], k.tag)


def _prologue(k, p: _Plan, b, pos, j0):
    if p.kfp4 and p.q4:
        _prologue4k(k, p, b, pos, j0)


def _wts_specs(k, p: _Plan, b, j0):
    wt = fx.min(k.tid, fx.Int32(p.g * HEADS - 1))
    wg, wh = wt // HEADS, wt % HEADS
    wr = b * p.spec + fx.min(j0 + wg, fx.Int32(p.spec - 1))
    return wt, [
        (k.mb[p.wpart], wr * (p.kslices * HEADS) + s * HEADS + wh, 1)
        for s in range(p.kslices)
    ]


def _wts_finish(k, p: _Plan, wt, wspecs):
    wv = fx.Float32(0.0)
    for gw in mb.poll(wspecs, k.tag):
        wv = wv + gw[0].bitcast(fx.Float32)
    wv = bf16_round(bf16_round(wv) * fx.Float32(W_SCALE))
    _lds_put(k.ldsf + (p.lds_q + wt), wv, k.tid < p.g * HEADS)


@traced
def _lds_put(ptr, v, ok):
    if ok:
        fx.ptr_store(v, ptr)


@traced
def _put_pair(k, a, e, v0, v1, ok):
    if ok:
        chan.put_bf16(a, e, [v0, v1], k.tag)


@traced
def _put_score(k, a, i, v, ok):
    if ok:
        mb.put(a, i, v, k.tag)


def _score_body(k, t, p: _Plan, b, j0, t0, pos, first=None):
    st = _tile_loads(
        k, p, b, j0, t0, pos, first
    )  # the keys first: plain loads, in flight through the q poll
    _prologue(k, p, b, pos, j0)
    _tile_new(k, p, b, pos, st)
    gpu.barrier()
    _tile_compute(k, p, b, j0, pos, st)
    gpu.barrier()  # the next task reuses the LDS


def _tile_loads(k, p: _Plan, b, j0, t0, pos, coff=None) -> SimpleNamespace:
    """A tile's loads (positions ``t0`` ..): the keys, their scales, (candidate readers) the bitmap words; ``coff``
    (split): the first position of this rank's columns."""
    lane, wave = k.lane, k.wave
    n_hi = (pos + p.spec) // p.ratio  # the request's last row sees this many keys
    n_old = (
        pos // p.ratio
    )  # keys from earlier launches (the cache); the rest this step's (``new``)
    kg = (
        lane // 16
    )  # the lane's K quarter (32 elements: the A / B K map, the same on both operands)
    cols, pcs = [], []
    if (
        p.clist
    ):  # compact: column c is position clist[c // 8] x 8 + c % 8 (a task a row: j0)
        cols = [t0 + (wave * p.nt + s) * 16 + lane % 16 for s in range(p.nt)]
        rc = (b * p.spec + j0) * CTOPK
        got = mb.poll([(k.mb[p.clist], rc + c // CBLOCK, 1) for c in cols], k.tag)
        pcs = [g[0] * CBLOCK + c % CBLOCK for g, c in zip(got, cols)]
    # the keys first: plain loads, in flight through the q poll
    keys, poss, kscale = [], [], []
    for s in range(p.nt):
        pp = pcs[s] if p.clist else t0 + (wave * p.nt + s) * 16 + lane % 16
        poss.append(pp)
        pc = fx.min(fx.max(pp, fx.Int32(0)), fx.Int32(p.cache_len - 1))
        # the lane's 32-group of key ``pc`` in ATOM's FP4 plane (``paged``): its 16 B of codes, the scale dword
        # holding its E8M0 byte
        unit, t = paged.key_unit(k, b, pc, p.ratio, uni=False)
        keys.append(
            fx.Vector(
                global_load(
                    paged.key_value_addr(k, p.keys, unit, t, kg, mfma=p.kmfma), 4
                )
            )
        )
        if p.kmfma:  # the word holding its byte (``paged.mfma_scale_word``)
            sa, byte = paged.mfma_scale_word(k, key_scales(p.keys), unit, t, kg)
            kscale.append((fx.Int32(global_load(sa)) >> (8 * byte)) & 255)
        else:
            sw = global_load(
                paged.key_scale_addr(k, key_scales(p.keys), unit, t & 4, kg)
            )
            kscale.append((fx.Int32(sw) >> (8 * (t % 4))) & 255)
    cws = {}
    return SimpleNamespace(
        keys=keys,
        kscale=kscale,
        poss=poss,
        cws=cws,
        cols=cols,
        t0=t0,
        kg=kg,
        n_hi=n_hi,
        n_old=n_old,
        coff=coff,
    )


def _tile_new(k, p: _Plan, b, pos, st) -> None:
    """The keys this step emitted (a KV source) into the tile's key registers."""
    keys, kscale, poss, t0, kg, n_old, n_hi = (
        st.keys,
        st.kscale,
        st.poss,
        st.t0,
        st.kg,
        st.n_old,
        st.n_hi,
    )
    wave = k.wave
    if (
        p.new and p.kfp4
    ):  # keys the step emitted: the channel's bf16 group -> FP4 codes in the lane
        for s in range(p.nt):
            p0 = (
                t0 + (wave * p.nt + s) * 16
            )  # the subtile's first position (wave-uniform)
            any_new = (p0 + 15 >= n_old) & (p0 < n_hi)
            key, ksc = _new_fp4(k, p, b, pos, poss[s], n_old, n_hi, kg, keys[s][0], keys[s][1], keys[s][2],
                                keys[s][3], kscale[s], any_new)  # fmt: skip
            keys[s] = fx.Vector.from_elements(key, fx.Int32)
            kscale[s] = ksc


def _tile_compute(k, p: _Plan, b, j0, pos, st) -> None:
    """The tile's scores (the rows' q and weights in LDS) -> the scores channel (and the block maxima)."""
    keys, kscale, kg = st.keys, st.kscale, st.kg
    lane = k.lane
    zero = fx.Float32(0.0)
    vt = fx.Vector.make_type(4, fx.Float32)
    vi = fx.Vector.make_type(4, fx.Int32)
    for g in range(p.g):
        wts = [fx.Vector(fx.ptr_load(k.ldsf + (p.lds_q + g * HEADS + ht * 16 + 4 * kg), result_type=vt))
               for ht in range(2)]  # fmt: skip
        for s in range(p.nt):
            tot = None
            for ht in range(2):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                if (
                    p.kfp4
                ):  # one scaled FP4 MFMA: q (A: head lane % 16, group lane // 16) x the 16 keys (B)
                    head = g * HEADS + ht * 16 + lane % 16
                    a = fx.Vector(
                        fx.ptr_load(k.lds + (head * HP4 + 4 * kg), result_type=vi)
                    )
                    sa = fx.ptr_load(k.lds + (p.g * HEADS * HP4 + head * 4 + kg))
                    acc = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32), [a, keys[s], acc, E2M1, FP4_FMT, 0, sa, 0, kscale[s]]))  # fmt: skip
                for i in range(4):
                    if (
                        p.f32
                    ):  # (``scores_f32``) the MFMA's f32 q . k, relu(s) w in f32: no rounding
                        v = (acc[i] > zero).select(acc[i], zero) * fx.Float32(
                            wts[ht][i]
                        )
                    tot = v if tot is None else tot + v
            _score_out(k, p, b, j0, g, s, pos, st, tot)


def _put_nb(k, a, word, words, ok, nbytes) -> None:
    """``words`` at word ``word`` of region ``a`` where ``ok``; elsewhere the store goes past the region's buffer range
    and is dropped: no branch (a DPP reduction feeding a store in a lane-conditional branch may be sunk into it by
    LLVM and read stale lanes)."""
    note(a.region, a.space, "put")
    off = ok.select(word, fx.Int32(OOB_WORD))
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(a.value, nbytes),
        off,
        cache_modifier=CM_DEV,
    )


def _score_out(k, p: _Plan, b, j0, g, s, pos, st, tot) -> None:
    """Row j0 + g's head sum ``tot`` (this lane's heads) at subtile ``s``: the bf16 score -> the scores channel (and
    the block maxima)."""
    poss, cols, lane, zero, j = st.poss, st.cols, k.lane, fx.Float32(0.0), j0 + g
    tot = butterfly(tot, (16, 32))
    tot = (
        (tot + zero) if p.f32 else bf16_round(tot + zero)
    )  # -0 -> +0 (the reference's sum starts at +0)
    pp = poss[s]
    r = b * p.spec + fx.min(j, fx.Int32(p.spec - 1))
    if (
        p.clist
    ):  # compact column cols[s]: -inf at or past the row's bound (and on -1 padding)
        nrow = (pos + fx.min(j, fx.Int32(p.spec - 1)) + 1) // p.ratio
        tot = ((pp >= 0) & (pp < nrow)).select(tot, fx.Float32(-float("inf")))
        if p.f32:  # an F32 channel: a value a lane
            _put_score(
                k, k.mb[p.scores], r * CW + cols[s], tot, (lane < 16) & (j < p.spec)
            )
            return
        nxt = xshfl(tot, 1)
        ok = (lane < 16) & (j < p.spec) & (lane % 2 == 0)
        _put_pair(k, k.mb[p.scores], r * CW + cols[s], tot, nxt, ok)
        return
    ok = (lane < 16) & (j < p.spec) & (pp < p.n_max)
    blk = fx.min(pp, fx.Int32(p.n_max - 1)) // CBLOCK
    nxt = xshfl(tot, 1)  # the scores channel is BF16: an even lane puts its pair
    if p.per or p.sp:  # (the persistent / split stages: branch-free puts, ``_put_nb``)
        lc = (
            pp - st.coff if p.sp else pp
        )  # (split: this rank's columns: position - its first)
        ok = ok & (lc < p.width)
        if p.f32:  # an F32 channel: a (value, tag) pair a lane
            at = 2 * (r * p.width + fx.min(lc, fx.Int32(p.width - 1)))
            _put_nb(
                k,
                k.mb[p.scores],
                at,
                [tot.bitcast(fx.Int32), k.tag],
                ok,
                p.rows * p.width * 8,
            )
        if (
            p.bmax
        ):  # (the pair's maximum from ``nxt``: a DPP xor-1 maximum came out unreduced in this loop's code)
            bound = (pos + fx.min(j, fx.Int32(p.spec - 1)) + 1) // p.ratio
            ninf = fx.Float32(-float("inf"))
            m = fx.max(
                (pp < bound).select(tot, ninf), (pp + 1 < bound).select(nxt, ninf)
            )  # (even lanes: their pair)
            m = butterfly(m, (2, 4), fx.max)
            m = (blk == (bound - 1) // CBLOCK).select(fx.Float32(float("inf")), m)
            at = 2 * (
                r * (p.width // CBLOCK) + fx.min(lc, fx.Int32(p.width - 1)) // CBLOCK
            )
            _put_nb(k, k.mb[p.bmax], at, [m.bitcast(fx.Int32), k.tag], ok & (lane % CBLOCK == 0),
                    p.rows * (p.width // CBLOCK) * 8)  # fmt: skip
        return


@traced
def _score_if(k, t, p, b, j0, t0, pos, live, first=None):
    if live:
        _score_body(k, t, p, b, j0, t0, pos, first)


def _share(k, p: _Plan, pos):
    """(first, n, n_hi): request at ``pos``'s positions this task's rank scores, [first, first + n) (first None: all
    of them, n_hi capped at the scores' width), and n_hi the positions its last row sees (``sp``: this rank's share of
    them, Q = ceil(n_hi / (sp tile)) tile)."""
    n_hi = (pos + p.spec) // p.ratio
    n_hi = fx.min(n_hi, fx.Int32(p.n_max))
    q = (n_hi + (p.sp * p.su - 1)) // (p.sp * p.su) * p.su
    first = k.rank * q
    n = fx.max(fx.min(n_hi - first, q), fx.Int32(0))
    if (
        p.scap
    ):  # a request up to ``scap`` positions: whole on every rank (its TopKs rank it whole: ``TopK(scap=)``)
        whole = n_hi <= p.scap
        first, n = whole.select(fx.Int32(0), first), whole.select(n_hi, n)
    return first, n, n_hi


def _live_tiles(k, p: _Plan):
    """The tiles the requests' longest row reaches (one at least): the live score tasks' tiles a row group (``mode``
    4: of the requests up to ``cap`` positions, 5: of those past it; ``sp``: of this rank's shares).
    """
    n = fx.Int32(0)
    for b in range(p.batch):
        pos = uniform(ld_i32(k.args[p.pos], b))
        if p.sp or p.mode in (4, 5):
            _, nb, n_hi = _share(k, p, pos)
            if p.mode == 4:
                nb = (n_hi <= p.cap).select(nb, fx.Int32(0))
            n = fx.max(n, nb)
    return fx.max((fx.min(n, fx.Int32(p.width)) + (p.tile - 1)) // p.tile, fx.Int32(1))


def _score_live_tasks(k, p: _Plan):
    return fx.Int32(p.batch * p.groups) * _live_tiles(k, p)


def _score_live(q: _Plan):
    """A score stage's ``Stage.live_tasks``: the tile tasks' (``live``), the persistent stage's of a pair (mode 3)."""
    if q.mode == 3:
        return partial(_long_live_tasks, p=q)
    if q.live and q.mode in (0, 4, 5) and not q.per:
        return partial(_score_live_tasks, p=q)
    return None


def _long_live_tasks(k, p: _Plan):
    """``mode`` 3 (the persistent stage of a length-chosen pair): every task if a request is past ``cap``, else none."""
    on = fx.Int32(0)
    for b in range(p.batch):
        on = fx.max(
            on,
            ((uniform(ld_i32(k.args[p.pos], b)) + p.spec) // p.ratio > p.cap).select(
                fx.Int32(1), on
            ),
        )
    return on * p.tasks


def _t0(p: _Plan, tile, first):
    return tile * p.tile if first is None else first + tile * p.tile


@traced
def _persist_loop(k, p: _Plan, b, j0, c, pos, tiles, first=None, step=None):
    step = (
        p.per if step is None else step
    )  # (``_Plan.items``: the request's share, a uniform Int32)
    if const_expr(p.pu == 1):
        for tile in range(c, tiles, step):
            st = _tile_loads(k, p, b, j0, _t0(p, tile, first), pos, first)
            _tile_new(k, p, b, pos, st)
            _tile_compute(k, p, b, j0, pos, st)


@traced
def _persist_if(k, p: _Plan, b, j0, c, pos, tiles, live, first=None, step=None):
    if live:
        _prologue(k, p, b, pos, j0)
        gpu.barrier()
        _persist_loop(k, p, b, j0, c, pos, tiles, first, step)
        gpu.barrier()  # the next task reuses the LDS


def _score_persist(k, t, p: _Plan):
    """``_Plan.per``: task (request, row group, c) scores tiles c, c + per, .. below the tiles its request's last row
    reaches (the same tiles, positions and arithmetic as a task a tile)."""
    c, rest = t % p.per, t // p.per
    grp, b = rest % p.groups, rest // p.groups
    pos = uniform(ld_i32(k.args[p.pos], b))
    first, n, n_hi = _share(
        k, p, pos
    )  # (sp: this rank's share: positions [r Q, r Q + Q) of the request's)
    tiles = (n + (p.tile - 1)) // p.tile
    live = c < tiles
    if (
        p.mode == 3
    ):  # the long stage of a length-chosen pair: requests past ``cap`` only
        live = live & (n_hi > p.cap)
    _persist_if(k, p, b, grp * p.g, c, pos, tiles, live, first)


def _score(k, t, p: _Plan):
    if p.per:
        _score_persist(k, t, p)
        return
    live_tiles = p.live and p.mode in (0, 4, 5)
    tiles = (
        fx.max(k.live // (p.batch * p.groups), fx.Int32(1)) if live_tiles else p.tiles
    )  # (live)
    tile = t % tiles
    rest = t // tiles
    grp, b = rest % p.groups, rest // p.groups
    pos = uniform(ld_i32(k.args[p.pos], b))
    t0 = tile * p.tile
    first = None
    if p.sp:  # this rank's share of the request's positions (``_share``)
        first, n, _n_hi = _share(k, p, pos)
        live = t0 < n
        t0 = first + t0
    elif p.clist:  # a row a task: its compact columns below min(its bound, CW)
        live = t0 < fx.min((pos + grp + 1) // p.ratio, fx.Int32(CW))
    if p.mode in (2, 4):  # (4: the tile stage of a length-chosen pair)
        live = live & ((pos + p.spec) // p.ratio <= p.cap)
    _score_if(k, t, p, b, grp * p.g, t0, pos, live, first)


def _cmask(k, t, p: _Plan, cand: str, out: str, lds: bool = False):
    """Row t: layer 20's candidate block list (``TopK``'s, ascending, -1 padded) -> bitmap words ``out``
    [rows, cwords] (bit b % 32 of word b // 32: block b kept). A row with at most ``CTOPK`` reachable blocks keeps
    every one of them (the list is all of them: ``_cmask_all``'s words, the list not read); else the list into LDS,
    thread w owns word w: a lower bound of 32 w in the list, then the next 32 entries (``lds``: the row's words in LDS,
    each entry's bit set there, ``_cmask_bits``: the same words)."""
    b, j = t // p.spec, t % p.spec
    nb = ((uniform(ld_i32(k.args[p.pos], b)) + j + 1) // p.ratio + CBLOCK - 1) // CBLOCK
    _cmask_list_if(k, t, p, cand, out, nb > CTOPK, lds)
    _cmask_all_if(k, t, p, out, nb <= CTOPK)


@traced
def _cmask_list_if(k, t, p, cand, out, live, lds=False):
    if live:  # noqa: SIM102 (traced)
        if const_expr(lds):
            _cmask_bits(k, t, p, cand, out)


@traced
def _cmask_all_if(k, t, p, out, live):
    if live:
        _cmask_all(k, t, p, out)


def _cmask_bits(k, t, p: _Plan, cand: str, out: str):
    """``_cmask``'s words built in LDS: the row's ``cwords`` words cleared (before the poll), each entry's bit
    OR'd into its word (an LDS atomic; -1 padding into a junk word past them), then the words stored. Every entry
    sets one bit of one word, so the words are ``_cmask``'s."""
    tid = k.tid
    for w0 in range(0, p.cwords + 1, THREADS):  # (+1: the junk word)
        fx.ptr_store(fx.Int32(0), k.lds + fx.min(tid + w0, fx.Int32(p.cwords)))
    specs = [(k.mb[cand], t * CTOPK + 4 * tid + 2 * e, 2) for e in range(2)]
    got = mb.poll(specs, k.tag)
    gpu.barrier()
    for w in got[0] + got[1]:
        ok = w >= fx.Int32(0)
        fx.atomic_or(
            k.lds + ok.select(w >> 5, fx.Int32(p.cwords)),
            fx.Int32(1) << (w & 31),
            syncscope="workgroup",
        )
    gpu.barrier()
    for w0 in range(0, p.cwords, THREADS):
        w = fx.min(tid + w0, fx.Int32(p.cwords - 1))
        _store_word(
            k.args[out], t * p.cwords + w, fx.ptr_load(k.lds + w), tid + w0 < p.cwords
        )
    gpu.barrier()  # the next task reuses the LDS


def _cmask_all(k, t, p: _Plan, out: str):
    """Row t, at most CTOPK blocks: every reachable block (ceil(bound / 8) of them) is a candidate."""
    b, j = t // p.spec, t % p.spec
    nb = ((uniform(ld_i32(k.args[p.pos], b)) + j + 1) // p.ratio + CBLOCK - 1) // CBLOCK
    for w0 in range(0, p.cwords, THREADS):
        w = fx.min(k.tid + w0, fx.Int32(p.cwords - 1))
        full = (w + 1) * 32 <= nb
        part = (fx.Int32(1) << (fx.max(nb - 32 * w, fx.Int32(0)) & 31)) - 1
        bits = full.select(fx.Int32(-1), (nb > 32 * w).select(part, fx.Int32(0)))
        _store_word(k.args[out], t * p.cwords + w, bits, k.tid + w0 < p.cwords)


def _clist(k, t, p: _Plan, cand: str):
    """Row t (a compact reader): layer 20's candidate bitmap (plain int32 [rows, cwords]) -> the kept blocks'
    ids, ascending, the first ``CTOPK`` (-1 padded) -> channel ``clist``. Thread i owns words [per i, per (i + 1)):
    their bits' count, a CTA prefix sum, then each set bit's id into LDS at its rank."""
    if const_expr(p.csparse):
        _clist_sparse(k, t, p, cand)
        return
    tid = k.tid
    per = -(-p.cwords // THREADS)
    junk = CTOPK  # an LDS slot for the bits past CTOPK and the clear bits
    w0 = tid * per
    words = [(w0 + i < p.cwords).select(ld_i32(k.args[cand], t * p.cwords + fx.min(w0 + i, fx.Int32(p.cwords - 1))),
                                        fx.Int32(0)) for i in range(per)]  # fmt: skip
    cnt = popcount(words[0])
    for w in words[1:]:
        cnt = cnt + popcount(w)
    for e in range(CTOPK // THREADS):
        fx.ptr_store(fx.Int32(-1), k.lds + (tid + e * THREADS))
    off, _ = block_excl_scan(
        cnt, k.lane, k.wave, k.lds + (junk + 4)
    )  # barriers inside: the -1s land first
    for i, w in enumerate(words):
        for bit in range(32):
            on = ((w >> bit) & 1) != 0
            fx.ptr_store(
                (w0 + i) * 32 + bit,
                k.lds + (on & (off < CTOPK)).select(off, fx.Int32(junk)),
            )
            off = off + on.select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    n = CTOPK // THREADS
    ids = fx.Vector(
        fx.ptr_load(k.lds + n * tid, result_type=fx.Vector.make_type(n, fx.Int32))
    )
    for e in range(0, n, 2):
        mb.put_words(
            k.mb[p.clist], t * CTOPK + n * tid + e, [ids[e], ids[e + 1]], k.tag
        )
    gpu.barrier()  # the next task reuses the LDS


def _clist_sparse(k, t, p: _Plan, cand: str):
    """``_clist`` (``Indexer(clist_sparse=)``) over the row's live words only: thread i owns words [i pl, (i + 1) pl)
    of them (pl = ceil(live / THREADS): one word a thread up to THREADS live words, coalesced), one CTA prefix sum of
    their bit counts, then each set bit stored at its rank: the same ids at the same ranks, O(the row's blocks). Layer
    20 writes the words past a row's bound 0."""
    junk = CTOPK
    b, j = t // p.spec, t % p.spec
    nb = ((uniform(ld_i32(k.args[p.pos], b)) + j + 1) // p.ratio + CBLOCK - 1) // CBLOCK
    nw = fx.min((nb + 31) // 32, fx.Int32(p.cwords))
    for e in range(CTOPK // THREADS):
        fx.ptr_store(fx.Int32(-1), k.lds + (k.tid + e * THREADS))
    pl = (nw + (THREADS - 1)) // THREADS
    w0 = k.tid * pl
    # every load issued at once (the build's words a thread at most), the ones past this thread's pl masked
    words = [((w0 + i < nw) & (fx.Int32(i) < pl)).select(
        ld_i32(k.args[cand], t * p.cwords + fx.min(w0 + i, fx.Int32(p.cwords - 1))), fx.Int32(0))
        for i in range(-(-p.cwords // THREADS))]  # fmt: skip
    cnt = popcount(words[0])
    for w in words[1:]:
        cnt = cnt + popcount(w)
    off, _ = block_excl_scan(
        cnt, k.lane, k.wave, k.lds + (junk + 4)
    )  # barriers inside: the -1s land first
    for i, w in enumerate(words):
        off = _clist_bits(k, w, (w0 + i) * 32, off, junk)
    gpu.barrier()
    n = CTOPK // THREADS
    ids = fx.Vector(
        fx.ptr_load(k.lds + n * k.tid, result_type=fx.Vector.make_type(n, fx.Int32))
    )
    for e in range(0, n, 2):
        mb.put_words(
            k.mb[p.clist], t * CTOPK + n * k.tid + e, [ids[e], ids[e + 1]], k.tag
        )
    gpu.barrier()  # the next task reuses the LDS


@traced
def _clist_bits(k, w, base, off, junk):
    """Word ``w``'s set bits, lowest first: block ``base + bit`` into LDS at rank ``off`` (past CTOPK: the junk slot);
    -> the rank after them."""
    while w != 0:
        bit = popcount((w & (fx.Int32(0) - w)) - 1)  # the lowest set bit's index
        fx.ptr_store(base + bit, k.lds + (off < CTOPK).select(off, fx.Int32(junk)))
        off = off + 1
        w = w & (w - 1)
    return off


@traced
def _store_word(ptr, i, w, ok):
    if ok:
        bo.buffer_store(w, rsrc(ptr), i)


# ------------------------------------------------------------------ plan


class Indexer(Block):
    """The indexer of one index-source layer (module docstring). ``qn``: ``q_norm``'s MXFP8 channel (1280); ``x``:
        ``attn_norm``'s BF16 channel (5120); ``keys``: the source's FP4 index plane in ATOM's storage, in place (its values;
    ``key_scales(keys)`` its E8M0 scales; ``device.gfx950.paged``);
        ``new_keys``: at a KV source, the Compressor's ``index_out`` channel (BF16 128, zeros on rows that do not
        emit), else None; ``bound``: plain int32 [rows]; ``out``: TopK's (a plain int32 [rows, 512] arg or an F32
        channel of width 512 carrying int32 bits), -1 padded, ascending."""

    def __init__(self, name: str, qn: Channel, x: Channel, ratio: int, n_max: int, out: Channel | str, pos: str,
                 rope: str, keys: str, cache_len: int, bound: str, new_keys: Channel | None = None,
                 rows_per_task: int = 0, subtiles: int = 0, kslices: int = 4, seg: int = 0,
                 qfmt: str = "auto", wq_b_rows: int | None = None, cand_out: str = "", cand_bound: str = "",
                 candidates: str = "", kfmt: str = "fp4", kpack_lanes: bool = False,
                 compact: int = 0, short_tiles: bool = False, live: bool = False,
                 persist: int = 0, cseg: int = 0, split: int = 0, punroll: int = 1,
                 pcap: int = 0, lean_epi: bool = False, pnarrow: bool = False,
                 clist_sparse: bool = False, topk_packed: bool = False, cmask_lds: bool = False,
                 bmax_bf16: bool = False, topk_xhoist: bool = False, scores_f32: bool = False,
                 rows_long: int = 0, rcap: int = 0, items: bool = False, kmfma: bool = False):  # fmt: skip
        if ratio not in (1, 2):
            raise ValueError(f"Indexer {name}: ratio 1 or 2, got {ratio}")
        if qn.fmt is not Fmt.MXFP8 or qn.width != QR:
            raise ValueError(f"Indexer {name}: qn is q_norm's MXFP8 channel ({QR})")
        if x.fmt is not Fmt.BF16 or x.flagged or x.width != HIDDEN:
            raise ValueError(
                f"Indexer {name}: x is a tagged BF16 channel of width {HIDDEN}"
            )
        if new_keys is not None and (
            new_keys.fmt is not Fmt.BF16 or new_keys.flagged or new_keys.width != IHD
        ):
            raise ValueError(
                f"Indexer {name}: new_keys is a tagged BF16 channel of width {IHD}"
            )
        if n_max % 4 or n_max > cache_len:
            raise ValueError(
                f"Indexer {name}: n_max a multiple of 4, <= cache_len ({n_max}, {cache_len})"
            )
        self.name, self.ratio, self.n_max, self.pos, self.rope = (
            name,
            ratio,
            n_max,
            pos,
            rope,
        )
        # f32 scores (module docstring): an engine's arithmetic (ATOM's), a numerics choice, not a
        # schedule knob; the bf16 scores' shortcuts (the lean epilogue's paired roundings, packed lists, bf16 maxima)
        # do not apply to them
        if scores_f32 and (lean_epi or topk_packed or bmax_bf16):
            raise ValueError(
                f"Indexer {name}: scores_f32 with lean_epi / topk_packed / bmax_bf16 (bf16 scores only)"
            )
        self.scores_f32 = scores_f32
        # a second, narrow score stage (one 16-position subtile a wave) for rows short enough that its tasks fit one
        # grid round, beside the wide one for the rest (``score_plans``; the same per-position arithmetic)
        self.short_tiles = short_tiles
        # live tasks (``Stage.live_tasks``): the score stage and the TopK trees run only the tasks the step's rows
        # reach (from ``pos`` / the bounds), so a large build costs a short context nothing for its tasks past it
        self.live = live
        # persistent score tasks (``_Plan.per``): ``persist`` tasks a (request, row group), each looping over its
        # tiles; -1: as many as fill the grid. Full-width sources only (not the compact readers' columns)
        self.persist = persist
        self.punroll = max(
            1, punroll
        )  # (persist: tiles a loop step, their keys in flight together)
        # (persist) ``pcap``: requests up to this many positions take the tile tasks (the block's rows a task, live
        # tasks), the rest the persistent ones: two score stages, each with no live tasks outside its lengths
        self.pcap = pcap if persist and live else 0
        self.pnarrow = pnarrow
        # ``rows_long``: the rows a score task for long requests (6 rows a task, the keys read
        # once a request, saves 1.4-4.6 us a source layer at 128K and costs ~7 us at 1K). Two tile stages (no
        # ``persist``): requests past ``rcap`` positions at ``rows_long`` rows a task, the rest at the block's; with
        # ``persist`` + ``pcap`` the persistent (long) stage takes ``rows_long`` (its cap ``pcap``). 0: the block's
        # rows a task for every request
        if rows_long and (rows_long < 1 or (not self.pcap and rcap < 1) or short_tiles):
            raise ValueError(
                f"Indexer {name}: rows_long needs rcap > 0 (or persist + pcap) and no short_tiles"
            )
        self.rows_long, self.rcap = rows_long, (
            rcap if rows_long else 0
        )  # (a compact reader: off, below)
        # the persistent score tasks shared among the requests by their tiles (``_Plan.items``; with ``persist`` only:
        # a tile a task has no per-request share to deal)
        self.items = items
        # ATOM's FP4 plane in its ``mfma`` layout (``_Plan.kmfma``)
        self.kmfma = kmfma
        # the score epilogue with fewer VALU ops: the same roundings, ReLU values and sum order
        self.lean_epi = lean_epi
        # a compact reader's candidate list (``_clist``) from the row's live bitmap words only (its bound: the words
        # past it are 0) and its set bits only (a loop over each word's set bits): O(the row's blocks), not O(n_max)
        self.clist_sparse = clist_sparse
        self.cmask_lds = cmask_lds  # layer 20's bitmap words built in LDS (``_cmask_bits``), not stored a word
        # positions split across ``split`` TP ranks (persistent score tasks, full-width sources): each rank scores and
        # ranks its share, the TopKs merge the ranks' answers (``TopK(split=)``: the same ids)
        self.split = split if split > 1 else 0
        if self.split and not persist:
            raise ValueError(
                f"Indexer {name}: split needs persistent score tasks (persist)"
            )
        self.keys, self.cache_len, self.new_keys, self.g, self.nt, self.kslices = (
            keys, cache_len, new_keys, rows_per_task, subtiles, kslices)  # fmt: skip
        self.wq_b_w = f"{name}_wq_b"
        self.q = Channel(f"{name}.q", QW, Fmt.BF16)
        # auto: short contexts (n_max <= RAW_MAX) at up to 8 rows a request skip the qprep hop (the score tasks prep
        # q: few tiles repeat it); 16-row steps keep it (decided per scenario in ``plan``)
        self.qauto = qfmt == "auto" and n_max <= RAW_MAX and kfmt == "fp4"
        qfmt = "fp4" if qfmt == "auto" else qfmt
        if qfmt not in ("fp4", "raw", "bf16") or (qfmt == "raw" and kfmt != "fp4"):
            raise ValueError(f"Indexer {name}: qfmt 'auto', 'fp4' (a qprep stage), 'raw' (RoPE / FP4 QAT in the "
                             "score tasks; FP4 keys) or 'bf16' (bf16 keys)")  # fmt: skip
        # FP4 q: per row 512 code words (8 E2M1 codes, element 8 i + e at bits 4 e) then 128 E8M0 scale words
        self.q4 = Channel(f"{name}.q4", Q4W, Fmt.F32) if qfmt == "fp4" else None
        self.wq_b = Linear(
            f"{name}.wq_b",
            (self.wq_b_w,),
            qn,
            QW,
            BF16Out(self.q),
            weight_format="mxfp8",
            group_rows=wq_b_rows,
        )  # 32: 128 tasks, so the weights' dot and qprep get CTAs of their own
        self.wproj = f"{name}.wproj"
        self.wdot = GatheredDot(f"{name}.w", x, self.wproj, Channel(f"{name}.w", HEADS, Fmt.F32), kslices=kslices,
                                sum_stage=False)  # fmt: skip
        if kfmt != "fp4" or qfmt == "bf16":
            raise ValueError(f"Indexer {name}: kfmt 'fp4' (the keys: ATOM's FP4 index plane, in place), qfmt fp4 / "
                             "raw")  # fmt: skip
        self.kfp4 = kfmt == "fp4"
        self.kpack_lanes = kpack_lanes  # kpack's FP4 conversion over 64 lanes (the same bits; ``_kpack``)
        self.knew = (
            Channel(f"{name}.knew", KW, Fmt.F32)
            if self.kfp4 and new_keys is not None
            else None
        )
        # a candidate reader past CTOPK blocks scores only its candidate blocks' positions (module docstring);
        # ``compact``: 0 off, else the least n_max that does
        if compact and not candidates:
            raise ValueError(
                f"Indexer {name}: compact is a candidate reader's (candidates)"
            )
        self.compact = bool(compact) and n_max // CBLOCK > CTOPK and n_max >= compact
        if (
            self.compact
        ):  # a compact reader scores a row a task already (its own candidate columns): no rows_long
            self.rows_long = self.rcap = 0
        self.clist = Channel(f"{name}.clist", CTOPK, Fmt.F32) if self.compact else None
        if self.compact or (n_max < WIDE and not subtiles):
            self.split = 0  # (a compact reader's columns are its candidates'; below WIDE the tile follows the rows)
        self.wtile = (
            WAVES * 16 * (subtiles or (4 if n_max >= WIDE else 1))
        )  # (split: the shares' unit)
        self.wloc = (
            -(-n_max // (self.split * self.wtile)) * self.wtile if self.split else 0
        )
        width = CW if self.compact else self.wloc or n_max
        # the scores are bf16 values: a BF16 channel (half the bytes) and TopK's bf16 mode (two radix passes); f32
        # scores an F32 channel and TopK's f32 mode
        self.scores = Channel(
            f"{name}.scores", width, Fmt.F32 if scores_f32 else Fmt.BF16
        )
        # TopK's segment: one level up to ONE_LEVEL positions (a level's hand-off costs more than the larger sort);
        # compact: CSEG (4 segments of the 16384 columns, then one merge)
        seg = seg or (CSEG if self.compact else width if width <= ONE_LEVEL else SEG)
        # (split + pcap: the requests the tile stage scores are whole on every rank, their TopKs too: at most a rank's
        # columns, a multiple of the shares' unit)
        if self.split and self.pcap:
            self.pcap = min(self.pcap, self.wloc) // self.wtile * self.wtile
        sp = (
            {"split": self.split, "unit": self.wtile, "gn": n_max, "scap": self.pcap}
            if self.split
            else {}
        )
        self.topk = TopK(f"{name}.topk", self.scores, width, TOPK_K, out, bound=bound, seg=seg, bf16=not scores_f32,
                         lift=self.clist, lift_unit=CBLOCK if self.compact else 0, live=live, packed=topk_packed,
                         xhoist=topk_xhoist,
                         **sp)  # fmt: skip
        # layer 20 (the candidate source): its block maxima -> TopK 2048 -> the bitmap ``cand_out`` (plain int32
        # [rows, ceil(n_max / 256)], for the index sources of later launches, ``candidates``); ``cand_bound``: the
        # host's per-row block count ceil(bound / 8)
        if (
            bool(cand_out) != bool(cand_bound)
            or (cand_out and candidates)
            or (n_max % (4 * CBLOCK))
        ):
            raise ValueError(
                f"Indexer {name}: cand_out with cand_bound, not with candidates; n_max % 32 == 0"
            )
        self.cand_out, self.candidates = cand_out, candidates
        self.bmax = self.cand = self.ctopk = None
        # up to CTOPK blocks (n_max <= 16384) the reference keeps every reachable block: the bitmap needs no scores
        self.cand_all = bool(cand_out) and n_max // CBLOCK <= CTOPK
        if cand_out and not self.cand_all:
            nb = (self.wloc or n_max) // CBLOCK  # (split: this rank's blocks)
            # (``bmax_bf16``: the maxima of bf16 scores (and +-inf) are bf16 values: the same keys, TopK's bf16 mode)
            self.bmax = Channel(f"{name}.bmax", nb, Fmt.BF16 if bmax_bf16 else Fmt.F32)
            self.cand = Channel(f"{name}.cand", CTOPK, Fmt.F32)
            # ``cseg``: the candidate TopK's segment (0: one level up to ONE_LEVEL blocks, else SEG)
            csp = ({"split": self.split, "unit": self.wtile // CBLOCK, "gn": n_max // CBLOCK,
                    "scap": -(-self.pcap // CBLOCK)} if self.split else {})  # fmt: skip
            self.ctopk = TopK(f"{name}.ctopk", self.bmax, nb, CTOPK, self.cand, bound=cand_bound,
                              seg=min(cseg, nb) if cseg else nb if nb <= ONE_LEVEL else SEG, live=live,
                              bf16=bmax_bf16, xhoist=topk_xhoist, **csp)  # fmt: skip

    def raw(self, sc: Scenario) -> bool:
        """No qprep stage: q's RoPE / QAT in the score tasks."""
        return self.q4 is None and self.kfp4 or self.qauto and sc.spec <= 8

    def plan(self, sc: Scenario) -> _Plan:
        # defaults: 2 rows x 128 positions a task up to 8 rows a request (1 row without
        # qprep), 4 x 256 above
        g = self.g or (4 if sc.spec > 8 else 1 if self.raw(sc) else 2)
        g = (
            1 if self.compact else g
        )  # compact: a row a task (each row its own candidate positions)
        if (
            min(g, sc.spec) * HEADS > THREADS
        ):  # ``_wts_specs``: a (row, head) weight a thread
            raise ValueError(
                f"Indexer {self.name}: rows a task x heads at most {THREADS} (g {g}, {HEADS} heads)"
            )
        if self.nt:
            nt = self.nt
        elif self.compact:  # 256 columns a task, 64 at 1 row
            nt = 4 if sc.spec > 1 else 1
        else:  # long contexts: 256 positions a task
            nt = 4 if self.n_max >= WIDE else 2 if sc.spec > 8 else 1
        q4 = "" if self.raw(sc) or self.q4 is None else self.q4.name
        return _Plan(self.name, sc.rows, sc.spec, sc.batch, self.ratio, self.n_max, self.cache_len,
                     min(g, sc.spec), nt, self.kslices, self.q.name, q4,
                     self.wdot.part.name, self.scores.name,
                     self.keys, self.new_keys.name if self.new_keys else "", self.pos, self.rope,
                     self.bmax.name if self.bmax else "", "" if self.compact else self.candidates,
                     self.knew.name if self.knew else "", self.kfp4, self.kpack_lanes,
                     self.clist.name if self.compact else "", live=self.live,
                     per=self._per(sc, -(-(self.wloc or self.n_max) // (WAVES * nt * 16)), min(g, sc.spec)),
                     sp=self.split, wloc=self.wloc, su=self.wtile if self.split else 0, lean=self.lean_epi,
                     scap=self.pcap if self.split else 0,
                     pu=self.punroll if self.persist else 1,
                     csparse=self.clist_sparse and self.compact,
                     bbf=self.bmax is not None and self.bmax.fmt is Fmt.BF16, f32=self.scores_f32,
                     items=self.items and bool(self.persist) and not self.compact, kmfma=self.kmfma)  # fmt: skip

    def _per(self, sc: Scenario, tiles: int, g: int) -> int:
        """Persistent score tasks a (request, row group): ``persist`` (-1: the grid's share), at most the tiles."""
        if not self.persist or self.compact:
            return 0
        units = sc.batch * -(-sc.spec // g)
        per = MI355X.cus // units if self.persist < 0 else self.persist
        return max(1, min(per, tiles))

    def score_plans(self, sc: Scenario) -> list[_Plan]:
        """The score stage's plan, or with ``short_tiles`` (a request a step, full width, wide tiles) the wide plan
        for rows past ``cap`` positions and a narrow one (one subtile a wave) for rows up to it, ``cap`` the most
        positions its narrow tasks cover in one grid round."""
        from dataclasses import replace

        p = self.plan(sc)
        if (
            self.pcap and p.per
        ):  # length-chosen: the persistent stage past ``pcap``, the tile tasks up to it
            g = (
                4 if sc.spec > 8 else 1 if self.raw(sc) else 2
            )  # (the block's rows a task)
            # (the tile stage: the build's tiles, or ``pnarrow`` one 16-position subtile a wave, as a short build's: a
            # short row's tiles in parallel, but 4x the tasks at mid lengths)
            nt = 1 if self.pnarrow else p.nt
            long = replace(p, mode=3, cap=self.pcap)
            return [
                long,
                replace(p, mode=4, cap=self.pcap, per=0, g=min(g, sc.spec), nt=nt),
            ]
        if not (
            self.short_tiles
            and sc.batch == 1
            and not self.compact
            and p.nt > 1
            and not p.per
        ):
            return [p]
        ntl = min(MI355X.cus // p.groups, -(-p.width // (WAVES * 16)))
        cap = ntl * WAVES * 16
        return [replace(p, mode=1, cap=cap), replace(p, nt=1, mode=2, cap=cap, ntl=ntl)]

    def stages(self, sc: Scenario) -> list[Stage]:
        if sc.spec > 16:
            raise ValueError(
                f"Indexer {self.name}: up to 16 rows a request (the weights' dense dot)"
            )
        p = self.plan(sc)
        plans = self.score_plans(sc)
        names = [f"{self.name}.score"] + [f"{self.name}.score_n"] * (len(plans) > 1)
        writes = self.scores.regions(sc.rows, tuple(names)) + (
            self.bmax.regions(sc.rows, tuple(names)) if self.bmax else ())  # fmt: skip
        scores = [Stage(n, q.tasks, partial(_score, p=q), lds_words=q.lds_words, writes=writes if i == 0 else (),
                        args=(self.pos, self.rope, rope_sin(self.rope), self.keys, key_scales(self.keys), *paged.REQ)
                        + ((q.cand_in,) if q.cand_in else ()),
                        key=("indexer_score",) + (q,),
                        live_tasks=_score_live(q))
                  for i, (n, q) in enumerate(zip(names, plans))]  # fmt: skip
        out = self.wq_b.stages(sc) + self.wdot.stages(sc)
        if (
            self.compact
        ):  # after wq_b / the dot (on CTAs of its own: it waits on nothing in the launch)
            lname = f"{self.name}.clist"
            out.append(Stage(lname, sc.rows, partial(_clist, p=p, cand=self.candidates), lds_words=CTOPK + 8,
                             writes=tuple(self.clist.regions(sc.rows, (lname,))),
                             args=(self.candidates,) + ((self.pos,) if p.csparse else ()),
                             key=("indexer_clist",) + (p,)))  # fmt: skip
        if p.q4:
            qname = f"{self.name}.qprep"
            out.append(Stage(qname, sc.rows, partial(_qprep, p=p), writes=self.q4.regions(sc.rows, (qname,)),
                             args=(self.pos, self.rope, rope_sin(self.rope)), key=("indexer_qprep",) + (p,)))  # fmt: skip
        if (
            self.knew is not None
        ):  # a KV source: its emitted keys -> the FP4 cache (later launches) and ``knew``
            kname = f"{self.name}.kpack"
            # the cache entry is for later launches: this launch's score tasks poll ``knew``
            out.append(Stage(kname, sc.rows, partial(_kpack, p=p),
                             args=(self.pos, self.keys, key_scales(self.keys), *paged.REQ),
                             writes=tuple(self.knew.regions(sc.rows, (kname,))), key=("indexer_kpack",) + (p,),
                             later_args=(self.keys, key_scales(self.keys))))  # fmt: skip
        out += scores
        if (
            self.cand_out
        ):  # the candidate path first: the bitmap is what later launches wait on
            cname = f"{self.name}.cmask"
            lds = {"lds": True} if self.cmask_lds else {}
            out += self.ctopk.stages(sc) + [
                Stage(cname, sc.rows, partial(_cmask, p=p, cand=self.cand.name, out=self.cand_out, **lds),
                      lds_words=max(CTOPK, p.cwords + 1) if lds else CTOPK, args=(self.pos, self.cand_out),
                      key=("indexer_cmask",) + (p,) + tuple(lds))
            ]  # fmt: skip
        return out + self.topk.stages(sc)

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """``layer``: the layer's tensors as the checkpoint names them (``attn.indexer.wq_b.{weight,
        scale}``, ``attn.indexer.weights_proj.weight``), any device; returns tensors on that device.
        """
        from types import SimpleNamespace

        from ....weights.mx import fp8_block_to_mxfp8

        wt, sc = layer["attn.indexer.wq_b.weight"], layer["attn.indexer.wq_b.scale"]
        m = fp8_block_to_mxfp8(wt.cpu().contiguous(), sc.cpu().contiguous())
        m.packed, m.codes = m.packed.to(wt.device), m.codes.to(wt.device)
        out = dict(self.wq_b.pack(SimpleNamespace(linears={self.wq_b_w: m})))
        out[self.wproj] = (
            layer["attn.indexer.weights_proj.weight"].to(torch.bfloat16).contiguous()
        )
        return out


def bounds(pos: torch.Tensor, spec: int, ratio: int) -> torch.Tensor:
    """The ``bound`` arg: int32 [batch x spec], row j of request b sees (pos[b] + j + 1) // ratio keys."""
    j = torch.arange(spec, device=pos.device)
    return ((pos.view(-1, 1).long() + j + 1) // ratio).flatten().int()
