# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Indexer block scores for K4's sparse selection (``select_blocks`` polls them).

The Triton decode scorer's math, one 128-key block of index-K per task: bf16(fp8
K) x bf16 index q on the MFMA, max over the causal keys, times fp32(sm_scale *
log2 e). (An NVFP4 index cache: K = e4m3_rn_satfinite(e2m1 x scale) and q e4m3,
one fp8 16x16x128 MFMA a tile, like vLLM's NVFP4 indexer.) A context of <= TOPK_BLOCKS blocks keeps every block and is not scored;
nor are the blocks the selector pins (the first ``init_blocks``, the last
``local_blocks``): their keys are forced. A scored block waits for the index keys
K1 inserts this step only when it may hold one (``wait_new_keys``).

A task is one block of one request and runs on one wave: the request's q tokens
(a speculative verify's rows) are the MFMA's B columns, so the block's index-K is
read once for all of them, and no task needs the CTA (no barrier).

Emitted at the end of K1 (``emit_index_scores``), which makes index q: K4's
selector then finds the scores in place instead of scoring after its launch.
``build_index_score_kernel`` runs the same tasks alone, for K4's harnesses.
"""

import struct
from dataclasses import dataclass, field
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.compiler.kernel_function import KernelFunction
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    LONG_FROM_BLOCKS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    SPARSE_BLOCK,
    TOPK_BLOCKS,
    TP,
    IndexHeads,
)
from atom.models.minimax_m3.mono.sources import SOURCES
from atom.mono.device.mx import UNIT_SCALE, mfma_scaled
from atom.mono.device.ops import (
    butterfly,
    fp8x8_bf16_pk,
    kernel_symbol,
    lane_gather,
    launcher,
    mfma_bf16,
    permlane_swap,
    readlane,
    rsrc,
    traced,
    uniform,
    unlikely,
    wave_incl_scan,
)
from atom.mono.device.sync import Addr, Mailbox
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.trace import Space

INDEX_BLOCK_BYTES = SPARSE_BLOCK * HEAD_DIM  # one fp8 index-K block
# an nvfp4 index page (one buffer): 128 keys x 64 B e2m1 (low nibble = even dim),
# then 128 x 8 B e4m3 scales, one per 16 dims (aiter / vLLM single-buffer layout)
NVFP4_PAGE_BYTES = 9216
NVFP4_SCALES_AT = SPARSE_BLOCK * HEAD_DIM // 2
KEY_TILES = SPARSE_BLOCK // 16  # MFMA row tiles of a block
_CURRENT_STREAM = fx.Stream(None)


E4M3_MAX = 448.0


def e4m3_dwords(vals):
    """Eight f32 -> two dwords of E4M3 (RNE, saturating at 448: the convert itself
    does not saturate), the quantizer of vLLM's NVFP4 indexer for K and for Q."""
    hi, lo = fx.Float32(E4M3_MAX), fx.Float32(-E4M3_MAX)
    out = []
    for d in range_constexpr(2):
        packed = as_ir_value(fx.Int32(0))
        for h in range_constexpr(2):
            a, b = (vals[4 * d + 2 * h + i].minimumf(hi).maximumf(lo) for i in (0, 1))
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, as_ir_value(a), as_ir_value(b), packed, bool(h)
            )
        out.append(fx.Int32(packed))
    return out


def nvfp4_dword_e4m3x8(word, scale_byte):
    """Eight e2m1 of ``word`` (low nibble first) times the e4m3 scale in the low
    byte of ``scale_byte``, rounded to E4M3 (``k_round="e4m3"`` of aiter's scorer):
    two dwords. ``v_cvt_scalef32_pk_f32_fp4`` takes only the exponent of its scale,
    so the elements are widened at scale 1.0 and multiplied by the scale decoded to
    f32 (exact: <= 6 significant bits)."""
    sc = fx.Vector(
        rocdl.cvt_pk_f32_fp8(
            res=fx.Vector.make_type(2, fx.Float32),
            src=as_ir_value(fx.Int32(scale_byte & 0xFF)),
            word_sel=False,
        )
    )[0]
    w = as_ir_value(fx.Int32(word))
    one = as_ir_value(fx.Float32(1.0))
    prod = []
    for sel in range_constexpr(4):
        pair = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(T.vec(2, T.f32), w, one, sel))
        prod += [fx.Float32(pair[0]) * sc, fx.Float32(pair[1]) * sc]
    return e4m3_dwords(prod)


# aiter's per-scale e4m3 table (``fp4_dword_to_e4m3`` of its scorer): for each positive
# e4m3 scale byte the 8 bytes e4m3_rn_satfinite(mag * scale), mag the e2m1 magnitudes
# (equal to ``nvfp4_dword_e4m3x8`` for every code and scale byte 0x00..0x7E: exhaustive
# check), 2 dwords an entry, in LDS: a K dword is then 2 v_perm_b32 + ~7 ALU, not ~33.
LUT_ENTRIES = 128
E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SIGN8 = -0x7F7F7F80  # 0x80808080 as an i32: the sign bit of each fp8 byte of a dword
MAG8 = 0x07070707  # the e2m1 magnitude bits of each nibble, spread one a byte
EVEN_ODD = (0, 2, 4, 6, 1, 3, 5, 7)  # a dword's elements as ``fp4_dword_e4m3x8_lut``


@fx.struct
class LutSmem:
    tab: fx.Array[fx.Int32, 2 * LUT_ENTRIES, 16]


def lut_lds():
    """The CTA's table in LDS: ``LUT_ENTRIES`` * 8 B more, allocated from the kernel's
    own allocator (a kernel may make one ``SharedAllocator``, and the callers' Smem
    structs are not ours to change). shortcut: reaches FlyDSL's current kernel's
    private ``_shared_allocator``, so it needs FlyDSL's static allocator mode (the
    default); upgrade when the callers' Smem structs carry the table."""
    alloc = KernelFunction.get_current()._shared_allocator
    assert alloc is not None and alloc.is_static, "the kernel's SharedAllocator"
    return alloc.allocate(LutSmem).peek().tab.ptr


def e4m3_lut_row(scale_byte):
    """The two dwords of the table entry of the scale byte ``scale_byte`` (0..127):
    e4m3_rn_satfinite(mag * scale) for the 8 e2m1 magnitudes. The scale decodes at f32
    exactly (``cvt_pk_f32_fp8``)."""
    sc = fx.Vector(
        rocdl.cvt_pk_f32_fp8(
            res=fx.Vector.make_type(2, fx.Float32),
            src=as_ir_value(fx.Int32(scale_byte)),
            word_sel=False,
        )
    )[0]
    return fx.Vector.from_elements(
        e4m3_dwords([fx.Float32(m) * sc for m in E2M1_MAGS]), fx.Int32
    )


def e4m3_lut_entry(tab, scale_byte):
    """The (t_lo, t_hi) dwords of the table entry of ``scale_byte``'s low 7 bits."""
    t = fx.Vector(
        fx.ptr_load(
            tab + (scale_byte & 0x7F) * 2, result_type=fx.Vector.make_type(2, fx.Int32)
        )
    )
    return t[0], t[1]


def fp4_dword_e4m3x8_lut(word, t_lo, t_hi):
    """Eight e2m1 of ``word`` -> e4m3 K by lookup in a scale's table entry: v_perm_b32
    takes magnitudes 0..7 (selector bytes) from the 8 table bytes, four a dword; the
    sign bit of each nibble is OR-ed in afterwards. Two dwords: the even elements
    (0, 2, 4, 6), then the odd ones (``EVEN_ODD``; ``q_e4m3`` orders q the same way,
    the MFMA sums over k in any order)."""
    w = fx.Int32(word)
    ev = rocdl.perm_b32(t_hi, t_lo, w & MAG8)
    od = rocdl.perm_b32(t_hi, t_lo, (w >> 4) & MAG8)
    return fx.Int32(ev) | ((w << 4) & SIGN8), fx.Int32(od) | (w & SIGN8)


def q_e4m3(frags, order=range(8)):
    """The four k-chunk fragments (bf16x8 each) of an index q -> the e4m3 B operand
    of one 16x16x128 MFMA (8 dwords), as vLLM quantizes the index q; each chunk's
    elements in ``order``."""
    words = []
    for f in frags:
        v = fx.Vector(f).to(fx.Float32)
        words += e4m3_dwords([fx.Float32(v[i]) for i in order])
    return fx.Vector.from_elements(words, fx.Int32)


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def index_scale_log2e(sm_scale: float) -> float:
    """fp32(sm_scale * log2 e), the scorer's scale."""
    return _f32(_f32(sm_scale) * _f32(LOG2E))


def scored_count(n_blk, init_blocks: int, local_blocks: int):
    """How many blocks (from block init_blocks on) a context of n_blk blocks scores."""
    return (n_blk > TOPK_BLOCKS).select(n_blk - (init_blocks + local_blocks), 0)


class StepRows(NamedTuple):
    """A decode step's rows as the indexer sees them (``step_rows``)."""

    seq: list  # each row's context, >= 1 (a graph's pad rows carry 0)
    n_blks: list  # each row's index blocks
    long: list  # each row's request is long

    def any_long(self):
        v = self.long[0]
        for lg in self.long[1:]:
            v = v | lg
        return v


def step_rows(seq_lens, tokens: int, q_len, heads: IndexHeads) -> StepRows:
    """The one place a request is decided long: past LONG_FROM_BLOCKS index
    blocks (with every index head its scores take the context-parallel layout,
    ``emit_index_scores``); K4 selects it past the split stage.

    Decided by the request's last row (a request is ``q_len`` consecutive rows):
    a speculative verify's rows can sit either side of the threshold, and a
    request's rows must all read the layout its scores were written in."""
    seq = [
        fx.max(uniform(bo.buffer_load(rsrc(seq_lens), k, vec_width=1, dtype=T.i32)), 1)
        for k in range(tokens)
    ]
    n_blks = [(s + SPARSE_BLOCK - 1) // SPARSE_BLOCK for s in seq]
    req_blks = list(n_blks)
    for k in range(tokens - 2, -1, -1):
        req_blks[k] = (fx.Int32(k) % q_len == q_len - 1).select(
            n_blks[k], req_blks[k + 1]
        )
    return StepRows(seq, n_blks, [nb > LONG_FROM_BLOCKS for nb in req_blks])


@traced
def emit_index_scores(
    slot, lane, wave, red, mb_put, iscore, q_frag, tokens, q_len, init_blocks,
    local_blocks, scale, index_cache, block_table, seq_lens, bt_width,
    wait_new_keys=None, prepare=None, heads=ONE_INDEX_HEAD, q_frag_head=None,
    prepare_heads=None, nvfp4=False, strip=(), wait_index_q=None, stamp=None,
):  # fmt: skip
    """This CTA's (``slot``) share of the step's score tasks: up to BLOCKS tasks,
    task u runs on CTA u, else on wave (u // BLOCKS) % WAVES of CTA u % BLOCKS
    (``red``: WAVES * 16 floats of LDS). The tokens form requests of ``q_len``
    consecutive rows sharing a block table; the last one has the longest context
    and lists the request's scored blocks. ``q_frag(k, s)``: this lane's B operand
    for token k, k-chunk s (bf16x8 of index q). Pair k * MAX_INDEX_BLOCKS + b of
    mailbox ``iscore`` (an ``Addr``) <- the score of token k's block b.

    ``wait_new_keys()`` (K1 fused into the layer kernel; wave-local) waits for the
    index keys this launch inserts: a block within MAX_TOKENS keys of a token's
    end may hold an earlier token of its request (speculative decode rows).
    Without it, every key this launch inserts must sit in a pinned block
    (one-token requests). ``prepare(first, last)`` (with CTA barriers) runs on a
    CTA with tasks before any index q is read: K1 makes rows first..last's
    there.

    Indexer context parallelism (``heads.count`` > 1): a long request
    (``step_rows``) scores only this rank's blocks, b = heads.own
    (mod TP) -- every index q head of them, a task's key block read once for all:
    pair k * MAX_INDEX_BLOCKS + h * (MAX_INDEX_BLOCKS / TP) + b / TP <- the score
    of token k's block b for head h. ``q_frag_head(k, s, h)``: head h's B operand
    (every lane its own h); ``prepare_heads()`` (with CTA barriers, on every CTA)
    makes those requests' rows' heads first. The other requests score as above.

    ``nvfp4``: the index cache holds nvfp4 pages (``NVFP4_PAGE_BYTES``); a lane's
    key operand of chunk s is dims 32 g4 + 8 s .. + 7 (its own 16 B of the key's
    data, no transpose), so ``q_frag`` must return the same dims (its ``perm``).

    More than MAX_TOKENS rows (``BIG``): the step's plan is lane-resident, not unrolled over
    the rows (n scalar chains do not fit the registers at 64 rows): lane k holds row k's
    context, block count and, for a request's last row, its task count, whose wave scan
    gives every request's first task; a lane decodes its task by a 6-step binary search
    over the scan (``lane_gather``). Wave-per-task mode only (the task counts of
    these steps are not bounded by BLOCKS), one index head (no context parallelism), and
    ``q_frag`` reads the index q the head tasks wrote, once ``wait_index_q`` says every
    row's is there."""
    assert local_blocks >= 1, "the block of the current token must be pinned"
    BIG = tokens > MAX_TOKENS
    assert not BIG or (
        heads.count == 1 and prepare is None and wait_index_q is not None
    )
    l16 = lane % 16
    g4 = lane // 16
    # nvfp4: the per-task-wave paths look K up in a CTA table (built before them)
    tab = lut_lds() if nvfp4 else None
    tid = wave * 64 + lane
    r_ic = rsrc(index_cache, 0 if "idx" in strip else None)
    if const_expr(BIG):
        rows = seq = pre = pre_cp = None
        cp, first_own, total_cp = False, 0, 0
        # lane k: row k (rows past ``tokens`` repeat the last: a lane's loads stay in range)
        row_l = fx.min(lane, tokens - 1)
        seq_k = fx.max(
            fx.Int32(bo.buffer_load(rsrc(seq_lens), row_l, vec_width=1, dtype=T.i32)), 1
        )
        nb_k = (seq_k + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        last_k = (lane < tokens) & (fx.Int32(lane) % q_len == q_len - 1)
        n_scored_k = last_k.select(
            scored_count(nb_k, init_blocks, local_blocks), fx.Int32(0)
        )
        end_k = wave_incl_scan(n_scored_k, lane)  # request tasks: [end - n_scored, end)
        start_k = end_k - n_scored_k
        total = readlane(end_k, 63)
    else:
        rows = step_rows(seq_lens, tokens, q_len, heads)
        seq = rows.seq
        # every request's scored blocks in one list, a task per wave across the grid;
        # context-parallel requests' (this rank's blocks) in a second
        cp = heads.count > 1
        first_own = init_blocks + (heads.own - init_blocks) % TP
        pre = [fx.Int32(0)]
        pre_cp = [fx.Int32(0)]
        for k in range_constexpr(tokens):
            last = fx.Int32(k) % q_len == q_len - 1
            n_scored = scored_count(rows.n_blks[k], init_blocks, local_blocks)
            if const_expr(cp):
                long = rows.long[k]
                n_own = fx.max(init_blocks + n_scored - first_own + TP - 1, 0) // TP
                pre.append(pre[-1] + (last & ~long).select(n_scored, fx.Int32(0)))
                pre_cp.append(pre_cp[-1] + (last & long).select(n_own, fx.Int32(0)))
            else:
                pre.append(pre[-1] + last.select(n_scored, fx.Int32(0)))
        total = pre[tokens]
        total_cp = pre_cp[-1]  # 0 without context parallelism

    r_bt = rsrc(block_table)

    def task_block(u, pre=pre, first=init_blocks, stride=1):
        """Task u of list ``pre`` -> (the request's last token, block b); per lane
        if u is."""
        tk = fx.Int32(0)
        for j in range_constexpr(1, tokens):
            tk = tk + (u >= pre[j]).select(fx.Int32(1), fx.Int32(0))
        return tk, first + stride * (u - pick(pre[:tokens], tk))

    def page_of(tk, b):
        return fx.Int32(
            bo.buffer_load(r_bt, tk * bt_width + b, vec_width=1, dtype=T.i32)
        )

    def task_block_big(u):
        """Task u (per lane) -> (the request's last row, block b): the number of lanes whose
        inclusive scan end is <= u is the row (a 6-step binary search)."""
        tk = fx.Int32(0)
        for step in (32, 16, 8, 4, 2, 1):
            e = lane_gather(end_k, tk + (step - 1))
            tk = tk + (e <= u).select(fx.Int32(step), fx.Int32(0))
        return tk, init_blocks + u - lane_gather(start_k, tk)

    def new_keys_in(tk, b):
        """Block b of the request ending at token tk may hold keys this launch
        inserts (a speculative verify's earlier rows)."""
        return (b + 1) * SPARSE_BLOCK > pick(seq, tk) - MAX_TOKENS

    def cols_of(tk):
        """B column l16's token (the request's token l16, the last one repeated)
        and its context."""
        col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
        return col, pick(seq, col)

    def load_raw(page, tiles):
        """Key tiles ``tiles`` (16 keys each) of a block, in flight: lane g4 =
        lane / 16 reads its key's bytes 32 g4 .. 32 g4 + 32 with two 16 B loads
        (half the load instructions of eight-byte operand reads, scattered over
        the rows: 1M context, S = 16, scores 125 -> 100 us); ``split_tiles``
        makes them MFMA operands."""
        words = []
        if const_expr(nvfp4):
            # per tile: 16 B of data (dims 32 g4 .. + 31) and the 2 B of scales
            # of its groups 2 g4, 2 g4 + 1
            for i in tiles:
                key = 16 * i + l16
                base = page * NVFP4_PAGE_BYTES
                w = fx.Vector(
                    bo.buffer_load(
                        r_ic,
                        (base + key * (HEAD_DIM // 2) + 16 * g4) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                sc = fx.Int32(
                    bo.buffer_load(
                        r_ic,
                        (base + NVFP4_SCALES_AT + key * 8 + 2 * g4) // 2,
                        vec_width=1,
                        dtype=T.i16,
                    )
                )
                words += [w[e] for e in range(4)] + [sc]
            return fx.Vector.from_elements(words, fx.Int32)
        for i in tiles:
            row = (page * INDEX_BLOCK_BYTES + (16 * i + l16) * HEAD_DIM + 32 * g4) // 4
            for h in range_constexpr(2):
                w = fx.Vector(
                    bo.buffer_load(r_ic, row + 4 * h, vec_width=4, dtype=T.i32)
                )
                words += [w[e] for e in range(4)]
        return fx.Vector.from_elements(words, fx.Int32)

    def q_operand(frags, lut=False):
        if const_expr(nvfp4):
            return q_e4m3(frags, EVEN_ODD if lut else range(8))
        return frags

    def split_tiles(raw, tiles, tab=None):
        """``load_raw``'s words -> per tile the MFMA A operands: tile i holds keys
        16 i + lane % 16, k = 32 s + 8 (lane / 16). A 4 x 4 transpose of the 8 B
        parts across the key's four lanes (permlane swaps) gives each lane part
        g4 of every chunk s. ``tab`` (nvfp4): the CTA's e4m3 table, the K operand's
        elements then in ``EVEN_ODD`` order within each dword pair."""
        out = []
        if const_expr(nvfp4 and tab is not None):
            for n in range_constexpr(len(tiles)):
                sc = raw[5 * n + 4]
                ent = [e4m3_lut_entry(tab, sc), e4m3_lut_entry(tab, sc >> 8)]
                words = []
                for s in range_constexpr(4):
                    words += fp4_dword_e4m3x8_lut(raw[5 * n + s], *ent[s // 2])
                out.append(fx.Vector.from_elements(words, fx.Int32))
            return out
        if const_expr(nvfp4):
            for n in range_constexpr(len(tiles)):
                out.append(
                    fx.Vector.from_elements(
                        [
                            w
                            for s in range(4)
                            for w in nvfp4_dword_e4m3x8(
                                raw[5 * n + s], raw[5 * n + 4] >> (8 * (s // 2))
                            )
                        ],
                        fx.Int32,
                    )
                )
            return out
        for n in range_constexpr(len(tiles)):
            d = [raw[8 * n + e] for e in range(8)]
            # part j = dwords 2 j, 2 j + 1; lanes g4 and g4 ^ 2, then g4 ^ 1, trade
            for off, pairs in ((32, ((0, 2), (1, 3))), (16, ((0, 1), (2, 3)))):
                for a, b in pairs:
                    for w in range_constexpr(2):
                        d[2 * a + w], d[2 * b + w] = permlane_swap(
                            off, d[2 * a + w], d[2 * b + w]
                        )
            out.append(
                [
                    fx.Vector.from_elements([d[2 * s], d[2 * s + 1]], fx.Int32)
                    for s in range(HEAD_DIM // 32)
                ]
            )
        return out

    def tiles_max(b, qb, col_seq, tiles, kw):
        """Max over key tiles ``tiles`` of block b (``kw``: ``split_tiles``) of this
        lane's column's causal scores (its B operand ``qb``, per k-chunk; the
        column's context ``col_seq``), the same in every lane of the column."""
        v = fx.Float32(float("-inf"))
        for n, i in enumerate(tiles):
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            if const_expr(nvfp4):  # K and q e4m3: one fp8 MFMA over all 128 dims
                c = mfma_scaled(
                    kw[n], qb, c, fx.Int32(UNIT_SCALE), fx.Int32(UNIT_SCALE)
                )
            else:
                for s in range_constexpr(HEAD_DIM // 32):
                    c = mfma_bf16(fp8x8_bf16_pk(kw[n][s]), qb[s], c)
            # lane holds keys 16 i + 4 (lane / 16) + e of column lane % 16
            key0 = b * SPARSE_BLOCK + 16 * i + 4 * g4
            for e in range_constexpr(4):
                v = fx.max(v, (key0 + e < col_seq).select(c[e], float("-inf")))
        return butterfly(v, (32, 16), fx.max)

    def fill_lut():
        """Fill the CTA's table: thread s < LUT_ENTRIES makes the entry of scale byte
        s. The caller barriers before a lookup."""
        if tid < LUT_ENTRIES:
            fx.ptr_store(e4m3_lut_row(tid), tab + tid * 2)

    def put(b, col, v):
        if (lane < 16) & (l16 < q_len):
            mb_put(iscore, col * MAX_INDEX_BLOCKS + b, v * scale)

    def emit_cp_tasks():
        """The context-parallel requests' tasks, a task a wave: B column f of pass
        p (f = 16 p + lane % 16) is head f / q_len's row f % q_len of the request,
        the key block's operands kept for every pass."""
        if const_expr(nvfp4):
            fill_lut()
        if const_expr(prepare_heads is not None):
            prepare_heads()
        if const_expr(nvfp4):
            gpu.barrier()
        tiles = range(KEY_TILES)
        seq_lane = pick(seq, fx.min(lane, tokens - 1))  # lane k: token k's context
        n_cols = TP * q_len

        def score_cp(tk, b, page):
            """Block b of the request ending at token tk, at ``page``: every head."""
            if const_expr(wait_new_keys is not None):  # noqa: SIM102
                if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                    wait_new_keys()
            kw = split_tiles(load_raw(page, tiles), tiles, tab)
            # every pass unrolled, the unused ones skipped: a runtime pass loop
            # slowed the whole kernel's short contexts (S = 1, 3k: +0.6 us)
            for p in range_constexpr(TP * MAX_TOKENS // 16):
                if p * 16 < n_cols:
                    f = p * 16 + l16
                    head = fx.min(f // q_len, TP - 1)
                    col = tk - (q_len - 1) + f % q_len
                    qb = q_operand(
                        [q_frag_head(col, s, head) for s in range(HEAD_DIM // 32)],
                        True,
                    )
                    v = tiles_max(b, qb, lane_gather(seq_lane, col), tiles, kw)
                    if (lane < 16) & (f < n_cols):
                        mb_put(
                            iscore,
                            col * MAX_INDEX_BLOCKS
                            + head * (MAX_INDEX_BLOCKS // TP)
                            + b // TP,
                            v * scale,
                        )

        # as the wave mode below: task u on wave (u // BLOCKS) % WAVES of CTA
        # u % BLOCKS, lane j decoding task c0 + step j
        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total_cp, step * 64):
            c0 = fx.Int32(c0)
            tk_j, b_j = task_block(
                fx.min(c0 + step * lane, total_cp - 1), pre_cp, first_own, TP
            )
            pages = page_of(tk_j, b_j)
            for i in range(fx.min((total_cp - c0 + step - 1) // step, 64)):
                i = fx.Int32(i)
                score_cp(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))

    # Up to BLOCKS tasks: one a CTA, a key tile a wave, the columns' maxima met in
    # LDS; the keys go out before prepare() (only the task's request's rows), so
    # their latency hides under the index q's: the scores are on the step's
    # critical path (keys after: S = 1, 3k context +1 us). More: a task a wave,
    # every row's index q first (a wave's task count differs: no barrier inside),
    # the pages of a wave's next 64 tasks fetched at once, a lane each.
    if const_expr(not BIG):  # noqa: SIM102
        if total <= BLOCKS:
            for u in range(slot, total, BLOCKS):
                tk, b = task_block(fx.Int32(u))
                page = uniform(page_of(tk, b))
                if const_expr(wait_new_keys is not None):  # noqa: SIM102
                    if new_keys_in(tk, b):
                        wait_new_keys()
                raw = load_raw(page, [wave])
                if const_expr(prepare is not None):
                    prepare(tk - (q_len - 1), tk)
                col, col_seq = cols_of(tk)
                qb = q_operand([q_frag(col, s) for s in range(HEAD_DIM // 32)])
                v = tiles_max(b, qb, col_seq, [wave], split_tiles(raw, [wave]))
                if lane < 16:
                    fx.ptr_store(v, red + (wave * 16 + l16))
                gpu.barrier()
                if wave == 0:
                    for w in range_constexpr(1, WAVES):
                        v = fx.max(v, fx.ptr_load(red + (w * 16 + l16)))
                    put(b, col, v)
                gpu.barrier()
    if total > (0 if BIG else BLOCKS):
        if const_expr(nvfp4):
            fill_lut()
        if const_expr(prepare is not None):
            prepare(fx.Int32(0), fx.Int32(tokens - 1))
        if const_expr(nvfp4):
            gpu.barrier()
        if const_expr(BIG):
            stamp(7)
            if (
                slot + BLOCKS * wave < total
            ):  # (the waves without a task need no index q)
                wait_index_q()
            stamp(8)
            seq_lane = seq_k
        else:
            seq_lane = pick(seq, fx.min(lane, tokens - 1))  # lane k: token k's context
        tiles = range(KEY_TILES)

        def score_task(tk, b, page):
            """Block b of the request ending at token tk, at ``page``."""
            if const_expr(wait_new_keys is not None):  # noqa: SIM102
                if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                    wait_new_keys()
            if const_expr(BIG):
                stamp(11)
            raw = load_raw(page, tiles)
            col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
            col_seq = lane_gather(seq_lane, col)
            qb = q_operand([q_frag(col, s) for s in range(HEAD_DIM // 32)], True)
            v = tiles_max(b, qb, col_seq, tiles, split_tiles(raw, tiles, tab))
            if const_expr(BIG):
                stamp(12)
            put(b, col, v)

        # Every wave the same task count: task u on wave
        # (u // BLOCKS) % WAVES of CTA u % BLOCKS; lane j decodes task c0 + step j
        # once -- token, block, page -- read back per task (a task's select chains
        # sat between one task's keys and the next: S = 16, +8 us).
        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total, step * 64):
            c0 = fx.Int32(c0)
            u_j = fx.min(c0 + step * lane, total - 1)
            tk_j, b_j = task_block_big(u_j) if BIG else task_block(u_j)
            pages = page_of(tk_j, b_j)
            if const_expr(BIG):
                stamp(9)
            for i in range(fx.min((total - c0 + step - 1) // step, 64)):
                i = fx.Int32(i)
                score_task(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))
        if const_expr(BIG):
            stamp(10)
    if const_expr(cp):  # noqa: SIM102
        # cold: a step without context-parallel requests runs none of it
        if unlikely(total_cp > 0):
            emit_cp_tasks()


def q_frag_from(iq, heads=ONE_INDEX_HEAD, nvfp4=False, cm=0):
    """(``q_frag``, ``q_frag_head``) reading index q from memory: [token][head]
    [HEAD_DIM] bf16, ``heads.count`` heads a token."""

    def q_frag_head(tk, s, head):
        lane = fx.thread_idx.x % 64
        return fx.Vector(
            bo.buffer_load(
                rsrc(iq),
                (
                    (tk * heads.count + head) * HEAD_DIM
                    + (
                        32 * (lane // 16) + 8 * s
                        if nvfp4
                        else 32 * s + 8 * (lane // 16)
                    )
                )
                // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        ).bitcast(fx.BFloat16)

    def q_frag(tk, s):
        return q_frag_head(tk, s, heads.own)

    return q_frag, q_frag_head


@dataclass(frozen=True)
class ScoreBuild:
    """Every parameter of a standalone score build (``atom.mono.plan.build_key``)."""

    tokens: int = field(metadata={"sym": "s"})
    init_blocks: int = field(metadata={"sym": "ib"})
    local_blocks: int = field(metadata={"sym": "lb"})
    index_heads: int = field(metadata={"sym": "ih"})
    index_own: int = field(metadata={"sym": "io"})
    sm_scale: float
    nvfp4: bool = False


def build_index_score_kernel(
    sm_scale: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    heads: IndexHeads = ONE_INDEX_HEAD,
    nvfp4: bool = False,
):
    """``@flyc.jit`` launcher of the score tasks alone (index q from memory,
    ``heads.count`` heads a token), for K4's harnesses: scores into mailbox
    ``iscore`` with K4's layer tag."""
    assert 1 <= tokens <= MAX_TOKENS
    key = ScoreBuild(
        tokens=tokens, init_blocks=init_blocks, local_blocks=local_blocks,
        index_heads=heads.count, index_own=heads.own, sm_scale=sm_scale, nvfp4=nvfp4,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)  # the launcher references it: keyed
    scale = index_scale_log2e(sm_scale)
    # the kernel body rebuilds IndexHeads from ints: an object in a closure is not
    # a cache-key value (``build_key`` keys them anyway)
    ih_count, ih_own = heads.count, heads.own

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 16, 16]

    kernel_name = kernel_symbol("minimax_m3_index_score", **symbol_params(key))

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def index_score_kernel(
        iq: Int64,
        index_cache: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red = lds.red.ptr
        mbox = Mailbox(layer)
        mb_put = mbox.put
        k_heads = IndexHeads(ih_count, ih_own)
        q_frag, q_frag_head = q_frag_from(iq, k_heads, nvfp4)
        emit_index_scores(
            fx.block_idx.x, tid % 64, uniform(tid // 64), red, mb_put,
            Addr("iscore", Space.SCRATCH, iscore),
            q_frag, tokens, q_len, init_blocks, local_blocks, scale, index_cache,
            block_table, seq_lens, bt_width, heads=k_heads, q_frag_head=q_frag_head,
            nvfp4=nvfp4,
        )  # fmt: skip

    @launcher(kernel_name)
    def launch_index_score(
        iq: Int64,
        index_cache: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key  # every build parameter in the JIT cache key
        index_score_kernel(
            iq,
            index_cache,
            block_table,
            seq_lens,
            bt_width,
            q_len,
            iscore,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch_index_score
