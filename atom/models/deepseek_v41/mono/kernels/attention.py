# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K2a's decode sparse attention: the scores' tasks put raw scores, a merge
task lands its values before one poll of them and runs the softmax itself, on
the original path's arithmetic: ``qk_scale`` and ``exp2``, p rounded to bf16
before PV and the output to bf16 before the inverse RoPE. Only the sums' order
differs from the original split + reduce (``_paged_decode_split_kernel`` /
``_paged_decode_reduce_kernel``).

The rank's heads (``Dims.heads``) are tiles of 16. A ragged tile's (TP8: 8
heads) dead heads read the last live head and write nothing.

A token's keys sit in ``ATTENTION_KEYS_MAX`` = 640 fixed slots
(``config_refusal``): the selection's 512 then the window's 128, or in a build
whose keys hold no selection (``AttnPostBuild.attn_sel``: the DSpark draft's
window and block) the window's 640, in chunks of 64 (``attention_plan``: a
pass's kind its position; ``c["sel_slots"]`` the build's selection slots).

score (S x tiles x CHUNKS tasks, task (t, tile, chunk)): waves 0 .. 3 take 16
slots each, their rows through the wave's LDS; C[key][head] = 16 chained
16x16x32 bf16 MFMAs over D (A = kv rows, B = q), x qk_scale, NEG at a slot
without a key -> AS [t][tile][chunk][head][slot].

merge (S x tiles x 16 / MH x 8 tasks, task (t, tile, head group, 64-dim
slice), MH ``attention_plan.merge_heads``): the slice of every key's value
into LDS [8-column block][key][8 columns], then the group's scores of every
chunk in one poll; m = the max over the keys, with the sink's; p = exp2(s - m),
the denominator sum(p) + exp2(sink - m) in f32, bf16 p into LDS [head][key];
P V on 16x16x16 bf16 MFMAs (A = V^T by transposing LDS reads, B = p), wave w
the 16 dims 16 (w % 4) .. over key half w // 4; o / den, bf16 ->
``att["out"]`` LDS [head][64], which the same task's inverse-RoPE quant reads.

An FP8 pool (``kv_fp8``: MXFP8 rows, 512 E4M3 in the data plane, 16 E8M0 in
the scale plane) follows the
original FP8 path (``mla_sparse_prefill_v41``): QK on FP8, PV on BF16. score:
q quantized to MXFP8 by the KV row's rule once a CTA (``_q_shared``), 4 chained
16x16x128 scaled MFMAs over D on the rows as stored; merge: a row's 8 columns
dequantized by their group's code (exact in bf16) into the same bf16 tiles.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import ATTENTION_KEYS_MAX
from atom.models.deepseek_v41.mono.kernels.dims import (
    HEAD_DIM,
    HEAD_TILE,
    kvrow_scale0,
)
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.step_meta import KMETA
from atom.mono.device.mx import (
    FP8_MAX,
    UNIT_SCALE,
    clamp_fp8,
    code_ceil,
    mfma_scaled,
    pow2,
    rcp_pow2,
)
from atom.mono.device.ops import (
    bf16_round,
    butterflies,
    butterfly,
    fp8_pack4,
    hw_exp2,
    ld_dev,
    ld_i32,
    mfma_bf16,
    mxfp8_to_bf16x8,
    readlane,
    readlane64,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.sync import Quad
from atom.mono.plan.execution import BLOCKS, THREADS
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import as_ir_value

CH, CHUNKS = aplan.CH, aplan.CHUNKS  # keys a chunk, chunks a token
BK = 16  # keys a score wave (an MFMA's 16 rows)
SCORE_WAVES = CH // BK  # a score task's waves with keys
# a score wave's 16 key rows in its LDS: a row's 1 KB and 16 B of padding (a
# lane's read-back of 16 rows at one column hits distinct banks)
SCORE_LDS_ROW = HEAD_DIM // 2 + 4  # words
SCORE_LDS_WORDS = BK * SCORE_LDS_ROW  # a wave's
# the layer kernel's q tile in LDS (``_q_tile``): a head's row padded alike, so
# the 16 heads a lane group reads at one column hit distinct banks
Q_LDS_ROW = HEAD_DIM // 2 + 4  # words
Q_LDS_WORDS = HEAD_TILE * Q_LDS_ROW
# an FP8 pool's q operands in LDS (``_q_shared``): [K step][lane] 8 words (32 B
# aligned for the vector access), then [K step][lane] their codes
Q_LDS_CODES = HEAD_DIM // 128 * 64 * 8
SCORE_Q_WORDS = Q_LDS_CODES + HEAD_DIM // 128 * 64
# The layer kernel's window rows of this step as KVT's tagged pairs, where it
# hands them over so (``layer.window_rows_tagged``, ``attn_pre._ring_row_fp8``):
# a row's FP8 data words a pair each, its group's E8M0 code in the tag's high
# half. A one-chunk score task's waves 4 .. 7 hold no keys: wave 4 + i polls
# the step rows of wave i's keys into its own LDS rows (``_poll_step_rows``),
# a row's data words, then its 4 code words (``_row_codes``' layout) from word
# KVT_CODES
KVT_ROW_PAIRS = HEAD_DIM // 4
KVT_LDS = SCORE_WAVES * SCORE_LDS_WORDS
KVT_CODES = HEAD_DIM // 4
assert KVT_CODES + HEAD_DIM // 128 <= SCORE_LDS_ROW
SLICE = 64  # a merge task's dims: two FP8 quant groups
SLICES = HEAD_DIM // SLICE
COLS = 32  # an FP8 quant group's dims
# the merge's LDS (words): V [8-column block][key][8 columns] (4 words a key's
# block), P [head][key] (bf16 pairs), the heads' chunk maxima and partial sums,
# the second key half's P V and the output [head][dim]
# V's block: V_SLOTS key slots of 4 words, 10 past ATTENTION_KEYS_MAX
# (a stride of 10 400 B): unpadded (10 240 B = 0 mod 256 B) a 16-B landing
# store's 8 lanes of one key, 8 blocks, hit the same banks (8-way); padded,
# distinct ones. The PV's transposing reads stay 2-way either way. The
# addresses keep the unpadded ones' form (slots, then words): the keys'
# offsets stay the stores' and reads' immediates
V_SLOTS = ATTENTION_KEYS_MAX + 10
MERGE_V_WORDS = SLICE // 8 * V_SLOTS * 4
# words a head's P row, padded: the PV reads 16 heads at one key
P_ROW = ATTENTION_KEYS_MAX // 2 + 4
MERGE_P_WORDS = HEAD_TILE * P_ROW
MERGE_STAT_WORDS = HEAD_TILE * CHUNKS
MERGE_SUM_WORDS = HEAD_TILE * CHUNKS
MERGE_TREE_WORDS = SLICE // 16 * 64 * 4
MERGE_OUT_WORDS = HEAD_TILE * SLICE
PV_GROUP = 4  # the PV's 16-slot blocks a branch (``_merge_pv``)


def merge_groups(tokens) -> int:
    """Head groups a 16-head tile in the merge (``attention_plan.merge_heads``)."""
    return HEAD_TILE // aplan.merge_heads(tokens)


def merge_tasks(tokens, tiles) -> int:
    """The merge stage's tasks a step: (token, tile, head group, slice)."""
    return tokens * tiles * merge_groups(tokens) * SLICES


NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
ROW_BYTES = HEAD_DIM * 2


def scratch_pairs(tokens: int, d) -> dict[str, int]:
    """AS: every (token, tile, chunk, head, key)'s raw score, a pair each."""
    return {"as": tokens * d.head_tiles.count * CHUNKS * HEAD_TILE * CH}


def tile_task(task, per, tiles):
    """Task ``task`` of a stage with ``per`` tasks a (token, head tile) -> (token,
    tile, the task's index in its tile)."""
    if tiles.count == 1:
        return task // per, 0, task % per
    return task // (per * tiles.count), task // per % tiles.count, task % per


def tile_head(d, tile, h):
    """Head h of ``tile`` -> (the rank's head it reads, live): a ragged tile's
    dead heads read the last live one."""
    tiles = d.head_tiles
    if tiles.count == 1 and not tiles.ragged:
        return h, True
    head = tile * HEAD_TILE + h
    if not tiles.ragged:
        return head, True
    return fx.min(head, d.heads - 1), head < d.heads


def _plane_ptr(c, row, scale=False):
    """An FP8 key row's data (or ``scale``) plane row. The row space
    concatenates this owner's CSA and this layer's SWA; in the layer kernel a
    row of this step is ``row`` = -1 - j for row j of ``kvrow``
    (``dims.kvrow_bytes``)."""
    a = c["args"]
    width = HEAD_DIM // 32 if scale else HEAD_DIM
    is_window = row >= a["ring_off"]
    base = is_window.select(
        a["window_scale"] if scale else a["window"],
        a["pool_scale"] if scale else a["pool"],
    )
    local = is_window.select(row - a["ring_off"], row)
    pool_row = base + fx.Int64(local) * width
    if const_expr(not c["fused"]):
        return pool_row
    plane = a["kvrow"] + (kvrow_scale0(c["S"]) if scale else 0)
    step_row = plane + fx.Int64(-1 - row) * width
    return fx.Int64((row < 0).select(step_row, pool_row))


def _step_cm(c):
    """The cache policy of a key row read that may be one of this launch's
    (``kvrow``): ``hand_cm`` (CM_DEV) where the launch rereads ``kvrow`` every
    iteration -- the run kernel / pair (``hand_cm`` set) without KVT; else 0,
    the default policy (the single-layer kernel reads it once past its launch's
    invalidation; KVT builds read the window rows from their tagged pairs)."""
    if const_expr(not c["fused"] or c.get("kvt_rows", False)):
        return 0
    return c["args"].get("hand_cm", 0)


def _ld_step(addr, words, align):
    """``words`` i32 (1, 2 or 4) at the 64-bit global address ``addr`` at device
    scope: relaxed agent-scope atomic loads (gfx94x/95x: ``global_load`` sc1,
    never a CU's resident line). Two words load as one i64; four as two (an
    atomic access is at most 64 bits)."""
    i32 = ir.IntegerType.get_signless(32)
    if words == 4:
        lo, hi = (fx.Vector(_ld_step(fx.Int64(addr) + 8 * h, 2, 8)) for h in range(2))
        return fx.Vector.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int32)
    ty = {1: i32, 2: ir.IntegerType.get_signless(64)}[words]
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    v = llvm.LoadOp(
        ty, ptr.result, alignment=align, ordering=llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    ).result
    if words == 2:
        v = llvm.BitcastOp(ir.VectorType.get([2], i32), v).result
    return v


def _row_ptr(pool, row, byte, row_bytes):
    """Pool row ``row`` at ``byte``: 64-bit, a row index times the row passing
    2^32 (``pool_index.row_offset``), so neither a buffer offset nor an i32
    product can carry it."""
    return pool + fx.Int64(row) * row_bytes + fx.Int64(byte)


def _key_base(c, pool, row, row_bytes):
    """Key row ``row``'s address (``_key_row``): a pool row, or in the layer
    kernel a row of this step, ``row`` = -1 - j for row j of ``kvrow``."""
    if const_expr(not c["fused"]):
        return _row_ptr(pool, row, 0, row_bytes)
    step_row = c["args"]["kvrow"] + fx.Int64(-1 - row) * row_bytes
    return fx.Int64((row < 0).select(step_row, _row_ptr(pool, row, 0, row_bytes)))


def _new_rows(c, t):
    """The layer kernel: token t's window keys written in this launch, the
    last ``n`` of its count, and the step row of its last key: (n, that row).
    Causal (the target): its own and its request's earlier rows of the step
    (a request's rows consecutive, their slot row base, ``kmeta`` word 3,
    alike), its last key its own. A block of ``block_rows`` (the DSpark
    draft's: every row sees all of it, a request's block at rows
    ``[r * block_rows, (r + 1) * block_rows)``): the whole block, its last
    key the block's last row. None in a K2 launch, whose keys are all in the
    pool."""
    if const_expr(not c["fused"]):
        return None
    block = c["block_rows"]
    if const_expr(block):
        return fx.Int32(block), t - t % block + (block - 1)
    kmeta = c["args"]["kmeta"]
    base = ld_i32(kmeta, t * KMETA + 3)
    run, own = fx.Int32(1), fx.Int32(1)
    for k in range(1, c["S"]):
        same = (t - k >= 0) & (ld_i32(kmeta, fx.max(t - k, 0) * KMETA + 3) == base)
        own = same.select(own, fx.Int32(0))
        run = run + own
    return run, t


def _slot_valid(meta, slot):
    """Whether key slot ``slot`` of the token (``_keys``' ``meta``) holds a key
    (``attention_plan.slot_valid``)."""
    n_sel, _, _, count, _, _, sel_slots, _ = meta
    return aplan.slot_valid(slot, n_sel, count, sel_slots)


@traced
def await_step_rows(c, meta, last_slot):
    """The layer kernel: before this wave loads key slots up to ``last_slot``
    of a token (``_keys``' ``meta``), this launch's window rows (KRDY, every
    token) written if one of them is among them, polled at K1's tag (the
    wave's own loads follow: no barrier). The selected keys and the older
    window rows, from earlier launches, need no wait."""
    if const_expr(c["fused"]):
        _, _, _, count, _, (n_new, _), sel_slots, _ = meta
        if last_slot >= sel_slots + count - n_new:
            c["k1_poll"]([(c["krdy"], c["lane"] % c["S"], 1)])


def _has_step_rows(c, meta, key0):
    """Whether key slots key0 .. + 16 of a token (``_keys``' ``meta``)
    hold one of this launch's window rows (``_window_row``: the last
    ``n_new`` of its ``count``)."""
    _, _, _, count, _, (n_new, _), sel_slots, _ = meta
    return (key0 < sel_slots + count) & (key0 + BK > sel_slots + count - n_new)


@traced
def _poll_step_rows(c, meta, key0):
    """Wave 4 + i (no keys of its own) lands the step rows among wave
    i's key slots key0 .. + 16 in its own LDS rows: lane (h, cc) of each
    pass p polls 16 B (two quads, column chunk cc) of step row 2 p + h, a row
    the slots do not hold out of range (``Mailbox.poll_need``); the words to
    the row of its slot, and from the tags' codes (chunk cc: group cc // 2)
    the row's code words, word j of groups 4 j .. 4 j + 3 (chunks 8 j, + 2,
    + 4, + 6: lanes xor 2, then xor 4). ``_q_shared``'s first barrier hands
    them to wave i (``_rows_from_lds``)."""
    lane, wave, s = c["lane"], c["wave"], c["S"]
    _, _, _, count, _, (n_new, last), sel_slots, _ = meta
    h, cc = lane // 32, lane % 32
    area = c["att"]["score_kv"] + wave * SCORE_LDS_WORDS
    specs, needs, rows = [], [], []
    for p in range_constexpr(-(-s // 2)):
        k = 2 * p + h
        # step row k: window slot count - 1 - (last - k) (``_window_row``)
        r = sel_slots + count - 1 - (last - k) - key0
        need = (k < s) & (k <= last) & (k > last - n_new) & (r >= 0) & (r < BK)
        q0 = fx.min(k, s - 1) * (KVT_ROW_PAIRS // 2) + 2 * cc
        specs += [(c["kvt"], Quad(q0 + e), 2) for e in range(2)]
        needs += [need, need]
        rows.append((need, r))
    got = c["k1_poll_need"](specs, needs, coded=True)
    for p in range_constexpr(len(rows)):
        need, r = rows[p]
        (w0, code), (w1, _) = got[2 * p], got[2 * p + 1]
        pair = code | (xshfl(code, 2) << 8)
        cw = pair | (xshfl(pair, 4) << 16)
        if need:
            at = area + r * SCORE_LDS_ROW
            fx.ptr_store(fx.Vector.from_elements([*w0, *w1], fx.Int32), at + 4 * cc)
            if cc % 8 == 0:
                fx.ptr_store(cw, at + (KVT_CODES + cc // 8))


@traced
def _q_tile(c, t, tile):
    """The layer kernel: token t's q of a head tile from wq_b's tagged pairs
    (QP, K1's tag), polled once a CTA into LDS (``red``, which the score stage
    has no other use for: exactly a tile's bf16 q, a head a ``Q_LDS_ROW`` row;
    a thread 16 values) -> its words there."""
    d, tid = c["d"], c["tid"]
    q = fx.recast_iter(fx.Int32, c["red"])
    # a thread 8 words of one head's row: 32 threads a row
    assert THREADS * 8 == HEAD_TILE * HEAD_DIM // 2
    row_words = HEAD_DIM // 2 // 8
    # a ragged tile's dead heads poll the last live one's row (``tile_head``):
    # past it the pairs are the next token's, or past the step's, never written
    head = tile_head(d, tile, tid // row_words)[0]
    base = (t * d.heads + head) * HEAD_DIM + tid % row_words * 16
    got = c["k1_poll"]([(c["qp"], Quad(base // 4 + k), 2) for k in range(4)])
    words = [w for pair in got for w in pair]
    at = tid // row_words * Q_LDS_ROW + tid % row_words * 8
    fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), q + at)
    gpu.barrier()
    return q


def _q_tile_words(q, head, g):
    """``_q_words`` of head ``head`` of the tile in LDS ``q`` (``_q_tile``)."""
    return [
        fx.Vector(
            fx.ptr_load(
                q + (head * Q_LDS_ROW + (ch * 32 + 8 * g) // 2),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        for ch in range(HEAD_DIM // 32)
    ]


def _dequant8(w0, w1, code):
    """Eight E4M3 of two dwords times their E8M0 ``code``, as bf16 pairs (exact:
    a power of two times an E4M3 is a bf16) -- a BF16 pool's operand words."""
    return mxfp8_to_bf16x8(w0, w1, pow2(code)).bitcast(fx.Int32)


def _q_load4(c, at):
    """Four words of q at word ``at`` (q's [token][head][dim] layout): of
    ``q``, or in the layer kernel polled from wq_b's tagged pairs (QP, K1's
    tag), as ``_q_tile`` does."""
    if const_expr(not c["fused"]):
        return fx.Vector(
            bo.buffer_load(rsrc(c["args"]["q"]), at, vec_width=4, dtype=T.i32)
        )
    got = c["k1_poll"]([(c["qp"], Quad(at // 2 + k), 2) for k in range(2)])
    return fx.Vector.from_elements([w for pair in got for w in pair], fx.Int32)


def _q_step(c, row, g, st):
    """K step ``st`` of head ``row``'s q (bf16, ``row`` its word offset,
    ``_q_load4``) as the 16x16x128 scaled MFMA's B operand. Lane group j holds dims 16 j .. + 16
    of the step, then 16 from 64 past them, but its E8M0 scales block j (dims
    32 j .. + 32): a half's block, here dims 128 st + 16 g (+ 64) and the 16
    of lane group g ^ 1 (xor 16), takes their amax, and block g's code comes
    from a half of this pair or of the one xor 32. Quantized by the KV row's
    rule (amax floored at AMAX_FLOOR, the ceil code, an exact x / scale, E4M3
    RNE): 8 words, and block g's code."""
    words, codes = [], []
    for half in range_constexpr(2):
        at = row + (st * 128 + half * 64 + 16 * g) // 2
        v = [
            _q_load4(c, at + 4 * i).bitcast(fx.BFloat16).to(fx.Float32)
            for i in range(2)
        ]
        v = [v[i // 8][i % 8] for i in range(16)]
        amax = fx.Float32(0.0)
        for i in range_constexpr(16):
            amax = fx.max(amax, abs(v[i]))
        amax = butterfly(amax, (16,), fx.max)
        code = code_ceil(
            fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX)
        )
        inv = rcp_pow2(code)
        words += [
            fp8_pack4(*[clamp_fp8(v[4 * k + i] * inv) for i in range(4)])
            for k in range(4)
        ]
        codes.append(fx.Int32(code))
    lo, hi = codes
    mine = (g == 0).select(
        lo,
        (g == 1).select(xshfl(lo, 32), (g == 2).select(xshfl(hi, 32), hi)),
    )
    return words, mine


@traced
def _q_shared(c, row, g):
    """The task's q operands (``_q_step``, its words straight from ``q`` or
    from wq_b's pairs: no LDS staging first) for every K step, each quantized
    once a CTA: wave w < 4 its step w into LDS ``score_q`` (``Q_LDS_CODES``),
    then every wave reads all four. The second barrier keeps a fast wave's
    next task from overwriting what a slow one has yet to read."""
    lane, wave = c["lane"], c["wave"]
    q = c["att"]["score_q"]
    if wave < HEAD_DIM // 128:
        words, code = _q_step(c, row, g, wave)
        fx.ptr_store(
            fx.Vector.from_elements(words, fx.Int32), q + (wave * 64 + lane) * 8
        )
        fx.ptr_store(fx.Int32(code), q + (Q_LDS_CODES + wave * 64 + lane))
    gpu.barrier()
    out = []
    for st in range_constexpr(HEAD_DIM // 128):
        out.append(
            (
                fx.Vector(
                    fx.ptr_load(
                        q + (st * 64 + lane) * 8,
                        result_type=fx.Vector.make_type(8, fx.Int32),
                    )
                ),
                fx.Int32(
                    fx.ptr_load(q + (Q_LDS_CODES + st * 64 + lane), result_type=T.i32)
                ),
            )
        )
    rocdl.s_waitcnt(lgkmcnt=0)
    gpu.barrier()
    return out


@traced
def _rows_to_lds(c, pool, slot, kv):
    """A split's 16 key rows (row ``slot`` of lane k, k < 16) whole into this
    wave's LDS ``kv``, 16 B a lane, read back in the MFMA's layout
    (``_rows_from_lds``). Descriptors are row-local because the pool can exceed
    4 GiB. FP8: the data plane only, lanes 0..31 (its scales: ``_row_codes``);
    each lane's row's address is computed once (``slot``: its own row), a
    row's read from its lane before the lanes split (a lane branch would have
    the compiler compute it there, on its lanes only)."""
    if const_expr(c["kv_fp8"]):
        data = _plane_ptr(c, slot)
    # the run kernel / pair without KVT: its rows of this step at device scope
    # (``_step_cm``; the DMA's aux), else as before
    cm = _step_cm(c)
    for r in range_constexpr(BK):
        if const_expr(c["kv_fp8"]):
            row_data = readlane64(data, r)
            if c["lane"] < HEAD_DIM // 16:
                if const_expr(cm):
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc(row_data, HEAD_DIM),
                        fx.to_llvm_ptr(kv + r * SCORE_LDS_ROW),
                        16, c["lane"] * 16, 0, 0, cm,
                    )  # fmt: skip
                else:
                    rocdl.buffer_load_to_lds(
                        rsrc(row_data, HEAD_DIM),
                        fx.to_llvm_ptr(kv + r * SCORE_LDS_ROW),
                        c["lane"] * 16,
                        size_bytes=16,
                    )
        else:
            row = readlane(slot, r)
            if const_expr(cm):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc(_key_base(c, pool, row, ROW_BYTES)),
                    fx.to_llvm_ptr(kv + r * SCORE_LDS_ROW),
                    16, c["lane"] * 16, 0, 0, cm,
                )  # fmt: skip
            else:
                rocdl.buffer_load_to_lds(
                    rsrc(_key_base(c, pool, row, ROW_BYTES)),
                    fx.to_llvm_ptr(kv + r * SCORE_LDS_ROW),
                    c["lane"] * 16,
                    size_bytes=16,
                )


def _row_codes(c, slot):
    """FP8 row ``slot``'s 16 E8M0 codes (a word a K step of 128), straight
    into this lane's registers: lane l's row is l % 16 (``_rows_to_lds``)."""
    if const_expr(_step_cm(c)):
        return fx.Vector(_ld_step(_plane_ptr(c, slot, scale=True), 4, 16))
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
        _plane_ptr(c, slot, scale=True),
    )
    return fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))


def _rows_from_lds(c, kv, codes=None, slot=None):
    """The split's key operands back from LDS ``kv``, lane l row l % 16: a BF16
    row's bytes (ch 32 + 8 (l // 16)) 2, a K chunk ch each; an FP8 row's
    scaled-MFMA A operand (``_q_step``'s layout) -- K step st's 16 bytes from
    128 st + 16 (l // 16), then 16 from 64 past them -- with ``codes``
    (``_row_codes``). ``slot``: the lane's row, one of this launch's
    (< 0) read with its codes where wave 4 + i landed it (``_poll_step_rows``)."""
    lane = c["lane"]
    g = lane // 16
    row = kv + (lane % BK) * SCORE_LDS_ROW
    if const_expr(slot is not None):
        step = slot < 0
        row = row + step.select(fx.Int32(KVT_LDS), fx.Int32(0))
        landed = fx.Vector(
            fx.ptr_load(row + KVT_CODES, result_type=fx.Vector.make_type(4, fx.Int32))
        )
        codes = fx.Vector.from_elements(
            [step.select(landed[e], codes[e]) for e in range(4)], fx.Int32
        )
    if const_expr(not c["kv_fp8"]):
        return [
            fx.Vector(
                fx.ptr_load(
                    row + 4 * g + ch * 16, result_type=fx.Vector.make_type(4, fx.Int32)
                )
            )
            for ch in range(HEAD_DIM // 32)
        ]
    steps = []
    for st in range_constexpr(HEAD_DIM // 128):
        lo, hi = (
            fx.Vector(
                fx.ptr_load(
                    row + st * 32 + half * 16 + 4 * g,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            for half in range(2)
        )
        steps.append(
            fx.Vector.from_elements(
                [lo[e] for e in range(4)] + [hi[e] for e in range(4)], fx.Int32
            )
        )
    return steps, codes


def _ld_row4(c, pool, row, col):
    """Row ``row``'s 8 columns from ``col``, as loaded (``_tile_words`` turns
    them into bf16 pairs where they are used, so the loads stay in flight):
    a BF16 row's 16 B, or an FP8 row's 8 B (their scale: ``_ld_codes``)."""
    if const_expr(_step_cm(c)):
        if const_expr(not c["kv_fp8"]):
            return fx.Vector(_ld_step(_key_base(c, pool, row, ROW_BYTES) + fx.Int64(col * 2), 4, 16))
        return fx.Vector(_ld_step(_plane_ptr(c, row) + fx.Int64(col), 2, 8))
    if const_expr(not c["kv_fp8"]):
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
            _key_base(c, pool, row, ROW_BYTES) + fx.Int64(col * 2),
        )
        return fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))
    values = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 8),
        _plane_ptr(c, row) + fx.Int64(col),
    )
    return fx.Vector(fx.ptr_load(values, result_type=fx.Vector.make_type(2, fx.Int32)))


def _ld_codes(c, row, dc):
    """FP8 row ``row``'s E8M0 codes of column groups 4 (dc // 4) .. + 4, a byte
    each in one word: a PV task's consecutive groups take one load a row."""
    if const_expr(_step_cm(c)):
        return fx.Int32(_ld_step(_plane_ptr(c, row, scale=True) + fx.Int64(dc // 4 * 4), 1, 4))
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
        _plane_ptr(c, row, scale=True) + fx.Int64(dc // 4 * 4),
    )
    return fx.Int32(fx.ptr_load(ptr, result_type=T.i32))


def _tile_words(c, tile, code=None):
    """An ``_ld_row4`` result as 4 bf16-pair words (an FP8 one by its group's
    E8M0 ``code``)."""
    if const_expr(not c["kv_fp8"]):
        return tile
    return _dequant8(tile[0], tile[1], code)


def _keys(c, t):
    """Token t's key slots (``attention_plan``) and what ``_key_row`` needs:
    its selected count (the selection's valid ids, ``index_plan.selected`` of
    its scorer's bound), its first window position's ring slot and its slot's
    row base (word 2 of its metadata, the block table offset, is the
    indexer's: ``index_score._selected_rows``), its window count and, in the
    layer kernel, its window rows of this step (``_new_rows``)."""
    a = c["args"]
    first, count, base = [ld_i32(a["kmeta"], t * KMETA + i) for i in (0, 1, 3)]
    n_sel = ip.selected(fx.Int32(ld_dev(a["sbound"], t)))
    first_slot = first % a["ring_slots"]
    new = _new_rows(c, t)
    fresh = None
    if const_expr(c.get("cp_ratio")):
        # a compressing layer kernel: (ratio, the request's first position of
        # the step, the token's) -- which selected entries this launch writes;
        # the request's rows before t and t itself: ``new`` less those past it
        pos = ld_i32(a["pos"], 2 * t)
        n_new, last = new
        fresh = (c["cp_ratio"], pos - (n_new - (last - t) - 1), pos, c["S"])
    return n_sel, first_slot, base, count, t, new, c["sel_slots"], fresh


@traced
def await_selection(c, t):
    """A layer that selects: its token t's selection is written this launch. A
    build without the indexer (``index_bound_max`` 0: the DSpark draft) has no
    selection to wait for. A compressing layer kernel: this launch's new
    entries too (every compress plan row's CRDY, a lane each), before any of
    them is read from ``kvrow``."""
    fresh = c.get("cp_ratio")
    if const_expr("irdy" in c):  # noqa: SIM102
        if c["args"]["iact"] != 0:
            c["poll"]([(c["irdy"], t, 1)])
    if const_expr(fresh):  # noqa: SIM102
        if c["cp_ncap"] > 0:
            c["k1_poll"]([(c["crdy"], fx.min(c["lane"], c["cp_ncap"] - 1), 1)])


def _key_row(a, t, meta, slot):
    """Pool row of token t's key slot ``slot`` (``attention_plan``): a
    selection slot's (``_selected_row``, its word loaded here) or a window
    slot's (``_window_row``). For a slot whose kind is not known at trace time
    (the score's: its chunk is the task's); a pass over a known kind calls
    those directly."""
    at = t * a["topk"] + fx.min(slot, a["topk"] - 1)
    word = _sel_word(a, meta[7] is not None, at)
    sel_slots = meta[6]
    return fx.Int32(
        (slot < sel_slots).select(
            _selected_row(meta, slot, word), _window_row(a, t, meta, slot - sel_slots)
        )
    )


def _sel_word(a, fresh, at):
    """Selection word ``at`` as ``_selected_row`` takes it: (its pool row, and
    in a compressing layer kernel -- ``fresh`` -- its compressed index,
    ``isel``)."""
    ci = ld_dev(a["isel"], at) if fresh else None
    return ld_dev(a["sel"], at), ci


def _selected_row(meta, slot, word):
    """Selection slot ``slot``'s pool row (``word``: the selecting layer's
    indexer's ``(pool row, compressed index)``, ``index_score._selected_rows``),
    row 0 past the token's selection (its value masked; a layer without one
    passes a dummy ``sel``). A compressing layer kernel's entry this launch
    writes for the token -- index in [p_first // ratio, (pos + 1) // ratio):
    completed at or before its position, not before the step -- is
    -1 - (S + r), r the step row completing it (<= t): its ``kvrow`` row
    (``_key_base``)."""
    n_sel, fresh = meta[0], meta[7]
    row, ci = word
    if const_expr(fresh is not None):
        ratio, p_first, pos, s = fresh
        t = meta[4]
        step_row = -1 - (s + t - (pos - (ratio * ci + ratio - 1)))
        this_step = (ci >= p_first // ratio) & (ci < (pos + 1) // ratio)
        row = this_step.select(step_row, row)
    return fx.Int32((slot < n_sel).select(row, 0))


def _window_row(a, t, meta, w):
    """Window slot w's row (oldest first): the ring row of the first window
    position's slot plus w; in the layer kernel the last ``n_new`` of its
    ``count`` this step's (``_new_rows``), -1 - its step row (``_key_base``),
    counted back from the last key's."""
    _, first_slot, base, count, _, new, _, _ = meta
    # a window key's offset is below the window, which the ring holds: its
    # slot past the first one's wraps at most once. A slot past the token's
    # count (a chunk's tail, its value masked) takes the last key's row: its
    # load must stay in the ring, which a small one (the draft's) would leave
    slot = first_slot + fx.min(w, fx.max(count - 1, 0))
    slot = (slot >= a["ring_slots"]).select(slot - a["ring_slots"], slot)
    row = base + a["ring_off"] + slot
    if const_expr(new is None):
        return fx.Int32(row)
    n_new, last = new
    step = (w >= count - n_new) & (w < count)
    step_row = -1 - (last - (count - 1 - w))
    return fx.Int32(step.select(step_row, row))


def only_live(cond, live):
    """``cond`` and ``live`` (``tile_head``'s; True: every head of the tile is)."""
    return cond if live is True else cond & live


def _as(d, t, tile, chunk, head, key):
    """AS's pair of token t's head tile, chunk, head (of the tile), key (of the
    chunk)."""
    return (
        ((t * d.head_tiles.count + tile) * CHUNKS + chunk) * HEAD_TILE + head
    ) * CH + key


def score_tasks(tokens, tiles) -> tuple[int, int]:
    """(chunks a score task, the stage's tasks): ``attention_plan.score_chunks``."""
    per = aplan.score_chunks(tokens, tokens * tiles * CHUNKS, BLOCKS)
    return per, tokens * tiles * CHUNKS // per


@traced
def stage_attn_score(c, task, per):
    """Score task ``task`` (token, head tile, ``per`` chunks from per x its
    index): waves 4 i .. 4 i + 3 chunk i's 16-key rows through their LDS
    (``_rows_to_lds``), in flight through q's wait; the other waves join the
    CTA's barriers only. A chunk without a key of the token
    (``attention_plan.chunk_written``) is not written."""
    lane, wave, d = c["lane"], c["wave"], c["d"]
    a = c["args"]
    g = lane // 16
    head = lane % 16
    t, tile, cg = tile_task(task, CHUNKS // per, d.head_tiles)
    qh, live = tile_head(d, tile, head)
    # the FP8 pool's original attention reads q as MXFP8 (``rope_quant_window``'s
    # ``query_fp8``): quantized before a selection's wait, which it does not
    # depend on; without one, under the rows' DMA
    q_first = c["kv_fp8"] and "irdy" in c
    kvt_rows = c.get("kvt_rows", False)
    # KVT's step rows: a one-chunk task's waves 4 .. 7 free to poll them, the
    # FP8 pool's q quantized meanwhile
    assert not kvt_rows or (per == 1 and c["kv_fp8"] and not q_first)
    if const_expr(q_first):
        q8 = _q_shared(c, (t * d.heads + qh) * (HEAD_DIM // 2), g)
    # the token's metadata (the host's) before the selection's wait: after it
    # only the selection words' loads stand before the rows' DMA
    meta = _keys(c, t)
    await_selection(c, t)
    n_sel, count = meta[0], meta[3]
    sel_slots = c["sel_slots"]
    written = [
        aplan.chunk_written(cg * per + i, n_sel, count, sel_slots) for i in range(per)
    ]
    if written[0] if per == 1 else written[0] | written[1]:
        slot_w = fx.min(wave, SCORE_WAVES * per - 1)
        chunk = cg * per + slot_w // SCORE_WAVES
        key0 = chunk * CH + slot_w % SCORE_WAVES * BK
        slot = _key_row(a, t, meta, key0 + head)
        kv = c["att"]["score_kv"] + slot_w * SCORE_LDS_WORDS
        keyed = (wave < SCORE_WAVES * per) & aplan.chunk_written(
            chunk, n_sel, count, sel_slots
        )
        codes = fx.Vector.filled(4, 0, fx.Int32)
        if keyed:
            if const_expr(not kvt_rows):
                await_step_rows(c, meta, key0 + BK - 1)
            _rows_to_lds(c, a["pool"], slot, kv)
            if const_expr(c["kv_fp8"]):
                codes = _row_codes(c, slot)
        if const_expr(kvt_rows):
            # wave 4 + i: the step rows of wave i's keys, from KVT
            key_i = chunk * CH + (wave - SCORE_WAVES) * BK
            if (wave >= SCORE_WAVES) & _has_step_rows(c, meta, key_i):
                _poll_step_rows(c, meta, key_i)
        if const_expr(q_first):
            qws = q8
        elif const_expr(c["kv_fp8"]):
            qws = _q_shared(c, (t * d.heads + qh) * (HEAD_DIM // 2), g)
        elif const_expr(c["fused"]):
            # the layer kernel: the tile's q from K1's pairs, in LDS
            qws = _q_tile_words(_q_tile(c, t, tile), qh - tile * HEAD_TILE, g)
        else:
            qws = _q_words(a, d, t, qh, g)
        if keyed:
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(kvt_rows):
                kws = _rows_from_lds(c, kv, codes, slot)
            else:
                kws = _rows_from_lds(c, kv, codes)
            # the whole read-back before the first MFMA (left to the scheduler,
            # each MFMA waited on its own chunk's read)
            rocdl.s_waitcnt(lgkmcnt=0)
            _score_keys(c, t, tile, chunk, key0, qh, live, meta, kws, qws)
        # the next task's rows overwrite the LDS this one read
        gpu.barrier()


def _q_words(a, d, t, qh, g):
    """Token t's q of head ``qh``, this lane's 16 B of each 32-dim chunk (the
    score MFMA's B operand)."""
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["q"]),
                ((t * d.heads + qh) * HEAD_DIM + ch * 32 + 8 * g) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for ch in range(HEAD_DIM // 32)
    ]


@traced
def _score_keys(c, t, tile, chunk, key0, qh, live, meta, kws, qws):
    """Key slots key0 .. key0 + 16 of token t, head tile ``tile``, on this wave
    from its key and q operands: scores x qk_scale, NEG at a slot without a
    key -> AS (lane (g, head): slots 4 g .. + 4 of its head, two pairs a
    store)."""
    lane, d = c["lane"], c["d"]
    a = c["args"]
    g = lane // 16
    ok = _slot_valid(meta, key0 + lane % 16)
    sc = fx.Vector.filled(4, 0.0, fx.Float32)
    if const_expr(c["kv_fp8"]):
        # 4 scaled FP8 MFMAs (K = 128): a slot without a key zero at unit
        # scale (a zero operand does not hold back a NaN scale)
        steps, codes = kws
        for st in range_constexpr(HEAD_DIM // 128):
            kw = fx.Vector.from_elements(
                [ok.select(steps[st][e], fx.Int32(0)) for e in range(8)], fx.Int32
            )
            kcode = ok.select(codes[st] >> (8 * g) & 0xFF, fx.Int32(UNIT_SCALE))
            qw, qcode = qws[st]
            sc = mfma_scaled(kw, qw, sc, kcode, qcode)
    else:
        for ch in range_constexpr(HEAD_DIM // 32):
            kw = fx.Vector.from_elements(
                [ok.select(kws[ch][e], fx.Int32(0)) for e in range(4)], fx.Int32
            )
            qw = qws[ch]
            sc = mfma_bf16(kw.bitcast(fx.BFloat16), qw.bitcast(fx.BFloat16), sc)
    scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
    sv = [
        fx.Float32(
            _slot_valid(meta, key0 + 4 * g + i).select(sc[i] * scale, fx.Float32(NEG))
        ).bitcast(fx.Int32)
        for i in range(4)
    ]
    if only_live(True, live):
        at = _as(d, t, tile, chunk, qh - tile * HEAD_TILE, key0 % CH + 4 * g)
        c["put_words"](c["as"], Quad(at // 2), sv[:2])
        c["put_words"](c["as"], Quad(at // 2 + 1), sv[2:])


def merge_task(task, tokens, tiles):
    """Merge task ``task`` -> (token, head tile, head group, slice)."""
    t, tile, rest = tile_task(task, merge_groups(tokens) * SLICES, tiles)
    return t, tile, rest // SLICES, rest % SLICES


@traced
def stage_attn_merge(c, t, tile, hg, sl, mh):
    """Token t, heads ``mh`` hg .. of ``tile``, dims 64 sl .. -> ``att["out"]``
    [head][64] (f32 of bf16), barriers included: every key's values landed
    while the scores come, then one poll of them."""
    d, a = c["d"], c["args"]
    await_selection(c, t)
    sels = _sel_words(c, t)
    meta = _keys(c, t)
    h = c["tid"] // 32 % mh
    sink = fx.Float32(
        bo.buffer_load(
            rsrc(a["sink"]), tile_head(d, tile, hg * mh + h)[0], vec_width=1,
            dtype=T.f32,
        )
    )  # fmt: skip
    kvt_rows = c.get("kvt_rows", False)
    if const_expr(c["fused"] and not kvt_rows):
        # the task loads its last key, always one of this launch's window rows
        # (``_new_rows`` >= 1): the poll unconditional, in flight with the
        # metadata's loads instead of after them (``await_step_rows``). By
        # KVT the rows' words are polled themselves (``_merge_step_values``)
        c["k1_poll"]([(c["krdy"], c["lane"] % c["S"], 1)])
    _merge_values_land(c, _merge_values(c, t, meta, sl, sels), meta, sl)
    _merge_softmax(c, t, tile, hg, mh, meta, sink * fx.Float32(LOG2E))
    gpu.barrier()
    _merge_out(c, _merge_pv(c, meta, mh), mh)


def _sel_words(c, t):
    """``_merge_values``' selection words (``_sel_word``), a selection pass's
    slot a thread: loaded before the token's metadata decides which slots hold
    keys (a slot past them reads row 0, its value masked)."""
    a, tid = c["args"], c["tid"]
    fresh = bool(c.get("cp_ratio"))
    return [
        _sel_word(a, fresh, t * a["topk"] + fx.min(i * 64 + tid // 8, a["topk"] - 1))
        for i in range(aplan.window_chunk(c["sel_slots"]))
    ]


def _merge_values(c, t, meta, sl, sels):
    """Slice ``sl`` of the token's key slots, thread (slot tid // 8 of each
    pass of 64, a chunk, 8-column block tid % 8): its 16 B of a BF16 row (an
    FP8 row's 8 B and its group's code word), every pass's loads issued at
    once, a pass's kind (selection or window) its position; a slot without a
    key reads a row it is masked at where it lands. ``sels``: ``_sel_words``."""
    a = c["args"]
    tid = c["tid"]
    col = sl * SLICE + tid % 8 * 8
    out = []
    sel_slots = c["sel_slots"]
    for i in range_constexpr(CHUNKS):
        slot = i * 64 + tid // 8
        if const_expr(i < aplan.window_chunk(sel_slots)):
            row = _selected_row(meta, slot, sels[i])
        else:
            row = _window_row(a, t, meta, slot - sel_slots)
        code = _ld_codes(c, row, col // COLS) if c["kv_fp8"] else None
        out.append((_ld_row4(c, a["pool"], row, col), code))
    if const_expr(c.get("kvt_rows", False)):
        out = _merge_step_values(c, meta, col, out)
    return out


def _merge_step_values(c, meta, col, out):
    """``_merge_values``' window slots that hold this launch's rows
    (``_window_row``: -1 - j, ``kvrow`` not written) from KVT's pairs: a
    slot's 8 columns a quad, polled once every pass's loads are out (a slot
    of the pool's out of range, ``Mailbox.poll_need``); the code from its
    tag, at its byte of ``_ld_codes``' word (``_merge_values_land``'s
    shift)."""
    tid, s = c["tid"], c["S"]
    _, _, _, count, _, (n_new, last), sel_slots, _ = meta
    w0 = aplan.window_chunk(sel_slots)
    specs, needs = [], []
    for i in range_constexpr(w0, CHUNKS):
        w = i * 64 + tid // 8 - sel_slots
        needs.append((w >= count - n_new) & (w < count))
        j = fx.min(fx.max(last - (count - 1 - w), 0), s - 1)
        specs.append((c["kvt"], Quad(j * (KVT_ROW_PAIRS // 2) + col // 8), 2))
    got = c["k1_poll_need"](specs, needs, coded=True)
    shift = 8 * (col // COLS % 4)
    for k in range_constexpr(len(specs)):
        (words, code), need = got[k], needs[k]
        loaded, lcode = out[w0 + k]
        out[w0 + k] = (
            fx.Vector.from_elements(
                [need.select(words[e], loaded[e]) for e in range(2)], fx.Int32
            ),
            need.select(code << shift, lcode),
        )
    return out


@traced
def _merge_values_land(c, vals, meta, sl):
    """``_merge_values``' words (an FP8 row's dequantized by its group's code,
    exact in bf16) into LDS V [8-column block][slot][8 columns], zeros at a
    slot without a key."""
    tid = c["tid"]
    blk = tid % 8
    shift = 8 * ((sl * SLICE + blk * 8) // COLS % 4)
    for i in range_constexpr(len(vals)):
        key = i * 64 + tid // 8
        words, code = vals[i]
        words = _tile_words(c, words, None if code is None else code >> shift & 0xFF)
        ok = _slot_valid(meta, key)
        fx.ptr_store(
            fx.Vector.from_elements(
                [ok.select(words[e], fx.Int32(0)) for e in range(4)], fx.Int32
            ),
            c["att"]["vt"] + (blk * V_SLOTS + key) * 4,
        )


@traced
def _merge_softmax(c, t, tile, hg, mh, meta, sink2):
    """Thread (chunk group g, head h of the group, key pair j): the group's
    scores of chunks g, g + G, .. (``attention_plan.polled_chunk``; G =
    512 / (32 mh)) in one poll; each chunk's max into LDS, then m = the max over
    the chunks and the sink's (``sink2``: the sink x log2e); p = exp2(s - m),
    bf16 into LDS P [head][key]; each chunk's sum of p (f32), m and the sink's
    term's exponent per head into LDS."""
    tid, d, att = c["tid"], c["d"], c["att"]
    groups = THREADS // (mh * 32)
    g, h, j = tid // (mh * 32), tid // 32 % mh, tid % 32
    hh = tile_head(d, tile, hg * mh + h)[0] - tile * HEAD_TILE
    n_sel, count = meta[0], meta[3]
    ccs = [g + groups * i for i in range(-(-CHUNKS // groups))]
    got = c["poll"](
        [
            (c["as"], Quad(_as(d, t, tile, aplan.polled_chunk(fx.min(cc, CHUNKS - 1), n_sel, count, c["sel_slots"]), hh, 2 * j) // 2), 2)
            for cc in ccs
        ]
    )  # fmt: skip
    svs = []
    for i in range_constexpr(len(ccs)):
        cc = ccs[i]
        sv = []
        for e in range_constexpr(2):
            # a slot holding a key is in a written chunk: one polled in place
            ok = (cc < CHUNKS) & _slot_valid(meta, cc * CH + 2 * j + e)
            sv.append(
                fx.Float32(ok.select(got[i][e].bitcast(fx.Float32), fx.Float32(NEG)))
            )
        svs.append(sv)
    # every chunk's reduction a level at a time: their
    # cross-lane latencies overlap, each chunk's order unchanged
    mxs = butterflies([fx.max(sv[0], sv[1]) for sv in svs], (1, 2, 4, 8, 16), fx.max)
    for i in range_constexpr(len(ccs)):
        if (j == 0) & (ccs[i] < CHUNKS):
            fx.ptr_store(mxs[i], att["mx"] + (h * CHUNKS + ccs[i]))
    gpu.barrier()
    m = fx.Float32(NEG)
    for cc in range_constexpr(CHUNKS):
        m = fx.max(m, fx.ptr_load(att["mx"] + (h * CHUNKS + cc)))
    m = fx.max(m, sink2)
    ps = [[hw_exp2(svs[i][e] - m) for e in range(2)] for i in range(len(ccs))]
    # each chunk's sum apart, added in chunk order by ``_merge_out``: the same
    # order whatever the head group's size (``merge_heads``, a width's)
    lsums = butterflies([p[0] + p[1] for p in ps], (1, 2, 4, 8, 16))
    for i in range_constexpr(len(ccs)):
        cc = ccs[i]
        pw = (
            fx.Vector.from_elements(ps[i], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        )
        if cc < CHUNKS:
            fx.ptr_store(pw[0], att["p"] + (h * P_ROW + cc * (CH // 2) + j))
            if j == 0:
                fx.ptr_store(lsums[i], att["sum"] + (h * CHUNKS + cc))
    if (j == 0) & (g == 0):
        fx.ptr_store(m, att["stat"] + h)
        fx.ptr_store(sink2, att["stat"] + (HEAD_TILE + h))


@traced
def _merge_pv(c, meta, mh):
    """P V of the slice: wave w the 16 dims 16 (w % 4) .. over key half w // 4,
    a 16-slot block an MFMA (A = V^T by a transposing LDS read, B = P; a head
    past the group zero), PV_GROUP blocks a branch: a group's LDS reads issue
    together, one LDS round trip a group, not a block (20 a wave serially
    before). A group without a key is skipped (the wave's condition: the slots
    holding keys are a prefix of the selection's and of the window's, which a
    block does not straddle, so its first slot says; a group inside one of
    them likewise its first block). A block without a key in a taken group
    adds exact +0 products (V +0 at its slots, ``_merge_values_land``; P +0,
    exp2 of NEG - m): acc unchanged."""
    lane = c["lane"]
    wave, att = c["wave"], c["att"]
    g, n = lane // 16, lane % 16
    db, kh = wave % 4, wave // 4
    # ds_read_b64_tr_b16's source: lane (g, n) points at key 4 g + n // 4 of
    # the block, columns 4 (n % 4) .. + 4 of the dim block's 8-column pair
    tr = ((2 * db + n % 4 // 2) * V_SLOTS + 4 * g + n // 4) * 4 + n % 2 * 2
    nn = fx.min(n, mh - 1)
    blocks = ATTENTION_KEYS_MAX // 16 // 2
    sb = c["sel_slots"] // 16  # the first window block
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for b0 in range_constexpr(0, blocks, PV_GROUP):
        bs = list(range(b0, min(b0 + PV_GROUP, blocks)))
        nxt = acc
        # one condition a group inside the selection or the window (either
        # key half's: kh 0 or 1), its first block's; a group across their
        # boundary takes any block's. (An OR of a group's four held the
        # kernel's SGPRs, at their 104, to 36 B of scratch: refused.)
        has_key = _slot_valid(meta, (kh * blocks + bs[0]) * 16)
        if const_expr(any(h * blocks + b0 < sb <= h * blocks + bs[-1] for h in (0, 1))):
            for b in bs[1:]:
                has_key = has_key | _slot_valid(meta, (kh * blocks + b) * 16)
        if has_key:
            ops = []
            for b in bs:
                kb = kh * blocks + b
                av = rocdl.ds_read_tr16_b64(
                    fx.Vector.make_type(4, fx.BFloat16),
                    fx.to_llvm_ptr(att["vt"] + (tr + kb * 16 * 4)),
                ).result
                pw = fx.Vector(
                    fx.ptr_load(
                        att["p"] + (nn * P_ROW + (kb * 16 + 4 * g) // 2),
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                )
                ops.append((av, pw))
            group_acc = acc
            # the head mask's compare in the group: not held over the loop
            live = n < mh
            for av, pw in ops:
                pw = fx.Vector.from_elements(
                    [live.select(pw[e], fx.Int32(0)) for e in range(2)], fx.Int32
                )
                group_acc = fx.Vector(
                    rocdl.mfma_f32_16x16x16bf16_1k(
                        T.vec(4, T.f32),
                        [
                            fx.Vector(av).bitcast(fx.Int16),
                            pw.bitcast(fx.Int16),
                            group_acc,
                        ],
                    )
                )
            nxt = group_acc
        acc = nxt
    return acc


@traced
def _merge_out(c, acc, mh):
    """The two key halves' P V summed, / the denominator (the chunks' sums in
    chunk order and the sink's term), bf16 -> ``att["out"]`` [head][64]: lane
    (g, n) of wave w holds dims 16 (w % 4) + 4 g .. + 4 of head n."""
    lane, wave, att = c["lane"], c["wave"], c["att"]
    g, n = lane // 16, lane % 16
    db, kh = wave % 4, wave // 4
    tree = att["tree"] + (db * 64 + lane) * 4
    if kh == 1:
        for i in range_constexpr(4):
            fx.ptr_store(fx.Float32(acc[i]), tree + i)
    gpu.barrier()
    if (kh == 0) & (n < mh):
        nn = fx.min(n, mh - 1)
        den = fx.Float32(0.0)
        for cc in range_constexpr(CHUNKS):
            den = den + fx.ptr_load(att["sum"] + (nn * CHUNKS + cc))
        m = fx.ptr_load(att["stat"] + nn)
        sink2 = fx.ptr_load(att["stat"] + (HEAD_TILE + nn))
        den = den + hw_exp2(sink2 - m)
        for i in range_constexpr(4):
            o = (fx.Float32(acc[i]) + fx.ptr_load(tree + i)) / fx.max(
                den, fx.Float32(1e-30)
            )
            o = fx.Float32((den > fx.Float32(0.0)).select(o, fx.Float32(0.0)))
            fx.ptr_store(bf16_round(o), att["out"] + (nn * SLICE + db * 16 + 4 * g + i))
    gpu.barrier()
