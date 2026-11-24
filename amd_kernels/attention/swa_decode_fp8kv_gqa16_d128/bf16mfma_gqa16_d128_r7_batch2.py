# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSpark draft decode attention over fp8 K/V pages (MiniMax-M3.1, gfx950, FlyDSL).

The draft's sliding-window attention for its query block: every request has ``ql``
(<= 8) query rows at the end of its sequence, and its 16 query heads per KV head see
the keys of the window that ends at the sequence (non-causal inside the block, or
causal). K / V are e4m3 (one per-tensor scale each), stored per token as 128 K bytes
then 128 V bytes. K and V are widened to bf16 (exact), bf16 Q, fp32 scores,
online softmax in fp32, P rounded to bf16, fp32 accumulation; the K scale folds
into the score scale and the V scale into the output. The Triton unified
attention instead rounds ``(k * k_scale)`` to bf16 before the dot, so the two
differ by one bf16 rounding of K when the scale is not a power of two.

A workgroup is one (request, KV head, split), 4 waves; the request's window is cut
in 32-key chunks, and chunk c goes to wave (c mod 4 S) of the request's 4 S waves.
A wave step is one chunk for every query row of the block:

  S^T = K Q^T      v_mfma_f32_16x16x32_bf16, A = K (16 keys x 32 dims), B = Q^T
  online softmax   fp32, keys 8 g .. 8 g + 7 of the chunk in lane group g
  O^T += V^T P^T   A = V^T (16 dims x 32 keys), B = P^T straight from S^T

Lane (n, g) of an S^T tile st holds key 8 (n // 4) + 4 st + n % 4's dims 64 h + 16 g
.. + 16 (one 16-byte load per h), so its S^T outputs are keys 8 g .. 8 g + 7 of the
chunk, P^T's operand. Its V^T operand is dims 8 n .. 8 n + 7 of keys 8 g .. 8 g + 7
(one 8-byte load per key), byte-transposed; its O^T outputs are dims 32 g .. 32 g +
31 of head n. Q^T is staged once in LDS in operand order. The next chunk's loads are
in flight while one is computed.

O accumulates inside the MFMAs; it is rescaled only when a row's running max grows
(the step is skipped when no lane's max moved, where the rescale factor is exactly
1). Only the window's edge chunks are masked.

Past 6 rows (``ql`` 7, 8) the rows' O alone fills the AGPRs, so a chunk is attended
``ROW_GROUP`` rows at a time (scores, mask, softmax and PV of a group before the
next group's scores): only a group's S / P are live. Per row the operations and
their order are the same; K is widened once per chunk, V once.

The waves' (max, sum, O) are combined through LDS, row t assigned to wave t % 4. With
``splits`` > 1 each workgroup leaves its unnormalized O, max and sum per row in a
scratch row, and the last split of a request to arrive (a ticket) combines them in
split order, so the result does not depend on arrival order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import CH, MAX_QL, NW

D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
TOK_B = 2 * D  # bytes per (token, KV head): K then V
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 16 + 16  # dwords a lane loads per chunk: K (2 tiles x 2 x 16 B), V (8 x 8 B)
PART_F = 36  # split partials: fp32 per (split, row, lane): O (NDT x 4), max, sum, pad
# The split partials must reach the combining workgroup, and the XCDs' L2s are not
# coherent. With ``xcds`` (workgroup w runs on XCD w % xcds) the splits of a request
# run on one XCD and plain stores / loads meet in its L2; otherwise they are stored
# and loaded at agent scope (sc1).
AGENT_SCOPE = 0x10
# kernel arguments preloaded into SGPRs (the first 64 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
LOG2E = 1.4426950409
TINY_F32 = 1.1754943508222875e-38  # torch.finfo(float32).tiny, the quant's amax floor
# rows per attend pass (2: graph-timed ~4% faster than 1 at q8, splits 5/6)
ROW_GROUP = 2


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads)."""
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane_swap(op, d_a, d_b):
    """v_permlane{16,32}_swap (raw rocdl op; fx has no wrapper)."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def _maxf_nn(a, b):
    """v_max_f32 with nnan (no canonicalizing v_max x, x, x on the inputs)."""
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=_nnan()).result
    )


def _xlane4(x, op):
    """``op`` over lanes L, L^16, L^32, L^48, broadcast to all four (two swaps)."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _pk_bf16(a, b):
    """Two fp32 -> one dword of two bf16 (RNE, v_cvt_pk_bf16_f32), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _pk_fp8(a, b, old, hi):
    """Two fp32 -> two e4m3 bytes (RNE, v_cvt_pk_fp8_f32) in the low (``hi``: high)
    word of ``old``, ``a`` the lower byte (nvfp4_sparse_decode/kernel.py)."""
    return fx.Int32(
        _rocdl_ops.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _mxfp8_group(words):
    """A lane's 32 output values as 16 packed-bf16 dwords (dims in order) -> their
    MXFP8 quant as ``third_party/aiter/mxfp8_gemm._mxfp8_quantize_kernel`` computes
    it on the bf16 values: 8 dwords of e4m3 codes and the e8m0 scale (fp32,
    0..254). nvfp4_sparse_decode/kernel.py's ``_mxfp8_group`` with a smaller live
    set (fewer AGPR copies / MFMA-result hazards): the 32 fp32 values are never
    live together --
      * amax: the largest magnitude as an integer max (a tree) over the bf16 halves'
        magnitude bits (``w & 0x7FFF``: for non-negative floats, including
        subnormals and inf, the integer order is the float order), widened to
        fp32 and floored at TINY by the same ``maximumf`` -- the value the fp32
        ``maximumf(|x|)`` chain gives for every non-NaN group (a NaN group stays
        NaN: its magnitude bits exceed inf's);
      * codes: each dword's two values are widened right where they are scaled
        (the same ``x * rs`` products, the same conversions, in the same order)."""
    # integer max as a tree (depth 5 instead of a 32-step chain; max is exact)
    mags = []
    for wd in words:
        lo = wd & fx.Int32(0x7FFF)
        hi = (wd >> 16) & fx.Int32(0x7FFF)
        mags.append((lo > hi).select(lo, hi))
    while len(mags) > 1:
        mags = [
            (mags[i] > mags[i + 1]).select(mags[i], mags[i + 1])
            for i in range(0, len(mags), 2)
        ]
    mag = mags[0]
    amax = fx.arith.maximumf(fx.Float32(TINY_F32), _as_f32(mag << 16))
    sb = fx.math.ceil(fx.math.log2(amax / fx.Float32(448.0))) + fx.Float32(127.0)
    sb = fx.arith.minimumf(fx.arith.maximumf(sb, fx.Float32(0.0)), fx.Float32(254.0))
    rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sb))
    dws = []
    for b in range_constexpr(8):
        w0, w1 = words[2 * b], words[2 * b + 1]
        lo = _pk_fp8(
            _as_f32(w0 << 16) * rs,
            _as_f32(w0 & fx.Int32(-65536)) * rs,
            fx.Int32(0),
            False,
        )
        dws.append(
            _pk_fp8(
                _as_f32(w1 << 16) * rs, _as_f32(w1 & fx.Int32(-65536)) * rs, lo, True
            )
        )
    return dws, sb


def _mfma(a4, b4, acc):
    """v_mfma_f32_16x16x32_bf16: lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i]
    (4 dwords each), gets D[4q + e][i]."""
    a = Vec(a4).bitcast(fx.BFloat16)
    b = Vec(b4).bitcast(fx.BFloat16)
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def _exp2(x):
    """v_exp_f32 (no denormal range reduction: the P it feeds is rounded to bf16)."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _perm(hi, lo, sel):
    """v_perm_b32: byte i of the result is byte sel_i of (hi:lo), 0-3 = lo."""
    return fx.Int32(_rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(sel)))


def _bf16x2_e4m3(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` -> one dword of two bf16 (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
            Vec.make_type(2, fx.BFloat16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    ).bitcast(fx.Int32)[0]


def _bf16x8_e4m3(lo, hi):
    """Eight e4m3 bytes (``lo`` then ``hi``) -> four dwords of bf16 pairs, in order."""
    return Vec.from_elements(
        [
            _bf16x2_e4m3(lo, False),
            _bf16x2_e4m3(lo, True),
            _bf16x2_e4m3(hi, False),
            _bf16x2_e4m3(hi, True),
        ],
        fx.Int32,
    )


def _transpose4(a, b, c, d):
    """4 x 4 byte transpose: result k = (a.k, b.k, c.k, d.k)."""
    x = _perm(b, a, 0x06020400)  # a0 b0 a2 b2
    y = _perm(b, a, 0x07030501)  # a1 b1 a3 b3
    u = _perm(d, c, 0x06020400)  # c0 d0 c2 d2
    v = _perm(d, c, 0x07030501)  # c1 d1 c3 d3
    return [
        _perm(u, x, 0x05040100),
        _perm(v, y, 0x05040100),
        _perm(u, x, 0x07060302),
        _perm(v, y, 0x07060302),
    ]


def _global_atomic_add(addr_i64, value):
    """``old = *addr; *addr += value`` on a global i32 (agent scope, relaxed)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_batch2(
    ql: int,
    splits: int,
    window: int,
    causal: bool,
    block_size: int,
    xcds: int,
    waves: int = NW,
    quant: bool = False,
):
    """``ql``: query rows per request (1..8). ``splits``: workgroups per (request, KV
    head). ``window``: the sliding window (keys p - window < k for the row at p).
    ``block_size``: tokens per page (a multiple of 32). ``xcds`` > 0: workgroup w
    runs on XCD w % xcds, and a request's splits are placed on one XCD.
    ``quant``: instead of the bf16 output, write its MXFP8 quant
    as the o_proj's input, as nvfp4_sparse_decode's ``quant`` does: e4m3 codes
    row-major or, with ``i32_tile`` > 0, in the decode GEMM's (16, 16) tile layout,
    and e8m0 scales in AITER's shuffled layout (the bytes of
    ``third_party/aiter/mxfp8_gemm``'s quantizers on the bf16 output). The
    attention itself is unchanged; the quant / tile arguments are the kernel's
    last ones (unused, 0, without ``quant``)."""
    NW = waves
    assert 1 <= ql <= MAX_QL and NW >= 1 and splits >= 1 and block_size % CH == 0
    assert 0 < window <= 64 * CH * NW - max(ql, 6) - CH + 2
    QL, S, W, BS, XCDS = ql, splits, window, block_size, xcds
    assert QL == 7 and S > 1 and NW == 4
    RG = ROW_GROUP
    NG = (QL + RG - 1) // RG  # attend passes per chunk
    NWT = S * NW  # waves per request
    PART_SCOPE = 0 if XCDS else AGENT_SCOPE
    name = (
        f"swa_fp8kv_bf16mfma_gqa16_d128_r{QL}_s{S}_w{W}_p{BS}_n{NW}"
        + ("_c" if causal else "")
        + (f"_x{XCDS}" if XCDS else "")
        + f"_rg{RG}"
        + ("_mxfp8" if quant else "")
    )
    name += "_batch2"
    QUANT = quant
    # the wave partials are merged BR rows at a time (Q^T lives in the O buffer: it is
    # dead after the key loop, and the LDS stays under 160 KB)
    BR = 2
    OB = BR
    QSZ = QL * NKS * 64 * 4  # Q^T operand words

    @fx.struct
    class Shared:
        # OB rows' O per wave as 16 B tiles per (d-tile, lane); (max, sum) per lane
        o: fx.Array[fx.Int32, OB * NW * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * QL * 2 * 64]
        # (the Q^T operands, 16 B per (row, k-step, lane), live in ``o``)
        q: fx.Array[fx.Int32, 4, 16]
        ticket: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_reqs: fx.Int32,
        i32_bt_s: fx.Int32,  # block-table row stride (elements)
        i32_page_b: fx.Int32,  # cache page stride (bytes)
        i32_head_b: fx.Int32,  # cache KV-head stride (bytes)
        i32_H: fx.Int32,  # query heads
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_qsl: fx.Int64,  # [reqs + 1] int32
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, Hkv, BS, 256] e4m3
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_ks: fx.Int64,  # fp32 K scale
        arg_vs: fx.Int64,  # fp32 V scale
        arg_part: fx.Int64,  # splits: [grid requests * Hkv, S, QL, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # splits: [grid requests * Hkv] int32 tickets, zero
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
        arg_qc: fx.Int64,  # quant: [>= ceil(rows / 16) * 16, H * 128] e4m3 codes
        arg_qs: fx.Int64,  # quant: [>= ceil(rows / 32) * 32, H * 4] e8m0 scales
        i32_tile: fx.Int32,  # quant: 0 row-major codes, > 0 the (16, 16) tile layout
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        if const_expr(XCDS > 0):
            # grid.x is padded to XCDS x S: a request's splits share an XCD
            xcd = bx % XCDS
            slot = bx // XCDS
            req = (slot // S) * XCDS + xcd
            split = slot % S
        else:
            req = bx // S
            split = bx % S
        req_ok = req < i32_reqs
        reqc = req_ok.select(req, fx.Int32(0))
        seq_p = _global_i32_ptr(arg_seq)
        qsl_p = _global_i32_ptr(arg_qsl)
        bt = _global_i32_ptr(arg_bt)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))

        seq_len = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(seq_p[reqc])))
        q0 = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(qsl_p[reqc])))
        q1 = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(qsl_p[reqc + 1])))
        qlen = req_ok.select(q1 - q0, fx.Int32(0))
        qlen = (qlen > QL).select(fx.Int32(QL), qlen)
        qlen = (seq_len < qlen).select(fx.Int32(0), qlen)  # graph padding: no keys
        ctx = seq_len - qlen
        # the block's key range: row 0's window start .. the sequence end
        k_lo = ctx - (W - 1)
        k_lo = (k_lo < 0).select(fx.Int32(0), k_lo)
        c_first = k_lo // CH
        n_ch = (qlen > 0).select((seq_len + (CH - 1)) // CH - c_first, fx.Int32(0))
        gw = split * NW + w  # this wave's index among the request's waves
        n_my = (n_ch > gw).select((n_ch - gw + (NWT - 1)) // NWT, fx.Int32(0))
        n_my = fx.Int32(fx.rocdl.readfirstlane(T.i32, n_my))

        # lane l = this wave's chunk l: its page (logical block of key 32 c)
        my_c = c_first + gw + lane * NWT
        lane_ok = lane < n_my
        blk = (my_c * CH) // BS
        pg_lane = fx.Int32(bt[reqc * i32_bt_s + lane_ok.select(blk, fx.Int32(0))])
        pg_lane = lane_ok.select(pg_lane, fx.Int32(0))

        head_off = kvh * i32_head_b
        k_rows = [8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]

        def load_chunk(it):
            """This lane's dwords of its wave's chunk ``it`` (past the wave's chunks:
            its first one again): K tile st, half h (4 + 4 + 4 + 4), V keys 8 g + j
            (2 each). Slots at or past the sequence end read as zero (an unwritten
            slot may hold an e4m3 NaN)."""
            itc = (it < n_my).select(it, fx.Int32(0))
            pg = fx.Int32(fx.rocdl.readlane(T.i32, pg_lane, itc))
            cc = c_first + gw + itc * NWT
            tok0 = (cc * CH) % BS
            valid = seq_len - (cc * CH - tok0)
            valid = (valid > BS).select(fx.Int32(BS), valid)
            valid = (valid < 0).select(fx.Int32(0), valid)
            # (LLVM clamps with v_med3: keep the descriptor scalar, no waterfall)
            valid = fx.Int32(fx.rocdl.readfirstlane(T.i32, valid))
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(pg) * fx.Int64(i32_page_b) + fx.Int64(head_off),
                num_records_bytes=valid * TOK_B,
            )
            raw = []
            for st in range_constexpr(2):
                for h in range_constexpr(2):
                    kd = fx.Vector(
                        buffer_ops.buffer_load(
                            pr,
                            ((tok0 + k_rows[st]) * TOK_B + 64 * h + 16 * g) // 4,
                            vec_width=4,
                            dtype=fx.Int32,
                        )
                    )
                    raw += [kd[i] for i in range_constexpr(4)]
            for j in range_constexpr(8):
                vd = fx.Vector(
                    buffer_ops.buffer_load(
                        pr,
                        ((tok0 + 8 * g + j) * TOK_B + D + 8 * n) // 4,
                        vec_width=2,
                        dtype=fx.Int32,
                    )
                )
                raw += [vd[0], vd[1]]
            return raw

        # Q^T into LDS in operand order: row t, k-step ks = 2 h + s, lane (n, g) holds
        # dims 64 h + 16 g + 8 s .. + 8 of head n
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        q_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(QSZ, 1),
            ),
            fx.make_layout(4, 1),
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        NQI = (QL * HG * 16 + 64 * NW - 1) // (64 * NW)
        q_items = []
        for k in range_constexpr(NQI):
            item = tx + k * (64 * NW)  # (row, head, 16-byte piece j of the head)
            t_i = item // (HG * 16)
            h_i = (item // 16) % HG
            j = item % 16
            row = q0 + (t_i < qlen).select(t_i, fx.Int32(0))
            qv = fx.Vector(
                buffer_ops.buffer_load(
                    qr,
                    ((row * i32_H + kvh * HG + h_i) * D + 8 * j) // 2,
                    vec_width=4,
                    dtype=fx.Int32,
                )
            )
            q_items.append((item, t_i, h_i, j, qv))
        fx.rocdl.sched_barrier(0)  # the Q loads go out before the page lookup wait
        # the first chunk's K / V loads fly while Q^T is stored and the waves meet
        raw0 = load_chunk(fx.Int32(0))
        for k in range_constexpr(NQI):
            item, t_i, h_i, j, qv = q_items[k]
            ks_i = 2 * (j // 8) + j % 2
            g_i = (j % 8) // 2
            if item < QL * HG * 16:
                r_q = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r_q.store(qv)
                fx.copy(
                    lds_atom,
                    r_q,
                    fx.slice(q_lds, (None, (t_i * NKS + ks_i) * 64 + g_i * 16 + h_i)),
                )

        # per-row key bounds: row t at position ctx + t sees [ctx + t - W + 1, hi)
        klo = [ctx + (t - (W - 1)) for t in range_constexpr(QL)]
        khi = [(ctx + (t + 1)) if causal else seq_len for t in range_constexpr(QL)]
        # a chunk with keys in [all-rows-lo, all-rows-hi) needs no mask
        last_t = (qlen > 0).select(qlen - 1, fx.Int32(0))
        full_lo = ctx + last_t - (W - 1)
        full_hi = (ctx + 1) if causal else seq_len

        def widen_v(raw):
            """V^T frags vf[dt]: dims 8 n + dt of keys 8 g .. 8 g + 7 (bf16 pairs)."""
            vlo = [raw[16 + 2 * j] for j in range_constexpr(8)]
            vhi = [raw[16 + 2 * j + 1] for j in range_constexpr(8)]
            t_lo_a = _transpose4(*vlo[0:4])
            t_lo_b = _transpose4(*vlo[4:8])
            t_hi_a = _transpose4(*vhi[0:4])
            t_hi_b = _transpose4(*vhi[4:8])
            vf = []
            for dt in range_constexpr(NDT):
                a = t_lo_a[dt] if dt < 4 else t_hi_a[dt - 4]
                b = t_lo_b[dt] if dt < 4 else t_hi_b[dt - 4]
                vf.append(_bf16x8_e4m3(a, b))
            return vf

        zero4 = Vec.filled(4, 0.0, fx.Float32)

        def q_frag(t, ks):
            r_f = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(lds_atom, fx.slice(q_lds, (None, (t * NKS + ks) * 64 + lane)), r_f)
            return Vec(r_f.load())

        def attend_grouped(raw, k0, m_t, l_t, o_t):
            """One chunk (keys k0 ..) into every row's running (max, sum, O), RG rows at
            a time: a group's S^T, mask, online softmax and O^T update finish before
            the next group's S^T; the max-grew skip is per group
            (a skipped row's factor is exactly 1, or 0 on its zero O and sum)."""
            kf_all = [
                [
                    _bf16x8_e4m3(
                        raw[st * 8 + (ks // 2) * 4 + 2 * (ks % 2)],
                        raw[st * 8 + (ks // 2) * 4 + 2 * (ks % 2) + 1],
                    )
                    for st in range_constexpr(2)
                ]
                for ks in range_constexpr(NKS)
            ]
            edge = (k0 < full_lo) | (k0 + CH > full_hi)
            key = k0 + 8 * g
            m_o = list(m_t)
            l_o = list(l_t)
            o_o = list(o_t)
            vf = []
            for gi in range_constexpr(NG):
                t0 = gi * RG
                nr = min(RG, QL - t0)
                s_g = [[zero4, zero4] for _ in range_constexpr(nr)]
                for ks in range_constexpr(NKS):
                    for j in range_constexpr(nr):
                        qv = q_frag(t0 + j, ks)
                        s_g[j] = [
                            _mfma(kf_all[ks][st], qv, s_g[j][st])
                            for st in range_constexpr(2)
                        ]
                a_g = [
                    [s_g[j][i // 4][i % 4] for i in range_constexpr(8)]
                    for j in range_constexpr(nr)
                ]
                if edge:
                    a_g = [
                        [
                            ((key + i >= klo[t0 + j]) & (key + i < khi[t0 + j])).select(
                                a_g[j][i], neg_inf
                            )
                            for i in range_constexpr(8)
                        ]
                        for j in range_constexpr(nr)
                    ]
                m_g = []
                for j in range_constexpr(nr):
                    mx = a_g[j][0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf_nn(mx, a_g[j][i])
                    m_g.append(_maxf_nn(m_t[t0 + j], _xlane4(mx, _maxf_nn) * c))
                grew = m_g[0] > m_t[t0]
                for j in range_constexpr(1, nr):
                    grew = grew | (m_g[j] > m_t[t0 + j])
                mu_g = [
                    (m_g[j] == neg_inf).select(fx.Float32(0.0), m_g[j])
                    for j in range_constexpr(nr)
                ]
                # the branch below yields o_g / l_g only (no names new to it)
                al_g = [_exp2(m_t[t0 + j] - mu_g[j]) for j in range_constexpr(nr)]
                o_g = [o_t[t0 + j] for j in range_constexpr(nr)]
                l_g = [l_t[t0 + j] for j in range_constexpr(nr)]
                if fx.Int64(fx.rocdl.ballot(T.i64, grew)) != fx.Int64(0):
                    o_g = [
                        [
                            o_g[j][dt] * Vec.filled(4, al_g[j], fx.Float32)
                            for dt in range_constexpr(NDT)
                        ]
                        for j in range_constexpr(nr)
                    ]
                    l_g = [l_g[j] * al_g[j] for j in range_constexpr(nr)]
                if const_expr(gi == 0):
                    vf = widen_v(raw)
                for j in range_constexpr(nr):
                    p = [_exp2(fx.math.fma(x, c, -mu_g[j])) for x in a_g[j]]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    pf = Vec.from_elements(
                        [_pk_bf16(p[2 * i], p[2 * i + 1]) for i in range_constexpr(4)],
                        fx.Int32,
                    )
                    m_o[t0 + j] = m_g[j]
                    l_o[t0 + j] = l_g[j] + ps
                    o_o[t0 + j] = [
                        _mfma(vf[dt], pf, o_g[j][dt]) for dt in range_constexpr(NDT)
                    ]
            return m_o, l_o, o_o

        fx.gpu.barrier()  # Q^T is in LDS

        # Double buffer: chunk i + 1's loads are issued before chunk i is computed
        # (the loop-carried copy then waits only for what the compute did not hide).
        # Past the wave's count a chunk lies past the sequence end: fully masked.
        init = [neg_inf for _ in range_constexpr(QL)]
        init += [fx.Float32(0.0) for _ in range_constexpr(QL)]
        init += [zero4 for _ in range_constexpr(QL * NDT)]
        init += raw0
        n_st = 2 * QL + QL * NDT
        results = init
        for it, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_my),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            m_t = [fx.Float32(st_[t]) for t in range_constexpr(QL)]
            l_t = [fx.Float32(st_[QL + t]) for t in range_constexpr(QL)]
            o_t = [
                [Vec(st_[2 * QL + t * NDT + dt]) for dt in range_constexpr(NDT)]
                for t in range_constexpr(QL)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            it32 = fx.Int32(it)
            raw_n = load_chunk(it32 + 1)
            fx.rocdl.sched_barrier(0)
            k0 = (c_first + gw + it32 * NWT) * CH
            m_t, l_t, o_t = attend_grouped(raw, k0, m_t, l_t, o_t)
            state = m_t + l_t + [o for row in o_t for o in row] + raw_n
            results = yield [fx.as_ir_value(x) for x in state]

        # Combine rows through one LDS buffer; each wave owns rows t % NW == w.
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(OB * NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        ml = smem.ml.ptr
        for t in range_constexpr(QL):
            ml[((w * QL + t) * 2) * 64 + lane] = fx.Float32(results[t])
            ml[((w * QL + t) * 2 + 1) * 64 + lane] = fx.Float32(results[QL + t])

        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )

        def store_out(t, acc, inv):
            """Row t's O (``acc`` x ``inv``) in bf16: lane (n, g) holds dims 32 g + 8 e
            + dt of head n."""
            o_row = ((q0 + t) * i32_H + kvh * HG + n) * (D // 2) + g * 16
            for e in range_constexpr(4):
                val = Vec.from_elements(
                    [
                        _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=t < qlen)

        def store_q(t, acc, inv):
            """Row t's o_proj input (nvfp4_sparse_decode store_q): the MXFP8 codes
            of O (``acc`` x ``inv`` rounded to bf16, the bf16 output's values;
            row-major, or the (16, 16) tile layout when ``i32_tile`` > 0) and the
            e8m0 scale of the lane's 32 dims 32 g .. 32 g + 31 of head n."""
            row = q0 + t
            words = [
                _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                for e in range_constexpr(4)
                for j in range_constexpr(4)
            ]
            dws, sb = _mxfp8_group(words)
            k_row = i32_H * D  # row width (elements)
            head = kvh * HG + n
            k0 = head * D + g * 32  # the lane's first dim
            rows16 = (i32_rows + 15) // 16 * 16
            qcr = buffer_ops.create_buffer_resource_from_addr(
                arg_qc, num_records_bytes=fx.Int64(rows16) * fx.Int64(k_row)
            )
            shuffled = i32_tile > 0
            # (16, 16) tiles: byte (m, k) at (m//16)*16K + (k//64)*1024
            # + ((k%64)//16)*256 + (m%16)*16 + k%16; the lane's 32 bytes are two
            # 16 B runs 256 B apart
            off_t = (
                (row // 16) * (16 * k_row)
                + (k0 // 64) * 1024
                + ((k0 % 64) // 16) * 256
                + (row % 16) * 16
            ) // 4
            off_l = (row * k_row + k0) // 4
            off = shuffled.select(off_t, off_l)
            step = shuffled.select(fx.Int32(64), fx.Int32(4))
            for h in range_constexpr(2):
                val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                buffer_ops.buffer_store(val, qcr, off + h * step, mask=t < qlen)
            # scale (m, g): [M//32, K//256, klane 4, mlane 16, kpack 2, mpack 2]
            gi = head * 4 + g
            s_off = (
                (
                    (((row // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4) * 16
                    + row % 16
                )
                * 2
                + (gi % 8) // 4
            ) * 2 + (row % 32) // 16
            rows32 = (i32_rows + 31) // 32 * 32
            qsr = buffer_ops.create_buffer_resource_from_addr(
                arg_qs, num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32)
            )
            buffer_ops.buffer_store(
                sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=t < qlen
            )

        def store_row(t, acc, inv):
            if const_expr(QUANT):
                store_q(t, acc, inv)
            else:
                store_out(t, acc, inv)

        tid = req * i32_hkv + kvh
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_part, num_records_bytes=i64_part_bytes
        )

        def put_o(t, slot):
            """This wave's partial O of row t (static) into LDS slot ``slot``."""
            for dt in range_constexpr(NDT):
                r_w = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r_w.store(Vec(results[2 * QL + t * NDT + dt]).bitcast(fx.Int32))
                fx.copy(
                    lds_atom,
                    r_w,
                    fx.slice(o_lds, (None, ((slot * NW + w) * NDT + dt) * 64 + lane)),
                )

        def merge_row(t, slot):
            """Row t's waves merged from LDS slot ``slot``: (acc, max, sum)."""
            ms = [
                fx.Float32(ml[((ww * QL + t) * 2) * 64 + lane])
                for ww in range_constexpr(NW)
            ]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = _maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(_exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_w = fx.Float32(ml[((ww * QL + t) * 2 + 1) * 64 + lane])
                l_tot = l_tot + l_w * wt[ww]
            l_tot = _xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r_c = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(
                            o_lds, (None, ((slot * NW + ww) * NDT + dt) * 64 + lane)
                        ),
                        r_c,
                    )
                    acc[dt] = acc[dt] + Vec(r_c.load()).bitcast(fx.Float32) * wv
            return acc, big_raw, l_tot

        # batches of BR rows: wave w merges row b BR + w and stores its split partial
        # while the other wave does its row
        for b in range_constexpr((QL + BR - 1) // BR):
            nr = min(BR, QL - b * BR)
            # every wave's key loop (Q^T reads) / the previous batch's reads are done
            fx.gpu.barrier()
            for r in range_constexpr(nr):
                put_o(b * BR + r, r)
            fx.gpu.barrier()
            if w < nr:
                t_b = w + b * BR
                acc, big_raw, l_tot = merge_row(t_b, w)
                row_f = ((tid * S + split) * QL + t_b) * 64 + lane
                for dt in range_constexpr(NDT):
                    buffer_ops.buffer_store(acc[dt].bitcast(fx.Int32), pr,
                        row_f * PART_F + dt * 4, cache_modifier=PART_SCOPE)
                buffer_ops.buffer_store(
                    Vec.from_elements([_bits(big_raw), _bits(l_tot)], fx.Int32),
                    pr, row_f * PART_F + NDT * 4, cache_modifier=PART_SCOPE,
                )
        # every row's partial is stored before the ticket is taken
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx == 0:
            smem.ticket.ptr[0] = _global_atomic_add(arg_cnt + fx.Int64(tid) * 4, 1)
        fx.gpu.barrier()
        last = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0])))
        if last == S - 1:
            for batch in range_constexpr((QL + NW - 1) // NW):
                t_f = w + batch * NW
                if t_f < QL:
                    rows_f = [
                        ((tid * S + sp) * QL + t_f) * 64 + lane
                        for sp in range_constexpr(S)
                    ]
                    mls = [
                        Vec(
                            buffer_ops.buffer_load(
                                pr,
                                rf * PART_F + NDT * 4,
                                vec_width=2,
                                dtype=fx.Int32,
                                cache_modifier=PART_SCOPE,
                            )
                        )
                        for rf in rows_f
                    ]
                    bigs = [_as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, S):
                        top = _maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(_exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(S):
                        l_all = l_all + _as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(
                        v_scale / l_all, fx.Float32(0.0)
                    )
                    acc_all = [zero4 for _ in range_constexpr(NDT)]
                    for sp in range_constexpr(S):
                        wv = Vec.filled(4, ws[sp], fx.Float32)
                        for dt in range_constexpr(NDT):
                            part = Vec(
                                buffer_ops.buffer_load(
                                    pr,
                                    rows_f[sp] * PART_F + dt * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            ).bitcast(fx.Float32)
                            acc_all[dt] = acc_all[dt] + part * wv
                    store_row(t_f, acc_all, inv_all)
        if (last == S - 1) & (tx == 0):
            _global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)

    @flyc.jit
    def launch_swa_bf16_r7_batch2(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_qsl: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_reqs: fx.Int32,
        i32_rows: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        i32_head_b: fx.Int32,
        f32_c: fx.Float32,
        arg_qc: fx.Int64,
        arg_qs: fx.Int64,
        i32_tile: fx.Int32,
        i32_grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_reqs,
            i32_bt_s,
            i32_page_b,
            i32_head_b,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_qsl,
            arg_bt,
            arg_q,
            arg_cache,
            arg_out,
            arg_ks,
            arg_vs,
            arg_part,
            arg_cnt,
            i64_part_bytes,
            i32_rows,
            f32_c,
            arg_qc,
            arg_qs,
            i32_tile,
        ).launch(
            grid=(i32_grid_x, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_swa_bf16_r7_batch2.kernel_name = name
    # One wave per SIMD: the grid is about one workgroup per CU, and the rows' O
    # (32 registers each) spills to scratch under a 256-register budget from 4 rows.
    launch_swa_bf16_r7_batch2.compile_hints = {**PRELOAD_HINTS, "waves_per_eu": 1}
    return launch_swa_bf16_r7_batch2
