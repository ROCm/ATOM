# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse decode attention on bf16 (or exact fp16) MFMAs: one workgroup per (row, KV head, split).

K / V are dequantized in registers: bf16 e4m3_rn_satfinite(e2m1 x block scale), or the exact e2m1 x scale in fp16;
Q and P in the MFMA dtype; fp32 scores, online softmax and accumulation (contract: ``reference.py``). A workgroup
holds one row's 16 query heads of one KV head (the MFMA N dimension); wave w owns keys [32 w, 32 w + 32) of every
selected block and loads its NVFP4 words straight into the MFMA operands, two blocks per loop iteration:

  S^T = K Q^T      v_mfma_f32_16x16x32_bf16, A = K (16 keys x 32 dims), B = Q^T
  O^T += V^T P^T   A = V^T (16 dims x 32 keys), B = P^T straight from S^T

The waves' (max, sum, O) combine through LDS. With ``splits`` > 1 each split stores its unnormalized partial and the
row's last split to arrive (a ticket) combines them in split order, so the result does not depend on arrival order.
The dequant is about 260 of the ~480 instructions a wave issues per block: VALU-bound, a second wave per SIMD does not
help.
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

BLK = 128  # tokens per block (one page)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every block
KPW = BLK // NW
DATA_B = BLK * D // 2  # packed e2m1 bytes per head per side
SCALE_B = BLK * D // 16  # e4m3 block-scale bytes per head per side
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per block
PART_F = 36  # split partials: fp32 per (split, row, lane): O (NDT x 4), max, sum, pad
# The XCDs' L2s are not coherent: with ``xcds`` (workgroup w on XCD w % xcds) a row's splits run on one XCD and plain
# stores / loads of the partials meet in its L2; otherwise they go at agent scope (sc1). Never an agent-scope release
# fence: it writes back the whole L2.
AGENT_SCOPE = 0x10
# kernel arguments preloaded into SGPRs (the first 56 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
LOG2E = 1.4426950409
SIGN4 = 0x80808080 - (1 << 32)  # the four bytes' sign bits, as an i32
MAG4 = 0x07070707  # the four low-nibble magnitudes


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
    word of ``old``, ``a`` the lower byte."""
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
    """32 finite bf16 values (16 dwords) -> eight dwords of e4m3 codes and the e8m0 scale byte (as fp32).

    The scale byte is the largest magnitude's exponent - 8, plus 1 when its mantissa > 0x60, clamped at 0; byte 0
    scales by 2^-127. Where amax / 448 is subnormal it keeps byte 1 and the reference 0 (strict xfail, ``test_quant``).
    """
    m = Vec.filled(2, 0, fx.Int16)
    for d in words:
        a = Vec.from_elements([d & 0x7FFF7FFF], fx.Int32).bitcast(fx.Int16)
        m = Vec(fx.arith.maxui(m, a))
    lo, hi = m.bitcast(fx.Int32)[0] & 65535, m.bitcast(fx.Int32)[0] >> 16
    ab = (lo > hi).select(lo, hi)
    t = (ab + 31) >> 7
    sb = (t > 8).select(t - 8, fx.Int32(0))
    scale = (sb << 23).bitcast(fx.Float32)
    # byte 0 is the subnormal 2^-127, not 0.
    scale = (sb == 0).select(fx.Int32(0x00400000).bitcast(fx.Float32), scale)
    dws = []
    for b in range_constexpr(8):
        pair0 = Vec.from_elements([words[2 * b]], fx.Int32).bitcast(fx.BFloat16)
        pair1 = Vec.from_elements([words[2 * b + 1]], fx.Int32).bitcast(fx.BFloat16)
        v = fx.rocdl.cvt_scalef32_pk_fp8_bf16(
            T.i16x2, _llvm.mlir_undef(T.i16x2), pair0.ir_value(), scale.ir_value(), False
        )
        v = fx.rocdl.cvt_scalef32_pk_fp8_bf16(
            T.i16x2, v, pair1.ir_value(), scale.ir_value(), True
        )
        dws.append(fx.Int32(_llvm.bitcast(T.i32, v)))
    return dws, sb.to(fx.Float32)


def _mfma(a4, b4, acc, fp16=False):
    """v_mfma_f32_16x16x32_bf16 (``fp16``: _f16): lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i] (4 dwords each),
    gets D[4q + e][i]."""
    dtype = fx.Float16 if fp16 else fx.BFloat16
    a = Vec(a4).bitcast(dtype)
    b = Vec(b4).bitcast(dtype)
    op = fx.rocdl.mfma_f32_16x16x32_f16 if fp16 else fx.rocdl.mfma_f32_16x16x32_bf16
    return Vec(op(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def _exp2(x):
    """v_exp_f32 (no denormal range reduction: the P it feeds is rounded to bf16 or e4m3)."""
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


def _k_bf16x8(codes, tab_lo, tab_hi):
    """Eight K codes (one dword: byte b = dims 2 b / 2 b + 1 in its low / high nibble) under one block scale with
    magnitude table (tab_lo: magnitudes 0-3, tab_hi: 4-7; ``device.nvfp4_e4m3_lut``) -> four dwords of bf16 pairs, dims
    (0, 2) (4, 6) (1, 3) (5, 7) (the Q operand's order), of e4m3_rn_satfinite(e2m1 x scale): v_perm lookups, the
    code's sign merged, widened."""
    even = _perm(tab_hi, tab_lo, codes & MAG4) ^ ((codes << 4) & SIGN4)
    odd = _perm(tab_hi, tab_lo, (codes >> 4) & MAG4) ^ (codes & SIGN4)
    return [
        _bf16x2_e4m3(even, False),
        _bf16x2_e4m3(even, True),
        _bf16x2_e4m3(odd, False),
        _bf16x2_e4m3(odd, True),
    ]


def _f16x2_e4m3(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as an fp16 pair (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def pack_f16(a, b):
    return Vec.from_elements([a, b], fx.Float32).to(fx.Float16).bitcast(fx.Int32)[0]


def dequant_f16(codes, scales):
    """One dword of eight e2m1 codes times their four fp16 scale pairs -> four dwords of fp16 pairs (exact)."""
    out = []
    for j in range_constexpr(4):
        pair = Vec(
            fx.rocdl.cvt_scalef32_pk_f16_fp4(
                Vec.make_type(2, fx.Float16),
                fx.as_ir_value(fx.Int32(codes)),
                fx.as_ir_value(fx.Float32(1.0)),
                j,
            )
        )
        out.append((pair * scales[j]).bitcast(fx.Int32)[0])
    return out


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


def compile_nvfp4_sparse_decode(
    splits: int = 1,
    topk: int = 16,
    xcds: int = 0,
    quant: bool = False,
    exact_fp16: bool = False,
    tail: bool = False,
):
    """The launcher of one variant.

    Args:
        splits: workgroups per (row, KV head), each over a contiguous share of the row's blocks.
        topk: the top-k width (one lane per slot).
        xcds: > 0: workgroup w runs on XCD w % xcds; a request's rows (and a row's splits) share one XCD and its L2.
        quant: write the bf16 output's MXFP8 quant (the o_proj input) instead: e4m3 codes row-major or, at
            ``i32_tile`` 16, in the decode GEMM's (16, 16) tiles, and e8m0 scales in AITER's shuffled layout.
        exact_fp16: exact e2m1 x scale K / V with fp16 Q / P.
        tail: (needs ``xcds``) a 1-D grid: the whole CU rounds of (row, KV head) pairs at one split each, then the
            leftover pairs at ``splits`` each.
    """
    assert 1 <= topk <= 64 and splits >= 1
    assert not tail or xcds > 0, "the tail grid places a pair's splits by XCD"
    SPLITS, TOPK, XCDS, TAIL = splits, topk, xcds, tail
    PART_SCOPE = 0 if XCDS else AGENT_SCOPE
    mfma = "f16" if exact_fp16 else "bf16"  # the MFMA operand dtype
    name = (
        f"nvfp4_{mfma}mfma_sparse_decode_gqa16_d128_p128_top{TOPK}_s{SPLITS}"
        + (f"_xcd{XCDS}" if XCDS else "")
        + ("_mxfp8" if quant else "")
        + ("_tail" if tail else "")
    )

    @fx.struct
    class Shared:
        # partial O per (wave, d-tile, lane) as 16 B tiles, then (max, sum) per (wave, lane)
        o: fx.Array[fx.Int32, NW * NDT * 64 * 4, 16]
        # LLVM places fields by alignment, then size (both descending): the 16 B alignment keeps ml before the table
        ml: fx.Array[fx.Float32, NW * 2 * 64, 16]
        # the e4m3 magnitude table (lo, hi word per scale byte); its 8 B alignment folds the LDS offset into ds_read_b64
        lut: fx.Array[fx.Int32, 512, 8]
        ticket: fx.Array[fx.Int32, 4]

    def store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv):
        """O of ``row`` (``acc`` x ``inv``) in bf16."""
        n, g = lane % 16, lane // 16
        # lane (n, g) holds dims 8 (4 g + e) + dt of head n: per e, 8 consecutive dims
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        o_row = (row * i32_H + kvh * HG + n) * (D // 2) + g * 16
        for e in range_constexpr(4):
            val = Vec.from_elements(
                [
                    _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                    for j in range_constexpr(4)
                ],
                fx.Int32,
            )
            buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=row < i32_rows)

    def store_q(arg_qc, arg_qs, i32_rows, i32_H, i32_tile, row, kvh, lane, acc, inv):
        """The o_proj input of ``row``: MXFP8 codes of O (row-major, or the (16, 16) tiles when ``i32_tile`` is 16)
        and e8m0 scales in AITER's layout."""
        n, g = lane % 16, lane // 16
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
        # (16, 16) tiles: byte (m, k) at (m//16)*16K + (k//64)*1024 + ((k%64)//16)*256 + (m%16)*16 + k%16; the lane's
        # 32 bytes are two 16 B runs 256 B apart
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
            buffer_ops.buffer_store(val, qcr, off + h * step, mask=row < i32_rows)
        # scale (m, g): [M//32, K//256, klane 4, mlane 16, kpack 2, mpack 2]
        gi = head * 4 + g
        s_off = (
            ((((row // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4) * 16 + row % 16)
            * 2
            + (gi % 8) // 4
        ) * 2 + (row % 32) // 16
        rows32 = (i32_rows + 31) // 32 * 32
        qsr = buffer_ops.create_buffer_resource_from_addr(
            arg_qs, num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32)
        )
        buffer_ops.buffer_store(
            sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=row < i32_rows
        )

    # the arguments of the first (dependent) loads lead: the first 56 bytes arrive preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk stride over KV heads (elements)
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_lut: fx.Int64,  # [256] int64 K magnitude tables
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # splits: [grid rows * Hkv, S, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # splits: [grid rows * Hkv] int32 tickets, zero
        arg_qc: fx.Int64,  # quant: [ceil(rows / 16) * 16, H * 128] e4m3 codes
        arg_qs: fx.Int64,  # quant: [ceil(rows / 32) * 32, H * 4] e8m0 scales
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,  # cache page stride (bytes)
        i32_tile: fx.Int32,  # quant: 0 row-major codes, 16 the tile layout
        f32_c: fx.Float32,  # sm_scale * log2(e)
        i32_heavy: fx.Int32,  # tail: whole-round pairs (a multiple of XCDS)
        i32_tpx: fx.Int32,  # tail: tail pairs per XCD
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _global_i32_ptr(arg_topk)
        bt = _global_i32_ptr(arg_bt)
        seq_p = _global_i32_ptr(arg_seq)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        if const_expr(not exact_fp16):
            lut_g = _global_i32_ptr(arg_lut)
            smem.lut.ptr[tx] = fx.Int32(lut_g[tx])
            smem.lut.ptr[tx + 256] = fx.Int32(lut_g[tx + 256])
        lut64 = fx.recast_iter(fx.Int64, smem.lut.ptr)

        def tab(sc):
            """(lo, hi) words of scale byte ``sc``'s table: one ds_read_b64."""
            pair = Vec.from_elements([fx.Int64(lut64[sc])], fx.Int64).bitcast(fx.Int32)
            return [pair[0], pair[1]]

        split = fx.block_idx.z
        if const_expr(TAIL):
            # 1-D grid. Workgroups [0, heavy) are the pairs p = row * Hkv + kvh in [0, heavy), one split each, XCD x
            # taking the contiguous pairs [x heavy / XCDS, (x + 1) heavy / XCDS) (a request's pairs stay together).
            # Then the tail pairs heavy + x tpx + j at SPLITS splits each, all splits of a pair on XCD x (their
            # partials meet in its L2). Pairs past rows x Hkv are padding.
            xcd = bx % XCDS
            heavy = bx < i32_heavy
            t_slot = (bx - i32_heavy) // XCDS
            tid_t = xcd * i32_tpx + t_slot // SPLITS  # the tail pair (scratch row)
            pair = heavy.select(
                xcd * (i32_heavy // XCDS) + bx // XCDS, i32_heavy + tid_t
            )
            split = heavy.select(fx.Int32(0), t_slot % SPLITS)
            split = fx.Int32(fx.rocdl.readfirstlane(T.i32, split))
            tid_t = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid_t))
            heavy_i = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, heavy.select(fx.Int32(1), fx.Int32(0)))
            )
            row = fx.Int32(fx.rocdl.readfirstlane(T.i32, pair // i32_hkv))
            kvh = fx.Int32(fx.rocdl.readfirstlane(T.i32, pair % i32_hkv))
        else:
            if const_expr(XCDS > 0):
                # grid.x is padded to XCDS x ql: a permutation that puts each request's ql rows on one XCD (padding
                # rows map past i32_rows)
                xcd = bx % XCDS
                slot = bx // XCDS
                row = ((slot // i32_ql) * XCDS + xcd) * i32_ql + slot % i32_ql
            else:
                row = bx
        row_ok = row < i32_rows
        rowc = row_ok.select(row, fx.Int32(0))
        req = rowc // i32_ql
        pos = fx.Int32(seq_p[req]) - i32_ql + rowc % i32_ql
        # a graph's padding rows (seq_len 0) and the grid's padding see nothing
        pos = (row_ok & (pos >= 0)).select(pos, fx.Int32(-1))
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # lane = top-k slot: its logical block and page (-1: none)
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[kvh * i32_topk_hs + rowc * TOPK + slot_ok.select(lane, fx.Int32(0))]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int64(fx.rocdl.ballot(T.i64, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))
        if const_expr(SPLITS > 1):
            # this split's blocks: ranks [lo, lo + chunk) in slot order
            below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below).to(fx.Int32))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            if const_expr(TAIL):
                chunk = (heavy_i == fx.Int32(1)).select(n_v, chunk)  # whole row
            lo = split * chunk
            vmask = fx.Int64(
                fx.rocdl.ballot(T.i64, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))

        # causal limit of this wave's key offsets in the own block; Q^T
        lim = pos % BLK - w * KPW
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        q_row = (rowc * i32_H + kvh * HG + n) * (D // 2) + g * 16
        qf = [
            fx.Vector(
                buffer_ops.buffer_load(qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32)
            )
            for ks in range_constexpr(NKS)
        ]

        if const_expr(exact_fp16):
            qf = [x.bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32) for x in qf]
        else:
            # Q pairs in ``_k_bf16x8``'s dim order (0, 2) (4, 6) (1, 3) (5, 7): the K bytes need no interleave (16
            # v_perm fewer per block) for the same score bits
            qf = [
                Vec.from_elements(
                    [
                        _perm(x[1], x[0], 0x05040100),
                        _perm(x[3], x[2], 0x05040100),
                        _perm(x[1], x[0], 0x07060302),
                        _perm(x[3], x[2], 0x07060302),
                    ],
                    fx.Int32,
                )
                for x in qf
            ]

        # block offsets (bytes) of this KV head's K data / scales, V data / scales
        side_b = i32_hkv * (DATA_B + SCALE_B)
        k_data = kvh * DATA_B
        k_scl = i32_hkv * DATA_B + kvh * SCALE_B
        v_data = side_b + kvh * DATA_B
        v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
        # lane map: K rows 32 w + 8 (n // 4) + 4 st + n % 4, bytes [16 g, 16 g + 16) (dims 32 g..), their scales
        # (groups 2 g, 2 g + 1) one dword; V keys 32 w + 8 g + j of dims 8 n.. one 32-byte tile, their scales (group
        # n // 2) 8 contiguous bytes
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
        k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
        v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
        v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def load_block(pg):
            """This lane's NVFP4 dwords of page ``pg`` (RAW): K rows st = 0, 1 (4 + 4), their scales (1), V dims
            8 n + dt (8), the scales of their keys (2)."""
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(pg) * fx.Int64(i32_page_byte),
                num_records_bytes=i32_page_byte,
            )
            raw = []
            for st in range_constexpr(2):
                kd = fx.Vector(
                    buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
                )
                raw += [kd[i] for i in range_constexpr(4)]
            raw.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        pr, k_scale_load, vec_width=1, dtype=fx.Int32
                    )
                )
            )
            for h in range_constexpr(2):
                vd = fx.Vector(
                    buffer_ops.buffer_load(
                        pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32
                    )
                )
                raw += [vd[i] for i in range_constexpr(4)]
            vs = fx.Vector(
                buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
            )
            raw += [vs[i] for i in range_constexpr(2)]
            return raw

        def dequant_k(raw):
            """K frags kf[st][ks] of a block (4 dwords of bf16 pairs each)."""
            if const_expr(exact_fp16):
                kf = []
                for st in range_constexpr(2):
                    ks = _f16x2_e4m3(raw[8], st == 1)
                    kf.append(
                        [
                            Vec.from_elements(
                                dequant_f16(
                                    raw[st * 4 + j],
                                    [
                                        Vec.from_elements(
                                            [ks[j // 2], ks[j // 2]], fx.Float16
                                        )
                                        for _ in range_constexpr(4)
                                    ],
                                ),
                                fx.Int32,
                            )
                            for j in range_constexpr(NKS)
                        ]
                    )
                return kf
            kf = []
            for st in range_constexpr(2):
                tabs = []
                for grp in range_constexpr(2):
                    sb = raw[8].shrui(fx.Int32(8 * (2 * st + grp))) & fx.Int32(0xFF)
                    tabs.append(tuple(tab(sb)))
                kf.append(
                    [
                        Vec.from_elements(
                            _k_bf16x8(raw[st * 4 + ks], *tabs[ks // 2]), fx.Int32
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )
            return kf

        def dequant_v(raw, cur_blk):
            """V^T frags vf[dt] of a block (4 dwords of bf16 pairs each)."""
            if const_expr(exact_fp16):
                sp = []
                for j in range_constexpr(4):
                    pair = _f16x2_e4m3(raw[17 + j // 2], j % 2 == 1)
                    # unwritten tail scale bytes can be NaN, and P = 0 cannot mask a NaN V
                    sp.append(
                        Vec.from_elements(
                            [
                                (
                                    cur_blk * BLK + w * KPW + 8 * g + 2 * j + k <= pos
                                ).select(pair[k], fx.Float16(0.0))
                                for k in range_constexpr(2)
                            ],
                            fx.Float16,
                        )
                    )
                return [
                    Vec.from_elements(dequant_f16(raw[9 + dt], sp), fx.Int32)
                    for dt in range_constexpr(NDT)
                ]
            # V through the same table: regroup the eight dwords so each holds one key pair of four dims (one key's
            # table per dword), look the magnitudes up, merge the signs, interleave two keys per dim, widen to bf16
            def perm(hi, lo, sel):
                return fx.Int32(fx.rocdl.perm_b32(hi, lo, sel))

            x = [raw[9 + dt] for dt in range_constexpr(8)]
            for dt in range_constexpr(4):
                a, b = x[dt], x[dt + 4]
                x[dt] = perm(b, a, fx.Int32(0x05040100))
                x[dt + 4] = perm(b, a, fx.Int32(0x07060302))
            for r in (0, 1, 4, 5):
                a, b = x[r], x[r + 2]
                x[r] = perm(b, a, fx.Int32(0x06020400))
                x[r + 2] = perm(b, a, fx.Int32(0x07030501))
            vt = [tab((raw[17 + k // 4] >> (8 * (k % 4))) & 255) for k in range_constexpr(8)]
            z = []
            for kp in range_constexpr(4):
                even, odd = [], []
                for par in range_constexpr(2):
                    c = x[2 * kp + par]
                    te, to = vt[2 * kp], vt[2 * kp + 1]
                    even.append(perm(te[1], te[0], c & MAG4) ^ ((c << 4) & SIGN4))
                    odd.append(perm(to[1], to[0], (c >> 4) & MAG4) ^ (c & SIGN4))
                z.extend([even, odd])
            # z[k][par] holds key k's four dims of half ``par``. One interleave puts keys (2 j, 2 j + 1) side by side
            # per dim, [k d0, k' d0, k d1, k' d1], and the widening reads bytes (0, 1) or (2, 3): each result dword is
            # the operand pair (key 2 j, key 2 j + 1) of one dim, no second regroup.
            pairs = [[None] * 4 for _ in range_constexpr(8)]  # [d-tile][key pair j]
            for par in range_constexpr(2):
                for j in range_constexpr(4):
                    a, b = z[2 * j][par], z[2 * j + 1][par]
                    u = [perm(b, a, fx.Int32(0x05010400)), perm(b, a, fx.Int32(0x07030602))]
                    for i in range_constexpr(4):
                        pairs[par + 2 * i][j] = _bf16x2_e4m3(u[i // 2], i % 2 == 1)
            return [Vec.from_elements(p, fx.Int32) for p in pairs]

        if const_expr(not exact_fp16):
            fx.gpu.barrier()  # the LUT is in LDS

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # offset (within this wave's 32 keys) of S^T tile st, row 4 g + e
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        if const_expr(not exact_fp16):
            # The score masks as the S^T MFMAs' start values, per wave in its own part of ``o`` (idle until the wave's
            # combine stores): variant 0 no mask, 1 the own block's causal mask, 2 a masked block (all -inf), per S^T
            # tile. K is finite (the table maps NaN scale bytes to 0), so -inf plus the products stays -inf: the
            # scores of selecting -inf after the MFMAs, without the 8 selects per block.
            bias_base = w * (NDT * 64) + lane
            for var in range_constexpr(3):
                for st in range_constexpr(2):
                    if const_expr(var == 1):
                        vals = [
                            (key_off[4 * st + e] > lim).select(neg_inf, fx.Float32(0.0))
                            for e in range_constexpr(4)
                        ]
                    else:
                        vals = [neg_inf if var == 2 else fx.Float32(0.0)] * 4
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(Vec.from_elements(vals, fx.Float32).bitcast(fx.Int32))
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(o_lds, (None, bias_base + (2 * var + st) * 64)),
                    )

        def bias_of(var):
            """The two S^T tiles' start values of mask variant ``var`` (uniform)."""
            out = []
            for st in range_constexpr(2):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    lds_atom,
                    fx.slice(o_lds, (None, bias_base + var * 128 + st * 64)),
                    r,
                )
                out.append(Vec(r.load()).bitcast(fx.Float32))
            return out

        def mask_var(cur_blk, live):
            return (live == fx.Int32(0)).select(
                fx.Int32(2), (cur_blk == self_blk).select(fx.Int32(1), fx.Int32(0))
            )

        first = (n_v > 0).select(
            fx.Int32(fx.math.cttz(vmask).to(fx.Int32)), fx.Int32(0)
        )

        def slot_at(mask):
            """The lowest slot left in ``mask`` (the first one again when empty)."""
            return (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(kf, vf, cur_blk, live, m_t, l_t, o_t, bias=None):
            """One dequantized block into the running (max, sum, O): S^T, the own block's causal mask, the online
            softmax, O^T += V^T P^T. A block that is not ``live`` is masked whole (max unmoved, alpha 1, P 0);
            ``bias``: the masks as the S^T start values (``bias_of``) instead."""
            a = []
            for st in range_constexpr(2):
                acc = zero4 if bias is None else bias[st]
                for ks in range_constexpr(NKS):
                    acc = _mfma(kf[st][ks], qf[ks], acc, exact_fp16)
                a += [acc[e] for e in range_constexpr(4)]
            if const_expr(bias is None):
                diag = cur_blk == self_blk
                dead = live == fx.Int32(0)
                a = [
                    (dead | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                    for i in range_constexpr(8)
                ]
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = _maxf_nn(mx, a[i])
            # the running max is kept in scaled units: max(a) * c == max(a * c) (c > 0)
            m_new = _maxf_nn(m_t, _xlane4(mx, _maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            alpha = _exp2(m_t - m_use)
            p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            pack = pack_f16 if exact_fp16 else _pk_bf16
            pf = Vec.from_elements(
                [pack(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                fx.Int32,
            )
            # alpha == 1 leaves O's bits unchanged: the fp16 path skips that rescale
            av = Vec.filled(4, alpha, fx.Float32)
            if const_expr(exact_fp16):
                if alpha != fx.Float32(1.0):
                    o_t = [o_t[dt] * av for dt in range_constexpr(NDT)]
                o_t = [_mfma(vf[dt], pf, o_t[dt], True) for dt in range_constexpr(NDT)]
            else:
                o_t = [_mfma(vf[dt], pf, o_t[dt] * av) for dt in range_constexpr(NDT)]
            return m_new, l_t * alpha + ps, o_t

        # two blocks (A, B) per iteration, the next A's loads in flight; an odd last B is masked. fp16 dequantizes V
        # after the next load, bf16 before it.
        init = [vmask, neg_inf, fx.Float32(0.0)]
        init += [zero4 for _ in range_constexpr(NDT)]
        init += load_block(page_of(slot_at(vmask)))
        n_st = 3 + NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + 1) // 2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m_t = fx.Float32(st_[1])
            l_t = fx.Float32(st_[2])
            o_t = [Vec(st_[3 + dt]) for dt in range_constexpr(NDT)]
            raw_a = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            blk_a = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem)))
            rem_b = rem & (rem - fx.Int64(1))
            raw_b = load_block(page_of(slot_at(rem_b)))
            if const_expr(not exact_fp16):
                blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
                live_b = (rem_b != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
                bias_a = bias_of(mask_var(blk_a, fx.Int32(1)))
                bias_b = bias_of(mask_var(blk_b, live_b))
            fx.rocdl.sched_barrier(0)
            kf_a = dequant_k(raw_a)
            if const_expr(not exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            rem_n = rem_b & (rem_b - fx.Int64(1))
            fx.rocdl.sched_barrier(0)
            raw_n = load_block(page_of(slot_at(rem_n)))
            fx.rocdl.sched_barrier(0)
            if const_expr(exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            if const_expr(exact_fp16):
                res = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, l_t, o_t)
            else:
                res = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, l_t, o_t, bias_a)
            m_t, l_t, o_t = res[0], res[1], res[2]
            if const_expr(exact_fp16):
                blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
                live_b = (rem_b != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
            kf_b = dequant_k(raw_b)
            vf_b = dequant_v(raw_b, blk_b)
            if const_expr(exact_fp16):
                res = attend(kf_b, vf_b, blk_b, live_b, m_t, l_t, o_t)
            else:
                res = attend(kf_b, vf_b, blk_b, live_b, m_t, l_t, o_t, bias_b)
            m_t, l_t, o_t = res[0], res[1], res[2]
            state = [rem_n, m_t, l_t] + o_t + raw_n
            results = yield [fx.as_ir_value(x) for x in state]

        # combine the waves' partials (wave 0)
        ml = smem.ml.ptr
        ml[(w * 2) * 64 + lane] = fx.Float32(results[1])
        ml[(w * 2 + 1) * 64 + lane] = fx.Float32(results[2])
        for dt in range_constexpr(NDT):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(Vec(results[3 + dt]).bitcast(fx.Int32))
            fx.copy(lds_atom, r, fx.slice(o_lds, (None, (w * NDT + dt) * 64 + lane)))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = _maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = _xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(o_lds, (None, (ww * NDT + dt) * 64 + lane)),
                        r,
                    )
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(SPLITS > 1):
                # this split's partial; the row's last split combines. In the tail grid the whole-round workgroups
                # (heavy_i) write directly.
                if const_expr(TAIL):
                    tid = tid_t
                    zsp = split
                    split_on = heavy_i == fx.Int32(0)
                else:
                    tid = row * i32_hkv + kvh
                    zsp = fx.block_idx.z
                    split_on = True
                if split_on:
                    pr = buffer_ops.create_buffer_resource_from_addr(
                        arg_part, num_records_bytes=i64_part_bytes
                    )
                    row_f = (tid * SPLITS + zsp) * 64 + lane
                    for dt in range_constexpr(NDT):
                        buffer_ops.buffer_store(
                            acc[dt].bitcast(fx.Int32),
                            pr,
                            row_f * PART_F + dt * 4,
                            cache_modifier=PART_SCOPE,
                        )
                    buffer_ops.buffer_store(
                        Vec.from_elements([_bits(big_raw), _bits(l_tot)], fx.Int32),
                        pr,
                        row_f * PART_F + NDT * 4,
                        cache_modifier=PART_SCOPE,
                    )
                    # the partial stores are done before the ticket is taken
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    if lane == 0:
                        smem.ticket.ptr[0] = _global_atomic_add(
                            arg_cnt + fx.Int64(tid) * 4, 1
                        )
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    if (
                        fx.Int32(
                            fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                        )
                        == SPLITS - 1
                    ):
                        rows_f = [
                            (tid * SPLITS + sp) * 64 + lane
                            for sp in range_constexpr(SPLITS)
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
                        for sp in range_constexpr(1, SPLITS):
                            top = _maxf_nn(top, bigs[sp])
                        top = (top == neg_inf).select(fx.Float32(0.0), top)
                        ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                        l_all = fx.Float32(0.0)
                        for sp in range_constexpr(SPLITS):
                            l_all = l_all + _as_f32(mls[sp][1]) * ws[sp]
                        inv_all = (l_all > fx.Float32(0.0)).select(
                            v_scale / l_all, fx.Float32(0.0)
                        )
                        acc_all = [zero4 for _ in range_constexpr(NDT)]
                        for sp in range_constexpr(SPLITS):
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
                        if const_expr(quant):
                            store_q(
                                arg_qc,
                                arg_qs,
                                i32_rows,
                                i32_H,
                                i32_tile,
                                row,
                                kvh,
                                lane,
                                acc_all,
                                inv_all,
                            )
                        else:
                            store_out(
                                arg_out,
                                i32_rows,
                                i32_H,
                                row,
                                kvh,
                                lane,
                                acc_all,
                                inv_all,
                            )
                        if lane == 0:
                            cnt_p = _global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)
                            cnt_p[0] = fx.Int32(0)
                if const_expr(TAIL):  # noqa: SIM102 (compile-time, then runtime)
                    if heavy_i == fx.Int32(1):
                        inv = (l_tot > fx.Float32(0.0)).select(
                            v_scale / l_tot, fx.Float32(0.0)
                        )
                        if const_expr(quant):
                            store_q(
                                arg_qc,
                                arg_qs,
                                i32_rows,
                                i32_H,
                                i32_tile,
                                row,
                                kvh,
                                lane,
                                acc,
                                inv,
                            )
                        else:
                            store_out(
                                arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv
                            )
            else:
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                if const_expr(quant):
                    store_q(
                        arg_qc,
                        arg_qs,
                        i32_rows,
                        i32_H,
                        i32_tile,
                        row,
                        kvh,
                        lane,
                        acc,
                        inv,
                    )
                else:
                    store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        arg_qc: fx.Int64,
        arg_qs: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_tile: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        i32_grid_x: fx.Int32,
        i32_heavy: fx.Int32,
        i32_tpx: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_lut,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            arg_qc,
            arg_qs,
            i64_part_bytes,
            i32_page_b,
            i32_tile,
            f32_c,
            i32_heavy,
            i32_tpx,
        ).launch(
            grid=(i32_grid_x, 1, 1) if TAIL else (i32_grid_x, i32_hkv, SPLITS),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
