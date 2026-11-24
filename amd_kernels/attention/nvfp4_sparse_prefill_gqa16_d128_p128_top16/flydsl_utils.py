# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FlyDSL code shared by the prefill kernels: bit casts, cross-lane ops, conversions, MFMAs, the page load, the
dequants and the output epilogue."""

import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

BLK = 128  # tokens per page
D = 128
HG = 16  # query heads per KV head
KPW = BLK // 4  # keys per wave and page
DATA_B = BLK * D // 2  # packed e2m1 bytes per head per side
SCALE_B = BLK * D // 16  # e4m3 block-scale bytes per head per side
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
PACK_ROW_PER_HEAD = 132  # packed row bytes per head: 128 e4m3 codes + 4 e8m0 scales
LAZY_LOG2 = 8.0  # the lazy rescale's re-base bound in log2 units: fp16 P <= 2^8 = 256


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


def _maxf_nn(a, b):
    """v_max_f32 with nnan (no canonicalizing v_max x, x, x on the inputs)."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _xlane4(x, op):
    """``op`` over lanes L, L^16, L^32, L^48, broadcast to all four (two swaps)."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _xlane4_or(x):
    """Bitwise or over lanes L, L^16, L^32, L^48 (Int32), broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, fx.Int32(x), fx.Int32(x))
    m = a | b
    a, b = _permlane_swap(_rocdl.permlane16_swap, m, m)
    return a | b


def _pk_bf16(a, b):
    """Two fp32 -> one dword of two bf16 (RNE, v_cvt_pk_bf16_f32), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _pk_f16(a, b):
    """Two fp32 -> one dword of two fp16 (RNE), ``a`` low."""
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
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


def _pack_fp8(x):
    """Eight fp32 -> eight e4m3 bytes in an i64, ``x[0]`` the lowest."""
    words = []
    for i in range_constexpr(2):
        lo = _pk_fp8(x[4 * i], x[4 * i + 1], fx.Int32(_llvm.mlir_undef(T.i32)), False)
        words.append(_pk_fp8(x[4 * i + 2], x[4 * i + 3], lo, True))
    return Vec.from_elements(words, fx.Int32).bitcast(fx.Int64)[0]


def _join4(vs):
    """Four i64 -> one 8-dword vector, ``vs[0]`` first."""
    words = []
    for i in range_constexpr(4):
        v = Vec.from_elements([vs[i]], fx.Int64).bitcast(fx.Int32)
        words += [v[j] for j in range_constexpr(2)]
    return Vec.from_elements(words, fx.Int32)


def _exp2(x):
    """v_exp_f32 (no denormal range reduction: the P it feeds is rounded to 16 or 8 bits)."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _fp4x2_f16(word, sel):
    """E2M1 codes of byte ``sel`` of ``word`` (low nibble first) as two fp16."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(word),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _e4m3x2_f16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two fp16 (exact)."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(word),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def _v_e4m3x2_f16(word, hi):
    """``_e4m3x2_f16`` with NaN mapped to -448 (MaxNum; finite E4M3 scales are >= -448): unwritten V tails."""
    return Vec(
        _arith.MaxNumFOp(
            fx.as_ir_value(_e4m3x2_f16(word, hi)),
            fx.as_ir_value(Vec.filled(2, -448.0, fx.Float16)),
        ).result
    )


def _q_f16(v):
    """Four dwords of bf16 pairs -> fp16 pairs (exact for |q| in [2^-17, 65504])."""
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def _p_f16x(pk):
    """A bf16 pair of P (0 <= P < 4, low half first) -> the fp16 pair of P x 2^14, per
    16-bit half max(b, 98 << 7) x 8 + 0x7800: bf16 exponent e -> fp16 exponent e - 98,
    the 7 mantissa bits -> the top 7 of fp16's 10. Bit exact for P >= 2^-28 (fp16
    normal); smaller P (< 2^-28 of the running max) give 0 or a value below 2^-28."""
    v = Vec.from_elements([fx.Int32(pk)], fx.Int32).bitcast(fx.Int16)
    m = _arith.MaxUIOp(
        fx.as_ir_value(v), fx.as_ir_value(Vec.filled(2, 98 << 7, fx.Int16))
    ).result
    r = _arith.MulIOp(m, fx.as_ir_value(Vec.filled(2, 8, fx.Int16))).result
    r = _arith.AddIOp(r, fx.as_ir_value(Vec.filled(2, 0x7800, fx.Int16))).result
    return Vec(r).bitcast(fx.Int32)[0]


def _p_f16x_usub(pk):
    """``_p_f16x`` as usub.sat(b, 98 << 7) x 8 per half: the same bits modulo 2^16."""
    v = Vec.from_elements([fx.Int32(pk)], fx.Int32).bitcast(fx.Int16)
    r = _llvm.call_intrinsic(
        Vec.make_type(2, fx.Int16), "llvm.usub.sat.v2i16",
        [fx.as_ir_value(v), fx.as_ir_value(Vec.filled(2, 98 << 7, fx.Int16))], [], [],
    )
    r = _arith.MulIOp(r, fx.as_ir_value(Vec.filled(2, 8, fx.Int16))).result
    return Vec(r).bitcast(fx.Int32)[0]


def _mfma_f16(a4, b4, acc):
    """v_mfma_f32_16x16x32_f16: lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i] (4 dwords each), gets D[4q + e][i]."""
    a = Vec(a4).bitcast(fx.Float16)
    b = Vec(b4).bitcast(fx.Float16)
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_f16(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def _mfma_fp8(a, b, acc):
    """v_mfma_f32_16x16x32_fp8_fp8 (e4m3 operands, one i64 each)."""
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def _mfma128(a, b, acc):
    """v_mfma_scale_f32_16x16x128_f8f6f4 with e4m3 operands (8 dwords each) and unit scales."""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    sel = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ops.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(acc),
            e4m3,
            e4m3,
            sel,
            zero,
            sel,
            zero,
        ).result
    )


def _page_offsets(i32_hkv, kvh, w, n, g):
    """Lane (n, g) of wave w: the page dword offsets of its two K rows, their scales, its V dims and their scales."""
    # page offsets (bytes) of this KV head's K data / scales, V data / scales
    side_b = i32_hkv * (DATA_B + SCALE_B)
    k_data = kvh * DATA_B
    k_scl = i32_hkv * DATA_B + kvh * SCALE_B
    v_data = side_b + kvh * DATA_B
    v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
    # K: rows 32 w + 8 (n // 4) + 4 st + n % 4, bytes [16 g, 16 g + 16) (dims 32 g..); the scales of both rows
    # (groups 2 g, 2 g + 1) are one dword
    k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
    k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
    k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
    # V: keys 32 w + 8 g + j of dims 8 n.. are one 32-byte (token octet, dim octet) tile, dword dt = dim 8 n + dt
    # (a byte = two consecutive keys); their scales (group n // 2) are 8 contiguous bytes
    v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
    v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4
    return k_load, k_scale_load, v_load, v_scale_load


def _load_page(page, arg_cache, i32_page_byte, k_load, k_scale_load, v_load, v_scale_load):
    """This lane's 19 NVFP4 dwords of ``page``: K rows st = 0, 1 (4 + 4), their scales (1), V dims 8 n + dt (8), the
    scales of their keys (2)."""
    pr = buffer_ops.create_buffer_resource_from_addr(
        arg_cache + fx.Int64(page) * fx.Int64(i32_page_byte),
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


def _dequant_f16(raw, v_scale_f16=_e4m3x2_f16):
    """K frags kf[st][ks] and V^T frags vf[dt] of a page as exact fp16 (4 dwords of fp16 pairs each).

    Args:
        raw: the page's dwords from ``_load_page``.
        v_scale_f16: the V scale conversion (``_v_e4m3x2_f16`` maps a NaN tail scale to a finite one).
    """
    kf = []
    for st in range_constexpr(2):
        ksc = _e4m3x2_f16(raw[8], st == 1)  # row st: groups 2 g, 2 g + 1
        row = []
        for ks in range_constexpr(NKS):
            sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
            row.append(
                Vec.from_elements(
                    [
                        (_fp4x2_f16(raw[st * 4 + ks], i) * sv).bitcast(
                            fx.Int32
                        )[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
            )
        kf.append(row)
    # vs[i]: the block scales of key slots 2 i, 2 i + 1
    vs = []
    for h in range_constexpr(2):
        vs += [
            v_scale_f16(raw[17 + h], False),
            v_scale_f16(raw[17 + h], True),
        ]
    # V^T frag dt (dim 8 n + dt): data byte i of its dword holds key slots 2 i, 2 i + 1, the operand's fp16 pair
    vf = [
        Vec.from_elements(
            [
                (_fp4x2_f16(raw[9 + dt], i) * vs[i]).bitcast(fx.Int32)[0]
                for i in range_constexpr(4)
            ],
            fx.Int32,
        )
        for dt in range_constexpr(NDT)
    ]
    return kf, vf


def _tab(smem, sc):
    """The fp8 values of the 8 E2M1 magnitudes under scale byte ``sc`` (``query.nvfp4_e4m3_lut``), two dwords."""
    # Shared.lut sits at LDS offset 0: the address is sc * 8 (one SDWA shift)
    value = fx.Int64(smem.lut.ptr[sc])
    pair = Vec.from_elements([value], fx.Int64).bitcast(fx.Int32)
    return [pair[0], pair[1]]


def _perm(hi, lo, sel):
    return fx.Int32(_rocdl.perm_b32(hi, lo, sel))


def _dq8(c, t):
    """The e4m3 bytes of dword ``c``'s eight E2M1 codes via the table pair ``t`` (low nibbles first), signs kept."""
    even = _perm(t[1], t[0], c & 0x07070707) ^ ((c << 4) & -2139062144)
    odd = _perm(t[1], t[0], (c >> 4) & 0x07070707) ^ (c & -2139062144)
    return Vec.from_elements([even, odd], fx.Int32).bitcast(fx.Int64)[0]


def _dequant_fp8(raw, smem):
    """K frags kf[st][ks] and V^T frags vf[dt] of a page as e4m3 bytes (one i64 each), from the table in LDS."""
    kt = [_tab(smem, (raw[8] >> (8 * i)) & 255) for i in range_constexpr(4)]
    kf = [
        [
            _dq8(raw[st * 4 + ks], kt[st * 2 + ks // 2])
            for ks in range_constexpr(NKS)
        ]
        for st in range_constexpr(2)
    ]
    x = [raw[9 + dt] for dt in range_constexpr(8)]
    for dt in range_constexpr(4):
        a, b = x[dt], x[dt + 4]
        x[dt] = _perm(b, a, fx.Int32(0x05040100))
        x[dt + 4] = _perm(b, a, fx.Int32(0x07060302))
    for r in (0, 1, 4, 5):
        a, b = x[r], x[r + 2]
        x[r] = _perm(b, a, fx.Int32(0x06020400))
        x[r + 2] = _perm(b, a, fx.Int32(0x07030501))
    vt = [
        _tab(smem, (raw[17 + k // 4] >> (8 * (k % 4))) & 255)
        for k in range_constexpr(8)
    ]
    z = []
    for kp in range_constexpr(4):
        even, odd = [], []
        for par in range_constexpr(2):
            c = x[2 * kp + par]
            te, to = vt[2 * kp], vt[2 * kp + 1]
            even.append(
                _perm(te[1], te[0], c & 0x07070707) ^ ((c << 4) & -2139062144)
            )
            odd.append(
                _perm(to[1], to[0], (c >> 4) & 0x07070707) ^ (c & -2139062144)
            )
        z.extend([even, odd])
    words = [[fx.Int32(0), fx.Int32(0)] for dt in range_constexpr(8)]
    for par in range_constexpr(2):
        for half in range_constexpr(2):
            a, b, c, d = [z[4 * half + k][par] for k in range_constexpr(4)]
            u0 = _perm(b, a, fx.Int32(0x05010400))
            u1 = _perm(b, a, fx.Int32(0x07030602))
            v0 = _perm(d, c, fx.Int32(0x05010400))
            v1 = _perm(d, c, fx.Int32(0x07030602))
            vals = [
                _perm(v0, u0, fx.Int32(0x05040100)),
                _perm(v0, u0, fx.Int32(0x07060302)),
                _perm(v1, u1, fx.Int32(0x05040100)),
                _perm(v1, u1, fx.Int32(0x07060302)),
            ]
            for i in range_constexpr(4):
                words[par + 2 * i][half] = vals[i]
    vf = [Vec.from_elements(w, fx.Int32).bitcast(fx.Int64)[0] for w in words]
    return kf, vf


def _store_out(words, tok, kvh, n, g, i32_T, i32_H, arg_out, arg_pack, pack, quant):
    """Store lane (n, g)'s output row piece: dims [32 g, 32 g + 32) of head ``kvh`` * 16 + n of token ``tok``.

    Args:
        words: the 16 bf16-pair dwords (``words[e][j]``: dims 8 (4 g + e) + 2 j, + 1).
        pack: the MXFP8 quant (``reference.mxfp8_quantize``; the lane's 32 dims are one group) as sequence-parallel
            rows to ``arg_pack``.
        quant: the MXFP8 codes row-major to ``arg_out``, the scales AITER-shuffled to ``arg_pack``.
    Otherwise the bf16 row goes to ``arg_out``.
    """
    if pack or quant:
        mag = Vec.filled(2, 0x0080, fx.Int16)
        for e in range_constexpr(4):
            for j in range_constexpr(4):
                v = Vec.from_elements([words[e][j] & 0x7FFF7FFF], fx.Int32).bitcast(fx.Int16)
                mag = Vec(_arith.MaxUIOp(fx.as_ir_value(mag), fx.as_ir_value(v)).result)
        lo = mag[0].to(fx.Int32)
        hi = mag[1].to(fx.Int32)
        amax = (lo > hi).select(lo, hi)
        # 448 = 1.75 * 2^8: the bf16 mantissa decides whether ceil(log2(amax / 448)) rounds up.
        sb = (amax >> 7) - 8 + ((amax & 127) > 96).select(fx.Int32(1), fx.Int32(0))
        sb = (sb < 0).select(fx.Int32(0), sb)
        # The down-conversion divides by the scale; e8m0 zero is the fp32 subnormal 2^-127.
        rs = (sb == 0).select(fx.Int32(0x00400000), sb << 23).bitcast(fx.Float32)
        flat = [words[e][j] for e in range_constexpr(4) for j in range_constexpr(4)]
        dws = []
        for b in range_constexpr(8):
            dst = Vec.from_elements([fx.Int32(0)], fx.Int32).bitcast(fx.Int16)
            for h in range_constexpr(2):
                src = Vec.from_elements([flat[2 * b + h]], fx.Int32).bitcast(fx.BFloat16)
                dst = Vec(
                    _rocdl_ops.cvt_scalef32_pk_fp8_bf16(
                        Vec.make_type(2, fx.Int16), fx.as_ir_value(dst), fx.as_ir_value(src),
                        fx.as_ir_value(rs), h == 1,
                    )
                )
            dws.append(dst.bitcast(fx.Int32)[0])
        if quant:
            k_row = i32_H * D
            head = kvh * HG + n
            qcr = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(k_row)
            )
            q_off = (tok * k_row + head * D + g * 32) // 4
            for h in range_constexpr(2):
                val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                buffer_ops.buffer_store(
                    val, qcr, q_off + h * 4, mask=tok < i32_T
                )
            # AITER scales: [M / 32, K / 256, 4, 16, 2, 2]
            gi = head * 4 + g
            s_off = (
                (
                    (((tok // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4)
                    * 16
                    + tok % 16
                )
                * 2
                + (gi % 8) // 4
            ) * 2 + (tok % 32) // 16
            rows32 = (i32_T + 31) // 32 * 32
            qsr = buffer_ops.create_buffer_resource_from_addr(
                arg_pack,
                num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32),
            )
            buffer_ops.buffer_store(
                sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=tok < i32_T
            )
        else:
            row_dw = i32_H * (PACK_ROW_PER_HEAD // 4)
            pack_bytes = (
                fx.Int64(i32_T) * fx.Int64(i32_H) * PACK_ROW_PER_HEAD
            )
            packr = buffer_ops.create_buffer_resource_from_addr(
                arg_pack, num_records_bytes=pack_bytes
            )
            p_row = tok * row_dw + (kvh * HG + n) * (D // 4) + g * 8
            for h in range_constexpr(2):
                val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                buffer_ops.buffer_store(
                    val, packr, p_row + h * 4, mask=tok < i32_T
                )
            # the head's four scale bytes: one dword after the codes
            sdw = _xlane4_or(sb.to(fx.Int32) << (g * 8))
            buffer_ops.buffer_store(
                sdw,
                packr,
                tok * row_dw + i32_H * (D // 4) + kvh * HG + n,
                mask=(tok < i32_T) & (g == 0),
            )
    else:
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out,
            num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2),
        )
        o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + g * 16
        for e in range_constexpr(4):
            val = Vec.from_elements(words[e], fx.Int32)
            buffer_ops.buffer_store(
                val, outr, o_row + e * 4, mask=tok < i32_T
            )
