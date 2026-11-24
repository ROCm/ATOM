# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SwiGLU-OAI of a split-layout gate_up row + MXFP8 (1x32, E8M0) quant of the result, FlyDSL (gfx950),
I = 3072 output columns per row (the M3.1 dense FFN at TP4: gate_up N 6144 = [gate 3072 | up 3072]).

``form`` (compile time, the caller's numerics; never chosen by row count):
  ``bf16_chain``: the aten chain of ``SiluAndMulWithClamp.forward_native`` (each op in fp32, rounded to
  bf16): g = min(gate, limit); u = clamp(up, -limit, limit); t1 = bf(alpha * g); t2 = bf(1 / (1 + exp(-t1)));
  t3 = bf(g * t2); t4 = bf(u + beta); act = bf(t3 * t4)  (the DSpark draft).
  ``fp32``: ``swiglu_oai_split``'s form: act = bf((g * (1 / (1 + exp(-alpha * g)))) * (u + beta)), fp32
  throughout, one rounding (the target's dense layers 0-2).
The quant then reads the bf16 ``act`` exactly as the Triton MXFP8 quantizer (``_mxfp8_quantize_kernel``)
does: amax over 32 columns, scale byte ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)``, codes by
``v_cvt_scalef32_pk_fp8_f32`` of ``act * 2^(127 - byte)`` with MODE.fp16_ovfl. Codes row-major [m, I] or in the
decode GEMM's (16, 16) layout; scales in AITER's shuffled layout.

Layout: six independent workgroups per row, one 64-lane wave per 512-column piece.
Thread t owns eight columns; each 32-column MX group stays within four adjacent lanes.
The XCD mapping deals out pieces; the activation and quantization order is unchanged."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

I = 3072  # output columns per row
PIECES = 6
NT = I // 8 // PIECES  # one wave per 512-column piece
SCALE_K1 = I // 256  # AITER scale layout: 256-column blocks per row
TINY = 0x00800000  # fp32 tiny, the amax floor
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
FORMS = ("bf16_chain", "fp32")

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _max_dpp(a, ctrl):
    b = fx.Int32(
        _rocdl_ops.update_dpp(
            T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(_bits(a)), ctrl, 0xF, 0xF, True
        )
    ).bitcast(fx.Float32)
    return _maxnum(a, b)


def _swap_xor1(words):
    return [
        fx.Int32(
            _rocdl_ops.update_dpp(
                T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(w), 0xB1, 0xF, 0xF, True
            )
        )
        for w in words
    ]


def _unpack8(v):
    """8 fp32 of 4 dwords of bf16 (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pk(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _bf(vals):
    """Round 8 fp32 to bf16 (RNE) and back to fp32."""
    return _unpack8([_pk(vals[2 * i], vals[2 * i + 1]) for i in range(4)])


def _set_fp16_ovfl():
    _llvm.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()],
        [],
        [],
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _e8m0(y):
    """(scale byte as fp32, 2^(127 - byte)) of the 32-column group whose 4 lanes hold ``y``."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = _maxnum(a, fx.math.absf(v))
    a = _max_dpp(a, 0x4E)  # quad_perm [2, 3, 0, 1]
    a = _maxnum(_max_dpp(a, 0xB1), _f32(TINY))  # quad_perm [1, 0, 3, 2]
    d = a / fx.Float32(448.0)
    small = d < _f32(TINY)
    lg = fx.Float32(
        _rocdl_ops.log(T.f32, fx.as_ir_value(small.select(d * fx.Float32(2.0**32), d)))
    ) - small.select(fx.Float32(32.0), fx.Float32(0.0))
    sb = _minnum(_maxnum(fx.math.ceil(lg) + fx.Float32(127.0), fx.Float32(0.0)), fx.Float32(254.0))
    rescale = fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(127.0) - sb)))
    return sb, rescale


def _codes(y, rescale):
    q = [v * rescale for v in y]
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[4], q[5], False), q[6], q[7], True)
    return [lo, hi]


def _scale_off(m, g):
    """Byte of (row, group) in AITER's shuffled scale layout."""
    return (
        ((((m // 32) * SCALE_K1 + g // 8) * 4 + g % 4) * 16 + m % 16) * 2 + (g % 8) // 4
    ) * 2 + (m % 32) // 16


def _code_off(m, p):
    """Byte of (row, 16-column piece) in the (16, 16) code layout."""
    return (m // 16) * (16 * I) + (p // 4) * 1024 + (p % 4) * 256 + (m % 16) * 16


def _xcd_row(bid, m):
    x, k = bid % 8, bid // 8
    q8, r8 = m // 8, m % 8
    extra = x < r8
    row = x * q8 + extra.select(x, r8) + k
    return row, k < q8 + extra.select(fx.Int32(1), fx.Int32(0))


def _xcd_grid(m):
    return (m + 7) // 8 * 8


def _swiglu_bf16_chain(gate, up, alpha, beta, limit):
    g = [_minnum(v, limit) for v in gate]
    u = [_minnum(_maxnum(v, -limit), limit) for v in up]
    t1 = _bf([alpha * v for v in g])
    t2 = _bf([fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-v)) for v in t1])
    t3 = _bf([a * b for a, b in zip(g, t2)])
    t4 = _bf([v + beta for v in u])
    return _bf([a * b for a, b in zip(t3, t4)])


def _swiglu_fp32(gate, up, alpha, beta, limit):
    """swiglu_oai_split's form: fp32 math, one bf16 rounding of the result."""
    out = []
    for gv, uv in zip(gate, up):
        g = _minnum(gv, limit)
        u = _minnum(_maxnum(uv, -limit), limit)
        sig = fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-(alpha * g)))
        out.append((g * sig) * (u + beta))
    return _bf(out)


def compile_swiglu_oai_mxfp8_partitioned(form="bf16_chain", shuffled=False):
    assert form in FORMS, form
    name = f"swiglu_oai_partition{PIECES}_{form}_mxfp8_{'tiled' if shuffled else 'rowmajor'}_i{I}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sx: fx.Int32,  # gate_up row stride (elements)
        arg_x: fx.Int64,  # [m, 2I] bf16
        arg_q: fx.Int64,  # codes e4m3
        arg_s: fx.Int64,  # scales u8, AITER layout
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_alpha: fx.Int32,  # fp32 bits
        i32_beta: fx.Int32,
        i32_limit: fx.Int32,
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        item, live = _xcd_row(fx.block_idx.x, i32_m*PIECES)
        row=item//PIECES
        tx=tx+(item%PIECES)*NT
        if live:
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(row) * fx.Int64(i32_sx * 2), num_records_bytes=4 * I
            )
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i64_q_bytes)
            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i64_s_bytes)
            gv = _unpack8(Vec(buffer_ops.buffer_load(xr, tx * 4, vec_width=4, dtype=fx.Int32)))
            uv = _unpack8(Vec(buffer_ops.buffer_load(xr, I // 2 + tx * 4, vec_width=4, dtype=fx.Int32)))
            alpha = i32_alpha.bitcast(fx.Float32)
            beta = i32_beta.bitcast(fx.Float32)
            limit = i32_limit.bitcast(fx.Float32)
            if const_expr(form == "bf16_chain"):
                y = _swiglu_bf16_chain(gv, uv, alpha, beta, limit)
            else:
                y = _swiglu_fp32(gv, uv, alpha, beta, limit)
            sb, rescale = _e8m0(y)
            codes = _codes(y, rescale)
            if const_expr(shuffled):
                recv = _swap_xor1(codes)
                piece = Vec.from_elements([codes[0], codes[1], recv[0], recv[1]], fx.Int32)
                buffer_ops.buffer_store(piece, qr, _code_off(row, tx // 2) // 4, mask=(tx % 2) == 0)
            else:
                buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), qr, row * (I // 4) + tx * 2)
            if (tx % 4) == 0:
                buffer_ops.buffer_store(
                    fx.Int32(sb.to(fx.Int32)).to(fx.Int8), sr, _scale_off(row, tx // 4), offset_is_bytes=True
                )

    @flyc.jit
    def launch_swiglu_oai_mxfp8_partitioned(
        arg_x: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_alpha: fx.Int32,
        i32_beta: fx.Int32,
        i32_limit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, i32_sx, arg_x, arg_q, arg_s, i64_q_bytes, i64_s_bytes, i32_alpha, i32_beta, i32_limit
        ).launch(grid=(fx.Int64(_xcd_grid(i32_m*PIECES)), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_swiglu_oai_mxfp8_partitioned.kernel_name = name
    launch_swiglu_oai_mxfp8_partitioned.compile_hints = PRELOAD_HINTS
    return launch_swiglu_oai_mxfp8_partitioned
