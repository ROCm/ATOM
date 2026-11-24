# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSpark draft attention input, FlyDSL (gfx950): per-head Gemma RMS norm of q and k,
partial NeoX RoPE (rotary_dim 64 of 128), and the e4m3 K / V insert into the
draft's cache pages, in one launch. Bit for bit the draft's compiled path: the
Triton ``_gemma_rmsnorm_kernel`` (q and k), the Inductor RoPE and
``triton_reshape_and_cache_flash``.

A wave is one head of one token, lane l holding dims l and 64 + l: the Triton norm's
lane map (its waves 0 and 1 hold dims 0..63 and 64..127), so each half keeps its
sum of squares order: lanes 8..15 of a row fma their square onto lane l - 8's,
row_shr 4 / 2 / 1 adds, (row 0 + row 1) + (row 2 + row 3), then half 0 + half 1.
var = sum / 128 in fp32; then, as Inductor compiles the kernel (eps an fp64 argument),
fp64: rstd = 1 / sqrt(var + eps), y = RN_bf16((rstd x) (1 + w)), all correctly
rounded. RoPE on the bf16 y: fma(y, cos, -/+ y' sin) (y' = dim l ^ 32; a product of
two bf16 is exact in fp32, so any contraction gives these bits), rounded to bf16.
The insert divides by the per-tensor scale (correctly rounded) and converts with
v_cvt_scalef32_pk_fp8_f32 under MODE.fp16_ovfl, as Triton does.

Waves of a token: [q heads | k heads | v heads] (no v waves without the insert).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

NW = 4  # waves per workgroup
D = 128
ROT = 64  # rotary dims (NeoX halves of 32)
TOK_B = 2 * D  # cache bytes per (token, KV head): K then V
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
# kernel arguments preloaded into SGPRs (the first 64 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bf16_f32(h):
    """bf16 bits in the low half of an i32 -> fp32 (exact)."""
    return (fx.Int32(h) << 16).bitcast(fx.Float32)


def _rn_bf16(x):
    """fp32 -> RNE bf16 (v_cvt_pk_bf16_f32) as its low-half i32 bits."""
    return (
        fx.Int32(
            fx.rocdl.cvt_pk_bf16_f32(
                fx.as_ir_value(fx.Float32(x)), fx.as_ir_value(fx.Float32(0.0))
            )
        )
        & 0xFFFF
    )


def _rn_bf16_f64(d):
    """fp64 -> RNE bf16 (fptrunc: round to odd fp32, then v_cvt_pk_bf16_f32) as
    low-half i32 bits."""
    return fx.Float64(d).to(fx.BFloat16).bitcast(fx.Int16).to(fx.Int32) & 0xFFFF


def _dpp_f(v, ctrl):
    """v_mov_b32_dpp of an fp32; source lanes outside the row read 0 (bound_ctrl)."""
    r = _rocdl_ops.update_dpp(
        T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(_bits(v)), ctrl, 0xF, 0xF, True
    )
    return fx.Int32(r).bitcast(fx.Float32)


def _readlane_f(v, lane):
    return fx.Int32(fx.rocdl.readlane(T.i32, _bits(v), fx.Int32(lane))).bitcast(
        fx.Float32
    )


def _half_sum_sq(x):
    """Sum of squares of the wave's 64 values in the Triton reduction order."""
    a = fx.math.fma(x, x, _dpp_f(x * x, 0x118))  # row_shr:8
    for sh in (4, 2, 1):
        a = a + _dpp_f(a, 0x110 + sh)
    r = [_readlane_f(a, 16 * i + 15) for i in range_constexpr(4)]
    return (r[0] + r[1]) + (r[2] + r[3])


def _swap32(v):
    """Lane l ^ 32's i32 (v_permlane32_swap of ``v`` with itself)."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl_ops.permlane32_swap(
        pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False
    )
    a = fx.Int32(_llvm.extractvalue(T.i32, res, [0]))
    b = fx.Int32(_llvm.extractvalue(T.i32, res, [1]))
    return a, b


def _set_fp16_ovfl():
    """MODE.fp16_ovfl = 1, which Triton sets before its fp8 conversions."""
    _llvm.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()],
        [],
        [],
    )


def _e4m3x2(a, b):
    """Two fp32 -> e4m3 bytes 0 (a) and 1 (b) of an i32 (v_cvt_scalef32_pk_fp8_f32,
    scale 1)."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(fx.Int32(0))),
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(False).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _ld16(rsrc, off):
    """One bf16 (element offset ``off``) as the low half of an i32."""
    v = buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=T.i16)
    return fx.Int16(v).to(fx.Int32)


def compile_qkv_norm_rope_insert(nq: int, nkv: int, block_size: int, insert: bool):
    """``nq`` / ``nkv``: query / KV heads of a qkv row ([q | k | v] heads of 128).
    ``insert``: write K / V into the cache pages (``block_size`` tokens a page)."""
    S = nq + (2 * nkv if insert else nkv)  # waves per token
    BS = block_size
    name = f"qkv_norm_rope_fp8_insert_d128_rot64_q{nq}_kv{nkv}_p{BS}" + (
        "" if insert else "_noinsert"
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_qkv: fx.Int64,  # [n, row stride] bf16, rows of ROW
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # slots [n] i64, < 0 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, 64] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16
        arg_kw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, nq * 128] bf16
        arg_kv: fx.Int64,  # pages [blocks, nkv, BS, 256] e4m3
        arg_ks: fx.Int64,  # k / v scales, fp32 scalars
        arg_vs: fx.Int64,
        i64_page_b: fx.Int64,  # page stride (bytes)
        i32_head_b: fx.Int32,  # KV-head stride (bytes)
        i32_tok_b: fx.Int32,  # token stride in a page (bytes)
        i32_row_st: fx.Int32,  # qkv row stride (elements)
        i64_eps: fx.Int64,  # fp64 bits
        i32_nsl: fx.Int32,  # slots given: tokens >= i32_nsl are not inserted
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        tok = wid // S
        s = wid % S
        live = tok < i32_n
        qkv = buffer_ops.create_buffer_resource_from_addr(
            arg_qkv, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_row_st) * 2
        )
        row0 = tok * i32_row_st
        if live:
            if s < nq + nkv:
                is_q = s < nq
                w_r = buffer_ops.create_buffer_resource_from_addr(
                    is_q.select(arg_qw, arg_kw), num_records_bytes=D * 2
                )
                pos_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
                )
                cs_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 2)
                )
                x0 = _bf16_f32(_ld16(qkv, row0 + s * D + lane))
                x1 = _bf16_f32(_ld16(qkv, row0 + s * D + 64 + lane))
                w0 = _bf16_f32(_ld16(w_r, lane))
                w1 = _bf16_f32(_ld16(w_r, lane + 64))
                pos = fx.Int32(
                    buffer_ops.buffer_load(
                        pos_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True
                    )
                )
                pos = (pos < 0).select(pos + i32_cs_rows, pos)
                j = lane % 32
                cos = _bf16_f32(_ld16(cs_r, pos * ROT + j))
                sin = _bf16_f32(_ld16(cs_r, pos * ROT + 32 + j))

                total = _half_sum_sq(x0) + _half_sum_sq(x1)
                var = total / fx.Float32(float(D))
                vd = var.to(fx.Float64) + i64_eps.bitcast(fx.Float64)
                sq = fx.Float64(
                    _llvm.call_intrinsic(
                        T.f64, "llvm.sqrt.f64", [fx.as_ir_value(vd)], [], []
                    )
                )
                rstd = fx.Float64(1.0) / sq
                y0 = _rn_bf16_f64(
                    (rstd * x0.to(fx.Float64)) * (fx.Float32(1.0) + w0).to(fx.Float64)
                )
                y1 = _rn_bf16_f64(
                    (rstd * x1.to(fx.Float64)) * (fx.Float32(1.0) + w1).to(fx.Float64)
                )
                a, b = _swap32(y0)
                part = _bf16_f32((lane < 32).select(b, a))
                yf = _bf16_f32(y0)
                ps = part * sin
                roped = fx.math.fma(yf, cos, (lane < 32).select(-ps, ps))
                r0 = _rn_bf16(roped)
                if is_q:
                    q_r = buffer_ops.create_buffer_resource_from_addr(
                        arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
                    )
                    qo = tok * (nq * D) + s * D + lane
                    buffer_ops.buffer_store(r0.to(fx.Int16), q_r, qo)
                    buffer_ops.buffer_store(y1.to(fx.Int16), q_r, qo + 64)
                if const_expr(insert):  # noqa: SIM102 (compile-time, then runtime)
                    if s >= nq:
                        sl_r = buffer_ops.create_buffer_resource_from_addr(
                            arg_slot, num_records_bytes=fx.Int64(i32_nsl) * 8
                        )
                        slot = fx.Int32(
                            buffer_ops.buffer_load(
                                sl_r,
                                tok * 2,
                                vec_width=1,
                                dtype=fx.Int32,
                                is_scalar=True,
                            )
                        )
                        f32p = fx.PointerType.get(
                            T.f32, address_space=fx.AddressSpace.Global, alignment=4
                        )
                        ks = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
                        k0 = _bf16_f32(r0) / ks
                        k1 = _bf16_f32(y1) / ks
                        _set_fp16_ovfl()
                        codes = _e4m3x2(k0, k1)
                        rec = (
                            arg_kv
                            + fx.Int64(slot // BS) * i64_page_b
                            + fx.Int64((s - nq) * i32_head_b + (slot % BS) * i32_tok_b)
                        )
                        kv_r = buffer_ops.create_buffer_resource_from_addr(
                            rec, num_records_bytes=TOK_B
                        )
                        ok = (slot >= 0) & (tok < i32_nsl)
                        buffer_ops.buffer_store(
                            (codes & 0xFF).to(fx.Int8),
                            kv_r,
                            lane,
                            mask=ok,
                            offset_is_bytes=True,
                        )
                        buffer_ops.buffer_store(
                            ((codes >> 8) & 0xFF).to(fx.Int8),
                            kv_r,
                            lane + 64,
                            mask=ok,
                            offset_is_bytes=True,
                        )
            else:
                g = s - (nq + nkv)
                sl_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_slot, num_records_bytes=fx.Int64(i32_nsl) * 8
                )
                slot = fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True
                    )
                )
                vw = fx.Int32(
                    buffer_ops.buffer_load(
                        qkv,
                        (row0 + (nq + nkv + g) * D) // 2 + lane,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                f32p = fx.PointerType.get(
                    T.f32, address_space=fx.AddressSpace.Global, alignment=4
                )
                vs = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
                v0 = (vw << 16).bitcast(fx.Float32) / vs
                v1 = (vw & -65536).bitcast(fx.Float32) / vs
                _set_fp16_ovfl()
                codes = _e4m3x2(v0, v1)
                rec = (
                    arg_kv
                    + fx.Int64(slot // BS) * i64_page_b
                    + fx.Int64(g * i32_head_b + (slot % BS) * i32_tok_b)
                )
                kv_r = buffer_ops.create_buffer_resource_from_addr(
                    rec, num_records_bytes=TOK_B
                )
                buffer_ops.buffer_store(
                    (codes & 0xFFFF).to(fx.Int16),
                    kv_r,
                    D + 2 * lane,
                    mask=(slot >= 0) & (tok < i32_nsl),
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(
        arg_qkv: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_cs: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        i64_page_b: fx.Int64,
        i32_head_b: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_row_st: fx.Int32,
        i64_eps: fx.Int64,
        i32_nsl: fx.Int32,
        stream: fx.Stream,
    ):
        waves = i32_n * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_qkv,
            arg_pos,
            arg_slot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_q,
            arg_kv,
            arg_ks,
            arg_vs,
            i64_page_b,
            i32_head_b,
            i32_tok_b,
            i32_row_st,
            i64_eps,
            i32_nsl,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
