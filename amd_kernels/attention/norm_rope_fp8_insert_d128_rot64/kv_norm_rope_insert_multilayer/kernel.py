# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSpark draft context-KV tail, FlyDSL (gfx950): the fused context-KV GEMM's
bf16 output [C, L * 2 * kv] -> per-head Gemma RMS norm of K + partial NeoX RoPE
(rotary 64 of 128) + the e4m3 K/V insert into every layer's cache pages, in one
launch, instead of the permute / stack copies, the per-layer K norms, the
batched C++ RoPE and the per-layer Triton reshape-and-cache inserts.

Bit for bit the BASE chain: the K norm is the eager Triton
``gemma_rmsnorm._gemma_rmsnorm_kernel`` — its exact DPP reduction tree (same
lane map, so the same bits by construction) and its all-fp32 rstd pipeline (the
eager JIT binds the python-float ``eps`` as fp32: fp32 add, ``v_sqrt_f32``,
IEEE fp32 divide, fp32 muls, RNE bf16 store; NOT the fp64 pipeline of the
Inductor-compiled norm that ``../qkv_norm_rope_insert`` matches on the query
path). RoPE is the C++ ``vllm::rotary_embedding_kernel`` (bf16 cache values;
bf16 x cos/sin products are exact in fp32, so any fp-contraction order is
correctly rounded and all forms coincide). The insert divides by the
per-tensor scale (IEEE fp32) and converts with v_cvt_scalef32_pk_fp8_f32 under
MODE.fp16_ovfl, as ``triton_reshape_and_cache_flash``.

A wave is one (side, head) of one layer of one context row, lane l holding dims
l and 64 + l for K (the Triton norm's lane map), dims 2l, 2l+1 for V. K rows
occupy columns (l * 2) * nkv * D + h * D of the GEMM output, V the next
nkv * D.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from ..qkv_norm_rope_insert.kernel import (
    _bf16_f32,
    _bits,
    _e4m3x2,
    _ld16,
    _readlane_f,
    _rn_bf16,
    _set_fp16_ovfl,
    _swap32,
)

NW = 4  # waves per workgroup
D = 128
ROT = 64  # rotary dims (NeoX halves of 32)
TOK_B = 2 * D  # cache bytes per (token, KV head): K then V


def _dpp_row(v, ctrl, row_mask=0xF):
    """v_mov_b32_dpp of an fp32 with the given row mask (bound_ctrl)."""
    r = _rocdl_ops.update_dpp(
        T.i32,
        _llvm.mlir_undef(T.i32),
        fx.as_ir_value(_bits(v)),
        ctrl,
        row_mask,
        0xF,
        True,
    )
    return fx.Int32(r).bitcast(fx.Float32)


def _eager_sum_sq(x):
    """Sum of squares of the wave's 64 values in the eager Triton
    ``_gemma_rmsnorm_kernel``'s order (its ISA: mul, row_shr:8 fmac, row_shr
    4/2/1 adds, row_bcast:15 / row_bcast:31 tail, lane 63). The context-K norm
    runs the eager Triton kernel — NOT the Inductor one the qk-norm-rope
    kernel's ``_half_sum_sq`` recipe matches — so this replicates its exact DPP
    sequence (same lane map, hence the same bits by construction)."""
    a = fx.math.fma(x, x, _dpp_row(x * x, 0x118))  # row_shr:8
    for sh in (4, 2, 1):
        a = a + _dpp_row(a, 0x110 + sh)
    a = _dpp_row(a, 0x142, row_mask=0xA) + a  # row_bcast:15, rows 1 and 3
    a = a + _dpp_row(a, 0x143)  # row_bcast:31
    return _readlane_f(a, 63)


# kernel arguments preloaded into SGPRs (the first 64 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _global_i64_ptr(addr_i64):
    """Typed global i64 pointer at a raw device address (``p[i]`` loads)."""
    ptr_ty = fx.PointerType.get(
        T.i64, address_space=fx.AddressSpace.Global, alignment=8
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def compile_kv_norm_rope_insert_multilayer(
    nl: int, nkv: int, block_size: int, rope_form: int = 0, split_k: int = 0
):
    """``nl``: draft layers (their K/V rows share one GEMM output row).
    ``nkv``: KV heads per layer. ``rope_form`` 0: fma(y, cos, -+ x*sin) (the
    Inductor order); 1: fma(-+ y, sin, x*cos) — whichever the C++ rotary kernel
    compiles to. ``split_k`` > 0: the input is the GEMM's fp32 split-K
    slabs [split_k, n, row stride] (``gemm.mxfp8_common.SkPartials``) instead of its bf16
    output; each value is the split-K reduce's (0 + slab 0 + slab 1 + ... in fp32,
    RNE to bf16), so the bits equal reduce-then-insert."""
    SK = split_k
    S = nl * 2 * nkv  # waves per context row
    BS = block_size
    KV = nkv * D  # one side's width
    name = (
        f"kv_norm_rope_fp8_insert_d128_rot64_l{nl}_kv{nkv}_p{BS}"
        + ("_rf1" if rope_form else "")
        + (f"_sk{SK}" if SK else "")
    )

    def _slot(arg_slots, slot_stride, lay, i32_nsl, tok):
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slots + fx.Int64(lay * slot_stride) * 8,
            num_records_bytes=fx.Int64(i32_nsl) * 8,
        )
        return fx.Int32(
            buffer_ops.buffer_load(
                sl_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )

    def _scale(ptrs, lay, which):
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        return fx.Float32(fx.inttoptr(f32p, fx.Int64(ptrs[(1 + which) * nl + lay]))[0])

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,  # context rows
        i32_cs_rows: fx.Int32,
        arg_kv: fx.Int64,  # [n, row stride] bf16 GEMM output
        arg_pos: fx.Int64,  # positions [n] i64
        arg_k_norm_weight: fx.Int64,  # K norm weights [nl, 128] bf16
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, 64] bf16
        arg_ptrs: fx.Int64,  # i64: [nl caches][nl ks][nl vs]
        arg_slots: fx.Int64,
        slot_stride: fx.Int32,
        i64_page_b: fx.Int64,  # page stride (bytes)
        i32_head_b: fx.Int32,  # KV-head stride (bytes)
        i32_tok_b: fx.Int32,  # token stride in a page (bytes)
        i32_row_st: fx.Int32,  # kv row stride (elements)
        i32_eps: fx.Int32,  # fp32 bits
        i32_nsl: fx.Int32,  # slots given: rows >= i32_nsl are not inserted
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        tok = wid // S
        s = wid % S
        lay = s // (2 * nkv)
        r = s % (2 * nkv)
        is_k = r < nkv
        h = r % nkv
        live = tok < i32_n
        slab = i32_n * i32_row_st  # elements per split-K slab
        kv = buffer_ops.create_buffer_resource_from_addr(
            arg_kv,
            num_records_bytes=fx.Int64(slab) * (4 * SK if SK else 2),
        )

        def _kv16(off):
            """GEMM output element ``off`` as bf16 bits in the low half of an i32."""
            if const_expr(SK == 0):
                return _ld16(kv, off)
            acc = fx.Float32(0.0)
            for k in range_constexpr(SK):
                acc = acc + fx.Float32(
                    buffer_ops.buffer_load(kv, k * slab + off, vec_width=1, dtype=T.f32)
                )
            return _rn_bf16(acc)

        ptrs = _global_i64_ptr(arg_ptrs)
        if live:
            if is_k:
                w_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_k_norm_weight, num_records_bytes=nl * D * 2
                )
                pos_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
                )
                cs_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 2)
                )
                row0 = tok * i32_row_st + lay * 2 * KV + h * D
                x0 = _bf16_f32(_kv16(row0 + lane))
                x1 = _bf16_f32(_kv16(row0 + 64 + lane))
                w0 = _bf16_f32(_ld16(w_r, lay * D + lane))
                w1 = _bf16_f32(_ld16(w_r, lay * D + lane + 64))
                pos = fx.Int32(
                    buffer_ops.buffer_load(
                        pos_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True
                    )
                )
                pos = (pos < 0).select(pos + i32_cs_rows, pos)
                j = lane % 32
                cos = _bf16_f32(_ld16(cs_r, pos * ROT + j))
                sin = _bf16_f32(_ld16(cs_r, pos * ROT + 32 + j))

                total = _eager_sum_sq(x0) + _eager_sum_sq(x1)
                var = total / fx.Float32(float(D))
                vd = var + i32_eps.bitcast(fx.Float32)
                # the eager kernel's tl.sqrt lowers to v_sqrt_f32, which is NOT
                # correctly rounded on gfx950 (up to ~2 ulp low) — take the same
                # hardware sqrt, not llvm.sqrt.f32's CR expansion
                sq = fx.Float32(
                    _llvm.call_intrinsic(
                        T.f32, "llvm.amdgcn.sqrt.f32", [fx.as_ir_value(vd)], [], []
                    )
                )
                rstd = fx.Float32(1.0) / sq
                y0 = _rn_bf16((x0 * rstd) * (fx.Float32(1.0) + w0))
                y1 = _rn_bf16((x1 * rstd) * (fx.Float32(1.0) + w1))
                a, b = _swap32(y0)
                part = _bf16_f32((lane < 32).select(b, a))
                yf = _bf16_f32(y0)
                ps = part * sin
                if const_expr(rope_form == 0):
                    roped = fx.math.fma(yf, cos, (lane < 32).select(-ps, ps))
                else:
                    roped = fx.math.fma((lane < 32).select(-part, part), sin, yf * cos)
                r0 = _rn_bf16(roped)
                slot = _slot(arg_slots, slot_stride, lay, i32_nsl, tok)
                ks = _scale(ptrs, lay, 0)
                k0 = _bf16_f32(r0) / ks
                k1 = _bf16_f32(y1) / ks
                _set_fp16_ovfl()
                codes = _e4m3x2(k0, k1)
                rec = (
                    fx.Int64(ptrs[lay])
                    + fx.Int64(slot // BS) * i64_page_b
                    + fx.Int64(h * i32_head_b + (slot % BS) * i32_tok_b)
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
                row0 = tok * i32_row_st + (lay * 2 + 1) * KV + h * D
                if const_expr(SK == 0):
                    vw = fx.Int32(
                        buffer_ops.buffer_load(
                            kv, row0 // 2 + lane, vec_width=1, dtype=fx.Int32
                        )
                    )
                    f0 = (vw << 16).bitcast(fx.Float32)
                    f1 = (vw & -65536).bitcast(fx.Float32)
                else:
                    f0 = _bf16_f32(_kv16(row0 + 2 * lane))
                    f1 = _bf16_f32(_kv16(row0 + 2 * lane + 1))
                slot = _slot(arg_slots, slot_stride, lay, i32_nsl, tok)
                vs = _scale(ptrs, lay, 1)
                v0 = f0 / vs
                v1 = f1 / vs
                _set_fp16_ovfl()
                codes = _e4m3x2(v0, v1)
                rec = (
                    fx.Int64(ptrs[lay])
                    + fx.Int64(slot // BS) * i64_page_b
                    + fx.Int64(h * i32_head_b + (slot % BS) * i32_tok_b)
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
        arg_kv: fx.Int64,
        arg_pos: fx.Int64,
        arg_k_norm_weight: fx.Int64,
        arg_cs: fx.Int64,
        arg_ptrs: fx.Int64,
        arg_slots: fx.Int64,
        slot_stride: fx.Int32,
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        i64_page_b: fx.Int64,
        i32_head_b: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_row_st: fx.Int32,
        i32_eps: fx.Int32,
        i32_nsl: fx.Int32,
        stream: fx.Stream,
    ):
        waves = i32_n * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_kv,
            arg_pos,
            arg_k_norm_weight,
            arg_cs,
            arg_ptrs,
            arg_slots,
            slot_stride,
            i64_page_b,
            i32_head_b,
            i32_tok_b,
            i32_row_st,
            i32_eps,
            i32_nsl,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
