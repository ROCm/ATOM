# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone gfx1250 K1 Q/KV normalization and KV RoPE/QAT stage.

The input is the BF16 [rows, 1792] wqkv_a result. Q uses the original
FP32 norm and round-to-nearest MXFP8 scale rule; KV rounds through BF16
before RoPE and again before ceil-scale FP8 QAT. One wave32 CTA per row
reconstructs the gfx950 dual norm's four wave64 partials in LDS.

Callers supply bounded, contiguous [rows, 32] BF16 cosine/sine rows and all
outputs. This compute stage does not publish mailboxes or index a KV ring;
the complete gfx1250 mono backend remains gated.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, check_e8m0, ptr_arg
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_norm_wqkv import _load_bf8
from atom.models.deepseek_v41.mono.kernels.mx import code_round, fp8_value
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2, rcp_pow2
from atom.mono.device.ops import bf_hi, bf_lo, bf16_round, div_rn, fp8_pack4, hw_rsq

Q_RANK = 1280
KV_DIM = 512
QKV_DIM = Q_RANK + KV_DIM
ROPE = 64
THREADS = 256
EPS = 1e-20


def _kernel_name(rows, debug):
    return f"atom_gfx1250_qkv_norm_rope_r{rows}_d{int(debug)}"


def _dual_total(partials):
    # Wave32 halves form each original wave64 partial. The original CTA
    # reduction uses xor 2, then xor 1 over its four wave64 partials.
    pairs = [partials[2 * i] + partials[2 * i + 1] for i in range(4)]
    return (pairs[0] + pairs[2]) + (pairs[1] + pairs[3])


@lru_cache(maxsize=4)
def _launcher(rows: int, debug: bool):
    if rows not in (1, 2):
        raise ValueError("gfx1250 qkv supports only one or two rows")

    @flyc.jit
    def launch(qkv: fx.Pointer, qnw: fx.Pointer, kvnw: fx.Pointer,
               cos: fx.Pointer, sin: fx.Pointer, qout: fx.Pointer,
               qsout: fx.Pointer, kvout: fx.Pointer, qnorm: fx.Pointer,
               knorm: fx.Pointer, krot: fx.Pointer, kfp8: fx.Pointer,
               kscale: fx.Pointer, stream: fx.Stream):
        @flyc.kernel(name=_kernel_name(rows, debug), known_block_size=[THREADS, 1, 1])
        def kernel(qkv: fx.Pointer, qnw: fx.Pointer, kvnw: fx.Pointer,
                   cos: fx.Pointer, sin: fx.Pointer, qout: fx.Pointer,
                   qsout: fx.Pointer, kvout: fx.Pointer, qnorm: fx.Pointer,
                   knorm: fx.Pointer, krot: fx.Pointer, kfp8: fx.Pointer,
                   kscale: fx.Pointer):
            tid = fx.Int32(fx.thread_idx.x)
            lane, wave = tid % 32, tid // 32
            row = fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(64)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            ld32, st32 = make_lds_copy_ops(32)
            rx = bo.create_buffer_resource_from_addr(fx.ptrtoint(qkv), num_records_bytes=rows * QKV_DIM * 2)
            rq = bo.create_buffer_resource_from_addr(fx.ptrtoint(qnw), num_records_bytes=Q_RANK * 2)
            rk = bo.create_buffer_resource_from_addr(fx.ptrtoint(kvnw), num_records_bytes=KV_DIM * 2)
            rc = bo.create_buffer_resource_from_addr(fx.ptrtoint(cos), num_records_bytes=rows * ROPE)
            rs = bo.create_buffer_resource_from_addr(fx.ptrtoint(sin), num_records_bytes=rows * ROPE)
            qo = bo.create_buffer_resource_from_addr(fx.ptrtoint(qout), num_records_bytes=rows * Q_RANK)
            qso = bo.create_buffer_resource_from_addr(fx.ptrtoint(qsout), num_records_bytes=rows * Q_RANK // 32)
            ko = bo.create_buffer_resource_from_addr(fx.ptrtoint(kvout), num_records_bytes=rows * KV_DIM * 2)
            if const_expr(debug):
                qno = bo.create_buffer_resource_from_addr(fx.ptrtoint(qnorm), num_records_bytes=rows * Q_RANK * 4)
                kno = bo.create_buffer_resource_from_addr(fx.ptrtoint(knorm), num_records_bytes=rows * KV_DIM * 2)
                kro = bo.create_buffer_resource_from_addr(fx.ptrtoint(krot), num_records_bytes=rows * KV_DIM * 2)
                kfo = bo.create_buffer_resource_from_addr(fx.ptrtoint(kfp8), num_records_bytes=rows * KV_DIM)
                kso = bo.create_buffer_resource_from_addr(fx.ptrtoint(kscale), num_records_bytes=rows * KV_DIM // 32)

            col = tid * 8
            qc = fx.min(col, Q_RANK - 8)
            qvals = _load_bf8(rx, row * QKV_DIM + qc)
            xq = [(col < Q_RANK).select(v, fx.Float32(0.0)) for v in qvals]
            qw = _load_bf8(rq, qc)
            kw = fx.Int32(bo.buffer_load(rk, tid, vec_width=1, dtype=T.i32))
            kv = fx.Int32(bo.buffer_load(rx, (row * QKV_DIM + Q_RANK) // 2 + tid, vec_width=1, dtype=T.i32))
            xk = [bf_lo(kv), bf_hi(kv)]
            weights = [bf_lo(kw), bf_hi(kw)]
            phase = tid - (KV_DIM - ROPE) // 2
            table_col = fx.max(phase, 0)
            table_index = row * (ROPE // 2) + table_col
            cw = fx.Int32(bo.buffer_load(rc, table_index // 2, vec_width=1, dtype=T.i32))
            sw = fx.Int32(bo.buffer_load(rs, table_index // 2, vec_width=1, dtype=T.i32))
            cosine = (phase >= 0).select((table_index % 2 == 0).select(bf_lo(cw), bf_hi(cw)), fx.Float32(1.0))
            sine = (phase >= 0).select((table_index % 2 == 0).select(bf_lo(sw), bf_hi(sw)), fx.Float32(0.0))

            sq = xq[0] * xq[0]
            for j in range_constexpr(1, 8):
                sq = sq + xq[j] * xq[j]
            sk = xk[0] * xk[0] + xk[1] * xk[1]
            for offset in (8, 4, 2, 1, 16):
                sq = sq + sq.shuffle_xor(offset, 32)
                sk = sk + sk.shuffle_xor(offset, 32)
            if lane == 31:
                st32(lds, wave * 4, Vec.from_elements([sq.bitcast(fx.Int32)]))
                st32(lds, 32 + wave * 4, Vec.from_elements([sk.bitcast(fx.Int32)]))
            fx.gpu.barrier()
            qp = [fx.Int32(ld32(lds, i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
            kp = [fx.Int32(ld32(lds, 32 + i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
            nq = hw_rsq(div_rn(_dual_total(qp), fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK)) + fx.Float32(EPS))
            nk = hw_rsq(div_rn(_dual_total(kp), fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM)) + fx.Float32(EPS))

            # Q deliberately stays FP32 until MXFP8 packing.
            ys = [(xq[j] * nq) * qw[j] for j in range(8)]
            amax = fx.Float32(0.0)
            for y in ys:
                amax = fx.max(amax, abs(y))
            for offset in (1, 2):
                amax = fx.max(amax, amax.shuffle_xor(offset, 32))
            code = code_round(amax)
            mul = pow2(254 - code)
            if col < Q_RANK:
                bo.buffer_store(fp8_pack4(*[ys[j] * mul for j in range(4)]), qo, (row * Q_RANK + col) // 4)
                bo.buffer_store(fp8_pack4(*[ys[j + 4] * mul for j in range(4)]), qo, (row * Q_RANK + col) // 4 + 1)
                if tid % 4 == 0:
                    bo.buffer_store(code.to(fx.Int8), qso, row * (Q_RANK // 32) + col // 32)
                if const_expr(debug):
                    for j in range_constexpr(8):
                        bo.buffer_store(ys[j], qno, row * Q_RANK + col + j)

            kn = [bf16_round((xk[j] * nk) * weights[j]) for j in range(2)]
            kr = [bf16_round(kn[0] * cosine - kn[1] * sine),
                  bf16_round(kn[0] * sine + kn[1] * cosine)]
            amax = fx.max(abs(kr[0]), abs(kr[1]))
            for offset in (1, 2, 4, 8):
                amax = fx.max(amax, amax.shuffle_xor(offset, 32))
            code = code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX))
            step, mul = pow2(code), rcp_pow2(code)
            scaled = [clamp_fp8(v * mul) for v in kr]
            qat = [fp8_value(v) * step for v in scaled]
            for j in range_constexpr(2):
                bo.buffer_store(qat[j].to(fx.BFloat16), ko, row * KV_DIM + tid * 2 + j)
            if const_expr(debug):
                for j in range_constexpr(2):
                    bo.buffer_store(kn[j].to(fx.BFloat16), kno, row * KV_DIM + tid * 2 + j)
                    bo.buffer_store(kr[j].to(fx.BFloat16), kro, row * KV_DIM + tid * 2 + j)
                neighbor = [v.shuffle_xor(1, 32) for v in scaled]
                packed = fp8_pack4(scaled[0], scaled[1], neighbor[0], neighbor[1])
                if tid % 2 == 0:
                    bo.buffer_store(packed, kfo, (row * KV_DIM + tid * 2) // 4)
                if tid % 16 == 0:
                    bo.buffer_store(code.to(fx.Int8), kso, row * (KV_DIM // 32) + tid // 16)

        kernel(qkv, qnw, kvnw, cos, sin, qout, qsout, kvout,
               qnorm, knorm, krot, kfp8, kscale).launch(
            grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def _arguments(qkv, qnw, kvnw, cos, sin, qout, qsout, kvout, debug):
    owner = "gfx1250 qkv norm/rope"
    rows = qkv.shape[0] if qkv.ndim == 2 else 0
    if rows not in (1, 2):
        raise ValueError(f"[{owner}] qkv must have one or two rows")
    specs = [(qkv, (rows, QKV_DIM), torch.bfloat16, "qkv"),
             (qnw, (Q_RANK,), torch.bfloat16, "qnw"),
             (kvnw, (KV_DIM,), torch.bfloat16, "kvnw"),
             (cos, (rows, ROPE // 2), torch.bfloat16, "cos"),
             (sin, (rows, ROPE // 2), torch.bfloat16, "sin"),
             (qout, (rows, Q_RANK), torch.float8_e4m3fn, "qout"),
             (kvout, (rows, KV_DIM), torch.bfloat16, "kvout")]
    check_e8m0(qsout, "qsout", owner, (rows, Q_RANK // 32))
    if debug is not None:
        if len(debug) != 5:
            raise ValueError(f"[{owner}] debug must contain qnorm, knorm, krot, kfp8, kscale")
        specs += [(debug[0], (rows, Q_RANK), torch.float32, "qnorm"),
                  (debug[1], (rows, KV_DIM), torch.bfloat16, "knorm"),
                  (debug[2], (rows, KV_DIM), torch.bfloat16, "krot"),
                  (debug[3], (rows, KV_DIM), torch.float8_e4m3fn, "kfp8")]
        check_e8m0(debug[4], "kscale", owner, (rows, KV_DIM // 32))
    for tensor, shape, dtype, name in specs:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"[{owner}] {name} must be contiguous {dtype} with shape {shape}")
    tensors = [qkv, qnw, kvnw, cos, sin, qout, qsout, kvout] + ([] if debug is None else list(debug))
    if not qkv.is_cuda or any(t.device != qkv.device for t in tensors):
        raise ValueError(f"[{owner}] all buffers must be on the same GPU")
    if torch.cuda.get_device_properties(qkv.device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError(f"[{owner}] requires gfx1250")
    for i in range(5, len(tensors)):
        a = tensors[i]
        lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        for b in tensors[:i]:
            if lo < b.data_ptr() + b.numel() * b.element_size() and b.data_ptr() < hi:
                raise ValueError(f"[{owner}] output buffers must not overlap")
    dbg = [flyc.from_c_void_p(fx.Uint8, 0)] * 5 if debug is None else [ptr_arg(t) for t in debug]
    return tuple(ptr_arg(t) for t in tensors[:8]) + tuple(dbg) + (torch.cuda.current_stream(qkv.device).cuda_stream,)


def qkv_norm_rope_gfx1250(qkv, qnw, kvnw, cos, sin, qout, qsout, kvout, *, debug=None):
    """Write the Q MXFP8 pair and BF16 KV QAT using caller-owned buffers.

    Cosine/sine rows must already be selected from the model's GPT-J RoPE
    table. Inputs and weights must be finite. The API allocates no GPU buffer
    and is HIP graph compatible; KV cache insertion remains the caller's.
    """
    args = _arguments(qkv, qnw, kvnw, cos, sin, qout, qsout, kvout, debug)
    _run_compiled(_launcher(qkv.shape[0], debug is not None), *args)
    return qout, qsout, kvout
