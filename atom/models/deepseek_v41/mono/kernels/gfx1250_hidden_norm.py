# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K1 hidden norm/MXFP8 producer, one CTA per token.

Preserves stage_norm's H5120 reduction, BF16 rounding and ceil-scale group32
quantization. Caller-owned outputs may be shared by all projection tasks.
This standalone producer establishes the buffer contract; publishing tagged
mailboxes within a persistent K1 still needs independent progress validation.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, check_e8m0, ptr_arg
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from atom.models.deepseek_v41.mono.kernels.gfx1250_norm_wqkv import HIDDEN, _load_bf8
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import bf_lo, fp8_pack4, hw_rsq

THREADS = 256


def _weighted_bf16(value, weight):
    # The original gfx1250 stage fuses FP32 * BF16 into BF16 directly.
    # An intermediate FP32 product can double-round a BF16 halfway case.
    # The BF16 coefficient occupies weight's high 16 bits as an FP32 value.
    bits = fx.Int32(llvm.inline_asm(
        T.i32, [value.ir_value(), weight.ir_value()],
        "v_fma_mixlo_bf16 $0, $1, $2, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]",
        "=v,v,v", has_side_effects=False))
    return bf_lo(bits)


def _kernel_name(rows):
    return f"atom_gfx1250_hidden_norm_producer_r{rows}"


@lru_cache(maxsize=2)
def _launcher(rows):
    if rows not in (1, 2):
        raise ValueError("gfx1250 hidden norm supports rows1/2")

    @flyc.jit
    def launch(x: fx.Pointer, norm_weight: fx.Pointer, normed: fx.Pointer,
               quantized: fx.Pointer, codes: fx.Pointer, stream: fx.Stream):
        @flyc.kernel(name=_kernel_name(rows), known_block_size=[THREADS, 1, 1])
        def kernel(x: fx.Pointer, norm_weight: fx.Pointer, normed: fx.Pointer,
                   quantized: fx.Pointer, codes: fx.Pointer):
            tid = fx.Int32(fx.thread_idx.x)
            lane, wave = tid % 32, tid // 32
            row = fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(32)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            ld32, st32 = make_lds_copy_ops(32)
            rx = bo.create_buffer_resource_from_addr(fx.ptrtoint(x), num_records_bytes=rows * HIDDEN * 2)
            rw = bo.create_buffer_resource_from_addr(fx.ptrtoint(norm_weight), num_records_bytes=HIDDEN * 2)
            rn = bo.create_buffer_resource_from_addr(fx.ptrtoint(normed), num_records_bytes=rows * HIDDEN * 2)
            rq = bo.create_buffer_resource_from_addr(fx.ptrtoint(quantized), num_records_bytes=rows * HIDDEN)
            rs = bo.create_buffer_resource_from_addr(fx.ptrtoint(codes), num_records_bytes=rows * HIDDEN // 32)
            cols = [tid * 8 + 2048 * ch for ch in range(3)]
            weights = [_load_bf8(rw, fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
            loaded = [_load_bf8(rx, row * HIDDEN + fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
            xs = [[(cols[ch] < HIDDEN).select(v, fx.Float32(0.0)) for v in loaded[ch]] for ch in range(3)]
            accum = fx.Float32(0.0)
            for ch in range_constexpr(3):
                for j in range_constexpr(8):
                    accum = accum + xs[ch][j] * xs[ch][j]
            for off in range_constexpr(5):
                accum = accum + accum.shuffle_xor(1 << off, 32)
            if lane == 0:
                st32(lds, wave * 4, Vec.from_elements([accum.bitcast(fx.Int32)]))
            fx.gpu.barrier()
            partials = [fx.Int32(ld32(lds, i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
            pairs = [partials[2 * i] + partials[2 * i + 1] for i in range(4)]
            total = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3])
            rstd = hw_rsq(fx.fma(total, fx.Float32(1.0 / HIDDEN), fx.Float32(1e-20)))
            for ch in range_constexpr(3):
                ys = [_weighted_bf16(xs[ch][j] * rstd, weights[ch][j]) for j in range(8)]
                amax = fx.Float32(0.0)
                for y in ys:
                    amax = fx.max(amax, abs(y))
                amax = fx.max(amax, amax.shuffle_xor(1, 32))
                amax = fx.max(amax, amax.shuffle_xor(2, 32))
                code = code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX))
                mul = rcp_pow2(code)
                if cols[ch] < HIDDEN:
                    col = row * HIDDEN + cols[ch]
                    for j in range_constexpr(8):
                        bo.buffer_store(ys[j].to(fx.BFloat16), rn, col + j)
                    for j in range_constexpr(2):
                        bo.buffer_store(fp8_pack4(*[clamp_fp8(ys[4 * j + k] * mul) for k in range(4)]),
                                        rq, col // 4 + j)
                    if tid % 4 == 0:
                        bo.buffer_store(code.to(fx.Int8), rs, row * (HIDDEN // 32) + cols[ch] // 32)

        kernel(x, norm_weight, normed, quantized, codes).launch(
            grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def _arguments(x, norm_weight, normed, quantized, codes):
    owner = "gfx1250 hidden norm producer"
    rows = x.shape[0] if x.ndim == 2 else 0
    if rows not in (1, 2):
        raise ValueError(f"[{owner}] supports rows1/2")
    specs = [(x, (rows, HIDDEN), torch.bfloat16, "x"),
             (norm_weight, (HIDDEN,), torch.bfloat16, "norm_weight"),
             (normed, (rows, HIDDEN), torch.bfloat16, "normed"),
             (quantized, (rows, HIDDEN), torch.float8_e4m3fn, "quantized")]
    check_e8m0(codes, "codes", owner, (rows, HIDDEN // 32))
    for tensor, shape, dtype, name in specs:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"[{owner}] {name} must be contiguous {dtype} with shape {shape}")
    tensors = [x, norm_weight, normed, quantized, codes]
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError(f"[{owner}] all buffers must be on the same GPU")
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError(f"[{owner}] requires gfx1250")
    for i in range(2, len(tensors)):
        a = tensors[i]
        lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        for b in tensors[:i]:
            if lo < b.data_ptr() + b.numel() * b.element_size() and b.data_ptr() < hi:
                raise ValueError(f"[{owner}] output buffers must not overlap")
    return tuple(ptr_arg(t) for t in tensors) + (torch.cuda.current_stream(x.device).cuda_stream,)


def hidden_norm_quant_gfx1250(x, norm_weight, normed, quantized, codes):
    """Produce shared BF16 norm, FP8 activation and group32 E8M0 codes.

    Inputs and norm weights must be finite BF16. Caller supplies all outputs;
    no GPU buffer is allocated, and the producer supports HIP graph replay.
    """
    args = _arguments(x, norm_weight, normed, quantized, codes)
    _run_compiled(_launcher(x.shape[0]), *args)
    return normed, quantized, codes
