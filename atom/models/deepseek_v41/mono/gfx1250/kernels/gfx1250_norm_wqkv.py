# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 RMSNorm -> group32 MXFP8 -> Flash wqkv_a, for one or two rows.

This standalone migration stage is not registered as a complete mono backend.
Each CTA quantizes its own activation in LDS and produces 16 output features.
Eight wave32 groups split K and reduce partials in LDS, following gfx950's
strided K steps and wave-order reduction. There is no inter-CTA or peer
synchronization; repeated normalization remains a migration-stage tradeoff.

Weights use AITER's gfx1250 native 16-row preshuffle and raw [N/32, K/32]
E8M0 block scales. Inputs and norm weights must be finite BF16 values. The
normalization follows K1's 256-thread reduction tree, epsilon 1e-20 and BF16
rounding before quantization. Padded WMMA rows read a valid row and never store.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, check_e8m0, ptr_arg
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import bf_hi, bf_lo, bf16_round, fp8_pack4, hw_rsq

HIDDEN = 5120
OUTPUT = 1792
TILE_N = 16
THREADS = 256
K_SPLIT = 8


def _weighted_bf16(value, weight):
    # The original gfx1250 stage fuses FP32 * BF16 into BF16 directly.
    # An intermediate FP32 product can double-round a BF16 halfway case.
    # The BF16 coefficient occupies weight's high 16 bits as an FP32 value.
    bits = fx.Int32(
        llvm.inline_asm(
            T.i32,
            [value.ir_value(), weight.ir_value()],
            "v_fma_mixlo_bf16 $0, $1, $2, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]",
            "=v,v,v",
            has_side_effects=False,
        )
    )
    return bf_lo(bits)


def _kernel_name(rows, debug):
    return f"atom_gfx1250_norm_wqkv_r{rows}_d{int(debug)}_k8"


def _load_bf8(rsrc, base):
    """Eight BF16 elements in one aligned 16-byte load; base is a BF16 index."""
    words = Vec(bo.buffer_load(rsrc, base // 2, vec_width=4, dtype=T.i32))
    return [unpack(words[i]) for i in range(4) for unpack in (bf_lo, bf_hi)]


@lru_cache(maxsize=4)
def _launcher(rows: int, debug: bool):
    if rows not in (1, 2):
        raise ValueError("gfx1250 norm_wqkv supports only one or two rows")
    sa_off = rows * HIDDEN
    red_off = sa_off + rows * (HIDDEN // 32)
    partial_off = red_off + 32
    lds_bytes = partial_off + K_SPLIT * rows * TILE_N * 4

    @flyc.jit
    def launch(x: fx.Pointer, norm_w: fx.Pointer, weight: fx.Pointer,
               scale: fx.Pointer, out: fx.Pointer, normed: fx.Pointer,
               quantized: fx.Pointer, codes: fx.Pointer, stream: fx.Stream):
        @flyc.kernel(name=_kernel_name(rows, debug),
                     known_block_size=[THREADS, 1, 1])
        def kernel(x: fx.Pointer, norm_w: fx.Pointer, weight: fx.Pointer,
                   scale: fx.Pointer, out: fx.Pointer, normed: fx.Pointer,
                   quantized: fx.Pointer, codes: fx.Pointer):
            tid = fx.Int32(fx.thread_idx.x)
            lane = tid % 32
            wave = tid // 32
            bid = fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(lds_bytes)
            base = arena.base_ptr
            lds = fx.Int64(fx.ptrtoint(base))
            ld32, st32 = make_lds_copy_ops(32)
            ld128, st128 = make_lds_copy_ops(128)
            rx = bo.create_buffer_resource_from_addr(fx.ptrtoint(x), num_records_bytes=rows * HIDDEN * 2)
            rn = bo.create_buffer_resource_from_addr(fx.ptrtoint(norm_w), num_records_bytes=HIDDEN * 2)
            rw = bo.create_buffer_resource_from_addr(fx.ptrtoint(weight), num_records_bytes=OUTPUT * HIDDEN)
            rs = bo.create_buffer_resource_from_addr(fx.ptrtoint(scale), num_records_bytes=OUTPUT * HIDDEN // 1024)
            ro = bo.create_buffer_resource_from_addr(fx.ptrtoint(out), num_records_bytes=rows * OUTPUT * 2)
            if const_expr(debug):
                rd = bo.create_buffer_resource_from_addr(fx.ptrtoint(normed), num_records_bytes=rows * HIDDEN * 2)
                rq = bo.create_buffer_resource_from_addr(fx.ptrtoint(quantized), num_records_bytes=rows * HIDDEN)
                rc = bo.create_buffer_resource_from_addr(fx.ptrtoint(codes), num_records_bytes=rows * HIDDEN // 32)

            # Preserve K1's element ownership and wave64 tree with eight wave32
            # partials paired in LDS. Invalid chunks use legal clamped loads.
            cols = [tid * 8 + 2048 * ch for ch in range(3)]
            wts = [_load_bf8(rn, fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
            for row in range_constexpr(rows):
                loaded = [_load_bf8(rx, row * HIDDEN + fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
                xs = [[(cols[ch] < HIDDEN).select(v, fx.Float32(0.0)) for v in loaded[ch]] for ch in range(3)]
                acc = fx.Float32(0.0)
                for ch in range_constexpr(3):
                    for j in range_constexpr(8):
                        acc = acc + xs[ch][j] * xs[ch][j]
                for off in range_constexpr(5):
                    acc = acc + acc.shuffle_xor(1 << off, 32)
                if lane == 0:
                    st32(lds, red_off + wave * 4, Vec.from_elements([acc.bitcast(fx.Int32)]))
                fx.gpu.barrier()
                partials = [fx.Int32(ld32(lds, red_off + i * 4)[0]).bitcast(fx.Float32)
                            for i in range(8)]
                pairs = [partials[2 * i] + partials[2 * i + 1] for i in range(4)]
                total = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3])
                rstd = hw_rsq(fx.fma(total, fx.Float32(1.0 / HIDDEN), fx.Float32(1e-20)))
                for ch in range_constexpr(3):
                    ys = [_weighted_bf16(xs[ch][j] * rstd, wts[ch][j]) for j in range(8)]
                    amax = fx.Float32(0.0)
                    for j in range_constexpr(8):
                        amax = fx.max(amax, abs(ys[j]))
                    amax = fx.max(amax, amax.shuffle_xor(1, 32))
                    amax = fx.max(amax, amax.shuffle_xor(2, 32))
                    code = code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX))
                    # E8M0 is a power of two; this reciprocal multiplication
                    # preserves the FP8 result without a full IEEE division.
                    mul = rcp_pow2(code)
                    if cols[ch] < HIDDEN:
                        q0 = fp8_pack4(*[clamp_fp8(ys[j] * mul) for j in range(4)])
                        q1 = fp8_pack4(*[clamp_fp8(ys[j + 4] * mul) for j in range(4)])
                        st32(lds, row * HIDDEN + cols[ch], Vec.from_elements([q0]))
                        st32(lds, row * HIDDEN + cols[ch] + 4, Vec.from_elements([q1]))
                        if tid % 4 == 0:
                            fx.ptr_store(code.to(fx.Int8), base + sa_off + row * (HIDDEN // 32) + cols[ch] // 32)
                        if const_expr(debug):
                            if bid == 0:
                                for j in range_constexpr(8):
                                    bo.buffer_store(ys[j].to(fx.BFloat16), rd, row * HIDDEN + cols[ch] + j)
                                bo.buffer_store(q0, rq, (row * HIDDEN + cols[ch]) // 4)
                                bo.buffer_store(q1, rq, (row * HIDDEN + cols[ch]) // 4 + 1)
                                if tid % 4 == 0:
                                    bo.buffer_store(code.to(fx.Int8), rc, row * (HIDDEN // 32) + cols[ch] // 32)
                # Protect the reused reduction slots as well as activation LDS.
                fx.gpu.barrier()

            if wave < K_SPLIT:
                n0 = bid * TILE_N
                lane16, half = lane % 16, lane // 16
                active_row = fx.min(lane16, rows - 1)
                atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
                    16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32))
                c = fx.make_rmem_tensor(8, fx.Float32)
                c.store(fx.full(8, 0.0, fx.Float32))
                # gfx950 GEMV assigns steps w, w+8, ... to wave w and issues
                # every weight load before consuming the activation operands.
                # A gfx1250 operand is 16 dwords/lane, rather than gfx950's 8.
                steps = [wave + K_SPLIT * i for i in range(HIDDEN // 128 // K_SPLIT)]
                weights = []
                scales = []
                for st in steps:
                    bv = [Vec(bo.buffer_load(rw, (n0 // 16 * HIDDEN * 16 + st * 2048 + half * 256
                                                  + lane16 * 16 + j * 512) // 4,
                                            vec_width=4, dtype=T.i32)) for j in range(4)]
                    weights.append(bv[0].shuffle(bv[1], list(range(8))).shuffle(
                        bv[2].shuffle(bv[3], list(range(8))), list(range(16))))
                    scales.append(bo.buffer_load(rs, (n0 // 32) * (HIDDEN // 128) + st,
                                                  vec_width=1, dtype=T.i32))
                # Keep the five weight groups in flight before consuming them.
                fx.rocdl.sched_barrier(0)
                for i in range_constexpr(HIDDEN // 128 // K_SPLIT):
                    st = steps[i]
                    av = [Vec(ld128(lds, active_row * HIDDEN + st * 128 + half * 16 + j * 32))
                          for j in range_constexpr(4)]
                    avec = av[0].shuffle(av[1], list(range(8))).shuffle(
                        av[2].shuffle(av[3], list(range(8))), list(range(16)))
                    a = fx.make_rmem_tensor(16, fx.Int32)
                    b = fx.make_rmem_tensor(16, fx.Int32)
                    a.store(avec)
                    b.store(weights[i])
                    sa = ld32(lds, sa_off + active_row * (HIDDEN // 32) + st * 4)[0]
                    # AITER WMMA orientation: weight is A, activation is B.
                    fx.gemm(atom, c, b, a, c, scale_a=scales[i], scale_b=sa)
                if lane16 < rows:
                    values = c.load()
                    offset = partial_off + (wave * rows * TILE_N + lane16 * TILE_N + half * 8) * 4
                    for j in range_constexpr(2):
                        st128(lds, offset + j * 16, Vec.from_elements(
                            [values[j * 4 + i].bitcast(fx.Int32) for i in range(4)]))
            fx.gpu.barrier()
            if tid < rows * TILE_N:
                total = fx.Float32(0.0)
                for w in range_constexpr(K_SPLIT):
                    part = fx.Int32(ld32(lds, partial_off + (w * rows * TILE_N + tid) * 4)[0]).bitcast(fx.Float32)
                    total = total + part
                bo.buffer_store(total.to(fx.BFloat16), ro,
                                (tid // TILE_N) * OUTPUT + bid * TILE_N + tid % TILE_N)

        kernel(x, norm_w, weight, scale, out, normed, quantized, codes).launch(
            grid=(OUTPUT // TILE_N, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def _arguments(x, norm_weight, weight, weight_scale, out, debug):
    owner = "gfx1250 norm_wqkv"
    rows = x.shape[0] if x.ndim == 2 else 0
    specs = [(x, (rows, HIDDEN), torch.bfloat16, "x"),
             (norm_weight, (HIDDEN,), torch.bfloat16, "norm_weight"),
             (weight, (OUTPUT, HIDDEN), torch.float8_e4m3fn, "weight"),
             (out, (rows, OUTPUT), torch.bfloat16, "out")]
    if rows not in (1, 2):
        raise ValueError(f"[{owner}] x must have one or two rows")
    check_e8m0(weight_scale, "weight_scale", owner, (OUTPUT // 32, HIDDEN // 32))
    if debug is not None:
        if len(debug) != 3:
            raise ValueError(f"[{owner}] debug must contain normed, quantized, codes")
        specs += [(debug[0], (rows, HIDDEN), torch.bfloat16, "debug_normed"),
                  (debug[1], (rows, HIDDEN), torch.float8_e4m3fn, "debug_quantized")]
        check_e8m0(debug[2], "debug_codes", owner, (rows, HIDDEN // 32))
    for t, shape, dtype, name in specs:
        if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"[{owner}] {name} must be contiguous {dtype} with shape {shape}")
    tensors = [x, norm_weight, weight, weight_scale, out] + ([] if debug is None else list(debug))
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError(f"[{owner}] all buffers must be on the same GPU")
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError(f"[{owner}] requires gfx1250")
    # Written buffers must not alias inputs or each other. Read-only aliases
    # are legal. Check intervals on the host without device synchronization.
    for i in range(4, len(tensors)):
        a = tensors[i]
        a0, a1 = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        for b in tensors[:i]:
            b0, b1 = b.data_ptr(), b.data_ptr() + b.numel() * b.element_size()
            if a0 < b1 and b0 < a1:
                raise ValueError(f"[{owner}] output buffers must not overlap")
    dbg = [flyc.from_c_void_p(fx.Uint8, 0)] * 3 if debug is None else [ptr_arg(t) for t in debug]
    return tuple(ptr_arg(t) for t in tensors[:5]) + tuple(dbg) + (torch.cuda.current_stream(x.device).cuda_stream,)


def norm_wqkv_gfx1250(x, norm_weight, weight, weight_scale, out, *, debug=None):
    """Write BF16 [rows,1792] using caller-owned buffers (HIP graph compatible).

    ``debug=(normed_bf16, quantized_fp8, scale_codes)`` optionally records the
    intermediate tensors. ``weight`` must already be natively preshuffled;
    shape validation cannot distinguish a shuffled matrix from a row-major one.
    """
    args = _arguments(x, norm_weight, weight, weight_scale, out, debug)
    _run_compiled(_launcher(x.shape[0], debug is not None), *args)
    return out
