# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K1 wq_b GEMV with BF16 rounding and query-head RoPE.

Consumes the Q MXFP8 pair from gfx1250_qkv. Eight wave32 groups take K128
steps w, w+8, ... and reduce FP32 partials in wave order, as gfx950 K1 does.
The last wave round is ragged (K1280 has ten steps); all loads stay in the
last valid step and inactive operands use zeros with unit E8M0 scales.

Weights use AITER's native 16-row preshuffle and raw 32x32 E8M0 scales.
TP2/TP4 shapes (32/16 heads) and rows1/2 are supported. This is an independent
compute stage with caller-owned buffers, without persistent/peer mailboxes.
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

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_gemv import (
    grouped_lds_bytes,
    grouped_wqb_device,
)
from atom.mono.device.ops import bf_hi, bf_lo, bf16_round

Q_RANK = 1280
HEAD_DIM = 512
ROPE = 64
TILE_N = 16
K_SPLIT = 8
THREADS = 256
UNIT_SCALES = 0x7F7F7F7F


def _kernel_name(rows, heads, debug):
    return f"atom_gfx1250_wqb_rope_r{rows}_h{heads}_d{int(debug)}_k8"


def _lds_bytes(rows):
    partial_off = (rows * (Q_RANK + Q_RANK // 32) + 15) // 16 * 16
    return partial_off + K_SPLIT * rows * TILE_N * 4


@lru_cache(maxsize=8)
def _launcher(rows: int, heads: int, debug: bool):
    if rows not in (1, 2) or heads not in (16, 32):
        raise ValueError("gfx1250 wq_b supports rows1/2 and 16/32 heads")
    output = heads * HEAD_DIM
    sa_off = rows * Q_RANK
    partial_off = (sa_off + rows * (Q_RANK // 32) + 15) // 16 * 16

    @flyc.jit
    def launch(q: fx.Pointer, qs: fx.Pointer, weight: fx.Pointer,
               scale: fx.Pointer, cos: fx.Pointer, sin: fx.Pointer,
               out: fx.Pointer, unrotated: fx.Pointer, stream: fx.Stream):
        @flyc.kernel(name=_kernel_name(rows, heads, debug),
                     known_block_size=[THREADS, 1, 1])
        def kernel(q: fx.Pointer, qs: fx.Pointer, weight: fx.Pointer,
                   scale: fx.Pointer, cos: fx.Pointer, sin: fx.Pointer,
                   out: fx.Pointer, unrotated: fx.Pointer):
            tid = fx.Int32(fx.thread_idx.x)
            lane, wave = tid % 32, tid // 32
            bid = fx.Int32(fx.block_idx.x)
            n0 = bid * TILE_N
            lane16, half = lane % 16, lane // 16
            arena = fx.SharedAllocator(static=False)
            arena.allocate(_lds_bytes(rows))
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            ld32, st32 = make_lds_copy_ops(32)
            ld128, st128 = make_lds_copy_ops(128)
            rx = bo.create_buffer_resource_from_addr(fx.ptrtoint(q), num_records_bytes=rows * Q_RANK)
            rxs = bo.create_buffer_resource_from_addr(fx.ptrtoint(qs), num_records_bytes=rows * Q_RANK // 32)
            rw = bo.create_buffer_resource_from_addr(fx.ptrtoint(weight), num_records_bytes=output * Q_RANK)
            rs = bo.create_buffer_resource_from_addr(fx.ptrtoint(scale), num_records_bytes=output * Q_RANK // 1024)
            rc = bo.create_buffer_resource_from_addr(fx.ptrtoint(cos), num_records_bytes=rows * ROPE)
            rn = bo.create_buffer_resource_from_addr(fx.ptrtoint(sin), num_records_bytes=rows * ROPE)
            ro = bo.create_buffer_resource_from_addr(fx.ptrtoint(out), num_records_bytes=rows * output * 2)
            if const_expr(debug):
                rd = bo.create_buffer_resource_from_addr(fx.ptrtoint(unrotated), num_records_bytes=rows * output * 2)

            steps = [wave + K_SPLIT * i for i in range(2)]
            weights, scales = [], []
            # Prefetch both rounds before consuming activation operands.
            for step in steps:
                st = fx.min(step, Q_RANK // 128 - 1)
                bv = [Vec(bo.buffer_load(rw, (n0 // 16 * Q_RANK * 16 + st * 2048 + half * 256
                                              + lane16 * 16 + j * 512) // 4,
                                        vec_width=4, dtype=T.i32)) for j in range(4)]
                weights.append(bv[0].shuffle(bv[1], list(range(8))).shuffle(
                    bv[2].shuffle(bv[3], list(range(8))), list(range(16))))
                scales.append(bo.buffer_load(rs, (n0 // 32) * (Q_RANK // 128) + st,
                                              vec_width=1, dtype=T.i32))
            fx.rocdl.sched_barrier(0)
            if tid < rows * Q_RANK // 16:
                values = Vec(bo.buffer_load(rx, tid * 4, vec_width=4, dtype=T.i32))
                st128(lds, tid * 16, values)
            if tid < rows * (Q_RANK // 128):
                codes = fx.Int32(bo.buffer_load(rxs, tid, vec_width=1, dtype=T.i32))
                st32(lds, sa_off + tid * 4, Vec.from_elements([codes]))
            fx.gpu.barrier()

            active_row = fx.min(lane16, rows - 1)
            atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
                16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32))
            accum = fx.make_rmem_tensor(8, fx.Float32)
            accum.store(fx.full(8, 0.0, fx.Float32))
            for i in range_constexpr(2):
                step = steps[i]
                st = fx.min(step, Q_RANK // 128 - 1)
                live = step < Q_RANK // 128
                av = [Vec(ld128(lds, active_row * Q_RANK + st * 128 + half * 16 + j * 32))
                      for j in range_constexpr(4)]
                avec = av[0].shuffle(av[1], list(range(8))).shuffle(
                    av[2].shuffle(av[3], list(range(8))), list(range(16)))
                a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(16, fx.Int32)
                a.store(Vec.from_elements([live.select(avec[j], fx.Int32(0)) for j in range(16)]))
                b.store(Vec.from_elements([live.select(weights[i][j], fx.Int32(0)) for j in range(16)]))
                sa = ld32(lds, sa_off + active_row * (Q_RANK // 32) + st * 4)[0]
                fx.gemm(atom, accum, b, a, accum,
                        scale_a=live.select(scales[i], fx.Int32(UNIT_SCALES)),
                        scale_b=live.select(sa, fx.Int32(UNIT_SCALES)))
            if lane16 < rows:
                values = accum.load()
                offset = partial_off + (wave * rows * TILE_N + lane16 * TILE_N + half * 8) * 4
                for j in range_constexpr(2):
                    st128(lds, offset + j * 16, Vec.from_elements(
                        [values[j * 4 + k].bitcast(fx.Int32) for k in range(4)]))
            fx.gpu.barrier()

            if tid < rows * TILE_N // 2:
                row, pair = tid // (TILE_N // 2), tid % (TILE_N // 2)
                column = n0 + 2 * pair
                sums = [fx.Float32(0.0), fx.Float32(0.0)]
                for w in range_constexpr(K_SPLIT):
                    for j in range_constexpr(2):
                        part = fx.Int32(ld32(lds, partial_off +
                                       (w * rows * TILE_N + row * TILE_N + 2 * pair + j) * 4)[0]).bitcast(fx.Float32)
                        sums[j] = sums[j] + part
                values = [bf16_round(v) for v in sums]
                dim = column % HEAD_DIM
                table_pair = fx.max(dim - (HEAD_DIM - ROPE), 0) // 2
                at = row * (ROPE // 2) + table_pair
                cw = fx.Int32(bo.buffer_load(rc, at // 2, vec_width=1, dtype=T.i32))
                sw = fx.Int32(bo.buffer_load(rn, at // 2, vec_width=1, dtype=T.i32))
                cs = (at % 2 == 0).select(bf_lo(cw), bf_hi(cw))
                sn = (at % 2 == 0).select(bf_lo(sw), bf_hi(sw))
                live = dim >= HEAD_DIM - ROPE
                rotated = [live.select(values[0] * cs - values[1] * sn, values[0]),
                           live.select(values[0] * sn + values[1] * cs, values[1])]
                for j in range_constexpr(2):
                    bo.buffer_store(rotated[j].to(fx.BFloat16), ro, row * output + column + j)
                    if const_expr(debug):
                        bo.buffer_store(values[j].to(fx.BFloat16), rd, row * output + column + j)

        kernel(q, qs, weight, scale, cos, sin, out, unrotated).launch(
            grid=(output // TILE_N, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def _arguments(q, qs, weight, scale, cos, sin, out, debug):
    owner = "gfx1250 wq_b/rope"
    rows = q.shape[0] if q.ndim == 2 else 0
    heads = out.shape[1] if out.ndim == 3 else 0
    if rows not in (1, 2) or heads not in (16, 32):
        raise ValueError(f"[{owner}] requires rows1/2 and 16/32 query heads")
    output = heads * HEAD_DIM
    specs = [(q, (rows, Q_RANK), torch.float8_e4m3fn, "q"),
             (weight, (output, Q_RANK), torch.float8_e4m3fn, "weight"),
             (cos, (rows, ROPE // 2), torch.bfloat16, "cos"),
             (sin, (rows, ROPE // 2), torch.bfloat16, "sin"),
             (out, (rows, heads, HEAD_DIM), torch.bfloat16, "out")]
    check_e8m0(qs, "qs", owner, (rows, Q_RANK // 32))
    check_e8m0(scale, "scale", owner, (output // 32, Q_RANK // 32))
    if debug is not None:
        specs.append((debug, (rows, heads, HEAD_DIM), torch.bfloat16, "unrotated"))
    for tensor, shape, dtype, name in specs:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"[{owner}] {name} must be contiguous {dtype} with shape {shape}")
    tensors = [q, qs, weight, scale, cos, sin, out] + ([] if debug is None else [debug])
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError(f"[{owner}] all buffers must be on the same GPU")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError(f"[{owner}] requires gfx1250")
    for i in range(6, len(tensors)):
        a = tensors[i]
        lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        for b in tensors[:i]:
            if lo < b.data_ptr() + b.numel() * b.element_size() and b.data_ptr() < hi:
                raise ValueError(f"[{owner}] output buffers must not overlap")
    dbg = flyc.from_c_void_p(fx.Uint8, 0) if debug is None else ptr_arg(debug)
    return tuple(ptr_arg(t) for t in tensors[:7]) + (dbg, torch.cuda.current_stream(q.device).cuda_stream)


def wqb_rope_gfx1250(q, qs, weight, scale, cos, sin, out, *, debug=None):
    """Write BF16 [rows,heads,512] queries using caller-owned buffers.

    Inputs, native-preshuffled weights and selected RoPE table rows must be
    finite. ``debug`` optionally receives the BF16 projection before RoPE.
    Output shape selects TP2/TP4's 32/16 heads; this validates compute geometry
    only and does not enable either complete mono backend.
    """
    args = _arguments(q, qs, weight, scale, cos, sin, out, debug)
    _run_compiled(_launcher(q.shape[0], out.shape[1], debug is not None), *args)
    return out


@lru_cache(maxsize=4)
def _grouped_launcher(rows, heads):
    @flyc.jit
    def launch(
        q: fx.Pointer,
        qs: fx.Pointer,
        weight: fx.Pointer,
        scale: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        out: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(
            name=f"atom_gfx1250_wqb_rope_r{rows}_h{heads}_n64_k8",
            known_block_size=[THREADS, 1, 1],
        )
        def kernel(
            q: fx.Pointer,
            qs: fx.Pointer,
            weight: fx.Pointer,
            scale: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            out: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(grouped_lds_bytes(rows, 4))
            grouped_wqb_device(
                q,
                qs,
                weight,
                scale,
                cos,
                sin,
                out,
                fx.Int64(fx.ptrtoint(arena.base_ptr)),
                tid,
                bid,
                rows,
                heads * HEAD_DIM,
            )

        kernel(q, qs, weight, scale, cos, sin, out).launch(
            grid=(heads * HEAD_DIM // 64, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


def wqb_rope_grouped_gfx1250(q, qs, weight, scale, cos, sin, out):
    """Four N16 tiles per CTA, with the original eight partials and rounding.

    Each pair of wave32 groups computes one N16 tile. Independent original
    K partials reduce in their original order; no reassociation is introduced.
    This variant removes all masked WMMA steps for K1280 and reduces the
    TP2 grid from 1024 to 256 CTAs. The original debug-capable API is retained.
    The N64 grouping serves cooperative K1 scheduling; standalone callers
    should retain the faster original N16 path unless measured otherwise.
    """
    args = _arguments(q, qs, weight, scale, cos, sin, out, None)
    _run_compiled(_grouped_launcher(q.shape[0], out.shape[1]), *args[:7], args[-1])
    return out
