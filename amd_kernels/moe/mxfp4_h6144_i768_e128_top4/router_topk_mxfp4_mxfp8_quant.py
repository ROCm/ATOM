# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``router_topk.py``'s top-k (one wave per token) with ``quant.py``'s MXFP4 and ``shared_quant``'s MXFP8 quant of the
token's row (one amax for both) in the same launch (gfx950): lane l takes the 32-value blocks l, l + 64, l + 128, so ids,
weights and both code / scale pairs are bit-equal to the standalone kernels. The row loads issue right behind the
partial loads, so their latency hides under the top-k. The host call takes ``_sorted_tasks`` at 1-256 rows."""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common.loaders import _as_f32
from .i3072.gemm1_prefill_h6144_i3072_persistent import _cvt_pk_fp4, _e8m0_even_e2m1
from .mxfp8.gemm1_prefill_h6144_i3072 import _fmax3_abs
from .router_topk import DPP_TREE, NEG_LOG2E, NW, PRELOAD_HINTS, _dpp, _run_compiled

from .shared_quant import _fp8, _e8m0_fp8
from .mxfp8.shared_dense_4wave import _scale_off
from amd_kernels.warm_up import launcher

H = 6144
BLOCKS = H // 32  # 32-value blocks per row
BPL = BLOCKS // 64  # blocks per lane
assert BLOCKS % 64 == 0


@launcher
def compile_router_topk_mxfp4_mxfp8_quant(num_experts: int, topk: int, split_k: int, s8_tiles: bool = False):
    """``s8_tiles``: the FP8 scales in ``shared_dense._shuffle_scales``'s 32-row / 256-K tiled layout; the padding
    rows' waves write their zeros."""
    assert num_experts % 4 == 0 and num_experts <= 256 and 1 <= topk <= 64
    E, K, S = num_experts, topk, split_k
    name = f"router_splitk_biased_top{K}_mxfp4_mxfp8_quant_h{H}_e{E}_s{S}" + ("_s8tiles" if s8_tiles else "")

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_part: fx.Int64,  # split-K partials, fp32, [M, S, E] by strides
        arg_bias: fx.Int64,  # correction bias [E] fp32
        arg_w: fx.Int64,  # top-k weights fp32, row stride i32_stk
        arg_ids: fx.Int64,  # top-k ids int32, row stride i32_stk
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,  # partial strides (elements): token, split
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,  # routed_scaling_factor (fp32 bits)
        arg_x: fx.Int64,  # bf16 [M, H] row-major (the router GEMM's input)
        arg_q: fx.Int64,  # e2m1 codes [M, H / 2]
        arg_s: fx.Int64,  # routed e8m0 [M,H/32]
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        t = pid * NW + wave
        if const_expr(s8_tiles):
            m32 = (i32_m + 31) // 32 * 32
            if t >= i32_m:
                if t < m32:
                    zr = buffer_ops.create_buffer_resource_from_addr(arg_s8, num_records_bytes=fx.Int64(m32) * (H // 32))
                    for b in range_constexpr(BPL):
                        buffer_ops.buffer_store(fx.Int32(0).to(fx.Int8), zr,
                                                _scale_off(t, lane + 64 * b, H // 256), offset_is_bytes=True)
        if t < i32_m:
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_part, num_records_bytes=i64_part_bytes
            )
            br = buffer_ops.create_buffer_resource_from_addr(
                arg_bias, num_records_bytes=E * 4
            )
            out_bytes = fx.Int64(i32_m) * fx.Int64(i32_stk) * 4
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=out_bytes
            )
            ir = buffer_ops.create_buffer_resource_from_addr(
                arg_ids, num_records_bytes=out_bytes
            )
            has = lane < E // 4
            base = t * i32_sm + lane * 4
            parts = [
                Vec(
                    buffer_ops.buffer_load(
                        pr, base + k * i32_sk, vec_width=4, dtype=fx.Float32, mask=has
                    )
                )
                for k in range_constexpr(S)
            ]
            # token t's row, loaded behind the partials: lane l takes 32-value blocks l + 64 b (4 x 16 B each)
            row_bytes = fx.Int64(i32_m) * (H * 2)
            xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=row_bytes)
            xv = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            xr, (t * BLOCKS + lane + 64 * b) * 16 + j * 4, vec_width=4, dtype=fx.Int32
                        )
                    )
                    for j in range_constexpr(4)
                ]
                for b in range_constexpr(BPL)
            ]
            bias = Vec(
                buffer_ops.buffer_load(
                    br, lane * 4, vec_width=4, dtype=fx.Float32, mask=has
                )
            )
            neg_inf = fx.Float32(float("-inf"))
            sig, choice = [], []
            for i in range_constexpr(4):
                x = fx.Float32(0.0)
                for k in range_constexpr(S):
                    x = x + fx.Float32(parts[k][i])
                y = (x.to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
                s = fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + y.exp2()))
                sig.append(s)
                choice.append(has.select(s + fx.Float32(bias[i]), neg_inf))
            total = fx.Float32(0.0)
            out_id = fx.Int32(0)
            out_val = fx.Float32(0.0)
            for k in range_constexpr(K):
                mv = neg_inf
                mi = fx.Int32(k)
                for i in range_constexpr(4):
                    take = choice[i] > mv
                    mv = take.select(choice[i], mv)
                    mi = take.select(lane * 4 + i, mi)
                for ctrl in DPP_TREE:
                    rv = _dpp(mv.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    ri = _dpp(mi, ctrl)
                    take = rv > mv
                    mv = take.select(rv, mv)
                    mi = take.select(ri, mi)
                idx = fx.Int32(fx.rocdl.readlane(T.i32, mi, fx.Int32(63)))
                owner = idx // 4
                comp = idx % 4
                # the selected expert's unbiased sigmoid, from its owner lane
                sv = [
                    fx.Int32(
                        fx.rocdl.readlane(T.i32, sig[i].bitcast(fx.Int32), owner)
                    ).bitcast(fx.Float32)
                    for i in range_constexpr(4)
                ]
                w = sv[0]
                for i in range_constexpr(1, 4):
                    w = (comp == i).select(sv[i], w)
                for i in range_constexpr(4):
                    choice[i] = ((lane == owner) & (comp == i)).select(
                        neg_inf, choice[i]
                    )
                total = total + w
                out_id = (lane == k).select(idx, out_id)
                out_val = (lane == k).select(w, out_val)
            scale = i32_scale.bitcast(fx.Float32)
            factor = scale / total
            off = t * i32_stk + lane
            buffer_ops.buffer_store(out_val * factor, wr, off, mask=lane < K)
            buffer_ops.buffer_store(out_id, ir, off, mask=lane < K)
            # both quants, quant.py's and shared_quant's arithmetic op for op
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=fx.Int64(i32_m) * (H // 2)
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_s, num_records_bytes=fx.Int64(i32_m) * (H // 32)
            )
            q8r = buffer_ops.create_buffer_resource_from_addr(arg_q8, num_records_bytes=fx.Int64(i32_m) * H)
            s8_rows = i32_m
            if const_expr(s8_tiles):
                s8_rows = (i32_m + 31) // 32 * 32
            s8r = buffer_ops.create_buffer_resource_from_addr(arg_s8, num_records_bytes=fx.Int64(s8_rows) * (H//32))
            for b in range_constexpr(BPL):
                blk = t * BLOCKS + lane + 64 * b
                vals = []
                for j in range_constexpr(4):
                    for kk in range_constexpr(4):
                        w = fx.Int32(xv[b][j][kk])
                        vals.append(_as_f32(w << 16))
                        vals.append(_as_f32(w & fx.Int32(-65536)))
                amax = _fmax3_abs(vals[0], vals[1], vals[2])
                for i in range_constexpr(3, 31, 2):
                    amax = _fmax3_abs(vals[i], vals[i + 1], amax)
                amax = _fmax3_abs(vals[31], vals[31], amax)
                e8 = _e8m0_fp8(amax)
                sf8 = (e8 << 23).bitcast(fx.Float32)
                for half in range_constexpr(2):
                    words = []
                    for dw in range_constexpr(4):
                        j = half * 16 + dw * 4
                        pk = _fp8(fx.Int32(0), vals[j], vals[j+1], sf8, False)
                        pk = _fp8(pk, vals[j+2], vals[j+3], sf8, True)
                        words.append(pk)
                    buffer_ops.buffer_store(Vec.from_elements(words, fx.Int32), q8r, blk*8+half*4)
                if const_expr(s8_tiles):
                    buffer_ops.buffer_store(e8.to(fx.Int8), s8r, _scale_off(t, lane + 64 * b, H // 256),
                                            offset_is_bytes=True)
                else:
                    buffer_ops.buffer_store(e8.to(fx.Int8), s8r, blk)
                e8m0 = _e8m0_even_e2m1(amax)
                scale_f = _as_f32(e8m0 << 23)
                qo = []
                for d in range_constexpr(4):
                    p = fx.Int32(0)
                    for kk in range_constexpr(4):
                        p = _cvt_pk_fp4(p, vals[8 * d + 2 * kk], vals[8 * d + 2 * kk + 1], scale_f, kk)
                    qo.append(p)
                buffer_ops.buffer_store(Vec.from_elements(qo, fx.Int32), qr, blk * 4)
                buffer_ops.buffer_store(fx.Int32(e8m0).to(fx.Int8), sr, blk)

    @flyc.jit
    def launch_router_topk_mxfp4_mxfp8_quant(
        arg_part: fx.Int64,
        arg_bias: fx.Int64,
        arg_w: fx.Int64,
        arg_ids: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,
        arg_x: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
        stream: fx.Stream,
    ):
        rows = i32_m
        if const_expr(s8_tiles):
            rows = (i32_m + 31) // 32 * 32
        kernel(
            arg_part,
            arg_bias,
            arg_w,
            arg_ids,
            i64_part_bytes,
            i32_m,
            i32_sm,
            i32_sk,
            i32_stk,
            i32_scale,
            arg_x,
            arg_q,
            arg_s,
            arg_q8,
            arg_s8,
        ).launch(
            grid=(fx.Int64((rows + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_router_topk_mxfp4_mxfp8_quant.kernel_name = name
    launch_router_topk_mxfp4_mxfp8_quant.compile_hints = PRELOAD_HINTS
    return launch_router_topk_mxfp4_mxfp8_quant


@functools.cache
def _get_kernel(num_experts: int, topk: int, split_k: int, s8_tiles: bool = False):
    return compile_router_topk_mxfp4_mxfp8_quant(num_experts, topk, split_k, s8_tiles)


def splitk_biased_topk_mxfp4_mxfp8_quant(
    partials: torch.Tensor,  # [M, split_k, E] fp32 (any strides, last dim dense)
    correction_bias: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int32, same row stride
    routed_scaling_factor: float,
    x: torch.Tensor,  # bf16 [M, H] contiguous
    *,
    s8_tiles: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Top-k and both input quants from one bf16 read -> ((FP4 codes, e8m0 scales), (FP8 codes, e8m0 scales)).

    ``s8_tiles`` (> 256 rows): the FP8 scales [ceil32(M), H/32] in the shared GEMMs' tiled layout, padding zero.
    """
    m, split_k, num_experts = partials.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape == (m, H)
    assert partials.dtype == torch.float32 and partials.stride(2) == 1
    assert correction_bias.dtype == torch.float32 and correction_bias.is_contiguous()
    assert topk_weights.stride(1) == 1 and topk_ids.stride(1) == 1
    assert topk_weights.stride(0) == topk_ids.stride(0)
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((m, H // 32), dtype=torch.uint8, device=x.device)
    q8 = torch.empty((m,H),dtype=torch.uint8,device=x.device)
    s8 = torch.empty((((m + 31) // 32 * 32) if s8_tiles else m, H // 32), dtype=torch.uint8, device=x.device)
    end = (m - 1) * partials.stride(0) + (split_k - 1) * partials.stride(1) + num_experts
    get_kernel = _get_kernel
    # separate top-k and quant workgroups to 256 rows (the 16-row expert tier): the whole entry faster; above, noise
    if 0 < m <= 256:
        assert not s8_tiles
        from .router_topk_mxfp4_mxfp8_quant_sorted_tasks import _get_kernel as get_kernel
        token_major = split_k == 48 and partials.stride(0) == split_k * num_experts and partials.stride(1) == num_experts
        launch = (get_kernel(num_experts, topk_ids.shape[1], split_k, True) if token_major
                  else get_kernel(num_experts, topk_ids.shape[1], split_k))
    else:
        launch = get_kernel(num_experts, topk_ids.shape[1], split_k, s8_tiles)
    _run_compiled(
        launch,
        partials.data_ptr(),
        correction_bias.data_ptr(),
        topk_weights.data_ptr(),
        topk_ids.data_ptr(),
        end * 4,
        m,
        partials.stride(0),
        partials.stride(1),
        topk_ids.stride(0),
        struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0],
        x.data_ptr(),
        q.data_ptr(),
        s.data_ptr(),
        q8.data_ptr(),
        s8.data_ptr(),
        torch.cuda.current_stream(),
    )
    return (q, s), (q8, s8)
