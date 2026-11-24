# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router split-K partials -> AITER's biased top-k (``grouped_topk_kernel``, one group) in one launch (gfx950): lane l
of a token's wave sums experts 4 l .. 4 l + 3 in split order from 0.0, takes the sigmoid and the biased choice, runs k
rounds of a lane-local strict argmax and AITER's DPP tree read at lane 63, and renormalizes the unbiased sigmoids by
scale / sum: AITER's ids and weights bit for bit (ties included)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.warm_up import launcher

# one token (wave) per workgroup spreads the partial loads over more CUs: 172 rows 3.75 -> 3.33 us after the router
NW = 1
# kernel arguments preloaded into SGPRs instead of s_load'ed first
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
NEG_LOG2E = -1.44269504088896340736
# AITER's wave_reduce on gfx9: quad_perm [1,0,3,2], quad_perm [2,3,0,1],
# row_half_mirror, row_mirror, row_bcast:15, row_bcast:31
DPP_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp(v, ctrl):
    """v_mov_b32_dpp, all rows / banks: an invalid source lane keeps its value (the lane-63 result never reads one)."""
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _ir(v), _ir(v), ctrl, 0xF, 0xF, False))


@launcher
def compile_router_topk(num_experts: int, topk: int, split_k: int, tpw: int = 1):
    """``tpw``: tokens per wave, 1 or 2 (lanes 0-31 and 32-63)."""
    assert num_experts % 4 == 0 and num_experts <= 256 and 1 <= topk <= 64
    assert tpw == 1 or (tpw == 2 and num_experts <= 128 and topk <= 32)
    E, K, S = num_experts, topk, split_k
    LPT = 64 // tpw  # lanes per token
    # tpw 2: the first five steps never move a value across lanes 31 / 32 toward the lanes read, so lanes 31 and 63
    # hold the one-token result without the last step (row_bcast:31)
    TREE = DPP_TREE if tpw == 1 else DPP_TREE[:-1]
    name = f"router_splitk_biased_top{K}_e{E}_s{S}" + ("" if tpw == 1 else f"_tpw{tpw}")

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
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        t0 = (pid * NW + wave) * tpw
        if t0 < i32_m:
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
            if const_expr(tpw == 1):
                t, lh = t0, lane
                has = lane < E // 4
            else:
                t, lh = t0 + lane // LPT, lane % LPT
                has = (lh < E // 4) & (t < i32_m)
            base = t * i32_sm + lh * 4
            parts = [
                Vec(
                    buffer_ops.buffer_load(
                        pr, base + k * i32_sk, vec_width=4, dtype=fx.Float32, mask=has
                    )
                )
                for k in range_constexpr(S)
            ]
            bias = Vec(
                buffer_ops.buffer_load(
                    br, lh * 4, vec_width=4, dtype=fx.Float32, mask=lh < E // 4
                )
            )
            neg_inf = fx.Float32(float("-inf"))
            sig, choice = [], []
            for i in range_constexpr(4):
                x = fx.Float32(0.0)
                for k in range_constexpr(S):
                    x = x + fx.Float32(parts[k][i])
                # AITER's -log2(e) is a double literal: the product is a double rounded to fp32
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
                    mi = take.select(lh * 4 + i, mi)
                for ctrl in TREE:
                    rv = _dpp(mv.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    ri = _dpp(mi, ctrl)
                    take = rv > mv
                    mv = take.select(rv, mv)
                    mi = take.select(ri, mi)
                picks = []
                for h in range_constexpr(tpw):
                    hidx = fx.Int32(
                        fx.rocdl.readlane(T.i32, mi, fx.Int32(LPT * h + LPT - 1))
                    )
                    howner = hidx // 4 + LPT * h
                    hcomp = hidx % 4
                    # the selected expert's unbiased sigmoid, from its owner lane
                    sv = [
                        fx.Int32(
                            fx.rocdl.readlane(T.i32, sig[i].bitcast(fx.Int32), howner)
                        ).bitcast(fx.Float32)
                        for i in range_constexpr(4)
                    ]
                    hw = sv[0]
                    for i in range_constexpr(1, 4):
                        hw = (hcomp == i).select(sv[i], hw)
                    picks.append((hidx, howner, hw))
                idx, owner, w = picks[0]
                if const_expr(tpw == 2):
                    upper = lane >= LPT
                    idx = upper.select(picks[1][0], idx)
                    owner = upper.select(picks[1][1], owner)
                    w = upper.select(picks[1][2], w)
                comp = idx % 4
                for i in range_constexpr(4):
                    choice[i] = ((lane == owner) & (comp == i)).select(
                        neg_inf, choice[i]
                    )
                total = total + w
                out_id = (lh == k).select(idx, out_id)
                out_val = (lh == k).select(w, out_val)
            scale = i32_scale.bitcast(fx.Float32)
            factor = scale / total
            off = t * i32_stk + lh
            keep = lh < K
            if const_expr(tpw == 2):
                keep = keep & (t < i32_m)
            buffer_ops.buffer_store(out_val * factor, wr, off, mask=keep)
            buffer_ops.buffer_store(out_id, ir, off, mask=keep)

    @flyc.jit
    def launch_router_topk_mxfp4_moe(
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
        stream: fx.Stream,
    ):
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
        ).launch(
            grid=(fx.Int64((i32_m + (NW * tpw - 1)) // (NW * tpw)), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_router_topk_mxfp4_moe.kernel_name = name
    launch_router_topk_mxfp4_moe.compile_hints = PRELOAD_HINTS
    return launch_router_topk_mxfp4_moe


# ---- host: compile cache, launch
import functools  # noqa: E402
import struct  # noqa: E402

import torch  # noqa: E402


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


# two tokens per wave from 4097 tokens: -0.3 .. -5.9 us at 4097-32768, +0.35 at 4096 (split 8)
TPW2_FROM = 4097


@functools.cache
def _get_kernel(num_experts: int, topk: int, split_k: int, tpw: int = 1):
    from .router_topk_vec2 import VEC2_SPLITS, compile_router_topk_vec2
    if (num_experts, topk, tpw) == (128, 4, 1) and split_k in VEC2_SPLITS:
        # the one-token router splits: two experts per lane, the same bits
        return compile_router_topk_vec2(num_experts, topk, split_k, tpw)
    return compile_router_topk(num_experts, topk, split_k, tpw)


def splitk_biased_topk(
    partials: torch.Tensor,  # [M, split_k, E] fp32 (any strides, last dim dense)
    correction_bias: torch.Tensor,  # [E] fp32
    topk_weights: torch.Tensor,  # [M, topk] fp32 (row stride free)
    topk_ids: torch.Tensor,  # [M, topk] int32, same row stride
    routed_scaling_factor: float,
) -> None:
    """Sums the router's split-K partials and runs AITER's biased top-k (one group, renormalized) on the logits: AITER's
    ids and weights bit for bit."""
    m, split_k, num_experts = partials.shape
    if m == 0:
        return
    assert partials.dtype == torch.float32 and partials.stride(2) == 1
    assert correction_bias.dtype == torch.float32 and correction_bias.is_contiguous()
    assert topk_weights.stride(1) == 1 and topk_ids.stride(1) == 1
    assert topk_weights.stride(0) == topk_ids.stride(0)
    end = (
        (m - 1) * partials.stride(0) + (split_k - 1) * partials.stride(1) + num_experts
    )
    tpw = 2 if m >= TPW2_FROM and num_experts <= 128 else 1
    _run_compiled(
        _get_kernel(num_experts, topk_ids.shape[1], split_k, tpw),
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
        torch.cuda.current_stream(),
    )
