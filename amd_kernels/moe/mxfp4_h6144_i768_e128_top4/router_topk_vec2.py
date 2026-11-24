# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``router_topk.py``'s top-4 of 128 experts (one token per wave) with two experts per lane (gfx950): all 64 lanes sum
two experts and take their sigmoids, and two bpermutes restore the four-expert layout in lanes 0-31 before the same
argmax tree: the same ids and weights bit for bit."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
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
# the router splits this variant serves; S12 (2049-2688 rows): -0.15..-0.27 us vs four experts per lane
VEC2_SPLITS = (8, 12, 16, 48)


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp(v, ctrl):
    """v_mov_b32_dpp, all rows / banks: an invalid source lane keeps its value (the lane-63 result never reads one)."""
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _ir(v), _ir(v), ctrl, 0xF, 0xF, False))


@launcher
def compile_router_topk_vec2(num_experts: int, topk: int, split_k: int, tpw: int = 1):
    assert (num_experts, topk, tpw) == (128, 4, 1) and split_k in VEC2_SPLITS
    E, K, S = num_experts, topk, split_k
    name = f"router_splitk_biased_top{K}_vec2_e{E}_s{S}"

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
        t0 = pid
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
            t, lh = t0, lane
            has = lane < E // 4
            # lane l: experts 4 (l % 32) + 2 (l // 32) + 0 / 1, so lanes l and l + 32 hold one quartet
            el = lane % 32 * 4 + lane // 32 * 2
            base = t * i32_sm + el
            parts = [Vec(buffer_ops.buffer_load(
                pr, base + k * i32_sk, vec_width=2, dtype=fx.Float32
            )) for k in range_constexpr(S)]
            bias = Vec(buffer_ops.buffer_load(br, lane * 4, vec_width=4, dtype=fx.Float32, mask=has))
            neg_inf = fx.Float32(float("-inf"))
            ss = []
            for i in range_constexpr(2):
                x = fx.Float32(0.0)
                for k in range_constexpr(S):
                    x = x + fx.Float32(parts[k][i])
                y = (x.to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
                ss.append(fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + y.exp2())))
            # components 2 / 3 back to the quartet's lower lane, for the same argmax tree and ties
            sig = [ss[0], ss[1],
                   fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane ^ 32) * 4, ss[0].bitcast(fx.Int32))).bitcast(fx.Float32),
                   fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane ^ 32) * 4, ss[1].bitcast(fx.Int32))).bitcast(fx.Float32)]
            choice = [has.select(sig[i] + fx.Float32(bias[i]), neg_inf) for i in range_constexpr(4)]
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
                for ctrl in DPP_TREE:
                    rv = _dpp(mv.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    ri = _dpp(mi, ctrl)
                    take = rv > mv
                    mv = take.select(rv, mv)
                    mi = take.select(ri, mi)
                idx = fx.Int32(fx.rocdl.readlane(T.i32, mi, fx.Int32(63)))
                owner = idx // 4
                comp = idx % 4
                # the component first: one readlane per pick instead of four
                local_w = sig[0]
                for i in range_constexpr(1, 4):
                    local_w = (comp == i).select(sig[i], local_w)
                w = fx.Int32(fx.rocdl.readlane(
                    T.i32, local_w.bitcast(fx.Int32), owner
                )).bitcast(fx.Float32)
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
            buffer_ops.buffer_store(out_val * factor, wr, off, mask=keep)
            buffer_ops.buffer_store(out_id, ir, off, mask=keep)

    @flyc.jit
    def launch_router_topk_vec2_mxfp4_moe(
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
            grid=(fx.Int64(i32_m), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_router_topk_vec2_mxfp4_moe.kernel_name = name
    launch_router_topk_vec2_mxfp4_moe.compile_hints = PRELOAD_HINTS
    return launch_router_topk_vec2_mxfp4_moe
