# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM-owned composable GDR emitter, adapted from AITER gdr_decode.

The recurrence is derived from AITER revision
48cdd1db73691e42d834968f87110cab2721864e (MIT). Composition, device-visible
output stores and BF16 beta rounding are local to ATOM; no AITER
branch or factory modification is required.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.gdr_common import _gview, _load_vec, _store_vec
from aiter.ops.flydsl.kernels.kernels_common import LOG2E as _LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
from flydsl.expr import range_constexpr

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.shapes import GDN_HEAD_DIM
from atom.mono.device.ops import CM_DEV
from atom.mono.plan.execution import WAVES


def _fast_exp(x):
    return fx.exp2(fx.Float32(x) * _LOG2E, fastmath="afn")


@functools.lru_cache(maxsize=16)
def create_qwen_gdr_decode(dims: Dims, *, complete_head=False):
    """Compose a q1 BF16/VK recurrence; batch norm owns a complete V head."""

    num_k_heads = dims.gdn_key_heads
    num_v_heads = dims.gdn_value_heads
    head_k_dim = head_v_dim = GDN_HEAD_DIM
    q_strides = k_strides = (dims.gdn_key_dim, dims.gdn_key_dim, GDN_HEAD_DIM, 1)
    v_strides = (dims.gdn_value_dim, dims.gdn_value_dim, GDN_HEAD_DIM, 1)
    a_strides = b_strides = (num_v_heads, num_v_heads, 1)
    # Native state is exposed as logical KV; the emitter consumes its physical VK base.
    state_strides = (
        num_v_heads * GDN_HEAD_DIM * GDN_HEAD_DIM,
        GDN_HEAD_DIM * GDN_HEAD_DIM,
        GDN_HEAD_DIM,
        1,
    )
    seq_length = 1
    softplus_beta = 1.0
    softplus_threshold = 20.0
    NUM_BLOCKS_PER_V_DIM = 1 if complete_head else 2
    NUM_WARPS = WAVES
    WARP_THREADS_K = 16
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K
    VALUES_PER_THREAD_K = 4
    data_num = A_log_num = fx.BFloat16
    state_num = fx.Float32

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    assert NUM_WARPS * WARP_SIZE == 512
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = (1, 2, 4, 8)

    @flyc.jit
    def emit_gdr_decode_task(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        tidx,
        bidx,
        output_smem=None,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        width_i32 = _to_raw(fx.Int32(WARP_SIZE))

        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_data = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), data_num)
        cp_out = fx.make_copy_atom(fx.rocdl.BufferCopy16b(CM_DEV), data_num)
        cp_data_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy(data_num.width * VALUES_PER_THREAD_K), data_num
        )
        cp_A_log = fx.make_copy_atom(fx.rocdl.BufferCopy(A_log_num.width), A_log_num)
        cp_state_vec = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), state_num)

        read_indices_view = _gview(read_indices, None, (batch_size, 1), (1, 1))
        write_indices_view = _gview(write_indices, None, (batch_size, 1), (1, 1))
        read_pool_idx = _load_vec(
            cp_i32, fx.slice(read_indices_view, (b_i, None)), 1, fx.Int32
        )
        write_pool_idx = _load_vec(
            cp_i32, fx.slice(write_indices_view, (b_i, None)), 1, fx.Int32
        )

        def projection_view(tensor, heads, dim, strides):
            return _gview(
                tensor,
                None,
                (batch_size, 1, heads, dim // VALUES_PER_THREAD_K, VALUES_PER_THREAD_K),
                (*strides[:-1], VALUES_PER_THREAD_K, 1),
            )

        q_view = projection_view(query, num_k_heads, head_k_dim, q_strides)
        k_view = projection_view(key, num_k_heads, head_k_dim, k_strides)
        v_view = _gview(
            value, None, (batch_size, 1, num_v_heads, head_v_dim, 1), (*v_strides, 1)
        )
        a_view = _gview(a, None, (batch_size, 1, num_v_heads, 1), (*a_strides, 1))
        b_view = _gview(b, None, (batch_size, 1, num_v_heads, 1), (*b_strides, 1))
        dt_bias_view = _gview(dt_bias, None, (num_v_heads, 1), (1, 1))
        A_log_view = _gview(A_log, None, (num_v_heads, 1), (1, 1))
        out_view = _gview(
            out,
            None,
            (batch_size, seq_length, num_v_heads, head_v_dim, 1),
            (
                seq_length * num_v_heads * head_v_dim,
                num_v_heads * head_v_dim,
                head_v_dim,
                1,
                1,
            ),
        )

        state_shape = (
            num_v_heads,
            head_v_dim,
            head_k_dim // VALUES_PER_THREAD_K,
            VALUES_PER_THREAD_K,
        )
        state_stride = (
            state_strides[1],
            state_strides[2],
            VALUES_PER_THREAD_K,
            1,
        )
        read_state_view = _gview(
            state,
            fx.Int64(read_pool_idx) * fx.Int64(state_strides[0]),
            state_shape,
            state_stride,
        )
        write_state_view = _gview(
            state,
            fx.Int64(write_pool_idx) * fx.Int64(state_strides[0]),
            state_shape,
            state_stride,
        )

        # Skip CG-pad slots (indices sentinel < 0). The guarded body is a
        # closure so the runtime `if` sees an opaque call and lowers to scf.if.
        def _do_decode():
            r_A_log = _load_vec(
                cp_A_log, fx.slice(A_log_view, (hv_i, None)), 1, A_log_num
            )
            r_A_log = r_A_log.to(fx.Float32)
            r_dt_bias = _load_vec(
                cp_data, fx.slice(dt_bias_view, (hv_i, None)), 1, data_num
            ).to(fx.Float32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = _load_vec(
                        cp_state_vec,
                        fx.slice(
                            read_state_view,
                            (
                                hv_i,
                                global_v_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        state_num,
                    )

            sq_i = 0
            r_a = _load_vec(
                cp_data,
                fx.slice(a_view, (b_i, sq_i, hv_i, None)),
                1,
                data_num,
            ).to(fx.Float32)
            r_b = _load_vec(
                cp_data,
                fx.slice(b_view, (b_i, sq_i, hv_i, None)),
                1,
                data_num,
            ).to(fx.Float32)
            x = r_a + r_dt_bias
            beta_x = softplus_beta_ * x

            # For beta_x > threshold, softplus(x) == x; both arms run and
            # the overflowing one is dropped.
            softplus_big = (f32_1 / softplus_beta_) * fx.math.log1p(_fast_exp(beta_x))
            softplus_x = (fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)).select(
                softplus_big, x
            )

            r_g_value = -_fast_exp(r_A_log) * softplus_x
            r_beta = f32_1 / (f32_1 + _fast_exp(-r_b))
            bits = fx.Float32(r_beta).bitcast(fx.Uint32)
            rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
            r_beta = fx.Float32((rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
            r_g = _fast_exp(r_g_value)

            r_g_vec = fx.Vector.filled(VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32)

            sq_vecs = [0] * WARP_TILE_K_ITERS
            sk_vecs = [0] * WARP_TILE_K_ITERS

            scale_vec = fx.Vector.filled(
                VALUES_PER_THREAD_K, fx.Float32(scale), fx.Float32
            )

            for ki in range_constexpr(WARP_TILE_K_ITERS):
                warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                q_vec = _load_vec(
                    cp_data_vec,
                    fx.slice(
                        q_view,
                        (
                            b_i,
                            sq_i,
                            hk_i,
                            warp_k_vec_i // VALUES_PER_THREAD_K,
                            None,
                        ),
                    ),
                    VALUES_PER_THREAD_K,
                    data_num,
                )
                k_vec = _load_vec(
                    cp_data_vec,
                    fx.slice(
                        k_view,
                        (
                            b_i,
                            sq_i,
                            hk_i,
                            warp_k_vec_i // VALUES_PER_THREAD_K,
                            None,
                        ),
                    ),
                    VALUES_PER_THREAD_K,
                    data_num,
                )
                sq_vecs[ki] = q_vec.to(fx.Float32)
                sk_vecs[ki] = k_vec.to(fx.Float32)

            sum_q_partial_vec = fx.Vector.from_elements(
                [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                fx.Float32,
            )
            sum_k_partial_vec = fx.Vector.from_elements(
                [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                fx.Float32,
            )
            for ki in range_constexpr(WARP_TILE_K_ITERS):
                sum_q_partial_vec = sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                sum_k_partial_vec = sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
            sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(fx.ReductionOp.ADD)
            sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(fx.ReductionOp.ADD)
            for offset in WARP_THREADS_K_SHFL_OFFSETS:
                sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                    offset, WARP_SIZE
                )
                sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                    offset, WARP_SIZE
                )
            local_sum_q = fx.gpu.shuffle_idx(
                sum_q_partial,
                fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K),
                width_i32,
            )
            local_sum_k = fx.gpu.shuffle_idx(
                sum_k_partial,
                fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K),
                width_i32,
            )
            inv_norm_q = fx.math.rsqrt(local_sum_q + 1e-6)
            inv_norm_k = fx.math.rsqrt(local_sum_k + 1e-6)
            inv_norm_q_vec = fx.Vector.filled(
                VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
            )
            inv_norm_k_vec = fx.Vector.filled(
                VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
            )
            for ki in range_constexpr(WARP_TILE_K_ITERS):
                sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec

            dot_kq_vec = fx.Vector.from_elements(
                [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
            )
            for ki in range_constexpr(WARP_TILE_K_ITERS):
                dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
            dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
            for offset in WARP_THREADS_K_SHFL_OFFSETS:
                dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                r_v = _load_vec(
                    cp_data,
                    fx.slice(v_view, (b_i, sq_i, hv_i, global_v_i, None)),
                    1,
                    data_num,
                ).to(fx.Float32)

                sum_hk = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                    fx.Float32,
                )
                sum_hq_old = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                    fx.Float32,
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                    h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                    sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                    sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(offset, WARP_SIZE)

                v_new = (r_v - sum_hk) * r_beta
                v_new = fx.gpu.shuffle_idx(
                    v_new,
                    fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K),
                    width_i32,
                )
                sum_hq = sum_hq_old + v_new * dot_kq
                v_new_bcast = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    h_new = fx.math.fma(
                        sk_vecs[ki],
                        v_new_bcast,
                        state_vecs[vi * WARP_TILE_K_ITERS + ki],
                    )
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                sum_hq = sum_hq.to(data_num)

                # Only k-vec lane 0 writes the q output.
                def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                    if output_smem is None:
                        _store_vec(
                            cp_out,
                            fx.slice(out_view, (b_i, _sq, hv_i, _gv, None)),
                            _sum_hq,
                            1,
                            data_num,
                        )
                    else:
                        fx.ptr_store(_sum_hq, output_smem + _gv)

                if warp_k_vec_start == 0:
                    _write_q()

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    _store_vec(
                        cp_state_vec,
                        fx.slice(
                            write_state_view,
                            (
                                hv_i,
                                global_v_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        out_vec,
                        VALUES_PER_THREAD_K,
                        state_num,
                    )

        def _zero_padding_output():
            zero = fx.Float32(0.0).to(data_num)
            sq_i = 0
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V

                def _write_zero(_sq=sq_i, _gv=global_v_i):
                    if output_smem is None:
                        _store_vec(
                            cp_out,
                            fx.slice(out_view, (b_i, _sq, hv_i, _gv, None)),
                            zero,
                            1,
                            data_num,
                        )
                    else:
                        fx.ptr_store(zero, output_smem + _gv)

                if warp_k_vec_start == 0:
                    _write_zero()

        if (read_pool_idx >= 0) & (write_pool_idx >= 0):
            _do_decode()
        else:
            _zero_padding_output()

    return emit_gdr_decode_task
