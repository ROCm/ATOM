# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_GDN_FIX_BT, check_shared_mem

# The Triton kernel's launch configuration, fixed: an autotuner would time its
# configurations on the first launch of every process, in the serving loop.
# gfx1250 runs `chunk_o_gfx1250`.
_TRITON_BK = _TRITON_BV = 64 if check_shared_mem() else 32
_TRITON_NUM_WARPS, _TRITON_NUM_STAGES = 4, 2


@functools.lru_cache(maxsize=None)
def _use_gfx1250_kernel(device_index: int) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    return props.gcnArchName.split(":")[0] == "gfx1250"


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NT": lambda args: args["nt_ptr"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NT: tl.constexpr,
    HEAD_MAJOR_VK: tl.constexpr = False,
    USE_EXP2: tl.constexpr = False,
):
    T_FLAT = T
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN and HAS_NT:
        if i_t >= tl.load(nt_ptr):
            return
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    if HEAD_MAJOR_VK:
        v += (i_h.to(tl.int64) * T_FLAT + bos) * V
    else:
        v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        # [BT, BK]
        _p_q_0 = (i_t * BT) + tl.arange(0, BT)
        _p_q_1 = (i_k * BK) + tl.arange(0, BK)
        b_q = tl.load(
            q + _p_q_0[:, None] * (Hg * K) + _p_q_1[None, :] * (1),
            mask=(_p_q_0[:, None] < (T)) & (_p_q_1[None, :] < (K)),
            other=0.0,
        )
        # [BK, BT]
        _p_k_0 = (i_k * BK) + tl.arange(0, BK)
        _p_k_1 = (i_t * BT) + tl.arange(0, BT)
        b_k = tl.load(
            k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (Hg * K),
            mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
            other=0.0,
        )
        # [BK, BV]
        _p_h_0 = (i_k * BK) + tl.arange(0, BK)
        _p_h_1 = (i_v * BV) + tl.arange(0, BV)
        h_offsets = _p_h_0[:, None] * V + _p_h_1[None, :]
        if HEAD_MAJOR_VK:
            h_offsets = _p_h_0[:, None] + _p_h_1[None, :] * K
        b_h = tl.load(
            h + h_offsets,
            mask=(_p_h_0[:, None] < (K)) & (_p_h_1[None, :] < (V)),
            other=0.0,
        )

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if HEAD_MAJOR_VK:
            b_h = b_h.to(b_q.dtype)
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        if HEAD_MAJOR_VK:
            g += i_h.to(tl.int64) * T_FLAT + bos
        else:
            g += bos * H + i_h
        _p_g_0 = (i_t * BT) + tl.arange(0, BT)
        g_stride = 1 if HEAD_MAJOR_VK else H
        b_g = tl.load(g + _p_g_0 * g_stride, mask=(_p_g_0 < T), other=0.0)
        if USE_EXP2:
            b_o = b_o * tl.exp2(b_g)[:, None]
            b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
        else:
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    _p_v_0 = (i_t * BT) + tl.arange(0, BT)
    _p_v_1 = (i_v * BV) + tl.arange(0, BV)
    b_v = tl.load(
        v + _p_v_0[:, None] * (V if HEAD_MAJOR_VK else H * V) + _p_v_1[None, :],
        mask=(_p_v_0[:, None] < (T)) & (_p_v_1[None, :] < (V)),
        other=0.0,
    )

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    _p_o_0 = (i_t * BT) + tl.arange(0, BT)
    _p_o_1 = (i_v * BV) + tl.arange(0, BV)
    tl.store(
        o + _p_o_0[:, None] * (H * V) + _p_o_1[None, :] * (1),
        b_o.to(o.dtype.element_ty),
        mask=(_p_o_0[:, None] < (T)) & (_p_o_1[None, :] < (V)),
    )


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    o: torch.Tensor | None = None,
    *,
    head_major_vk: bool = False,
    chunk_indices: torch.Tensor | None = None,
    plan=None,
    gate=None,
) -> torch.Tensor:
    """Returns the attention output tensor.

    If ``o`` is provided, the kernel writes into it inplace and ``o`` is
    returned. The caller's buffer MUST match ``v``'s shape and dtype and
    be contiguous — the Triton kernel assumes stride ``(H * V, 1)`` along
    ``(T, V)`` for a ``[B, T, H, V]`` layout. The public chunk_gated_delta_rule
    entry point asserts these contracts before .apply() (so input_guard's
    silent .contiguous() clone can't defeat them); we re-assert here as a
    defense-in-depth backstop for any caller that bypasses the public API.

    ``gate`` = (z, norm weight, eps): ``o`` receives the layer's gated RMSNorm
    ``norm(o) * weight * silu(z)`` (groups of V, z ``[B, T, H, V]`` or
    ``[T, H, V]`` with contiguous ``[H, V]`` rows) instead of the raw output.
    """
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[1] if head_major_vk else v.shape[-2]
    BT = 64 if FLA_GDN_FIX_BT else min(chunk_size, max(16, triton.next_power_of_2(T)))
    if head_major_vk:
        if B != 1:
            raise ValueError("head-major VK output currently requires B=1")
        BT = chunk_size
    nt_ptr = None
    if plan is not None:
        chunk_indices, NT, nt_ptr = plan.chunk_indices_64, plan.nt64_max, plan.nt64
    else:
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if o is None:
        o = (
            torch.empty((B, T, H, V), dtype=v.dtype, device=v.device)
            if head_major_vk
            else torch.empty_like(v)
        )
    else:
        assert o.shape == (B, T, H, V), (
            f"chunk_fwd_o: caller-provided o.shape {tuple(o.shape)} != "
            f"v.shape {tuple(v.shape)}"
        )
        assert (
            o.dtype == v.dtype
        ), f"chunk_fwd_o: caller-provided o.dtype {o.dtype} != v.dtype {v.dtype}"
        assert o.is_contiguous(), (
            "chunk_fwd_o: caller-provided o must be contiguous (kernel "
            "assumes stride (H*V, 1) on the (T, V) plane)"
        )

    if (
        not head_major_vk
        and _use_gfx1250_kernel(q.device.index)
        and K == 128
        and V == 128
        and BT == 64
        and q.dtype == torch.bfloat16
    ):
        from .chunk_o_gfx1250 import chunk_o_gfx1250

        return chunk_o_gfx1250(
            q, k, v, h, g, o, scale, cu_seqlens, chunk_indices, nt_ptr, NT, T, B, H,
            Hg, K, V, BT, gate=gate,
        )

    chunk_fwd_kernel_o[(triton.cdiv(V, _TRITON_BV), NT, B * H)](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        nt_ptr,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=_TRITON_BK,
        BV=_TRITON_BV,
        HEAD_MAJOR_VK=head_major_vk,
        USE_EXP2=head_major_vk,
        num_warps=_TRITON_NUM_WARPS,
        num_stages=_TRITON_NUM_STAGES,
    )
    if gate is not None:
        from .layernorm_guard import layer_norm_fwd

        z, weight, eps = gate
        rows = o.view(-1, V)
        layer_norm_fwd(
            rows,
            weight,
            None,
            eps,
            z=z.reshape(-1, V),
            out=rows,
            norm_before_gate=True,
            is_rms_norm=True,
        )
    return o
