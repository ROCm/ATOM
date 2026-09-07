# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

import triton
import triton.language as tl

from .index import prepare_chunk_indices
from .op import exp


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    _p_beta_0 = (i_t * BT) + tl.arange(0, BT)
    b_beta = tl.load(
        beta + bos * H + i_h + _p_beta_0 * (H), mask=(_p_beta_0 < (T)), other=0.0
    )

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        _p_k_0 = (i_t * BT) + tl.arange(0, BT)
        _p_k_1 = (i_k * BK) + tl.arange(0, BK)
        b_k = tl.load(
            k
            + (bos * Hg + i_h // (H // Hg)) * K
            + _p_k_0[:, None] * (Hg * K)
            + _p_k_1[None, :] * (1),
            mask=(_p_k_0[:, None] < (T)) & (_p_k_1[None, :] < (K)),
            other=0.0,
        )
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    if USE_G:
        _p_g_0 = (i_t * BT) + tl.arange(0, BT)
        b_g = tl.load(g + bos * H + i_h + _p_g_0 * (H), mask=(_p_g_0 < (T)), other=0.0)
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    _p_A_0 = (i_t * BT) + tl.arange(0, BT)
    _p_A_1 = (0) + tl.arange(0, BT)
    tl.store(
        A + (bos * H + i_h) * BT + _p_A_0[:, None] * (BT * H) + _p_A_1[None, :] * (1),
        b_A.to(A.dtype.element_ty),
        mask=(_p_A_0[:, None] < (T)) & (_p_A_1[None, :] < (BT)),
    )


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
    )
    return A
