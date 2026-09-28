# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=4),  # swept-best on MI308
    ],
    key=["H", "Hg", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _fused_cumsum_kkt_kernel(
    g_ptr,
    k_ptr,
    beta_ptr,
    g_cumsum_ptr,
    A_ptr,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_seq = eos - bos
        i_t = i_t_local
    else:
        bos = i_b * T
        T_seq = T

    o_t = tl.arange(0, BT)

    _p_g_0 = (i_t * BT) + tl.arange(0, BT)
    b_g = tl.load(
        g_ptr + bos * H + i_h + _p_g_0 * (H), mask=(_p_g_0 < (T_seq)), other=0.0
    ).to(tl.float32)
    b_g_cumsum = tl.cumsum(b_g, axis=0)
    _p_g_out_0 = (i_t * BT) + tl.arange(0, BT)
    tl.store(
        g_cumsum_ptr + bos * H + i_h + _p_g_out_0 * (H),
        b_g_cumsum.to(g_cumsum_ptr.dtype.element_ty),
        mask=(_p_g_out_0 < (T_seq)),
    )

    _p_beta_0 = (i_t * BT) + tl.arange(0, BT)
    b_beta = tl.load(
        beta_ptr + bos * H + i_h + _p_beta_0 * (H),
        mask=(_p_beta_0 < (T_seq)),
        other=0.0,
    ).to(tl.float32)

    _p_k_0 = (i_t * BT) + tl.arange(0, BT)
    _p_k_1 = (0) + tl.arange(0, K)
    b_k = tl.load(
        k_ptr
        + (bos * Hg + i_h // (H // Hg)) * K
        + _p_k_0[:, None] * (Hg * K)
        + _p_k_1[None, :] * (1),
        mask=(_p_k_0[:, None] < (T_seq)) & (_p_k_1[None, :] < (K)),
        other=0.0,
    ).to(tl.float32)

    b_A = tl.dot(b_k, tl.trans(b_k))
    b_g_diff = b_g_cumsum[:, None] - b_g_cumsum[None, :]
    b_A = b_A * safe_exp(b_g_diff) * b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0.0)

    _p_A_0 = (i_t * BT) + tl.arange(0, BT)
    _p_A_1 = (0) + tl.arange(0, BT)
    tl.store(
        A_ptr
        + (bos * H + i_h) * BT
        + _p_A_0[:, None] * (BT * H)
        + _p_A_1[None, :] * (1),
        b_A.to(A_ptr.dtype.element_ty),
        mask=(_p_A_0[:, None] < (T_seq)) & (_p_A_1[None, :] < (BT)),
    )


def fused_cumsum_kkt(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused cumsum + KKT.

    Args:
        g: [B, T, H]
        k: [B, T, Hg, K]
        beta: [B, T, H]

    Returns:
        g_cumsum: [B, T, H]
        A: [B, T, H, chunk_size], strictly lower triangular
    """
    B, T, H = g.shape
    Hg, K = k.shape[2], k.shape[3]

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)

    g_cumsum = torch.empty(B, T, H, device=g.device, dtype=torch.float32)
    A = torch.empty(B, T, H, chunk_size, device=k.device, dtype=torch.float32)

    _fused_cumsum_kkt_kernel[(NT, B * H)](
        g,
        k,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        Hg,
        K,
        chunk_size,
        # num_warps/num_stages selected by autotune
    )
    return g_cumsum, A
