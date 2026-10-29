# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gates and the strictly lower triangular A of every 64-token chunk.

One program per chunk and key head: k and k.k^T once for the key head's value
heads. The gates come in as [64, H] tiles of the [T, H] tensors, every row one
contiguous run, rather than one strided word per token and head; the key head 0
program stores g_cumsum the same way. For `fused_inverse_recompute` it also
stores beta and beta exp(g_cumsum) head-major, so that kernel reads each as one
contiguous run.

The launch configuration is fixed: no autotuning.
"""

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NT": lambda args: args["nt_ptr"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _fused_cumsum_kkt_kernel(
    g_ptr,
    k_ptr,
    beta_ptr,
    g_cumsum_ptr,
    A_ptr,
    beta_hm_ptr,
    scale_hm_ptr,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    HP: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NT: tl.constexpr,
):
    i_t, i_bk = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN and HAS_NT:
        # a bucket-shaped grid: programs past the step's chunk count have no work
        if i_t >= tl.load(nt_ptr):
            return
    i_b, i_hk = i_bk // Hg, i_bk % Hg
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T_seq = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    else:
        bos = i_b * T
        T_seq = T

    o = tl.arange(0, BT)
    r = i_t * BT + o
    m = r < T_seq
    hh = tl.arange(0, HP)
    m_tile = m[:, None] & (hh < H)[None, :]
    tile_off = (bos + r[:, None]) * H + hh[None, :]
    gc_tile = tl.cumsum(
        tl.load(g_ptr + tile_off, mask=m_tile, other=0.0).to(tl.float32), axis=0
    )
    if i_hk == 0:
        tl.store(
            g_cumsum_ptr + tile_off,
            gc_tile.to(g_cumsum_ptr.dtype.element_ty),
            mask=m_tile,
        )
    beta_tile = tl.load(beta_ptr + tile_off, mask=m_tile, other=0.0).to(tl.float32)

    b_k = tl.load(
        k_ptr + ((bos + r[:, None]) * Hg + i_hk) * K + tl.arange(0, K)[None, :],
        mask=m[:, None],
        other=0.0,
    )
    # k is bf16, so its products are exact in fp32: bf16 WMMA loses nothing. k.k^T
    # is symmetric (every entry the same products in the same order), and its
    # transpose holds the WMMA result with each lane's values along a row, so the
    # A stores below are contiguous instead of one word per 8 KB row.
    b_kk = tl.trans(tl.dot(b_k, tl.trans(b_k)))
    lower = o[:, None] > o[None, :]
    for j in tl.static_range(H // Hg):
        i_h = i_hk * (H // Hg) + j
        pick = (hh == i_h)[None, :]
        b_gc = tl.sum(tl.where(pick, gc_tile, 0.0), axis=1)
        b_beta = tl.sum(tl.where(pick, beta_tile, 0.0), axis=1)
        b_A = tl.where(
            lower,
            b_kk
            * tl.exp(tl.where(lower, b_gc[:, None] - b_gc[None, :], 0.0))
            * b_beta[:, None],
            0.0,
        )
        tl.store(
            A_ptr + ((bos + r[:, None]) * H + i_h) * BT + o[None, :],
            b_A.to(A_ptr.dtype.element_ty),
            mask=m[:, None],
        )
        hm = (i_b * H + i_h) * T + (bos - i_b * T) + r
        tl.store(beta_hm_ptr + hm, b_beta, mask=m)
        tl.store(scale_hm_ptr + hm, b_beta * tl.exp(b_gc), mask=m)


def fused_cumsum_kkt(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
    plan=None,
):
    """
    Fused cumsum + KKT. ``plan`` (`varlen_plan.VarlenPlan`) launches the
    bucket's grid and lets the kernel read the step's chunk count.

    Args:
        g: [B, T, H]
        k: [B, T, Hg, K]
        beta: [B, T, H]

    Returns:
        g_cumsum: [B, T, H] fp32
        A: [B, T, H, chunk_size] fp32, strictly lower triangular
        beta_hm, scale_hm: [B, H, T] fp32, beta and beta * exp(g_cumsum) head-major
    """
    B, T, H = g.shape
    Hg, K = k.shape[2], k.shape[3]

    nt_ptr = None
    if plan is not None:
        chunk_indices, NT, nt_ptr = plan.chunk_indices_64, plan.nt64_max, plan.nt64
    elif cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)

    g_cumsum = torch.empty(B, T, H, device=g.device, dtype=torch.float32)
    A = torch.empty(B, T, H, chunk_size, device=k.device, dtype=torch.float32)
    beta_hm = torch.empty(B, H, T, device=g.device, dtype=torch.float32)
    scale_hm = torch.empty(B, H, T, device=g.device, dtype=torch.float32)

    # gfx1250 sweep (T = 16k, 30 sequences, 16 key heads; cold caches): 2 warps
    # 24.4 us, 4 warps 28.2 us, 8 warps 30.8 us
    _fused_cumsum_kkt_kernel[(NT, B * Hg)](
        g,
        k,
        beta,
        g_cumsum,
        A,
        beta_hm,
        scale_hm,
        cu_seqlens,
        chunk_indices,
        nt_ptr,
        T,
        H=H,
        Hg=Hg,
        HP=triton.next_power_of_2(H),
        K=K,
        BT=chunk_size,
        num_warps=2,
        num_stages=1,
    )
    return g_cumsum, A, beta_hm, scale_hm
