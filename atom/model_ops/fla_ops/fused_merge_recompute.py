# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=4),  # swept-best on MI308
    ],
    key=["H", "Hg", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _fused_merge_recompute_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    g_cumsum_ptr,
    A_ptr,
    Ai16_ptr,
    w_ptr,
    u_ptr,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
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

    Ai16_base = Ai16_ptr + (bos * H + i_h) * 16

    _p_Ai11_0 = (i_t * 64) + tl.arange(0, 16)
    _p_Ai11_1 = (0) + tl.arange(0, 16)
    b_Ai11 = tl.load(
        Ai16_base + _p_Ai11_0[:, None] * (H * 16) + _p_Ai11_1[None, :] * (1),
        mask=(_p_Ai11_0[:, None] < (T_seq)) & (_p_Ai11_1[None, :] < (16)),
        other=0.0,
    ).to(tl.float32)
    _p_Ai22_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_Ai22_1 = (0) + tl.arange(0, 16)
    b_Ai22 = tl.load(
        Ai16_base + _p_Ai22_0[:, None] * (H * 16) + _p_Ai22_1[None, :] * (1),
        mask=(_p_Ai22_0[:, None] < (T_seq)) & (_p_Ai22_1[None, :] < (16)),
        other=0.0,
    ).to(tl.float32)
    _p_Ai33_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_Ai33_1 = (0) + tl.arange(0, 16)
    b_Ai33 = tl.load(
        Ai16_base + _p_Ai33_0[:, None] * (H * 16) + _p_Ai33_1[None, :] * (1),
        mask=(_p_Ai33_0[:, None] < (T_seq)) & (_p_Ai33_1[None, :] < (16)),
        other=0.0,
    ).to(tl.float32)
    _p_Ai44_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_Ai44_1 = (0) + tl.arange(0, 16)
    b_Ai44 = tl.load(
        Ai16_base + _p_Ai44_0[:, None] * (H * 16) + _p_Ai44_1[None, :] * (1),
        mask=(_p_Ai44_0[:, None] < (T_seq)) & (_p_Ai44_1[None, :] < (16)),
        other=0.0,
    ).to(tl.float32)

    A_base = A_ptr + (bos * H + i_h) * BT

    _p_A21_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_A21_1 = (0) + tl.arange(0, 16)
    b_A21 = tl.load(
        A_base + _p_A21_0[:, None] * (H * BT) + _p_A21_1[None, :] * (1),
        mask=(_p_A21_0[:, None] < (T_seq)) & (_p_A21_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)
    _p_A31_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_A31_1 = (0) + tl.arange(0, 16)
    b_A31 = tl.load(
        A_base + _p_A31_0[:, None] * (H * BT) + _p_A31_1[None, :] * (1),
        mask=(_p_A31_0[:, None] < (T_seq)) & (_p_A31_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)
    _p_A32_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_A32_1 = (16) + tl.arange(0, 16)
    b_A32 = tl.load(
        A_base + _p_A32_0[:, None] * (H * BT) + _p_A32_1[None, :] * (1),
        mask=(_p_A32_0[:, None] < (T_seq)) & (_p_A32_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)
    _p_A41_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_A41_1 = (0) + tl.arange(0, 16)
    b_A41 = tl.load(
        A_base + _p_A41_0[:, None] * (H * BT) + _p_A41_1[None, :] * (1),
        mask=(_p_A41_0[:, None] < (T_seq)) & (_p_A41_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)
    _p_A42_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_A42_1 = (16) + tl.arange(0, 16)
    b_A42 = tl.load(
        A_base + _p_A42_0[:, None] * (H * BT) + _p_A42_1[None, :] * (1),
        mask=(_p_A42_0[:, None] < (T_seq)) & (_p_A42_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)
    _p_A43_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_A43_1 = (32) + tl.arange(0, 16)
    b_A43 = tl.load(
        A_base + _p_A43_0[:, None] * (H * BT) + _p_A43_1[None, :] * (1),
        mask=(_p_A43_0[:, None] < (T_seq)) & (_p_A43_1[None, :] < (BT)),
        other=0.0,
    ).to(tl.float32)

    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision="ieee"), b_Ai11, input_precision="ieee"
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision="ieee"), b_Ai22, input_precision="ieee"
    )
    b_Ai43 = -tl.dot(
        tl.dot(b_Ai44, b_A43, input_precision="ieee"), b_Ai33, input_precision="ieee"
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision="ieee")
        + tl.dot(b_A32, b_Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai42 = -tl.dot(
        b_Ai44,
        tl.dot(b_A42, b_Ai22, input_precision="ieee")
        + tl.dot(b_A43, b_Ai32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai41 = -tl.dot(
        b_Ai44,
        tl.dot(b_A41, b_Ai11, input_precision="ieee")
        + tl.dot(b_A42, b_Ai21, input_precision="ieee")
        + tl.dot(b_A43, b_Ai31, input_precision="ieee"),
        input_precision="ieee",
    )

    k_base = k_ptr + (bos * Hg + i_h // (H // Hg)) * K
    beta_base = beta_ptr + bos * H + i_h
    g_base = g_cumsum_ptr + bos * H + i_h

    _p_k1_0 = (i_t * 64) + tl.arange(0, 16)
    _p_k1_1 = (0) + tl.arange(0, K)
    b_k1 = tl.load(
        k_base + _p_k1_0[:, None] * (Hg * K) + _p_k1_1[None, :] * (1),
        mask=(_p_k1_0[:, None] < (T_seq)) & (_p_k1_1[None, :] < (K)),
        other=0.0,
    ).to(tl.float32)
    _p_k2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_k2_1 = (0) + tl.arange(0, K)
    b_k2 = tl.load(
        k_base + _p_k2_0[:, None] * (Hg * K) + _p_k2_1[None, :] * (1),
        mask=(_p_k2_0[:, None] < (T_seq)) & (_p_k2_1[None, :] < (K)),
        other=0.0,
    ).to(tl.float32)
    _p_k3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_k3_1 = (0) + tl.arange(0, K)
    b_k3 = tl.load(
        k_base + _p_k3_0[:, None] * (Hg * K) + _p_k3_1[None, :] * (1),
        mask=(_p_k3_0[:, None] < (T_seq)) & (_p_k3_1[None, :] < (K)),
        other=0.0,
    ).to(tl.float32)
    _p_k4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_k4_1 = (0) + tl.arange(0, K)
    b_k4 = tl.load(
        k_base + _p_k4_0[:, None] * (Hg * K) + _p_k4_1[None, :] * (1),
        mask=(_p_k4_0[:, None] < (T_seq)) & (_p_k4_1[None, :] < (K)),
        other=0.0,
    ).to(tl.float32)

    _p_beta1_0 = (i_t * 64) + tl.arange(0, 16)
    b_beta1 = tl.load(
        beta_base + _p_beta1_0 * (H), mask=(_p_beta1_0 < (T_seq)), other=0.0
    ).to(tl.float32)
    _p_beta2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    b_beta2 = tl.load(
        beta_base + _p_beta2_0 * (H), mask=(_p_beta2_0 < (T_seq)), other=0.0
    ).to(tl.float32)
    _p_beta3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    b_beta3 = tl.load(
        beta_base + _p_beta3_0 * (H), mask=(_p_beta3_0 < (T_seq)), other=0.0
    ).to(tl.float32)
    _p_beta4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    b_beta4 = tl.load(
        beta_base + _p_beta4_0 * (H), mask=(_p_beta4_0 < (T_seq)), other=0.0
    ).to(tl.float32)

    _p_g1_0 = (i_t * 64) + tl.arange(0, 16)
    b_g1 = tl.exp(
        tl.load(g_base + _p_g1_0 * (H), mask=(_p_g1_0 < (T_seq)), other=0.0).to(
            tl.float32
        )
    )
    _p_g2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    b_g2 = tl.exp(
        tl.load(g_base + _p_g2_0 * (H), mask=(_p_g2_0 < (T_seq)), other=0.0).to(
            tl.float32
        )
    )
    _p_g3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    b_g3 = tl.exp(
        tl.load(g_base + _p_g3_0 * (H), mask=(_p_g3_0 < (T_seq)), other=0.0).to(
            tl.float32
        )
    )
    _p_g4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    b_g4 = tl.exp(
        tl.load(g_base + _p_g4_0 * (H), mask=(_p_g4_0 < (T_seq)), other=0.0).to(
            tl.float32
        )
    )

    b_rhs_w1 = b_k1 * b_beta1[:, None] * b_g1[:, None]
    b_rhs_w2 = b_k2 * b_beta2[:, None] * b_g2[:, None]
    b_rhs_w3 = b_k3 * b_beta3[:, None] * b_g3[:, None]
    b_rhs_w4 = b_k4 * b_beta4[:, None] * b_g4[:, None]

    b_w1 = tl.dot(b_Ai11, b_rhs_w1, input_precision="ieee")
    b_w2 = tl.dot(b_Ai21, b_rhs_w1, input_precision="ieee") + tl.dot(
        b_Ai22, b_rhs_w2, input_precision="ieee"
    )
    b_w3 = (
        tl.dot(b_Ai31, b_rhs_w1, input_precision="ieee")
        + tl.dot(b_Ai32, b_rhs_w2, input_precision="ieee")
        + tl.dot(b_Ai33, b_rhs_w3, input_precision="ieee")
    )
    b_w4 = (
        tl.dot(b_Ai41, b_rhs_w1, input_precision="ieee")
        + tl.dot(b_Ai42, b_rhs_w2, input_precision="ieee")
        + tl.dot(b_Ai43, b_rhs_w3, input_precision="ieee")
        + tl.dot(b_Ai44, b_rhs_w4, input_precision="ieee")
    )

    w_base = w_ptr + (bos * H + i_h) * K
    _p_w1_0 = (i_t * 64) + tl.arange(0, 16)
    _p_w1_1 = (0) + tl.arange(0, K)
    tl.store(
        w_base + _p_w1_0[:, None] * (H * K) + _p_w1_1[None, :] * (1),
        b_w1.to(w_ptr.dtype.element_ty),
        mask=(_p_w1_0[:, None] < (T_seq)) & (_p_w1_1[None, :] < (K)),
    )
    _p_w2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_w2_1 = (0) + tl.arange(0, K)
    tl.store(
        w_base + _p_w2_0[:, None] * (H * K) + _p_w2_1[None, :] * (1),
        b_w2.to(w_ptr.dtype.element_ty),
        mask=(_p_w2_0[:, None] < (T_seq)) & (_p_w2_1[None, :] < (K)),
    )
    _p_w3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_w3_1 = (0) + tl.arange(0, K)
    tl.store(
        w_base + _p_w3_0[:, None] * (H * K) + _p_w3_1[None, :] * (1),
        b_w3.to(w_ptr.dtype.element_ty),
        mask=(_p_w3_0[:, None] < (T_seq)) & (_p_w3_1[None, :] < (K)),
    )
    _p_w4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_w4_1 = (0) + tl.arange(0, K)
    tl.store(
        w_base + _p_w4_0[:, None] * (H * K) + _p_w4_1[None, :] * (1),
        b_w4.to(w_ptr.dtype.element_ty),
        mask=(_p_w4_0[:, None] < (T_seq)) & (_p_w4_1[None, :] < (K)),
    )

    v_base = v_ptr + (bos * H + i_h) * V

    _p_v1_0 = (i_t * 64) + tl.arange(0, 16)
    _p_v1_1 = (0) + tl.arange(0, V)
    b_v1 = tl.load(
        v_base + _p_v1_0[:, None] * (H * V) + _p_v1_1[None, :] * (1),
        mask=(_p_v1_0[:, None] < (T_seq)) & (_p_v1_1[None, :] < (V)),
        other=0.0,
    ).to(tl.float32)
    _p_v2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_v2_1 = (0) + tl.arange(0, V)
    b_v2 = tl.load(
        v_base + _p_v2_0[:, None] * (H * V) + _p_v2_1[None, :] * (1),
        mask=(_p_v2_0[:, None] < (T_seq)) & (_p_v2_1[None, :] < (V)),
        other=0.0,
    ).to(tl.float32)
    _p_v3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_v3_1 = (0) + tl.arange(0, V)
    b_v3 = tl.load(
        v_base + _p_v3_0[:, None] * (H * V) + _p_v3_1[None, :] * (1),
        mask=(_p_v3_0[:, None] < (T_seq)) & (_p_v3_1[None, :] < (V)),
        other=0.0,
    ).to(tl.float32)
    _p_v4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_v4_1 = (0) + tl.arange(0, V)
    b_v4 = tl.load(
        v_base + _p_v4_0[:, None] * (H * V) + _p_v4_1[None, :] * (1),
        mask=(_p_v4_0[:, None] < (T_seq)) & (_p_v4_1[None, :] < (V)),
        other=0.0,
    ).to(tl.float32)

    b_rhs_u1 = b_v1 * b_beta1[:, None]
    b_rhs_u2 = b_v2 * b_beta2[:, None]
    b_rhs_u3 = b_v3 * b_beta3[:, None]
    b_rhs_u4 = b_v4 * b_beta4[:, None]

    b_u1 = tl.dot(b_Ai11, b_rhs_u1, input_precision="ieee")
    b_u2 = tl.dot(b_Ai21, b_rhs_u1, input_precision="ieee") + tl.dot(
        b_Ai22, b_rhs_u2, input_precision="ieee"
    )
    b_u3 = (
        tl.dot(b_Ai31, b_rhs_u1, input_precision="ieee")
        + tl.dot(b_Ai32, b_rhs_u2, input_precision="ieee")
        + tl.dot(b_Ai33, b_rhs_u3, input_precision="ieee")
    )
    b_u4 = (
        tl.dot(b_Ai41, b_rhs_u1, input_precision="ieee")
        + tl.dot(b_Ai42, b_rhs_u2, input_precision="ieee")
        + tl.dot(b_Ai43, b_rhs_u3, input_precision="ieee")
        + tl.dot(b_Ai44, b_rhs_u4, input_precision="ieee")
    )

    u_base = u_ptr + (bos * H + i_h) * V
    _p_u1_0 = (i_t * 64) + tl.arange(0, 16)
    _p_u1_1 = (0) + tl.arange(0, V)
    tl.store(
        u_base + _p_u1_0[:, None] * (H * V) + _p_u1_1[None, :] * (1),
        b_u1.to(u_ptr.dtype.element_ty),
        mask=(_p_u1_0[:, None] < (T_seq)) & (_p_u1_1[None, :] < (V)),
    )
    _p_u2_0 = (i_t * 64 + 16) + tl.arange(0, 16)
    _p_u2_1 = (0) + tl.arange(0, V)
    tl.store(
        u_base + _p_u2_0[:, None] * (H * V) + _p_u2_1[None, :] * (1),
        b_u2.to(u_ptr.dtype.element_ty),
        mask=(_p_u2_0[:, None] < (T_seq)) & (_p_u2_1[None, :] < (V)),
    )
    _p_u3_0 = (i_t * 64 + 32) + tl.arange(0, 16)
    _p_u3_1 = (0) + tl.arange(0, V)
    tl.store(
        u_base + _p_u3_0[:, None] * (H * V) + _p_u3_1[None, :] * (1),
        b_u3.to(u_ptr.dtype.element_ty),
        mask=(_p_u3_0[:, None] < (T_seq)) & (_p_u3_1[None, :] < (V)),
    )
    _p_u4_0 = (i_t * 64 + 48) + tl.arange(0, 16)
    _p_u4_1 = (0) + tl.arange(0, V)
    tl.store(
        u_base + _p_u4_0[:, None] * (H * V) + _p_u4_1[None, :] * (1),
        b_u4.to(u_ptr.dtype.element_ty),
        mask=(_p_u4_0[:, None] < (T_seq)) & (_p_u4_1[None, :] < (V)),
    )


def fused_merge_recompute(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    Ai16: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused merge + recompute.

    Args:
        k: [B, T, Hg, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g_cumsum: [B, T, H]
        A: [B, T, H, chunk_size]
        Ai16: [B, T, H, 16], diagonal block inverses from solve_tril_16x16

    Returns:
        w: [B, T, H, K]
        u: [B, T, H, V]
    """
    B, T, H = g_cumsum.shape
    Hg, K = k.shape[2], k.shape[3]
    V = v.shape[3]

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)

    w = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)

    _fused_merge_recompute_kernel[(NT, B * H)](
        k,
        v,
        beta,
        g_cumsum,
        A,
        Ai16,
        w,
        u,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        Hg,
        K,
        V,
        chunk_size,
        # num_warps/num_stages selected by autotune
    )

    return w, u
