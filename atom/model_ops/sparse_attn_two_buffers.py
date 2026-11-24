# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`sparse_attn` over two key buffers read where they lie (no concatenated copy).

`sparse_attn_two_buffers` of ROCm/ATOM#3645, in a module of its own so that no
kernel of `sparse_attn_v4.py` moves (Triton keys a kernel by its source and its
line number). Used only behind ATOM_V41_DRAFT_ATTN_NO_CAT
(`atom.spec_decode.dspark_tail`).
"""

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_tile(q, kv, valid, h_mask, m_i, l_i, acc, softmax_scale):
    """One key tile of `_sparse_attn_triton_kernel`'s online softmax, in its
    operations and their order."""
    neg_large = -3.4028234663852886e38
    scores = tl.dot(q, tl.trans(kv)) * softmax_scale
    scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

    m_block = tl.max(scores, axis=1)
    m_new = tl.maximum(m_i, m_block)
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(scores - m_new[:, None])
    p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
    l_new = l_i * alpha + tl.sum(p, axis=1)

    acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
    return m_new, l_new, acc


@triton.jit
def _sparse_attn_2buf_triton_kernel(
    q_ptr,
    kv_ptr,
    kv2_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_b: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    kv2_stride_b: tl.constexpr,
    kv2_stride_n: tl.constexpr,
    kv2_stride_d: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_m: tl.constexpr,
    topk_stride_k: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    N1: tl.constexpr,
    K_FIRST: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`_sparse_attn_triton_kernel` over `cat((kv, kv2), 1)` without the copy:
    index `i < N1` is row `i` of `kv`, index `i >= N1` row `i - N1` of `kv2`.

    The first `K_FIRST` list positions (whole key tiles) hold only indices of
    `kv` or -1 -- the caller's contract -- so those tiles load one buffer; the
    tiles after them load both and select by the index. Tiles, arithmetic and
    order are that kernel's, so the output is its output bit for bit."""
    pid_bm = tl.program_id(0)
    pid_h = tl.program_id(1)
    m = pid_bm % M
    b = pid_bm // M

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q_base = b * q_stride_b + m * q_stride_m
    q = tl.load(
        q_ptr + q_base + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    idx_base = topk_idxs_ptr + b * topk_stride_b + m * topk_stride_m
    for k_start in tl.range(0, K_FIRST, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < K
        idx = tl.load(idx_base + k_pos * topk_stride_k, mask=in_range, other=-1)
        valid = in_range & (idx >= 0)
        kv = tl.load(
            kv_ptr
            + b * kv_stride_b
            + idx[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        m_i, l_i, acc = _sparse_attn_tile(
            q, kv, valid, h_mask, m_i, l_i, acc, softmax_scale
        )
    for k_start in tl.range(K_FIRST, K, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < K
        idx = tl.load(idx_base + k_pos * topk_stride_k, mask=in_range, other=-1)
        valid = in_range & (idx >= 0)
        first = idx < N1
        kv_first = tl.load(
            kv_ptr
            + b * kv_stride_b
            + idx[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=(valid & first)[:, None] & d_mask[None, :],
            other=0.0,
        )
        kv_second = tl.load(
            kv2_ptr
            + b * kv2_stride_b
            + (idx - N1)[:, None] * kv2_stride_n
            + d_offs[None, :] * kv2_stride_d,
            mask=(valid & (idx >= N1))[:, None] & d_mask[None, :],
            other=0.0,
        )
        kv = tl.where(first[:, None], kv_first, kv_second)
        m_i, l_i, acc = _sparse_attn_tile(
            q, kv, valid, h_mask, m_i, l_i, acc, softmax_scale
        )

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    out_base = b * out_stride_b + m * out_stride_m
    tl.store(
        out_ptr
        + out_base
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


# Waves a program of the two-buffer kernel; `_sparse_attn_triton` launches 8.
# The wave count changes no bit of the output. MI355X (256 CUs), 64 heads,
# 133 + 5 keys, in a CUDA graph: the 8-wave kernel takes 15.8 / 16.0 / 16.3 /
# 29.0 us at 25 / 40 / 50 / 65 rows (260 programs no longer get a CU each),
# with 4 waves 13.9 / 14.0 / 14.4 / 18.1 us.
_TWO_BUFFER_WARPS = 4


@functools.lru_cache
def _cu_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def sparse_attn_two_buffers(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv2: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    *,
    first_only: int = 0,
    num_warps: int = _TWO_BUFFER_WARPS,
) -> torch.Tensor:
    """`sparse_attn(q, cat((kv, kv2), 1), ...)` without the concatenated copy.

    q `[B, M, H, D]`, kv `[B, N1, D]`, kv2 `[B, N2, D]`, topk_idxs `[B, M, K]`
    indexing the `N1 + N2` rows of the virtual concatenation, -1 skipped.
    `first_only`: how many leading list positions the caller guarantees to
    hold only rows of `kv` (or -1); 0 makes no assumption. The result is
    `_sparse_attn_triton`'s on the concatenation, bit for bit.
    """
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    assert kv.shape[0] == B and kv2.shape[0] == B, "batch mismatch"
    assert kv.shape[-1] == D and kv2.shape[-1] == D, "head_dim mismatch"
    assert kv.dtype == kv2.dtype == q.dtype, "the two sources share the query dtype"
    assert 0 <= first_only <= K
    out = torch.empty_like(q)
    topk_idxs = topk_idxs.to(torch.int32)
    block_h = 16
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    # The one-buffer tiles are worth their second loop only while every
    # program has a CU to itself. Past that the single-loop form (every tile
    # selects) shares a CU better: MI355X, 65 rows = 260 programs, 21.8 us
    # against 28.4 us; at 25-60 rows it is the slower one, 18.0-18.6 us
    # against 15.2-15.9 us. Either way the bytes out are the same.
    programs = B * M * triton.cdiv(H, block_h)
    k_first = (first_only // block_k) * block_k
    if programs > _cu_count(q.device):
        k_first = 0
    _sparse_attn_2buf_triton_kernel[(B * M, triton.cdiv(H, block_h))](
        q,
        kv,
        kv2,
        attn_sink,
        topk_idxs,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        kv2.stride(0),
        kv2.stride(1),
        kv2.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        M,
        H,
        D,
        K,
        kv.shape[1],
        k_first,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out
