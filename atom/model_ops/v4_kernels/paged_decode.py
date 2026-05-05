# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse decode attention over a unified KV pool with per-token paged indices.

Designed for V4 decode + CUDAGraph: replaces the per-fwd `kv_flat_sa`
materialization (whose shape depends on `n_committed_per_seq` → varies per
fwd → blocks CG capture) with a single unified KV pool indexed via paged
indices, mirroring `aiter.mla.mla_decode_fwd`'s API style.

Caller contract:
  unified_kv:       [total_pages, D] BF16  (page_size=1)
    Conceptually merges the SWA ring buffer and the compressor paged cache
    of a single V4 layer. Slots in `[0, swa_pages)` reference SWA entries
    (state_slot * win + ring); slots in `[swa_pages, ...)` reference
    compressed-K entries (block_id * K_PER_BLOCK + slot_in_block).
  kv_indices: [total_indices] int32 — per-token slot lists, flat.
    Per-token entries live in
    `kv_indices[kv_indptr[t] : kv_indptr[t+1]]`.
    `-1` entries are skipped (sentinel for unused tail).
  kv_indptr:  [N+1] int32 — true prefix sum (variable per-token len).
  attn_sink:        [H] per-head learnable softmax-denom bias (V4 specific).
  softmax_scale:    float.

Per-token K loop iterates exactly `kv_indptr[t+1] - kv_indptr[t]`
slots (rounded up to BLOCK_K granularity for the partial last block) — short
seqs do not pay for long-seq worst-case work. CUDAGraph-compatible: the trip
count is loaded from `kv_indptr` at replay, kernel binary stays
unchanged across batches.

Returns:
  out: [N, H, D] same dtype as q.

Numerics: standard online-softmax with attention sink as a virtual K.
Bit-exact against the PyTorch reference (`_sparse_attn_ragged_torch`)
when invoked with the same per-token gather indices.
"""

import os

import torch
import triton
import triton.language as tl

from atom.model_ops.sparse_attn_v4 import _sparse_attn_ragged_torch


@triton.jit
def _sparse_attn_v4_paged_decode_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D]
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # Runtime trip count: each token iterates ceil(kv_len / BLOCK_K) blocks —
    # short seqs do not pay for long-seq worst-case work. The last block needs
    # `k_pos < kv_len` masking for the partial tail.
    k_offs = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)

        kv = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_v4_paged_decode_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_decode requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_decode expects fp16/bf16 q, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )

    T, H, D = q.shape
    out = torch.empty_like(q)
    kv_indices = kv_indices.to(torch.int32).contiguous()
    kv_indptr = kv_indptr.to(torch.int32).contiguous()

    block_h = 16  # AMD MFMA min tile
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    _sparse_attn_v4_paged_decode_kernel[(T, triton.cdiv(H, block_h))](
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def sparse_attn_v4_paged_decode_reference(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure-torch reference. Materialises per-token KV via gather and reuses
    `_sparse_attn_ragged_torch`. Slow but correct — for unit tests / dump-bisect.

    Uses the longest per-token span as the K dimension for the dense
    `topk_idxs` tensor; shorter spans are tail-padded with `-1`.
    """
    T = q.size(0)
    indptr = kv_indptr.to(torch.int64)
    spans = (indptr[1:] - indptr[:T]).clamp(min=0)
    k_dim = int(spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1
    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        s = int(indptr[t].item())
        n = int(spans[t].item())
        if n > 0:
            topk_idxs[t, :n] = kv_indices[s : s + n].to(torch.int32)
    return _sparse_attn_ragged_torch(q, unified_kv, attn_sink, topk_idxs, softmax_scale)


def sparse_attn_v4_paged_decode(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """V4 decode sparse attention over a unified KV pool with paged indices."""
    if os.environ.get("ATOM_USE_TRITON_ATTN", "1") == "1":
        return _sparse_attn_v4_paged_decode_triton(
            q,
            unified_kv,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
        )
    return sparse_attn_v4_paged_decode_reference(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
    )
