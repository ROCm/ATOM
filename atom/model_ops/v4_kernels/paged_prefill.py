# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse prefill attention with two KV sources: paged `unified_kv` (history)
and per-fwd flat `kv` (current chunk's input).

Designed for V4 prefill: indexes the two KV sources directly without
materialising a per-fwd `kv_flat_sa` packed tensor. See
`atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_PREFILL_DESIGN.zh.md` §1, §3
for design rationale.

Caller contract:
  unified_kv:        [total_pages, D] BF16 — prefix source. Same buffer as
    decode kernel: SWA ring slots in `[0, swa_pages)`, compress pages in
    `[swa_pages, total_pages)`. For prefill, prefix indices select
    (a) prior-chunk SWA history, (b) CSA topk, (c) HCA all-committed.
  kv_indices_prefix: [total_prefix_indices] int32 — flat per-token slot
    lists. Per-token entries live in
    `kv_indices_prefix[kv_indptr_prefix[t] : kv_indptr_prefix[t+1]]`.
    `-1` entries are skipped (sentinel).
  kv_indptr_prefix:  [N+1] int32 — true prefix sum (variable per-token len).

  kv:                [total_tokens, D] BF16 — extend source = current
    fwd's just-computed K (NOT yet written to swa_kv ring). Layout matches
    `swa_write` input.
  kv_indices_extend: [total_extend_indices] int32 — flat per-token row idx
    lists into `kv`. Per-token entries live in
    `kv_indices_extend[kv_indptr_extend[t] : kv_indptr_extend[t+1]]`.
    `-1` entries are skipped (rare for extend; usually all valid).
  kv_indptr_extend:  [N+1] int32 — true prefix sum.

  attn_sink:         [H] per-head learnable softmax-denom bias (V4 specific).
  softmax_scale:     float.

Per-token K loop iterates two regions sequentially, sharing the online
softmax accumulator (m_i, l_i, acc) across regions. Order of regions does
not affect correctness (online softmax is order-invariant).

Returns:
  out: [N, H, D] same dtype as q.

Numerics: identical online-softmax + sink finalization to
`sparse_attn_v4_paged_decode` — bit-exact when the extend region is empty
(then equivalent to a decode call with the same prefix indices).
"""

import torch
import triton
import triton.language as tl

from atom.utils.decorators import mark_trace

from atom.utils import envs

try:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    _HAS_OPUS = True
except ImportError:
    pa_sparse_prefill_opus = None
    _HAS_OPUS = False

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse

# Fixed best-known Triton meta for DeepSeek-V4 TP=8 / local H=8 prefill.
_BLOCK_H = 16
_BLOCK_K = 32
_NUM_WARPS = 4
_NUM_STAGES = 1
_WAVES_PER_EU = 2


@triton.jit
def _sparse_attn_v4_paged_prefill_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D]   — prefix source
    kv_indices_prefix_ptr,  # [total_prefix_indices] int32
    kv_indptr_prefix_ptr,  # [N+1] int32
    kv_ptr,  # [total_tokens, D]    — extend source
    kv_indices_extend_ptr,  # [total_extend_indices] int32
    kv_indptr_extend_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,  # unified_kv stride 0 (= D usually)
    pkv_stride_d: tl.constexpr,  # unified_kv stride 1 (= 1 usually)
    ekv_stride_n: tl.constexpr,  # kv stride 0
    ekv_stride_d: tl.constexpr,  # kv stride 1
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREFIX_BLOCK_K: tl.constexpr,
    CHECK_NEG_ONE_SENTINEL: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    FULL_D: tl.constexpr,
):
    # int64 to avoid 32-bit overflow in per-token address offsets
    # (t * q_stride_t / t * out_stride_t).
    t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q_ptrs = (
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d
    )
    # FULL_D keeps the D=512 path free of d_mask overhead, while preserving a
    # masked fallback for future V4 variants with non-power-of-two head dims.
    if FULL_D:
        q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    else:
        q = tl.load(q_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0)

    neg_large = -3.4028234663852886e38
    # Use exp2/log2 softmax form, matching the faster attention convention on
    # this ROCm/Triton path: exp(x) == exp2(x * log2(e)).
    qk_scale = softmax_scale * 1.4426950408889634  # log2(e)
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    p_k_offs = tl.arange(0, PREFIX_BLOCK_K)

    # ===== Region 1: prefix from unified_kv =====
    # HAS_PREFIX specializes ratio=0/Dense-SWA to skip the whole prefix region.
    if HAS_PREFIX:
        p_start = tl.load(kv_indptr_prefix_ptr + t)
        p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
        p_len = p_end - p_start

        for k_start in tl.range(0, p_len, PREFIX_BLOCK_K):
            k_pos = k_start + p_k_offs
            in_range = k_pos < p_len
            slot = tl.load(
                kv_indices_prefix_ptr + p_start + k_pos,
                mask=in_range,
                other=-1,
            )
            # Some legacy index buffers can use -1 as a sentinel. HCA's
            # short-prefix path can disable this check via constexpr.
            if CHECK_NEG_ONE_SENTINEL:
                valid = in_range & (slot >= 0)
            else:
                valid = in_range

            kv_ptrs = (
                unified_kv_ptr
                + slot[:, None] * pkv_stride_n
                + d_offs[None, :] * pkv_stride_d
            )
            if FULL_D:
                kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)
            else:
                kv = tl.load(
                    kv_ptrs,
                    mask=valid[:, None] & d_mask[None, :],
                    other=0.0,
                )

            scores = tl.dot(q, tl.trans(kv)) * qk_scale
            scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(scores - m_new[:, None])
            p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            m_i = m_new
            l_i = l_new

    # ===== Region 2: extend from kv (per-fwd flat) =====
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=in_range,
            other=-1,
        )
        # Same sentinel policy as prefix; production extend indices are fully
        # written, but the check keeps compatibility for generic fallback.
        if CHECK_NEG_ONE_SENTINEL:
            valid = in_range & (slot >= 0)
        else:
            valid = in_range

        kv_ptrs = kv_ptr + slot[:, None] * ekv_stride_n + d_offs[None, :] * ekv_stride_d
        if FULL_D:
            kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)
        else:
            kv = tl.load(kv_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # ===== Sink finalization =====
    # Online softmax + sink integration: sink is a virtual extra K with V=0,
    # contributing only to the denominator. After main loops, (m_i, l_i, acc)
    # are in m_i frame; sink may shift max to m_final = max(m_i, sink), so
    # rescale BOTH l_i (for denom) AND acc (for numerator) by alpha to switch
    # to m_final frame. The sink itself adds exp(sink - m_final) to l_final
    # but contributes 0 to acc since V_sink = 0.
    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink * 1.4426950408889634
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp2(m_i - m_final)
    l_final = l_i * alpha + tl.exp2(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d
    )
    if FULL_D:
        tl.store(out_ptrs, out, mask=h_mask[:, None])
    else:
        tl.store(out_ptrs, out, mask=h_mask[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_v4_paged_prefill_csa_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,
    kv_indices_prefix_ptr,
    kv_indptr_prefix_ptr,
    kv_ptr,
    kv_indices_extend_ptr,
    kv_indptr_extend_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,
    pkv_stride_d: tl.constexpr,
    ekv_stride_n: tl.constexpr,
    ekv_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREFIX_BLOCK_K: tl.constexpr,
):
    # CSA-only fast path for DeepSeek-V4 D=512. It removes generic prefix/HCA
    # options from the shared kernel and keeps the proven BLOCK_K/PREFIX_BLOCK_K
    # shape for ratio=4.
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H

    q_ptrs = (
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    neg_large = -3.4028234663852886e38
    qk_scale = softmax_scale * 1.4426950408889634
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    p_k_offs = tl.arange(0, PREFIX_BLOCK_K)
    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    for k_start in tl.range(0, p_len, PREFIX_BLOCK_K):
        k_pos = k_start + p_k_offs
        valid = k_pos < p_len
        slot = tl.load(kv_indices_prefix_ptr + p_start + k_pos, mask=valid, other=0)
        kv_ptrs = (
            unified_kv_ptr
            + slot[:, None] * pkv_stride_n
            + d_offs[None, :] * pkv_stride_d
        )
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    k_offs = tl.arange(0, BLOCK_K)
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for e_k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = e_k_start + k_offs
        valid = k_pos < e_len
        slot = tl.load(kv_indices_extend_ptr + e_start + k_pos, mask=valid, other=0)
        kv_ptrs = kv_ptr + slot[:, None] * ekv_stride_n + d_offs[None, :] * ekv_stride_d
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink * 1.4426950408889634
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp2(m_i - m_final)
    l_final = l_i * alpha + tl.exp2(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    out_ptrs = (
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d
    )
    tl.store(out_ptrs, out, mask=h_mask[:, None])


def _sparse_attn_v4_paged_prefill_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_prefill requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill expects fp16/bf16 q, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    T, H, D = q.shape
    if out is None:
        # Callers can pass `out` to reuse a buffer and avoid this allocation.
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError(
            f"out shape/dtype/device mismatch: got shape={tuple(out.shape)} "
            f"dtype={out.dtype} device={out.device}, expected shape={tuple(q.shape)} "
            f"dtype={q.dtype} device={q.device}"
        )
    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    block_h = _BLOCK_H
    block_k = _BLOCK_K
    num_warps = _NUM_WARPS
    num_stages = _NUM_STAGES
    waves_per_eu = _WAVES_PER_EU
    block_d = triton.next_power_of_2(D)
    full_d = D == block_d
    avg_prefix_len = kv_indices_prefix.numel() / max(T, 1)
    # HCA prefill has a very short prefix, while CSA has a long prefix. A
    # smaller prefix tile helps HCA but hurts CSA, so choose it by prefix size.
    prefix_block_k = min(block_k, 16) if 0 < avg_prefix_len <= 16 else block_k
    # Prefix/extend indices are fully written in production; keep sentinel
    # checks only for non-short-prefix shapes for compatibility with legacy data.
    check_neg_one_sentinel = not (0 < avg_prefix_len <= 16)
    has_prefix = kv_indices_prefix.numel() > 0
    # Only long-prefix CSA with D=512 uses the CSA fast kernel. Other prefix
    # shapes, including HCA and non-512 V4 variants, use the generic kernel.
    use_csa_fast_kernel = has_prefix and avg_prefix_len > 16 and D == 512 and full_d
    if use_csa_fast_kernel:
        _sparse_attn_v4_paged_prefill_csa_kernel[(T, triton.cdiv(H, block_h))](
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            float(softmax_scale),
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            PREFIX_BLOCK_K=prefix_block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            waves_per_eu=waves_per_eu,
        )
        return out

    _sparse_attn_v4_paged_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        PREFIX_BLOCK_K=prefix_block_k,
        CHECK_NEG_ONE_SENTINEL=check_neg_one_sentinel,
        HAS_PREFIX=has_prefix,
        FULL_D=full_d,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    return out


def sparse_attn_v4_paged_prefill_reference(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure-torch reference via virtual-pool concatenation.

    Builds a virtual KV pool `pool = cat([unified_kv, kv])`, offsets extend
    indices by `len(unified_kv)`, then delegates to the proven
    `_sparse_attn_ragged_torch` (joint softmax with sink — same impl used by
    the decode kernel's reference). Slow but correct — for unit tests /
    dump-bisect.
    """
    # Local import: PostToolUse formatter strips unused top-level imports
    # between consecutive Edits (project memory note).
    from atom.model_ops.sparse_attn_v4 import _sparse_attn_ragged_torch

    T = q.size(0)
    n_pages = unified_kv.size(0)
    pool = torch.cat([unified_kv, kv], dim=0)  # [n_pages + total_tokens, D]

    p_indptr = kv_indptr_prefix.to(torch.int64)
    e_indptr = kv_indptr_extend.to(torch.int64)
    p_spans = (p_indptr[1:] - p_indptr[:T]).clamp(min=0)
    e_spans = (e_indptr[1:] - e_indptr[:T]).clamp(min=0)
    total_spans = p_spans + e_spans
    k_dim = int(total_spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1

    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        ps = int(p_indptr[t].item())
        pe = int(p_indptr[t + 1].item())
        es = int(e_indptr[t].item())
        ee = int(e_indptr[t + 1].item())
        p_n = pe - ps
        e_n = ee - es
        if p_n > 0:
            # prefix indices point into unified_kv, no offset; -1 stays -1
            topk_idxs[t, :p_n] = kv_indices_prefix[ps:pe].to(torch.int32)
        if e_n > 0:
            # extend indices point into kv → offset by n_pages; preserve -1
            e_idx = kv_indices_extend[es:ee].to(torch.int64)
            shifted = torch.where(
                e_idx >= 0,
                e_idx + n_pages,
                torch.full_like(e_idx, -1),
            )
            topk_idxs[t, p_n : p_n + e_n] = shifted.to(torch.int32)

    return _sparse_attn_ragged_torch(q, pool, attn_sink, topk_idxs, softmax_scale)


@mark_trace
def sparse_attn_v4_paged_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    *,
    unified_kv_rope: torch.Tensor | None = None,
    q_packed: torch.Tensor | None = None,
    q_rope: torch.Tensor | None = None,
    k_packed: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
    prefix: str = "",
) -> torch.Tensor:
    """V4 prefill sparse attention over two KV sources (paged unified_kv +
    flat per-fwd kv), dispatching on the kv-cache layout.

    Native 2buff fp8 (``unified_kv_rope`` provided): routes to aiter's
    ``pa_sparse_prefill_fp8_opus`` (op4). ``unified_kv`` is the packed fp8 NoPE
    prefix pool and ``unified_kv_rope`` the bf16 RoPE prefix pool; the
    op-quantized fp8 Q (``q_packed`` / ``q_rope``) and extend K
    (``k_packed`` / ``k_rope``) are fed directly, no dequant of the prefix and no
    torch quant. gfx950/gfx1250. (fp8 + PCP is out of scope: the extend K is this
    fwd's shard and is not PCP all-gathered — the caller must pass the sharded
    ``k_packed`` / ``k_rope``.)

    Otherwise (bf16): the existing OPUS / Triton / reference path over ``q`` and
    the bf16 extend ``kv``.

    Args:
      q:                 [T, H, D] BF16/FP16 — query (bf16 path).
      unified_kv:        [total_pages, D] — prefix source (paged). BF16/FP16 on
        the bf16 path; packed fp8 NoPE pool on the fp8 2buff path.
      kv_indices_prefix: [total_prefix] int32 — flat per-token slot lists into
        unified_kv. -1 sentinels skipped.
      kv_indptr_prefix:  [T+1] int32 — true prefix sum.
      kv:                [total_tokens, D] BF16/FP16 — extend source (bf16 path;
        this fwd's input K, NOT yet in swa_kv ring).
      kv_indices_extend: [total_extend] int32 — flat per-token row idx lists
        into kv. -1 sentinels skipped.
      kv_indptr_extend:  [T+1] int32 — true prefix sum.
      attn_sink:         [H] — per-head softmax-denom bias.
      softmax_scale:     float.
      out:               optional output buffer. Can alias q when the caller no
        longer needs q after attention (bf16 path only).
      unified_kv_rope:   [total_pages, rd] bf16 RoPE prefix pool — presence
        selects the native fp8 2buff path.
      q_packed / q_rope: fp8 2buff Q ([T, H, 512] fp8 / [T, H, 64] bf16).
      k_packed / k_rope: fp8 2buff extend K ([T, 512] fp8 / [T, 64] bf16, or the
        [T, 1, *] views produced by the quant kernel — flattened here).

    Returns:
      out: [T, H, D] bf16.
    """
    if unified_kv_rope is not None:
        # Native fp8 prefill (op4): 2buff fp8 prefix pool + op-quantized fp8 Q
        # and extend K fed directly. No dequant of the prefix, no torch quant.
        from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

        return pa_sparse_prefill_fp8_opus(
            q_packed,
            q_rope,
            unified_kv,
            unified_kv_rope,
            kv_indices_prefix,
            kv_indptr_prefix,
            k_packed.view(k_packed.shape[0], -1),
            k_rope.view(k_rope.shape[0], -1),
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
        )  # [S, H, head_dim] bf16
    if get_gfx() == "gfx1250":
        return pa_prefill_sparse(
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
        )
    # Backend selection: prefer OPUS when available; fall back to Triton on
    # import failure, env override, or runtime error (e.g. unsupported GPU).
    if not envs.ATOM_FORCE_ATTN_TRITON and _HAS_OPUS:
        try:
            return pa_sparse_prefill_opus(
                q,
                unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv,
                kv_indices_extend,
                kv_indptr_extend,
                attn_sink,
                softmax_scale,
                out=out,
            )
        except RuntimeError:
            pass
    return _sparse_attn_v4_paged_prefill_triton(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
    )
