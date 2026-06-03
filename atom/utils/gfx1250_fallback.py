# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch fallbacks for aiter C++ modules not built on gfx1250.

Gated by ATOM_GFX1250_FALLBACK. Correctness-only — no perf guarantees.
Smoke-tested on gfx950 via serve.sh + curl.sh.

Covers:
  - module_mhc                          → handled in deepseek_v4.py (existing torch path)
  - module_moe_topk                     → in atom/model_ops/moe.py (sqrtsoftplus torch impl)
  - module_dsv4_rotate_quant            → torch_rope_rotate_activation here
  - module_rope_1c_cached_positions_fwd → torch_rope_cached_positions_fwd_inplace_1c here
  - module_top_k_per_row                → torch_top_k_per_row_{prefill,decode} here
  - deepgemm_fp8_paged_mqa_logits       → torch_deepgemm_fp8_paged_mqa_logits here
"""

from typing import Optional

import torch


# ---------- RoPE (GPT-J / interleaved, reuse_freqs_front_part=True, nope_first=False) ----------

def _apply_rope_gptj_inplace(
    x: torch.Tensor,   # [..., rope_dim] last dim
    cos: torch.Tensor, # [max_seq, 1, 1, rope_dim//2] (or 2-D collapse)
    sin: torch.Tensor,
    positions: torch.Tensor,  # any shape broadcastable to x[..., 0]
    rope_dim: int,
) -> None:
    """In-place GPT-J style RoPE on trailing ``rope_dim`` of ``x``.

    Forward (per pair):
        out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    """
    rd_half = rope_dim // 2
    # Flatten cos/sin to [max_seq, rd_half] via stride 0 (the [1,1] middle dims).
    cos_flat = cos.reshape(cos.shape[0], -1)[:, :rd_half]
    sin_flat = sin.reshape(sin.shape[0], -1)[:, :rd_half]
    pos_flat = positions.reshape(-1)
    # Gather per-token cos/sin → [N, rd_half]
    c = cos_flat[pos_flat]
    s = sin_flat[pos_flat]
    # Reshape x to [N, H, rope_dim], view as [N, H, rd_half, 2]
    orig_shape = x.shape
    x2d = x.reshape(-1, *orig_shape[len(positions.shape):]) if False else x.view(-1, orig_shape[-2], rope_dim)
    pair = x2d.view(-1, orig_shape[-2], rd_half, 2)
    even = pair[..., 0]
    odd = pair[..., 1]
    # c/s: [N, rd_half] → unsqueeze head dim
    c_b = c.unsqueeze(1).to(x.dtype)
    s_b = s.unsqueeze(1).to(x.dtype)
    new_even = even * c_b - odd * s_b
    new_odd = even * s_b + odd * c_b
    pair[..., 0] = new_even
    pair[..., 1] = new_odd


def torch_rope_cached_positions_fwd_inplace_1c(
    input_x: torch.Tensor,  # [s, b, h, d]
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,  # [s, b]
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
) -> None:
    assert rotate_style == 1, "only GPT-J style supported in fallback"
    assert reuse_freqs_front_part is True
    assert nope_first is False
    rope_dim = cos.reshape(cos.shape[0], -1).shape[-1] * 2
    # All input dims past last are the rope slice (head_size == rotary_dim per
    # call-site contract in deepseek_v4.py:706).
    _apply_rope_gptj_inplace(input_x, cos, sin, positions, rope_dim)


def torch_rope_cached_positions_2c_fwd_inplace(
    input_x: torch.Tensor,
    input_y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
) -> None:
    torch_rope_cached_positions_fwd_inplace_1c(
        input_x, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first
    )
    torch_rope_cached_positions_fwd_inplace_1c(
        input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first
    )


# ---------- DSV4 rotate (RoPE on trailing rope_dim + Hadamard on full last dim) ----------

def torch_rope_rotate_activation(
    out: torch.Tensor,    # [N, H, D]
    input: torch.Tensor,  # [N, H, D]
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> None:
    """Torch fallback for aiter `rope_rotate_activation`: in-place RoPE on
    trailing ``rope_dim``, then Hadamard rotate on the full last dim.
    """
    # Local import to avoid circular import at module load time.
    from atom.model_ops.quant_v4 import rotate_activation as _hadamard

    if out.data_ptr() != input.data_ptr():
        out.copy_(input)
    # RoPE on trailing rope_dim (mutates out in place).
    rope_slice = out[..., -rope_dim:]
    _apply_rope_gptj_inplace(rope_slice, cos, sin, positions, rope_dim)
    # Hadamard on full last dim. `_hadamard` is functional, so write back.
    rotated = _hadamard(out)
    out.copy_(rotated)


# ---------- top_k_per_row (radix top-k with -1 sentinels in the tail) ----------

def torch_top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,  # [numRows] int — per-row valid range [start, end)
    rowEnds: torch.Tensor,    # [numRows] int
    indices: torch.Tensor,    # [numRows, k] int32 — written
    values: Optional[torch.Tensor],
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
) -> None:
    """Per-row top-k of logits in [rowStarts[i], rowEnds[i]). Writes -1 in the
    tail when the valid range is shorter than k. Matches the aiter
    `top_k_per_row_prefill` contract.
    """
    # logits view: [numRows, L] in the contract used by DS-V4 callers.
    L = logits.shape[-1] if logits.dim() == 2 else (logits.numel() // numRows)
    log2d = logits.view(numRows, L)
    # Per-row mask of valid columns [rowStarts, rowEnds).
    col = torch.arange(L, device=logits.device).unsqueeze(0)  # [1, L]
    starts = rowStarts.view(-1, 1).to(col.dtype)
    ends = rowEnds.view(-1, 1).to(col.dtype)
    valid = (col >= starts) & (col < ends)
    NEG_INF = torch.finfo(log2d.dtype).min if log2d.is_floating_point() else -2**31
    masked = torch.where(valid, log2d, torch.full_like(log2d, NEG_INF))
    k_eff = min(k, L)
    vals, idx = torch.topk(masked, k_eff, dim=-1)
    # Mark entries that landed on -inf (i.e., out-of-range) as -1.
    invalid_pick = (vals == NEG_INF)
    idx = idx.to(torch.int32)
    idx[invalid_pick] = -1
    indices[:, :k_eff].copy_(idx)
    if k_eff < k:
        indices[:, k_eff:].fill_(-1)
    if values is not None:
        values[:, :k_eff].copy_(vals.to(values.dtype))
        if k_eff < k:
            values[:, k_eff:].fill_(NEG_INF)


def torch_top_k_per_row_decode(
    logits: torch.Tensor,     # [numRows, L]
    next_n: int,
    seqLens: torch.Tensor,    # per-seq upper bound; row r belongs to seq r // next_n
    indices: torch.Tensor,    # [numRows, k] int32
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
) -> None:
    """Per-row top-k bounded by seqLens[r // next_n]. Tail filled with -1.
    Matches the aiter `top_k_per_row_decode` contract.
    """
    L = logits.shape[-1] if logits.dim() == 2 else (logits.numel() // numRows)
    log2d = logits.view(numRows, L)
    row_idx = torch.arange(numRows, device=logits.device) // max(next_n, 1)
    per_row_len = seqLens.view(-1)[row_idx].view(-1, 1)
    col = torch.arange(L, device=logits.device).unsqueeze(0)  # [1, L]
    valid = col < per_row_len
    NEG_INF = torch.finfo(log2d.dtype).min if log2d.is_floating_point() else -2**31
    masked = torch.where(valid, log2d, torch.full_like(log2d, NEG_INF))
    k_eff = min(k, L)
    vals, idx = torch.topk(masked, k_eff, dim=-1)
    invalid_pick = (vals == NEG_INF)
    idx = idx.to(torch.int32)
    idx[invalid_pick] = -1
    indices[:, :k_eff].copy_(idx)
    if k_eff < k:
        indices[:, k_eff:].fill_(-1)


# ---------- deepgemm_fp8_paged_mqa_logits (Gluon MFMA kernel → torch fallback) ----------

def _unshuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Reverse of ``aiter.ops.shuffle.shuffle_weight`` (non-guinterleave)."""
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size()
    BN = IN
    x_ = x.view(-1, x.shape[-2] // BN, BK // K, x.shape[-1] // BK, BN, K)
    x_ = x_.permute(0, 1, 4, 3, 2, 5).contiguous()
    return x_.view(*x.shape)


# Block-chunk size for the streaming fallback below. Override with
# ATOM_DG_FB_CHUNK_BLOCKS to trade peak temporary memory for kernel-launch
# overhead. 64 keeps the per-chunk gather under ~512 MiB even at large capture
# batch sizes; bump up for fewer launches if memory allows.
import os as _os
_DG_FB_CHUNK_BLOCKS = int(_os.getenv("ATOM_DG_FB_CHUNK_BLOCKS", "64"))


def torch_deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,       # [bs, next_n, heads, head_dim] fp8
    kv_cache: torch.Tensor,    # [num_blocks, block_size, 1, index_dim] uint8
    weights: torch.Tensor,     # [total_tokens, heads] fp32
    out_logits: torch.Tensor,  # [total_tokens, max_model_len] fp32
    context_lens: torch.Tensor,  # [bs] int32
    kv_indices: torch.Tensor,  # [bs, max_blocks_per_seq] int32 (block_tables)
    max_model_len: int,
    KVBlockSize: int = 1,
    Preshuffle: bool = False,
    **_kwargs,
) -> None:
    """Torch fallback for the Gluon ``deepgemm_fp8_paged_mqa_logits`` kernel.

    Streams over `max_blocks_per_seq` in chunks of `_DG_FB_CHUNK_BLOCKS` so the
    peak temporary is O(bs * chunk * block_size * head_dim) instead of the full
    padded KV (which can reach tens of GiB at max_model_len=1M). Intermediate
    dequant/matmul runs in bf16 (halved vs fp32, accuracy fine for the
    downstream top-k); only the final write to `out_logits` is fp32.

    Fully batched — no .tolist() / .item() / Python data-dependent ops on
    tensors — so this is safe inside CUDAGraph capture.
    """
    from aiter import dtypes as _dtypes

    bs, next_n, heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, index_dim = kv_cache.shape
    max_blk = kv_indices.shape[1]  # max_blocks_per_seq

    kv_flat = kv_cache.view(num_blocks, block_size * index_dim)
    kv_data_bytes = kv_flat[:, :block_size * head_dim]
    kv_scale_bytes = kv_flat[:, block_size * head_dim: block_size * head_dim + 4 * block_size]

    kv_fp8 = kv_data_bytes.view(_dtypes.fp8).view(num_blocks, block_size, head_dim)
    kv_scale = kv_scale_bytes.view(torch.float32).view(num_blocks, block_size, 1)

    # Precompute things that don't depend on chunk.
    # Reshape Q for a broadcast-free bmm: fold (heads, next_n) into the M dim
    # so matmul becomes [bs, heads*next_n, head_dim] @ [bs, head_dim, chunk_len].
    # This avoids ROCm materializing the broadcasted K of shape
    # [bs, heads, head_dim, chunk_len] (which costs heads× memory).
    q_bf16_mn = (
        q_fp8.to(torch.bfloat16)
        .permute(0, 2, 1, 3)            # [bs, heads, next_n, head_dim]
        .reshape(bs, heads * next_n, head_dim)
        .contiguous()
    )
    # weights: [total_tokens, heads] → [bs, heads, next_n, 1]
    w = weights.view(bs, next_n, heads).permute(0, 2, 1).unsqueeze(-1).to(torch.bfloat16)
    n_idx = torch.arange(next_n, device=q_fp8.device, dtype=context_lens.dtype)
    q_abs_pos = context_lens.unsqueeze(1) - next_n + n_idx.unsqueeze(0)  # [bs, next_n]

    out_logits.fill_(float("-inf"))
    out_view = out_logits.view(bs, next_n, max_model_len)
    neg_inf_t = torch.full((), float("-inf"), dtype=torch.bfloat16, device=q_fp8.device)

    chunk_blk = _DG_FB_CHUNK_BLOCKS
    for blk_start in range(0, max_blk, chunk_blk):
        blk_end = min(blk_start + chunk_blk, max_blk)
        n_chunk_blk = blk_end - blk_start
        kv_start = blk_start * block_size
        kv_end = blk_end * block_size
        chunk_len = kv_end - kv_start

        blk_ids = kv_indices[:, blk_start:blk_end].clamp(min=0).long()  # [bs, n_chunk_blk]
        k_chunk_fp8 = kv_fp8[blk_ids]      # [bs, n_chunk_blk, block_size, head_dim]
        s_chunk = kv_scale[blk_ids]        # [bs, n_chunk_blk, block_size, 1]
        if Preshuffle:
            k_chunk_fp8 = _unshuffle_weight(k_chunk_fp8, layout=(16, 16))

        # bf16 dequant: [bs, chunk_len, head_dim]
        k_chunk = (k_chunk_fp8.to(torch.bfloat16) * s_chunk.to(torch.bfloat16)).reshape(
            bs, chunk_len, head_dim
        )

        # bmm: [bs, heads*next_n, head_dim] @ [bs, head_dim, chunk_len]
        #   -> [bs, heads*next_n, chunk_len] -> [bs, heads, next_n, chunk_len]
        score = torch.bmm(q_bf16_mn, k_chunk.transpose(-1, -2)).view(
            bs, heads, next_n, chunk_len
        )

        kv_pos = torch.arange(
            kv_start, kv_end, device=q_fp8.device, dtype=context_lens.dtype
        )
        valid = kv_pos.unsqueeze(0) < context_lens.unsqueeze(1)            # [bs, chunk_len]
        causal = kv_pos.view(1, 1, -1) <= q_abs_pos.unsqueeze(2)            # [bs, next_n, chunk_len]
        mask = (valid.unsqueeze(1) & causal).unsqueeze(1)                   # [bs, 1, next_n, chunk_len]

        score = torch.where(mask, score, neg_inf_t)
        score = torch.relu(score)
        score = (score * w).sum(dim=1)  # [bs, next_n, chunk_len] bf16

        # Write to out_logits; clip to max_model_len.
        write_end = min(kv_end, max_model_len)
        if write_end > kv_start:
            wlen = write_end - kv_start
            out_view[:, :, kv_start:write_end] = score[:, :, :wlen].float()
