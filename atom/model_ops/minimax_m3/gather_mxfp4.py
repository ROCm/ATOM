# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense cached-prefix gather of the MXFP4 KV pool (``--kv-cache-dtype fp4``).

The MXFP4 counterpart of ``cp_mha_gather_cache`` (base_attention.py): same grid
(one program a token and kv head), same metadata, but the pages are the
``mxfp4_pages`` pool fields and the output is dequantized to BF16, which is what the
prefix-hit prefill attention (``flash_attn_varlen_func`` over the full KV) reads.
The writer has already stored the new tokens, so one pass reads cached + new:
``cu_seqlens_kv`` is the cumulative TOTAL kv length and ``seq_starts`` is zero.

e8m0 scale invariant (valid rows): bytes 1..254. The fused writer floors the row
amax so it never emits 0, and emits 255 only for a non-finite amax; ``pa_decode``
documents 0 and 255 as unsupported in valid rows. The scale is decoded as aiter's
device code does (``byte << 23`` bitcast to f32), so 0 -> 0.0 and 255 -> inf, NOT
``formats.e8m0_to_float`` (0 -> 2^-127, 255 -> NaN). Rows past a sequence's length
are never read here (the output has exactly ``total_tokens`` rows).
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.minimax_m3.mxfp4_pages import (
    DATA_BYTES,
    DATA_PAGE_BYTES,
    HEAD_DIM,
    PAGE_BYTES,
    PAGE_TOKENS,
    SCALE_BYTES,
)

# jit code reads globals only as constexpr
_HEAD_DIM = tl.constexpr(HEAD_DIM)
_DATA_BYTES = tl.constexpr(DATA_BYTES)
_DATA_PAGE_BYTES = tl.constexpr(DATA_PAGE_BYTES)
_SCALE_BYTES = tl.constexpr(SCALE_BYTES)
_PAGE_TOKENS = tl.constexpr(PAGE_TOKENS)


@triton.jit
def _gather_one(
    pages_ptr,
    out_ptr,
    block_id,
    head_id,
    slot_id,
    token_id,
    stride0,
    stride1,
    num_heads,
    valid,
):
    col = tl.arange(0, _HEAD_DIM)
    page = pages_ptr + block_id * stride0 + head_id * stride1
    byte = tl.load(page + slot_id * _DATA_BYTES + col // 2, mask=valid, other=0).to(
        tl.int32
    )
    code = (byte >> ((col % 2) * 4)) & 0xF  # low nibble is the even element
    mag = code & 7
    exp = mag >> 1
    man = (mag & 1).to(tl.float32)
    val = tl.where(
        exp == 0, man * 0.5, (1.0 + 0.5 * man) * (1 << tl.maximum(exp - 1, 0))
    )
    val = tl.where(code >> 3 == 1, -val, val)
    sbyte = tl.load(
        page + _DATA_PAGE_BYTES + slot_id * _SCALE_BYTES + col // 32,
        mask=valid,
        other=0,
    ).to(tl.int32)
    val = val * (sbyte << 23).to(tl.float32, bitcast=True)
    tl.store(out_ptr + (token_id * num_heads + head_id) * _HEAD_DIM + col, val)


@triton.jit
def _gather_mxfp4_kernel(
    k_pages_ptr,
    v_pages_ptr,
    k_out_ptr,
    v_out_ptr,
    block_table_ptr,
    cu_seqlens_kv_ptr,
    batch_id_per_k_token_ptr,
    seq_start_ptr,
    k_stride0,
    k_stride1,
    v_stride0,
    v_stride1,
    num_heads,
    max_block_num,
    num_pages,
):
    # int64: the packed output offset exceeds int32 on long cached batches
    token_id = tl.program_id(0).to(tl.int64)
    head_id = tl.program_id(1).to(tl.int64)
    batch_idx = tl.load(batch_id_per_k_token_ptr + token_id)
    pos = (
        token_id
        - tl.load(cu_seqlens_kv_ptr + batch_idx)
        + tl.load(seq_start_ptr + batch_idx)
    )
    # Triton's // and % truncate toward zero, so a negative pos must be masked itself
    blk = pos // _PAGE_TOKENS
    block_id = tl.load(
        block_table_ptr + max_block_num * batch_idx + blk,
        mask=(pos >= 0) & (blk < max_block_num),
        other=-1,
    ).to(tl.int64)
    # a page id outside the pool, or a position outside the table (negative or past
    # it), reads nothing and gathers zeros: no out-of-bounds access, the bad metadata
    # stays upstream's bug. batch_idx itself is NOT checked: batch_id_per_k_token
    # must hold batch indices in [0, batch), as for cp_mha_gather_cache.
    valid = (block_id >= 0) & (block_id < num_pages)
    slot_id = pos % _PAGE_TOKENS
    _gather_one(
        k_pages_ptr, k_out_ptr, block_id, head_id, slot_id, token_id,
        k_stride0, k_stride1, num_heads, valid,
    )  # fmt: skip
    _gather_one(
        v_pages_ptr, v_out_ptr, block_id, head_id, slot_id, token_id,
        v_stride0, v_stride1, num_heads, valid,
    )  # fmt: skip


@torch.no_grad()
def gather_mxfp4_prefix(
    key_cache: torch.Tensor,  # uint8 [pages, Hkv, 8704], Mxfp4KvLayout.kv_views
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [batch, max_blocks], 128-token blocks
    cu_seqlens_kv: torch.Tensor,  # [batch + 1] cumulative total (cached + new) kv
    batch_id_per_k_token: torch.Tensor,  # [total_tokens]
    seq_starts: torch.Tensor,  # [batch], zeros for a prefix at position 0
    total_tokens: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantized ``(k, v)``, each ``[total_tokens, Hkv, 128]`` ``out_dtype``.

    Pages are read in place by stride, so a pool field with a larger page stride
    is fine; no host sync, one launch.
    """
    for name, pages in (("key", key_cache), ("value", value_cache)):
        if (
            pages.dtype != torch.uint8
            or pages.dim() != 3
            or pages.shape[-1] != PAGE_BYTES
            or pages.stride(-1) != 1
        ):
            raise ValueError(
                f"{name}_cache pages are uint8 [pages, Hkv, {PAGE_BYTES}] with a "
                f"contiguous byte axis, got {pages.dtype} {tuple(pages.shape)} "
                f"strides {tuple(pages.stride())}"
            )
    if key_cache.shape[:2] != value_cache.shape[:2]:
        raise ValueError("key and value pages must have the same [pages, Hkv]")
    if out_dtype != torch.bfloat16:
        raise ValueError(f"the MXFP4 gather dequantizes to bfloat16, not {out_dtype}")
    # the cp_mha_gather_cache metadata contract: int32, on the cache's device, the
    # block table a dense row-major [batch, max_blocks] (the kernel indexes
    # max_blocks * batch + block). Metadata only: nothing here syncs the device.
    for name, t in (
        ("block_tables", block_tables),
        ("cu_seqlens_kv", cu_seqlens_kv),
        ("batch_id_per_k_token", batch_id_per_k_token),
        ("seq_starts", seq_starts),
    ):
        if t.dtype != torch.int32 or t.device != key_cache.device:
            raise ValueError(
                f"{name} must be int32 on {key_cache.device}, got {t.dtype} on "
                f"{t.device}"
            )
    if block_tables.dim() != 2 or not block_tables.is_contiguous():
        raise ValueError("block_tables must be a contiguous [batch, max_blocks]")
    if batch_id_per_k_token.numel() < total_tokens:
        raise ValueError(
            f"batch_id_per_k_token covers {batch_id_per_k_token.numel()} tokens, "
            f"need {total_tokens}"
        )
    num_heads = key_cache.shape[1]
    k = torch.empty(
        (total_tokens, num_heads, HEAD_DIM), dtype=out_dtype, device=key_cache.device
    )
    v = torch.empty_like(k)
    if total_tokens == 0:
        return k, v
    _gather_mxfp4_kernel[(total_tokens, num_heads)](
        key_cache,
        value_cache,
        k,
        v,
        block_tables,
        cu_seqlens_kv,
        batch_id_per_k_token,
        seq_starts,
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        num_heads,
        block_tables.size(1),
        key_cache.shape[0],
    )
    return k, v
