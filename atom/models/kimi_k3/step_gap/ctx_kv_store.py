# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft layers' context cache writes in one launch.

Production writes a layer's context rows with
``atom.model_ops.triton_fused_mla_ctx_kv._fused_mla_ctx_kv_kernel`` (RMSNorm of
the kv_a columns, rope of the rope columns, the paged store), a launch a layer.
``_ctx_kv_store_layers_kernel`` is that kernel's program with a layer axis: a
program is one (token, layer), reading the token's row of the layer from
``kv_lora`` [layers, N, 576] and the layer's norm weight and cache through a
pointer table. The arithmetic statements are the production kernel's, in its
order, at its block size and warp count, so the compiler lays the reduction
out alike; that the caches come out bit for bit is checked on the GPU
(``tools/kimi_k3_step_gap_ctx_check.py``), not argued.

Kept beside its only caller rather than in ``atom.model_ops``: a change to the
production kernel's arithmetic has to be mirrored here (the check fails until
it is).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ctx_kv_store_layers_kernel(
    kv_lora_ptr,  # [layers, N, kv_lora_rank + pe_dim] pre-norm, pre-rope
    tab_ptr,  # [layers, 2] int64: the layer's norm weight, its cache
    positions_ptr,  # [N]
    cos_ptr,  # [max_position, pe_dim // 2]
    sin_ptr,  # [max_position, pe_dim // 2]
    slot_mapping_ptr,  # [N]
    cache_like_ptr,  # a layer's cache: its element type only, never read
    k_scale_ptr,  # fp32 scalar
    num_tokens,
    stride_cos_p,
    stride_cache_block,
    stride_cache_slot,
    eps,
    KV_LORA_RANK: tl.constexpr,
    PE_DIM: tl.constexpr,
    PE_HALF: tl.constexpr,
    BLOCK_LORA: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    IS_NEOX: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    token = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    # slot < 0 marks a padded / non-owned row: skipped, as production's does
    if slot >= 0:
        row = kv_lora_ptr + (layer.to(tl.int64) * num_tokens + token.to(tl.int64)) * (
            KV_LORA_RANK + PE_DIM
        )
        norm_weight_ptr = tl.load(tab_ptr + 2 * layer).to(
            tl.pointer_type(kv_lora_ptr.dtype.element_ty)
        )
        kv_cache_ptr = tl.load(tab_ptr + 2 * layer + 1).to(
            tl.pointer_type(cache_like_ptr.dtype.element_ty)
        )

        lora_offs = tl.arange(0, BLOCK_LORA)
        lora_mask = lora_offs < KV_LORA_RANK
        x = tl.load(row + lora_offs, mask=lora_mask, other=0.0).to(tl.float32)
        w = tl.load(norm_weight_ptr + lora_offs, mask=lora_mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / KV_LORA_RANK + eps)
        nope = (x * rstd * w).to(kv_lora_ptr.dtype.element_ty)

        pe_offs = tl.arange(0, PE_DIM)
        if IS_NEOX:
            pair_offs = tl.where(
                pe_offs < PE_HALF, pe_offs + PE_HALF, pe_offs - PE_HALF
            )
            freq_offs = tl.where(pe_offs < PE_HALF, pe_offs, pe_offs - PE_HALF)
            negate = pe_offs < PE_HALF
        else:
            pair_offs = tl.where(pe_offs % 2 == 0, pe_offs + 1, pe_offs - 1)
            freq_offs = pe_offs // 2
            negate = pe_offs % 2 == 0
        pe_row = row + KV_LORA_RANK
        x_pe = tl.load(pe_row + pe_offs).to(tl.float32)
        x_pair = tl.load(pe_row + pair_offs).to(tl.float32)
        pos = tl.load(positions_ptr + token).to(tl.int64)
        freq = pos * stride_cos_p + freq_offs
        cos = tl.load(cos_ptr + freq).to(tl.float32)
        sin = tl.load(sin_ptr + freq).to(tl.float32)
        rot = tl.where(negate, -x_pair, x_pair)
        pe = (x_pe * cos + rot * sin).to(kv_lora_ptr.dtype.element_ty)

        dst = (
            kv_cache_ptr
            + (slot // CACHE_BLOCK_SIZE) * stride_cache_block
            + (slot % CACHE_BLOCK_SIZE) * stride_cache_slot
        )
        if APPLY_SCALE:
            inv_scale = 1.0 / tl.load(k_scale_ptr)
            nope_out = (nope.to(tl.float32) * inv_scale).to(
                cache_like_ptr.dtype.element_ty
            )
            pe_out = (pe.to(tl.float32) * inv_scale).to(cache_like_ptr.dtype.element_ty)
        else:
            nope_out = nope.to(cache_like_ptr.dtype.element_ty)
            pe_out = pe.to(cache_like_ptr.dtype.element_ty)
        tl.store(dst + lora_offs, nope_out, mask=lora_mask)
        tl.store(dst + KV_LORA_RANK + pe_offs, pe_out)


def store_layers(
    kv_lora: torch.Tensor,  # [layers, N, kv_lora_rank + qk_rope_head_dim]
    tab: torch.Tensor,  # [layers, 2] int64 on the device: norm weight, cache
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_like: torch.Tensor,
    k_scale: torch.Tensor,
    eps: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_neox: bool,
    apply_scale: bool,
) -> None:
    """Every layer's ``fused_mla_ctx_norm_rope_cache`` of its rows of
    ``kv_lora``, in one launch. The caches the table points at share
    ``cache_like``'s dtype, block size and strides; the norm weights are
    ``kv_lora``'s dtype, ``kv_lora_rank`` long."""
    layers, num_tokens, entry = kv_lora.shape
    if num_tokens == 0:
        return
    assert kv_lora.is_contiguous() and entry == kv_lora_rank + qk_rope_head_dim
    _ctx_kv_store_layers_kernel[(num_tokens, layers)](
        kv_lora,
        tab,
        positions,
        cos_cache,
        sin_cache,
        slot_mapping,
        cache_like,
        k_scale,
        num_tokens,
        cos_cache.stride(0),
        cache_like.stride(0),
        cache_like.stride(1),
        eps,
        KV_LORA_RANK=kv_lora_rank,
        PE_DIM=qk_rope_head_dim,
        PE_HALF=qk_rope_head_dim // 2,
        BLOCK_LORA=triton.next_power_of_2(kv_lora_rank),
        CACHE_BLOCK_SIZE=cache_like.shape[1],
        IS_NEOX=is_neox,
        APPLY_SCALE=apply_scale,
        # production's launch parameters: the reduction's layout follows them
        num_warps=4,
        num_stages=2,
    )
