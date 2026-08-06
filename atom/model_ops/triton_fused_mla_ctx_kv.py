# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm + RoPE + paged-cache write for MLA *context* latent rows.

One launch for what the per-op path spends four on: RMSNorm over the
``kv_lora_rank``-wide latent, RoPE over the ``qk_rope_head_dim``-wide positional
lane, the concat, and the two-segment ``concat_and_cache_mla`` store (plus the
throwaway query allocation aiter's 2-component RoPE op forces on a k-only
caller).

The caller is a drafter writing target-derived context rows into its own paged
MLA cache -- Kimi-K3 DSpark's ``write_context_kv``, which runs that chain once
per draft layer (x5) per drafting step over every scheduled target token. At
decode widths (bs*(1+T) rows) the chain is pure launch overhead; at prefill
widths it re-reads and re-writes the 512-wide latent three times. Both cases
collapse to one read of the projection output and one write of the cache.

Why a new kernel rather than an existing op:

  * ``aiter.fused_qk_rope_concat_and_cache_mla`` (and its ``_seg`` / Triton
    siblings) fuse rope+concat+cache but not the norm, and are built around a
    query: they demand ``q_nope``/``q_pe`` and write ``q_out``. This path has no
    query at all, so they would add a fabricated tensor and still leave the norm
    outside.
  * ``aiter.indexer_qk_rope_quant_and_cache`` does fuse norm+rope+cache, but for
    the DSA indexer: LayerNorm (mean-centred, with bias) rather than RMSNorm,
    ``static_assert(HEAD_DIM == 128)``, and an fp8 store with a per-token scale
    tail. None of that survives a 512+64 MLA latent.

Numerics are the per-op path's, kernel for kernel:

  * RMSNorm accumulates the sum of squares in fp32 and applies ``(x * rstd) * w``
    in fp32 with a single cast back to the activation dtype -- what aiter's
    ``rmsnorm2d_fwd`` does (``rmsnorm_quant_kernels.cu``).
  * RoPE indexes the SAME ``cos_cache`` / ``sin_cache`` buffers ``get_rope``
    built, so YaRN scaling, cache dtype and any upstream quirk come along
    unchanged; it promotes to fp32 for the two-term FMA and rounds back to the
    activation dtype BEFORE the store, which is the "materialize k_pe first"
    step aiter's own fused indexer kernel flags as load-bearing
    (``cache_kernels.cu:1451``).
  * The store reproduces ``concat_and_cache_mla``: a straight copy for an
    ``auto`` cache, ``cast(x * (1/k_scale))`` for an fp8 one.

Bit-exactness against the per-op path is not claimed and is not achievable from
Triton: the fp32 sum-of-squares reduction tree and ``rsqrt``'s approximation
differ from the HIP kernel's. Both differences are ~1e-7 relative on a quantity
that is then rounded to 8 mantissa bits, so the stored latent is expected to be
bitwise equal in practice -- the accompanying GPU test asserts exactly that.

Only the plain per-token / paged layout is covered (``[num_blocks, block_size,
kv_lora_rank + qk_rope_head_dim]``, last dim contiguous, which is also what a
per-token ``[num_slots, 1, entry]`` cache is). The seg (``ATOM_MLA_PAGE_SIZE``
> 1) and shuffled-KV layouts keep their own write kernels; the routing lives in
:meth:`atom.model_ops.attention_mla.MLAAttention.write_context_kv_latent`.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard


@triton.jit
def _fused_mla_ctx_kv_kernel(
    kv_lora_ptr,  # [N, kv_lora_rank + pe_dim] pre-norm, pre-rope (may be strided)
    norm_weight_ptr,  # [kv_lora_rank]
    positions_ptr,  # [N]
    cos_ptr,  # [max_position, pe_dim // 2]
    sin_ptr,  # [max_position, pe_dim // 2]
    slot_mapping_ptr,  # [N]
    kv_cache_ptr,  # [num_blocks, cache_block_size, kv_lora_rank + pe_dim]
    k_scale_ptr,  # fp32 scalar
    stride_kv_lora_n,
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
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    # slot < 0 marks a padded / non-owned row; concat_and_cache_mla returns
    # early on those and so must this, or a cudagraph pad would clobber slot 0.
    if slot >= 0:
        row = kv_lora_ptr + token.to(tl.int64) * stride_kv_lora_n

        lora_offs = tl.arange(0, BLOCK_LORA)
        lora_mask = lora_offs < KV_LORA_RANK
        # Masked lanes read 0 and contribute 0 to the sum of squares, which is
        # divided by the true rank -- so a padded BLOCK_LORA is exact, not a
        # widened mean.
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
            # GPT-J / interleaved: pairs are (2i, 2i+1) and share frequency i.
            pair_offs = tl.where(pe_offs % 2 == 0, pe_offs + 1, pe_offs - 1)
            freq_offs = pe_offs // 2
            negate = pe_offs % 2 == 0
        # The partner lane is re-read from memory rather than shuffled out of
        # registers: it is 64 elements against the 512 the norm already moved,
        # and it keeps the kernel free of cross-lane ops.
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
            # fp8 cache: dequant scale is per-tensor and static, applied to the
            # already-rounded activation exactly as the HIP copy kernel does.
            inv_scale = 1.0 / tl.load(k_scale_ptr)
            nope_out = (nope.to(tl.float32) * inv_scale).to(
                kv_cache_ptr.dtype.element_ty
            )
            pe_out = (pe.to(tl.float32) * inv_scale).to(kv_cache_ptr.dtype.element_ty)
        else:
            nope_out = nope.to(kv_cache_ptr.dtype.element_ty)
            pe_out = pe.to(kv_cache_ptr.dtype.element_ty)
        tl.store(dst + lora_offs, nope_out, mask=lora_mask)
        tl.store(dst + KV_LORA_RANK + pe_offs, pe_out)


@torch_compile_guard(mutates_args=["kv_cache"], gen_fake=lambda *a, **k: None)
def fused_mla_ctx_norm_rope_cache(
    kv_lora: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    k_scale: torch.Tensor,
    eps: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_neox: bool,
    apply_scale: bool,
) -> None:
    """RMSNorm(kv_c) + RoPE(k_pe) + concat + paged store, in one launch.

    Args:
        kv_lora: ``[N, kv_lora_rank + qk_rope_head_dim]`` raw projection output.
            Only the last dim must be contiguous, so the caller can pass the
            strided ``[..., q_lora_rank:]`` slice of a fused q/kv projection
            without materialising it.
        norm_weight: ``[kv_lora_rank]`` RMSNorm weight (``kv_a_layernorm``).
        positions: ``[N]`` absolute positions indexing ``cos_cache``.
        cos_cache, sin_cache: ``get_rope``'s buffers, squeezed to
            ``[max_position, qk_rope_head_dim // 2]``.
        slot_mapping: ``[N]`` flat destination slots; negative entries skipped.
        kv_cache: ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``,
            written in place.
        k_scale: fp32 scalar; consulted only when ``apply_scale``.
        is_neox: rotate style, ``rotary_emb.is_neox_style``.
        apply_scale: True iff the cache dtype is fp8 (an ``auto`` cache is a
            plain copy and must NOT scale, matching aiter's kAuto branch).
    """
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    _fused_mla_ctx_kv_kernel[(num_tokens,)](
        kv_lora,
        norm_weight,
        positions,
        cos_cache,
        sin_cache,
        slot_mapping,
        kv_cache,
        k_scale,
        kv_lora.stride(0),
        cos_cache.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        eps,
        KV_LORA_RANK=kv_lora_rank,
        PE_DIM=qk_rope_head_dim,
        PE_HALF=qk_rope_head_dim // 2,
        BLOCK_LORA=triton.next_power_of_2(kv_lora_rank),
        CACHE_BLOCK_SIZE=kv_cache.shape[1],
        IS_NEOX=is_neox,
        APPLY_SCALE=apply_scale,
        # One program per row over a 576-wide row: 4 waves give each lane a
        # handful of elements, which is where the 512-element fp32 reduction
        # stops being the bottleneck. Matches the width-driven choice in
        # lm_head_argmax / fused_dual_rmsnorm_cat.
        num_warps=4,
        num_stages=2,
    )
