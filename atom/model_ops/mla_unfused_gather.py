# SPDX-License-Identifier: MIT
"""Unfused torch fallback for aiter's Triton ``gather_kv_b_proj``.

The fused kernel does the whole MLA cached-prefix expansion in one launch: row
gather, KV-cache dequant, ``kv_b_proj``, the k_nope/v split and the k_pe concat.
On some architectures Triton has no working backend for the shapes a *chunked*
prefill produces and codegen asserts inside LLVM before any kernel runs::

    llvm/IR/Instructions.h: PHINode::setIncomingValue:
    Assertion `getType() == V->getType()' failed.

That is a compile-time failure with no fallback: the process aborts, so a long
prompt at concurrency takes the engine down. The FlyDSL gather is not an
alternative on such a target -- it is gfx950-only and aborts in its own
compiler if forced.

Every step of the chain is an ordinary torch op, so an unfused version needs no
codegen and cannot miscompile. It is slower and allocates per chunk; it exists
to keep a bring-up target serving, not to compete with the fused kernel, and is
off by default (``ATOM_UNFUSED_GATHER_KV_B_PROJ``).

What it deliberately does NOT implement, raising instead of guessing: the
shuffled-KV cache layout and a preshuffled weight (both private layouts of the
fused kernel) and MXFP4 ``kv_b_proj`` weights. Producing subtly wrong K/V would
be far worse than refusing -- nothing downstream would notice.
"""

from __future__ import annotations

import torch

__all__ = ["unfused_gather_kv_b_proj"]

# Rows per chunk. Bounds the temporary [rows, num_heads * (nope + v_dim)] GEMM
# output; the whole prefix at once is a large transient on long contexts.
_DEFAULT_ROWS = 4096


def _mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` for a ``[N, K]`` weight, routed away from hipBLASLt.

    ``torch.matmul`` lands on hipBLASLt, i.e. a Tensile ``Cijk_*_SAV_UserArgs_``
    kernel that reads its pointers from a device-side argument buffer. Combined
    with hipGraph replay and prefix caching that page-faults. aiter's Triton
    a16w16 GEMM is correct here and much faster on this shape. Called directly
    rather than through ``tgemm.mm`` so a tuned-CSV row cannot route it back.
    """
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    except Exception:  # noqa: BLE001 -- optional kernel; torch is the fallback
        return torch.matmul(x, w.t())
    # gemm_a16w16 reads N from w.shape[0] and returns [M, N].
    return gemm_a16w16(x, w, bias=None, dtype=x.dtype)


def _dequant_weight(
    weight: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize ``kv_b_proj`` once per call, matching the fused kernel.

    Three shapes of scale, all of which appear in shipped checkpoints: none at
    all (a bf16 ``kv_b_proj``, e.g. Kimi-K3), one per output row, and the
    128x128 block scale DeepSeek ships.
    """
    if scale is None:
        return weight if weight.dtype == dtype else weight.to(dtype)
    if scale.dim() == 1 or (scale.dim() == 2 and scale.shape[1] == 1):
        return (weight.to(torch.float32) * scale.reshape(-1, 1).to(torch.float32)).to(
            dtype
        )
    scale_n, scale_k = scale.shape
    n, k = weight.shape
    blocked = weight.to(torch.float32).view(scale_n, n // scale_n, scale_k, k // scale_k)
    return (blocked * scale[:, None, :, None].to(torch.float32)).reshape(n, k).to(dtype)


def _row_addresses(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_kv: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map each output row to its ``(block, slot)`` in the paged cache.

    Vectorized on purpose: a per-sequence Python loop would force a host sync
    per request in the prefill hot path. ``page_size`` 1 needs no arithmetic at
    all; the paged form walks ``kv_indptr`` / ``cu_seqlens_k`` the way the fused
    kernel's non-flat grid does.
    """
    if block_size == 1:
        return kv_indices[:total_kv].long(), None
    token = torch.arange(total_kv, device=device)
    # cu_seqlens_k = [0, n0, n0 + n1, ...]: a token's sequence is the last entry
    # that is <= it.
    seq = torch.searchsorted(cu_seqlens_k[1:].contiguous(), token, right=True)
    within = token - cu_seqlens_k[seq]
    block = kv_indices[(kv_indptr[seq] + within // block_size).long()].long()
    return block, (within % block_size).long()


def unfused_gather_kv_b_proj(
    k_buffer: torch.Tensor,
    k_scale: torch.Tensor | None,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_prefix_sum_context_lens: torch.Tensor,
    kv_proj_weight: torch.Tensor,
    kv_proj_scale: torch.Tensor | None,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    weight_preshuffle: bool = False,
    shuffled_kv_cache: bool = False,
    rows_per_chunk: int = _DEFAULT_ROWS,
) -> None:
    """Drop-in replacement for ``aiter.ops.triton.gather_kv_b_proj``.

    Writes into ``k_prefix`` / ``v_prefix`` in place and returns None, exactly
    as the fused op does.
    """
    if shuffled_kv_cache:
        raise NotImplementedError(
            "ATOM_UNFUSED_GATHER_KV_B_PROJ does not implement the shuffled-KV "
            "cache layout (the intra-block shuffle is the fused kernel's "
            "private layout). Set ATOM_USE_TRITON_MLA_SHUFFLE_KV=0."
        )
    if weight_preshuffle:
        raise NotImplementedError(
            "ATOM_UNFUSED_GATHER_KV_B_PROJ cannot read a preshuffled "
            "kv_b_proj weight (is_shuffled=True)."
        )
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is not None and kv_proj_weight.dtype == fp4:
        raise NotImplementedError(
            "ATOM_UNFUSED_GATHER_KV_B_PROJ does not implement MXFP4 kv_b_proj "
            "weights; unset the env for this checkpoint."
        )

    _, block_size, hidden = k_buffer.shape
    total_kv, num_heads, k_width = k_prefix.shape
    v_dim = v_prefix.shape[2]
    kv_c_dim = kv_proj_weight.shape[1]
    pe_dim = hidden - kv_c_dim
    nope = k_width - pe_dim
    dtype = k_prefix.dtype

    weight = _dequant_weight(kv_proj_weight, kv_proj_scale, dtype)
    block, slot = _row_addresses(
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        total_kv,
        block_size,
        k_prefix.device,
    )

    # A bf16 cache must NOT be scaled. The fused kernel hardcodes exactly this
    # (its k_scalar_scale is 1.0 for a bf16 cache and a load of k_scale
    # otherwise); applying k_scale anyway scales all of K and V by a constant,
    # which nothing downstream would flag. The GEMM is linear in kv_c, so
    # scaling after the matmul is exact.
    scale_cache = k_scale is not None and k_buffer.dtype is not torch.bfloat16

    for start in range(0, total_kv, rows_per_chunk):
        stop = min(start + rows_per_chunk, total_kv)
        rows = block[start:stop]
        latent = (
            k_buffer[rows, 0] if slot is None else k_buffer[rows, slot[start:stop]]
        ).to(dtype)

        projected = _mm(latent[:, :kv_c_dim], weight)
        k_pe = latent[:, kv_c_dim:]
        if scale_cache:
            projected = projected * k_scale
            k_pe = k_pe * k_scale
        projected = projected.view(stop - start, num_heads, nope + v_dim)

        k_prefix[start:stop, :, :nope] = projected[:, :, :nope]
        k_prefix[start:stop, :, nope:] = k_pe.unsqueeze(1).expand(-1, num_heads, -1)
        v_prefix[start:stop] = projected[:, :, nope:]
