# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small, CPU-testable geometry contracts for GLM-5.3's pooled indexer."""

import torch

from atom.utils import envs


def pooled_path_enabled(index_kpool: int) -> bool:
    """Whether the pooled indexer path is in force."""
    return index_kpool > 1 and envs.ATOM_GLM5_KPOOL


def effective_kpool_size(index_kpool: int) -> int:
    """Configured pool size, or one when the pooled path is disabled."""
    return index_kpool if pooled_path_enabled(index_kpool) else 1


def topk_output_width(topk: int, index_kpool: int) -> int:
    """Physical row width shared by the indexer producer and MLA metadata."""
    kpool = effective_kpool_size(index_kpool)
    if kpool <= 1:
        return topk
    return ((topk + kpool - 1 + 127) // 128) * 128


def get_query_request_indices(
    cu_seqlens_q: torch.Tensor,
    num_query_tokens: int,
) -> torch.Tensor:
    """Map each packed query row to its request index without a GPU kernel."""
    # Unlike repeat_interleave, this has no data-dependent output allocation.
    return torch.bucketize(
        torch.arange(num_query_tokens, device=cu_seqlens_q.device),
        cu_seqlens_q[1:],
        right=True,
    ).clamp_max(cu_seqlens_q.numel() - 2)


def speculative_verify_enabled(
    *,
    is_prefill: bool,
    num_spec_decodes: int,
    max_seqlen_q: int,
) -> bool:
    """Whether pooled indexing must use the ragged verification path."""
    return not is_prefill and (num_spec_decodes > 0 or max_seqlen_q > 1)


def speculative_kpool_history_size(
    pool_size: int,
    num_speculative_tokens: int | None,
) -> int:
    """Return ring rows needed across speculative rejection.

    A verification has ``W = num_speculative_tokens + 1`` query rows (the
    accepted token plus its draft). Before its accept/reject result is applied,
    the ring must retain the incomplete committed-pool suffix (at most
    ``pool_size - 1`` rows), that verification's ``W`` rows, and the following
    verification's ``W`` rows. Including the pool-closing row gives the safe
    integer bound ``pool_size + 2 * W``.

    Ring addressing uses modulo and is correct for any size at least that
    bound. Rounding to a power of two is only a performance/kernel-shape choice;
    it makes the modulo cheap and buckets graph shapes. For pool size 4 and
    three speculative tokens the bound is 12 rows and the allocation is 16.
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if num_speculative_tokens is None:
        return pool_size
    if num_speculative_tokens < 0:
        raise ValueError(
            "num_speculative_tokens must be non-negative, got "
            f"{num_speculative_tokens}"
        )
    verification_width = num_speculative_tokens + 1
    required_rows = pool_size + 2 * verification_width
    return 1 << (required_rows - 1).bit_length()


def speculative_pool_scratch_width(max_seqlen_k: int, pool_size: int) -> int:
    """Upper-bound pooled scoring columns by this batch's live KV length."""
    if max_seqlen_k < 0:
        raise ValueError(f"max_seqlen_k must be non-negative, got {max_seqlen_k}")
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    return (max_seqlen_k + pool_size - 1) // pool_size
