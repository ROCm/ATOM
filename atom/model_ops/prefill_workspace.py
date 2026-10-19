# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bound dense cached-prefill K/V materialization at whole-request boundaries."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from atom.utils.forward_context import ForwardContext


def prefill_workspace_reserve_bytes(models, *, overlapping_batches: int = 1) -> int:
    """Reserve the largest enabled local dense gather, including the draft.

    A warmup's short or uncached prefill does not exercise long cached KV
    materialization. Keep its configured bound outside the persistent KV pool.
    Layers and request groups reuse storage on one stream, so take their max;
    TBO may keep two microbatches live. Sparse/MLA/linear implementations do
    not use this dense gather. Dense layers are conservatively counted even
    when a particular request can dispatch to a direct-paged alternative.
    """
    largest = max(
        (
            getattr(module, "gqa_prefill_kv_workspace_bytes", 0)
            for model in models
            for module in model.modules()
            if not getattr(module, "is_indexed_sparse_attention", False)
        ),
        default=0,
    )
    return largest * overlapping_batches


@dataclass(frozen=True)
class PrefillRequestLengths:
    """Owned CPU prefix sums for the real requests in one prefill pass.

    Both tuples start at zero and have ``num_requests + 1`` entries.
    ``cu_q_lens`` counts this pass's query tokens; ``cu_k_lens`` counts the
    K/V context visible to each request, including any cached prefix.
    Adjacent differences give per-request lengths. Padding is excluded.
    """

    cu_q_lens: tuple[int, ...]
    cu_k_lens: tuple[int, ...]

    def __post_init__(self):
        if (
            len(self.cu_q_lens) < 2
            or len(self.cu_q_lens) != len(self.cu_k_lens)
            or self.cu_q_lens[0] != 0
            or self.cu_k_lens[0] != 0
        ):
            raise ValueError("Prefill cumulative lengths must start at zero and match")
        for i in range(len(self.cu_q_lens) - 1):
            q_len = self.cu_q_lens[i + 1] - self.cu_q_lens[i]
            k_len = self.cu_k_lens[i + 1] - self.cu_k_lens[i]
            if q_len <= 0 or k_len < q_len:
                raise ValueError("Each prefill request must have 0 < query <= context")

    def slice_requests(
        self,
        *,
        request_start: int,
        request_end: int,
        q_token_start: int,
        q_token_end: int,
        has_cached: bool = True,
    ) -> PrefillRequestLengths:
        """Match a request slice and a query-token window in the input batch.

        ``[request_start, request_end)`` indexes requests. The independent
        ``[q_token_start, q_token_end)`` indexes rows of the flattened Q tensor
        described by ``cu_q_lens``; it is not a position in cached K/V. TBO
        may split inside the first or last selected request, so request
        indices alone cannot describe the retained query tokens.

        For example, Q lengths (2, 3, 4) give (0, 2, 5, 9). Requests [1, 3)
        and Q rows [3, 9) retain query lengths (2, 4), rebased to (0, 2, 6).

        Cached KV retains the selected requests' context spans at this step;
        without a cache, K follows the clipped Q window. A backend that later
        changes visible KV must update this snapshot too. No device readback.
        """
        q_lens = tuple(
            min(max(value, q_token_start), q_token_end)
            for value in self.cu_q_lens[request_start : request_end + 1]
        )
        k_lens = (
            self.cu_k_lens[request_start : request_end + 1] if has_cached else q_lens
        )
        return PrefillRequestLengths(
            cu_q_lens=tuple(value - q_lens[0] for value in q_lens),
            cu_k_lens=tuple(value - k_lens[0] for value in k_lens),
        )

    def groups(self, max_tokens: int) -> list[tuple[int, int]]:
        """Return half-open request ranges, preserving order and all context."""
        groups = []
        start = 0
        for end in range(len(self.cu_k_lens) - 1):
            length = self.cu_k_lens[end + 1] - self.cu_k_lens[end]
            if length > max_tokens:
                raise ValueError(
                    f"Prefill request {end} needs {length} KV tokens, but the "
                    f"workspace fits {max_tokens}; increase "
                    "--gqa-prefill-kv-workspace-mb or set it to 0 to disable "
                    "grouping. Context is never truncated."
                )
            if self.cu_k_lens[end + 1] - self.cu_k_lens[start] > max_tokens:
                groups.append((start, end))
                start = end
        groups.append((start, len(self.cu_k_lens) - 1))
        return groups


def run_grouped_prefill(
    attention: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    fwd_ctx: ForwardContext,
    max_workspace_bytes: int,
) -> torch.Tensor:
    """Run the normal gather+attention once per bounded group, without recursion.

    The caller has already written new KV to the paged cache. Only the dense
    K/V tensors are bounded, not Q/output, persistent caches, kernel scratch or
    allocator-reserved memory. Both gathered tensors use k.dtype. Group calls
    stay on the caller's stream, ordering attention before workspace reuse.
    Lengths belong to this metadata view (including TBO slices), not to the
    runner's reusable buffers. No device readback occurs.
    """
    meta = fwd_ctx.attn_metadata
    bytes_per_token = 2 * k.shape[1] * k.shape[2] * k.element_size()
    max_tokens = max_workspace_bytes // bytes_per_token
    if meta.total_kv <= max_tokens:
        return attention(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)

    lengths = meta.prefill_request_lengths
    if lengths is None:
        raise ValueError("Prefill workspace grouping requires CPU request lengths")
    cu_q_lens, cu_k_lens = lengths.cu_q_lens, lengths.cu_k_lens
    if cu_q_lens[-1] != q.shape[0] or cu_k_lens[-1] != meta.total_kv:
        raise ValueError("CPU prefill lengths disagree with attention geometry")
    groups = lengths.groups(max_tokens)

    import torch

    output = torch.empty_like(q)
    for start, end in groups:
        qs, qe = cu_q_lens[start], cu_q_lens[end]
        ks, ke = cu_k_lens[start], cu_k_lens[end]
        group_meta = copy.copy(meta)
        group_meta.cu_seqlens_q = meta.cu_seqlens_q[start : end + 1] - qs
        group_meta.cu_seqlens_k = meta.cu_seqlens_k[start : end + 1] - ks
        group_meta.block_tables = meta.block_tables[start:end]
        group_meta.seq_starts = meta.seq_starts[start:end]
        group_meta.context_lens = meta.context_lens[start:end]
        group_meta.batch_id_per_k_token = meta.batch_id_per_k_token[ks:ke] - start
        group_meta.total_kv = ke - ks
        q_lengths = [cu_q_lens[i + 1] - cu_q_lens[i] for i in range(start, end)]
        group_meta.max_seqlen_q = max(q_lengths)
        group_meta.min_seqlen_q = min(q_lengths)
        group_meta.max_seqlen_k = max(
            cu_k_lens[i + 1] - cu_k_lens[i] for i in range(start, end)
        )
        group_meta.prefill_request_lengths = PrefillRequestLengths(
            cu_q_lens=tuple(value - qs for value in cu_q_lens[start : end + 1]),
            cu_k_lens=tuple(value - ks for value in cu_k_lens[start : end + 1]),
        )
        group_ctx = copy.copy(fwd_ctx)
        group_ctx.attn_metadata = group_meta
        output[qs:qe].copy_(
            attention(
                q[qs:qe],
                k[qs:qe],
                v[qs:qe],
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                group_ctx,
            )
        )
    return output
