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


@dataclass(frozen=True)
class PrefillRequestLengths:
    """Owned CPU cumulative lengths for the real requests in one prefill pass."""

    cu_q: tuple[int, ...]
    cu_k: tuple[int, ...]

    def __post_init__(self):
        if (
            len(self.cu_q) < 2
            or len(self.cu_q) != len(self.cu_k)
            or self.cu_q[0] != 0
            or self.cu_k[0] != 0
        ):
            raise ValueError("Prefill cumulative lengths must start at zero and match")
        for i in range(len(self.cu_q) - 1):
            q_len = self.cu_q[i + 1] - self.cu_q[i]
            k_len = self.cu_k[i + 1] - self.cu_k[i]
            if q_len <= 0 or k_len < q_len:
                raise ValueError("Each prefill request must have 0 < query <= context")

    def slice_requests(
        self,
        start: int,
        end: int,
        token_start: int,
        token_end: int,
        *,
        has_cached: bool = True,
    ) -> PrefillRequestLengths:
        """Match a metadata request/token slice without reading device tensors.

        Cached KV keeps complete contexts; uncached KV follows the query
        slice. A backend that later changes visible KV must update the snapshot.
        Padding is excluded from this owned description.
        """
        q = tuple(
            min(max(value, token_start), token_end)
            for value in self.cu_q[start : end + 1]
        )
        k = self.cu_k[start : end + 1] if has_cached else q
        return PrefillRequestLengths(
            tuple(value - q[0] for value in q),
            tuple(value - k[0] for value in k),
        )

    def groups(self, max_tokens: int) -> list[tuple[int, int]]:
        """Return half-open request ranges, preserving order and all context."""
        groups = []
        start = 0
        for end in range(len(self.cu_k) - 1):
            length = self.cu_k[end + 1] - self.cu_k[end]
            if length > max_tokens:
                raise ValueError(
                    f"Prefill request {end} needs {length} KV tokens, but the "
                    f"workspace fits {max_tokens}; increase "
                    "--prefill-kv-workspace-mb or set it to 0 to disable "
                    "grouping. Context is never truncated."
                )
            if self.cu_k[end + 1] - self.cu_k[start] > max_tokens:
                groups.append((start, end))
                start = end
        groups.append((start, len(self.cu_k) - 1))
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
    cu_q, cu_k = lengths.cu_q, lengths.cu_k
    if cu_q[-1] != q.shape[0] or cu_k[-1] != meta.total_kv:
        raise ValueError("CPU prefill lengths disagree with attention geometry")
    groups = lengths.groups(max_tokens)

    import torch

    output = torch.empty_like(q)
    for start, end in groups:
        qs, qe = cu_q[start], cu_q[end]
        ks, ke = cu_k[start], cu_k[end]
        group_meta = copy.copy(meta)
        group_meta.cu_seqlens_q = meta.cu_seqlens_q[start : end + 1] - qs
        group_meta.cu_seqlens_k = meta.cu_seqlens_k[start : end + 1] - ks
        group_meta.block_tables = meta.block_tables[start:end]
        group_meta.seq_starts = meta.seq_starts[start:end]
        group_meta.context_lens = meta.context_lens[start:end]
        group_meta.batch_id_per_k_token = meta.batch_id_per_k_token[ks:ke] - start
        group_meta.total_kv = ke - ks
        q_lengths = [cu_q[i + 1] - cu_q[i] for i in range(start, end)]
        group_meta.max_seqlen_q = max(q_lengths)
        group_meta.min_seqlen_q = min(q_lengths)
        group_meta.max_seqlen_k = max(cu_k[i + 1] - cu_k[i] for i in range(start, end))
        group_meta.prefill_request_lengths = PrefillRequestLengths(
            tuple(value - qs for value in cu_q[start : end + 1]),
            tuple(value - ks for value in cu_k[start : end + 1]),
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
