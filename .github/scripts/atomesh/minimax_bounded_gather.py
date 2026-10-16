# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only request grouping for bounded EAGLE3 prefill KV workspace."""

import copy


def request_groups(cu_q, cu_k, max_tokens):
    """Partition whole requests, preserving each request's complete context."""
    if max_tokens <= 0 or len(cu_q) != len(cu_k) or not len(cu_q):
        raise ValueError("Invalid prefill lengths or workspace limit")
    if int(cu_q[0]) != 0 or int(cu_k[0]) != 0:
        raise ValueError("Prefill cumulative lengths must start at zero")
    groups = []
    start = 0
    for end in range(len(cu_k) - 1):
        q_len = int(cu_q[end + 1]) - int(cu_q[end])
        k_len = int(cu_k[end + 1]) - int(cu_k[end])
        if q_len <= 0 or k_len < q_len:
            raise ValueError("Each prefill request must have 0 < query <= context")
        if k_len > max_tokens:
            raise ValueError("One request exceeds the bounded draft KV workspace")
        if int(cu_k[end + 1]) - int(cu_k[start]) > max_tokens:
            groups.append((start, end))
            start = end
    if start < len(cu_k) - 1:
        groups.append((start, len(cu_k) - 1))
    return groups


def bounded_prefill_attention(
    attention,
    q,
    k,
    v,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    fwd_ctx,
    max_workspace_bytes=8 << 30,
):
    """Return None for the ordinary path; otherwise run independent groups.

    No request/token is dropped and causal attention is unchanged. Only the
    dense K/V materialization is grouped; target scheduling and draft steps
    keep their original batch. Lengths come from host mirrors, so partitioning
    never synchronizes the GPU. Each group's tensors die before the next group
    allocates; the same compute stream orders attention before storage reuse.
    """
    meta = fwd_ctx.attn_metadata
    if not getattr(fwd_ctx.context, "is_draft", False) or not meta.has_cached:
        return None
    bytes_per_token = k.shape[1] * k.shape[2] * (k.element_size() + v.element_size())
    max_tokens = max_workspace_bytes // bytes_per_token
    if meta.total_kv <= max_tokens:
        return None

    import torch

    cu_q = meta._ci_prefill_cu_q
    cu_k = meta._ci_prefill_cu_k
    if int(cu_q[-1]) != q.shape[0] or int(cu_k[-1]) != meta.total_kv:
        raise ValueError("Host prefill lengths disagree with draft attention geometry")
    groups = request_groups(cu_q, cu_k, max_tokens)
    output = torch.empty_like(q)
    print(
        f"[bounded-draft-kv] tokens={meta.total_kv} groups={len(groups)} "
        f"max_tokens={max_tokens} max_workspace_bytes={max_workspace_bytes}",
        flush=True,
    )
    for start, end in groups:
        qs, qe = int(cu_q[start]), int(cu_q[end])
        ks, ke = int(cu_k[start]), int(cu_k[end])
        group_meta = copy.copy(meta)
        group_meta.cu_seqlens_q = meta.cu_seqlens_q[start : end + 1] - qs
        group_meta.cu_seqlens_k = meta.cu_seqlens_k[start : end + 1] - ks
        group_meta.block_tables = meta.block_tables[start:end]
        group_meta.seq_starts = meta.seq_starts[start:end]
        group_meta.context_lens = meta.context_lens[start:end]
        group_meta.batch_id_per_k_token = meta.batch_id_per_k_token[ks:ke] - start
        group_meta.total_kv = ke - ks
        group_meta.max_seqlen_q = max(
            int(cu_q[i + 1] - cu_q[i]) for i in range(start, end)
        )
        group_meta.max_seqlen_k = max(
            int(cu_k[i + 1] - cu_k[i]) for i in range(start, end)
        )
        group_ctx = copy.copy(fwd_ctx)
        group_ctx.attn_metadata = group_meta
        output[qs:qe].copy_(
            attention.prefill_attention(
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


def instrument(runtime):
    attention = runtime / "atom/model_ops/attention_mha.py"
    source = attention.read_text()
    old = """        if attn_metadata.has_cached:
            q, k, v, k_cache, v_cache, k_scale, v_scale = (
"""
    new = """        from atom._ci_bounded_gather import bounded_prefill_attention

        bounded = bounded_prefill_attention(
            self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
        )
        if bounded is not None:
            return bounded
        if attn_metadata.has_cached:
            q, k, v, k_cache, v_cache, k_scale, v_scale = (
"""
    if source.count(old) != 1:
        raise ValueError("Expected one MHA prefill gather entry")
    attention.write_text(source.replace(old, new))
    proposer = runtime / "atom/spec_decode/eagle_proposer.py"
    source = proposer.read_text()
    old = """        var = self.runner.forward_vars
        # Eaale3 only support mha currently
"""
    new = """        var = self.runner.forward_vars
        if context.is_prefill:
            # Own the host lengths for this eager draft pass; no GPU readback.
            attn_metadata._ci_prefill_cu_q = var["cu_seqlens_q"].np[: scheduled_bs + 1].copy()
            attn_metadata._ci_prefill_cu_k = var["cu_seqlens_k"].np[: scheduled_bs + 1].copy()
        # Eaale3 only support mha currently
"""
    if source.count(old) != 1:
        raise ValueError("Expected one EAGLE3 metadata handoff")
    proposer.write_text(source.replace(old, new))
