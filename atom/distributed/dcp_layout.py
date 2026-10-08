# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP token-ownership arithmetic shared by attention and KV transfer.

These helpers are pure ``//`` / ``%`` so they stay elementwise over Python
ints, numpy arrays, and torch tensors. They live outside ``dcp_ops`` so P/D
relayout can follow the same rule without importing Triton or aiter.
"""


def dcp_owner_rank(pos, dcp_size, cp_kv_cache_interleave_size=1):
    """Which DCP rank owns global token ``pos`` under interleaved KV storage.

    Interleaving groups tokens into chunks of ``cp_kv_cache_interleave_size``
    (= S); chunk ``c = pos // S`` is stored on rank ``c % dcp_size``. For
    ``S == 1`` this reduces to the round-robin ``pos % dcp_size``.
    """
    return (pos // cp_kv_cache_interleave_size) % dcp_size


def dcp_local_index(pos, dcp_size, cp_kv_cache_interleave_size=1):
    """Local KV-sequence index of global token ``pos`` on its owning rank.

    Each ``S * W`` super-block contributes ``S`` tokens to a rank, so the local
    index is ``(pos // (S*W)) * S + (pos % S)``. For ``S == 1`` this reduces to
    the round-robin ``pos // dcp_size``.
    """
    sw = cp_kv_cache_interleave_size * dcp_size
    return (pos // sw) * cp_kv_cache_interleave_size + (
        pos % cp_kv_cache_interleave_size
    )


def dcp_global_pos(local_index, dcp_rank, dcp_size, cp_kv_cache_interleave_size=1):
    """Inverse of ``dcp_local_index``: global token position of local KV index
    ``local_index`` held on ``dcp_rank``.

    Local index ``j`` on rank ``r`` sits in local S-group ``j // S`` at offset
    ``j % S``; that group is global chunk ``(j//S)*W + r``, so the global
    position is ``((j//S)*W + r) * S + (j % S)``. For ``S == 1`` this reduces
    to the round-robin ``j*W + r``.
    """
    return (
        (local_index // cp_kv_cache_interleave_size) * dcp_size + dcp_rank
    ) * cp_kv_cache_interleave_size + (local_index % cp_kv_cache_interleave_size)


def dcp_local_prefix_count(n, dcp_rank, dcp_size, cp_kv_cache_interleave_size=1):
    """How many of the global positions ``[0, n)`` ``dcp_rank`` owns.

    Each ``S * W`` super-block hands every rank ``S`` positions; the tail
    remainder is handed out ``S`` at a time in rank order. This is the same
    split ``dcp_ops.dcp_local_context_lens`` applies to a context length,
    hoisted here so the sparse-prefill metadata builder can apply it per query
    token without importing Triton.

    The inverse-ish relation to the rest of this module: ``dcp_local_index``
    maps one owned position to its local slot, while this counts the owned
    positions below ``n`` -- which is what a causal window's end becomes once
    the sequence is sharded.
    """
    s = cp_kv_cache_interleave_size
    sw = s * dcp_size
    full = n // sw
    rem = n - full * sw - dcp_rank * s
    if hasattr(rem, "clip"):
        rem = rem.clip(0, s)
    else:
        rem = max(0, min(rem, s))
    return full * s + rem


def dcp_prefill_local_window(
    cu_pad, g_lens, q_counts, dcp_rank, dcp_size, cp_kv_cache_interleave_size=1
):
    """Per-query-token causal window in one DCP rank's LOCAL column space.

    A sparse-prefill query token sees ``[0, p]`` of its own sequence, where
    ``p = cached_len + offset_in_chunk``. When the sequence is sharded, that
    window becomes ``[cu_pad[b], cu_pad[b] + (owned positions below p+1))`` --
    the request's base in the concatenated local plane, plus
    ``dcp_local_prefix_count``.

    ``cu_pad`` is the cumsum of the PADDED local lengths
    ``ceil(g / (S*W)) * S``, the same buffer ``cp_gather_indexer_k_quant_cache``
    is driven by, so the end is bounded by that gather's own extent and the
    scorer cannot reach the inter-rank padding rows it leaves uninitialized.

    Needs numpy only; lives here rather than in the attention metadata builder
    so it can be checked against ``dcp_owner_rank`` directly.

    Returns ``(local_ks, local_ke)``, both int64 arrays of ``sum(q_counts)``.
    """
    import numpy as np

    q_counts = np.asarray(q_counts, dtype=np.int64)
    g_lens = np.asarray(g_lens, dtype=np.int64)
    cu_pad = np.asarray(cu_pad, dtype=np.int64)
    bs = q_counts.shape[0]
    if bs == 0 or int(q_counts.sum()) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()

    offset = np.concatenate([np.arange(c, dtype=np.int64) for c in q_counts])
    p_next = np.repeat(g_lens - q_counts, q_counts) + offset + 1
    local_ks = np.repeat(cu_pad[:bs], q_counts)
    local_ke = local_ks + dcp_local_prefix_count(
        p_next, dcp_rank, dcp_size, cp_kv_cache_interleave_size
    )
    return local_ks, local_ke
