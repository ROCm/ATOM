# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP token-ownership arithmetic shared by attention and KV transfer.

These helpers are pure ``//`` / ``%`` so they stay elementwise over Python
ints, numpy arrays, and torch tensors. They live outside ``dcp_ops`` so P/D
relayout can follow the same rule without importing Triton or aiter.
"""

import numpy as np


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


def dcp_concat_gather_plan(g_cu, first: int, end: int, world: int, interleave: int):
    """Requests `[first, end)`'s DCP index-key gather, off host `g_cu` (the
    step's committed-key cumsum, int64): each request's committed keys
    `g_lens`; their local lengths padded to the interleave super-block,
    `lpad`, the max over ranks so every rank gathers the same count; the
    padded cumsum `cu_pad`; the shard length `local_total`; and `src`, each
    of the requests' keys -- end to end from the first one's -- as a row of
    the rank-major all-gathered `[world, local_total]` shards. Global
    position `p` of a request lives on rank `(p // S) % W` at local index
    `(p // (S * W)) * S + p % S`, hence
    `src = owner(p) * local_total + cu_pad[b] + local_index(p)`."""
    W, S = world, interleave
    g_lens = g_cu[first + 1 : end + 1] - g_cu[first:end]
    lpad = ((g_lens + S * W - 1) // (S * W)) * S
    cu_pad = np.zeros(end - first + 1, dtype=np.int64)
    np.cumsum(lpad, out=cu_pad[1:])
    local_total = int(cu_pad[-1])
    total = int(g_cu[end] - g_cu[first])
    pos = np.arange(total, dtype=np.int64) - np.repeat(
        g_cu[first:end] - g_cu[first], g_lens
    )
    src = (
        dcp_owner_rank(pos, W, S) * local_total
        + np.repeat(cu_pad[:-1], g_lens)
        + dcp_local_index(pos, W, S)
    )
    return g_lens, lpad, cu_pad, local_total, src
