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
