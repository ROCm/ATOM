# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP token-ownership arithmetic shared by attention and KV transfer.

These helpers are ``//``, ``%`` and ``np.clip``, so they stay elementwise over
Python ints, numpy arrays, and torch tensors. They live outside ``dcp_ops`` so P/D
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


def get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size=1):
    """Compute per-DCP-rank local sequence lengths.

    With interleaved storage, token i is stored on rank
    (i // cp_kv_cache_interleave_size) % dcp_size.

    Args:
        seq_lens: numpy array of sequence lengths
        dcp_size: DCP world size
        dcp_rank: this rank's DCP rank
        cp_kv_cache_interleave_size: interleaving granularity (default 1 = token-level)

    Returns:
        local_seq_lens: numpy array of local sequence lengths
    """
    full_chunks = seq_lens // (cp_kv_cache_interleave_size * dcp_size)
    base = full_chunks * cp_kv_cache_interleave_size

    remainder_total = seq_lens - base * dcp_size
    remainder = np.clip(
        remainder_total - dcp_rank * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    return base + remainder
