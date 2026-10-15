# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chained prompt-block hash, shared by the frontend and the BlockManager.
Must match `BlockManager.compute_hash` bit for bit."""

import array

import numpy as np
import xxhash

from atom.utils import envs

_xxh64 = xxhash.xxh64_intdigest


def chained_block_hashes(
    token_ids, hash_block_size: int, start: int, end: int, prev: int
) -> list[int]:
    """Chained hashes of hash blocks `[start, end)` of `token_ids`, from `prev`
    (block `start - 1`'s hash, or the seed; `-1`: no parent). Same bytes as
    `BlockManager.compute_hash`; a short final block hashes what it holds."""
    if end <= start:
        return []
    step = hash_block_size * 8
    raw = memoryview(
        np.asarray(
            token_ids[start * hash_block_size : end * hash_block_size],
            dtype=np.int64,
        ).tobytes()
    )
    out: list[int] = []
    append = out.append
    h = prev
    for off in range(0, (end - start) * step, step):
        block = raw[off : off + step]
        h = _xxh64(block) if h == -1 else _xxh64(h.to_bytes(8, "little") + block)
        append(h)
    return out


def ship_prompt_hash_chain(seq, hash_block_size: int) -> None:
    """Attach `seq`'s full-prompt-block hash chain as `(hash_block_size,
    cache_seed, array("Q"))`, for `BlockManager._adopt_shipped_chain`."""
    if not envs.ATOM_FRONTEND_PREFIX_HASH or hash_block_size <= 0:
        return
    n = seq.num_prompt_tokens // hash_block_size
    chain = chained_block_hashes(seq.token_ids, hash_block_size, 0, n, seq.cache_seed)
    seq.prompt_hash_chain = (hash_block_size, seq.cache_seed, array.array("Q", chain))
