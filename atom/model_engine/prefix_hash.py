# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chained prompt-block hash, shared by the frontend and the BlockManager.

One definition for both sides because a chain the frontend computes and the
engine's index disagree on is not an error anyone sees: every lookup misses
and prefix caching silently stops working. `BlockManager.compute_hash` is the
per-block form of the same function and the reference the tests hold this to.
"""

import array

import numpy as np
import xxhash

from atom.utils import envs

_xxh64 = xxhash.xxh64_intdigest


def chained_block_hashes(
    token_ids, hash_block_size: int, start: int, end: int, prev: int
) -> list[int]:
    """Chained hashes of hash blocks `[start, end)` of `token_ids`.

    `prev` is block `start - 1`'s hash, or the chain seed for `start == 0`
    (`-1` hashes the first block with no parent). Each hash is
    `xxh64(parent as 8 little-endian bytes + the block's tokens as int64)`,
    the bytes `BlockManager.compute_hash` hashes, so a short final block
    hashes what it holds. ~4x cheaper per block than `compute_hash` in a loop:
    the int64 conversion is done once for the whole range and each block is a
    view into it.
    """
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
    """Attach `seq`'s full-prompt-block hash chain, for the engine to adopt.

    Runs where the request is built (the API server's preprocessing executor,
    or the offline engine's caller), so the engine thread does not hash the
    prompt itself. Carried as `(hash_block_size, cache_seed, array("Q"))`: the
    engine checks the first two against its own and the length against the
    prompt, and hashes the prompt itself on any disagreement
    (`BlockManager._adopt_shipped_chain`). Covers only full prompt blocks,
    the ones a chain may cache.
    """
    if not envs.ATOM_FRONTEND_PREFIX_HASH or hash_block_size <= 0:
        return
    n = seq.num_prompt_tokens // hash_block_size
    chain = chained_block_hashes(seq.token_ids, hash_block_size, 0, n, seq.cache_seed)
    seq.prompt_hash_chain = (hash_block_size, seq.cache_seed, array.array("Q", chain))
