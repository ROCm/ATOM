# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chained prompt-block hash, shared by the frontend and the BlockManager
(`BlockManager.compute_hash` is `block_hash`)."""

import array
from typing import NamedTuple

import numpy as np
import xxhash

from atom.utils import envs

_xxh64 = xxhash.xxh64_intdigest


def hash_block_size_for(block_size: int, dcp_world_size: int) -> int:
    """Tokens per prefix-cache hash block: one block-table entry under DCP."""
    return block_size * dcp_world_size


def full_prompt_blocks(seq, hash_block_size: int) -> int:
    """Hash blocks `seq`'s prompt fills: the part of its chain that cannot change."""
    return seq.num_prompt_tokens // hash_block_size


def _token_bytes(token_ids) -> bytes:
    # int64 whatever the caller passes: `np.asarray` alone infers int32 from an
    # `array("i")` and int64 from a list.
    return np.asarray(token_ids, dtype=np.int64).tobytes()


def _link(prev: int, raw) -> int:
    """Hash of one block's token bytes `raw` after `prev` (-1: no parent)."""
    return _xxh64(raw) if prev == -1 else _xxh64(prev.to_bytes(8, "little") + raw)


def block_hash(token_ids, prev: int = -1) -> int:
    """Chained hash of one block's tokens."""
    return _link(prev, _token_bytes(token_ids))


def chained_block_hashes(
    token_ids, hash_block_size: int, start: int, end: int, prev: int
) -> list[int]:
    """`block_hash` of hash blocks `[start, end)` of `token_ids`, chained from
    `prev` (block `start - 1`'s hash, or the seed); a short final block hashes
    what it holds."""
    if end <= start:
        return []
    step = hash_block_size * 8
    raw = memoryview(
        _token_bytes(token_ids[start * hash_block_size : end * hash_block_size])
    )
    out: list[int] = []
    append = out.append
    h = prev
    for off in range(0, (end - start) * step, step):
        h = _link(h, raw[off : off + step])
        append(h)
    return out


def prompt_fingerprint(seq) -> int:
    """xxh64 of `seq`'s prompt token bytes, without a copy."""
    return _xxh64(memoryview(seq.token_ids)[: seq.num_prompt_tokens])


class ShippedChain(NamedTuple):
    """`seq.prompt_hash_chain`: its full-prompt-block chain as the frontend
    hashed it, and what it was minted from."""

    hash_block_size: int
    cache_seed: int
    num_prompt_tokens: int
    fingerprint: int  # `prompt_fingerprint`
    hashes: array.array  # "Q"


def ship_prompt_hash_chain(seq, hash_block_size: int) -> None:
    """Attach `seq`'s chain for `BlockManager._adopt_shipped_chain`."""
    if not envs.ATOM_FRONTEND_PREFIX_HASH or hash_block_size <= 0:
        return
    n = full_prompt_blocks(seq, hash_block_size)
    chain = chained_block_hashes(seq.token_ids, hash_block_size, 0, n, seq.cache_seed)
    seq.prompt_hash_chain = ShippedChain(
        hash_block_size,
        seq.cache_seed,
        seq.num_prompt_tokens,
        prompt_fingerprint(seq),
        array.array("Q", chain),
    )


def take_shipped_chains(seqs) -> list | None:
    """`seqs`' shipped chains, detached for the ADD frame that carries them
    beside the seqs (a pickled `Sequence` drops its chain); None if none has
    one. The frontend keeps no copy."""
    chains = [getattr(seq, "prompt_hash_chain", None) for seq in seqs]
    if not any(chains):
        return None
    for seq in seqs:
        seq.prompt_hash_chain = None
    return chains


def attach_shipped_chains(seqs, chains=None) -> None:
    """Hand an ADD frame's chains (`take_shipped_chains`) back to its seqs."""
    if chains:
        for seq, chain in zip(seqs, chains, strict=True):
            seq.prompt_hash_chain = chain
