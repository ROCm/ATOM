# SPDX-License-Identifier: MIT
"""The prompt block-hash chain, kept with a tokenizer-cached prefix.

`prefix_hash.chained_block_hashes` of a prompt depends only on its tokens, so a
conversation's next turn -- which shares the cached prefix's tokens up to the
tokenizer's matched cut -- can take those blocks' hashes from the cached record
and hash only the blocks after them. These are content hashes: they say nothing
about whether a block's KV is still resident (BlockManager decides that).

A record lives inside the tokenizer cache's stored prefix, under its byte
budget, and refers to no earlier record, so evicting an old turn frees it.
The LMCache chunk chain at the end of this module is kept the same way.
"""

import array
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixBlockHashes:
    """`chained_block_hashes(tokens[:num_tokens], block_size, 0, n, seed)` for
    the record's complete blocks, packed as little-endian u64."""

    block_size: int
    seed: int
    num_tokens: int
    hashes: bytes

    @property
    def payload_bytes(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.hashes)
            + sum(
                sys.getsizeof(v) for v in (self.block_size, self.seed, self.num_tokens)
            )
        )

    @property
    def num_blocks(self) -> int:
        return len(self.hashes) // 8

    def as_array(self) -> array.array:
        values = array.array("Q")
        values.frombytes(self.hashes)
        return values

    def truncated(self, num_tokens: int) -> "PrefixBlockHashes":
        """The complete blocks of the first `num_tokens` tokens, for storing
        up to the tokenizer's cut: the wrapper's EOS and the unfinished tail
        must never seed a later turn."""
        if not 0 <= num_tokens <= self.num_tokens:
            raise ValueError("invalid stored prefix length")
        return PrefixBlockHashes(
            self.block_size,
            self.seed,
            num_tokens,
            self.hashes[: num_tokens // self.block_size * 8],
        )


def extend_block_hashes(tokens, block_size, seed=-1, *, base=None, shared_tokens=0):
    """`tokens`' block hashes, taking the first blocks from `base`.

    `shared_tokens` is how many leading tokens `tokens` shares with `base`'s,
    and must come from SegmentTokenizer's matched added-token cut -- an
    internal fact, never a client-supplied cached-token count. Only complete
    blocks inside it are reused; the rest is hashed here.
    """
    from atom.model_engine.prefix_hash import chained_block_hashes

    if block_size <= 0:
        raise ValueError("invalid block size")
    if not 0 <= shared_tokens <= len(tokens):
        raise ValueError("invalid shared token count")
    n = len(tokens) // block_size
    keep = 0
    if base is not None and (base.block_size, base.seed) == (block_size, seed):
        if shared_tokens > base.num_tokens:
            raise ValueError("shared prefix exceeds its stored record")
        keep = min(shared_tokens // block_size, base.num_blocks, n)
    packed = array.array("Q")
    if keep:
        packed.frombytes(base.hashes[: keep * 8])
    packed.extend(
        chained_block_hashes(
            tokens, block_size, keep, n, packed[-1] if packed else seed
        )
    )
    return PrefixBlockHashes(block_size, seed, len(tokens), packed.tobytes())


# -- the LMCache chunk chain ---------------------------------------------------
#
# With an LMCache MP offload whose server takes client-computed chunk hashes,
# every lookup, save and load names the prompt by LMCache's own chain instead
# of its tokens. It is kept beside the block chain above, in the same record and
# under the same budget, so a later turn hashes only its new chunks. The engine
# checks the shipped chain against LMCache's hasher before adopting it
# (`atom.kv_transfer.offload.mp.chunk_hashes.prompt_chunk_hashes`).

# Bytes per LMCache chunk hash (blake3).
LMCACHE_CHUNK_HASH_BYTES = 32


def _lmcache_none_hash() -> bytes:
    """The hash LMCache chains a prompt's first chunk from: its blake3
    `TokenHasher.none_hash`, the hash of prefix 0 and the one token 0."""
    import blake3

    return blake3.blake3((0).to_bytes(8, "big", signed=True) + bytes(4)).digest()


def lmcache_chunk_chain(tokens, chunk_size: int, prev: bytes | None = None) -> bytes:
    """LMCache's chunk hashes of `tokens`' complete chunks, concatenated.

    Chunk `i` hashes chunk `i - 1`'s hash (`prev` before the first; None starts
    from LMCache's none hash) followed by its tokens as big-endian u32, as an
    LMCache server hashing with blake3 does.
    """
    import blake3
    import numpy as np

    n = len(tokens) // chunk_size
    if n == 0:
        return b""
    raw = memoryview(np.asarray(tokens[: n * chunk_size], dtype=">u4").tobytes())
    step = 4 * chunk_size
    h = _lmcache_none_hash() if prev is None else bytes(prev)
    out = bytearray()
    for off in range(0, len(raw), step):
        hasher = blake3.blake3(h)
        hasher.update(raw[off : off + step])
        h = hasher.digest()
        out += h
    return bytes(out)


@dataclass(frozen=True, slots=True)
class LMCacheChunkHashes:
    """`lmcache_chunk_chain(tokens[:num_tokens], chunk_size)`."""

    chunk_size: int
    num_tokens: int
    hashes: bytes

    @property
    def payload_bytes(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.hashes)
            + sum(sys.getsizeof(v) for v in (self.chunk_size, self.num_tokens))
        )

    @property
    def num_chunks(self) -> int:
        return len(self.hashes) // LMCACHE_CHUNK_HASH_BYTES

    def truncated(self, num_tokens: int) -> "LMCacheChunkHashes":
        """The complete chunks of the first `num_tokens` tokens (as
        `PrefixBlockHashes.truncated`)."""
        if not 0 <= num_tokens <= self.num_tokens:
            raise ValueError("invalid stored prefix length")
        return LMCacheChunkHashes(
            self.chunk_size,
            num_tokens,
            self.hashes[: num_tokens // self.chunk_size * LMCACHE_CHUNK_HASH_BYTES],
        )


def extend_lmcache_chunk_hashes(tokens, chunk_size, *, base=None, shared_tokens=0):
    """`tokens`' LMCache chunk chain, taking the first chunks from `base`.

    `shared_tokens` is as for `extend_block_hashes`: the tokenizer's matched
    cut, never a client-supplied count.
    """
    if chunk_size <= 0:
        raise ValueError("invalid chunk size")
    if not 0 <= shared_tokens <= len(tokens):
        raise ValueError("invalid shared token count")
    n = len(tokens) // chunk_size
    keep = 0
    if base is not None and base.chunk_size == chunk_size:
        if shared_tokens > base.num_tokens:
            raise ValueError("shared prefix exceeds its stored record")
        keep = min(shared_tokens // chunk_size, base.num_chunks, n)
    kept = base.hashes[: keep * LMCACHE_CHUNK_HASH_BYTES] if keep else b""
    new = lmcache_chunk_chain(
        tokens[keep * chunk_size : n * chunk_size],
        chunk_size,
        kept[-LMCACHE_CHUNK_HASH_BYTES:] if keep else None,
    )
    return LMCacheChunkHashes(chunk_size, len(tokens), kept + new)
