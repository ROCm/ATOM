# SPDX-License-Identifier: MIT
"""The prompt block-hash chain, kept with a tokenizer-cached prefix.

`prefix_hash.chained_block_hashes` of a prompt depends only on its tokens, so a
conversation's next turn -- which shares the cached prefix's tokens up to the
tokenizer's matched cut -- can take those blocks' hashes from the cached record
and hash only the blocks after them. These are content hashes: they say nothing
about whether a block's KV is still resident (BlockManager decides that).

A record lives inside the tokenizer cache's stored prefix, under its byte
budget, and refers to no earlier record, so evicting an old turn frees it.
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
