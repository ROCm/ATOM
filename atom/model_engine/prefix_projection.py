# SPDX-License-Identifier: MIT
"""Immutable hash views of a tokenizer's exactly matched token prefix.

These records describe content, never KV residency. A stored tokenizer prefix
owns its projection under the same ByteLRU budget. There are no parent-record
references: evicting an old turn cannot leave an unbounded ancestor chain alive.
"""

import array
import sys
from dataclasses import dataclass, replace

import numpy as np
import xxhash


def token_fingerprint(tokens) -> int:
    """Same native int32 token bytes as Sequence's prompt_fingerprint."""
    return xxhash.xxh64_intdigest(np.asarray(tokens, dtype=np.int32))


@dataclass(frozen=True, slots=True)
class PrefixProjection:
    """A versioned, pickleable content hint; the packed array is immutable."""

    block_size: int
    seed: int
    num_tokens: int
    fingerprint: int
    kv_hashes: bytes
    version: int = 1

    @property
    def payload_bytes(self):
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.kv_hashes)
            + sum(
                sys.getsizeof(v)
                for v in (
                    self.block_size,
                    self.seed,
                    self.num_tokens,
                    self.fingerprint,
                    self.version,
                )
            )
        )

    def kv_array(self):
        values = array.array("Q")
        values.frombytes(self.kv_hashes)
        return values

    def for_stored_prefix(self, tokens, length):
        """Keep only immutable complete blocks before the tokenizer's cut.

        Wrapper EOS and the uncommitted tail must never seed a later turn.
        """
        if not 0 <= length <= self.num_tokens:
            raise ValueError("invalid stored prefix length")
        return replace(
            self,
            num_tokens=length,
            fingerprint=token_fingerprint(tokens[:length]),
            kv_hashes=self.kv_hashes[: length // self.block_size * 8],
        )

    def matches(self, tokens):
        return (
            self.version == 1
            and self.num_tokens == len(tokens)
            and self.fingerprint == token_fingerprint(tokens)
        )


def project_prefix(tokens, block_size, seed=-1, *, base=None, shared_tokens=0):
    """Extend hashes from an exact tokenizer match, not a hash-only guess.

    `shared_tokens` must come from SegmentTokenizer's matched added-token cut;
    it is an internal contract, never a client-supplied cached-token count.
    Full arrays are still materialized for the existing ADD-frame protocol.
    """
    from atom.model_engine.prefix_hash import chained_block_hashes

    if block_size <= 0:
        raise ValueError("invalid prefix projection geometry")
    if shared_tokens < 0 or shared_tokens > len(tokens):
        raise ValueError("invalid shared token count")
    if base is None or base.version != 1:
        shared_tokens = 0
    elif shared_tokens > base.num_tokens:
        raise ValueError("shared prefix exceeds its stored record")
    n = len(tokens) // block_size
    keep = 0
    if base is not None and (base.block_size, base.seed) == (block_size, seed):
        keep = min(shared_tokens // block_size, len(base.kv_hashes) // 8, n)
    packed = array.array("Q")
    if keep:
        packed.frombytes(base.kv_hashes[: keep * 8])
    packed.extend(
        chained_block_hashes(
            tokens, block_size, keep, n, packed[-1] if packed else seed
        )
    )
    return PrefixProjection(
        block_size,
        seed,
        len(tokens),
        token_fingerprint(tokens),
        packed.tobytes(),
    )
