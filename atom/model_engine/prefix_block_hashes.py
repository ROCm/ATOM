# SPDX-License-Identifier: MIT
"""The prompt block-hash chain, kept with a tokenizer-cached prefix or, for a
prompt sent as token ids, with its tokens (`TokenPrefixChains`).

`prefix_hash.chained_block_hashes` of a prompt depends only on its tokens, so a
conversation's next turn -- which shares the cached prefix's tokens up to the
tokenizer's matched cut -- can take those blocks' hashes from the cached record
and hash only the blocks after them. These are content hashes: they say nothing
about whether a block's KV is still resident (BlockManager decides that).

A text prompt's record lives in the tokenizer cache's stored prefix, a token-id
prompt's in `TokenPrefixChains`' own ByteLRU; each counts toward its store's
byte budget and refers to no earlier record, so evicting an old turn frees it.
"""

import array
import logging
import sys
import threading
from bisect import bisect_left, insort
from dataclasses import dataclass
from operator import attrgetter
from typing import NamedTuple

import numpy as np

from atom.model_engine.byte_lru import ByteLRU

logger = logging.getLogger("atom")

# Reuses checked against a full `chained_block_hashes` before reuse is trusted.
_VERIFY_FIRST = 4


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
    and must come from SegmentTokenizer's matched added-token cut or
    TokenPrefixChains' token compare -- never a client-supplied count. Only
    complete blocks inside it are reused; the rest is hashed here.
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


class _TokenRecord(NamedTuple):
    """A stored token-id prompt: its int32 tokens as bytes, its chain, its LRU key."""

    tokens: bytes
    hashes: PrefixBlockHashes
    rid: int

    @property
    def payload_bytes(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.tokens)
            + self.hashes.payload_bytes
            + sys.getsizeof(self.rid)
        )


_TOKENS = attrgetter("tokens")


def _shared_tokens(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the common prefix of `a` and `b`."""
    n, start, step = min(len(a), len(b)), 0, 4096
    while start < n:
        end = min(start + step, n)
        diff = np.flatnonzero(a[start:end] != b[start:end])
        if len(diff):
            return start + int(diff[0])
        start, step = end, step * 2
    return n


class TokenPrefixChains:
    """Token-id prompts' chains: each reuses the complete blocks it shares with
    the stored prompt sharing the most leading tokens, then is stored itself."""

    def __init__(self, budget_bytes: int, block_size: int, enabled: bool = True):
        self.block_size = block_size
        self.enabled = enabled and budget_bytes > 0
        self._lock = threading.Lock()
        self._lru = ByteLRU(budget_bytes, on_evict=self._unlist)
        # Stored records in their tokens' bytes order, which is lexicographic: the
        # one sharing the most leading tokens with a prompt sorts beside the prompt.
        self._order: list[_TokenRecord] = []
        self._next_id = 0
        self._verified = 0
        self._counts = dict.fromkeys(
            ("prompts", "blocks_hashed", "blocks_reused", "blocks_rehashed"), 0
        )

    def stats(self) -> dict:
        """Prompts seen and their chains' complete blocks: hashed, reused from a
        stored prompt, and hashed again to check a reuse."""
        with self._lock:
            return {"enabled": self.enabled, **self._counts}

    def block_hashes(self, tokens) -> PrefixBlockHashes:
        """`chained_block_hashes` of `tokens`' complete blocks (seed -1), hashing
        only those past what it shares with a stored prompt."""
        from atom.model_engine.prefix_hash import chained_block_hashes

        bs, raw = self.block_size, np.asarray(tokens, dtype=np.int32).tobytes()
        toks = np.frombuffer(raw, dtype=np.int32)
        n = len(toks) // bs
        with self._lock:
            enabled, verify = self.enabled, self._verified < _VERIFY_FIRST
        base, shared = self._find(raw, toks) if enabled and n else (None, 0)
        chain = extend_block_hashes(
            toks, bs, base=None if base is None else base.hashes, shared_tokens=shared
        )
        reused, rehash = shared // bs, False
        if reused:  # check the first reuses, and any made while reuse turned off
            with self._lock:
                rehash = verify or not self.enabled
        if rehash:
            full = array.array("Q", chained_block_hashes(toks, bs, 0, n, -1)).tobytes()
        with self._lock:
            if rehash and full != chain.hashes:
                self._turn_off()
                chain = PrefixBlockHashes(bs, -1, len(toks), full)
            elif rehash and verify and self.enabled:
                self._verified += 1
            if enabled and self.enabled and n:
                whole = reused and reused == base.hashes.num_blocks
                self._store(raw, chain, base if whole else None)
            counts = self._counts
            counts["prompts"] += 1
            counts["blocks_hashed"] += n - reused
            counts["blocks_reused"] += reused
            counts["blocks_rehashed"] += n if rehash else 0
        return chain

    def _find(self, raw: bytes, toks: np.ndarray):
        """`(record, shared tokens)` of the stored prompt sharing the most leading
        tokens with `toks`, one of the two `raw` sorts between; `(None, 0)` if none."""
        with self._lock:
            i = bisect_left(self._order, raw, key=_TOKENS)
            near = self._order[max(i - 1, 0) : i + 1]
        best, shared = None, 0
        for record in near:
            count = _shared_tokens(toks, np.frombuffer(record.tokens, dtype=np.int32))
            if count > shared:
                best, shared = record, count
        if best is not None:
            with self._lock:
                self._lru.get(best.rid)
        return best, shared

    def _store(self, raw: bytes, chain, replaces) -> None:
        """Keep `raw` and its chain, in place of `replaces`, a record whose every
        block it reused (lock held)."""
        if replaces is not None and self._lru.pop(replaces.rid) is not None:
            self._unlist(replaces.rid, replaces)
        record = _TokenRecord(raw, chain, self._next_id)
        self._next_id += 1
        self._lru.put(record.rid, record, record.payload_bytes)
        if record.rid in self._lru:  # not larger than the whole budget
            insort(self._order, record, key=_TOKENS)

    def _unlist(self, _rid, record) -> None:
        """Drop `record` from the bytes order (lock held; also ByteLRU's eviction
        callback)."""
        i = bisect_left(self._order, record.tokens, key=_TOKENS)
        while self._order[i] is not record:
            i += 1
        del self._order[i]

    def _turn_off(self) -> None:
        """A reused chain differed from a full rehash: stop reusing (lock held)."""
        if self.enabled:
            logger.warning(
                "token prefix chains: a reused chain differs from a full rehash; "
                "reuse is off"
            )
        self.enabled = False
        self._lru.clear()
        self._order.clear()
