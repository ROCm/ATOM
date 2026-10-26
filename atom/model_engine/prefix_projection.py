# SPDX-License-Identifier: MIT
"""Immutable hash views of a tokenizer's exactly matched token prefix.

These records describe content, never KV residency. A stored tokenizer prefix
owns its projections under the same ByteLRU budget. There are no parent-record
references: evicting an old turn cannot leave an unbounded ancestor chain alive.
"""

import array
from dataclasses import dataclass, replace
import hashlib
import struct
import sys

import numpy as np
import xxhash

PAGE_SEED = hashlib.sha256(b"atom-dualpath-page-v1").digest()


def page_chunk_from_config(kv_config):
    """Resolve the PAGE view through a MultiConnector, without reading secrets.

    Differing child geometries use the ordinary adapter hashing fallback.
    """
    chunks = set()

    def visit(config):
        if not isinstance(config, dict):
            return
        dualpath = (config.get("kv_connector_extra_config") or {}).get("dualpath")
        if isinstance(dualpath, dict):
            chunk = int(dualpath.get("chunk_tokens", 256))
            if chunk > 0:
                chunks.add(chunk)
        for child in config.get("connectors") or ():
            visit(child)

    visit(kv_config)
    return next(iter(chunks)) if len(chunks) == 1 else 0


def token_fingerprint(tokens) -> int:
    """Same native int32 token bytes as Sequence's prompt_fingerprint."""
    return xxhash.xxh64_intdigest(np.asarray(tokens, dtype=np.int32))


def page_link(previous: bytes, tokens) -> bytes:
    values = [int(t) for t in tokens]
    if any(t < 0 or t > 0xFFFFFFFF for t in values):
        raise ValueError("invalid token id")
    return hashlib.sha256(previous + struct.pack(f"!{len(values)}I", *values)).digest()


@dataclass(frozen=True, slots=True)
class PrefixProjection:
    """A versioned, pickleable content hint; packed arrays are immutable."""

    block_size: int
    seed: int
    num_tokens: int
    fingerprint: int
    kv_hashes: bytes
    page_chunk: int = 0
    page_digests: bytes = b""
    reused_blocks: int = 0
    reused_pages: int = 0
    version: int = 1

    @property
    def payload_bytes(self):
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.kv_hashes)
            + sys.getsizeof(self.page_digests)
            + sum(
                sys.getsizeof(v)
                for v in (
                    self.block_size,
                    self.seed,
                    self.num_tokens,
                    self.fingerprint,
                    self.page_chunk,
                    self.reused_blocks,
                    self.reused_pages,
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
            page_digests=(
                self.page_digests[: length // self.page_chunk * 32]
                if self.page_chunk
                else b""
            ),
            reused_blocks=0,
            reused_pages=0,
        )

    def matches(self, tokens):
        return (
            self.version == 1
            and self.num_tokens == len(tokens)
            and self.fingerprint == token_fingerprint(tokens)
        )

    def for_lookup_prefix(self, tokens, length):
        """Bind a shorter restore query to this verified complete prompt.

        Native-state lookup excludes the last prompt token and rounds down to
        a checkpoint boundary. Validate the original content before rebinding
        its hashes: fingerprinting only the new slice could bless stale hashes
        from an unrelated prompt. The consumer still validates PAGE geometry,
        query fingerprint and sampled links, and checks actual PAGE/STATE data.
        """
        if (
            self.block_size <= 0
            or self.page_chunk < 0
            or not 0 <= length <= len(tokens)
            or not self.matches(tokens)
        ):
            return None
        if length == self.num_tokens:
            return self
        return self.for_stored_prefix(tokens, length)

    def checked_page_digests(self, tokens, chunk, check=lambda: None):
        """Validate an internal IPC hint before the external tier uses it.

        The full token fingerprint binds the prompt, and first/middle/last
        links check the representation, matching shipped HBM-chain guards.
        An incompatible hint is ignored, never interpreted as a cache miss.
        """
        check()
        n = len(tokens) // chunk
        if (
            self.page_chunk != chunk
            or len(self.page_digests) != n * 32
            or not self.matches(tokens)
        ):
            return None
        for i in ({0, n // 2, n - 1} if n else ()):
            check()
            prev = self.page_digests[(i - 1) * 32 : i * 32] if i else PAGE_SEED
            actual = page_link(prev, tokens[i * chunk : (i + 1) * chunk])
            if actual != self.page_digests[i * 32 : (i + 1) * 32]:
                return None
        check()
        return self.page_digests


def project_prefix(
    tokens, block_size, seed=-1, page_chunk=0, *, base=None, shared_tokens=0
):
    """Extend hashes from an exact tokenizer match, not a hash-only guess.

    `shared_tokens` must come from SegmentTokenizer's matched added-token cut;
    it is an internal contract, never a client-supplied cached-token count.
    Full arrays are still materialized for the existing ADD-frame protocol.
    """
    from atom.model_engine.prefix_hash import chained_block_hashes

    if block_size <= 0 or page_chunk < 0:
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

    pages = bytearray()
    keep_pages = 0
    if page_chunk:
        count = len(tokens) // page_chunk
        if base is not None and base.page_chunk == page_chunk:
            keep_pages = min(
                shared_tokens // page_chunk, len(base.page_digests) // 32, count
            )
        if keep_pages:
            pages.extend(base.page_digests[: keep_pages * 32])
        prev = bytes(pages[-32:]) if pages else PAGE_SEED
        for i in range(keep_pages, count):
            prev = page_link(prev, tokens[i * page_chunk : (i + 1) * page_chunk])
            pages.extend(prev)
    return PrefixProjection(
        block_size,
        seed,
        len(tokens),
        token_fingerprint(tokens),
        packed.tobytes(),
        page_chunk,
        bytes(pages),
        keep,
        keep_pages,
    )
