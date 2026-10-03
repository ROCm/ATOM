# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Store keys for KV chunks: a layout namespace, a rank, and a prefix hash.

One Store object holds one PP/TP rank's bytes of one ``chunk_tokens`` chunk of
a prompt, exactly as the dense codec packs them. Its key is

    ``{namespace}/w{rank}of{world}/{digest}``

* ``namespace`` fingerprints everything that changes those bytes -- model,
  dtypes, block and chunk size, world, the HF geometry, the speculative config
  and each PP stage's layer span -- so a Store shared by differently configured
  servers never serves one of them another's layout.
* ``digest`` is the chunk's link in a prefix chain over the prompt's tokens, so
  a chunk's key names the whole prefix up to and including it.

The scheduler (which has no GPU and no KV tensors) and every worker compute the
namespace from the same ``Config``; only the scheduler hashes tokens, and the
digests travel to the workers in the request metadata.
"""

from __future__ import annotations

import array
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from atom.kv_transfer.offload import config as offcfg

DIGEST_BYTES = 16
_CHAIN_PERSON = b"atom-kv-chain-v1"
_MEDIA_SEED_PERSON = b"atom-kv-media-v1"
_NAMESPACE_PREFIX = "atomkv1-"
# Bumped when the dense codec's per-chunk byte layout changes.
CODEC_ID = "dense-opaque-block-v1"
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1


def pp_stage_layer_spans(config: Any) -> list[tuple[int, int]]:
    """Each PP stage's ``[start, end)`` slice of the target model's layers.

    Follows ``get_pp_indices``, so ``VLLM_PP_LAYER_PARTITION`` applies.
    """
    num_hidden_layers = int(config.hf_config.num_hidden_layers)
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if pp_size <= 1:
        return [(0, num_hidden_layers)]

    from atom.models.utils import get_pp_indices

    return [
        tuple(get_pp_indices(num_hidden_layers, pp_rank, pp_size))
        for pp_rank in range(pp_size)
    ]


def store_namespace(config: Any, chunk_tokens: int) -> str:
    """The key prefix every chunk of this server's KV layout is stored under.

    Built on the offload page namespace, which already covers the model,
    dtypes, block/chunk size, world (PP x TP), HF geometry and speculative
    config, plus what it lacks for a Store that outlives the server: the PP
    layer split -- PP4 x TP1 and PP1 x TP4 have the same world -- the online
    quantization, and the codec's byte layout.
    """
    world = offcfg.lmcache_replica_world_size(config)
    document = {
        "page": offcfg.build_page_namespace(
            config, SimpleNamespace(chunk_size=int(chunk_tokens)), world
        ),
        "pp_layers": [list(span) for span in pp_stage_layer_spans(config)],
        "online_quant": json.dumps(
            getattr(config, "online_quant_config", None), sort_keys=True, default=str
        ),
        "codec": CODEC_ID,
    }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.blake2b(canonical, digest_size=12).hexdigest()
    return f"{_NAMESPACE_PREFIX}{digest}"


def chain_seed(cache_seed: int | None = None) -> bytes:
    """The link before a prompt's first chunk.

    Sixteen zero bytes for a text prompt. A multimodal prompt carries its
    media's content identity (``Sequence.cache_seed``) because its placeholder
    tokens are the same for every image: without it two prompts that differ
    only in their pictures would share every key.
    """
    if cache_seed is None or int(cache_seed) == -1:
        return bytes(DIGEST_BYTES)
    return hashlib.blake2b(
        int(cache_seed).to_bytes(16, "little", signed=True),
        digest_size=DIGEST_BYTES,
        person=_MEDIA_SEED_PERSON,
    ).digest()


def chunk_hash_chain(
    token_ids: Any,
    chunk_tokens: int,
    *,
    num_tokens: int | None = None,
    previous: bytes = b"",
    seed: bytes | None = None,
) -> bytes:
    """Digests of every full chunk of ``token_ids[:num_tokens]``, concatenated.

    ``h_i = blake2b(h_{i-1} || int32-LE(tokens[i*C:(i+1)*C]))`` with a 16-byte
    digest, starting from ``seed`` (``chain_seed()`` by default). ``previous``
    holds the digests of a prefix already hashed; only the chunks after it are
    hashed, so a caller may extend a chain incrementally. Any partial chunk at
    the end is left out.

    ``token_ids`` may be a list, an ``array("i")`` -- read in place, without a
    copy -- or an integer ndarray.

    Raises:
        ValueError: ``previous`` is not whole digests or covers more chunks
            than the tokens hold, or a token does not fit in int32.
    """
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if len(previous) % DIGEST_BYTES:
        raise ValueError("previous chunk digests must be whole 16-byte digests")
    tokens = _int32_tokens(token_ids)
    if num_tokens is not None:
        tokens = tokens[: int(num_tokens)]
    total = len(tokens) // chunk_tokens
    done = len(previous) // DIGEST_BYTES
    if done > total:
        raise ValueError(
            f"previous digests cover {done} chunks but the tokens hold {total}"
        )
    if done == total:
        return bytes(previous)
    link = previous[-DIGEST_BYTES:] if done else (seed or chain_seed())
    if len(link) != DIGEST_BYTES:
        raise ValueError("a chain seed is one 16-byte digest")
    raw = memoryview(tokens[done * chunk_tokens : total * chunk_tokens].tobytes())
    stride = chunk_tokens * 4
    out = bytearray(previous)
    for offset in range(0, len(raw), stride):
        digest = hashlib.blake2b(link, digest_size=DIGEST_BYTES, person=_CHAIN_PERSON)
        digest.update(raw[offset : offset + stride])
        link = digest.digest()
        out += link
    return bytes(out)


def _int32_tokens(token_ids: Any) -> np.ndarray:
    """Token ids as little-endian int32, without copying an ``array("i")``."""
    if (
        isinstance(token_ids, array.array)
        and token_ids.typecode == "i"
        and token_ids.itemsize == 4
    ):
        # The view must not outlive the caller: an array exporting its
        # buffer cannot grow, and the scheduler appends to `Sequence.token_ids`.
        tokens = np.frombuffer(token_ids, dtype=np.int32)
    else:
        tokens = np.asarray(token_ids)
        if tokens.size == 0:
            tokens = tokens.astype(np.int32)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
            raise ValueError("token ids must be a flat sequence of integers")
        if tokens.dtype != np.int32 and (
            int(tokens.min()) < _INT32_MIN or int(tokens.max()) > _INT32_MAX
        ):
            raise ValueError("token ids must fit in int32")
    return tokens.astype("<i4", copy=False)


def chunk_digest(hashes: bytes, index: int) -> bytes:
    """The 16-byte digest of chunk ``index`` in a concatenated chain."""
    return hashes[index * DIGEST_BYTES : (index + 1) * DIGEST_BYTES]


def rank_key_prefix(namespace: str, rank: int, world: int) -> str:
    """The part of a chunk key every chunk of one rank shares."""
    return f"{namespace}/w{int(rank)}of{int(world)}/"


def chunk_key(namespace: str, rank: int, world: int, digest: bytes) -> str:
    """Store key of one rank's object for the chunk whose digest is ``digest``."""
    return rank_key_prefix(namespace, rank, world) + bytes(digest).hex()


def chunk_keys(
    namespace: str, rank: int, world: int, hashes: bytes, start: int, end: int
) -> list[str]:
    """Keys of one rank's chunks ``[start, end)`` of a concatenated chain."""
    prefix = rank_key_prefix(namespace, rank, world)
    return [prefix + chunk_digest(hashes, index).hex() for index in range(start, end)]


def probe_key(namespace: str, rank: int, nonce: str) -> str:
    """A key outside every chunk key, for the worker's startup round trip."""
    return f"{namespace}/probe/w{int(rank)}/{nonce}"
