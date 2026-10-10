# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Store keys for KV chunks: a layout namespace, a rank, and a prefix hash.

One Store object holds one PP/TP rank's bytes of one ``chunk_tokens`` chunk of
a prompt, exactly as the dense codec packs them. Its key is

    ``{namespace}/w{rank}of{world}/{digest}``

* ``namespace`` fingerprints everything that changes those bytes -- model,
  dtypes and the formats this GPU stores them in, block and chunk size, world,
  the HF geometry, RoPE settings and overrides, the speculative config, each
  PP stage's layer span and the environment switches that pick the attention
  backend's KV layout -- so a Store shared by differently configured servers
  never serves one of them another's layout.
* ``digest`` is the chunk's link in a prefix chain over the prompt's tokens, so
  a chunk's key names the whole prefix up to and including it.

Every rank's object of a chunk is put in one Mooncake group,
``{namespace}/group/{digest}``. The master evicts a group whole, so the ranks
of a chunk do not outlive each other: a chunk any rank lacks is a miss, and
the others' objects would only hold memory.

The scheduler (which has no GPU and no KV tensors) and every worker compute the
namespace from the same ``Config``; only the scheduler hashes tokens, and the
digests travel to the workers in the request metadata.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from atom.kv_transfer.offload import config as offcfg
from atom.utils import envs

DIGEST_BYTES = 16
_CHAIN_PERSON = b"atom-kv-chain-v1"
_MEDIA_SEED_PERSON = b"atom-kv-media-v1"
_NAMESPACE_PREFIX = "atomkv1-"
# Bumped when the dense codec's per-chunk byte layout changes.
CODEC_ID = "dense-opaque-block-v1"


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


def kv_layout_selectors() -> dict[str, Any]:
    """The environment switches that rearrange the bytes inside a KV block.

    Each picks a layout the attention backend writes at the same block size --
    the segmented MLA cache (``ATOM_MLA_PAGE_SIZE``), the Triton MLA backend's
    shuffled one (``ATOM_USE_TRITON_MLA`` with
    ``ATOM_USE_TRITON_MLA_SHUFFLE_KV``), the MHA kernels' block
    (``ATOM_USE_UNIFIED_ATTN``) and the Triton MHA path's fp8 KV, quantized
    with one fixed scale instead of the per-token scales the default path
    writes (``ATOM_FORCE_ATTN_TRITON``) -- so neither the codec's opaque bytes
    nor the size a load checks tell them apart. The scheduler and the workers
    it spawns read the same environment. A switch a model does not use costs a
    miss across servers that differ in it, never another layout's KV.
    """
    return {
        "mla_page_size": int(envs.ATOM_MLA_PAGE_SIZE),
        "triton_mla": bool(envs.ATOM_USE_TRITON_MLA),
        "triton_mla_shuffle_kv": bool(envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV),
        "unified_attn": bool(envs.ATOM_USE_UNIFIED_ATTN),
        "force_attn_triton": bool(envs.ATOM_FORCE_ATTN_TRITON),
    }


def kv_storage_formats(config: Any) -> dict[str, str]:
    """The GPU arch and the torch dtypes the KV and index caches are stored in.

    The configured names do not say which bytes a block holds: AITER stores
    ``fp8`` as e4m3fnuz on gfx942 and e4m3fn on gfx950, one byte each, so a
    load's size check cannot tell a chunk saved on one from the other. A DSA
    index cache is fp8 even under a bf16 KV cache, and other arch-dependent
    choices (the MHA kernels' fixed fp8 scale) follow the arch too, so it is
    named as well. The scheduler resolves them as its workers do, with the
    same AITER on the same node, as the dense LMCache scheduler already does.
    A name AITER does not know (``fp4``) stands as it is.
    """
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx

    def resolve(name: Any) -> str:
        return str(dtypes.d_dtypes.get(str(name), name))

    kv_cache_dtype = getattr(config, "kv_cache_dtype", "auto")
    index_cache_dtype = getattr(config, "index_cache_dtype", None)
    return {
        "gfx": str(get_gfx()),
        "kv_cache": resolve(kv_cache_dtype),
        "index_cache": resolve(
            kv_cache_dtype if index_cache_dtype is None else index_cache_dtype
        ),
    }


def store_namespace(config: Any, chunk_tokens: int) -> str:
    """The key prefix every chunk of this server's KV layout is stored under.

    Built on the offload page namespace, which already covers the model path,
    the configured dtypes, block/chunk size, world (PP x TP), HF geometry and
    speculative config, plus what it lacks for a Store that outlives the
    server and is shared by others: the PP layer split -- PP4 x TP1 and
    PP1 x TP4 have the same world -- the online quantization, the KV layout
    the environment selects (`kv_layout_selectors`), the storage formats the
    GPU resolves them to (`kv_storage_formats`), the RoPE settings K is cached
    with, every ``--hf-overrides`` field, the checkpoint revision when the
    config names one, and the codec's byte layout.
    """
    world = offcfg.lmcache_replica_world_size(config)
    hf = config.hf_config
    # Taken when the Config was built: a worker changes the live hf_config
    # while it builds the model (`snapshot_rope_config`).
    rope = getattr(config, "offload_rope_config", None)
    if rope is None:
        rope = offcfg.snapshot_rope_config(hf)
    document = {
        "page": offcfg.build_page_namespace(
            config, SimpleNamespace(chunk_size=int(chunk_tokens)), world
        ),
        "pp_layers": [list(span) for span in pp_stage_layer_spans(config)],
        "online_quant": json.dumps(
            getattr(config, "online_quant_config", None), sort_keys=True, default=str
        ),
        "kv_layout": kv_layout_selectors(),
        "storage": kv_storage_formats(config),
        "rope": rope,
        "hf_overrides": json.dumps(
            getattr(config, "hf_overrides", None), sort_keys=True, default=str
        ),
        "revision": str(getattr(hf, "_commit_hash", None)),
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
    seed: bytes | None = None,
) -> bytes:
    """Digests of every full chunk of ``token_ids[:num_tokens]``, concatenated.

    ``h_i = blake2b(h_{i-1} || int32-LE(tokens[i*C:(i+1)*C]))`` with a 16-byte
    digest, starting from ``seed`` (``chain_seed()`` by default). Any partial
    chunk at the end is left out. An ``array("i")`` is read in place, without
    a copy.
    """
    chunk_tokens = int(chunk_tokens)
    # The view must not outlive the call: an array exporting its buffer
    # cannot grow, and the scheduler appends to `Sequence.token_ids`.
    tokens = np.asarray(token_ids, dtype="<i4")[:num_tokens]
    total = len(tokens) // chunk_tokens
    raw = memoryview(tokens[: total * chunk_tokens].tobytes())
    stride = chunk_tokens * 4
    link = seed or chain_seed()
    out = bytearray()
    for offset in range(0, len(raw), stride):
        digest = hashlib.blake2b(link, digest_size=DIGEST_BYTES, person=_CHAIN_PERSON)
        digest.update(raw[offset : offset + stride])
        link = digest.digest()
        out += link
    return bytes(out)


def chunk_digest(hashes: bytes, index: int) -> bytes:
    """The 16-byte digest of chunk ``index`` in a concatenated chain."""
    return hashes[index * DIGEST_BYTES : (index + 1) * DIGEST_BYTES]


def rank_key_prefix(namespace: str, rank: int, world: int) -> str:
    """The part of a chunk key every chunk of one rank shares."""
    return f"{namespace}/w{int(rank)}of{int(world)}/"


def chunk_keys(
    namespace: str, rank: int, world: int, hashes: bytes, start: int, end: int
) -> list[str]:
    """Keys of one rank's chunks ``[start, end)`` of a concatenated chain."""
    prefix = rank_key_prefix(namespace, rank, world)
    return [prefix + chunk_digest(hashes, index).hex() for index in range(start, end)]


def chunk_group_id(namespace: str, digest: bytes) -> str:
    """Mooncake group of every rank's object of the chunk ``digest`` names."""
    return f"{namespace}/group/{bytes(digest).hex()}"


def chunk_group_ids(namespace: str, hashes: bytes, start: int, end: int) -> list[str]:
    """Groups of chunks ``[start, end)`` of a concatenated chain, one per chunk."""
    return [
        chunk_group_id(namespace, chunk_digest(hashes, index))
        for index in range(start, end)
    ]


def probe_key(namespace: str, rank: int, nonce: str) -> str:
    """A key outside every chunk key, for the worker's startup round trip."""
    return f"{namespace}/probe/w{int(rank)}/{nonce}"
