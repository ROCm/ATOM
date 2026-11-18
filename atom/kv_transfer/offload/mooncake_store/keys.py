# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Store keys for KV chunks: the Mooncake Store's namespace, groups and marker.

The key scheme -- ``{namespace}/w{rank}of{world}/{digest}``, the layout
namespace and the prompt's chunk hash chain -- is the store-agnostic one of
:mod:`atom.kv_transfer.offload.kv_store.keys`, re-exported here. This module
adds what is the Mooncake Store's own:

* the namespace prefix, ``atomkv1-``, and the codec id of the layouts this
  connector moves (``layouts``);
* the groups: every rank's object of a chunk is put in one Mooncake group,
  ``{namespace}/group/{digest}``. The master evicts a group whole, so the ranks
  of a chunk do not outlive each other: a chunk any rank lacks is a miss, and
  the others' objects would only hold memory;
* the Store's taint marker key (see ``taint``).
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.offload.kv_store import keys as _store_keys
from atom.kv_transfer.offload.kv_store.keys import (
    DIGEST_BYTES,
    chain_seed,
    chunk_digest,
    chunk_hash_chain,
    chunk_keys,
    kv_layout_selectors,
    kv_storage_formats,
    pp_stage_layer_spans,
    probe_key,
    rank_key_prefix,
)
from atom.kv_transfer.offload.mooncake_store.layouts import mooncake_codec_id

_NAMESPACE_PREFIX = "atomkv1-"


def store_namespace(config: Any, chunk_tokens: int) -> str:
    """The key prefix every chunk of this server's KV layout is stored under.

    See :func:`atom.kv_transfer.offload.kv_store.keys.store_namespace`.
    """
    return _store_keys.store_namespace(
        config,
        chunk_tokens,
        prefix=_NAMESPACE_PREFIX,
        codec_id=mooncake_codec_id(config),
    )


def chunk_group_id(namespace: str, digest: bytes) -> str:
    """Mooncake group of every rank's object of the chunk ``digest`` names."""
    return f"{namespace}/group/{bytes(digest).hex()}"


def chunk_group_ids(namespace: str, hashes: bytes, start: int, end: int) -> list[str]:
    """Groups of chunks ``[start, end)`` of a concatenated chain, one per chunk."""
    return [
        chunk_group_id(namespace, chunk_digest(hashes, index))
        for index in range(start, end)
    ]


# Present in a Store that a put Mooncake gave up on may still write (see
# `taint`). Outside every namespace: the late write can land in an object of
# any namespace.
STORE_TAINT_KEY = "atomkv/store-tainted"

__all__ = [
    "DIGEST_BYTES",
    "STORE_TAINT_KEY",
    "chain_seed",
    "chunk_digest",
    "chunk_group_id",
    "chunk_group_ids",
    "chunk_hash_chain",
    "chunk_keys",
    "kv_layout_selectors",
    "kv_storage_formats",
    "pp_stage_layer_spans",
    "probe_key",
    "rank_key_prefix",
    "store_namespace",
]
