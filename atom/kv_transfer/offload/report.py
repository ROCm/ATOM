# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The offload tier as a router outside ATOM reads it (`/kv_event_sources`).

From a Mooncake Store's report a router can ask the masters which of a prompt's
chunks the Store holds, under the keys `keys.chunk_keys` builds; LMCache's
reports only the chunk a load moves and the shortest load ATOM makes.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.mooncake_store import keys
from atom.kv_transfer.offload.mooncake_store.config import parse_mooncake_store_config
from atom.utils import envs

_LMCACHE_CONNECTORS = ("lmcache_offload", "lmcache_mp")


def offload_report(config: Any) -> dict[str, Any] | None:
    """The report of the offload connector `config` runs, None without one a
    router can read. Each value is what the connector computes, on its own view
    of `config` (a `multi` entry's)."""
    for name, leaf in KVConnectorFactory.leaf_connectors(config) or ():
        if name == "mooncake_store":
            return _mooncake_store_report(leaf)
        if name in _LMCACHE_CONNECTORS:
            return {
                "backend": "lmcache",
                "chunk_tokens": offcfg.lmcache_chunk_tokens(leaf.kv_transfer_config),
                "min_load_tokens": envs.OFFLOAD_MIN_LOAD_TOKENS,
            }
    return None


def _mooncake_store_report(config: Any) -> dict[str, Any]:
    store = parse_mooncake_store_config(config.kv_transfer_config)
    chunk_tokens = store.chunk_tokens
    return {
        "backend": "mooncake_store",
        "masters": list(dict.fromkeys(pool.master for pool in store.store_masters())),
        "namespace": keys.store_namespace(config, chunk_tokens),
        "chunk_tokens": chunk_tokens,
        "world": offcfg.lmcache_replica_world_size(config),
        "min_load_tokens": envs.OFFLOAD_MIN_LOAD_TOKENS,
        # The digest of a text prompt's first chunk when its tokens are
        # 0..chunk_tokens-1: a router that computes another must not read keys.
        "chain_check": keys.chunk_hash_chain(range(chunk_tokens), chunk_tokens).hex(),
    }
