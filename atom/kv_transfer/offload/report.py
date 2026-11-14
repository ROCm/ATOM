# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The offload tier as the Atomesh router reads it (`/kv_event_sources`).

From a Mooncake Store's report the router finds the masters to ask which of a
prompt's chunks the Store holds, under the keys `keys.chunk_keys` builds.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.mooncake_store import keys
from atom.kv_transfer.offload.mooncake_store.config import parse_mooncake_store_config
from atom.utils import envs


def offload_report(config: Any) -> dict[str, Any] | None:
    """The report of the Mooncake Store connector `config` runs, on its own
    view of `config` (a `multi` entry's); None without one."""
    for name, leaf in KVConnectorFactory.leaf_connectors(config) or ():
        if name == "mooncake_store":
            return _mooncake_store_report(leaf)
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
