# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Settings of the UMBP offload connector (``umbp_store``).

They come from the connector's ``kv_connector_extra_config`` -- the
top-level transfer config when it has none -- under keys prefixed
``umbp_store.``. The scheduler and every worker parse the same dict, so both
sides agree on the chunk size and on the server they address. No environment
variable is read.

Also what the server holds: the layouts this connector moves (dense and M3
complete PAGEs; ``UMBP_LAYOUTS``) and the key namespace, the store-agnostic one
of :mod:`atom.kv_transfer.offload.kv_store.keys` under UMBP's own prefix, so
no key of this connector is ever a Mooncake Store key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.kv_store import keys as store_keys
from atom.kv_transfer.offload.kv_store.layouts import (
    StoreLayouts,
    check_store_engine,
)

KEY_PREFIX = "umbp_store."
_MIB = 1 << 20
# UMBP's own: no key of this connector is ever a Mooncake Store key.
NAMESPACE_PREFIX = "atomumbp1-"

UMBP_LAYOUTS = StoreLayouts(
    ("dense", "m3"),
    refusal="the UMBP offload moves only the dense KV layout or M3 PAGE layout",
)


def store_namespace(config: Any, chunk_tokens: int) -> str:
    """The key prefix every chunk of this engine's KV layout is stored under."""
    return store_keys.store_namespace(
        config,
        chunk_tokens,
        prefix=NAMESPACE_PREFIX,
        codec_id=UMBP_LAYOUTS.codec_id(config),
    )


@dataclass(frozen=True)
class UMBPStoreOffloadConfig:
    """Parsed ``umbp_store.*`` settings."""

    # The standalone server's gRPC address, e.g. ``unix:///tmp/umbp/server.sock``.
    # Started outside ATOM; every worker and the scheduler connect to it.
    address: str
    # How long a client waits for the server to answer when it connects, and
    # how long the startup probe asks again while its put is answered False.
    startup_timeout_ms: int
    # How long a lookup waits for the server; past it every key is absent and
    # the scheduler stops asking for a while. Lookups block the scheduler
    # thread, so 0 (Mori's "wait for ever") is refused.
    lookup_timeout_ms: int
    chunk_tokens: int
    load_pool_mib: int
    save_pool_mib: int
    save_abandon_timeout_s: float

    @property
    def load_pool_bytes(self) -> int:
        return self.load_pool_mib * _MIB

    @property
    def save_pool_bytes(self) -> int:
        return self.save_pool_mib * _MIB


# The known setting names, after KEY_PREFIX.
_FIELDS = tuple(field.name for field in fields(UMBPStoreOffloadConfig))


def store_chunk_tokens(kv_transfer_config: Mapping[str, Any] | None) -> int:
    """The transfer granularity shared by routing and the UMBP connector."""
    return offcfg.connector_integer(
        offcfg.connector_extras(kv_transfer_config), f"{KEY_PREFIX}chunk_tokens", 256
    )


def parse_umbp_store_config(
    kv_transfer_config: Mapping[str, Any] | None,
) -> UMBPStoreOffloadConfig:
    """Parse and validate the ``umbp_store.*`` keys of a transfer config.

    Keys outside that prefix belong to other parts of ATOM and are ignored.

    Raises:
        ValueError: An unknown ``umbp_store.*`` key, a value of the wrong type
            or out of range, or no server address.
    """
    extra = offcfg.connector_extras(kv_transfer_config)
    values: dict[str, Any] = {}
    for key, value in extra.items():
        if not (isinstance(key, str) and key.startswith(KEY_PREFIX)):
            continue
        name = key[len(KEY_PREFIX) :]
        if name not in _FIELDS:
            raise ValueError(
                f"unknown UMBP offload setting {key!r}; known: "
                f"{', '.join(KEY_PREFIX + field for field in _FIELDS)}"
            )
        values[name] = value

    address = values.get("address")
    if address is None:
        raise ValueError(
            f"the UMBP offload needs {KEY_PREFIX}address: the standalone "
            "server's address, e.g. unix:///tmp/umbp/server.sock"
        )
    return UMBPStoreOffloadConfig(
        address=offcfg._strict_text(f"{KEY_PREFIX}address", address),
        startup_timeout_ms=offcfg._strict_integer(
            f"{KEY_PREFIX}startup_timeout_ms",
            values.get("startup_timeout_ms", 10000),
            minimum=1,
        ),
        lookup_timeout_ms=offcfg._strict_integer(
            f"{KEY_PREFIX}lookup_timeout_ms",
            values.get("lookup_timeout_ms", 1000),
            minimum=1,
        ),
        chunk_tokens=offcfg.connector_integer(extra, f"{KEY_PREFIX}chunk_tokens", 256),
        load_pool_mib=offcfg._strict_integer(
            f"{KEY_PREFIX}load_pool_mib", values.get("load_pool_mib", 1024), minimum=1
        ),
        save_pool_mib=offcfg._strict_integer(
            f"{KEY_PREFIX}save_pool_mib", values.get("save_pool_mib", 256), minimum=1
        ),
        save_abandon_timeout_s=offcfg._positive_seconds(
            f"{KEY_PREFIX}save_abandon_timeout_s",
            values.get("save_abandon_timeout_s", 300.0),
        ),
    )


def check_engine_compatibility(cfg: UMBPStoreOffloadConfig, config: Any) -> None:
    """Accept dense and M3 PAGE bytes; refuse stateful layouts, DCP and a TP
    group over several nodes.

    M3 uses complete PAGE regions, including its NSA index cache. Stateful
    layouts require a separate state tier, while a DCP rank's pages cover a
    virtual block that the chunk keys do not name. One server serves one
    node: with TP over several nodes each node's ranks would put to their own
    server while the scheduler asks only its own for every rank's keys, so
    every lookup would miss.

    Raises:
        ValueError: The model uses neither dense nor M3 PAGE layout, DCP > 1,
            TP spans nodes, or the chunk is not a whole number of KV blocks.
    """
    UMBP_LAYOUTS.require(offcfg.select_offload_layout(config))
    if config.is_multinode_tp:
        raise ValueError(
            "the UMBP offload serves one node: a tensor-parallel group over "
            "several nodes is not supported (each node's ranks reach only "
            "that node's UMBP standalone server)"
        )
    check_store_engine(
        config,
        cfg.chunk_tokens,
        offload="the UMBP offload",
        chunk_setting=f"{KEY_PREFIX}chunk_tokens",
    )


__all__ = [
    "KEY_PREFIX",
    "UMBP_LAYOUTS",
    "UMBPStoreOffloadConfig",
    "check_engine_compatibility",
    "parse_umbp_store_config",
    "store_namespace",
]
