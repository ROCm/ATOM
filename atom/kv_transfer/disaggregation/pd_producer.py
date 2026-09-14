# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Classify P/D producer connectors without importing attention backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import DEFAULT_SHARDED_STAGING_WORKERS

# Connectors that push KV across the P/D boundary. Offload backends are not
# producers even when ``kv_role`` is omitted (they default to ``offload``).
_PD_TRANSFER_CONNECTORS = frozenset({"mooncake", "moriio"})
# Only Mooncake consumes DSA index staging callbacks / pool slots.
_INDEX_STAGING_CONNECTORS = frozenset({"mooncake"})


def _canonical(connector: dict, *, path: str) -> str | None:
    try:
        return KVConnectorFactory.canonical_name(
            connector.get("kv_connector"), path=path
        )
    except (TypeError, ValueError):
        return None


def iter_connector_configs(
    transfer_config: Any,
) -> Iterator[tuple[dict, str]]:
    """Yield ``(connector_dict, path)`` for each real connector entry."""

    if not isinstance(transfer_config, dict) or not transfer_config:
        return
    name = _canonical(transfer_config, path="kv_transfer_config")
    if name is None:
        return
    if name == "multi":
        connectors = transfer_config.get("connectors")
        if not isinstance(connectors, (list, tuple)):
            return
        for index, connector in enumerate(connectors):
            if isinstance(connector, dict):
                yield connector, f"kv_transfer_config.connectors[{index}]"
        return
    yield transfer_config, "kv_transfer_config"


def _is_named_pd_producer(connector: dict, *, path: str, names: frozenset[str]) -> bool:
    name = _canonical(connector, path=path)
    if name not in names:
        return False
    return connector.get("kv_role", "kv_producer") == "kv_producer"


def _producer_connectors(config, names: frozenset[str]) -> tuple[dict, ...]:
    """Configured ``kv_producer`` entries whose connector name is in ``names``."""
    transfer_config = getattr(config, "kv_transfer_config", None)
    return tuple(
        connector
        for connector, path in iter_connector_configs(transfer_config)
        if _is_named_pd_producer(connector, path=path, names=names)
    )


def pd_producer_configured(config) -> bool:
    return bool(_producer_connectors(config, _PD_TRANSFER_CONNECTORS))


def mooncake_pd_producer_configured(config) -> bool:
    return bool(_producer_connectors(config, _INDEX_STAGING_CONNECTORS))


def index_staging_pool_size(config) -> int:
    """Slots for one Mooncake producer's send-worker concurrency.

    Multiple Mooncake producer connector entries in one process would share
    one ``KVTransferTensors`` buffer while maintaining independent free lists,
    so that local ``MultiConnector`` configuration is refused here. Independent
    producer server processes in a multi-P/one-D deployment remain supported.
    """

    connectors = _producer_connectors(config, _INDEX_STAGING_CONNECTORS)
    if len(connectors) > 1:
        raise ValueError(
            "DSA index staging cannot be shared by multiple Mooncake P/D producer "
            "connector entries in one process; list only one kv_producer mooncake "
            "connector per local MultiConnector configuration"
        )
    if not connectors:
        return 0
    count = connectors[0].get("num_worker_threads", DEFAULT_SHARDED_STAGING_WORKERS)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            "P/D producer num_worker_threads must be a positive integer, "
            f"got {count!r}"
        )
    return count
