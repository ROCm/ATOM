# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Classify P/D producer connectors without importing attention backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import DEFAULT_SHARDED_STAGING_WORKERS
from atom.utils import envs

# Connectors that push KV across the P/D boundary. Offload backends are not
# producers even when ``kv_role`` is omitted (they default to ``offload``).
_PD_TRANSFER_CONNECTORS = frozenset({"mooncake", "moriio"})
# Only Mooncake consumes DSA index staging callbacks / pool slots.
_INDEX_STAGING_CONNECTORS = frozenset({"mooncake"})
# MLA staging: one slot of this size per send worker, the pool capped in bytes
# so a large ``num_worker_threads`` makes workers share slots.
MLA_STAGING_SLOT_BYTES = 8 << 20
MLA_STAGING_POOL_BYTES = 256 << 20


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
    return send_worker_count(connectors[0])


def send_worker_count(connector: dict) -> int:
    """A P/D producer connector entry's validated ``num_worker_threads``."""

    count = connector.get("num_worker_threads", DEFAULT_SHARDED_STAGING_WORKERS)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            "P/D producer num_worker_threads must be a positive integer, "
            f"got {count!r}"
        )
    return count


def mla_staging_slot_count(num_send_workers: int, slot_bytes: int) -> int:
    """MLA staging slots for one producer: one per send worker, capped in bytes.

    ``MLA_STAGING_POOL_BYTES`` bounds the pool, so a large
    ``num_worker_threads`` makes workers share slots instead of growing HBM.
    """

    return min(num_send_workers, max(1, MLA_STAGING_POOL_BYTES // slot_bytes))


def mla_staging_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for Mooncake MLA staging pools.

    An upper bound of what ``MooncakeConnector`` allocates after the KV cache
    is sized: every Mooncake producer entry on an MLA model without producer
    DCP gets a pool of at most ``mla_staging_slot_count`` slots of
    ``MLA_STAGING_SLOT_BYTES``.
    """

    if (
        not envs.ATOM_PD_MLA_STAGING
        or getattr(config, "decode_context_parallel_size", 1) > 1
        or not getattr(getattr(config, "hf_config", None), "kv_lora_rank", None)
    ):
        return 0
    return sum(
        min(
            send_worker_count(connector) * MLA_STAGING_SLOT_BYTES,
            max(MLA_STAGING_POOL_BYTES, MLA_STAGING_SLOT_BYTES),
        )
        for connector in _producer_connectors(config, _INDEX_STAGING_CONNECTORS)
    )
