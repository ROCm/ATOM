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
# Only Mooncake consumes staging callbacks / pool slots (DSA index and MLA).
_MOONCAKE_CONNECTORS = frozenset({"mooncake"})
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
    return bool(_producer_connectors(config, _MOONCAKE_CONNECTORS))


def index_staging_pool_size(config) -> int:
    """Slots for one Mooncake producer's send-worker concurrency.

    Multiple Mooncake producer connector entries in one process would share
    one ``KVTransferTensors`` buffer while maintaining independent free lists,
    so that local ``MultiConnector`` configuration is refused here. Independent
    producer server processes in a multi-P/one-D deployment remain supported.
    """

    connectors = _producer_connectors(config, _MOONCAKE_CONNECTORS)
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


def mla_staging_pool_bytes(num_send_workers: int) -> int:
    """HBM one Mooncake producer holds back for its MLA staging pool.

    One ``MLA_STAGING_SLOT_BYTES`` slot per send worker, capped at
    ``MLA_STAGING_POOL_BYTES``, so a large ``num_worker_threads`` makes
    workers share slots instead of growing HBM.
    """

    return min(num_send_workers * MLA_STAGING_SLOT_BYTES, MLA_STAGING_POOL_BYTES)


def mla_staging_slot_count(num_send_workers: int, slot_bytes: int) -> int:
    """MLA staging slots for one producer: one per send worker, within its reserve.

    ``slot_bytes`` follows the MLA layout and exceeds ``MLA_STAGING_SLOT_BYTES``
    when one block is wider. The pool then gets fewer slots, or none (the
    producer keeps per-token sends), rather than outgrow what
    ``mla_staging_reserve_bytes`` held back from the KV cache.
    """

    return min(num_send_workers, mla_staging_pool_bytes(num_send_workers) // slot_bytes)


def plain_mla_model(config) -> bool:
    """Whether ``config``'s model keeps plain MLA KV, all MLA staging handles.

    One latent row per token and no per-request state (``Family.MLA``), which
    is what the producer stages and a DCP consumer lands. DeepSeek-V4 carries
    a ``kv_lora_rank`` too (its V3 schema defaults one) but is its own family,
    and the MLA hybrids register slot regions, which both paths skip, so
    reserving for either would hold back HBM nothing uses.
    """

    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        return False
    # Local import: config-time callers must not import transformers early.
    from atom.utils import get_hf_text_config
    from atom.utils.selector import Family, attn_family

    return attn_family(get_hf_text_config(hf_config)) == Family.MLA


def mla_staging_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for Mooncake MLA staging pools.

    An upper bound of what ``MooncakeConnector`` allocates after the KV cache
    is sized: every Mooncake producer entry on a plain MLA model without
    producer DCP gets ``mla_staging_pool_bytes``, which its slots fit into.
    """

    if (
        not envs.ATOM_PD_MLA_STAGING
        or getattr(config, "decode_context_parallel_size", 1) > 1
        or not plain_mla_model(config)
    ):
        return 0
    return sum(
        mla_staging_pool_bytes(send_worker_count(connector))
        for connector in _producer_connectors(config, _MOONCAKE_CONNECTORS)
    )
