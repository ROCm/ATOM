# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sizing of the decode-side MLA landing pool (see ``landing_scatter``)."""

from __future__ import annotations

from atom.kv_transfer.disaggregation.pd_producer import (
    _canonical,
    iter_connector_configs,
)
from atom.utils import envs

# Only Mooncake consumers grant landing slots.
_LANDING_CONNECTORS = frozenset({"mooncake"})
# One decode rank's landing pool, held back from the KV budget and split
# evenly across the prefill stages that send to the rank.
MLA_LANDING_SLOT_BYTES = 8 << 20
MLA_LANDING_POOL_BYTES = 256 << 20


def _mooncake_consumer_configured(config) -> bool:
    transfer_config = getattr(config, "kv_transfer_config", None)
    return any(
        _canonical(connector, path=path) in _LANDING_CONNECTORS
        and connector.get("kv_role") == "kv_consumer"
        for connector, path in iter_connector_configs(transfer_config)
    )


def mla_landing_pool_shape() -> tuple[int, int]:
    """``(slots, slot_bytes)`` of one decode rank's landing pool, or ``(0, 0)``."""

    if not envs.ATOM_PD_MLA_LANDING:
        return 0, 0
    return MLA_LANDING_POOL_BYTES // MLA_LANDING_SLOT_BYTES, MLA_LANDING_SLOT_BYTES


def mla_landing_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for this rank's MLA landing pool.

    Only a Mooncake ``kv_consumer`` running DCP on an MLA model allocates the
    pool, in ``register_kv_caches`` after the KV cache is sized.
    """

    if (
        getattr(config, "decode_context_parallel_size", 1) <= 1
        or not getattr(getattr(config, "hf_config", None), "kv_lora_rank", None)
        or not _mooncake_consumer_configured(config)
    ):
        return 0
    slots, slot_bytes = mla_landing_pool_shape()
    return slots * slot_bytes
