# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Admission and memory budgets for validated staging/landing configurations.

The MLA switches gate supported recipes here, not transport algorithms.
Runtime plans select ownership using peer topology and declared region layouts.
"""

from dataclasses import dataclass

from atom.kv_transfer.disaggregation.landing_scatter import segment_table_capacity
from atom.kv_transfer.disaggregation.mooncake.landing import (
    LANDING_POOL_SLOTS,
    LANDING_SLOT_BYTES,
)
from atom.kv_transfer.disaggregation.pd_producer import (
    mooncake_producer_connectors,
    send_worker_count,
    staging_pool_bytes,
)
from atom.kv_transfer.disaggregation.transfer_plan import TOKEN_UNITS
from atom.utils import envs, get_hf_text_config
from atom.utils.selector import Family, attn_family


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
    return attn_family(get_hf_text_config(hf_config)) == Family.MLA


def mla_staging_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for Mooncake MLA staging pools.

    An upper bound of what ``MooncakeConnector`` allocates after the KV cache
    is sized: every Mooncake producer entry on a plain MLA model without
    producer DCP gets ``staging_pool_bytes``, which its slots fit into.
    """

    if (
        not envs.ATOM_PD_MLA_STAGING
        or getattr(config, "decode_context_parallel_size", 1) > 1
        or not plain_mla_model(config)
    ):
        return 0
    return sum(
        staging_pool_bytes(send_worker_count(connector))
        for connector in mooncake_producer_connectors(config)
    )


def mla_landing_configured(config) -> bool:
    """Whether a Mooncake connector with ``config`` gets an MLA landing pool.

    ``config`` holds one Mooncake connector's ``kv_transfer_config`` (see
    ``KVConnectorFactory.kv_budget_reserve_bytes``). Only a ``kv_consumer``
    running DCP on a plain MLA model (``plain_mla_model``) lands. The reserve
    and the allocation in ``register_kv_caches`` both ask this, so no pool
    goes unreserved.
    """

    return bool(
        envs.ATOM_PD_MLA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
        and plain_mla_model(config)
    )


def mla_landing_table_segments(config) -> int:
    """Segments a landing rank's scatter table holds, from ``config`` alone.

    One per destination page of a full pool, plus 64 item splits per slot.
    An MLA row takes at least ``kv_lora_rank`` bytes, so a page takes at
    least ``block_size * kv_lora_rank``. A batch needing more segments is
    copied in several chunks.
    """

    kv_lora_rank = get_hf_text_config(config.hf_config).kv_lora_rank
    page_bytes = config.kv_cache_block_size * kv_lora_rank
    pages_per_slot = LANDING_SLOT_BYTES // page_bytes
    return segment_table_capacity(LANDING_POOL_SLOTS * (pages_per_slot + 64))


@dataclass(frozen=True)
class LandingSpec:
    units_per_page: int
    table_segments: int
    wire_layout: str


def landing_spec(config):
    if mla_landing_configured(config):
        return LandingSpec(
            config.kv_cache_block_size,
            mla_landing_table_segments(config),
            TOKEN_UNITS,
        )
    return None


def staging_reserve_bytes(config):
    return mla_staging_reserve_bytes(config)


def landing_reserve_bytes(config):
    """HBM to hold back from the KV budget for this rank's KV landing.

    The pool plus the scatter's int64 ``[3, segments]`` device table, both
    allocated in ``register_kv_caches`` after the KV cache is sized.
    """

    spec = landing_spec(config)
    if spec is None:
        return 0
    table_bytes = 3 * 8 * spec.table_segments
    return LANDING_POOL_SLOTS * LANDING_SLOT_BYTES + table_bytes
