# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Admission and memory budgets for validated staging/landing configurations.

The MLA and MHA switches gate supported recipes here, not transport algorithms.
Runtime plans select ownership using peer topology and declared region layouts.
"""

from dataclasses import dataclass

from atom.kv_transfer.disaggregation.landing_scatter import segment_table_capacity
from atom.kv_transfer.disaggregation.mooncake.landing import (
    LANDING_POOL_SLOTS,
    LANDING_SLOT_BYTES,
)
from atom.kv_transfer.disaggregation.pd_producer import (
    mla_staging_reserve_bytes,
    plain_mla_model,
    send_worker_count,
    staging_pool_bytes,
)
from atom.kv_transfer.disaggregation.transfer_plan import PAGE_UNITS, TOKEN_UNITS
from atom.utils import envs, get_hf_text_config

MHA_SCATTER_SEGMENTS = 4096


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


def mha_landing_model(config) -> bool:
    from atom.config import _is_minimax_m3_config

    return bool(
        envs.ATOM_PD_MHA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) == 1
        and _is_minimax_m3_config(getattr(config, "hf_config", None))
    )


def mha_staging_reserve_bytes(config) -> int:
    if (
        not mha_landing_model(config)
        or getattr(config, "tensor_parallel_size", 1) != 1
        or config.kv_transfer_config.get("kv_role") not in ("kv_producer", "kv_both")
    ):
        return 0
    return staging_pool_bytes(send_worker_count(config.kv_transfer_config))


def mha_landing_configured(config) -> bool:
    return bool(
        mha_landing_model(config)
        and getattr(config, "tensor_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
    )


@dataclass(frozen=True)
class LandingSpec:
    units_per_page: int
    table_segments: int
    wire_layout: str
    supports_chunks: bool = False


def landing_spec(config):
    # MLA lands only with DCP and MHA only without it, so at most one applies.
    if mla_landing_configured(config):
        return LandingSpec(
            config.kv_cache_block_size,
            mla_landing_table_segments(config),
            TOKEN_UNITS,
        )
    if mha_landing_configured(config):
        return LandingSpec(1, MHA_SCATTER_SEGMENTS, PAGE_UNITS, supports_chunks=True)
    return None


def staging_reserve_bytes(config):
    return mla_staging_reserve_bytes(config) + mha_staging_reserve_bytes(config)


def landing_reserve_bytes(config):
    spec = landing_spec(config)
    return (
        0
        if spec is None
        else LANDING_POOL_SLOTS * LANDING_SLOT_BYTES + 24 * spec.table_segments
    )
