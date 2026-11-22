# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Admission and memory budgets for validated staging/landing configurations.

The MLA and MiniMax switches gate supported recipes here, not transport
algorithms.
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
from atom.kv_transfer.disaggregation.state_gather import (
    STATE_POOL_BYTES,
    STATE_SLOT_BYTES,
)
from atom.kv_transfer.disaggregation.transfer_plan import PAGE_UNITS, TOKEN_UNITS
from atom.model_ops.glm5_next.geometry import pooled_path_enabled
from atom.utils import envs, get_hf_text_config
from atom.utils.selector import Family, attn_family

MINIMAX_SCATTER_SEGMENTS = 4096


def plain_mla_model(config) -> bool:
    """Whether ``config``'s model keeps plain MLA KV, all MLA staging handles.

    One latent row per token and no per-request state (``Family.MLA``), which
    is what the producer stages and a DCP consumer lands. DeepSeek-V4 carries
    a ``kv_lora_rank`` too (its V3 schema defaults one) but is its own family,
    and the MLA hybrids register slot regions, which both paths skip, so
    reserving for either would hold back HBM nothing uses. Kimi-K3, the one
    hybrid whose rows the slot path stages, has its own gate
    (``kimi_mla_admitted``).
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


def minimax_landing_admitted(config) -> bool:
    """Whether ``ATOM_PD_MINIMAX_LANDING`` admits ``config``: MiniMax-M3 without
    DCP, the only model the PAGE head-cut staging and landing ran on."""
    from atom.config import _is_minimax_m3_config

    return bool(
        envs.ATOM_PD_MINIMAX_LANDING
        and getattr(config, "decode_context_parallel_size", 1) == 1
        and _is_minimax_m3_config(getattr(config, "hf_config", None))
    )


def minimax_page_staging(config) -> bool:
    """Whether a Mooncake connector with ``config`` stages its MiniMax head cuts.

    Only a TP1 (PP) producer cuts heads for a TP-sharded decode, and the
    producer role is decided as Mooncake decides it
    (``mooncake_producer_connectors``), so the reserve and the connector agree.
    """

    return bool(
        minimax_landing_admitted(config)
        and getattr(config, "tensor_parallel_size", 1) == 1
        and mooncake_producer_connectors(config)
    )


def minimax_staging_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for MiniMax head-cut staging pools."""

    if not minimax_page_staging(config):
        return 0
    return sum(
        staging_pool_bytes(send_worker_count(connector))
        for connector in mooncake_producer_connectors(config)
    )


def minimax_landing_configured(config) -> bool:
    """Whether a Mooncake connector with ``config`` gets a MiniMax landing pool."""

    return bool(
        minimax_landing_admitted(config)
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
    if mla_landing_configured(config) or kimi_mla_landing_configured(config):
        return LandingSpec(
            config.kv_cache_block_size,
            mla_landing_table_segments(config),
            TOKEN_UNITS,
        )
    if minimax_landing_configured(config):
        return LandingSpec(
            1, MINIMAX_SCATTER_SEGMENTS, PAGE_UNITS, supports_chunks=True
        )
    return None


def kimi_state_model(config) -> bool:
    """Whether ``config`` is Kimi-K3, the model whose KDA slots are staged."""

    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        return False
    return getattr(get_hf_text_config(hf_config), "model_type", None) == "kimi_linear"


def kimi_mla_admitted(config) -> bool:
    """Whether the slot path stages and lands ``config``'s MLA rows.

    Kimi-K3 with a per-token indexer only. A pooled indexer (``index_kpool``
    > 1 with ``ATOM_GLM5_KPOOL``) caches one key per pool, which the PP
    backend refuses anyway; GLM-5.3-Flash shares the family but not the gate.
    """

    if not kimi_state_model(config):
        return False
    hf_config = config.hf_config
    return not any(
        pooled_path_enabled(int(getattr(c, "index_kpool", 1) or 1))
        for c in (hf_config, get_hf_text_config(hf_config))
    )


def kimi_mla_staging_reserve_bytes(config) -> int:
    """HBM a Kimi producer without DCP holds back for its MLA row pool.

    The same ``staging_pool_bytes`` per Mooncake producer entry as a plain
    MLA model, beside the KDA state pool.
    """

    if (
        not envs.ATOM_PD_MLA_STAGING
        or getattr(config, "decode_context_parallel_size", 1) > 1
        or not kimi_mla_admitted(config)
    ):
        return 0
    return sum(
        staging_pool_bytes(send_worker_count(connector))
        for connector in mooncake_producer_connectors(config)
    )


def kimi_mla_landing_configured(config) -> bool:
    """Whether a Kimi DCP decode gets the MLA row landing pool."""

    return bool(
        envs.ATOM_PD_MLA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
        and kimi_mla_admitted(config)
    )


def _state_pool_bytes(num_send_workers: int) -> int:
    return min(num_send_workers * STATE_SLOT_BYTES, STATE_POOL_BYTES)


def kimi_state_staging_reserve_bytes(config) -> int:
    """HBM a Kimi producer holds back for its KDA state staging pool.

    One 12 MiB slot per send worker, capped at 256 MiB. The pool is what a
    PP stage gathers a decode rank's TP shard into.
    """

    if (
        not envs.ATOM_PD_MLA_STAGING
        or getattr(config, "decode_context_parallel_size", 1) > 1
        or not kimi_state_model(config)
    ):
        return 0
    return sum(
        _state_pool_bytes(send_worker_count(connector))
        for connector in mooncake_producer_connectors(config)
    )


def kimi_state_landing_configured(config) -> bool:
    """Whether a Kimi DCP decode gets a landing pool for KDA state."""

    return bool(
        envs.ATOM_PD_MLA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
        and kimi_state_model(config)
    )


def kimi_state_landing_reserve_bytes(config) -> int:
    """The state landing pool plus its scatter table, held back from KV."""

    if not kimi_state_landing_configured(config):
        return 0
    # One segment per slot region, plus the same 64-item slack MLA uses.
    # Kimi-K3.1 publishes a few hundred regions; 1024 covers a stage.
    table_bytes = 3 * 8 * segment_table_capacity(1024)
    return STATE_POOL_BYTES + table_bytes


def kv_staging_reserve_bytes(config):
    """The KV staging pool's share: what ``_build_kv_staging`` sizes slots from."""
    return (
        mla_staging_reserve_bytes(config)
        + minimax_staging_reserve_bytes(config)
        + kimi_mla_staging_reserve_bytes(config)
    )


def staging_reserve_bytes(config):
    return kv_staging_reserve_bytes(config) + kimi_state_staging_reserve_bytes(config)


def landing_reserve_bytes(config):
    """HBM to hold back from the KV budget for this rank's KV landing.

    The pool plus the scatter's int64 ``[3, segments]`` device table, both
    allocated in ``register_kv_caches`` after the KV cache is sized.
    """

    spec = landing_spec(config)
    mla = 0
    if spec is not None:
        table_bytes = 3 * 8 * spec.table_segments
        mla = LANDING_POOL_SLOTS * LANDING_SLOT_BYTES + table_bytes
    return mla + kimi_state_landing_reserve_bytes(config)
