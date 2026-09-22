# SPDX-License-Identifier: MIT
"""Connect scheduler commits and sampled load to the common catalog."""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

from atom.cache_routing.catalog import CacheCatalog
from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import HASH_PROTOCOL
from atom.cache_routing.server import CatalogServer
from atom.kv_transfer.offload.config import (
    build_page_namespace,
    page_parallel_layout,
    select_offload_layout,
)
from atom.kv_transfer.topology import resolve_transfer_role


def start_catalog(config):
    cfg = CacheRoutingConfig.from_env()
    if cfg is None:
        return None
    parallel = getattr(config, "parallel_config", None)
    if getattr(parallel, "pipeline_parallel_rank", 0):
        return None
    # A serving endpoint must identify one queue. Multi-DP endpoints and stateful
    # models retain load routing until their exact execution/state view exists.
    try:
        dense = select_offload_layout(config) == "dense"
    except ValueError:
        dense = False
    if (
        getattr(parallel, "data_parallel_size", 1) > 1
        or not config.enable_prefix_caching
        or not dense
        or getattr(config.hf_config, "sliding_window", None)
    ):
        logging.getLogger(__name__).warning(
            "Exact cache catalog disabled for unsupported execution/state layout"
        )
        return None
    layout = page_parallel_layout(config)
    world = layout["pp_size"] * layout["tp_size"]
    namespace = build_page_namespace(
        config, SimpleNamespace(chunk_size=cfg.canonical_block_size), world
    )
    span = config.kv_cache_block_size * layout["dcp_size"]
    catalog = CacheCatalog(cfg, namespace, span, world)
    kv_config = getattr(config, "kv_transfer_config", None) or {}
    role, _ = resolve_transfer_role(kv_config)
    role = {"kv_producer": "prefill", "kv_consumer": "decode"}.get(role, "regular")
    connectors = (
        kv_config.get("connectors", [])
        if kv_config.get("kv_connector") == "multi"
        else [kv_config]
    )
    mooncake_consumer = any(
        c.get("kv_connector") == "mooncake" and c.get("kv_role") == "kv_consumer"
        for c in connectors
    )
    catalog.info = {
        "protocol_version": 1,
        "execution_id": cfg.execution_id,
        "role": role,
        "content_namespace": cfg.content_namespace,
        "layout_id": namespace,
        "canonical_hash": HASH_PROTOCOL,
        "canonical_block_size_tokens": cfg.canonical_block_size,
        "hash_block_size_tokens": span,
        "physical_block_size_tokens": config.kv_cache_block_size,
        "parallel_layout": layout,
        "storage_domain_id": cfg.storage_domain_id or cfg.execution_id,
        "catalog_http": cfg.catalog_url.rstrip("/") + "/v1/cache",
        "min_load_tokens": max(0, int(os.getenv("OFFLOAD_MIN_LOAD_TOKENS", "8192"))),
        "costs": kv_config.get("routing_topology", {}).get("costs", {}),
        "transfer_paths": kv_config.get("routing_topology", {}).get(
            "transfer_paths", []
        ),
        "capabilities": {
            "exact_prefix_reuse": True,
            "cache_load_policy_hint": True,
            "pd_delta_receive": mooncake_consumer and config.index_cache_dtype != "fp4",
            "pd_decode_cpu_preload": False,
            "stateful_prefix_reuse": False,
            "cross_layout_store_load": False,
        },
    }
    return CatalogServer(catalog)


def update_load(
    catalog: CacheCatalog, metrics: dict, pending_prefill_tokens: int
) -> None:
    """Use exactly the engine metrics sample also delivered to Prometheus."""
    with catalog.lock:
        catalog.load = {
            **metrics,
            "execution_id": catalog.config.execution_id,
            "source_epoch": catalog.epoch,
            "sample_seq": str(int(catalog.load.get("sample_seq", "0")) + 1),
            "sample_time_unix_ns": str(time.time_ns()),
            "pending_prefill_tokens": pending_prefill_tokens,
            "accepted_dispatch_ids": list(catalog.accepted_dispatches),
        }
