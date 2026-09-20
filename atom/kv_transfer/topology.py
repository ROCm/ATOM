# SPDX-License-Identifier: MIT
"""Shared, metadata-only discovery for the Python and embedded frontends."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from atom.kv_transfer.offload.config import page_parallel_layout


def resolve_transfer_role(kv_cfg: dict) -> tuple[str | None, int]:
    kv_role = kv_cfg.get("kv_role")
    handshake_port = kv_cfg.get("handshake_port", 6301)
    if kv_role is not None or kv_cfg.get("kv_connector") != "multi":
        return kv_role, handshake_port

    # MultiConnector wraps the real transfer connector. Surface the producer
    # role so a P/D proxy can recognize multi[mooncake-producer + offload] as a
    # prefill node.
    fallback_role = None
    fallback_port = handshake_port
    for sub_cfg in kv_cfg.get("connectors", []):
        sub_role = sub_cfg.get("kv_role")
        if sub_role is None:
            continue
        if fallback_role is None:
            fallback_role = sub_role
            fallback_port = sub_cfg.get("handshake_port", handshake_port)
        if sub_role == "kv_producer":
            return sub_role, sub_cfg.get("handshake_port", handshake_port)
    return fallback_role, fallback_port


def execution_topology(config) -> dict[str, Any]:
    """Return validated rank placement, without inferring network reachability.

    ``kv_transfer_config.routing_topology`` is an optional deployment manifest.
    Its rank list must cover one entire PP x TP execution. DCP reuses TP ranks.
    A host may own any number of ranks or executions. Unspecified placement is
    reported as unknown; a local hostname cannot prove remote rank placement.
    """
    layout = page_parallel_layout(config)
    topology = deepcopy(
        (getattr(config, "kv_transfer_config", None) or {}).get("routing_topology", {})
    )
    ranks = topology.get("ranks", [])
    expected = {
        (pp, tp) for pp in range(layout["pp_size"]) for tp in range(layout["tp_size"])
    }
    seen = set()
    for rank in ranks:
        identity = (rank.get("pp"), rank.get("tp"))
        if identity not in expected or identity in seen:
            raise ValueError(
                "routing_topology ranks must uniquely cover the PP x TP grid"
            )
        if not isinstance(rank.get("host_id"), str) or not rank["host_id"]:
            raise ValueError("routing_topology ranks require a nonempty host_id")
        if rank.get("device_id") is None:
            raise ValueError("routing_topology ranks require device_id")
        seen.add(identity)
        rank["dcp"] = identity[1] % layout["dcp_size"]
        rank["layer_range"] = layout["pp_layer_ranges"][identity[0]]
    if ranks and seen != expected:
        raise ValueError("routing_topology ranks must cover every PP x TP rank")
    bindings = topology.get("cache_bindings", [])
    for binding in bindings:
        if not binding.get("storage_domain_id"):
            raise ValueError("cache binding requires storage_domain_id")
        pieces = binding.get("rank_set", [])
        if not pieces or any(tuple(piece) not in expected for piece in pieces):
            raise ValueError("cache binding rank_set must name existing PP x TP ranks")
    return {**topology, **layout, "ranks": ranks}


def transfer_info(config) -> dict[str, Any]:
    """Describe configured geometry; connector handshakes remain authoritative."""
    kv_config = getattr(config, "kv_transfer_config", None) or {}
    role, port = resolve_transfer_role(kv_config)
    topology = execution_topology(config)
    return {
        "tp_size": topology["tp_size"],
        "pp_size": topology["pp_size"],
        "dcp_size": topology["dcp_size"],
        "dp_size": getattr(
            getattr(config, "parallel_config", None), "data_parallel_size", 1
        ),
        "kv_role": role,
        "handshake_port": port,
        "physical_block_size_tokens": config.kv_cache_block_size,
        "hash_block_size_tokens": config.kv_cache_block_size * topology["dcp_size"],
        "topology": topology,
    }


def server_info(config, model_name: str) -> dict[str, Any]:
    """Return the same engine identity from either serving frontend."""
    from atom.cache_routing.config import CacheRoutingConfig

    routing = CacheRoutingConfig.from_env()
    extra = (
        {}
        if routing is None
        else {
            "cache_routing": {
                "catalog_http": routing.catalog_url.rstrip("/") + "/v1/cache",
                "execution_id": routing.execution_id,
                "content_namespace": routing.content_namespace,
            }
        }
    )
    return {
        "model_id": model_name,
        "served_model_name": model_name,
        **transfer_info(config),
        **extra,
    }
