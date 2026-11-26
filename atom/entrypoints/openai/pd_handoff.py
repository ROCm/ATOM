# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A P/D worker's side of the Atomesh router's handoff, for every frontend
that serves one: the OpenAI server (`api_server.py`) and the Rust frontend's
`TokenBridge`.

* `kv_transfer_info` / `server_info`: what the router reads of a worker at
  registration (its role, TP and DP sizes, chunked transfer's endpoint).
* `chunked_receive_refusal` / `check_media_handoff` (`protocol.py`): what a
  request's `kv_transfer_params` asks that this worker cannot serve, refused
  before it is enqueued.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.disaggregation.chunked_prefill import chunked_topology_supported
from atom.kv_transfer.disaggregation.mooncake.endpoint import (
    DEFAULT_HANDSHAKE_PORT,
    producer_endpoint,
)
from atom.model_engine.llm_engine import InputOutputProcessor


def is_chunked_receive(params: dict[str, Any] | None) -> bool:
    """A decode leg that receives its KV while its prefill still runs."""
    return bool(
        params and params.get("chunked_transfer") and params.get("do_remote_prefill")
    )


def chunked_receive_refusal(params: dict[str, Any], config: Any) -> str | None:
    """Why this decode cannot take a chunked receive, or None.

    Checked per request: chunked transfer is P's setting, and P learns D's
    DCP only from a request. A refusal must reach the router as an HTTP error
    before admission, so it cancels the waiting P request.
    """
    if not chunked_topology_supported(
        params.get("remote_tp_size", 1),
        config.tensor_parallel_size,
        config.pipeline_parallel_size,
    ):
        return "Chunked P/D requires D PP=1 and D TP a multiple of positive P TP"
    if (
        config.decode_context_parallel_size > 1
        and config.hf_config.model_type
        in InputOutputProcessor._per_req_cache_model_types()
    ):
        return "Chunked P/D does not send per-request state to a DCP decode"
    return None


def resolve_kv_transfer_role(kv_cfg: dict) -> tuple[str | None, int]:
    kv_role = kv_cfg.get("kv_role")
    handshake_port = kv_cfg.get("handshake_port", DEFAULT_HANDSHAKE_PORT)
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


def kv_transfer_info(config: Any) -> dict[str, Any]:
    """`GET /kv_transfer_info`."""
    kv_cfg = config.kv_transfer_config or {}
    kv_role, handshake_port = resolve_kv_transfer_role(kv_cfg)
    info = {
        "tp_size": config.tensor_parallel_size,
        "dp_size": config.parallel_config.data_parallel_size,
        "kv_role": kv_role,
        "handshake_port": handshake_port,
    }
    from atom.kv_transfer.disaggregation.pd_producer import iter_connector_configs
    from atom.utils.network import get_ip

    for sub, _ in iter_connector_configs(kv_cfg):
        if (
            sub.get("kv_connector") == "mooncake"
            and sub.get("kv_role", "kv_producer") == "kv_producer"
            and sub.get("enable_chunked_transfer", False)
        ):
            info["chunked_transfer"] = producer_endpoint(
                host=get_ip(),
                handshake_port=sub.get("handshake_port", DEFAULT_HANDSHAKE_PORT),
                pp_size=config.pipeline_parallel_size,
                block_size=config.kv_cache_block_size,
                dcp_size=config.decode_context_parallel_size,
            )
            break
    return info


def server_info(config: Any, model_name: str) -> dict[str, Any]:
    """`GET /server_info`: the router's dp-aware discovery reads ``dp_size``
    here to expand the per-DP-rank worker set."""
    return {
        "model_id": model_name,
        "served_model_name": model_name,
        "tp_size": config.tensor_parallel_size,
        "dp_size": config.parallel_config.data_parallel_size,
    }
