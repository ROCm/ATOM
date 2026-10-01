# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How this engine meets LMCache's standalone multiprocess server.

Configuration validation, TP/DP topology and rank collapse, the model
namespace, and the adapters both connector halves open to the server.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.utils import envs

logger = logging.getLogger("atom")

_MP_LAYOUT_VERSION = 3


def _extra_config(config: Any) -> dict[str, Any]:
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    extra = kvc.get("kv_connector_extra_config", {}) or {}
    if not isinstance(extra, dict):
        raise TypeError("kv_connector_extra_config must be a dictionary")
    return extra


def _storage_kv_transfer_config(config: Any) -> dict[str, Any]:
    """Remove MP transport-only options before LMCache storage parsing."""

    kvc = dict(getattr(config, "kv_transfer_config", {}) or {})
    extra = kvc.get("kv_connector_extra_config")
    if isinstance(extra, dict):
        kvc["kv_connector_extra_config"] = {
            key: value
            for key, value in extra.items()
            if not (isinstance(key, str) and key.startswith("lmcache.mp."))
        }
    return kvc


def _mp_session_id(config: Any, request_id: Any) -> str:
    """Scope one LMCache MP request session to its global DP replica."""

    return f"{offcfg.lmcache_engine_id(config)}:{request_id}"


def _transfer_mode(config: Any) -> str:
    extra = _extra_config(config)
    configured_mode = extra.get("lmcache.mp.mp_transfer_mode")
    if configured_mode is None:
        configured_mode = envs.LMCACHE_MP_TRANSFER_MODE
    transfer_mode = str(configured_mode).strip().lower()
    if transfer_mode not in ("auto", "lmcache_driven", "engine_driven"):
        raise ValueError(
            "LMCache MP transfer mode must be 'auto', 'lmcache_driven', or "
            f"'engine_driven', got {configured_mode!r}"
        )
    if transfer_mode == "engine_driven":
        raise NotImplementedError(
            "ATOM lmcache_mp requires LMCache's lmcache_driven transfer path "
            "because engine_driven does not support multiple physical "
            "cache groups"
        )
    return transfer_mode


def _validate_mp_config(config: Any) -> tuple[int, int]:
    """Validate topology constraints shared by generic PAGE layouts."""
    tp_size = offcfg._strict_integer(
        "tensor_parallel_size",
        getattr(config, "tensor_parallel_size", 1) or 1,
        minimum=1,
    )
    pp_size = offcfg._strict_integer(
        "pipeline_parallel_size",
        getattr(config, "pipeline_parallel_size", 1) or 1,
        minimum=1,
    )
    dcp_size = offcfg._strict_integer(
        "decode_context_parallel_size",
        getattr(config, "decode_context_parallel_size", 1) or 1,
        minimum=1,
    )
    pcp_size = offcfg._strict_integer(
        "prefill_context_parallel_size",
        getattr(config, "prefill_context_parallel_size", 1) or 1,
        minimum=1,
    )
    parallel_config = getattr(config, "parallel_config", None)
    dp_size = offcfg._strict_integer(
        "data_parallel_size",
        getattr(
            parallel_config,
            "data_parallel_size",
            getattr(config, "data_parallel_size", 1),
        )
        or 1,
        minimum=1,
    )
    dp_size_local = offcfg._strict_integer(
        "data_parallel_size_local",
        getattr(parallel_config, "data_parallel_size_local", dp_size) or dp_size,
        minimum=1,
    )
    if pp_size != 1:
        raise NotImplementedError("lmcache_mp does not support PP yet")
    if dcp_size != 1:
        raise NotImplementedError("lmcache_mp does not support DCP yet")
    if pcp_size != 1:
        raise NotImplementedError("lmcache_mp does not support PCP yet")
    # Single-host DP replicas deliberately share one (model_name, worker_id,
    # world_size) identity: it is the content-addressed storage namespace, so
    # replicas deduplicate identical prefixes. It is not a registration key.
    # The server registers GPU memory per unique instance_id, refcounts layout
    # descriptors per (model_name, world_size) (all replicas publish the same
    # one), and _mp_session_id scopes request sessions and their locks per
    # replica.
    if dp_size_local != dp_size:
        raise NotImplementedError(
            "lmcache_mp supports DP and DP-attention only within one host; "
            "multi-node DP requires one LMCache server per host and local "
            "server routing "
            f"(data_parallel_size={dp_size}, local={dp_size_local})"
        )

    _transfer_mode(config)
    return tp_size, pp_size


def _config_has_fully_replicated_tp_pages(config: Any) -> bool:
    """Conservatively identify configs whose complete PAGE cache is TP-replicated.

    MLA caches the shared latent before the TP-sharded KV-B projection. ATOM's
    sparse MLA index-key projection is replicated too, so its auxiliary PAGE
    plane has the same property.

    The worker validates this config-time prediction against the attention
    backend's ``KVTransferTensors.tp_replication_factor`` declaration before
    registering any cache memory.
    """

    if _config_has_own_pool_draft(config):
        # A DSpark draft whose backend owns a KV pool appends per-rank-sharded
        # PAGE regions (`draft_kv.py` declares factor 1), so the complete PAGE
        # object is not replicated even when the target's is.
        return False
    hf_config = getattr(config, "hf_config", None)
    # MiniMax-M3 is GQA. Some TP ranks can happen to own the same KV head when
    # TP exceeds the global KV-head count, but the complete PAGE object is not
    # replicated across the whole TP group and must remain one shard per rank.
    if offcfg._is_minimax_m3(hf_config):
        return False
    hf_config = getattr(hf_config, "text_config", hf_config)
    # Kimi-K3's MLA KV is replicated, but its KDA checkpoint images -- stored
    # in those same PAGE units -- hold TP-sharded heads, so every rank keeps
    # its own copy.
    if getattr(hf_config, "model_type", None) == "kimi_linear":
        return False
    return getattr(hf_config, "kv_lora_rank", None) is not None


def _config_has_own_pool_draft(config: Any) -> bool:
    """Whether a speculative draft caches into a KV pool of its own.

    Mirrors `spec_decode.draft_kv.draft_kv_builder`, which only DSpark calls:
    the draft's own backend answers through `DRAFT_OWNS_KV_POOL`.
    """

    speculative = getattr(config, "speculative_config", None)
    draft_hf = getattr(speculative, "draft_model_hf_config", None)
    if getattr(speculative, "method", None) != "dspark" or draft_hf is None:
        return False
    from atom.utils.selector import attn_family, get_attn_backend

    return bool(get_attn_backend(attn_family(draft_hf)).DRAFT_OWNS_KV_POOL)


def _tp_replication_factor(config: Any) -> int:
    """Return the PAGE rank-collapse factor selected before workers start.

    ``auto`` uses only structural cache information available in the shared
    engine config. An explicit boolean is useful for a new attention backend:
    ``True`` requests full TP collapse, but worker registration still fails
    closed unless that backend declares every PAGE region byte-identical.
    """

    tp_size, _ = _validate_mp_config(config)
    configured = _extra_config(config).get("lmcache.mp.tp_rank_collapse", "auto")
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        collapse = _config_has_fully_replicated_tp_pages(config)
    elif type(configured) is bool:
        collapse = configured
    else:
        raise TypeError("lmcache.mp.tp_rank_collapse must be true, false, or 'auto'")
    return tp_size if collapse else 1


def _published_tp_replication_factor(
    transfer_tensors: Any,
    *,
    tp_size: int,
    native_state: bool = False,
) -> int:
    """Validate a backend's whole-object TP replication declaration."""

    attribute = (
        "native_state_tp_replication_factor"
        if native_state
        else "tp_replication_factor"
    )
    label = "native STATE" if native_state else "KV PAGE"
    factor = offcfg._strict_integer(
        f"{label} TP replication factor",
        getattr(transfer_tensors, attribute, 1),
        minimum=1,
    )
    if tp_size % factor:
        raise ValueError(
            f"{label} TP replication factor {factor} must divide TP size {tp_size}"
        )
    if factor not in (1, tp_size):
        raise NotImplementedError(
            "lmcache_mp currently supports only sharded or fully TP-replicated "
            f"{label} layouts, got replication factor {factor} for TP size {tp_size}"
        )
    return factor


def _server_urls(config: Any) -> list[str]:
    extra = _extra_config(config)
    configured = extra.get("lmcache.mp.server_urls")
    if configured is not None:
        if isinstance(configured, (list, tuple)):
            urls = [str(value).strip() for value in configured if str(value).strip()]
        else:
            urls = [
                value.strip() for value in str(configured).split(",") if value.strip()
            ]
    else:
        host = str(extra.get("lmcache.mp.host", "tcp://localhost")).strip()
        if not host:
            raise ValueError("lmcache.mp.host must be non-empty")
        port = offcfg._strict_integer(
            "lmcache.mp.port",
            extra.get("lmcache.mp.port", 5555),
            minimum=1,
        )
        if not 1 <= port <= 65535:
            raise ValueError("lmcache.mp.port must be in [1, 65535]")
        urls = [f"{host}:{port}"]
    urls = [url if "://" in url else f"tcp://{url}" for url in urls]
    if len(urls) != 1:
        raise NotImplementedError(
            "lmcache_mp currently supports exactly one LMCache server"
        )
    return urls


def _model_namespace(config: Any, *, checkpoint_spec: Any = None) -> str:
    """Build a model/layout namespace shared by scheduler and workers."""

    cfg = offcfg.build_lmcache_config(_storage_kv_transfer_config(config))
    world_size = offcfg.lmcache_replica_world_size(config)
    page_namespace = offcfg.build_page_namespace(
        config,
        cfg,
        world_size,
    )
    namespace = f"{page_namespace}::lmcache-mp-v{_MP_LAYOUT_VERSION}"
    if checkpoint_spec is not None:
        hf = getattr(config, "hf_config", None)
        hf = getattr(hf, "text_config", hf)
        document = {
            "checkpoint": checkpoint_spec.to_wire(),
            "hf_commit": getattr(hf, "_commit_hash", None),
            "revision": getattr(config, "revision", None),
            "model_revision": _extra_config(config).get("lmcache.mp.model_revision"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(document, sort_keys=True).encode()
        ).hexdigest()[:32]
        namespace += f"::native-state-v1-{fingerprint}"
    return namespace


def _parallel_strategy(config: Any, worker_id: int) -> Any:
    from lmcache.integration.atom import AtomMPParallelConfig

    tp_size, pp_size = _validate_mp_config(config)
    if worker_id < 0 or worker_id >= tp_size:
        raise ValueError(
            f"LMCache MP worker rank {worker_id} is outside [0, {tp_size})"
        )
    replication_factor = _tp_replication_factor(config)
    return AtomMPParallelConfig(
        world_size=tp_size * pp_size // replication_factor,
        worker_id=worker_id // replication_factor,
        tp_size=tp_size,
    )


def _make_scheduler_adapter(config: Any, *, checkpoint_spec: Any = None) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPSchedulerAdapter

    num_kv_readers = _tp_replication_factor(config)

    class _ReaderAwareSchedulerAdapter(AtomMPSchedulerAdapter):
        """Reserve one LMCache read lock for every collapsed TP consumer."""

        def _create_key(self, *args: Any, **kwargs: Any) -> Any:
            key = super()._create_key(*args, **kwargs)
            return replace(key, num_kv_readers=num_kv_readers)

    extra = _extra_config(config)
    return _ReaderAwareSchedulerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config, checkpoint_spec=checkpoint_spec),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, 0),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
    )


def _make_worker_adapter(config: Any, rank: int, *, checkpoint_spec: Any = None) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPWorkerAdapter

    extra = _extra_config(config)
    return AtomMPWorkerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config, checkpoint_spec=checkpoint_spec),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, rank),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
        heartbeat_interval=float(extra.get("lmcache.mp.heartbeat_interval", 10.0)),
        transfer_mode=_transfer_mode(config),
    )
