# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How this engine meets LMCache's standalone multiprocess server.

Configuration validation, TP/DP topology and rank collapse, the model
namespace, and the adapters both connector halves open to the server.
"""

from __future__ import annotations

import functools
import hashlib
import ipaddress
import json
import logging
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

from atom.kv_transfer.offload import config as offcfg
from atom.model_engine.dp_routing.topology import prefix_spread_enabled
from atom.utils import envs

logger = logging.getLogger("atom")

_MP_LAYOUT_VERSION = 3


def _extra_config(config: Any) -> dict[str, Any]:
    return dict(offcfg.connector_extras(getattr(config, "kv_transfer_config", None)))


def _storage_kv_transfer_config(config: Any) -> dict[str, Any]:
    """Remove MP transport-only options before LMCache storage parsing."""

    kvc = dict(getattr(config, "kv_transfer_config", {}) or {})
    extra = {
        key: value
        for key, value in offcfg.connector_extras(kvc).items()
        if not (isinstance(key, str) and key.startswith("lmcache.mp."))
    }
    if "kv_connector_extra_config" in kvc:
        kvc["kv_connector_extra_config"] = extra
    else:
        kvc = extra
    return kvc


def require_chunk_matches_server(adapter: Any, config: Any) -> None:
    """Under `prefix_spread`, the chunk the router predicts loads with
    (`offcfg.lmcache_chunk_tokens`) must be the one the MP server runs on.

    The router reads the configured chunk before any server is asked; a
    different server chunk would skew its load estimates. Nothing else reads
    the configured value -- the connector itself runs on the server's chunk --
    so other strategies and runs with only one routable engine are not
    checked. Called by both worker registrations (PAGE-only and native-state).
    """
    if not prefix_spread_enabled(config):
        return
    configured = offcfg.lmcache_chunk_tokens(_storage_kv_transfer_config(config))
    server = int(adapter.lmcache_tokens_per_chunk)
    if configured != server:
        raise ValueError(
            "--dp-load-balance prefix_spread with LMCache MP needs the "
            f"configured chunk to match the server: configured {configured} "
            f"(lmcache.chunk_size or LMCACHE_CHUNK_SIZE), server {server}"
        )


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
    #
    # Across hosts the server cannot be shared: it moves KV through GPU IPC,
    # which reaches only its own host's devices. A loopback server address
    # therefore means one server per host, each serving that host's replicas,
    # which is the multi-node layout this connector supports. A prefix is then
    # reusable only on the host that stored it; DP session affinity routes a
    # session's later turns back to its owner rank, and so to that host.
    if dp_size_local != dp_size and not _server_is_host_local(config):
        raise NotImplementedError(
            "lmcache_mp supports multi-node DP only with one LMCache server per "
            "host: set lmcache.mp.host to a loopback address and start a server "
            "on every host "
            f"(data_parallel_size={dp_size}, local={dp_size_local})"
        )

    _transfer_mode(config)
    return tp_size, pp_size


def _server_is_host_local(config: Any) -> bool:
    """Whether every rank reaches the LMCache server at a loopback address."""

    try:
        urls = _server_urls(config)
    except (TypeError, ValueError, NotImplementedError):
        return False
    for url in urls:
        host = urlsplit(url).hostname or ""
        if host == "localhost":
            continue
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return False
        except ValueError:
            return False
    return True


def _config_has_fully_replicated_tp_pages(config: Any) -> bool:
    """Conservatively identify configs whose complete PAGE cache is TP-replicated.

    MLA caches the shared latent before the TP-sharded KV-B projection. ATOM's
    sparse MLA index-key projection is replicated too, so its auxiliary PAGE
    plane has the same property.

    The worker validates this config-time prediction against the attention
    backend's ``KVTransferTensors.tp_replication_factor`` declaration before
    registering any cache memory.
    """

    from atom.spec_decode.draft_kv import draft_owns_kv_pool

    if draft_owns_kv_pool(config):
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


def _model_namespace(
    config: Any, *, checkpoint_spec: Any = None, images: bool = True
) -> str:
    """Build a model/layout namespace shared by scheduler and workers.

    A native-state layout without images (`images=False`: PAGE KV alone, no
    image group) gets a namespace of its own, so a scheduler and workers that
    disagree about images find nothing rather than each other's objects.
    """

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
        kind = "native-state" if images else "native-page"
        namespace += f"::{kind}-v1-{fingerprint}"
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


def _make_scheduler_adapter(
    config: Any, *, checkpoint_spec: Any = None, images: bool = True
) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPSchedulerAdapter

    num_kv_readers = _tp_replication_factor(config)

    class _SaltedSchedulerAdapter(AtomMPSchedulerAdapter):
        """Salt a session's keys with its prompt's media, and reserve one
        LMCache read lock for every collapsed TP consumer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # session -> `tier_cache_salt`, for salted prompts only.
            self.cache_salts: dict[str, str] = {}

        def _create_key(self, *args: Any, **kwargs: Any) -> Any:
            key = super()._create_key(*args, **kwargs)
            key = replace(key, num_kv_readers=num_kv_readers)
            salt = self.cache_salts.get(key.request_id)
            return replace(key, cache_salt=salt) if salt else key

        def salt_session(self, request_id: str, salt: str) -> None:
            self.cache_salts[request_id] = salt

        def end_session(self, request_id: str) -> None:
            self.cache_salts.pop(request_id, None)
            super().end_session(request_id)

    extra = _extra_config(config)
    return _SaltedSchedulerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(
            config, checkpoint_spec=checkpoint_spec, images=images
        ),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, 0),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
    )


def _make_worker_adapter(
    config: Any, rank: int, *, checkpoint_spec: Any = None, images: bool = True
) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPWorkerAdapter

    class _SaltedWorkerAdapter(AtomMPWorkerAdapter):
        """Salt a transfer's keys with its prompt's media (`transfer_spec`)."""

        def _create_key(self, request_id: str, spec: Any) -> Any:
            key = super()._create_key(request_id, spec)
            return replace(key, cache_salt=spec.cache_salt)

    extra = _extra_config(config)
    return _SaltedWorkerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(
            config, checkpoint_spec=checkpoint_spec, images=images
        ),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, rank),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
        heartbeat_interval=float(extra.get("lmcache.mp.heartbeat_interval", 10.0)),
        transfer_mode=_transfer_mode(config),
    )


def transfer_spec(**fields: Any) -> Any:
    """LMCache's `AtomMPTransferSpec` plus the prompt's media
    (`LMCacheReqMeta.cache_salt`) for the key. Built on first use: LMCache is
    imported only where a transfer is submitted."""
    from lmcache.integration.atom import AtomMPTransferSpec

    return _salted_transfer_spec(AtomMPTransferSpec)(**fields)


@functools.cache
def _salted_transfer_spec(base: type) -> type:
    @dataclass
    class TransferSpec(base):
        cache_salt: str = ""

    return TransferSpec


def require_native_state_server(adapter: Any, config: Any) -> None:
    """Validate native transfer geometry shared with the LMCache server."""
    configured_chunk = int(
        offcfg.build_lmcache_config(_storage_kv_transfer_config(config)).chunk_size
    )
    if configured_chunk != int(adapter.lmcache_tokens_per_chunk):
        raise ValueError(
            "Native-state LMCache configured chunk size must match the MP server: "
            f"configured={configured_chunk}, server={adapter.lmcache_tokens_per_chunk}"
        )
