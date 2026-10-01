# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How this engine meets LMCache's standalone multiprocess server.

Configuration validation, TP/PP/DP topology and rank collapse, the model
namespace, and the adapters both connector halves open to the server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import replace
from http.client import HTTPException
from typing import Any
from urllib.parse import urlsplit

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


def _pp_rank(config: Any) -> int:
    """Return this engine's PP stage, set per stage by the engine core manager."""

    _, pp_size = _validate_mp_config(config)
    pp_rank = offcfg._strict_integer(
        "pipeline_parallel_rank",
        getattr(getattr(config, "parallel_config", None), "pipeline_parallel_rank", 0)
        or 0,
    )
    if pp_rank >= pp_size:
        raise ValueError(f"LMCache MP PP rank {pp_rank} is outside [0, {pp_size})")
    return pp_rank


def _reject_native_state_pp(config: Any) -> None:
    """Native-state transfers have not been validated across PP stages."""

    _, pp_size = _validate_mp_config(config)
    if pp_size != 1:
        raise NotImplementedError(
            "native-state lmcache_mp does not support PP "
            f"(pipeline_parallel_size={pp_size})"
        )


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


def _pp_stage_layout(config: Any) -> dict[str, Any]:
    """Describe how the model's KV layers are split across PP stages.

    ``stage_layers`` counts the layers each stage binds: its target-model span,
    plus the draft layers on the last stage.
    """

    _, pp_size = _validate_mp_config(config)
    draft_layers = offcfg.speculative_draft_layer_count(config)
    return {
        "pp_size": pp_size,
        "spans": [list(span) for span in offcfg.pp_stage_layer_spans(config)],
        "draft_layers": draft_layers,
        "draft_shares_target_pool": (
            not _config_has_own_pool_draft(config) if draft_layers else None
        ),
        "stage_layers": offcfg.pp_stage_layer_counts(config),
    }


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
    _, pp_size = _validate_mp_config(config)
    if pp_size > 1:
        # The PAGE namespace's world is PP x TP, so PP4/TP1 and PP1/TP4 would
        # otherwise share keys. Every stage hashes the same whole-model
        # layout, so all stages of one replica stay in one namespace.
        stage_layout = json.dumps(
            _pp_stage_layout(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        stage_digest = hashlib.blake2b(stage_layout, digest_size=8).hexdigest()
        namespace = f"{namespace}::pp-{stage_digest}"
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


def _kv_worker_grid(config: Any, tp_rank: int) -> tuple[int, int]:
    """Return this rank's LMCache ``(worker_id, world_size)``.

    Every PP stage holds a disjoint layer slice, so it is its own group of
    LMCache kv ranks: ``worker_id = pp_rank * ranks_per_stage + collapsed TP
    rank``. A scheduler lookup without a worker id fans out over all of them.
    """

    tp_size, pp_size = _validate_mp_config(config)
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"LMCache MP TP rank {tp_rank} is outside [0, {tp_size})")
    replication_factor = _tp_replication_factor(config)
    ranks_per_stage = tp_size // replication_factor
    return (
        _pp_rank(config) * ranks_per_stage + tp_rank // replication_factor,
        pp_size * ranks_per_stage,
    )


def _parallel_strategy(config: Any, tp_rank: int) -> Any:
    from lmcache.integration.atom import AtomMPParallelConfig

    tp_size, _ = _validate_mp_config(config)
    worker_id, world_size = _kv_worker_grid(config, tp_rank)
    return AtomMPParallelConfig(
        world_size=world_size,
        worker_id=worker_id,
        # Legacy wire field; LMCache ignores it.
        tp_size=tp_size,
    )


def _validate_pp_l2_layouts(config: Any) -> None:
    """Refuse an LMCache L2 under PP unless the server keeps per-rank layouts.

    LMCache keeps one layout per ``(model_name, world_size)``, last writer
    wins, and sizes L2 -> L1 prefetch buffers from it. PP stages register
    their own PAGE layouts, which differ even when their layer counts match:
    GLM-5.2's DSA index cache has rows only for non-``shared`` indexer layers,
    hybrid models have MLA rows only for full-attention layers, and the last
    stage adds the draft. Stages would then prefetch into wrongly sized
    buffers. L1 stores and retrieves use each instance's own layout and are
    unaffected.

    ``lmcache.mp.l2`` declares the server's L2: ``none``, ``present``, or
    ``auto`` (default), which asks ``<lmcache.mp.http_url>/config/adapters``
    and refuses when that cannot be answered.
    ``lmcache.mp.server_per_rank_layouts: true`` asserts the server keeps one
    layout per kv rank, which lifts the restriction.
    """

    _, pp_size = _validate_mp_config(config)
    if pp_size == 1:
        return
    extra = _extra_config(config)
    per_rank_layouts = extra.get("lmcache.mp.server_per_rank_layouts", False)
    if type(per_rank_layouts) is not bool:
        raise TypeError("lmcache.mp.server_per_rank_layouts must be true or false")
    if per_rank_layouts:
        return
    configured = extra.get("lmcache.mp.l2", "auto")
    l2 = str(configured).strip().lower()
    if l2 not in ("none", "auto", "present"):
        raise ValueError(
            f"lmcache.mp.l2 must be 'none', 'auto', or 'present', got {configured!r}"
        )
    if l2 == "none":
        return
    reason = f"lmcache.mp.l2={l2!r}"
    if l2 == "auto":
        url = _http_url(config) + "/config/adapters"
        try:
            adapters = _fetch_l2_adapters(url)
        except (OSError, TypeError, ValueError, HTTPException) as exc:
            reason = f"cannot list the server's L2 adapters at {url} ({exc})"
        else:
            if not adapters:
                return
            names = [
                str(a.get("type_name", a)) if isinstance(a, dict) else str(a)
                for a in adapters
            ]
            reason = f"the server has L2 adapters {names}"
    raise NotImplementedError(
        f"lmcache_mp with {pp_size} PP stages cannot use an LMCache L2: "
        f"{reason}. LMCache keeps one layout per model and would size L2 "
        "prefetch buffers for one stage's layout. Set lmcache.mp.l2='none' for "
        "an L1-only server, or lmcache.mp.server_per_rank_layouts=true for a "
        "server that keeps per-rank layouts"
    )


def _http_url(config: Any) -> str:
    """Return the LMCache server's HTTP frontend URL.

    Defaults to the MP server's host on LMCache's default HTTP port.
    """

    extra = _extra_config(config)
    configured = extra.get("lmcache.mp.http_url")
    if configured is not None:
        url = str(configured).strip().rstrip("/")
        if not url:
            raise ValueError("lmcache.mp.http_url must be non-empty")
        return url if "://" in url else f"http://{url}"
    host = urlsplit(_server_urls(config)[0]).hostname or "localhost"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:8080"


def _fetch_l2_adapters(url: str, timeout: float = 5.0) -> list[Any]:
    """Return the ``adapters`` list LMCache's ``GET /config/adapters`` reports."""

    # A control-plane probe of a local server: an http_proxy from the
    # environment would answer for it, or refuse a valid server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        document = json.loads(response.read().decode("utf-8"))
    adapters = document.get("adapters") if isinstance(document, dict) else None
    if not isinstance(adapters, list):
        raise TypeError(f"unexpected response {document!r}")
    return adapters


def _make_scheduler_adapter(config: Any, *, checkpoint_spec: Any = None) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPSchedulerAdapter

    _validate_pp_l2_layouts(config)
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


def _make_worker_adapter(
    config: Any, tp_rank: int, *, checkpoint_spec: Any = None
) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPWorkerAdapter

    _validate_pp_l2_layouts(config)
    extra = _extra_config(config)
    return AtomMPWorkerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config, checkpoint_spec=checkpoint_spec),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, tp_rank),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
        heartbeat_interval=float(extra.get("lmcache.mp.heartbeat_interval", 10.0)),
        transfer_mode=_transfer_mode(config),
    )
