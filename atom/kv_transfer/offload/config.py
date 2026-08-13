# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Build the per-rank ``LMCacheEngineConfig`` + ``LMCacheMetadata`` for the
ATOM standalone offload connector.

LMCache is driven by ``LMCACHE_*`` env vars (``LMCACHE_LOCAL_CPU``,
``LMCACHE_MAX_LOCAL_CPU_SIZE``, ``LMCACHE_CHUNK_SIZE``, ``LMCACHE_LOCAL_DISK``,
``LMCACHE_MAX_LOCAL_DISK_SIZE`` …) exactly like the vLLM recipe. We additionally
allow overrides via ``kv_transfer_config`` extras keyed ``lmcache.<field>`` and
force ``use_gds=False`` (cufile GDS init hangs without NVMe-GDS hardware).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from math import isfinite
from typing import Any

# Version 2 adds explicit DSV4 KV/index dimensions and DCP virtual-block byte
# mapping to the PAGE identity. Keeping it separate prevents version-1 objects
# from being reused after the layout correction.
PAGE_LAYOUT_VERSION = 2
_PAGE_FINGERPRINT_BYTES = 16
_OFFLOAD_LAYOUT_ALIASES = {
    "hybrid": "hybrid",
    "dense": "dense",
    # Compatibility with the names used while the layout split was developed.
    "terminal_unit": "hybrid",
    "page_slot": "hybrid",
    "chunked": "dense",
    "chunked_mla": "dense",
}
_HF_PAGE_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "head_dim",
    "kv_head_dim",
    "index_head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "compress_ratios",
    "indexer_dtype",
)

logger = logging.getLogger("atom")


def select_offload_layout(config) -> str:
    """Resolve the config-only dense/hybrid layout selection.

    The worker, scheduler, and LMCache key namespace must all use this exact
    selector. Otherwise an explicit layout override can make two incompatible
    byte layouts share one storage namespace.
    """

    kvc = getattr(config, "kv_transfer_config", {}) or {}
    override = kvc.get("offload_layout")
    if override is not None:
        mapped = _OFFLOAD_LAYOUT_ALIASES.get(str(override))
        if mapped is not None:
            return mapped
        logger.warning(
            "lmcache_offload: unknown offload_layout=%r; falling back to auto",
            override,
        )

    hf_config = getattr(config, "hf_config", None)
    if hf_config is not None and getattr(hf_config, "compress_ratios", None):
        return "hybrid"
    return "dense"


def _stable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("PAGE namespace config values must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value):
        return _stable_config_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _stable_config_value(model_dump(mode="json"))
    if hasattr(value, "value") and isinstance(value.value, (bool, int, float, str)):
        return _stable_config_value(value.value)
    if hasattr(value, "__dict__"):
        return _stable_config_value(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_") and not callable(item)
            }
        )
    raise TypeError(
        "PAGE namespace config contains unsupported value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def build_page_namespace(
    config,
    cfg,
    world_size: int,
    *,
    layout_version: int = PAGE_LAYOUT_VERSION,
) -> str:
    """Return the stable CacheEngineKey domain for ATOM PAGE bytes."""

    if isinstance(layout_version, bool) or not isinstance(layout_version, int):
        # Preserve the public configuration-validation API.
        raise ValueError("PAGE layout version must be an integer")  # noqa: TRY004
    if layout_version <= 0:
        raise ValueError("PAGE layout version must be positive")
    hf = config.hf_config
    base_model_name = str(
        getattr(config, "model_tag", None) or getattr(config, "model", "atom-model")
    )
    document = {
        "schema": "atom-page-namespace",
        "layout_version": layout_version,
        "model": base_model_name,
        "page_mode": (
            "dsv4-page-regions"
            if select_offload_layout(config) == "hybrid"
            else "dense-opaque-block"
        ),
        "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
        "block_size": int(config.kv_cache_block_size),
        "chunk_size": int(cfg.chunk_size),
        "tp_size": int(world_size),
        "dcp_size": int(getattr(config, "decode_context_parallel_size", 1) or 1),
        "hf_geometry": {
            name: _stable_config_value(getattr(hf, name, None))
            for name in _HF_PAGE_FIELDS
        },
        "speculative_config": _stable_config_value(
            getattr(config, "speculative_config", None)
        ),
    }
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(
        canonical,
        digest_size=_PAGE_FINGERPRINT_BYTES,
        person=b"ATOM-PAGE-CFG-v2",
    ).hexdigest()
    return f"{base_model_name}::atom-page-v{layout_version}-{digest}"


def build_lmcache_config(
    kv_transfer_config: dict[str, Any] | None = None,
) -> Any:
    """Return a validated ``LMCacheEngineConfig`` for ATOM offload.

    LMCache's local-disk backend always uses the local-CPU allocator as its
    host staging pool, even when the CPU hot-cache tier is disabled. Validate
    that relationship here so an incomplete NVMe configuration fails during
    connector startup instead of silently running CPU-only.

    Args:
        kv_transfer_config: Optional ATOM connector configuration containing
            ``lmcache.<field>`` overrides.

    Returns:
        The finalized LMCache engine configuration.

    Raises:
        ValueError: If the local-disk path, capacity, or CPU staging capacity
            is incomplete.
    """
    from lmcache.v1.config import LMCacheEngineConfig

    cfg = LMCacheEngineConfig.from_env()
    apply_extra_overrides(cfg, kv_transfer_config)
    # cufile GDS has no NVMe-GDS hardware here and hangs on init; force off.
    if getattr(cfg, "use_gds", False):
        cfg.use_gds = False
    # TP>1 fix: only rank 0 serves/answers the ZMQ lookup. Without this the
    # client queries all ranks and takes min() over results; we observed rank!=0
    # engine.lookup returning 0 even though that rank stored the chunk
    # (contains()=True) -> min(0, hit)=0 -> the scheduler never sees the hit and
    # always recomputes. Our connector saves on ALL ranks in lockstep, so rank 0
    # is authoritative for "is it offloaded?"; each rank still loads its own KV
    # shard, and _do_load is all-or-nothing (re-prefills if a shard is missing).
    cfg.lookup_server_worker_ids = [0]
    validate_lmcache_storage_config(cfg)
    return cfg


def apply_extra_overrides(cfg, kv_transfer_config: dict[str, Any] | None) -> None:
    """Apply ``{"lmcache.<field>": value}`` extras from kv_transfer_config."""
    if not kv_transfer_config:
        return
    extra = kv_transfer_config.get("kv_connector_extra_config", kv_transfer_config)
    for key, value in (extra or {}).items():
        if isinstance(key, str) and key.startswith("lmcache."):
            field = key[len("lmcache.") :]
            if hasattr(cfg, field):
                setattr(cfg, field, value)


def validate_lmcache_storage_config(cfg: Any) -> None:
    """Validate the host-staged LMCache local-disk configuration.

    Args:
        cfg: An LMCache engine configuration object.

    Raises:
        ValueError: If only one of the local-disk path/capacity is configured,
            or if no local-CPU staging capacity is available for disk I/O.
    """
    local_disk = getattr(cfg, "local_disk", None)
    disk_path_configured = bool(str(local_disk).strip()) if local_disk else False
    try:
        disk_size_gib = float(getattr(cfg, "max_local_disk_size", 0.0) or 0.0)
        cpu_size_gib = float(getattr(cfg, "max_local_cpu_size", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LMCache CPU/disk capacities must be numeric GiB values"
        ) from exc

    if disk_path_configured and disk_size_gib <= 0:
        raise ValueError(
            "LMCACHE_LOCAL_DISK is set but LMCACHE_MAX_LOCAL_DISK_SIZE must be > 0"
        )
    if not disk_path_configured and disk_size_gib > 0:
        raise ValueError(
            "LMCACHE_MAX_LOCAL_DISK_SIZE is set but LMCACHE_LOCAL_DISK is missing"
        )
    if disk_path_configured and cpu_size_gib <= 0:
        raise ValueError(
            "LMCache local-disk offload requires LMCACHE_MAX_LOCAL_CPU_SIZE > 0 "
            "for the host staging allocator, even when LMCACHE_LOCAL_CPU=False"
        )


def build_lmcache_metadata(config, cfg, world_size: int, worker_id: int):
    """Build ``LMCacheMetadata`` for this rank from ATOM ``config`` + LMCache cfg.

    ``kv_shape`` follows LMCache's ``(num_layers, 2, chunk_size, num_kv_heads,
    head_dim)`` convention. For our opaque BINARY-style storage the exact dims
    are only used for key/shape bookkeeping (we override the byte layout in the
    codec), but we fill them faithfully from hf_config so logging/keys are sane.
    """
    from aiter import dtypes
    from lmcache.v1.metadata import LMCacheMetadata

    hf = config.hf_config
    num_layers = int(hf.num_hidden_layers)
    tp = int(getattr(config, "tensor_parallel_size", world_size) or 1)
    kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
    model_name = build_page_namespace(config, cfg, world_size)

    # MLA (DeepSeek R1/V3, Kimi) stores a single replicated per-layer latent
    # cache (kv_lora_rank + qk_rope_head_dim), not TP-sharded K/V heads. These
    # dims are bookkeeping only — the codec moves opaque bytes either way. We
    # keep use_mla=False because our BINARY storage bypasses LMCache's own MLA
    # GPU-connector format path; only kv_shape needs to reflect reality.
    if getattr(hf, "kv_lora_rank", None) is not None:
        latent = int(hf.kv_lora_rank) + int(getattr(hf, "qk_rope_head_dim", 0))
        kv_shape = (num_layers, 1, int(cfg.chunk_size), 1, latent)
    else:
        num_kv_heads = int(getattr(hf, "num_key_value_heads", hf.num_attention_heads))
        num_kv_heads_local = max(1, num_kv_heads // tp)
        head_dim = int(
            getattr(hf, "head_dim", 0) or (hf.hidden_size // hf.num_attention_heads)
        )
        kv_shape = (num_layers, 2, int(cfg.chunk_size), num_kv_heads_local, head_dim)

    return LMCacheMetadata(
        model_name=model_name,
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=int(cfg.chunk_size),
        # Shared id so the scheduler's ZMQ LookupClient and each worker's
        # LookupServer derive the SAME ipc socket path (get_zmq_rpc_path_lmcache).
        engine_id="atom-offload",
    )
