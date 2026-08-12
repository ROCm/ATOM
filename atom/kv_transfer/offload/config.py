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

import logging
from typing import Any

logger = logging.getLogger(__name__)


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


def scale_cpu_size_for_pp(cfg, config) -> None:
    """Split the CPU offload budget across PP stages by layer count.

    Every stage reads the same ``LMCACHE_MAX_LOCAL_CPU_SIZE``, but a stage only
    holds its own slice of layers, so its bytes-per-token scales with that
    slice. Equal GB therefore buys unequal token horizons: under an uneven
    split like 18,20,20,20 the 18-layer stage caches ~11% more tokens than the
    others. Loads are all-or-nothing across stages, so those extra tokens are
    never usable — a prefix that survived only on the widest stage still forces
    a full recompute, and the bytes that stage read back are wasted. Measured
    on a 4-stage GLM-5.2 agentic replay: 22.4% of requests fell back this way
    while one stage had the whole prefix in hand.

    Give each stage a share proportional to its layer count, leaving the total
    (``pp_size * configured``) unchanged. The horizon then works out to
    ``configured * pp_size / (total_layers * bytes_per_token_per_layer)`` on
    every stage, independent of how the layers were split.

    Speculative decode binds the draft's KV layer on the last PP stage only, and
    the offload codec moves every bound layer — so that stage's bytes-per-token
    covers one more layer than its slice of the target. Counting only target
    layers would under-budget it and, since loads are all-or-nothing, drag the
    whole node's horizon down to the short stage.
    """
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if pp_size <= 1:
        return

    configured = float(getattr(cfg, "max_local_cpu_size", 0.0) or 0.0)
    if configured <= 0:
        return

    from atom.models.utils import get_pp_indices

    num_hidden_layers = int(config.hf_config.num_hidden_layers)
    pp_rank = int(
        getattr(
            getattr(config, "parallel_config", None),
            "pipeline_parallel_rank",
            0,
        )
    )
    start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    local_layers = end - start

    spec = getattr(config, "speculative_config", None)
    draft_layers = 0
    if spec is not None:
        draft_hf = getattr(spec, "draft_model_hf_config", None)
        draft_layers = int(getattr(draft_hf, "num_nextn_predict_layers", 1) or 0)
    total_layers = num_hidden_layers + draft_layers
    if pp_rank == pp_size - 1:
        local_layers += draft_layers

    if local_layers <= 0 or total_layers <= 0:
        return

    scaled = configured * pp_size * local_layers / total_layers
    try:
        cfg.max_local_cpu_size = scaled
    except Exception:
        return
    logger.info(
        "LMCache CPU budget for PP stage %d/%d: %.2fGB -> %.2fGB "
        "(%d/%d layers; total across stages unchanged at %.2fGB)",
        pp_rank,
        pp_size,
        configured,
        scaled,
        local_layers,
        total_layers,
        configured * pp_size,
    )


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

    from atom.models.utils import get_pp_indices

    hf = config.hf_config
    total_layers = int(hf.num_hidden_layers)
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    pp_rank = getattr(
        getattr(config, "parallel_config", None),
        "pipeline_parallel_rank",
        0,
    )
    if pp_size > 1:
        start, end = get_pp_indices(total_layers, pp_rank, pp_size)
        num_layers = end - start
    else:
        num_layers = total_layers
    tp = int(getattr(config, "tensor_parallel_size", world_size) or 1)
    kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
    model_name = str(getattr(config, "model", "atom-model"))

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
