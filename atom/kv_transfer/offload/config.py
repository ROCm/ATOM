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

from typing import Any


def build_lmcache_config():
    """Return an ``LMCacheEngineConfig`` from ``LMCACHE_*`` env + extras."""
    from lmcache.v1.config import LMCacheEngineConfig

    cfg = LMCacheEngineConfig.from_env()
    # cufile GDS has no NVMe-GDS hardware here and hangs on init; force off.
    if getattr(cfg, "use_gds", False):
        try:
            cfg.use_gds = False
        except Exception:
            pass
    # TP>1 fix: only rank 0 serves/answers the ZMQ lookup. Without this the
    # client queries all ranks and takes min() over results; we observed rank!=0
    # engine.lookup returning 0 even though that rank stored the chunk
    # (contains()=True) -> min(0, hit)=0 -> the scheduler never sees the hit and
    # always recomputes. Our connector saves on ALL ranks in lockstep, so rank 0
    # is authoritative for "is it offloaded?"; each rank still loads its own KV
    # shard, and _do_load is all-or-nothing (re-prefills if a shard is missing).
    try:
        cfg.lookup_server_worker_ids = [0]
    except Exception:
        pass
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
                try:
                    setattr(cfg, field, value)
                except Exception:
                    pass


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
    num_layers = int(getattr(hf, "num_hidden_layers"))
    tp = int(getattr(config, "tensor_parallel_size", world_size) or 1)
    kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
    model_name = str(getattr(config, "model", "atom-model"))

    # MLA (DeepSeek R1/V3, Kimi) stores a single replicated per-layer latent
    # cache (kv_lora_rank + qk_rope_head_dim), not TP-sharded K/V heads. These
    # dims are bookkeeping only — the codec moves opaque bytes either way. We
    # keep use_mla=False because our BINARY storage bypasses LMCache's own MLA
    # GPU-connector format path; only kv_shape needs to reflect reality.
    if getattr(hf, "kv_lora_rank", None) is not None:
        latent = int(getattr(hf, "kv_lora_rank")) + int(
            getattr(hf, "qk_rope_head_dim", 0)
        )
        kv_shape = (num_layers, 1, int(cfg.chunk_size), 1, latent)
    else:
        num_kv_heads = int(
            getattr(hf, "num_key_value_heads", getattr(hf, "num_attention_heads"))
        )
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
