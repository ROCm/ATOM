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
from numbers import Integral
from typing import Any

import torch

# Version 3 adds the effective index-cache dtype to the PAGE identity. FP4 and
# FP8 DSV4 indexers have different region counts and byte layouts, so they must
# never reuse one another's objects even when the HF model config is identical.
PAGE_LAYOUT_VERSION = 3
_PAGE_FINGERPRINT_BYTES = 16
_OFFLOAD_LAYOUT_ALIASES = {
    "hybrid": "hybrid",
    "dense": "dense",
    "kimi_k3": "kimi_k3",
    "m3": "m3",
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
_HF_INTEGER_GEOMETRY_FIELDS = frozenset(_HF_PAGE_FIELDS) - {
    "compress_ratios",
    "indexer_dtype",
}
# GDN/linear model types carrying a per-request recurrent state but NOT in the
# `kimi_linear` family `kimi_k3` owns. Same set as `ModelRunner.is_qwen_next()`,
# but note the attribute skew: the guard below reads `hf_config.model_type`
# while `is_qwen_next` reads `hf_text_config.model_type`. On the native path
# they agree only because `get_hf_config` flattens the text type up; `is_vllm()`
# drops `qwen3_5`/`qwen3_5_moe` from that flatten, so if offload is ever wired
# into plugin mode this guard must switch to the text config to stay in sync.
# Unreachable today -- no plugin caller passes a `kv_transfer_config`.
# MiniMax-M3 is not here either: it is sparse *paged* attention
# (`MiniMaxM3SparseForCausalLM` -> `SparseMHAPagedAttentionImpl`), no recurrent
# state, so it is NOT a GDN/linear model. It does get its own PAGE-only `m3`
# layout (see `_is_minimax_m3` and offload/hybrid/m3/): its NSA index_cache is
# delivered only through transfer_tensors.block_regions, so the dense codec --
# which builds from per-layer KVCacheTensor K/V -- would silently drop it.
_GDN_LINEAR_MODEL_TYPES = frozenset(
    {"qwen3_next", "qwen3_next_mtp", "qwen3_5_text", "qwen3_5_moe_text"}
)

# Layouts that own a tier for a model's per-request *recurrent* state. Today only
# `kimi_k3` (the kimi_linear KDA state); `hybrid` (DSV4 sparse-attention
# checkpoints) is not recurrent-state and dense<->hybrid is a legitimate operator
# override for namespace separation. An explicit `offload_layout` override may not
# downgrade a recurrent-state model to a layout that owns no tier for it (silent
# wrong output) -- see `select_offload_layout`. GDN/linear model types are refused
# outright in `_layout_from_model`, so no override can resurrect them.
_STATE_OWNING_LAYOUTS = frozenset({"kimi_k3"})

logger = logging.getLogger("atom")


def _strict_integer(name: str, value: object, *, minimum: int = 0) -> int:
    """Parse integer geometry without silently truncating or coercing text."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    normalized = int(value)
    if normalized < minimum:
        relation = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return normalized


def _is_minimax_m3(hf_config) -> bool:
    """MiniMax-M3: PAGE-only NSA sparse attention, no SLOT/compression.

    Detected by architecture / model_type name so the offload layout
    resolves identically in the scheduler and worker processes
    (config-only selection, matching select_offload_layout's contract).
    """

    if hf_config is None:
        return False
    architectures = getattr(hf_config, "architectures", None) or ()
    for arch in architectures:
        name = str(arch).lower()
        if "minimaxm3" in name or "minimax_m3" in name:
            return True
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    return "minimax_m3" in model_type or "minimaxm3" in model_type


def select_offload_layout(config) -> str:
    """Resolve the config-only dense/hybrid layout selection.

    The worker, scheduler, and LMCache key namespace must all use this exact
    selector. Otherwise an explicit layout override can make two incompatible
    byte layouts share one storage namespace.
    """

    kvc = getattr(config, "kv_transfer_config", {}) or {}
    override = kvc.get("offload_layout")
    # Inspect the model *first*, always -- even with an override present. An
    # override returning before this ran let `offload_layout: dense` on a
    # kimi_linear/Qwen3.5 checkpoint skip the GDN refusal below and restore a KV
    # prefix over stale recurrent state (silent wrong output). The natural layout
    # is also what an override is validated against.
    natural = _layout_from_model(config)
    if override is None:
        return natural
    mapped = (
        _OFFLOAD_LAYOUT_ALIASES.get(override) if isinstance(override, str) else None
    )
    if mapped is None:
        raise ValueError(
            f"lmcache_offload: unknown offload_layout={override!r}; "
            f"expected one of {sorted(_OFFLOAD_LAYOUT_ALIASES)}"
        )
    # An override may pick between compatible layouts, but it may not strip a
    # state-owning model down to a layout with no tier for its per-request state
    # -- the same silent-wrong-output the GDN refusal pre-empts, reached by a
    # different door (an explicit `dense` rather than a fall-through).
    if natural in _STATE_OWNING_LAYOUTS and mapped not in _STATE_OWNING_LAYOUTS:
        raise ValueError(
            f"lmcache_offload: offload_layout={override!r} would run a "
            f"{natural!r}-family model (per-request recurrent state) on the "
            f"{mapped!r} layout, which owns no tier for that state -- its KV "
            "prefix would be restored over stale state (silent wrong output). "
            "Drop the override, or disable offload for this model."
        )
    return mapped


def _layout_from_model(config) -> str:
    """The dense/hybrid/kimi_k3 layout the model itself implies (no override).

    May raise: a GDN/linear model that no layout owns a state tier for is
    refused here (the message names the cause) rather than left to die deep in
    `register_kv_caches` as a byte-layout mismatch.
    """
    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        return "dense"
    # MiniMax-M3 (NSA sparse *paged* attention, no recurrent state) gets a
    # PAGE-only layout whose codec is sourced from transfer_tensors.block_regions
    # -- the NSA index_cache rides there, never as a KVCacheTensor the dense codec
    # inspects, so `dense` would offload K/V and silently drop the index cache,
    # corrupting restored prefixes. Checked before the text-config family probe
    # below because M3 is identified by architecture name, not model_type family.
    if _is_minimax_m3(hf_config):
        return "m3"
    # Resolve the family off the *text* config, exactly as the model-side
    # predicates this mirrors do (`ModelRunner.is_qwen_next`/`is_kimi_linear`
    # read `self.hf_text_config`). Two names in `_GDN_LINEAR_MODEL_TYPES` --
    # `qwen3_5_text`, `qwen3_5_moe_text` -- are literally text-config model
    # types, so the bare wrapper `model_type` would be the multimodal wrapper's
    # and neither the `kimi_linear` nor the GDN branch would fire: Qwen3.5 would
    # fall through to `dense` and die deep in `register_kv_caches` (the exact
    # failure the refusal below exists to pre-empt), and a `kimi_linear`
    # checkpoint wrapped in a `text_config` would silently run with no state tier.
    # Local import: this module is on the early config path, and `atom.utils`
    # pulls in transformers -- keep it out of module import time.
    from atom.utils import get_hf_text_config

    text_config = get_hf_text_config(hf_config)
    model_type = getattr(text_config, "model_type", None)
    # K3's paged KV is ordinary dense MLA; what makes it its own family is the
    # KDA per-request state that a reusable prefix also needs.
    if model_type == "kimi_linear":
        return "kimi_k3"
    if getattr(hf_config, "compress_ratios", None):
        return "hybrid"
    # GDN/linear models carry a per-request recurrent state no layout owns a
    # tier for; `dense` would restore a KV prefix over stale state (silent wrong
    # output). The dense codec fails closed on the state tensor, but that fires
    # deep in `register_kv_caches` as a byte-layout mismatch -- refuse here where
    # the message can name the cause.
    if model_type in _GDN_LINEAR_MODEL_TYPES:
        raise ValueError(
            "lmcache_offload does not support GDN/linear-attention models "
            f"(model_type={model_type!r}; e.g. "
            "Qwen3-Next, Qwen3.5). These carry a per-request recurrent state "
            "that no offload layout owns a tier for -- restoring their KV "
            "prefix while that state is stale is silent wrong output. Disable "
            "offload for this model, or use a state-owning layout once one "
            "exists for it (kimi_k3 owns only the kimi_linear family)."
        )
    return "dense"


def _stable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
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


def _stable_hf_geometry(hf: object) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    for name in _HF_PAGE_FIELDS:
        value = getattr(hf, name, None)
        if value is not None and name in _HF_INTEGER_GEOMETRY_FIELDS:
            value = _strict_integer(f"PAGE {name}", value)
        elif value is not None and name == "compress_ratios":
            if not isinstance(value, (list, tuple)):
                raise ValueError("PAGE compress_ratios must be a sequence of integers")
            value = [
                _strict_integer(f"PAGE compress_ratios[{index}]", ratio)
                for index, ratio in enumerate(value)
            ]
        geometry[name] = _stable_config_value(value)
    return geometry


def build_page_namespace(
    config,
    cfg,
    world_size: int,
    *,
    layout_version: int = PAGE_LAYOUT_VERSION,
) -> str:
    """Return the stable CacheEngineKey domain for ATOM PAGE bytes."""

    layout_version = _strict_integer(
        "PAGE layout version",
        layout_version,
        minimum=1,
    )
    hf = config.hf_config
    base_model_name = str(
        getattr(config, "model_tag", None) or getattr(config, "model", "atom-model")
    )
    document = {
        "schema": "atom-page-namespace",
        "layout_version": layout_version,
        "model": base_model_name,
        "page_mode": {
            "hybrid": "dsv4-page-regions",
            "m3": "m3-page-regions",
        }.get(select_offload_layout(config), "dense-opaque-block"),
        "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
        "index_cache_dtype": str(getattr(config, "index_cache_dtype", "auto")),
        "block_size": _strict_integer(
            "PAGE block size",
            config.kv_cache_block_size,
            minimum=1,
        ),
        "chunk_size": _strict_integer(
            "PAGE chunk size",
            cfg.chunk_size,
            minimum=1,
        ),
        "tp_size": _strict_integer("PAGE TP size", world_size, minimum=1),
        "dcp_size": _strict_integer(
            "PAGE DCP size",
            (
                1
                if getattr(config, "decode_context_parallel_size", 1) is None
                else getattr(config, "decode_context_parallel_size", 1)
            ),
            minimum=1,
        ),
        "hf_geometry": _stable_hf_geometry(hf),
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
        person=b"ATOM-PAGE-CFG-v3",
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


def lmcache_engine_id(config) -> str:
    """Return the LMCache engine id for this process, namespaced by DP rank.

    A scheduler's ``LookupClient`` and its workers' ``LookupServer`` derive the
    same ZMQ ipc socket path from this id, so it must be identical inside one
    DP replica and distinct between replicas. Every replica owns a private
    CPU/NVMe pool; a shared id makes all replicas bind the same socket (only
    the first wins) and every scheduler then queries replica 0's hit counts,
    emitting loads its own workers cannot serve.

    Under ``--enable-dp-attention`` this is the normal case, not a corner case:
    ``CoreManager`` folds TP into DP, so each GPU becomes its own replica with
    ``tensor_parallel_size == 1``.
    """
    dp_rank = int(
        getattr(
            getattr(config, "parallel_config", None),
            "data_parallel_rank",
            0,
        )
        or 0
    )
    return f"atom-offload-dp{dp_rank}"


def lmcache_replica_world_size(config) -> int:
    """Return the number of LMCache workers inside one DP replica (PP x TP).

    Worker ids index this replica-local grid rather than the global one.
    LMCache selects its lookup servers by worker id
    (``cfg.lookup_server_worker_ids``, pinned to ``[0]`` above), so global
    numbering would leave every replica except the first without a server.
    Replica-local ids are also the right cache-key component: id ``i`` means
    "shard i of the model", which holds the same bytes in every replica, so
    replicas sharing a disk/remote backend share entries instead of
    duplicating them.
    """
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    tp_size = int(getattr(config, "tensor_parallel_size", 1) or 1)
    return max(1, pp_size * tp_size)


def scale_cpu_size_for_pp(cfg, config) -> None:
    """Split the CPU offload budget across PP stages by layer count.

    Give each stage a share proportional to its bound layer count so all
    stages reach the same token horizon. Loads are all-or-nothing across
    stages, so unequal horizons waste the longer stage's extra capacity.

    The last PP stage binds the draft KV layer (spec decode), so its
    layer count is one more than its target-model slice.
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
    except AttributeError:
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
            if not field or not hasattr(cfg, field):
                raise ValueError(f"unknown LMCache override {key!r}")
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
    total_layers = _strict_integer(
        "LMCache layer count",
        hf.num_hidden_layers,
        minimum=1,
    )
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
    raw_tp = getattr(config, "tensor_parallel_size", world_size)
    tp = _strict_integer(
        "LMCache tensor parallel size",
        world_size if raw_tp is None else raw_tp,
        minimum=1,
    )
    chunk_size = _strict_integer("LMCache chunk size", cfg.chunk_size, minimum=1)
    world_size = _strict_integer("LMCache world size", world_size, minimum=1)
    worker_id = _strict_integer("LMCache worker id", worker_id)
    if worker_id >= world_size:
        raise ValueError("LMCache worker id must be within the world")
    kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
    model_name = build_page_namespace(config, cfg, world_size)

    # MLA (DeepSeek R1/V3, Kimi) stores a single replicated per-layer latent
    # cache (kv_lora_rank + qk_rope_head_dim), not TP-sharded K/V heads. These
    # dims are bookkeeping only — the codec moves opaque bytes either way. We
    # keep use_mla=False because our BINARY storage bypasses LMCache's own MLA
    # GPU-connector format path; only kv_shape needs to reflect reality.
    if getattr(hf, "kv_lora_rank", None) is not None:
        latent = _strict_integer(
            "LMCache KV LoRA rank",
            hf.kv_lora_rank,
            minimum=1,
        ) + _strict_integer(
            "LMCache RoPE head dimension",
            getattr(hf, "qk_rope_head_dim", 0),
        )
        kv_shape = (num_layers, 1, chunk_size, 1, latent)
    else:
        num_kv_heads = _strict_integer(
            "LMCache KV head count",
            getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            minimum=1,
        )
        num_kv_heads_local = max(1, num_kv_heads // tp)
        raw_head_dim = getattr(hf, "head_dim", 0)
        normalized_head_dim = (
            0
            if raw_head_dim is None
            else _strict_integer("LMCache head dimension", raw_head_dim)
        )
        if normalized_head_dim == 0:
            hidden_size = _strict_integer(
                "LMCache hidden size",
                hf.hidden_size,
                minimum=1,
            )
            attention_heads = _strict_integer(
                "LMCache attention head count",
                hf.num_attention_heads,
                minimum=1,
            )
            head_dim = hidden_size // attention_heads
        else:
            head_dim = normalized_head_dim
        kv_shape = (num_layers, 2, chunk_size, num_kv_heads_local, head_dim)

    return LMCacheMetadata(
        model_name=model_name,
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=chunk_size,
        # Shared within a DP replica so the scheduler's ZMQ LookupClient and
        # each worker's LookupServer derive the SAME ipc socket path
        # (get_zmq_rpc_path_lmcache), distinct across replicas so they do not
        # collide on it.
        engine_id=lmcache_engine_id(config),
    )
