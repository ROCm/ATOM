# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Builds the Kimi-K3 state tier on a worker, or refuses to.

Every refusal is a warning and a tier-off, never a raise: this runs from
`register_kv_caches`, after the weights load, where a raise wedges startup
instead of reporting. Configurations that must not start are refused by the
engine before any worker exists (`refuse_state_tier_under_pp`).
"""

import logging
import os

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import pp_aware_rank_and_world
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier

logger = logging.getLogger("atom")


def build_state_tier(config, engine, transfer_tensors) -> StateOffloadTier | None:
    """The worker's state tier, or None when this configuration cannot host one."""
    # The config refusals come first, and deliberately above the aiter
    # import: whether this configuration can host a tier at all is decided
    # by config alone, so it must not depend on a GPU library being
    # importable -- otherwise the refusal cannot be reached, or tested, on a
    # CPU runner.
    # PP breaks the tier: the CacheEngineKey has no PP component, so two
    # stages at the same TP rank would overwrite each other. The engine
    # refuses this configuration before any worker exists
    # (`refuse_state_tier_under_pp`); a worker never raises here, because a
    # raise out of `register_kv_caches` lands after the weights are loaded
    # and wedges startup instead of reporting a config error.
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if pp_size > 1:
        logger.warning(
            "kimi_k3 offload: the state tier is unsupported under pipeline "
            "parallelism (pipeline_parallel_size=%d); state tier off.",
            pp_size,
        )
        return None

    backend = getattr(transfer_tensors, "state_backend", None)
    if backend is None:
        logger.warning(
            "kimi_k3 offload: no attention backend published; state tier off."
        )
        return None
    # The geometry the bytes are written under, folded into every key so a
    # changed build cannot read another's images. Read from the runtime, not
    # recomputed, so HBM and CPU sides share one owner of the string.
    spec = getattr(getattr(backend, "model_runner", None), "state_runtime", None)
    spec = getattr(spec, "checkpoint_spec", None)
    layout_id = getattr(spec, "layout_id", None)
    if not layout_id:
        logger.warning(
            "kimi_k3 offload: no checkpoint layout id published; state tier "
            "off. Without it a build that changed the state geometry could "
            "read another's images back as valid."
        )
        return None
    try:
        views = backend.state_entry_views(0)
        entry_bytes = sum(int(v.numel()) * v.element_size() for v in views)
    except (NotImplementedError, AttributeError):
        # No per-request state on this backend. IndexError is deliberately
        # not caught: a zero-entry pool with the tier on is a sizing bug.
        logger.warning(
            "kimi_k3 offload: %s owns no per-request state views; tier off.",
            type(backend).__name__,
        )
        return None

    # The store reads `page_unit_views`, a *different* method than the load's
    # `state_entry_views` (validated above). A backend with one but not the
    # other would build the tier, pass every load, then AttributeError on the
    # first store -- which the tier's blind `except` masks as an endlessly
    # "failed" store. Probe it here so the mismatch fails fast and visibly.
    if not callable(getattr(backend, "page_unit_views", None)):
        logger.warning(
            "kimi_k3 offload: %s has state_entry_views but no callable "
            "page_unit_views; the store path needs it, so a tier would fail "
            "every store silently. State tier off.",
            type(backend).__name__,
        )
        return None

    # The store reads PAGE units and the load writes an Active Slot, so the
    # blob must be the same length both ways (equal for K3: a checkpoint
    # covers the whole slot). A model where they differ would truncate the
    # store or over-read the load -- checked here, where both are in scope.
    image_bytes = int(getattr(spec, "image_bytes", 0) or 0)
    if image_bytes and image_bytes != entry_bytes:
        logger.warning(
            "kimi_k3 offload: a checkpoint image is %d B but an Active Slot "
            "is %d B; the store reads units and the load writes a slot, so "
            "they must match. State tier off.",
            image_bytes,
            entry_bytes,
        )
        return None

    # Imported here, at its one use site: everything above is a refusal
    # decidable without a GPU library.
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    rank, world = pp_aware_rank_and_world(config, tp)
    cfg = offcfg.build_lmcache_config(getattr(config, "kv_transfer_config", None))
    meta = offcfg.build_lmcache_metadata(config, cfg, world, rank)

    # One flat entry, packed on the tier's own `lmc-state` thread. Sized to
    # the entry rather than shared with the KV staging buffer, which is
    # sized in LMCache chunks and is routinely an order of magnitude smaller.
    gpu_connector = engine.gpu_connector
    staged = StagedTransfer(
        gpu_connector.device,
        staging_buffer_bytes=entry_bytes,
        release_after_transfer=gpu_connector.release_gpu_staging_after_transfer,
    )
    codec = StateByteCodec(
        backend,
        staged,
        entry_bytes,
        model_name=meta.model_name,
        world_size=world,
        worker_id=rank,
        layout_id=layout_id,
        max_cache_bytes=int(
            float(os.environ.get("OFFLOAD_STATE_CPU_SIZE", "32")) * (1 << 30)
        ),
    )
    # Share the allocator, but cap state retention separately so old
    # checkpoints cannot consume nearly the whole paged-KV CPU pool.
    codec.bind_storage_manager(engine.storage_manager)
    # No index here: StateOffloadIndex lives in the engine process; both
    # directions report and the engine applies.
    tier = StateOffloadTier(codec)
    logger.info(
        "kimi_k3 offload: state tier up, entry=%.2f MiB rank=%d, "
        "sharing the paged-KV CPU pool, max_state_entries=%d, layout=%s",
        entry_bytes / (1 << 20),
        rank,
        codec._max_entries,
        layout_id,
    )
    return tier
