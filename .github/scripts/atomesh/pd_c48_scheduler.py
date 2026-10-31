"""Record cache gates; optionally observe the unchanged scheduler methods."""
import copy
import hashlib
import json
import os
from pathlib import Path

from vllm.v1.simple_kv_offload import manager as simple_kv_offload_manager


def enable_mooncake_divergent_hits(scheduler, config):
    """Let a Mooncake decode reuse a local full-attention hit whose Mamba state lags, as MoRIIO READ does.

    The hybrid lookup otherwise needs a cached Mamba boundary, which a decode that only loads never stores,
    so every request pulled its whole prompt (job 10093: 0 local hits). Mooncake pulls the unhashed blocks
    of every group, the running Mamba slot included, so the state at the resume point still comes from P.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import MooncakeConnector
    connector = scheduler.connector
    if type(connector) is not MooncakeConnector or config.kv_transfer_config.kv_role != "kv_consumer":
        raise RuntimeError(f"C48_MOONCAKE_DIVERGENT_HITS needs a decode MooncakeConnector, got {type(connector)}")
    MooncakeConnector.supports_divergent_local_hybrid_hits = property(lambda self: True)


def enable_dspark_shared_prefix_checkpoint(scheduler, config):
    """Turn on the shared-prefix (junction) Mamba checkpoint for a DSpark drafter.

    Upstream gates the junction on an EAGLE group, but DSpark drops no EAGLE block (3966924d), so the gate
    leaves it off. The manager's EAGLE handling is additive (it also keeps junction - block_size); without
    EAGLE a sibling resumes at the junction itself, which is the plain path.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    if not (scheduler.cache_config.enable_mamba_shared_prefix_checkpoint
            and config.speculative_config is not None and config.speculative_config.use_dspark()
            and not config.speculative_config.use_eagle_block_drop() and not coordinator.eagle_group_ids
            and coordinator.enable_partial_hash_hits and coordinator.num_reprefillable_tokens == 0
            and scheduler.mamba_partial_cache_hit):
        raise RuntimeError("C48_DSPARK_JUNCTION needs a requested junction, DSpark without EAGLE groups, "
                           "partial hash hits, no re-prefill tail and partial Mamba hits")
    manager.mamba_shared_prefix_checkpoint = True
    for single in coordinator.single_type_managers:
        if isinstance(single, MambaManager):
            single.shared_prefix_checkpoint = True
    scheduler.mamba_shared_prefix_checkpoint = True


def align_tail_checkpoint_to_hash(scheduler):
    """Under no_tail_split, predict the final chunk's internal Mamba checkpoint on the hash grid.

    With partial Mamba hits off, `_mamba_block_aligned_split` places that checkpoint on the scheduler block
    (7168), while MambaManager and the worker export it on `cache_hit_alignment_tokens` (the hash block). The
    scheduler then misses the checkpoint the backend would write and stops at the last Mamba block boundary,
    so the tail runs one more forward over the whole cached context.
    """
    coordinator = scheduler.kv_cache_manager.coordinator
    if not (coordinator.enable_partial_hash_hits and scheduler.mamba_has_prefill_checkpoint_blocks
            and not scheduler.mamba_partial_cache_hit):
        raise RuntimeError("C48_P_TAIL_CKPT_HASH needs no_tail_split, partial hash hits and internal checkpoints")
    split = scheduler._mamba_block_aligned_split
    block_size = scheduler.block_size

    # The split reads scheduler.block_size only as the checkpoint grid; the scheduler is single-threaded.
    def hash_aligned_split(*args, **kwargs):
        scheduler.block_size = scheduler.hash_block_size
        try:
            return split(*args, **kwargs)
        finally:
            scheduler.block_size = block_size

    scheduler._mamba_block_aligned_split = hash_aligned_split


def subblock_fill(scheduler):
    """Let a non-final chunk end inside a Mamba block instead of flooring to 0 tokens.

    Under PP8 the full KDA state sits on one GPU, so the Mamba block is 6400 tokens and `_mamba_block_aligned_split`
    floors every non-final chunk to it. A long prefill then takes one block per 12288-token step, and any request
    starting on a block boundary gets 0 of the 5888 left. The split already allows a sub-block chunk when the block
    is wider than long_prefill_token_threshold; the next chunk re-aligns at the boundary, so cached states stay
    block-aligned.
    """
    config = scheduler.scheduler_config
    if config.long_prefill_token_threshold:
        raise RuntimeError("C48_P_SUBBLOCK_FILL needs long_prefill_token_threshold 0")
    split = scheduler._mamba_block_aligned_split
    block = scheduler.cache_config.block_size

    # The split reads the threshold only for that check; schedule() read its own copy before. Single-threaded.
    def fill_split(*args, **kwargs):
        config.long_prefill_token_threshold = block - 1
        try:
            return split(*args, **kwargs)
        finally:
            config.long_prefill_token_threshold = 0

    scheduler._mamba_block_aligned_split = fill_split


def track_superseded_states(store):
    """Mark each Mamba state a Prefill resumed from, or checkpointed past, as superseded by its next boundary state.

    Every turn writes a ~616 MB fp32 boundary state that only the conversation's next turn reads. The old
    ones filled mc19's Store, and LRU then evicted idle conversations' prefixes first. The worker removes
    marked keys (pd_c48_worker.drop_superseded_states).
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import BoundaryPut
    match, build = store.get_num_new_matched_tokens, store.build_connector_meta
    last = {}

    def get_num_new_matched_tokens(request, num_computed_tokens):
        result = match(request, num_computed_tokens)
        if result[0] is not None:
            last[request.request_id] = num_computed_tokens + result[0]
        return result

    def build_connector_meta(scheduler_output):
        meta = build(scheduler_output)
        for req_meta in meta.requests:
            mamba = [p for p in req_meta.boundary_puts or [] if p.group_id in store._boundary_state_group_ids]
            ends = sorted({p.num_tokens for p in mamba})
            start = last.get(req_meta.req_id, 0)
            if not ends or ends[0] <= start:
                continue
            # The worker keys a state by the hash ending at its position, so only hash-grid positions have one.
            groups = {p.group_id: p.block_id for p in mamba}
            req_meta.c48_superseded = [BoundaryPut(g, b, n) for n in [start, *ends[:-1]]
                                       if n and n % store._hash_block_size == 0 for g, b in groups.items()]
            last[req_meta.req_id] = ends[-1]
        # Finish-time tails ride the step that reports the request finished, so they were handled above.
        for req_id in scheduler_output.finished_req_ids:
            last.pop(req_id, None)
        return meta

    store.get_num_new_matched_tokens = get_num_new_matched_tokens
    store.build_connector_meta = build_connector_meta


def mooncake_store_scheduler(scheduler):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    stores = [c.connector_scheduler for c in getattr(scheduler.connector, "_connectors", [scheduler.connector])
              if isinstance(c, MooncakeStoreConnector)]
    if len(stores) != 1:
        raise RuntimeError(f"C48_STORE_DROP_SUPERSEDED needs one Prefill MooncakeStoreConnector, got {len(stores)}")
    return stores[0]


def create_scheduler(*args, **kwargs):
    config = kwargs.get("vllm_config") if "vllm_config" in kwargs else args[0]
    defaults = copy.copy(config.scheduler_config)
    defaults.scheduler_cls = None
    base = defaults.get_scheduler_cls()
    scheduler = base(*args, **kwargs)
    if os.environ.get("C48_MOONCAKE_DIVERGENT_HITS") == "1":
        enable_mooncake_divergent_hits(scheduler, config)
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    requested = scheduler.cache_config.enable_mamba_shared_prefix_checkpoint
    no_tail_split = (
        os.environ.get("C48_P_NO_TAIL_SPLIT") == "1"
        and os.environ["C48_DIAG_ROLE"] == "prefill"
    )
    if no_tail_split:
        if scheduler.mamba_shared_prefix_checkpoint:
            raise RuntimeError("C48_P_NO_TAIL_SPLIT does not support shared-prefix checkpoints")
        # Skip the extra Prefill step that only registers the prompt's partial-tail
        # Mamba state; internal checkpoints and block-boundary states still cache.
        scheduler.mamba_partial_cache_hit = False
    tail_ckpt_hash = no_tail_split and os.environ.get("C48_P_TAIL_CKPT_HASH") == "1"
    if tail_ckpt_hash:
        align_tail_checkpoint_to_hash(scheduler)
    fill = os.environ.get("C48_P_SUBBLOCK_FILL") == "1" and os.environ["C48_DIAG_ROLE"] == "prefill"
    if fill:
        subblock_fill(scheduler)
    if os.environ.get("C48_DSPARK_JUNCTION") == "1" and os.environ["C48_DIAG_ROLE"] == "prefill":
        enable_dspark_shared_prefix_checkpoint(scheduler, config)
    if os.environ.get("C48_STORE_DROP_SUPERSEDED") == "1" and os.environ["C48_DIAG_ROLE"] == "prefill":
        if scheduler.mamba_shared_prefix_checkpoint:
            raise RuntimeError("C48_STORE_DROP_SUPERSEDED would remove shared-prefix checkpoints")
        track_superseded_states(mooncake_store_scheduler(scheduler))
    if (os.environ.get("C48_EXPECT_SHARED_PREFIX_CHECKPOINT") == "1"
            and os.environ["C48_DIAG_ROLE"] == "prefill"
            and not (requested and scheduler.mamba_shared_prefix_checkpoint
                     and manager.mamba_shared_prefix_checkpoint)):
        raise RuntimeError("C72 junction candidate did not enable shared-prefix checkpoints")
    record = {
        "role": os.environ["C48_DIAG_ROLE"],
        "no_tail_split": no_tail_split,
        "tail_ckpt_hash": tail_ckpt_hash,
        "subblock_fill": fill,
        "scheduler_class": base.__module__ + "." + base.__name__,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "requested_shared_prefix_checkpoint": requested,
        "scheduler_shared_prefix_checkpoint": scheduler.mamba_shared_prefix_checkpoint,
        "manager_shared_prefix_checkpoint": manager.mamba_shared_prefix_checkpoint,
        "partial_hash_hits": coordinator.enable_partial_hash_hits,
        "mamba_partial_cache_hit": scheduler.mamba_partial_cache_hit,
        "divergent_local_hybrid_hits": bool(
            getattr(scheduler.connector, "supports_divergent_local_hybrid_hits", False)
        ),
        "scheduler_block_size": scheduler.block_size,
        "cache_config_block_size": scheduler.cache_config.block_size,
        "internal_prefill_checkpoints": scheduler.mamba_has_prefill_checkpoint_blocks,
        "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_num_active_reqs": scheduler.max_num_active_reqs,
        "long_prefill_token_threshold": config.scheduler_config.long_prefill_token_threshold,
        "hash_block_size": scheduler.hash_block_size,
        "num_prefill_lookahead": scheduler.num_prefill_lookahead,
        "num_reprefillable_tokens": coordinator.num_reprefillable_tokens,
        "prefill_timeline_enabled": os.environ.get("C48_PREFILL_TIMELINE") == "1",
        "eagle_group_ids": sorted(coordinator.eagle_group_ids),
        "simple_kv_offload_manager_sha256": hashlib.sha256(
            Path(simple_kv_offload_manager.__file__).read_bytes()
        ).hexdigest(),
        "groups": [
            {"manager": type(m).__name__, "block_size": m.block_size,
             "shared_prefix_checkpoint": m.shared_prefix_checkpoint}
            for m in coordinator.single_type_managers
        ],
    }
    directory = Path(os.environ["C48_DIAG_DIR"]).parent / "cache-manifests"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{record["role"]}-scheduler.json').write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("[C48-CACHE] " + json.dumps(record), flush=True)
    if bool(requested) != bool(scheduler.mamba_shared_prefix_checkpoint):
        raise RuntimeError("Requested shared-prefix checkpoint gate did not resolve as expected")
    if os.environ.get("C48_PREFILL_TIMELINE") == "1" and record["role"] == "prefill":
        from pd_c48_timeline import Timeline
        scheduler._c48_timeline = Timeline(
            scheduler, Path(os.environ["C48_DIAG_DIR"]).parent / "prefill-timeline"
        ).install()
    return scheduler
