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
    if os.environ.get("C48_DSPARK_JUNCTION") == "1" and os.environ["C48_DIAG_ROLE"] == "prefill":
        enable_dspark_shared_prefix_checkpoint(scheduler, config)
    if (os.environ.get("C48_EXPECT_SHARED_PREFIX_CHECKPOINT") == "1"
            and os.environ["C48_DIAG_ROLE"] == "prefill"
            and not (requested and scheduler.mamba_shared_prefix_checkpoint
                     and manager.mamba_shared_prefix_checkpoint)):
        raise RuntimeError("C72 junction candidate did not enable shared-prefix checkpoints")
    record = {
        "role": os.environ["C48_DIAG_ROLE"],
        "no_tail_split": no_tail_split,
        "tail_ckpt_hash": tail_ckpt_hash,
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
