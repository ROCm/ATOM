"""Record cache gates; optionally observe the unchanged scheduler methods."""
import copy
import hashlib
import json
import os
from pathlib import Path

from vllm.v1.simple_kv_offload import manager as simple_kv_offload_manager


def create_scheduler(*args, **kwargs):
    config = kwargs.get("vllm_config") if "vllm_config" in kwargs else args[0]
    defaults = copy.copy(config.scheduler_config)
    defaults.scheduler_cls = None
    base = defaults.get_scheduler_cls()
    scheduler = base(*args, **kwargs)
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    requested = scheduler.cache_config.enable_mamba_shared_prefix_checkpoint
    record = {
        "role": os.environ["C48_DIAG_ROLE"],
        "scheduler_class": base.__module__ + "." + base.__name__,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "requested_shared_prefix_checkpoint": requested,
        "scheduler_shared_prefix_checkpoint": scheduler.mamba_shared_prefix_checkpoint,
        "manager_shared_prefix_checkpoint": manager.mamba_shared_prefix_checkpoint,
        "partial_hash_hits": coordinator.enable_partial_hash_hits,
        "mamba_partial_cache_hit": scheduler.mamba_partial_cache_hit,
        "scheduler_block_size": scheduler.block_size,
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
    if os.environ.get("C48_CACHE_LIFECYCLE") == "1":
        if record["role"] != "prefill":
            raise ValueError("C48 cache lifecycle evidence is Prefill-only")
        from pd_c48_cache_lifecycle import install_prefill_cache_probes
        install_prefill_cache_probes(
            scheduler, Path(os.environ["C48_DIAG_DIR"]).parent / "cache-lifecycle"
        )
    return scheduler
