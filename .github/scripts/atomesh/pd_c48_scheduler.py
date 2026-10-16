"""Record resolved cache gates once; inherit all scheduling behavior."""
import copy
import json
import os
from pathlib import Path



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
        "hash_block_size": scheduler.hash_block_size,
        "num_prefill_lookahead": scheduler.num_prefill_lookahead,
        "num_reprefillable_tokens": coordinator.num_reprefillable_tokens,
        "eagle_group_ids": sorted(coordinator.eagle_group_ids),
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
    return scheduler
