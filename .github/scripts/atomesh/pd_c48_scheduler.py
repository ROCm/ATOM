"""Record cache gates and enable the isolated Prefill scheduling experiment."""
import copy
import hashlib
import json
import os
from pathlib import Path

from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.simple_kv_offload import manager as simple_kv_offload_manager


def create_scheduler(*args, **kwargs):
    role = os.environ["C48_DIAG_ROLE"]
    fairness = os.environ.get("C48_PREFILL_FAIRNESS", "0")
    if fairness not in ("0", "1") or (fairness == "1" and role != "prefill"):
        raise ValueError("C48 fairness requires Prefill role and a 0/1 gate")
    scheduler_sha = hashlib.sha256(Path(scheduler_module.__file__).read_bytes()).hexdigest()
    if scheduler_sha != "5e29800ef253c11ce4870eb0fe20cd4a35da3030d77c0724dd63911c0059467c":
        raise RuntimeError("C48 scheduler source differs from the pinned experiment")
    physical = os.environ.get("C48_PHYSICAL_CACHE_LIFECYCLE", "0")
    if physical not in ("0", "1") or (
        physical == "1" and (role != "prefill" or os.environ.get("C48_CACHE_LIFECYCLE") != "1")
    ):
        raise ValueError("C48 physical evidence requires Prefill lifecycle observation")
    lookup_mode = os.environ.get("C48_CACHE_LOOKUP_MODE", "all")
    inventory = os.environ.get("C48_PREFIX_INVENTORY", "0")
    if inventory not in ("0", "1") or (inventory == "1" and physical != "1"):
        raise ValueError("C48 prefix inventory requires physical Prefill observation")
    if lookup_mode not in ("all", "first_last") or (
        lookup_mode == "first_last"
        and (role != "prefill" or os.environ.get("C48_CACHE_LIFECYCLE") != "1")
    ):
        raise ValueError("C48 lookup coverage requires Prefill lifecycle observation")
    config = kwargs.get("vllm_config") if "vllm_config" in kwargs else args[0]
    defaults = copy.copy(config.scheduler_config)
    defaults.scheduler_cls = None
    base = defaults.get_scheduler_cls()
    scheduler = base(*args, **kwargs)
    scheduler._c48_prefill_fairness = fairness == "1"
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    requested = scheduler.cache_config.enable_mamba_shared_prefix_checkpoint
    record = {
        "role": role,
        "prefill_fairness_enabled": scheduler._c48_prefill_fairness,
        "physical_cache_lifecycle_enabled": physical == "1",
        "cache_lookup_mode": lookup_mode,
        "prefix_inventory_enabled": inventory == "1",
        "scheduler_source_sha256": scheduler_sha,
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
        "prefill_request_ledger_enabled": os.environ.get("C48_PREFILL_REQUEST_LEDGER") == "1",
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
    if record["prefill_request_ledger_enabled"]:
        if record["role"] != "prefill":
            raise ValueError("C48 request ledger is Prefill-only")
        from pd_c48_request_ledger import install_request_ledger
        install_request_ledger(
            scheduler, Path(os.environ["C48_DIAG_DIR"]).parent / "prefill-request-ledger"
        )
    return scheduler
