"""Candidate v2 metadata probe; local only, not deployed to the active run.

Observe actual map queries and publication/removal events without additional
cache lookups, token payloads, tensors, GPU calls, or changes to LRU order.
"""

import json
import time
from collections import OrderedDict
from contextlib import contextmanager


class CacheLifecycleProbe:
    def __init__(
        self,
        offload,
        stream,
        *,
        min_prompt=200000,
        max_history=100000,
        max_watched=4096,
        max_records=16384,
        max_positive_miss_examples=8,
        watch_published_group_ids=None,
        cache_kind="cpu",
        result_summary=None,
        lookup_mode="all",
        max_pending_requests=2048,
    ):
        if lookup_mode not in ("all", "first_last"):
            raise ValueError("Unknown cache lookup observation mode")
        if max_pending_requests <= 0:
            raise ValueError("Pending observation limit must be positive")
        self.offload = offload
        self.pool = offload.cpu_block_pool
        self.stream = stream
        self.min_prompt = min_prompt
        self.max_history = max_history
        self.max_watched = max_watched
        self.max_records = max_records
        self.max_positive_miss_examples = max_positive_miss_examples
        if watch_published_group_ids is None:
            coordinator = getattr(offload, "cpu_coordinator", None)
            watch_published_group_ids = _mamba_group_ids(coordinator)
        self.watch_published_group_ids = set(watch_published_group_ids)
        self.cache_kind = cache_kind
        self.result_summary = result_summary or (lambda result: result)
        self.history = OrderedDict()
        self.history_dropped = 0
        self.watched = {}
        self.watch_limit_misses = 0
        self.records = 0
        self.io_error = None
        self.context = None
        self.removal_reason = "mapping_update"
        self.reset_count = 0
        self.published = self.removed = 0
        self._installed_map = None
        self.history_started_empty = False
        self.installed = False
        self.lookup_mode = lookup_mode
        self.max_pending_requests = max_pending_requests
        self.pending_lookups = {}
        self.qualifying_lookup_calls = 0
        self.observed_lookup_calls = 0
        self.coverage_dropped_lookup_calls = 0
        self.pending_limit_dropped_lookup_calls = 0
        self.observed_request_instances = 0
        self.peak_pending_requests = 0
        self.coalesced_lookup_rows = 0
        self.control_records_dropped = 0

    def _can_observe(self, request_id):
        if self.io_error is not None:
            return False
        if self.lookup_mode == "all":
            return self.records < self.max_records
        if request_id in self.pending_lookups:
            return True
        if len(self.pending_lookups) >= self.max_pending_requests:
            self.pending_limit_dropped_lookup_calls += 1
            return False
        # Reserve one last-lookup slot for every pending request, plus both
        # first/last slots for a new request. Retries consume no extra slots.
        return self.records + len(self.pending_lookups) + 2 <= self.max_records

    def _emit_lookup(self, row):
        if self.lookup_mode == "all":
            self._emit(row)
            return
        request_id = row["request_id"]
        state = self.pending_lookups.get(request_id)
        if state is None:
            state = {"count": 1, "last": None}
            self.pending_lookups[request_id] = state
            self.observed_request_instances += 1
            self.peak_pending_requests = max(
                self.peak_pending_requests, len(self.pending_lookups)
            )
            row.update(lookup_sequence=1, lookup_capture="first")
            self._emit(row)
        else:
            state["count"] += 1
            if state["last"] is not None:
                self.coalesced_lookup_rows += 1
            row.update(lookup_sequence=state["count"], lookup_capture="last")
            state["last"] = row

    def finish_request(self, request_id, reason="request_finished"):
        state = self.pending_lookups.pop(request_id, None)
        if state is not None and state["last"] is not None:
            row = state["last"]
            row["lookup_flush_reason"] = reason
            self._emit(row)

    def flush_pending(self):
        for request_id in list(self.pending_lookups):
            self.finish_request(request_id, reason="probe_close")

    def lookup_coverage(self):
        return {
            "lookup_mode": self.lookup_mode,
            "qualifying_lookup_calls": self.qualifying_lookup_calls,
            "observed_lookup_calls": self.observed_lookup_calls,
            "coverage_dropped_lookup_calls": self.coverage_dropped_lookup_calls,
            "pending_limit_dropped_lookup_calls": self.pending_limit_dropped_lookup_calls,
            "observed_request_instances": self.observed_request_instances,
            "pending_requests": len(self.pending_lookups),
            "peak_pending_requests": self.peak_pending_requests,
            "coalesced_intermediate_lookup_rows": self.coalesced_lookup_rows,
            "control_records_dropped": self.control_records_dropped,
        }

    def _remember(self, key, event, block_id):
        watched = self.watched.get(key)
        value = self.history.pop(key, watched["events"] if watched else {})
        value[event] = {"unix_ns": time.time_ns(), "block_id": block_id}
        if event == "last_remove":
            value[event]["reason"] = self.removal_reason
        self.history[key] = value
        if watched is not None:
            watched["events"] = value
        if len(self.history) > self.max_history:
            self.history.popitem(last=False)
            self.history_dropped += 1
        if (
            event == "last_publish"
            and int.from_bytes(key[-4:], "big") in self.watch_published_group_ids
        ):
            self._watch(key, reason="state_publication")

    def _watch(self, key, *, reason="actual_query"):
        if key in self.watched:
            return
        if len(self.watched) >= self.max_watched:
            self.watch_limit_misses += 1
            return
        self.watched[key] = {
            "reason": reason,
            "events": self.history.get(key, {}),
            "complete_from_pool_start": self.history_started_empty
            and self.history_dropped == 0,
        }

    def _query(self, key, found):
        if self.context is None:
            return
        group = str(int.from_bytes(key[-4:], "big"))
        index = self.context["_hash_indices"].get(key[:-4])
        summary = self.context["groups"].setdefault(
            group, {"queries": 0, "hits": 0, "misses": 0}
        )
        if summary["queries"] == 0:
            # Preserve the actual first queried key in each group across a
            # long idle gap, even after the rolling history has turned over.
            self._watch(key)
        summary["queries"] += 1
        if found is not None:
            summary["hits"] += 1
            if index is not None:
                summary["max_hit_hash_index"] = max(
                    index, summary.get("max_hit_hash_index", -1)
                )
            return
        summary["misses"] += 1
        watched = self.watched.get(key)
        history = watched["events"] if watched else self.history.get(key)

        def describe_missing():
            return {
                "hash_index": index,
                "hash_with_group_hex": key.hex(),
                "observed_history": dict(history) if history else None,
                "history_source": "watched_" + watched["reason"]
                if watched
                else "rolling",
                "history_absence_is_conclusive": (
                    watched["complete_from_pool_start"]
                    if watched
                    else self.history_started_empty and self.history_dropped == 0
                ),
                "note": "A map miss, not a tensor validity or token equality check.",
            }

        # Backward state scans can pass an evicted checkpoint and end at an
        # earlier never-published hash. Keep bounded positive evidence apart
        # from the lowest miss, using only history of actual map queries.
        if history:
            summary["positive_history_miss_queries"] = (
                summary.get("positive_history_miss_queries", 0) + 1
            )
            examples = summary.setdefault("positive_miss_examples", [])
            if len(examples) < self.max_positive_miss_examples:
                examples.append(describe_missing())
            else:
                summary["positive_miss_examples_dropped"] = (
                    summary.get("positive_miss_examples_dropped", 0) + 1
                )
        previous = summary.get("lowest_missing_hash")
        if previous is None or (
            index is not None
            and (previous["hash_index"] is None or index < previous["hash_index"])
        ):
            summary["lowest_missing_hash"] = describe_missing()

    def _install_map(self):
        mapping = self.pool.cached_block_hash_to_block
        if mapping is self._installed_map:
            return
        self._installed_map = mapping
        self.history_started_empty = len(mapping) == 0
        original_insert, original_pop = mapping.insert, mapping.pop
        original_get = mapping.get_one_block

        def insert(key, block):
            result = original_insert(key, block)
            self.published += 1
            self._remember(key, "last_publish", block.block_id)
            return result

        def pop(key, block_id):
            result = original_pop(key, block_id)
            if result is not None:
                self.removed += 1
                self._remember(key, "last_remove", block_id)
            return result

        def get_one_block(key):
            result = original_get(key)
            self._query(key, result)
            return result

        mapping.insert, mapping.pop, mapping.get_one_block = insert, pop, get_one_block

    @contextmanager
    def _reason(self, reason):
        previous = self.removal_reason
        self.removal_reason = reason
        try:
            yield
        finally:
            self.removal_reason = previous

    def _emit(self, row):
        if self.records >= self.max_records or self.io_error is not None:
            return
        if (
            self.lookup_mode == "first_last"
            and row["event"] != "prefix_lookup"
            and self.records + len(self.pending_lookups) >= self.max_records
        ):
            self.control_records_dropped += 1
            return
        try:
            row["cache_kind"] = self.cache_kind
            self.stream.write(json.dumps(row) + "\n")
            self.stream.flush()
            self.records += 1
        except OSError as error:
            self.io_error = type(error).__name__

    def install(self):
        if self.installed:
            return self
        self.installed = True
        self._install_map()
        original_alloc = self.pool.get_new_blocks
        original_evict = self.pool.evict_blocks
        original_reset = self.pool.reset_prefix_cache
        original_lookup = self.offload.get_num_new_matched_tokens

        def allocate(*args, **kwargs):
            with self._reason("allocation_reuse"):
                return original_alloc(*args, **kwargs)

        def evict(*args, **kwargs):
            with self._reason("explicit_eviction"):
                return original_evict(*args, **kwargs)

        def reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            if result:
                self.reset_count += 1
                self.history.clear()
                self.watched.clear()
                self._install_map()
                self._emit({"event": "cache_reset", "reset_count": self.reset_count})
            return result

        def lookup(request, num_computed_tokens):
            if request.num_prompt_tokens < self.min_prompt:
                return original_lookup(request, num_computed_tokens)
            self.qualifying_lookup_calls += 1
            if not self._can_observe(request.request_id):
                self.coverage_dropped_lookup_calls += 1
                return original_lookup(request, num_computed_tokens)
            self.observed_lookup_calls += 1
            hashes = request.block_hashes
            hash_indices = {h: i for i, h in enumerate(hashes)}
            sampled_indices = sorted(
                {
                    i
                    for i in (0, 7, 63, 511, 4095, len(hashes) - 1)
                    if 0 <= i < len(hashes)
                }
            )
            row = {
                "event": "prefix_lookup",
                "unix_ns": time.time_ns(),
                "request_id": request.request_id,
                "prompt_tokens": request.num_prompt_tokens,
                "arrival_time": request.arrival_time,
                "local_tokens_presented_to_connector": num_computed_tokens,
                "skip_reading_prefix_cache": request.skip_reading_prefix_cache,
                "request_hash_count": len(hashes),
                "request_prefix_hash_samples": {
                    str(i): hashes[i].hex() for i in sampled_indices
                },
                "groups": {},
                "reset_count": self.reset_count,
                "_hash_indices": hash_indices,
            }
            previous = self.context
            self.context = row
            begin = time.monotonic_ns()
            try:
                result = original_lookup(request, num_computed_tokens)
            finally:
                self.context = previous
            row.pop("_hash_indices")
            row.update(
                lookup_result=self.result_summary(result),
                lookup_host_ms=(time.monotonic_ns() - begin) / 1e6,
                retained_history_keys=len(self.history),
                history_dropped=self.history_dropped,
                watched_keys=len(self.watched),
                watch_limit_misses=self.watch_limit_misses,
                map_publications=self.published,
                map_removals=self.removed,
            )
            self._emit_lookup(row)
            return result

        self.pool.get_new_blocks = allocate
        self.pool.evict_blocks = evict
        self.pool.reset_prefix_cache = reset
        self.offload.get_num_new_matched_tokens = lookup
        self._emit(
            {
                "event": "probe_installed",
                "unix_ns": time.time_ns(),
                "min_prompt": self.min_prompt,
                "max_history": self.max_history,
                "max_watched": self.max_watched,
                "max_records": self.max_records,
                "lookup_mode": self.lookup_mode,
                "max_pending_requests": self.max_pending_requests,
                "lookup_record_scope": "First and last actual lookup per request; intermediate attempts coalesced"
                if self.lookup_mode == "first_last" else "Each observed lookup",
                "max_positive_miss_examples_per_group": self.max_positive_miss_examples,
                "watch_published_group_ids": sorted(self.watch_published_group_ids),
                "history_start": "Installation; not earlier cache events.",
                "scope": "Host-side cache metadata; no GPU or tensor operations.",
            }
        )
        return self


def _mamba_group_ids(coordinator):
    if coordinator is None:
        return []
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    return [
        group_id
        for group_id, manager in enumerate(
            getattr(coordinator, "single_type_managers", ())
        )
        if isinstance(manager, MambaManager)
    ]


def install_gpu_cache_probe(scheduler, stream, **limits):
    """Observe the original local-prefix method without changing its result."""
    from types import SimpleNamespace

    original = scheduler._get_local_prefix_cache_hit
    adapter = SimpleNamespace(
        cpu_block_pool=scheduler.kv_cache_manager.coordinator.block_pool,
        cpu_coordinator=scheduler.kv_cache_manager.coordinator,
        get_num_new_matched_tokens=lambda request, unused_local: original(request),
    )
    probe = CacheLifecycleProbe(
        adapter,
        stream,
        cache_kind="gpu",
        result_summary=lambda result: {
            "local_tokens": result[1],
            "shared_prefix_boundary": result[2],
            "hit_diverged": result[3],
        },
        **limits,
    ).install()
    scheduler._get_local_prefix_cache_hit = lambda request: (
        adapter.get_num_new_matched_tokens(request, None)
    )
    return probe


def install_engine_shutdown_flush(engine_core_class, scheduler, close):
    """Persist this scheduler's metadata before executor resource teardown."""
    original = engine_core_class.shutdown

    def shutdown(engine, *args, **kwargs):
        if getattr(engine, "scheduler", None) is scheduler:
            close()
        return original(engine, *args, **kwargs)

    engine_core_class.shutdown = shutdown


def install_prefill_cache_probes(scheduler, directory, **limits):
    """Attach both observers to a constructed Prefill scheduler before warmup.

    Caller selects the Prefill role. A missing/ambiguous SimpleCPU connector
    fails installation instead of silently producing a one-tier cache trace.
    """
    import atexit
    import hashlib
    import os
    from pathlib import Path

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
        SimpleCPUOffloadConnector,
    )

    if getattr(scheduler, "_c48_cache_lifecycle", None) is not None:
        return scheduler._c48_cache_lifecycle
    pending = [scheduler.connector]
    managers = []
    while pending:
        connector = pending.pop()
        if isinstance(connector, MultiConnector):
            pending.extend(connector.sub_connectors)
        elif isinstance(connector, SimpleCPUOffloadConnector):
            managers.append(connector.scheduler_manager)
    if len(managers) != 1 or managers[0] is None:
        raise ValueError("Cache probe requires exactly one SimpleCPU scheduler manager")
    cpu = managers[0]
    physical_enabled = os.environ.get("C48_PHYSICAL_CACHE_LIFECYCLE", "0")
    if physical_enabled not in ("0", "1") or (
        physical_enabled == "1" and os.environ.get("C48_DIAG_ROLE") != "prefill"
    ):
        raise ValueError("Physical cache evidence is Prefill-only and requires a 0/1 gate")
    physical_probes = {}
    inventory_enabled = os.environ.get("C48_PREFIX_INVENTORY", "0")
    if inventory_enabled not in ("0", "1") or (
        inventory_enabled == "1" and physical_enabled != "1"
    ):
        raise ValueError("Prefix inventory requires physical Prefill observation")
    inventories = []
    inventory_stream = None
    limits.setdefault("lookup_mode", os.environ.get("C48_CACHE_LOOKUP_MODE", "all"))
    physical_source_sha256 = None
    if physical_enabled == "1":
        import pd_c48_physical_lifecycle
        physical_source_sha256 = hashlib.sha256(
            Path(pd_c48_physical_lifecycle.__file__).read_bytes()
        ).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"cache-lifecycle-{os.getpid()}"
    stream = (directory / f"{prefix}.jsonl").open("w", buffering=65536)
    gpu = scheduler.kv_cache_manager.coordinator
    manifest = {
        "probe_schema_version": 3,
        "lookup_mode": limits["lookup_mode"],
        "max_pending_requests_per_pool": limits.get("max_pending_requests", 2048),
        "physical_last_events_enabled": physical_enabled == "1",
        "physical_source_sha256": physical_source_sha256,
        "physical_history_scope": "last events, not complete access history",
        "pid": os.getpid(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "min_prompt": limits.get("min_prompt", 200000),
        "max_history_per_pool": limits.get("max_history", 100000),
        "max_watched_keys_per_pool": limits.get("max_watched", 4096),
        "max_records_per_pool": limits.get("max_records", 16384),
        "max_positive_miss_examples_per_group": limits.get(
            "max_positive_miss_examples", 8
        ),
        "lazy_offload": cpu._lazy_mode,
        "cpu_blocks": cpu.num_cpu_blocks,
        "gpu_blocks": len(gpu.block_pool.blocks),
        "cpu_full_attention_group": cpu.fa_gidx,
        "cpu_prefix_cacheable_groups": list(cpu.prefix_cacheable_group_ids),
        "scheduler_block_size": scheduler.block_size,
        "hash_block_size": scheduler.hash_block_size,
        "groups": [
            {
                "group_id": group_id,
                "gpu_manager": type(manager).__name__,
                "gpu_block_size": manager.block_size,
                "cpu_block_size": cpu.group_block_sizes[group_id],
                "eagle": group_id in gpu.eagle_group_ids,
            }
            for group_id, manager in enumerate(gpu.single_type_managers)
        ],
    }
    try:
        (directory / f"{prefix}-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        probes = (
            install_gpu_cache_probe(scheduler, stream, **limits),
            CacheLifecycleProbe(cpu, stream, **limits).install(),
        )
        if physical_enabled == "1":
            for probe in probes:
                physical = pd_c48_physical_lifecycle.PhysicalLifecycle(probe.pool).install()
                physical.attach_to_map_probe(probe)
                if probe.cache_kind == "cpu":
                    physical.observe_cpu_refresh(cpu)
                physical_probes[probe.cache_kind] = physical
            manifest["physical_installation"] = {
                name: probe.coverage() for name, probe in physical_probes.items()
            }
            (directory / f"{prefix}-manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        if inventory_enabled == "1":
            import pd_c48_prefix_inventory
            inventory_dir = directory.parent / "prefix-inventory"
            inventory_dir.mkdir(parents=True, exist_ok=True)
            inventory_stream = (inventory_dir / f"prefix-inventory-{os.getpid()}.jsonl").open(
                "x", buffering=65536
            )
            for probe in probes:
                inventories.append(pd_c48_prefix_inventory.PrefixInventory(
                    probe, physical_probes[probe.cache_kind], inventory_stream,
                    group_ids=cpu.prefix_cacheable_group_ids,
                ).install())
            manifest["prefix_inventory"] = {
                "enabled": True,
                "source_sha256": hashlib.sha256(
                    Path(pd_c48_prefix_inventory.__file__).read_bytes()
                ).hexdigest(),
                "initial_coverage": [item.coverage() for item in inventories],
            }
            (directory / f"{prefix}-manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    except BaseException:
        stream.close()
        if inventory_stream is not None:
            inventory_stream.close()
        raise

    def close():
        if stream.closed:
            return
        try:
            if inventory_stream is not None:
                try:
                    for inventory in inventories:
                        inventory_stream.write(json.dumps({
                            "event": "prefix_inventory_closed",
                            "cache_kind": inventory.probe.cache_kind,
                            **inventory.coverage(),
                        }) + "\n")
                    inventory_stream.flush()
                except OSError:
                    pass
            # Outside the per-lookup record cap so exhausted coverage is visible.
            for probe in probes:
                probe.flush_pending()
                stream.write(
                    json.dumps(
                        {
                            "event": "probe_closed",
                            "cache_kind": probe.cache_kind,
                            "records": probe.records,
                            "record_limit_reached": probe.records >= probe.max_records,
                            "history_dropped": probe.history_dropped,
                            "retained_history_keys": len(probe.history),
                            "watched_keys": len(probe.watched),
                            "watch_limit_misses": probe.watch_limit_misses,
                            "map_publications": probe.published,
                            "map_removals": probe.removed,
                            "io_error": probe.io_error,
                            "lookup_coverage": probe.lookup_coverage(),
                            "physical_last_events": (
                                physical_probes[probe.cache_kind].coverage()
                                if probe.cache_kind in physical_probes else None
                            ),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        finally:
            stream.close()
            if inventory_stream is not None:
                inventory_stream.close()

    original_shutdown = scheduler.shutdown

    def shutdown(*args, **kwargs):
        try:
            return original_shutdown(*args, **kwargs)
        finally:
            close()

    scheduler.shutdown = shutdown
    original_free_request = scheduler._free_request

    def free_request(request, *args, **kwargs):
        try:
            return original_free_request(request, *args, **kwargs)
        finally:
            for probe in probes:
                probe.finish_request(request.request_id)

    scheduler._free_request = free_request
    # EngineCore tears down the executor before invoking scheduler.shutdown.
    # Flush at its entry so slow/interrupted executor teardown does not lose
    # the observer's final coverage counters. The original teardown is intact.
    from vllm.v1.engine.core import EngineCore

    install_engine_shutdown_flush(EngineCore, scheduler, close)
    atexit.register(close)
    scheduler._c48_cache_lifecycle = probes, close
    return scheduler._c48_cache_lifecycle
