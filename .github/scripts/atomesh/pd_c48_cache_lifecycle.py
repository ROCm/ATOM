"""Bounded GPU/CPU cache metadata probe, prepared locally and not yet deployed.

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
        max_records=2048,
        cache_kind="cpu",
        result_summary=None,
    ):
        self.offload = offload
        self.pool = offload.cpu_block_pool
        self.stream = stream
        self.min_prompt = min_prompt
        self.max_history = max_history
        self.max_records = max_records
        self.cache_kind = cache_kind
        self.result_summary = result_summary or (lambda result: result)
        self.history = OrderedDict()
        self.history_dropped = 0
        self.records = 0
        self.io_error = None
        self.context = None
        self.removal_reason = "mapping_update"
        self.reset_count = 0
        self.published = self.removed = 0
        self._installed_map = None
        self.history_started_empty = False
        self.installed = False

    def _remember(self, key, event, block_id):
        value = self.history.pop(key, {})
        value[event] = {"unix_ns": time.time_ns(), "block_id": block_id}
        if event == "last_remove":
            value[event]["reason"] = self.removal_reason
        self.history[key] = value
        if len(self.history) > self.max_history:
            self.history.popitem(last=False)
            self.history_dropped += 1

    def _query(self, key, found):
        if self.context is None:
            return
        group = str(int.from_bytes(key[-4:], "big"))
        index = self.context["_hash_indices"].get(key[:-4])
        summary = self.context["groups"].setdefault(
            group, {"queries": 0, "hits": 0, "misses": 0}
        )
        summary["queries"] += 1
        if found is not None:
            summary["hits"] += 1
            if index is not None:
                summary["max_hit_hash_index"] = max(
                    index, summary.get("max_hit_hash_index", -1)
                )
            return
        summary["misses"] += 1
        previous = summary.get("lowest_missing_hash")
        if previous is None or (
            index is not None
            and (previous["hash_index"] is None or index < previous["hash_index"])
        ):
            history = self.history.get(key)
            summary["lowest_missing_hash"] = {
                "hash_index": index,
                "hash_with_group_hex": key.hex(),
                "observed_history": dict(history) if history is not None else None,
                "history_absence_is_conclusive": self.history_started_empty
                and self.history_dropped == 0,
                "note": "A map miss, not a tensor validity or token equality check.",
            }

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
                self._install_map()
                self._emit({"event": "cache_reset", "reset_count": self.reset_count})
            return result

        def lookup(request, num_computed_tokens):
            observe = (
                request.num_prompt_tokens >= self.min_prompt
                and self.records < self.max_records
                and self.io_error is None
            )
            if not observe:
                return original_lookup(request, num_computed_tokens)
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
                map_publications=self.published,
                map_removals=self.removed,
            )
            self._emit(row)
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
                "max_records": self.max_records,
                "history_start": "Installation; not earlier cache events.",
                "scope": "Host-side cache metadata; no GPU or tensor operations.",
            }
        )
        return self


def install_gpu_cache_probe(scheduler, stream, **limits):
    """Observe the original local-prefix method without changing its result."""
    from types import SimpleNamespace

    original = scheduler._get_local_prefix_cache_hit
    adapter = SimpleNamespace(
        cpu_block_pool=scheduler.kv_cache_manager.coordinator.block_pool,
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
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"cache-lifecycle-{os.getpid()}"
    stream = (directory / f"{prefix}.jsonl").open("w", buffering=65536)
    gpu = scheduler.kv_cache_manager.coordinator
    manifest = {
        "pid": os.getpid(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "min_prompt": limits.get("min_prompt", 200000),
        "max_history_per_pool": limits.get("max_history", 100000),
        "max_records_per_pool": limits.get("max_records", 2048),
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
    except BaseException:
        stream.close()
        raise

    def close():
        if stream.closed:
            return
        try:
            # Outside the per-lookup record cap so exhausted coverage is visible.
            for probe in probes:
                stream.write(
                    json.dumps(
                        {
                            "event": "probe_closed",
                            "cache_kind": probe.cache_kind,
                            "records": probe.records,
                            "record_limit_reached": probe.records >= probe.max_records,
                            "history_dropped": probe.history_dropped,
                            "retained_history_keys": len(probe.history),
                            "map_publications": probe.published,
                            "map_removals": probe.removed,
                            "io_error": probe.io_error,
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        finally:
            stream.close()

    original_shutdown = scheduler.shutdown

    def shutdown(*args, **kwargs):
        try:
            return original_shutdown(*args, **kwargs)
        finally:
            close()

    scheduler.shutdown = shutdown
    atexit.register(close)
    scheduler._c48_cache_lifecycle = probes, close
    return scheduler._c48_cache_lifecycle
