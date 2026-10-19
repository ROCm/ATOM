"""Bounded, read-only prefix inventories for the unresolved C48 cache loss.

Snapshot before the ordinary lookup. The real matcher still makes all serving
decisions. Inventory reads run outside the lifecycle query-recording context,
so they neither move blocks nor count as matcher queries.
"""

import json
import time


class PrefixInventory:
    def __init__(
        self,
        probe,
        physical,
        stream,
        *,
        group_ids,
        prompt_ranges=((451600, 451750), (544000, 544200), (686300, 686450)),
        max_requests=16,
        max_hashes=8192,
    ):
        self.probe = probe
        self.physical = physical
        self.stream = stream
        self.group_ids = tuple(group_ids)
        self.prompt_ranges = tuple(prompt_ranges)
        self.max_requests = max_requests
        self.max_hashes = max_hashes
        self.seen = set()
        self.captured = 0
        self.cap_rejections = 0
        self.hash_rejections = 0
        self.errors = []
        self.total_ns = 0
        self.max_ns = 0
        self.installed = False

    def capture(self, request, num_local_tokens):
        if not any(
            lo <= request.num_prompt_tokens <= hi for lo, hi in self.prompt_ranges
        ):
            return
        if request.request_id in self.seen:
            return
        if len(self.seen) >= self.max_requests:
            self.cap_rejections += 1
            return
        self.seen.add(request.request_id)
        if len(request.block_hashes) > self.max_hashes:
            self.hash_rejections += 1
            return
        begin = time.monotonic_ns()
        try:
            self._capture(request, num_local_tokens)
        except Exception as error:  # noqa: BLE001 -- evidence failure must not alter serving
            if len(self.errors) < 16:
                self.errors.append(type(error).__name__)
        finally:
            duration = time.monotonic_ns() - begin
            self.total_ns += duration
            self.max_ns = max(self.max_ns, duration)

    def _capture(self, request, num_local_tokens):
        probe = self.probe
        if probe.context is not None:
            raise RuntimeError("Inventory must run outside actual matcher observation")
        mapping = probe.pool.cached_block_hash_to_block
        hashes = request.block_hashes
        row = {
            "event": "prefix_inventory",
            "cache_kind": probe.cache_kind,
            "unix_ns": time.time_ns(),
            "request_id": request.request_id,
            "prompt_tokens": request.num_prompt_tokens,
            "num_local_tokens": num_local_tokens,
            "skip_reading_prefix_cache": request.skip_reading_prefix_cache,
            "hashes": [h.hex() for h in hashes],
            "history_complete_from_installation": probe.history_started_empty
            and probe.history_dropped == 0
            and probe.reset_count == 0,
            "history_dropped": probe.history_dropped,
            "reset_count": probe.reset_count,
            "map_publications": probe.published,
            "map_removals": probe.removed,
            "free_blocks": probe.pool.get_num_free_blocks(),
            "groups": {},
            "physical_blocks": {},
            "scope": "All target hashes before real lookup; map metadata, not tensor validation",
        }
        for group_id in self.group_ids:
            suffix = group_id.to_bytes(4, "big")
            entries = []
            resident = 0
            for index, block_hash in enumerate(hashes):
                key = block_hash + suffix
                block = mapping.get_one_block(key)
                watched = probe.watched.get(key)
                history = watched["events"] if watched else probe.history.get(key)
                if block is None and not history:
                    continue
                entry = {"hash_index": index, "block_id": None, "history": history}
                if block is not None:
                    resident += 1
                    entry["block_id"] = block.block_id
                    block_key = str(block.block_id)
                    if block_key not in row["physical_blocks"]:
                        row["physical_blocks"][block_key] = self.physical.snapshot(
                            block.block_id
                        )
                entries.append(entry)
            row["groups"][str(group_id)] = {
                "examined_hashes": len(hashes),
                "resident_hashes": resident,
                "entries": entries,
                "omitted_entry_meaning": "Absent from map and retained history",
            }
        row["snapshot_end_unix_ns"] = time.time_ns()
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        self.captured += 1

    def install(self):
        if self.installed:
            return self
        original = self.probe.offload.get_num_new_matched_tokens

        def lookup(request, num_local_tokens):
            self.capture(request, num_local_tokens)
            return original(request, num_local_tokens)

        self.probe.offload.get_num_new_matched_tokens = lookup
        self.installed = True
        return self

    def coverage(self):
        return {
            "captured": self.captured,
            "seen": len(self.seen),
            "cap_rejections": self.cap_rejections,
            "hash_rejections": self.hash_rejections,
            "errors": self.errors,
            "total_host_ns": self.total_ns,
            "max_host_ns": self.max_ns,
            "max_requests": self.max_requests,
            "max_hashes": self.max_hashes,
            "prompt_ranges": self.prompt_ranges,
            "group_ids": self.group_ids,
        }
