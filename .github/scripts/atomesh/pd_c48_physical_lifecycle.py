"""Local-only evidence candidate: bounded last-event state for each physical block.

No deployment or GPU claim. Hook actual queue mutations and record summaries;
this does not reconstruct every historical touch or compare tensor contents.
"""

import time


class PhysicalLifecycle:
    def __init__(self, pool):
        self.pool = pool
        self.sequence = 0
        self.failures = []
        self.failure_count = 0
        self.installed_unix_ns = None
        self.initial_cache_empty = None
        self.queue = None
        self.installed = False
        self.state = {}
        self.context = None

    def record(self, callback, *args):
        try:
            callback(*args)
        except Exception as error:  # noqa: BLE001 - evidence must not alter cache execution
            self.failure_count += 1
            if len(self.failures) < 8:
                self.failures.append(type(error).__name__)

    def tick(self):
        self.sequence += 1
        return {"sequence": self.sequence, "unix_ns": time.time_ns()}

    def queued(self, blocks, operation):
        event = self.tick()
        for index, block in enumerate(blocks):
            state = self.state.setdefault(block.block_id, {"generation": 0})
            state["last_queue_entry"] = dict(
                event, operation=operation, batch_index=index
            )
            state["queued"] = True

    def popped(self, blocks, header):
        event = self.tick()
        for index, block in enumerate(blocks):
            state = self.state.setdefault(block.block_id, {"generation": 0})
            state["generation"] += 1
            state["last_allocation_pop"] = dict(
                event, **header, batch_index=index, ref_cnt=block.ref_cnt
            )
            state["queued"] = False

    def touched(self, blocks, event):
        for block in blocks:
            state = self.state.setdefault(block.block_id, {"generation": 0})
            state["last_touch"] = dict(event, ref_cnt_after=block.ref_cnt)
            if self.context is not None:
                state["last_touch"]["context"] = dict(self.context)
            if not block.is_null:
                state["queued"] = False

    def removed(self, block):
        state = self.state.setdefault(block.block_id, {"generation": 0})
        state["last_queue_remove"] = self.tick()
        state["queued"] = False

    def install(self):
        if self.installed:
            return self
        self.installed = True
        queue = self.pool.free_block_queue
        self.queue = queue
        self.installed_unix_ns = time.time_ns()
        self.initial_cache_empty = len(self.pool.cached_block_hash_to_block) == 0
        # A complete initial queue snapshot is taken once, before workload start.
        self.record(self.queued, queue.get_all_free_blocks(), "initial")
        for name in ("append_n", "prepend_n"):
            original = getattr(queue, name)

            def insertion(blocks, original=original, name=name):
                result = original(blocks)
                if blocks:
                    self.record(self.queued, blocks, name)
                return result

            setattr(queue, name, insertion)
        original_remove = queue.remove

        def remove(block):
            result = original_remove(block)
            self.record(self.removed, block)
            return result

        queue.remove = remove
        original_pop = queue.popleft_n

        def pop(count):
            head = queue.fake_free_list_head.next_free_block
            header = {
                "requested": count,
                "free_before": queue.num_free_blocks,
                "head_before": head.block_id if head is not None else None,
            }
            result = original_pop(count)
            self.record(self.popped, result, header)
            return result

        queue.popleft_n = pop
        original_touch = self.pool.touch

        def touch(blocks):
            # touch's contract is a Sequence, so observe the same objects after
            # the original call without consuming/reordering a lazy iterable.
            event = self.tick()
            result = original_touch(blocks)
            self.record(self.touched, blocks, event)
            return result

        self.pool.touch = touch
        return self

    def observe_cpu_refresh(self, offload):
        assert offload.cpu_block_pool is self.pool
        original = offload._refresh_cpu_prefix

        def refresh(request, num_local_tokens):
            previous = self.context
            self.context = {
                "operation": "cpu_prefix_refresh",
                "request_id": request.request_id,
                "num_local_tokens": num_local_tokens,
            }
            try:
                return original(request, num_local_tokens)
            finally:
                self.context = previous

        offload._refresh_cpu_prefix = refresh
        return self

    def snapshot(self, block_id):
        state = {}
        for key, value in self.state.get(block_id, {}).items():
            if isinstance(value, dict):
                value = value.copy()
                if "context" in value:
                    value["context"] = value["context"].copy()
            state[key] = value
        block = self.pool.blocks[block_id]
        state["ref_cnt_now"] = block.ref_cnt
        state["evidence_failures"] = list(self.failures)
        state["scope"] = "last physical events; not complete touch history"
        return state

    def attach_to_map_probe(self, probe):
        assert probe.pool is self.pool
        original = probe._remember

        def remember(key, event, block_id):
            original(key, event, block_id)

            def attach():
                probe.history[key][event]["physical"] = self.snapshot(block_id)

            self.record(attach)

        probe._remember = remember
        return self

    def coverage(self):
        return {
            "schema_version": 1,
            "installed_unix_ns": self.installed_unix_ns,
            "initial_cache_empty": self.initial_cache_empty,
            "same_queue_instance": self.queue is self.pool.free_block_queue,
            "pool_blocks": self.pool.num_gpu_blocks,
            "tracked_physical_blocks": len(self.state),
            "last_sequence": self.sequence,
            "observation_errors": self.failure_count,
            "error_examples": list(self.failures),
            "history_scope": "last physical events; not complete access history",
        }
