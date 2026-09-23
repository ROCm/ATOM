# SPDX-License-Identifier: MIT
"""Engram host prefetch and flat staging; the runner owns committed history.

Host-side throughout. Page-locking a shard and voting on the outcome live here
because they are `cudart` and Gloo calls that a CPU-only machine can run and
test; hashing on the device and the gather that reads a registered shard are
`device.runtime.EngramUva`, which this module never imports -- the caller
supplies it, and a caller that supplies nothing gets the host path.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from atom.model_ops.engram.tables import HostEmbeddingTable
from atom.utils import CpuGpuBuffer, envs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngramRequest:
    """Immutable lookup snapshot, including the identity of tentative tokens.

    History contains compressed IDs/DEAD, oldest first. Position and generation
    are supplied by the request owner; this helper never commits model state.
    Token/history tuples prevent stale reuse after rollback or slot recycling.
    """

    request_id: int
    generation: int
    position: int
    token_ids: tuple[int, ...]
    history: tuple[int, ...]
    token_mask: tuple[bool, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "history", tuple(self.history))
        if self.token_mask is not None:
            object.__setattr__(self, "token_mask", tuple(self.token_mask))
            if len(self.token_mask) != len(self.token_ids):
                raise ValueError("Engram token mask must match token IDs")
        if self.position < 0 or self.generation < 0 or not self.token_ids:
            raise ValueError("Engram lookup needs tokens and nonnegative identity")


class EngramPrefetchCache:
    """Bounded lookup results keyed by the full request snapshot and layer."""

    def __init__(self, capacity: int = 4096):
        if capacity < 1:
            raise ValueError("Engram cache capacity must be positive")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[EngramRequest, int], torch.Tensor] = (
            OrderedDict()
        )

    def put(self, request: EngramRequest, layer_id: int, value: torch.Tensor):
        with self._lock:
            key = (request, layer_id)
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def take(self, request: EngramRequest, layer_id: int):
        with self._lock:
            return self._store.pop((request, layer_id), None)

    def drop(self, request_id: int):
        with self._lock:
            for key in [k for k in self._store if k[0].request_id == request_id]:
                del self._store[key]

    def __len__(self):
        with self._lock:
            return len(self._store)


class EngramPrefetcher:
    """One host worker, with the same lookup contract as synchronous fallback."""

    def __init__(
        self, hash_mapping, tables: dict[int, HostEmbeddingTable], cache_capacity=4096
    ):
        self._hash_mapping = hash_mapping
        self._tables = tables
        self.cache = EngramPrefetchCache(cache_capacity)
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engram-prefetch"
        )
        # The dtype the staging buffers hold. Gathering straight into it avoids
        # materializing the rows in float32 and then downcasting them on the way
        # into staging -- two passes over the widest form of the data, on the
        # host, on every step. `EngramHost` sets this from its own buffers.
        self.out_dtype = torch.float32
        self._lock = threading.Lock()
        self._pending: dict[EngramRequest, object] = {}
        self._inflight: Future | None = None

    @property
    def layer_ids(self):
        return self._hash_mapping.config.layer_ids

    def compute(self, requests):
        """Hash ragged chunks with history; gather all rows once per table."""
        if not requests:
            return {}
        compressed = [
            self._hash_mapping.compress_tokens(
                np.asarray([request.token_ids], dtype=np.int64),
                (
                    None
                    if request.token_mask is None
                    else np.asarray([request.token_mask])
                ),
            )
            for request in requests
        ]
        results = {}
        for layer_id in self.layer_ids:
            rows = []
            for request, tokens in zip(requests, compressed):
                hashes = self._hash_mapping.hash_layer(
                    tokens,
                    layer_id,
                    compress=False,
                    history=np.asarray([request.history], dtype=np.int64),
                )
                rows.append(self._hash_mapping.to_row_indices(hashes, layer_id)[0])
            gathered = self._tables[layer_id].gather(
                np.concatenate(rows), out_dtype=self.out_dtype
            )
            offset = 0
            for request in requests:
                end = offset + len(request.token_ids)
                results[(request, layer_id)] = gathered[offset:end]
                offset = end
        return results

    def row_indices(self, requests):
        """The table rows `requests` name, per layer, flattened in request order.

        `compute` without the gather: the hashing is cheap host arithmetic, and
        the UVA path wants only these indices -- the rows themselves are read by
        the device kernel.
        """
        compressed = [
            self._hash_mapping.compress_tokens(
                np.asarray([request.token_ids], dtype=np.int64),
                (
                    None
                    if request.token_mask is None
                    else np.asarray([request.token_mask])
                ),
            )
            for request in requests
        ]
        out = {}
        for layer_id in self.layer_ids:
            rows = []
            for request, tokens in zip(requests, compressed):
                hashes = self._hash_mapping.hash_layer(
                    tokens,
                    layer_id,
                    compress=False,
                    history=np.asarray([request.history], dtype=np.int64),
                )
                rows.append(self._hash_mapping.to_row_indices(hashes, layer_id)[0])
            # [tokens, num_hash_heads]: the UVA path hands the whole matrix to
            # every rank, which keeps the heads a rank does not own addressable
            # without an index exchange.
            out[layer_id] = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
        return out

    def submit_compute(self, requests):
        requests = tuple(requests)
        ticket = object()
        with self._lock:
            self._pending.update((request, ticket) for request in requests)

        def run():
            try:
                results = self.compute(requests)
                with self._lock:
                    for (request, layer), value in results.items():
                        if self._pending.get(request) is ticket:
                            self.cache.put(request, layer, value)
            finally:
                with self._lock:
                    for request in requests:
                        if self._pending.get(request) is ticket:
                            del self._pending[request]

        self._inflight = self._pool.submit(run)
        return self._inflight

    def wait(self, timeout=None):
        if self._inflight is None:
            return True
        try:
            self._inflight.result(timeout=timeout)
            return True
        except TimeoutError:
            return False

    def drop_requests(self, request_ids):
        request_ids = set(request_ids)
        with self._lock:
            for request in list(self._pending):
                if request.request_id in request_ids:
                    del self._pending[request]
            for request_id in request_ids:
                self.cache.drop(request_id)

    def shutdown(self):
        with self._lock:
            self._pending.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


class EngramHost:
    """Stage every token of ragged request chunks into a flat device buffer.

    Prefetch misses use the identical immutable snapshots inline. The runner
    supplies host token IDs and committed history; sampling/D2H and history
    commit belong to the runner's later request-state integration.
    """

    def __init__(
        self,
        prefetcher,
        max_num_tokens,
        num_hash_heads,
        head_dim,
        device,
        dtype=torch.bfloat16,
        device_lookup=None,
    ):
        self.prefetcher = prefetcher
        # Gather straight into the staging dtype rather than float32-then-downcast.
        prefetcher.out_dtype = dtype
        # Optional: let a device kernel read the tables over UVA instead of
        # gathering them on the host (ATOM_ENGRAM_UVA). All-or-nothing -- a
        # partially registered set would silently keep the host path for some
        # layers, which is the confusing half-state to avoid.
        #
        # `device_lookup` is the class that does the reading, supplied rather
        # than imported: it reaches Triton and this module must not. No class,
        # no UVA -- which is also what a CPU-only test wants to say.
        self.uva = False
        self._tp_group = None
        # Every table this rank page-locked, for the fallback and the shutdown.
        self._pinned_tables = []
        if device_lookup is not None and device.type == "cuda" and envs.ATOM_ENGRAM_UVA:
            self.uva = self._enable_uva(prefetcher, num_hash_heads)
        self.max_num_tokens = max_num_tokens
        self.embed_width = num_hash_heads * head_dim
        self.device = device
        # Not pinned under UVA: nothing writes the host half there, and page
        # -locked memory is the scarce kind on a rank already holding tens of
        # GiB of table. It is still allocated -- `CpuGpuBuffer` owns both.
        self.buffers = {
            layer: CpuGpuBuffer(
                max_num_tokens,
                self.embed_width,
                dtype=dtype,
                device=device,
                pin_memory=device.type == "cuda" and not self.uva,
                with_numpy=False,
            )
            for layer in prefetcher.layer_ids
        }
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.copy_done = torch.cuda.Event() if self.copy_stream is not None else None
        self._copy_pending = False
        self._staged_rows = 0
        # After the buffers: the lookup reads them, and the pin_memory above
        # reads `self.uva`, so registration has to come first and allocation
        # between the two.
        self.lookup = device_lookup(self, num_hash_heads) if self.uva else None

    @property
    def layer_ids(self):
        return self.prefetcher.layer_ids

    # What the device lookup holds, asked of the host because both readers may
    # run on a rank that has no lookup -- `None` is the answer there, not an
    # attribute error. Typed nowhere here: the objects come from the injected
    # class, and naming them would be the import this module exists to avoid.
    @property
    def overlap(self):
        """The side-stream staging, when there is a device lookup running one."""
        return None if self.lookup is None else self.lookup.overlap

    @property
    def hash_tables(self):
        """The device n-gram hash tables, for callers that hash their own batch."""
        return None if self.lookup is None else self.lookup.hash_tables

    def _prepare_staging(self, rows, padded_rows):
        staged = rows if padded_rows is None else padded_rows
        if rows < 0 or staged < rows or staged > self.max_num_tokens:
            raise ValueError(
                f"{staged} padded rows / {rows} tokens exceeds staging capacity {self.max_num_tokens} or truncates tokens"
            )
        # The pinned source cannot be rewritten until its previous H2D finishes.
        if self._copy_pending:
            self.copy_done.synchronize()
            self._copy_pending = False
        return staged

    def _copy_to_device(self, rows):
        if self.copy_stream is None:
            for buffer in self.buffers.values():
                buffer.copy_to_gpu(rows)
        else:
            compute_stream = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(compute_stream)
                for buffer in self.buffers.values():
                    buffer.copy_to_gpu(rows)
                self.copy_done.record(self.copy_stream)
            self._copy_pending = True
        self._staged_rows = rows
        return rows

    def stage_embeddings(self, requests, padded_rows=None, batch=None):
        requests = tuple(requests)
        # With a device batch the requests are not built at all, so the row
        # count comes off the batch itself rather than off their token tuples.
        rows = (
            batch.batch_ids.numel()
            if batch is not None
            else sum(len(request.token_ids) for request in requests)
        )
        staged = self._prepare_staging(rows, padded_rows)
        # Before the prefetch cache, not after: what that cache holds is
        # embedding rows, which the UVA path never stages from the host. Taking
        # them here only threw them away -- and evicted them on the way.
        if self.lookup is not None:
            return self.lookup.stage(requests, rows, staged, batch)
        values = {}
        missing = []
        for request in requests:
            cached = {
                layer: self.prefetcher.cache.take(request, layer)
                for layer in self.layer_ids
            }
            if any(value is None for value in cached.values()):
                missing.append(request)
            else:
                values.update(
                    ((request, layer), value) for layer, value in cached.items()
                )
        values.update(self.prefetcher.compute(missing))
        for layer, buffer in self.buffers.items():
            offset = 0
            for request in requests:
                end = offset + len(request.token_ids)
                buffer.cpu[offset:end].copy_(
                    values[(request, layer)].reshape(end - offset, self.embed_width)
                )
                offset = end
            buffer.cpu[rows:staged].zero_()
        return self._copy_to_device(staged)

    def _enable_uva(self, prefetcher, num_hash_heads):
        """Page-lock this rank's head shard of every table; all-or-nothing.

        Sharding is by hash HEAD. Each head owns a disjoint, contiguous row range
        (the mapping's head offsets are their running sum), so a whole number of
        heads is a contiguous row -- and byte -- range. A rank registers only that
        range: the full table on every rank is what a TP job cannot afford.

        The answer is the GROUP's. An empty shard and a refused registration are
        both per-rank outcomes, and the device lookup ends in an all-gather: a
        rank that fell back alone would strand every other rank in it.
        """
        from aiter.dist.parallel_state import get_tp_group

        group = get_tp_group()
        shards = group.world_size
        if shards > num_hash_heads:
            logger.info(
                "engram: UVA lookup needs at most one shard per hash head "
                "(%d shards, %d heads); using the host path",
                shards,
                num_hash_heads,
            )
            return False
        self._tp_group = group if shards > 1 else None
        per = -(-num_hash_heads // shards)
        self.head_start = group.rank_in_group * per
        self.local_heads = min(per, max(0, num_hash_heads - self.head_start))
        self.total_heads = num_hash_heads
        pinned = self._register_shard(prefetcher)
        if shards > 1:
            # Gloo, once, at load. MIN: one refusal is the group's answer.
            vote = torch.tensor([bool(pinned)], dtype=torch.int32)
            dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group.cpu_group)
            if not int(vote):
                if pinned:
                    logger.info("engram: a peer has no UVA shard")
                pinned = 0
        if not pinned:
            logger.info("engram: using the host path")
            self._release_shard()
            return False
        logger.info(
            "engram: UVA device lookup active -- heads [%d, %d) of %d, "
            "%.1f GiB page-locked on this rank",
            self.head_start,
            self.head_start + self.local_heads,
            num_hash_heads,
            pinned / 1024**3,
        )
        return True

    def _register_shard(self, prefetcher):
        """Bytes page-locked for this rank's head range, or 0 if any table said no.

        A partial set is left for `_enable_uva` to release: it releases on every
        other falsy outcome too, so there is one site that gives pages back.
        """
        if self.local_heads <= 0:
            logger.info("engram: UVA shard is empty on this rank")
            return 0
        mapping = prefetcher._hash_mapping
        last = self.head_start + self.local_heads - 1
        pinned = 0
        for layer_id, table in prefetcher._tables.items():
            offsets = mapping.head_offsets[layer_id]
            row_start = int(offsets[self.head_start])
            row_end = int(offsets[last]) + int(mapping.head_vocab_sizes[layer_id][last])
            if not table.enable_uva(row_start, row_end):
                logger.info("engram: UVA registration refused on layer %d", layer_id)
                return 0
            self._pinned_tables.append(table)
            pinned += (row_end - row_start) * table.head_dim
        return pinned

    def _release_shard(self):
        """Unregister every table this rank pinned; these pages are unswappable."""
        for table in self._pinned_tables:
            table.disable_uva()
        self._pinned_tables = []

    def mark_staged(self, staged):
        """Record rows the device lookup filled in place, with no H2D to await."""
        self._staged_rows = staged
        self._copy_pending = False
        return staged

    def stage_dummy(self, num_rows):
        """Zeros over the rows a synthetic forward will read.

        The UVA path zeroes where the rows are read instead of zeroing a host
        copy and shipping it: same bytes, no H2D, and it leaves the host half
        of these buffers untouched -- which is what lets them go unpinned.

        On the device for the same reason the UVA staging is: `prepare_model_inputs`
        runs before the capture region opens (`build_for_cudagraph_capture` is
        called, and only then is the graph captured), so this memset is issued
        outside it and is not recorded into any graph.
        """
        rows = self._prepare_staging(num_rows, None)
        if self.uva:
            for buffer in self.buffers.values():
                buffer.gpu[:rows].zero_()
            return self.mark_staged(rows)
        for buffer in self.buffers.values():
            buffer.cpu[:rows].zero_()
        return self._copy_to_device(rows)

    def wait_for_embeddings(self):
        if self.copy_done is not None and self._copy_pending:
            torch.cuda.current_stream(self.device).wait_event(self.copy_done)

    def embeddings(self, layer_id):
        return self.buffers[layer_id].gpu[: self._staged_rows]

    def prefetch(self, requests):
        return self.prefetcher.submit_compute(requests)

    def drop_requests(self, request_ids):
        self.prefetcher.drop_requests(request_ids)

    def shutdown(self):
        if self._copy_pending:
            self.copy_done.synchronize()
        # Ahead of the mappings these pages belong to, which is the order
        # `from_checkpoint`'s ExitStack unwinds in.
        self._release_shard()
        self.prefetcher.shutdown()
