# SPDX-License-Identifier: MIT
"""Engram host prefetch and flat staging; the runner owns committed history."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import logging

import numpy as np
import torch
from atom.model_ops.engram_lookup import HostEmbeddingTable

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
    ):
        self.prefetcher = prefetcher
        # Gather straight into the staging dtype rather than float32-then-downcast.
        prefetcher.out_dtype = dtype
        # Optional: let a device kernel read the tables over UVA instead of
        # gathering them on the host (ATOM_ENGRAM_UVA). All-or-nothing -- a
        # partially registered set would silently keep the host path for some
        # layers, which is the confusing half-state to avoid.
        self.uva = False
        self._tp_group = None
        self._ids_staging = None
        if device.type == "cuda" and envs.ATOM_ENGRAM_UVA:
            self.uva = self._enable_uva(prefetcher, num_hash_heads)
        self.max_num_tokens = max_num_tokens
        self.embed_width = num_hash_heads * head_dim
        self.device = device
        self.buffers = {
            layer: CpuGpuBuffer(
                max_num_tokens,
                self.embed_width,
                dtype=dtype,
                device=device,
                pin_memory=device.type == "cuda",
                with_numpy=False,
            )
            for layer in prefetcher.layer_ids
        }
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.copy_done = torch.cuda.Event() if self.copy_stream is not None else None
        self._copy_pending = False
        self._staged_rows = 0

    @property
    def layer_ids(self):
        return self.prefetcher.layer_ids

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

    def stage_embeddings(self, requests, padded_rows=None):
        requests = tuple(requests)
        rows = sum(len(request.token_ids) for request in requests)
        staged = self._prepare_staging(rows, padded_rows)
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
        if self.uva:
            return self._stage_uva(requests, rows, staged)
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
        if self.local_heads <= 0:
            logger.info("engram: UVA shard is empty on this rank; using the host path")
            return False
        mapping = prefetcher._hash_mapping
        registered = 0
        done = []
        for layer_id, table in prefetcher._tables.items():
            offsets = mapping.head_offsets[layer_id]
            sizes = mapping.head_vocab_sizes[layer_id]
            end_head = self.head_start + self.local_heads
            row_start = int(offsets[self.head_start])
            row_end = int(offsets[end_head - 1]) + int(sizes[end_head - 1])
            if not table.enable_uva(row_start, row_end):
                # Give back what the earlier tables already pinned: these pages
                # are unswappable and the host path has no use for them.
                for pinned in done:
                    pinned.disable_uva()
                logger.info("engram: UVA registration failed; using the host path")
                return False
            done.append(table)
            registered += (row_end - row_start) * table.head_dim
        logger.info(
            "engram: UVA device lookup active -- heads [%d, %d) of %d, "
            "%.1f GiB page-locked on this rank",
            self.head_start,
            self.head_start + self.local_heads,
            num_hash_heads,
            registered / 1024**3,
        )
        return True

    def _stage_uva(self, requests, rows, staged):
        """Device lookup: hash on the host, gather and dequantize on the GPU.

        Only the row INDICES cross to the device (a few KB); the kernel reads the
        table rows out of page-locked host memory and dequantizes them there, so
        neither the gather nor the fp8 decode runs on the host and there is no
        embedding H2D.

        Each rank owns a slice of the hash heads and writes zeros for the rest, so
        the all-gather that reassembles the full width is a concatenation. The
        projection that consumes this is replicated, so every rank needs it whole.
        """
        per_layer = self.prefetcher.row_indices(requests)
        # Stage the indices through pinned memory: `from_numpy(...).to(device)`
        # copies from PAGEABLE memory, where `non_blocking` is silently ignored
        # and the driver stages through its own bounce buffer every step.
        if self._ids_staging is None:
            self._ids_staging = CpuGpuBuffer(
                self.max_num_tokens,
                self.total_heads,
                dtype=torch.int64,
                device=self.device,
                pin_memory=True,
            )
        for layer, buffer in self.buffers.items():
            self._ids_staging.np[:rows] = per_layer[layer]
            ids = self._ids_staging.copy_to_gpu(rows)
            table = self.prefetcher._tables[layer]
            if ids.shape != (rows, self.total_heads):
                raise RuntimeError(
                    f"engram UVA indices are {tuple(ids.shape)}, expected "
                    f"{(rows, self.total_heads)}"
                )
            # empty, not zeros: the kernel stores every row it is given, writing
            # zeros itself for the heads this rank does not own. Flat
            # `[tokens, local_heads * head_dim]`, which is the same bytes the
            # kernel writes and the layout the all-gather below wants.
            flat = torch.empty(
                rows,
                self.local_heads * table.head_dim,
                dtype=buffer.gpu.dtype,
                device=self.device,
            )
            table.gather_into(
                ids,
                flat.view(rows, self.local_heads, table.head_dim),
                head_start=self.head_start,
                local_heads=self.local_heads,
                total_heads=self.total_heads,
            )
            out = flat
            if self._tp_group is not None:
                # Gather on the LAST dim of a 2-D view, which is what routes this
                # through aiter's IPC all-gather instead of NCCL: the custom path
                # needs dim 0 or a 16-byte-aligned last dim, and a head slice is
                # `local_heads * head_dim * 2` bytes wide. NCCL is not just
                # slower here -- its end event, recorded during a CUDAGraph
                # capture, is later read by the watchdog and crashes with
                # hipErrorCapturedEvent (see moe.all_gather_with_padding).
                # Rank-major concatenation puts head `h` back at column
                # `h * head_dim`, so the result needs no transpose; the trailing
                # columns are the padding an indivisible head count leaves.
                out = self._tp_group.all_gather(out, use_custom=True, dim=1)
                out = out[:, : self.embed_width]
            buffer.gpu[:rows].copy_(out.reshape(rows, self.embed_width))
            if staged > rows:
                buffer.gpu[rows:staged].zero_()
        self._staged_rows = staged
        self._copy_pending = False
        return staged

    def stage_dummy(self, num_rows):
        rows = self._prepare_staging(num_rows, None)
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
        self.prefetcher.shutdown()


@dataclass(frozen=True)
class EngramInputs:
    embeddings: dict[int, torch.Tensor]
    histories: np.ndarray
    compressed_rows: tuple[np.ndarray, ...]


class EngramInputPreparer:
    """Prepare finalized runtime tokens using the cache's restored history.

    No request-history dictionary: the returned history commits with the model
    state, and generic STATE checkpoints carry it across migration and reuse.
    """

    def __init__(self, mapping, host, resources=None):
        self.mapping, self.host, self.resources = mapping, host, resources

    @classmethod
    def from_checkpoint(cls, directory, config, max_tokens, device):
        from contextlib import ExitStack

        from atom.model_loader.deepseek_v41 import engram_tables
        from atom.model_ops.engram import (
            CompressedTokenizer,
            EngramConfig,
            NgramHashMapping,
        )
        from transformers import AutoTokenizer

        resources = ExitStack()
        try:
            tables = resources.enter_context(engram_tables(directory, config))
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            engram_config = EngramConfig.from_hf(config.to_dict())
            mapping = NgramHashMapping(
                engram_config,
                CompressedTokenizer(
                    tokenizer, expected_size=engram_config.compressed_vocab_size
                ),
            )
            host = EngramHost(
                EngramPrefetcher(mapping, tables),
                max_tokens,
                engram_config.num_hash_heads,
                engram_config.head_dim,
                device,
            )
            resources.callback(host.shutdown)
            return cls(mapping, host, resources)
        except BaseException:
            resources.close()
            raise

    def prepare(
        self,
        spans,
        token_ids,
        histories,
        *,
        dummy=False,
        token_mask=None,
        padded_rows=None,
    ):
        """Stage one embedding row per row the forward will run.

        `token_ids` are the rows the requests own. `padded_rows` is the width
        the forward runs, which is wider whenever the batch was padded up to a
        captured shape -- the tail belongs to no request, so it is staged as
        zeros rather than looked up, and it cannot be read off `token_ids`
        because the padding is applied to the model's input after this.
        """
        compressed_rows = []
        rows = token_ids.numel() if padded_rows is None else padded_rows
        if dummy:
            self.host.stage_dummy(rows)
            next_histories = histories
        else:
            # These are the final GPU IDs, including deferred decode tokens.
            # One D2H per batch is the eager host-lookup contract; a subsequent
            # HBM provider can replace it without changing the model or scheduler.
            ids = token_ids.detach().cpu().numpy()
            requests, next_histories = [], []
            for span, history in zip(spans, histories):
                tokens = ids[span.token_slice]
                mask = None if token_mask is None else token_mask[span.token_slice]
                requests.append(
                    EngramRequest(
                        span.request_id,
                        0,
                        span.position,
                        tuple(tokens),
                        tuple(history),
                        token_mask=None if mask is None else tuple(mask),
                    )
                )
                compressed = self.mapping.compress_tokens(
                    tokens[None, :], None if mask is None else mask[None, :]
                )
                compressed_rows.append(compressed[0])
                next_histories.append(
                    self.mapping.advance_history(history[None, :], compressed)[0]
                )
            self.host.stage_embeddings(requests, padded_rows=rows)
            next_histories = np.asarray(next_histories, dtype=np.int64).reshape(
                histories.shape
            )
        self.host.wait_for_embeddings()
        return EngramInputs(
            {
                layer: self.host.embeddings(layer).unsqueeze(0)
                for layer in self.host.layer_ids
            },
            next_histories,
            tuple(compressed_rows),
        )

    def close(self):
        if self.resources is not None:
            self.resources.close()
            self.resources = None
        else:
            self.host.shutdown()
