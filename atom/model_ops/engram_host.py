"""Host-resident engram embedding tables and the prefetch that hides them.

A DeepSeek-V4.1-Flash engram table is 384M x 256 fp8 (~98 GB), two of them. They
are memory-mapped from the checkpoint and never materialized: a step gathers at
most `batch x hash_heads` rows, so touching the full table would move five
orders of magnitude more bytes than the step needs.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HostEmbeddingTable:
    """One engram layer's table, memory-mapped and gathered row-wise.

    The reference keeps the table as a float32 numpy array, which for this model
    would be 393 GB per layer -- 786 GB of host RAM for the pair, before any
    staging buffer. Rows are kept in their stored dtype and converted only after
    the gather, so the resident cost is the page cache the OS chooses to keep.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        num_rows: int,
        head_dim: int,
        scale: torch.Tensor | None = None,
    ):
        if tensor.shape[0] != num_rows:
            raise ValueError(f"table has {tensor.shape[0]} rows, expected {num_rows}")
        if tensor.shape[1] != head_dim:
            raise ValueError(
                f"table row is {tensor.shape[1]} wide, expected {head_dim}"
            )
        self._tensor = tensor
        self.num_rows = num_rows
        self.head_dim = head_dim
        self._scale = scale
        self.block_size = 0
        if scale is not None:
            if scale.shape[0] != num_rows:
                raise ValueError(
                    f"scale has {scale.shape[0]} rows, expected {num_rows}"
                )
            if head_dim % scale.shape[1]:
                raise ValueError(
                    f"head_dim {head_dim} is not divisible by {scale.shape[1]} "
                    f"scale blocks"
                )
            self.block_size = head_dim // scale.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def gather(
        self, row_indices: np.ndarray, out_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Gather rows named by `row_indices` ([...] ints) -> [..., head_dim].

        Out-of-range indices are a bug in the hash layout rather than something
        to clamp away quietly: a clamp turns a wrong table into plausible
        numbers, which is far harder to notice than an exception.
        """
        flat = np.ascontiguousarray(row_indices.reshape(-1))
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_rows):
            raise IndexError(
                f"engram row index out of range: [{flat.min()}, {flat.max()}] "
                f"not within [0, {self.num_rows})"
            )
        index = torch.from_numpy(flat)
        # One fancy-index over the whole batch, not a row at a time: measured on
        # the real table that is ~0.9 us/row against ~8 us/row for a Python loop.
        rows = self._tensor[index].to(out_dtype)
        if self._scale is not None:
            # Block-quantized: each scale covers `block_size` consecutive values
            # of a row. Skipping this does not fail, it returns values two orders
            # of magnitude off, so it is not optional.
            scale = self._scale[index].to(out_dtype)
            rows = (
                rows.reshape(-1, scale.shape[1], self.block_size) * scale.unsqueeze(-1)
            ).reshape(-1, self.head_dim)
        return rows.reshape(*row_indices.shape, self.head_dim)


class EngramPrefetchCache:
    """Bounded per-request store of prefetched embeddings.

    The reference uses an unbounded module-level dict keyed by sequence id and
    never removes anything, so a long-running server accumulates one entry per
    request served. This is an LRU with an explicit `drop` for finished
    requests, and it is the only shared state between the worker and the runner.
    """

    def __init__(self, capacity: int = 4096):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def put(self, seq_id: int, layer_id: int, value: torch.Tensor) -> None:
        with self._lock:
            key = (seq_id, layer_id)
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def take(self, seq_id: int, layer_id: int) -> torch.Tensor | None:
        with self._lock:
            return self._store.pop((seq_id, layer_id), None)

    def contains(self, seq_id: int, layer_id: int) -> bool:
        """Non-destructive probe. `take` consumes, so miss detection needs this."""
        with self._lock:
            return (seq_id, layer_id) in self._store

    def drop(self, seq_id: int) -> None:
        """Forget everything for a finished request."""
        with self._lock:
            for key in [k for k in self._store if k[0] == seq_id]:
                del self._store[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class EngramPrefetcher:
    """Runs hash + host gather for the next step while the GPU works on this one.

    One worker thread, not a thread per step: the work is numpy and torch gather
    which release the GIL, and an unbounded thread-per-step (as in the reference)
    both races with its own consumer and contends with the runner for the GIL.

    `submit` returns a Future so the consumer can wait for a specific step rather
    than hoping the daemon finished. When the result is not ready in time the
    caller computes it inline -- correctness never depends on the race.
    """

    def __init__(
        self,
        hash_mapping,
        tables: dict[int, HostEmbeddingTable],
        cache_capacity: int = 4096,
    ):
        self._hash_mapping = hash_mapping
        self._tables = tables
        self.cache = EngramPrefetchCache(cache_capacity)
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engram-prefetch"
        )
        self._inflight: Future | None = None
        # DEBUG (ATOM_ENGRAM_DEBUG_VERIFY): the token each seq was last prefetched
        # with, so `stage` can compare it against the token it recomputes from.
        self._debug_prefetch_token: dict[int, int] = {}

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._hash_mapping.config.layer_ids

    def compute(
        self, seq_ids: list[int], token_ids: np.ndarray
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Hash `token_ids` ([B, T]) and gather, for every engram layer."""
        results: dict[tuple[int, int], torch.Tensor] = {}
        compressed = self._hash_mapping.tokenizer(token_ids)
        for layer_id in self.layer_ids:
            hashes = self._hash_mapping.hash_layer(compressed, layer_id, compress=False)
            rows = self._hash_mapping.to_row_indices(hashes, layer_id)
            gathered = self._tables[layer_id].gather(rows)
            for i, seq_id in enumerate(seq_ids):
                results[(seq_id, layer_id)] = gathered[i]
        return results

    def submit(self, seq_ids: list[int], token_ids: np.ndarray) -> Future:
        """Queue a prefetch. Cheap and non-blocking; the caller keeps going."""

        def _run() -> None:
            for key, value in self.compute(seq_ids, token_ids).items():
                self.cache.put(key[0], key[1], value)
            for i, sid in enumerate(seq_ids):
                self._debug_prefetch_token[sid] = int(token_ids[i][-1])

        self._inflight = self._pool.submit(_run)
        return self._inflight

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queued prefetch lands. True if it did."""
        if self._inflight is None:
            return True
        try:
            self._inflight.result(timeout=timeout)
            return True
        except TimeoutError:
            return False

    def drop_requests(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.cache.drop(seq_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class EngramRuntime:
    """Ties the host prefetch to the device step.

    Ordering per step, with nothing added to the critical path:

      1. the previous step's sampled ids land on the host (the runner already
         copies them asynchronously and synchronizes before use), and
         `prefetch_next` queues hash + gather for them on the worker;
      2. `stage` copies whatever the worker produced into a pinned buffer and
         issues the H2D on a side stream, recording an event;
      3. `wait_for_copy` makes the compute stream wait on that event, so the
         engram layers read staged rows rather than racing the copy.

    A sequence whose prefetch has not landed is computed inline in `stage`. The
    result is identical either way -- the prefetch only decides whether the work
    was already done, never what the answer is.
    """

    def __init__(
        self,
        prefetcher: EngramPrefetcher,
        max_num_tokens: int,
        num_hash_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        from atom.utils import CpuGpuBuffer

        self.prefetcher = prefetcher
        self.max_num_tokens = max_num_tokens
        self.embed_width = num_hash_heads * head_dim
        self.device = device
        self.buffers = {
            layer_id: CpuGpuBuffer(
                max_num_tokens,
                self.embed_width,
                dtype=dtype,
                device=device,
                with_numpy=False,
            )
            for layer_id in prefetcher.layer_ids
        }
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.copy_done = torch.cuda.Event() if device.type == "cuda" else None
        self._staged_rows = 0
        # ATOM_ENGRAM_DEBUG_VERIFY cumulative counters (see `stage`).
        self._verify_stages = 0
        self._verify_rows = 0
        self._verify_bad = 0
        self._verify_done = False

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.prefetcher.layer_ids

    def stage(
        self,
        seq_ids: list[int],
        token_ids: np.ndarray | None = None,
        verify_token_ids: np.ndarray | None = None,
    ) -> int:
        """Fill the staging buffers for `seq_ids`; returns the row count staged.

        `token_ids` is only consulted for sequences the prefetch missed -- the
        caller passes the cheap host-side source (the scheduler's committed
        anchor), correct for the just-admitted rows that are the only misses.

        `verify_token_ids`, when the debug check is on, is the GROUND-TRUTH token
        per row (this step's real model input) used to recompute the reference.
        Keeping it separate is what lets verify catch a bad `token_ids` source:
        a miss recomputed from the wrong anchor diverges from this reference.
        """
        num_rows = len(seq_ids)
        if num_rows > self.max_num_tokens:
            raise ValueError(
                f"{num_rows} rows exceeds staging capacity {self.max_num_tokens}"
            )
        self.prefetcher.wait(timeout=None)

        # Probe without consuming: a hit still has to be readable by the fill
        # loop below, which is what makes this `contains` and not `take`.
        missing = [
            i
            for i, seq_id in enumerate(seq_ids)
            if not self.prefetcher.cache.contains(seq_id, self.layer_ids[0])
        ]
        computed: dict[tuple[int, int], torch.Tensor] = {}
        if missing:
            if token_ids is None:
                raise RuntimeError(
                    f"engram prefetch missed {len(missing)} sequences and no token "
                    f"ids were supplied to recompute them"
                )
            miss_ids = [seq_ids[i] for i in missing]
            computed = self.prefetcher.compute(miss_ids, token_ids[missing])

        # DEBUG (ATOM_ENGRAM_DEBUG_VERIFY): the module's invariant is that the
        # async-prefetched rows equal a fresh SYNCHRONOUS recompute -- the
        # prefetch only decides whether the work was already done, never the
        # answer. Recompute every seq inline now and compare it against what is
        # about to be staged; for a cache hit that staged value came off the
        # prefetch queued on the PREVIOUS step, so a match proves the async
        # hand-off (per-seq token alignment, cache keying, staging) is correct.
        # `=1` logs mismatches; `=strict` also raises. Off by default; doubles the
        # host gather when on.
        verify = os.environ.get("ATOM_ENGRAM_DEBUG_VERIFY")
        reference = (
            self.prefetcher.compute(seq_ids, verify_token_ids)
            if verify and not self._verify_done and verify_token_ids is not None
            else None
        )
        n_checked = n_from_cache = n_bad = n_tok_bad = 0
        logged = 0

        for layer_id in self.layer_ids:
            cpu = self.buffers[layer_id].cpu
            for row, seq_id in enumerate(seq_ids):
                from_cache = (seq_id, layer_id) not in computed
                value = computed.get((seq_id, layer_id))
                if value is None:
                    value = self.prefetcher.cache.take(seq_id, layer_id)
                if value is None:
                    raise RuntimeError(
                        f"no engram embedding for seq {seq_id} layer {layer_id}"
                    )
                if reference is not None:
                    ref = reference[(seq_id, layer_id)]
                    n_checked += 1
                    n_from_cache += int(from_cache)
                    if not torch.equal(value, ref):
                        n_bad += 1
                        # The decisive split: did we recompute from a DIFFERENT
                        # token than the prefetch used (async alignment), or the
                        # SAME token yet still diverge (gather/table bug)?
                        ref_tok = int(verify_token_ids[row][-1])
                        cache_tok = self.prefetcher._debug_prefetch_token.get(seq_id)
                        tok_match = ref_tok == cache_tok
                        n_tok_bad += int(not tok_match)
                        if logged < 8:
                            logged += 1
                            d = (value.float() - ref.float()).abs()
                            logger.error(
                                "engram VERIFY MISMATCH seq=%d layer=%d "
                                "from_cache=%s ref_token=%s cache_token=%s "
                                "token_match=%s max|delta|=%.6g ndiff=%d/%d",
                                seq_id,
                                layer_id,
                                from_cache,
                                ref_tok,
                                cache_tok,
                                tok_match,
                                float(d.max()),
                                int((d > 0).sum()),
                                d.numel(),
                            )
                    # Show the actual numbers -- prefetched (computed one step
                    # ahead) vs the on-the-spot recompute -- for the first row of
                    # the first few steps, so equality is visible as values.
                    if (
                        row == 0
                        and layer_id == self.layer_ids[0]
                        and self._verify_stages < 6
                    ):
                        pv = [
                            round(x, 4) for x in value.reshape(-1)[:6].float().tolist()
                        ]
                        rv = [
                            round(x, 4) for x in ref.reshape(-1)[:6].float().tolist()
                        ]
                        logger.warning(
                            "engram VERIFY VALUES step %d seq=%d layer=%d token=%s "
                            "equal=%s prefetched[:6]=%s recomputed[:6]=%s",
                            self._verify_stages + 1,
                            seq_id,
                            layer_id,
                            int(verify_token_ids[row][-1]),
                            torch.equal(value, ref),
                            pv,
                            rv,
                        )
                cpu[row].copy_(value.reshape(-1)[: self.embed_width])

        if reference is not None:
            self._verify_stages += 1
            self._verify_rows += n_checked
            self._verify_bad += n_bad
            # One compact line per step over a short window, so the prefill->
            # decode transient (the first decode step's cache is stale because
            # prefill's prefetch had no sampled token yet under deferred output)
            # shows up apart from any steady-state divergence. wrong-token>0 means
            # the staged token != this step's model input; ==0 with mismatch would
            # mean the gather itself diverges.
            logger.warning(
                "engram VERIFY step %d: %d/%d rows mismatch, %d wrong-token",
                self._verify_stages,
                n_bad,
                n_checked,
                n_tok_bad,
            )
            if self._verify_stages >= 12:
                self._verify_done = True
            if n_bad and verify == "strict":
                raise AssertionError(
                    f"engram prefetch/inline mismatch on {n_bad} row(s)"
                )

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                for buffer in self.buffers.values():
                    buffer.copy_to_gpu(num_rows)
                self.copy_done.record(self.copy_stream)
        else:
            for buffer in self.buffers.values():
                buffer.copy_to_gpu(num_rows)
        self._staged_rows = num_rows
        return num_rows

    def stage_dummy(self, num_rows: int) -> int:
        """Stage zeros for a warmup or capture pass.

        A dummy forward has no real sequences to look up, but it still runs the
        engram layers, so the buffers have to be the right size and the H2D has
        to happen -- warmup exists to touch exactly this path. Zeros keep the
        shapes honest without inventing token ids.
        """
        num_rows = min(int(num_rows), self.max_num_tokens)
        for buffer in self.buffers.values():
            buffer.cpu[:num_rows].zero_()
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                for buffer in self.buffers.values():
                    buffer.copy_to_gpu(num_rows)
                self.copy_done.record(self.copy_stream)
        else:
            for buffer in self.buffers.values():
                buffer.copy_to_gpu(num_rows)
        self._staged_rows = num_rows
        return num_rows

    def wait_for_copy(self) -> None:
        """Order the compute stream behind the staging H2D."""
        if self.copy_done is not None:
            torch.cuda.current_stream(self.device).wait_event(self.copy_done)

    def embeddings(self, layer_id: int) -> torch.Tensor:
        """Staged rows for one layer, [staged_rows, num_hash_heads * head_dim]."""
        return self.buffers[layer_id].gpu[: self._staged_rows]

    def prefetch_next(self, seq_ids: list[int], token_ids: np.ndarray) -> None:
        if len(seq_ids):
            self.prefetcher.submit(seq_ids, token_ids)

    def drop_requests(self, seq_ids: list[int]) -> None:
        self.prefetcher.drop_requests(seq_ids)

    def shutdown(self) -> None:
        self.prefetcher.shutdown()
