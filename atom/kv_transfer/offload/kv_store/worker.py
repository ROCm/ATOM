# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker side of a KV-store offload, whatever the store.

Everything but the transport is the dense offload's: the executors and the
per-step producer fence (``start_load_kv``), the completion reports, the block
GPU connector and the codec of the layout the server resolves to. A packed
chunk moves through:

* save: KV blocks --pack, copy--> a slot of the registered transfer pool
  --``client.put``--> the store;
* load: store --``client.get``--> a pool slot --copy, unpack--> KV blocks.

The pool is in the worker GPU's HBM: the store's transport reads and writes
GPU memory directly, and no byte passes through host memory.

Completion protocol, per worker and operation:

* A save ends in exactly one store terminal (``_record_store_terminal``,
  directly or through ``_record_save_failure``). A failure once every GPU read
  of the save has finished -- a failed put, a pool with no slot left, the
  source read deadline -- also claims the source quiescent and reports the
  chunks it never read as source-safe, so the scheduler may retry at once and
  every PP stage reports the same per-chunk set. An exception from the GPU
  copy claims neither: it propagates, and ``_guard`` reports the failure
  without the quiescent claim.
* A chunk the store already holds when a save reaches its window is neither
  read nor put: it is source-safe at once and counts as stored, so a save
  whose every chunk was held ends in a success like any other.
* A load ends in exactly one of done or failed, all-or-nothing.
* A slot returns to the pool only once no GPU or store access can still reach
  it. One a store transfer or a GPU copy may still reach is quarantined for
  good, and the pool's memory then outlives the worker (``pool.close``).
* A put that leaves its slot quarantined may also still write store memory
  the store has since given to another key: from then on the worker loads
  nothing from the store, and marks it as the backend can
  (``_distrust_store``). Before every get, and after a load's last one, it
  asks whether it still may (``_load_refusal``).

:class:`KVStoreOffloadConnector` is that loop. A backend subclasses it with
its client, its settings, its startup probe, how it reads its client's
answers (``_put_chunks`` / ``_settle_put``, ``_get_chunks`` /
``_get_failure``) and how it marks a store it no longer trusts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from atom.kv_transfer.disaggregation.types import SaveOperationId, SaveSourceGroupId
from atom.kv_transfer.offload import atom_lmcache_staging
from atom.kv_transfer.offload._block_gpu_connector import (
    _DEFAULT_GPU_STAGING_BYTES,
    BlockGPUConnector,
)
from atom.kv_transfer.offload._offload_common import pp_aware_rank_and_world
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.kv_store.client import CallRefused, KVStoreClient
from atom.kv_transfer.offload.kv_store.keys import DIGEST_BYTES, chunk_keys
from atom.kv_transfer.offload.kv_store.pool import (
    Slot,
    SlotPoolExhausted,
    TransferSlotPool,
    pool_allocation_bytes,
)
from atom.kv_transfer.offload.kv_store.scheduler import (
    monotonic_clock_id,
    require_monotonic_clock_id,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds between two summaries of a worker's store traffic.
_STATS_LOG_INTERVAL_S = 60.0
# Seconds between two warnings of one kind; failures repeat while a store or
# fabric is unhealthy, and their count is in the next warning and the
# [OFFLOAD-STORE-STATS] line.
_WARNING_INTERVAL_S = 30.0
# A save stops reading its source blocks this long (at most) before the
# scheduler may reclaim them; see `_source_read_deadline`.
_MAX_SOURCE_READ_MARGIN_S = 60.0


def _staging_buffer_bound() -> int:
    """Bytes each staging buffer of the block GPU connector may hold.

    Windows whose slots are scattered go through it, one buffer per copy
    thread, allocated after the KV cache is sized. A bound from settings alone,
    not from the KV geometry, is one ``kv_budget_reserve_bytes`` can hold back.
    """
    bound = envs.OFFLOAD_GPU_STAGING_MAX_BYTES
    return _DEFAULT_GPU_STAGING_BYTES if bound is None else int(bound)


def _tp_group():
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group()


def _tail_first_windows(start: int, end: int, size: int) -> list[range]:
    """``[start, end)`` in windows of ``size`` chunks, the highest first."""
    windows = []
    high = end
    while high > start:
        low = max(start, high - size)
        windows.append(range(low, high))
        high = low
    return windows


def _head_first_windows(start: int, end: int, size: int) -> list[range]:
    """``[start, end)`` in windows of ``size`` chunks, the lowest first."""
    return [range(low, min(end, low + size)) for low in range(start, end, size)]


def _held_runs(window: range, held: list[bool]) -> list[tuple[range, bool]]:
    """``window`` in maximal runs of chunks alike in ``held``, the highest first.

    ``held[i]`` is chunk ``window[i]``'s.
    """
    runs: list[tuple[range, bool]] = []
    high = window.stop
    while high > window.start:
        state = held[high - 1 - window.start]
        low = high - 1
        while low > window.start and held[low - 1 - window.start] == state:
            low -= 1
        runs.append((range(low, high), state))
        high = low
    return runs


class KVStoreOffloadConnector(DenseOffloadConnector):
    """Offload worker whose tier is an external KV store, reached directly.

    A subclass sets the class attributes below and implements the hooks that
    raise ``NotImplementedError``.
    """

    # How logs name the backend.
    LABEL = "KV store offload"
    # The backend's transfer pool (a `TransferSlotPool` naming the backend).
    POOL_CLASS: type[TransferSlotPool] = TransferSlotPool
    # Whether the copies stage on the forward's stream (`set_forward_stream`)
    # instead of side streams of their own. Offload kernels on a side stream
    # run concurrently with the forward, and on ROCm a forward kernel that
    # needs every CU co-resident (a persistent kernel) can then wait for CUs
    # those kernels hold while they wait for it: the engine hangs. A backend
    # turns this off only where it has shown not to, and then picks its side
    # streams' priority (`_copy_stream_priority`).
    STAGES_ON_FORWARD_STREAM = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self._store_cfg = self._parse_store_config(kvc)
        self._check_engine_compatibility(self._store_cfg, config)
        self._require_monotonic_clock_id()
        self.chunk_size = self._store_cfg.chunk_tokens
        self._client: KVStoreClient | None = None
        self._pool: TransferSlotPool | None = None
        self._gpu_connector: BlockGPUConnector | None = None
        self._namespace: str | None = None
        self._world = 1
        self._chunk_bytes = 0
        self._stats_lock = threading.Lock()
        # Whether one-run windows are copied in place; see `register_kv_caches`.
        self._in_place = False
        # Per copy thread: the stream its in-place packs and unpacks run on.
        self._in_place_tls = threading.local()
        self._next_stats_log_at = 0.0
        self._next_warning_at: dict[str, float] = {}
        self._suppressed_warnings: Counter = Counter()
        # Per save: the windows looked up, and the chunks put and found held.
        self._save_counts: Counter = Counter()
        # Why this worker loads nothing from the store any more, once it does
        # not: its own put (`_distrust_store`) or the backend's mark of it.
        self._store_distrusted: str | None = None
        timeout = self._store_cfg.save_abandon_timeout_s
        self._source_read_window_s = timeout - min(
            _MAX_SOURCE_READ_MARGIN_S, timeout / 5
        )

    # -- the backend ------------------------------------------------------
    @classmethod
    def _parse_store_config(cls, kv_transfer_config: Any) -> Any:
        """The backend's parsed settings (``chunk_tokens``, pool sizes,
        ``save_abandon_timeout_s`` and its own)."""
        raise NotImplementedError

    @staticmethod
    def _check_engine_compatibility(store_cfg: Any, config: Any) -> None:
        """Refuse an engine the backend does not serve."""
        raise NotImplementedError

    @staticmethod
    def _require_monotonic_clock_id() -> str:
        return require_monotonic_clock_id()

    def _build_codec(
        self, kv_caches: Any, transfer_tensors: Any, num_blocks: int | None
    ) -> Any:
        """The codec of the layout the server resolves to."""
        raise NotImplementedError

    def _store_namespace(self) -> str:
        """This server's key namespace."""
        raise NotImplementedError

    def _select_endpoint(self, device: torch.device) -> Any:
        """What the client connects to, before anything is allocated."""
        raise NotImplementedError

    def _endpoint_log(self, endpoint: Any) -> str:
        """How the registration log names ``endpoint``."""
        raise NotImplementedError

    def _open_client(self, endpoint: Any) -> KVStoreClient:
        """A connected client of ``endpoint``."""
        raise NotImplementedError

    def _stop_client_users(self) -> bool:
        """Stop the backend's own users of the client; whether none is still
        inside a call (the client may then be closed)."""
        return True

    def _client_kept_open(self) -> None:
        """Called by `close` when `_stop_client_users` says the client is busy."""

    def _probe_store(
        self,
        client: KVStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
    ) -> None:
        """Send one chunk through the store and compare it back, or raise."""
        raise NotImplementedError

    def _store_marked(self, client: KVStoreClient, endpoint: Any) -> bool:
        """Whether the store is marked as one no client may load from.

        Raises when that cannot be told: registration then fails.
        """
        return False

    def _refuse_loads_from_startup_mark(self, endpoint: Any) -> None:
        """Load nothing more from a store `_store_marked` found marked."""
        raise NotImplementedError

    def _put_chunks(
        self,
        client: KVStoreClient,
        req: LMCacheReqMeta,
        run: range,
        keys: list[str],
        slots: list[Slot],
    ) -> Any:
        """Put chunks ``run`` of ``req`` from ``slots`` under ``keys``.

        Returns whatever `_settle_put` reads. Raises :class:`CallRefused`
        when nothing was sent; anything else it raises leaves the slots
        quarantined and the store distrusted.
        """
        raise NotImplementedError

    def _settle_put(
        self, pool: TransferSlotPool, slots: list[Slot], answer: Any
    ) -> str | None:
        """Release or quarantine a put's slots; why it failed, None if stored."""
        raise NotImplementedError

    def _get_chunks(
        self, client: KVStoreClient, keys: list[str], slots: list[Slot]
    ) -> Any:
        """Get ``keys`` into ``slots``; whatever `_get_failure` reads.

        Raises like `_put_chunks`; anything but :class:`CallRefused` leaves
        the slots quarantined.
        """
        raise NotImplementedError

    def _get_failure(
        self, pool: TransferSlotPool, slots: list[Slot], answer: Any
    ) -> str | None:
        """Why a get did not read every chunk whole, None when it did.

        On a failure the slots are released or quarantined here; on a success
        they stay leased for the unpack.
        """
        raise NotImplementedError

    def _copy_stream_priority(self, device: torch.device) -> int:
        """The priority of this worker's side pack, copy and in-place streams,
        on a backend that does not stage on the forward's stream: torch's
        default, 0, unless the backend asks for another."""
        return 0

    def _distrust_store(self, why: str) -> None:
        """Load nothing more from the store after a put that may still write.

        Calls `_refuse_loads` and marks the store as the backend can.
        """
        raise NotImplementedError

    def _load_refusal(self, client: KVStoreClient) -> str | None:
        """Why this worker may not load from its store now; None when it may."""
        raise NotImplementedError

    # -- sizing and lifecycle --------------------------------------------
    @classmethod
    def kv_budget_reserve_bytes(cls, config: Any) -> int:
        """HBM this worker allocates once the KV cache is sized.

        The transfer slot pool, which `register_kv_caches` allocates, and a
        staging buffer per save and load thread (the widths
        `_init_worker_common` gives the executors), each at most
        `_staging_buffer_bound`.
        """
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        cfg = cls._parse_store_config(kvc)
        threads = envs.OFFLOAD_COPY_WORKERS + envs.OFFLOAD_LOAD_WORKERS
        return (
            pool_allocation_bytes(cfg.save_pool_bytes, cfg.load_pool_bytes)
            + threads * _staging_buffer_bound()
        )

    def close(self) -> None:
        """Join the executors, then release the GPU connector, pool and client."""
        super().close()
        # A backend's own call still inside the client keeps it open: closing
        # it would pull it from under the call.
        client_idle = self._stop_client_users()
        gpu_connector, self._gpu_connector = self._gpu_connector, None
        if gpu_connector is not None:
            gpu_connector.close()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
        client, self._client = self._client, None
        if client is not None and client_idle:
            client.close()
        elif client is not None:
            self._client_kept_open()

    # The stream the forward runs on: seeded at registration, then reported
    # before each forward by `RapidServeModelRunner`, the one runner that picks
    # a stream per forward (`RapidServeModelRunner.prefill_forward`). Stays
    # None for a backend that stages on side streams.
    _forward_stream: Any = None

    def set_forward_stream(self, stream: Any) -> None:
        if stream is not None and self.STAGES_ON_FORWARD_STREAM:
            self._forward_stream = stream

    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        # The model runner's thread: its current stream is the forward's. The
        # first load is dispatched before the first forward reports its stream.
        if torch.cuda.is_available():
            self.set_forward_stream(torch.cuda.current_stream())
        rank, world = pp_aware_rank_and_world(self._config, _tp_group())
        self._rank = rank
        self._world = world
        # Scheduler blocks, threaded from the model runner. MLA stores its KV
        # token-major, so the codec cannot infer the count from shape[0].
        # The dense codec reads kv_caches. M3 and V4.1 read the PAGE region map.
        # V4.1 also requires the replay contract; its state is not stored.
        self._codec = self._build_codec(kv_caches, transfer_tensors, num_blocks)
        cfg = self._store_cfg
        namespace = self._store_namespace()
        device = self._codec.device
        endpoint = self._select_endpoint(device)
        gpu_connector = BlockGPUConnector(
            self._codec,
            self.block_size,
            chunk_size=self.chunk_size,
            virtual_block_size=self.virtual_block_size,
            source_safe_callback=self._source_group_safe,
            max_staging_bytes=_staging_buffer_bound(),
            # Staged on the forward's stream unless the backend opts out
            # (`STAGES_ON_FORWARD_STREAM`).
            compute_stream_provider=(
                (lambda: self._forward_stream)
                if self.STAGES_ON_FORWARD_STREAM
                else None
            ),
            stream_priority=self._copy_stream_priority(torch.device(device)),
        )
        client = pool = None
        try:
            client = self._open_client(endpoint)
            pool = self.POOL_CLASS(
                device=device,
                chunk_bytes=gpu_connector.gpu_staging_chunk_bytes,
                save_bytes=cfg.save_pool_bytes,
                load_bytes=cfg.load_pool_bytes,
                client=client,
            )
            self._probe_store(client, pool, namespace, rank)
            marked = self._store_marked(client, endpoint)
        except BaseException:
            client_idle = self._stop_client_users()
            if pool is not None:
                pool.close()
            if client is not None and client_idle:
                client.close()
            gpu_connector.close()
            raise
        if marked:
            self._refuse_loads_from_startup_mark(endpoint)
        self._gpu_connector = gpu_connector
        self._client = client
        self._pool = pool
        self._namespace = namespace
        self._chunk_bytes = gpu_connector.gpu_staging_chunk_bytes
        # A window whose slots are one run of the pool is a chunk-major buffer
        # the codec fills or drains with one kernel; the block GPU connector
        # instead stages it 48 MB at a time, with a copy per chunk and a
        # stream handshake per group, each a GIL round trip against the
        # forward thread. A codec with ``supports_strided_restore`` (the
        # DSV4/M3 PAGE codec) also restores a run of padded slots in place;
        # saves into padded slots still stage. The fused kernel exists only on
        # the GPU.
        self._in_place = self._codec.has_fused_chunk_major_staging
        self._next_stats_log_at = time.monotonic() + _STATS_LOG_INTERVAL_S
        logger.info(
            "%s worker rank=%d world=%d: namespace=%s %s "
            "bytes_per_block=%d chunk=%d chunk_bytes=%d "
            "pool=%s save_slots=%d load_slots=%d gpu_staging_buffer_bytes=%d "
            "save=%s load=%s save_workers=%d load_workers=%d",
            self.LABEL,
            rank,
            world,
            namespace,
            self._endpoint_log(endpoint),
            self._codec.bytes_per_block,
            self.chunk_size,
            self._chunk_bytes,
            pool.device,
            pool.capacity("save"),
            pool.capacity("load"),
            gpu_connector.gpu_staging_buffer_bytes,
            self._do_save,
            self._do_load,
            self.save_workers,
            self.load_workers,
        )

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if isinstance(metadata, LMCacheOffloadMetadata):
            # The source read deadline of a save counts from its dispatch, not
            # from whenever a save thread gets to it; see
            # `_source_read_deadline`. Placed here, once per save.
            received_at = time.monotonic()
            for req in metadata.requests:
                if req.save_spec is not None:
                    req._kv_store_dispatched_at = self._dispatch_time(req, received_at)
        super().start_load_kv(metadata)

    def _dispatch_time(self, req: LMCacheReqMeta, received_at: float) -> float:
        """When the scheduler dispatched ``req``, on this worker's monotonic clock.

        The scheduler stamps it on its CLOCK_MONOTONIC, the clock its reclaim
        deadline runs on. Every process of an engine runs on one host -- the PP
        stages pass metadata over ipc:// sockets, the TP workers their calls
        over shared memory -- where that is one clock, shared by all and moved
        by no wall-clock step: a save's age read off it is the age the
        scheduler counts, never less. A stamp from another clock has no age
        this worker can read, and one counted too short would let the save
        read blocks the scheduler already handed to other requests: the save
        is placed a whole read window before its receipt, so it reads nothing.
        A save without a stamp counts from its receipt. Both sides refuse to
        start where the kernel does not name the boot
        (`require_monotonic_clock_id`), so the clock here is always named.
        """
        if req.dispatched_at is None:
            return received_at
        clock = monotonic_clock_id()
        if req.dispatch_clock != clock:
            self._warn_rate_limited(
                "save:foreign_clock",
                "%s: req=%s was dispatched on another clock "
                "(boot %s; this worker's is %s); its save reads nothing",
                self.LABEL,
                req.req_id,
                req.dispatch_clock,
                clock,
            )
            return received_at - self._source_read_window_s
        # On one clock a dispatch never comes after its receipt.
        return min(float(req.dispatched_at), received_at)

    def _source_read_deadline(self, req: LMCacheReqMeta) -> float | None:
        """When a save must stop starting GPU reads of its source blocks.

        The scheduler reclaims the source of a save that has not reported for
        ``save_abandon_timeout_s``, counted from its dispatch, and hands the
        blocks to other requests. A save still queued then would pack their
        bytes under this prompt's keys -- a poisoned cache entry. Saves that
        reach a window past the deadline stop instead. The window counts from
        the dispatch too, not from when this worker received the save: a
        downstream PP stage receives it only after the forwards queued ahead
        of it, seconds later (`_dispatch_time`). The margin covers the copy in
        flight, which finishes within milliseconds.
        """
        dispatched_at = getattr(req, "_kv_store_dispatched_at", None)
        if dispatched_at is None:
            return None
        return float(dispatched_at) + self._source_read_window_s

    # -- copy daemon threads ---------------------------------------------
    def _registered(
        self,
    ) -> tuple[TransferSlotPool, KVStoreClient, BlockGPUConnector]:
        if self._pool is None or self._client is None or self._gpu_connector is None:
            raise RuntimeError(f"{self.LABEL}: KV caches are not registered")
        return self._pool, self._client, self._gpu_connector

    def _window_keys(self, req: LMCacheReqMeta, window: range) -> list[str]:
        """This rank's store keys of the chunks of ``req`` in ``window``."""
        return chunk_keys(
            self._namespace,
            self._rank,
            self._world,
            req.chunk_hashes,
            window.start,
            window.stop,
        )

    def _save_segments(
        self,
        client: KVStoreClient,
        req: LMCacheReqMeta,
        windows: list[range],
    ) -> Iterator[tuple[range, bool, list[str], float | None]]:
        """``windows`` in runs of chunks the store holds or not, highest first.

        Per run: its chunks, whether the store holds them, their keys and the
        milliseconds the lookup took, on a window's first run only (None on
        the rest). One
        lookup per window, made when the save reaches it. Another request's
        save often stored a prefix already -- the scheduler's lookup counts
        only the prefix every rank holds, so a save starts below what this
        rank has -- and putting it again would pack and send bytes the store
        then drops. A lookup that fails, or a key it has no answer for, holds
        nothing: the chunk is put as it would be without a lookup. The client
        answers one code per key or raises (`KVStoreClient.exists`).
        """
        for window in windows:
            keys = self._window_keys(req, window)
            t0 = time.perf_counter()
            try:
                held = [answer == 1 for answer in client.exists(keys)]
            except Exception as exc:  # noqa: BLE001  # the window is put whole
                self._warn_rate_limited(
                    "save:exists_failed",
                    "%s: a save's lookup failed (%r), req=%s; its "
                    "chunks are put without one",
                    self.LABEL,
                    exc,
                    req.req_id,
                )
                held = [False] * len(window)
            lookup_ms = (time.perf_counter() - t0) * 1000
            with self._stats_lock:
                self._save_counts["lookups"] += 1
            for run, run_held in _held_runs(window, held):
                offset = run.start - window.start
                yield run, run_held, keys[offset : offset + len(run)], lookup_ms
                lookup_ms = None

    def _do_save_req(self, req: LMCacheReqMeta, *, producer_event=None) -> None:
        ss = req.save_spec
        assert ss is not None
        self._pin_in_place_stream()
        t_total0 = time.perf_counter()
        chunk = self.chunk_size
        end = len(req.chunk_hashes or b"") // DIGEST_BYTES
        start = int(ss.skip_leading_tokens) // chunk
        if start >= end:
            self._record_store_terminal(req, True)
            return
        pool, client, gpu_connector = self._registered()
        deadline = self._source_read_deadline(req)
        windows = _tail_first_windows(
            start, end, pool.window("save", self.save_workers)
        )
        # The lowest chunk handed to the GPU or found held so far; [start,
        # read_from) is unread.
        read_from = end
        failure = detail = None
        segments_run = put_bytes = in_place_segments = 0
        windows_reached = put_chunks = held_chunks = 0
        pack_ms = put_ms = exists_ms = 0.0
        t_store0 = time.perf_counter()
        for segment, held, keys, lookup_ms in self._save_segments(client, req, windows):
            if lookup_ms is not None:  # the first segment of a window
                windows_reached += 1
                exists_ms += lookup_ms
            if held:
                # Already in the store: neither read nor put, and source-safe
                # at once, highest first like the rest.
                self._chunks_source_safe(req.save_operation, reversed(segment))
                read_from = segment.start
                held_chunks += len(segment)
                continue
            try:
                slots = pool.acquire("save", len(segment))
            except SlotPoolExhausted as exc:
                failure, detail = "pool_exhausted", str(exc)
                break
            # Checked after the wait for slots, right before the GPU read.
            if deadline is not None and time.monotonic() >= deadline:
                pool.release(slots)
                failure = "source_read_deadline"
                detail = (
                    f"queued past {self._source_read_window_s:.0f}s, after which "
                    "the scheduler may reclaim the source blocks"
                )
                break
            segments_run += 1
            run = pool.contiguous_view(slots) if self._in_place else None
            t_pack0 = time.perf_counter()
            try:
                if run is not None:
                    self._copy_in_place(
                        run,
                        segment,
                        req.block_ids,
                        pack=True,
                        producer_event=producer_event,
                    )
                else:
                    with gpu_connector.track_save_source(req.save_operation):
                        gpu_connector.batched_from_gpu(
                            slots,
                            [index * chunk for index in segment],
                            [(index + 1) * chunk for index in segment],
                            block_ids=req.block_ids,
                            producer_event=producer_event,
                        )
            except BaseException:
                self._release_once_device_idle(pool, slots)
                raise
            pack_ms += (time.perf_counter() - t_pack0) * 1000
            # Both paths return after their final stream sync: no GPU work of
            # this segment still reads the source or writes the slots.
            read_from = segment.start
            if run is not None:
                in_place_segments += 1
                # Highest first, as the block GPU connector reports them.
                self._chunks_source_safe(req.save_operation, reversed(segment))
            t_put0 = time.perf_counter()
            try:
                answer = self._put_chunks(client, req, segment, keys, slots)
            except CallRefused as exc:
                # Refused before the store had the call: nothing was sent.
                pool.release(slots)
                failure, detail = "put_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                pool.quarantine(slots, reason="a put raised")
                self._distrust_store(f"a put raised {exc!r}")
                failure, detail = "put_raised", repr(exc)
                break
            put_ms += (time.perf_counter() - t_put0) * 1000
            put_chunks += len(slots)
            problem = self._settle_put(pool, slots, answer)
            if problem is not None:
                failure, detail = "put_failed", problem
                break
            put_bytes += self._chunk_bytes * len(slots)
        with self._stats_lock:
            self._save_counts["put_chunks"] += put_chunks
            self._save_counts["held_chunks"] += held_chunks
        store_ms = (time.perf_counter() - t_store0) * 1000
        if failure is not None:
            self._warn_rate_limited(
                f"save:{failure}",
                "%s: a save failed (%s: %s), req=%s; its "
                "remaining chunks are not stored",
                self.LABEL,
                failure,
                detail,
                req.req_id,
            )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d skip=%d chunks=%d "
                "held_chunks=%d put_chunks=%d windows=%d segments=%d "
                "in_place_segments=%d put_bytes=%d producer_fenced=%d "
                "exists_ms=%.2f pack_ms=%.2f put_ms=%.2f store_ms=%.2f "
                "total_ms=%.2f status=%s",
                self._rank,
                req.req_id,
                end * chunk,
                start * chunk,
                end - start,
                held_chunks,
                put_chunks,
                windows_reached,
                segments_run,
                in_place_segments,
                put_bytes,
                int(producer_event is not None and segments_run > 0),
                exists_ms,
                pack_ms,
                put_ms,
                store_ms,
                (time.perf_counter() - t_total0) * 1000,
                failure or "ok",
            )
        self._maybe_log_store_stats()
        # The terminal goes last: anything raising before it leaves `_guard` to
        # report the one failure, never a second terminal after a first.
        if failure is None:
            self._record_store_terminal(req, True)
        else:
            self._abort_save(req, range(start, read_from))

    def _abort_save(self, req: LMCacheReqMeta, unread: range) -> None:
        """End a save whose GPU reads have all finished, as a failure.

        Quiescent, and every chunk it never read is source-safe on this
        worker: the other stages report their chunks as their copies finish,
        and a stage that stopped early must report the same set, or those
        groups never reach their all-stage quorum.
        """
        self._chunks_source_safe(req.save_operation, unread)
        self._record_save_failure(req, source_quiescent=True)

    def _chunks_source_safe(
        self, operation: SaveOperationId, indices: Iterable[int]
    ) -> None:
        """Report chunks ``indices`` of a save source-safe, one group each."""
        chunk = self.chunk_size
        for index in indices:
            self._source_group_safe(
                SaveSourceGroupId(operation, ((index * chunk, (index + 1) * chunk),))
            )

    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        ls = req.load_spec
        assert ls is not None
        self._pin_in_place_stream()
        hbm = int(ls.hbm_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        t_total0 = time.perf_counter()
        if lmc <= hbm:
            with self._lock:
                self._done_load.add(self._load_completion_id(req))
            return
        chunk = self.chunk_size
        end_tokens = (
            lmc if ls.transfer_end_tokens is None else int(ls.transfer_end_tokens)
        )
        first, end = hbm // chunk, end_tokens // chunk
        problem = None
        if hbm % chunk:
            problem = f"HBM prefix {hbm} is not chunk-aligned"
        elif end_tokens % chunk or end_tokens < lmc:
            problem = f"transfer end {end_tokens} does not cover {lmc} in chunks"
        elif len(req.chunk_hashes or b"") < end * DIGEST_BYTES:
            problem = "the request carries too few chunk digests"
        if problem is not None:
            logger.warning(
                "%s: cannot load req=%s hbm=%d lmc=%d chunk=%d: %s; re-prefill",
                self.LABEL,
                req.req_id,
                hbm,
                lmc,
                chunk,
                problem,
            )
            with self._lock:
                self._failed_load.add(self._load_completion_id(req))
                self._record_load_error_blocks(req)
            return

        pool, client, gpu_connector = self._registered()
        windows = _head_first_windows(
            first, end, pool.window("load", self.load_workers)
        )
        failure = detail = None
        windows_run = loaded_chunks = in_place_windows = 0
        get_ms = unpack_ms = 0.0
        for window in windows:
            # Checked before every get: a save thread may give up a put
            # while this load runs, and the store may be marked meanwhile.
            refusal = self._load_refusal(client)
            if refusal is not None:
                failure, detail = "store_distrusted", refusal
                break
            try:
                slots = pool.acquire("load", len(window))
            except SlotPoolExhausted as exc:
                failure, detail = "pool_exhausted", str(exc)
                break
            windows_run += 1
            keys = self._window_keys(req, window)
            t_get0 = time.perf_counter()
            try:
                answer = self._get_chunks(client, keys, slots)
            except CallRefused as exc:
                pool.release(slots)
                failure, detail = "get_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                # A get's late transfer writes only these slots: it reads the
                # store's memory and writes none of it, so quarantining them
                # is all it takes, unlike a put's (`_distrust_store`).
                pool.quarantine(slots, reason="a get raised")
                failure, detail = "get_raised", repr(exc)
                break
            get_ms += (time.perf_counter() - t_get0) * 1000
            # All-or-nothing: an object of another size is another layout.
            problem = self._get_failure(pool, slots, answer)
            if problem is not None:
                failure, detail = "get_failed", problem
                break
            run = pool.contiguous_view(slots) if self._in_place else None
            chunk_stride_bytes = None
            if run is None and self._in_place and self._codec.supports_strided_restore:
                run = pool.strided_view(slots)
                if run is not None:
                    chunk_stride_bytes = pool.slot_stride
            t_unpack0 = time.perf_counter()
            try:
                if run is not None:
                    self._copy_in_place(
                        run,
                        window,
                        req.block_ids,
                        pack=False,
                        chunk_stride_bytes=chunk_stride_bytes,
                    )
                    in_place_windows += 1
                else:
                    gpu_connector.batched_to_gpu(
                        slots,
                        [index * chunk for index in window],
                        [(index + 1) * chunk for index in window],
                        block_ids=req.block_ids,
                    )
            except BaseException:
                self._release_once_device_idle(pool, slots)
                raise
            unpack_ms += (time.perf_counter() - t_unpack0) * 1000
            # Both paths synchronized before returning: the slots are idle.
            pool.release(slots)
            loaded_chunks += len(window)
        if failure is None:
            # And once after the last get: a mark placed while a get ran is
            # one the look before it missed, and that get may have read what
            # the given-up put wrote. The load fails, so nothing it unpacked
            # is used.
            refusal = self._load_refusal(client)
            if refusal is not None:
                failure, detail = "store_distrusted", refusal
        if failure is not None:
            self._warn_rate_limited(
                f"load:{failure}",
                "%s: a load failed (%s: %s), req=%s; it is prefilled instead",
                self.LABEL,
                failure,
                detail,
                req.req_id,
            )
        if self._profile_enabled():
            retrieve_ms = get_ms + unpack_ms
            total_bytes = loaded_chunks * self._chunk_bytes
            logger.info(
                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d retrieved=%d "
                "status=%s chunks=%d windows=%d in_place_windows=%d total_bytes=%d "
                "get_ms=%.2f unpack_ms=%.2f retrieve_ms=%.2f total_ms=%.2f "
                "effective_gbps=%.2f",
                self._rank,
                req.req_id,
                hbm,
                lmc,
                loaded_chunks * chunk,
                failure or "ok",
                end - first,
                windows_run,
                in_place_windows,
                total_bytes,
                get_ms,
                unpack_ms,
                retrieve_ms,
                (time.perf_counter() - t_total0) * 1000,
                total_bytes / retrieve_ms / 1e6 if retrieve_ms > 0 else 0.0,
            )
        self._maybe_log_store_stats()
        # Last, for the reason `_do_save_req` gives.
        with self._lock:
            if failure is None:
                self._done_load.add(self._load_completion_id(req))
            else:
                self._failed_load.add(self._load_completion_id(req))
                self._record_load_error_blocks(req)

    # -- in-place copies ---------------------------------------------------
    def _in_place_stream(self) -> torch.cuda.Stream:
        """This copy thread's stream for in-place packs and unpacks."""
        # The forward's stream as pinned when this transfer began
        # (`_pin_in_place_stream`), so every window of it uses one stream.
        shared = getattr(self._in_place_tls, "pinned", None)
        if shared is not None:
            return shared
        stream = getattr(self._in_place_tls, "stream", None)
        if stream is None:
            device = torch.device(self._codec.device)
            stream = torch.cuda.Stream(
                device=device, priority=self._copy_stream_priority(device)
            )
            self._in_place_tls.stream = stream
        return stream

    def _pin_in_place_stream(self) -> None:
        """Pin the stream this thread's coming load or save stages on: the
        forward's, as it is now (None on a side-stream backend). A runner that
        picks a stream per forward may move on during the transfer; the switch
        happens between transfers."""
        self._in_place_tls.pinned = self._forward_stream

    def _copy_in_place(
        self,
        run: torch.Tensor,
        window: range,
        block_ids: list[int],
        *,
        pack: bool,
        producer_event: Any = None,
        chunk_stride_bytes: int | None = None,
    ) -> None:
        """Pack ``window``'s KV blocks into ``run``, or unpack it into them.
        With ``chunk_stride_bytes``, ``run`` holds padded Store slots and is
        unpacked in place; each slot's padding is skipped, not read.

        Returns once the copy is done. A pack is ordered after
        ``producer_event`` like the staged pack, so the kernels that produced
        the KV finish first.
        """
        codec = self._codec
        chunk = self.chunk_size
        stream = self._in_place_stream()
        if producer_event is not None:
            # Also on the forward's stream: it may have changed since the
            # producer recorded the event, and a same-stream wait is free.
            stream.wait_event(producer_event)
        # Only the window's part of the block table, with token offsets from
        # its first chunk: converting the whole table would cost every window
        # time in proportion to the prompt's length.
        per_chunk = chunk // self.virtual_block_size
        offsets = range(len(window))
        groups = self._gpu_connector._ranges_to_block_ids(
            [offset * chunk for offset in offsets],
            [(offset + 1) * chunk for offset in offsets],
            block_ids=block_ids[window.start * per_chunk : window.stop * per_chunk],
        )
        owner = codec.prepare_block_id_groups(
            [groups], device=codec.device, stream=stream
        )
        if chunk_stride_bytes is not None:
            codec.chunk_major_strided_buffer_to_gpu_prepared(
                run,
                owner,
                0,
                stream=stream,
                items_per_chunk=per_chunk,
                chunk_stride_bytes=chunk_stride_bytes,
            )
        elif pack:
            codec.gpu_to_chunk_major_device_buffer_prepared(
                run, owner, 0, stream=stream
            )
        else:
            codec.chunk_major_device_buffer_to_gpu_prepared(
                run, owner, 0, stream=stream
            )
        atom_lmcache_staging.fence_submitted_work(stream)

    # -- distrust -----------------------------------------------------------
    def _refuse_loads(self, why: str) -> bool:
        """Load nothing more from the store; True the first time only."""
        with self._stats_lock:
            if self._store_distrusted is not None:
                return False
            self._store_distrusted = why
        return True

    def _release_once_device_idle(
        self, pool: TransferSlotPool, slots: list[Slot]
    ) -> None:
        """After a failure: release the slots only if the device is idle.

        A GPU copy that raised may still run; a device that cannot be fenced
        leaves the slots quarantined.
        """
        try:
            _synchronize(pool.device)
        except Exception:
            logger.exception(
                "%s: device synchronize failed after a "
                "transfer slot's copy or transfer failed",
                self.LABEL,
            )
            pool.quarantine(slots, reason="a GPU copy failed and did not fence")
            return
        pool.release(slots)

    # -- observability -----------------------------------------------------
    def _warn_rate_limited(self, kind: str, message: str, *args: Any) -> None:
        """At most one warning of ``kind`` per interval, counting the rest."""
        now = time.monotonic()
        with self._stats_lock:
            if now < self._next_warning_at.get(kind, 0.0):
                self._suppressed_warnings[kind] += 1
                return
            self._next_warning_at[kind] = now + _WARNING_INTERVAL_S
            suppressed = self._suppressed_warnings.pop(kind, 0)
        logger.warning(message + " (%d more since the last warning)", *args, suppressed)

    def _maybe_log_store_stats(self) -> None:
        client, pool = self._client, self._pool
        if client is None or pool is None:
            return
        now = time.monotonic()
        with self._stats_lock:
            if now < self._next_stats_log_at:
                return
            self._next_stats_log_at = now + _STATS_LOG_INTERVAL_S
            save_counts = Counter(self._save_counts)
        stats = client.stats()
        counts = stats["counts"]
        failures = ",".join(
            f"{name}x{count}" for name, count in sorted(stats["failures"].items())
        )
        logger.info(
            "[OFFLOAD-STORE-STATS] rank=%s puts=%d put_keys=%d put_bytes=%d "
            "save_exists=%d save_put_chunks=%d save_held_chunks=%d "
            "save_held_bytes=%d "
            "gets=%d get_keys=%d get_bytes=%d failures=%s "
            "quarantined_slots=save:%d,load:%d",
            self._rank,
            counts.get("put_calls", 0),
            counts.get("put_keys", 0),
            counts.get("put_bytes", 0),
            save_counts["lookups"],
            save_counts["put_chunks"],
            save_counts["held_chunks"],
            save_counts["held_chunks"] * self._chunk_bytes,
            counts.get("get_calls", 0),
            counts.get("get_keys", 0),
            counts.get("get_bytes", 0),
            failures or "-",
            pool.quarantined("save"),
            pool.quarantined("load"),
        )


def _synchronize(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


__all__ = ["KVStoreOffloadConnector"]
