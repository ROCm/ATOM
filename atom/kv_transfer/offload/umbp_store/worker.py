# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker side of the UMBP offload.

Copied from the Mooncake Store offload's worker and rewritten for UMBP; a
shared layer is planned. Everything but the transport is the dense offload's:
the executors and the per-step producer fence (``start_load_kv``), the
completion reports, the block GPU connector and a dense or M3 PAGE codec. A
packed chunk moves through:

* save: KV blocks --pack, copy--> a slot of the registered transfer pool
  --``batch_put_from_ptr``--> the standalone server;
* load: server --``batch_get_into_ptr``--> a pool slot --copy, unpack--> KV
  blocks.

The pool is in the worker GPU's HBM, registered with the server, which copies
to and from it through HIP IPC before it answers each call.

Completion protocol, per worker and operation:

* A save ends in exactly one store terminal (``_record_store_terminal``,
  directly or through ``_record_save_failure``). A failure once every GPU read
  of the save has finished -- a put the server answered False for, a pool with
  no slot left, the source read deadline -- also claims the source quiescent
  and reports the chunks it never read as source-safe, so the scheduler may
  retry at once and every PP stage reports the same per-chunk set. An
  exception from the GPU copy claims neither: it propagates, and ``_guard``
  reports the failure without the quiescent claim.
* A load ends in exactly one of done or failed, all-or-nothing.
* A slot returns to the pool once its call returned, whatever it answered:
  the server copied before answering. A slot whose call raised is quarantined
  for good, and the pool's memory then outlives the worker (``pool.close``).
* A put that raised may also leave the server copying from a slot into an
  object; this worker then loads nothing more from the server and marks it in
  this process (``_distrust_store``, :mod:`.taint`). The mark is this
  process's only: other worker processes and other engines are not told.
* UMBP has no remove by key: the startup probe's object stays on the server
  until it is evicted.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch

from atom.kv_transfer.disaggregation.types import SaveOperationId, SaveSourceGroupId
from atom.kv_transfer.offload._block_gpu_connector import (
    _DEFAULT_GPU_STAGING_BYTES,
    BlockGPUConnector,
)
from atom.kv_transfer.offload._offload_common import pp_aware_rank_and_world
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.client import CallRefused, UMBPStoreClient
from atom.kv_transfer.offload.umbp_store.config import (
    check_engine_compatibility,
    parse_umbp_store_config,
)
from atom.kv_transfer.offload.umbp_store.keys import (
    DIGEST_BYTES,
    chunk_keys,
    probe_key,
    store_namespace,
)
from atom.kv_transfer.offload.umbp_store.layouts import build_umbp_codec
from atom.kv_transfer.offload.umbp_store.pool import (
    Slot,
    SlotPoolExhausted,
    TransferSlotPool,
    pool_allocation_bytes,
)
from atom.kv_transfer.offload.umbp_store.scheduler import (
    monotonic_clock_id,
    require_monotonic_clock_id,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds between two summaries of a worker's server traffic.
_STATS_LOG_INTERVAL_S = 60.0
# Seconds between two warnings of one kind; failures repeat while a server is
# unhealthy, and their count is in the next warning and the
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


class UMBPStoreOffloadConnector(DenseOffloadConnector):
    """Dense or M3 PAGE offload worker backed by a UMBP standalone server."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self._store_cfg = parse_umbp_store_config(kvc)
        check_engine_compatibility(self._store_cfg, config)
        require_monotonic_clock_id()
        self.chunk_size = self._store_cfg.chunk_tokens
        self._client: UMBPStoreClient | None = None
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
        # Why this worker loads nothing from the server any more, once it does
        # not: its own put (`_distrust_store`) or this process's mark of the
        # server (:mod:`.taint`).
        self._store_distrusted: str | None = None
        timeout = self._store_cfg.save_abandon_timeout_s
        self._source_read_window_s = timeout - min(
            _MAX_SOURCE_READ_MARGIN_S, timeout / 5
        )

    @classmethod
    def kv_budget_reserve_bytes(cls, config: Any) -> int:
        """HBM this worker allocates once the KV cache is sized.

        The transfer slot pool, which `register_kv_caches` allocates, and a
        staging buffer per save and load thread (the widths
        `_init_worker_common` gives the executors), each at most
        `_staging_buffer_bound`.
        """
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        cfg = parse_umbp_store_config(kvc)
        threads = envs.OFFLOAD_COPY_WORKERS + envs.OFFLOAD_LOAD_WORKERS
        return (
            pool_allocation_bytes(cfg.save_pool_bytes, cfg.load_pool_bytes)
            + threads * _staging_buffer_bound()
        )

    def close(self) -> None:
        """Join the executors, then release the GPU connector, pool and client.

        The pool unregisters its region unless a copy may still reach it, and
        the client then keeps its connection (``UMBPStoreClient.close``).
        """
        super().close()
        gpu_connector, self._gpu_connector = self._gpu_connector, None
        if gpu_connector is not None:
            gpu_connector.close()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
        client, self._client = self._client, None
        if client is not None:
            client.close()

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        rank, world = pp_aware_rank_and_world(self._config, _tp_group())
        self._rank = rank
        self._world = world
        # Scheduler blocks, threaded from the model runner. MLA stores its KV
        # token-major, so the codec cannot infer the count from shape[0].
        # The dense codec reads kv_caches. The M3 codec reads the PAGE region
        # map, which is where the NSA index cache is published.
        self._codec = build_umbp_codec(
            self._config,
            kv_caches,
            transfer_tensors,
            num_blocks,
            permit_per_request_state=self._permit_per_request_state,
        )
        cfg = self._store_cfg
        namespace = store_namespace(self._config, self.chunk_size)
        device = self._codec.device
        gpu_connector = BlockGPUConnector(
            self._codec,
            self.block_size,
            chunk_size=self.chunk_size,
            virtual_block_size=self.virtual_block_size,
            source_safe_callback=self._source_group_safe,
            max_staging_bytes=_staging_buffer_bound(),
        )
        client = pool = None
        try:
            client = UMBPStoreClient(
                address=cfg.address, startup_timeout_ms=cfg.startup_timeout_ms
            )
            pool = TransferSlotPool(
                device=device,
                chunk_bytes=gpu_connector.gpu_staging_chunk_bytes,
                save_bytes=cfg.save_pool_bytes,
                load_bytes=cfg.load_pool_bytes,
                client=client,
            )
            self._probe_store(client, pool, namespace, rank)
        except BaseException:
            if pool is not None:
                pool.close()
            if client is not None:
                client.close()
            gpu_connector.close()
            raise
        marked = taint.reason(cfg.address)
        if marked is not None:
            self._refuse_loads_from_marked_server(cfg.address, marked)
        self._gpu_connector = gpu_connector
        self._client = client
        self._pool = pool
        self._namespace = namespace
        self._chunk_bytes = gpu_connector.gpu_staging_chunk_bytes
        # A window whose slots are one run of the pool is a chunk-major buffer
        # the dense codec fills or drains with one kernel; the block GPU
        # connector instead stages it 48 MB at a time, with a copy per chunk
        # and a stream handshake per group, each a GIL round trip against the
        # forward thread. The fused kernel exists only on the GPU.
        self._in_place = self._codec.has_fused_chunk_major_staging
        self._next_stats_log_at = time.monotonic() + _STATS_LOG_INTERVAL_S
        logger.info(
            "UMBP offload worker rank=%d world=%d: namespace=%s server=%s "
            "bytes_per_block=%d chunk=%d chunk_bytes=%d "
            "pool=%s save_slots=%d load_slots=%d gpu_staging_buffer_bytes=%d "
            "save=%s load=%s save_workers=%d load_workers=%d",
            rank,
            world,
            namespace,
            cfg.address,
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

    def _probe_store(
        self,
        client: UMBPStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
    ) -> None:
        """Send one chunk of random bytes through the server and compare it back.

        An unregistered pool or an unreachable server otherwise shows up only
        as every lookup missing. A slot whose call raised stays quarantined
        when startup gives up. UMBP cannot remove by key, so the probe's object
        stays on the server until it is evicted: one chunk per worker start.
        """
        key = probe_key(namespace, rank, uuid.uuid4().hex)
        nbytes = pool.chunk_bytes
        [source] = pool.acquire("save", 1)
        [target] = pool.acquire("load", 1)
        # The probe's slots no copy may still reach.
        leased = [source, target]
        try:
            source.tensor.copy_(
                torch.randint(
                    0, 256, (nbytes,), dtype=torch.uint8, device=source.tensor.device
                )
            )
            target.tensor.zero_()
            _synchronize(pool.device)
            [stored] = self._probe_call(
                pool,
                leased,
                source,
                lambda: client.put([key], [source.ptr], [nbytes]),
                # A put that raised may still be read by the server.
                on_raise=self._distrust_store,
            )
            if not stored:
                raise RuntimeError(
                    f"UMBP startup probe: the server at {client.address} did not "
                    f"store a {nbytes}-byte chunk (full, or the pool is not "
                    "registered?)"
                )
            [present] = client.exists([key])
            if not present:
                raise RuntimeError(
                    "UMBP startup probe: a chunk just put is not on the server "
                    f"at {client.address}"
                )
            [read] = self._probe_call(
                pool,
                leased,
                target,
                lambda: client.get([key], [target.ptr], [nbytes]),
            )
            if not read:
                raise RuntimeError(
                    f"UMBP startup probe: the server at {client.address} did not "
                    f"return the {nbytes}-byte chunk just put"
                )
            _synchronize(pool.device)
            if not torch.equal(source.tensor, target.tensor):
                raise RuntimeError(
                    "UMBP startup probe: the chunk read back differs from the "
                    "one written"
                )
        except BaseException:
            self._release_once_device_idle(pool, leased)
            raise
        pool.release(leased)
        logger.warning(
            "UMBP startup probe: UMBP cannot remove by key; the probe object %s "
            "stays on the server until it is evicted",
            key,
        )

    @staticmethod
    def _probe_call(
        pool: TransferSlotPool,
        leased: list[Slot],
        slot: Slot,
        call: Any,
        *,
        on_raise: Any = None,
    ) -> list[bool]:
        """Run the probe's put or get on ``slot``, quarantining it if it raised.

        A call that returned is settled whatever it answered. A quarantined
        slot leaves ``leased``, the probe's slots still to release, and is
        reported to ``on_raise``, with why, when one is given.
        """
        try:
            return call()
        except CallRefused:
            raise  # nothing was sent: the slot is released with the rest
        except BaseException as exc:
            leased.remove(slot)
            pool.quarantine([slot], reason="a startup probe call raised")
            if on_raise is not None:
                on_raise(f"the startup probe's put raised {exc!r}")
            raise

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if isinstance(metadata, LMCacheOffloadMetadata):
            # The source read deadline of a save counts from its dispatch, not
            # from whenever a save thread gets to it; see
            # `_source_read_deadline`. Placed here, once per save.
            received_at = time.monotonic()
            for req in metadata.requests:
                if req.save_spec is not None:
                    req._umbp_store_dispatched_at = self._dispatch_time(
                        req, received_at
                    )
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
                "UMBP offload: req=%s was dispatched on another clock "
                "(boot %s; this worker's is %s); its save reads nothing",
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
        dispatched_at = getattr(req, "_umbp_store_dispatched_at", None)
        if dispatched_at is None:
            return None
        return float(dispatched_at) + self._source_read_window_s

    # -- copy daemon threads ---------------------------------------------
    def _registered(
        self,
    ) -> tuple[TransferSlotPool, UMBPStoreClient, BlockGPUConnector]:
        if self._pool is None or self._client is None or self._gpu_connector is None:
            raise RuntimeError("UMBP offload: KV caches are not registered")
        return self._pool, self._client, self._gpu_connector

    def _window_keys(self, req: LMCacheReqMeta, window: range) -> list[str]:
        """This rank's object keys of the chunks of ``req`` in ``window``."""
        return chunk_keys(
            self._namespace,
            self._rank,
            self._world,
            req.chunk_hashes,
            window.start,
            window.stop,
        )

    def _do_save_req(self, req: LMCacheReqMeta, *, producer_event=None) -> None:
        ss = req.save_spec
        assert ss is not None
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
        # The lowest chunk handed to the GPU so far; [start, read_from) is unread.
        read_from = end
        failure = detail = None
        windows_run = stored_bytes = in_place_windows = 0
        pack_ms = put_ms = 0.0
        t_store0 = time.perf_counter()
        for window in windows:
            try:
                slots = pool.acquire("save", len(window))
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
            windows_run += 1
            run = pool.contiguous_view(slots) if self._in_place else None
            t_pack0 = time.perf_counter()
            try:
                if run is not None:
                    self._copy_in_place(
                        run,
                        window,
                        req.block_ids,
                        pack=True,
                        producer_event=producer_event,
                    )
                else:
                    with gpu_connector.track_save_source(req.save_operation):
                        gpu_connector.batched_from_gpu(
                            slots,
                            [index * chunk for index in window],
                            [(index + 1) * chunk for index in window],
                            block_ids=req.block_ids,
                            producer_event=producer_event,
                        )
            except BaseException:
                self._release_once_device_idle(pool, slots)
                raise
            pack_ms += (time.perf_counter() - t_pack0) * 1000
            # Both paths return after their final stream sync: no GPU work of
            # this window still reads the source or writes the slots.
            read_from = window.start
            if run is not None:
                in_place_windows += 1
                # Highest first, as the block GPU connector reports them.
                self._chunks_source_safe(req.save_operation, reversed(window))
            keys = self._window_keys(req, window)
            t_put0 = time.perf_counter()
            try:
                stored = client.put(
                    keys,
                    [slot.ptr for slot in slots],
                    [self._chunk_bytes] * len(slots),
                )
            except CallRefused as exc:
                # Refused before the server had the call: nothing was sent.
                pool.release(slots)
                failure, detail = "put_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                pool.quarantine(slots, reason="a put raised")
                self._distrust_store(f"a put raised {exc!r}")
                failure, detail = "put_raised", repr(exc)
                break
            put_ms += (time.perf_counter() - t_put0) * 1000
            # The server copied before it answered: the slots are idle.
            pool.release(slots)
            not_stored = stored.count(False)
            if not_stored:
                failure = "put_failed"
                detail = (
                    f"the server did not store {not_stored} of {len(stored)} "
                    "chunks (UMBP gives no reason: full, or a failed RPC)"
                )
                break
            stored_bytes += self._chunk_bytes * len(slots)
        store_ms = (time.perf_counter() - t_store0) * 1000
        if failure is not None:
            self._warn_rate_limited(
                f"save:{failure}",
                "UMBP offload: a save failed (%s: %s), req=%s; its "
                "remaining chunks are not stored",
                failure,
                detail,
                req.req_id,
            )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d skip=%d chunks=%d "
                "windows=%d in_place_windows=%d total_bytes=%d producer_fenced=%d "
                "pack_ms=%.2f put_ms=%.2f store_ms=%.2f total_ms=%.2f status=%s",
                self._rank,
                req.req_id,
                end * chunk,
                start * chunk,
                end - start,
                windows_run,
                in_place_windows,
                stored_bytes,
                int(producer_event is not None and windows_run > 0),
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
                "UMBP offload: cannot load req=%s hbm=%d lmc=%d chunk=%d: "
                "%s; re-prefill",
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
            # Checked before every get: a save thread may distrust the
            # server while this load runs, and another worker of this process
            # may mark it.
            refusal = self._load_refusal()
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
                read = client.get(
                    keys,
                    [slot.ptr for slot in slots],
                    [self._chunk_bytes] * len(slots),
                )
            except CallRefused as exc:
                pool.release(slots)
                failure, detail = "get_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                # A get that raised may still be written into these slots by
                # the server; it writes no object, so quarantining them is all
                # it takes, unlike a put's (`_distrust_store`).
                pool.quarantine(slots, reason="a get raised")
                failure, detail = "get_raised", repr(exc)
                break
            get_ms += (time.perf_counter() - t_get0) * 1000
            # All-or-nothing: a miss, an eviction or an object of another size
            # (another layout) all answer False.
            not_read = read.count(False)
            if not_read:
                pool.release(slots)
                failure = "get_failed"
                detail = (
                    f"the server did not return {not_read} of {len(read)} chunks "
                    "(UMBP gives no reason: evicted, or a failed RPC)"
                )
                break
            run = pool.contiguous_view(slots) if self._in_place else None
            t_unpack0 = time.perf_counter()
            try:
                if run is not None:
                    self._copy_in_place(run, window, req.block_ids, pack=False)
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
            # the raised put left behind. The load fails, so nothing it
            # unpacked is used.
            refusal = self._load_refusal()
            if refusal is not None:
                failure, detail = "store_distrusted", refusal
        if failure is not None:
            self._warn_rate_limited(
                f"load:{failure}",
                "UMBP offload: a load failed (%s: %s), req=%s; it is "
                "prefilled instead",
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
        stream = getattr(self._in_place_tls, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self._codec.device)
            self._in_place_tls.stream = stream
        return stream

    def _copy_in_place(
        self,
        run: torch.Tensor,
        window: range,
        block_ids: list[int],
        *,
        pack: bool,
        producer_event: Any = None,
    ) -> None:
        """Pack ``window``'s KV blocks into ``run``, or unpack it into them.

        Returns once the copy is done. A pack is ordered after
        ``producer_event`` like the staged pack, so the kernels that produced
        the KV finish first.
        """
        codec = self._codec
        chunk = self.chunk_size
        stream = self._in_place_stream()
        if producer_event is not None:
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
        copy = (
            codec.gpu_to_chunk_major_device_buffer_prepared
            if pack
            else codec.chunk_major_device_buffer_to_gpu_prepared
        )
        copy(run, owner, 0, stream=stream)
        stream.synchronize()

    # -- distrust ------------------------------------------------------------
    def _distrust_store(self, why: str) -> None:
        """Load nothing more from the server, and mark it in this process.

        ``why`` is a put that raised: the server may still be copying from the
        quarantined slot into an object, and whether that copy can land in an
        object the server has since given to another key is not something
        UMBP documents. This worker stops loading, for good, and marks the
        server in this process (:mod:`.taint`). The mark is this process's
        only: other worker processes and other engines on the server go on
        loading from it, and a restarted engine does too. A load needs every
        rank's chunks, so every load of this engine fails and its request
        prefills. Saves go on.
        """
        address = self._store_cfg.address
        taint.mark(address, why)
        if self._refuse_loads(why):
            logger.error(
                "UMBP offload: %s, and the server at %s may still be copying it "
                "into an object. This worker loads nothing more from that "
                "server and marks it in this process only; other processes and "
                "engines are not told. Every load of this engine prefills "
                "instead; restart the server, then the engine, to load again",
                why,
                address,
            )

    def _refuse_loads_from_marked_server(self, address: str, why: str) -> str:
        """Load nothing more from a server this process has marked."""
        if self._refuse_loads(why):
            logger.error(
                "UMBP offload: this process marked the server at %s (%s); this "
                "worker loads nothing more from it, and every load of this "
                "engine prefills instead",
                address,
                why,
            )
        return self._store_distrusted or why

    def _refuse_loads(self, why: str) -> bool:
        """Load nothing more from the server; True the first time only."""
        with self._stats_lock:
            if self._store_distrusted is not None:
                return False
            self._store_distrusted = why
        return True

    def _load_refusal(self) -> str | None:
        """Why this worker may not load from its server now; None when it may.

        Its own verdict holds for good (`_distrust_store`), and so does this
        process's mark of the server, which another worker of the process may
        have placed a moment ago.
        """
        if self._store_distrusted is not None:
            return self._store_distrusted
        address = self._store_cfg.address
        marked = taint.reason(address)
        if marked is not None:
            return self._refuse_loads_from_marked_server(address, marked)
        return None

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
                "UMBP offload: device synchronize failed after a "
                "transfer slot's copy or transfer failed"
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
        stats = client.stats()
        counts = stats["counts"]
        failures = ",".join(
            f"{name}x{count}" for name, count in sorted(stats["failures"].items())
        )
        logger.info(
            "[OFFLOAD-STORE-STATS] rank=%s puts=%d put_keys=%d put_bytes=%d "
            "gets=%d get_keys=%d get_bytes=%d failures=%s "
            "quarantined_slots=save:%d,load:%d",
            self._rank,
            counts.get("put_calls", 0),
            counts.get("put_keys", 0),
            counts.get("put_bytes", 0),
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


__all__ = ["UMBPStoreOffloadConnector"]
