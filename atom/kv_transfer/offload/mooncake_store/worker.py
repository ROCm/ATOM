# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker side of the native Mooncake Store offload.

Everything but the transport is the dense offload's: the executors and the
per-step producer fence (``start_load_kv``), the completion reports, the block
GPU connector and the dense codec. What changes is where a packed chunk goes:

* save: KV blocks --pack, copy--> a slot of the registered transfer pool
  --``batch_put_from``--> the Store owners;
* load: owners --``batch_get_into``--> a pool slot --copy, unpack--> KV blocks.

The pool is in the worker GPU's HBM: the NIC reads and writes GPU memory
directly, and no byte passes through host memory.

Completion protocol, per worker and operation:

* A save ends in exactly one store terminal (``_record_store_terminal``,
  directly or through ``_record_save_failure``). A failure once every GPU read
  of the save has finished -- a failed put, a pool with no slot left, the
  source read deadline -- also claims the source quiescent and reports the
  chunks it never read as source-safe, so the scheduler may retry at once and
  every PP stage reports the same per-chunk set. An exception from the GPU
  copy claims neither: it propagates, and ``_guard`` reports the failure
  without the quiescent claim.
* A load ends in exactly one of done or failed, all-or-nothing.
* A slot returns to the pool only once no GPU or NIC access can still reach
  it. One a Store transfer or a GPU copy may still reach is quarantined for
  good, and the pool's memory then outlives the worker (``pool.close``).
* A put that leaves its slot quarantined may also still write the owner
  memory the Store took back, inside another key's object: from then on the
  worker loads nothing from the Store, and marks it tainted so that no client
  of it does (``_distrust_store``, :mod:`.taint`). Before every get, and
  after a load's last one, it looks for that mark (``_load_refusal``).
"""

from __future__ import annotations

import logging
import os
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
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.config import (
    check_engine_compatibility,
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.keys import (
    DIGEST_BYTES,
    STORE_TAINT_KEY,
    chunk_group_ids,
    chunk_keys,
    probe_key,
    store_namespace,
)
from atom.kv_transfer.offload.mooncake_store.nic import requester_rdma_device
from atom.kv_transfer.offload.mooncake_store.pool import (
    Slot,
    SlotPoolExhausted,
    TransferSlotPool,
    pool_allocation_bytes,
)
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    monotonic_clock_id,
    require_monotonic_clock_id,
)
from atom.kv_transfer.offload.mooncake_store.taint import TaintMarker
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds between two summaries of a worker's Store traffic.
_STATS_LOG_INTERVAL_S = 60.0
# Seconds between two warnings of one kind; failures repeat while a Store or
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


def _require_rdma_environment() -> None:
    """Refuse Mooncake settings that silently undo one-NIC-per-worker reads.

    ``MC_NUM_QP_PER_EP`` is read by every Mooncake transfer engine of the
    process, the P/D one included. With Mooncake's default of 2, concurrent
    reads from several workers stalled for 30-60 s and then failed.
    ``MC_MS_AUTO_DISC=1`` makes the Store client ignore its RDMA device and
    take every NIC.
    """
    qp = os.environ.get("MC_NUM_QP_PER_EP", "").strip()
    if qp != "1":
        raise ValueError(
            "the Mooncake Store offload needs MC_NUM_QP_PER_EP=1 in every "
            f"Mooncake process, the decode's P/D engine included (got {qp or 'unset'})"
        )
    if os.environ.get("MC_MS_AUTO_DISC", "").strip() == "1":
        raise ValueError(
            "MC_MS_AUTO_DISC=1 makes the Store client use every NIC; unset it, "
            "the offload picks one NIC per worker"
        )


def _device_index(device: torch.device) -> int:
    if device.type == "cuda" and device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


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


class MooncakeStoreOffloadConnector(DenseOffloadConnector):
    """Dense offload worker whose tier is a Mooncake Store, reached directly."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self._store_cfg = parse_mooncake_store_config(kvc)
        check_engine_compatibility(self._store_cfg, config)
        require_monotonic_clock_id()
        self.chunk_size = self._store_cfg.chunk_tokens
        self._client: store_client.MooncakeStoreClient | None = None
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
        # Why this worker loads nothing from the Store any more, once it does
        # not: its own put (`_distrust_store`) or the Store's taint marker.
        self._store_distrusted: str | None = None
        # Places the taint marker in this worker's Store; and when a load
        # next looks for one there (`_load_refusal`).
        self._taint_marker: TaintMarker | None = None
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
        cfg = parse_mooncake_store_config(kvc)
        threads = envs.OFFLOAD_COPY_WORKERS + envs.OFFLOAD_LOAD_WORKERS
        return (
            pool_allocation_bytes(cfg.save_pool_bytes, cfg.load_pool_bytes)
            + threads * _staging_buffer_bound()
        )

    def close(self) -> None:
        """Join the executors, then release the GPU connector, pool and client."""
        super().close()
        marker, self._taint_marker = self._taint_marker, None
        # A retry of the taint marker still inside a Store call keeps the
        # client open: closing it would pull it from under the call.
        client_idle = marker is None or marker.close()
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
            logger.warning(
                "Mooncake Store offload: the taint marker is still being placed; "
                "the Store client stays open"
            )

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        rank, world = pp_aware_rank_and_world(self._config, _tp_group())
        self._rank = rank
        self._world = world
        # Scheduler blocks, threaded from the model runner. MLA stores its KV
        # token-major, so the codec cannot infer the count from shape[0].
        self._codec = DenseKVByteCodec(
            kv_caches,
            num_blocks=num_blocks,
            permit_per_request_state=self._permit_per_request_state,
        )
        cfg = self._store_cfg
        namespace = store_namespace(self._config, self.chunk_size)
        device = self._codec.device
        nic = None
        if cfg.protocol == "rdma":
            _require_rdma_environment()
            nic = requester_rdma_device(_device_index(device), cfg)
        store_pool = cfg.pool_of(nic)
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
            client = store_client.MooncakeStoreClient(
                local_hostname=cfg.local_hostname,
                metadata_server=store_pool.metadata,
                master_server_addr=store_pool.master,
                protocol=cfg.protocol,
                rdma_devices=nic or "",
            )
            # Before the probe, whose put may leave a write behind.
            self._taint_marker = TaintMarker(client)
            pool = TransferSlotPool(
                device=device,
                chunk_bytes=gpu_connector.gpu_staging_chunk_bytes,
                save_bytes=cfg.save_pool_bytes,
                load_bytes=cfg.load_pool_bytes,
                client=client,
            )
            self._probe_store(client, pool, namespace, rank)
            tainted, detail = self._look_for_taint(client)
            if tainted is None:
                raise RuntimeError(
                    "Mooncake Store startup: cannot tell whether the Store at "
                    f"{store_pool.master} carries the taint marker {STORE_TAINT_KEY} "
                    f"({detail})"
                )
        except BaseException:
            marker, self._taint_marker = self._taint_marker, None
            client_idle = marker is None or marker.close()
            if pool is not None:
                pool.close()
            if client is not None and client_idle:
                client.close()
            gpu_connector.close()
            raise
        if tainted:
            self._refuse_loads_from_marked_store(store_pool.master)
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
            "Mooncake Store offload worker rank=%d world=%d: namespace=%s nic=%s "
            "master=%s bytes_per_block=%d chunk=%d chunk_bytes=%d "
            "pool=%s save_slots=%d load_slots=%d gpu_staging_buffer_bytes=%d "
            "save=%s load=%s save_workers=%d load_workers=%d",
            rank,
            world,
            namespace,
            nic or cfg.protocol,
            store_pool.master,
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
        client: store_client.MooncakeStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
    ) -> None:
        """Send one chunk of random bytes through the Store and compare it back.

        An unregistered pool, an unreachable owner or a misrouted NIC
        otherwise shows up only as every lookup missing. The probe's slots are
        settled as a save's and a load's are, so one its failed put or get
        may still reach stays quarantined when startup gives up.
        """
        key = probe_key(namespace, rank, uuid.uuid4().hex)
        nbytes = pool.chunk_bytes
        [source] = pool.acquire("save", 1)
        [target] = pool.acquire("load", 1)
        # The probe's slots no transfer may still reach.
        leased = [source, target]
        try:
            source.tensor.copy_(
                torch.randint(
                    0, 256, (nbytes,), dtype=torch.uint8, device=source.tensor.device
                )
            )
            target.tensor.zero_()
            _synchronize(pool.device)
            put_rc = self._probe_transfer(
                pool,
                leased,
                source,
                lambda: client.put([key], [source.ptr], [nbytes]),
                # A put that may still write marks the Store before startup
                # gives up.
                on_unsettled=self._distrust_store,
            )
            if put_rc != 0:
                raise RuntimeError(
                    f"Mooncake Store startup probe: put of {nbytes} bytes to "
                    f"{client.master_server_addr} failed "
                    f"({store_client.describe(put_rc)})"
                )
            try:
                [present] = client.exists([key])
                if present != 1:
                    raise RuntimeError(
                        "Mooncake Store startup probe: a chunk just put is not in "
                        f"the Store ({store_client.describe(present)}); is an "
                        "owner mounted and is the pool registered?"
                    )
                get_rc = self._probe_transfer(
                    pool,
                    leased,
                    target,
                    lambda: client.get([key], [target.ptr], [nbytes]),
                )
                if get_rc != nbytes:
                    raise RuntimeError(
                        "Mooncake Store startup probe: get returned "
                        f"{store_client.describe(get_rc)} for a {nbytes}-byte chunk"
                    )
                _synchronize(pool.device)
                if not torch.equal(source.tensor, target.tensor):
                    raise RuntimeError(
                        "Mooncake Store startup probe: the chunk read back differs "
                        "from the one written"
                    )
            finally:
                removed = client.remove(key, force=True)
                if removed != 0:
                    logger.warning(
                        "Mooncake Store startup probe: could not remove %s (%s); "
                        "it stays until evicted",
                        key,
                        store_client.describe(removed),
                    )
        except BaseException:
            self._release_once_device_idle(pool, leased)
            raise
        pool.release(leased)

    @staticmethod
    def _probe_transfer(
        pool: TransferSlotPool,
        leased: list[Slot],
        slot: Slot,
        call: Any,
        *,
        on_unsettled: Any = None,
    ) -> int:
        """Run the probe's put or get on ``slot``, quarantining it if unsettled.

        A quarantined slot leaves ``leased``, the probe's slots still to release,
        and is reported to ``on_unsettled``, with why, when one is given.
        """
        clock = store_client.CallClock()
        try:
            [code] = call()
        except store_client.CallRefused:
            raise  # nothing was posted: the slot is released with the rest
        except BaseException as exc:
            leased.remove(slot)
            pool.quarantine([slot], reason="a startup probe transfer raised")
            if on_unsettled is not None:
                on_unsettled(f"the startup probe's transfer raised {exc!r}")
            raise
        if not store_client.buffer_settled(code, clock.seconds()):
            why = (
                f"a startup probe transfer gave up after {clock.seconds():.0f}s "
                "with its RDMA work possibly still posted"
            )
            leased.remove(slot)
            pool.quarantine([slot], reason=why)
            if on_unsettled is not None:
                on_unsettled(why)
        return code

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if isinstance(metadata, LMCacheOffloadMetadata):
            # The source read deadline of a save counts from its dispatch, not
            # from whenever a save thread gets to it; see
            # `_source_read_deadline`. Placed here, once per save.
            received_at = time.monotonic()
            for req in metadata.requests:
                if req.save_spec is not None:
                    req._mooncake_store_dispatched_at = self._dispatch_time(
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
                "Mooncake Store offload: req=%s was dispatched on another clock "
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
        dispatched_at = getattr(req, "_mooncake_store_dispatched_at", None)
        if dispatched_at is None:
            return None
        return float(dispatched_at) + self._source_read_window_s

    # -- copy daemon threads ---------------------------------------------
    def _registered(
        self,
    ) -> tuple[TransferSlotPool, store_client.MooncakeStoreClient, BlockGPUConnector]:
        if self._pool is None or self._client is None or self._gpu_connector is None:
            raise RuntimeError("Mooncake Store offload: KV caches are not registered")
        return self._pool, self._client, self._gpu_connector

    def _window_keys(self, req: LMCacheReqMeta, window: range) -> list[str]:
        """This rank's Store keys of the chunks of ``req`` in ``window``."""
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
            group_ids = chunk_group_ids(
                self._namespace, req.chunk_hashes, window.start, window.stop
            )
            t_put0 = time.perf_counter()
            clock = store_client.CallClock()
            try:
                codes = client.put(
                    keys,
                    [slot.ptr for slot in slots],
                    [self._chunk_bytes] * len(slots),
                    group_ids=group_ids,
                )
            except store_client.CallRefused as exc:
                # Refused before Mooncake had the call: nothing was posted.
                pool.release(slots)
                failure, detail = "put_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                pool.quarantine(slots, reason="a put raised")
                self._distrust_store(f"a put raised {exc!r}")
                failure, detail = "put_raised", repr(exc)
                break
            put_ms += (time.perf_counter() - t_put0) * 1000
            call_seconds = clock.seconds()
            unsettled = self._settle(pool, slots, codes, call_seconds)
            if unsettled:
                self._distrust_store(
                    f"a put gave up on {unsettled} of {len(slots)} chunks after "
                    f"{call_seconds:.0f}s"
                )
            bad = [code for code in codes if code != 0]
            if bad:
                failure = "put_failed"
                detail = (
                    f"{len(bad)} of {len(codes)} chunks, first "
                    f"{store_client.describe(bad[0])}"
                )
                break
            stored_bytes += self._chunk_bytes * len(slots)
        store_ms = (time.perf_counter() - t_store0) * 1000
        if failure is not None:
            self._warn_rate_limited(
                f"save:{failure}",
                "Mooncake Store offload: a save failed (%s: %s), req=%s; its "
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
                "Mooncake Store offload: cannot load req=%s hbm=%d lmc=%d chunk=%d: "
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
            # Checked before every get: a save thread may give up a put
            # while this load runs, and another client may mark the Store.
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
            clock = store_client.CallClock()
            try:
                codes = client.get(
                    keys,
                    [slot.ptr for slot in slots],
                    [self._chunk_bytes] * len(slots),
                )
            except store_client.CallRefused as exc:
                pool.release(slots)
                failure, detail = "get_refused", str(exc)
                break
            except Exception as exc:  # noqa: BLE001  # reported below
                # A get's late RDMA writes only these slots: it reads the
                # owners' memory and writes none of it, so quarantining them
                # is all it takes, unlike a put's (`_distrust_store`).
                pool.quarantine(slots, reason="a get raised")
                failure, detail = "get_raised", repr(exc)
                break
            get_ms += (time.perf_counter() - t_get0) * 1000
            # All-or-nothing: an object of another size is another layout.
            bad = [code for code in codes if code != self._chunk_bytes]
            if bad:
                self._settle(pool, slots, codes, clock.seconds())
                failure = "get_failed"
                detail = f"{len(bad)} of {len(codes)} chunks, first " + (
                    store_client.describe(bad[0])
                    if bad[0] < 0
                    else f"{bad[0]} bytes where {self._chunk_bytes} were expected"
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
            # the given-up put wrote. The load fails, so nothing it unpacked
            # is used.
            refusal = self._load_refusal(client)
            if refusal is not None:
                failure, detail = "store_distrusted", refusal
        if failure is not None:
            self._warn_rate_limited(
                f"load:{failure}",
                "Mooncake Store offload: a load failed (%s: %s), req=%s; it is "
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

        # -- slot settlement -----------------------------------------------

    @staticmethod
    def _settle(
        pool: TransferSlotPool,
        slots: list[Slot],
        codes: list[int],
        call_seconds: float,
    ) -> int:
        """Release each slot whose Store transfer is over; quarantine the rest.

        ``call_seconds`` is how long the call that returned ``codes`` took: a
        failure is unsettled only after Mooncake's batch wait ran out. Returns
        how many slots were quarantined.
        """
        settled: list[Slot] = []
        unsettled: list[Slot] = []
        for slot, code in zip(slots, codes, strict=True):
            if store_client.buffer_settled(code, call_seconds):
                settled.append(slot)
            else:
                unsettled.append(slot)
        pool.release(settled)
        if unsettled:
            pool.quarantine(
                unsettled,
                reason=(
                    f"a Store transfer gave up after {call_seconds:.0f}s with "
                    "its RDMA work possibly still posted"
                ),
            )
        return len(unsettled)

    def _distrust_store(self, why: str) -> None:
        """Load nothing more from the Store, and mark it so no client does.

        ``why`` is a put that may still write owner memory the Store has since
        given to other objects (see :mod:`.taint`). Quarantining the slot keeps
        this worker's memory out of that write's reach; the owners' memory is
        out of every client's, so this worker stops loading, for good, and
        places the Store's taint marker, which every other client of it checks.
        A load needs every rank's chunks, so every load of the engine fails and
        its request prefills. Saves go on.
        """
        if self._refuse_loads(why):
            logger.error(
                "Mooncake Store offload: %s, and its RDMA writes may still land "
                "in owner memory the Store has since given to other objects. "
                "This worker loads nothing more from the Store and marks it "
                "tainted (%s), so no client of it loads from it until it "
                "restarts; every load of this engine prefills instead",
                why,
                STORE_TAINT_KEY,
            )
        marker = self._taint_marker
        if marker is not None:
            marker.place()

    def _refuse_loads_from_marked_store(self, master: str) -> str:
        """Load nothing more from a Store that carries the taint marker."""
        why = f"the Store carries the taint marker {STORE_TAINT_KEY}"
        if self._refuse_loads(why):
            logger.error(
                "Mooncake Store offload: the Store at %s carries the taint "
                "marker %s: a put another client gave up on may still write "
                "memory the Store has since given to other objects. This worker "
                "loads nothing more from it, and every load of this engine "
                "prefills instead; restart the Store, then this engine, to load "
                "from it again",
                master,
                STORE_TAINT_KEY,
            )
        return self._store_distrusted or why

    def _refuse_loads(self, why: str) -> bool:
        """Load nothing more from the Store; True the first time only."""
        with self._stats_lock:
            if self._store_distrusted is not None:
                return False
            self._store_distrusted = why
        return True

    def _load_refusal(self, client: store_client.MooncakeStoreClient) -> str | None:
        """Why this worker may not load from its Store now; None when it may.

        Its own verdict holds for good (`_distrust_store`). The Store's taint
        marker is looked for right before every get and after a load's last
        get, never trusted from an earlier look: another client may have
        marked the Store a moment ago. Once seen it holds for good too, and a
        look that gets no answer refuses this load only.
        """
        if self._store_distrusted is not None:
            return self._store_distrusted
        tainted, detail = self._look_for_taint(client)
        if tainted is None:
            return f"cannot tell whether the Store carries the taint marker ({detail})"
        if tainted:
            return self._refuse_loads_from_marked_store(client.master_server_addr)
        return None

    @staticmethod
    def _look_for_taint(
        client: store_client.MooncakeStoreClient,
    ) -> tuple[bool | None, str]:
        """Whether the Store carries the taint marker; None, and why, if unsaid."""
        try:
            [code] = client.exists([STORE_TAINT_KEY])
        except Exception as exc:  # noqa: BLE001  # an unanswered look refuses
            return None, repr(exc)
        if code < 0:
            return None, store_client.describe(code)
        return code == 1, ""

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
                "Mooncake Store offload: device synchronize failed after a "
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


__all__ = ["MooncakeStoreOffloadConnector"]
