# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses RDMA-based zero-copy transfers via the MoRIIO library for efficient
KV cache migration between producer (prefill) and consumer (decode) nodes.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgpack
import msgspec
import numpy as np
import zmq
from aiter.dist.parallel_state import get_dp_group, get_tp_group

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.moriio.moriio_common import (
    _MORIIO_AVAILABLE,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    get_port_offset,
)
from atom.kv_transfer.disaggregation.moriio.moriio_engine import MoRIIOWrapper
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    EngineId,
    KVConnectorOutput,
    ReqId,
    ReqMeta,
    TransferId,
)
from atom.kv_transfer.disaggregation.utils import chunk_tensor_for_rdma
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_port,
    make_zmq_path,
    zmq_socket_ctx,
)
from atom.utils.network import get_ip

if _MORIIO_AVAILABLE:
    from mori.io import (
        BackendType,
        IOEngine,
        IOEngineConfig,
        PollCqMode,
        RdmaBackendConfig,
    )

logger = logging.getLogger("atom")


# ===================================================================
# MoRIIOConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MoRIIOConnector(KVConnectorBase):
    """Worker-side KV cache connector for disaggregated P/D inference.

    Each tensor-parallel worker instantiates one ``MoRIIOConnector``.  It is
    responsible for:

    1. Registering local KV cache tensors for RDMA access.
    2. Performing handshakes with remote engines to exchange memory metadata.
    3. Moving KV cache blocks from the producer to the consumer, in one of two
       modes chosen by the consumer's ``moriio_mode``:

       - ``write`` (default): the consumer sends the producer its block ids
         and the producer pushes the KV with RDMA writes, then reports back.
       - ``read``: the consumer pulls the KV with RDMA reads.

    4. Tracking transfer completion and notifying the peer when done.
    5. Periodically pinging the proxy for service discovery.
    """

    def __init__(self, config: Config) -> None:
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size

        kv_transfer_config = config.kv_transfer_config
        self.local_ip = get_ip()

        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.transfer_mode = self._transfer_mode_from(kv_transfer_config)
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.base_handshake_port = kv_transfer_config.get(
            "handshake_port", MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
        )

        # Compute unique side-channel port for this (dp, tp) rank
        handshake_port = self.base_handshake_port
        # tp_size must be passed: it strides the DP ranks apart, and the
        # default of 1 would put every DP rank on the same port set.
        self.side_channel_port = handshake_port + get_port_offset(
            self.dp_rank, self.tp_rank, self.tp_size
        )
        self.engine_id = f"{self.local_ip}:{handshake_port}"

        # Remote metadata caches
        self.layer_name_to_remote_kv_cache_metadata: dict[str, dict[str, list[Any]]] = (
            {}
        )
        self.remote_moriio_metadata: dict[EngineId, MoRIIOAgentMetadata] = {}
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # RDMA engine and wrapper
        self.moriio_engine = IOEngine(
            f"atom:ip:{self.local_ip}+tp:{self.tp_rank}+dp:{self.dp_rank}",
            IOEngineConfig(host=str(self.local_ip), port=0),
        )
        self.moriio_wrapper = MoRIIOWrapper(moriio_engine=self.moriio_engine)

        qp_per_transfer = kv_transfer_config.get("qp_per_transfer", 4)
        post_batch_size = kv_transfer_config.get("post_batch_size", -1)
        num_worker_threads = kv_transfer_config.get("num_worker_threads", 4)
        poll_mode = PollCqMode.POLLING
        enable_notification = kv_transfer_config.get("enable_notification", False)

        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
            enable_notification,
        )
        rdma_cfg.max_send_wr = kv_transfer_config.get("max_send_wr", 0)
        rdma_cfg.max_cqe_num = kv_transfer_config.get("max_cqe_num", 0)
        rdma_cfg.max_msg_sge = kv_transfer_config.get("max_msg_sge", 0)
        logger.info(
            "RdmaBackendConfig: qp_per_transfer=%d, workers=%d, "
            "poll_mode=%s, notification=%s, transfer_mode=%s",
            qp_per_transfer,
            num_worker_threads,
            poll_mode.name,
            enable_notification,
            self.transfer_mode,
        )
        self.moriio_wrapper.set_backend_type(BackendType.RDMA, rdma_cfg)

        # Per-layer local metadata (populated in register_kv_caches)
        self.layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]] = {}
        self.local_kv_cache_metadata: list[bytes] = []
        self.remote_kv_cache_metadata: list[bytes] = []
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.kv_caches: dict[str, Any] | None = None
        self.kv_cache_block_size: int = config.kv_cache_block_size
        self.blocks_per_chunk: int | None = None
        self.num_k_chunks: int = 0

        # Session cache: remote_engine_id -> list[session]
        self._built_sessions: defaultdict[str, list] = defaultdict(list)

        # Handshake management
        self.zmq_context = zmq.Context()
        self._handshake_lock = threading.RLock()
        self._handshake_timeout_s = float(
            os.environ.get("ATOM_MORIIO_HANDSHAKE_TIMEOUT_S", "300")
        )
        self._handshake_futures: dict[EngineId, Future[set[str]]] = {}
        self._remote_agents: dict[EngineId, set[str]] = {}
        self._ready_requests: queue.Queue[tuple[ReqId, ReqMeta]] = queue.Queue()
        # MoRIIO is not guaranteed to be thread-safe, limit to 1 worker.
        self._handshake_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="atom-moriio-handshake-initiator",
        )

        # In-flight receive transfers
        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}

        # Opt-in correctness probe. A transfer that reports success only says
        # the RDMA verbs retired, not that every layer's KV arrived, and decode
        # cannot tell the difference -- it just attends over whatever is in the
        # block and degenerates. Digesting each layer's destination before and
        # after the transfer turns that silent case into a log line.
        self._verify_kv = os.environ.get("ATOM_MORIIO_VERIFY", "") not in ("", "0")
        self._verify_before: dict[ReqId, dict[str, int]] = {}
        self._verify_rows: dict[ReqId, Any] = {}
        self._logged_layout = False

        # Requests rejected before a single transfer was issued. They have no
        # status to poll, so get_finished has to report them from here.
        self._failed_before_read: set[ReqId] = set()

        # Serial WRITE mode, producer side. The listener thread only queues
        # requests; the worker thread, which owns every RDMA call, drains them,
        # waits out each consumer's handshake, and tracks the writes in flight.
        self._write_requests: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending_writes: list[tuple[dict[str, Any], float]] = []
        self._write_handshakes: dict[EngineId, Future[set[str]]] = {}
        self._sending_writes: dict[int, tuple[list[Any], dict[str, Any]]] = {}

        # Serial WRITE mode, consumer side: requests waiting on the producer's
        # report, and the reports the listener thread has collected.
        self._awaiting_write: dict[ReqId, float] = {}
        self._write_reports: dict[ReqId, bool] = {}
        self._write_reports_lock = threading.Lock()
        self._write_timeout_s = float(
            os.environ.get("ATOM_MORIIO_WRITE_TIMEOUT_S", "600")
        )

        # Completed send-side transfers (populated by handshake listener)
        self.done_sending: set[int] = set()

        # Transfer ID mapping (worker side)
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        """Register all KV cache tensors for RDMA and start the handshake listener.

        Must be called after model loading and KV cache allocation, before any
        transfers can occur.

        Each K (and V, when present) tensor is split into block-aligned
        chunks of < 2 GiB via :func:`chunk_tensor_for_rdma` and each
        chunk is registered independently with ``ibv_reg_mr``.  Per-layer
        metadata list layout:
        ``[k_chunk0, k_chunk1, ..., v_chunk0, v_chunk1, ...]``.
        ``self.num_k_chunks`` marks the K/V boundary.
        """
        self.kv_caches = kv_caches
        cache_tensor = None

        for layer_name, kv_cache in kv_caches.items():
            cache_tensor = kv_cache.k_cache
            v_cache = kv_cache.v_cache
            is_mla = v_cache is None

            if self.kv_cache_shape is None:
                self.kv_cache_shape = cache_tensor.shape

            device_id = (
                cache_tensor.device.index
                if cache_tensor.device.index is not None
                else -1
            )

            # MLA: dim 0 = num_blocks * block_size tokens, so block_size_in_dim0
            # equals the KV cache block_size (typically 16).
            # Non-MLA: dim 0 = num_blocks directly, so 1.
            bsd0 = self.kv_cache_block_size if is_mla else 1
            k_chunks, bpc = chunk_tensor_for_rdma(cache_tensor, bsd0)

            if self.blocks_per_chunk is None:
                self.blocks_per_chunk = bpc

            meta_list: list[bytes] = []
            for ptr, size in k_chunks:
                meta_list.append(
                    self.moriio_wrapper.register_local_buffer(ptr, size, device_id)
                )
            self.num_k_chunks = len(k_chunks)

            if not is_mla:
                v_device_id = (
                    v_cache.device.index if v_cache.device.index is not None else -1
                )
                v_chunks, _ = chunk_tensor_for_rdma(v_cache, 1)
                for ptr, size in v_chunks:
                    meta_list.append(
                        self.moriio_wrapper.register_local_buffer(
                            ptr, size, v_device_id
                        )
                    )

            self.layer_name_to_local_kv_cache_metadata[layer_name] = meta_list

        logger.info(
            "RDMA chunked registration: %d K chunks + %d V chunks, " "%d blocks/chunk",
            self.num_k_chunks,
            len(meta_list) - self.num_k_chunks,
            self.blocks_per_chunk,
        )

        # Extract block geometry from the last registered tensor
        is_mla = len(cache_tensor.shape) == 3
        self.block_len = self.kv_cache_block_size
        if is_mla:
            self.num_blocks = cache_tensor.shape[0] // self.block_len
        else:
            self.num_blocks = cache_tensor.shape[0]
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=None,
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name="aiter",
        )
        ready_event = threading.Event()
        self._handshake_listener_thread = threading.Thread(
            target=self._handshake_listener,
            args=(
                metadata,
                ready_event,
                self.side_channel_port,
                self.tp_rank,
                self.dp_rank,
                self.layer_name_to_local_kv_cache_metadata,
            ),
            daemon=True,
            name="moriio-handshake-listener",
        )
        self._handshake_listener_thread.start()

    @staticmethod
    def _transfer_mode_from(kv_transfer_config: dict, environ=os.environ) -> str:
        """``"write"`` or ``"read"``: which side posts the RDMA verbs.

        Only the consumer's mode matters: it either pulls with reads or asks
        for a push, and the producer serves both. Write is the default because
        on ionic AINIC (fw 1.117.5-a-56) an RDMA READ into GPU memory reports
        SUCCESS without placing any data, while writes land.
        """
        mode = environ.get("ATOM_MORIIO_MODE") or kv_transfer_config.get(
            "moriio_mode", "write"
        )
        mode = str(mode).lower()
        if mode not in ("read", "write"):
            raise ValueError(f"moriio_mode must be 'read' or 'write', got {mode!r}")
        return mode

    @staticmethod
    def _engine_name_with_dp(engine_name: str, dp_rank: int) -> str:
        """Build a unique engine identifier that includes the DP rank."""
        return f"{engine_name}_dp{dp_rank}"

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate RDMA reads for all pending receive requests.

        Called by the worker process each step.  For each request in
        ``metadata.reqs_to_recv``, this method either starts a handshake
        with the remote engine (if first contact) or issues RDMA reads
        immediately.
        """
        if self.is_producer:
            return

        if metadata is not None and metadata.reqs_to_recv:
            logger.debug("Starting KV load for %d requests", len(metadata.reqs_to_recv))

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        if self.transfer_mode == "write":
            for req_id, meta in metadata.reqs_to_recv.items():
                self._request_write(req_id, meta)
            return

        deferred = 0

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = f"{meta.remote_host}:{meta.remote_handshake_port}"
            meta.remote_engine_id = remote_engine_id
            dp0_id = self._engine_name_with_dp(remote_engine_id, 0)

            if dp0_id not in self._remote_agents:
                with self._handshake_lock:
                    # _remote_agents is keyed by the dp-suffixed id that the
                    # handshake callback registers, so re-check that one.
                    if dp0_id not in self._remote_agents:
                        self._initiate_background_handshake(
                            req_id, remote_engine_id, meta
                        )
                        deferred += 1
                        continue

            self._issue_read_for_req(req_id, meta)

        # Drain every deferred request, not just the first off the queue: the
        # rest would sit waiting on KV that was never asked for. One entry is
        # queued per deferral, so the count is exact.
        deadline = time.monotonic() + self._handshake_timeout_s
        while deferred:
            try:
                entry = self._ready_requests.get(timeout=0.01)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    logger.error(
                        "Timed out after %.0fs with %d handshake(s) pending; "
                        "those requests have no KV read in flight",
                        self._handshake_timeout_s,
                        deferred,
                    )
                    break
                continue
            self._issue_read_for_req(*entry)
            deferred -= 1

    def _issue_read_for_req(self, req_id: str, meta: ReqMeta) -> None:
        """Issue RDMA reads for a single request."""
        logger.debug(
            "Issuing RDMA read for req %s from engine %s (tp_rank=%d, remote_dp_rank=%d)",
            req_id,
            meta.remote_engine_id,
            self.tp_rank,
            meta.remote_dp_rank,
        )
        if self._rejects_transfer_subset(req_id, meta):
            self._failed_before_read.add(req_id)
            return
        try:
            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote_engine_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_host=meta.remote_host,
                remote_handshake_port=meta.remote_handshake_port,
                remote_dp_rank=meta.remote_dp_rank,
                remote_tp_size=int(meta.tp_size or 0),
            )
        except ValueError:
            # start_load_kv runs on the worker's busy_loop, so letting this
            # escape takes down the rank and with it every other request on
            # it. Failing just this one lets the scheduler retry or abort it.
            logger.exception("req %s: refusing to read KV", req_id)
            self._failed_before_read.add(req_id)

    @staticmethod
    def _rejects_transfer_subset(req_id: ReqId, meta: ReqMeta) -> bool:
        """Whether the request asks for a subset neither direction implements.

        Both paths move every advertised block, paired one to one. A request to
        skip already-computed blocks is safe to over-serve; a source stride is
        not, because the pairing would put blocks in the wrong place.
        """
        if getattr(meta, "num_computed_blocks", 0):
            logger.warning(
                "req %s asks to skip %d already-computed blocks, which this "
                "connector does not implement; transferring all blocks",
                req_id,
                meta.num_computed_blocks,
            )
        if getattr(meta, "src_block_skip_factor", 1) != 1:
            logger.error(
                "req %s: src_block_skip_factor=%s is not implemented by the "
                "MoRIIO connector; blocks would be paired one to one",
                req_id,
                meta.src_block_skip_factor,
            )
            return True
        return False

    def _request_write(self, req_id: ReqId, meta: ReqMeta) -> None:
        """Ask the producer to push this request's KV into our blocks.

        The consumer half of serial WRITE mode. The producer has already run
        the prefill and holds the source blocks until it reports back, so all
        that is left is to tell it where to write.
        """
        if self._rejects_transfer_subset(req_id, meta):
            self._failed_before_read.add(req_id)
            return
        dst = [int(b) for b in meta.local_block_ids]
        src = [int(b) for b in (meta.remote_block_ids or [])]
        # The producer's table also covers the token it generated, which we
        # never allocate for, so a longer source is normal. A shorter one would
        # leave the tail of the prompt with nothing to fill it.
        if len(src) < len(dst):
            logger.error(
                "req %s: %d local blocks but only %d remote blocks; the tail of "
                "the prompt's KV has no source",
                req_id,
                len(dst),
                len(src),
            )
            self._failed_before_read.add(req_id)
            return

        if self._verify_kv:
            first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
            is_mla = self.kv_caches[first_layer].v_cache is None
            rows = self._verify_rows_for(dst, is_mla)
            self._verify_rows[req_id] = rows
            self._verify_before[req_id] = self._verify_digest(rows)

        remote_tp_size = int(meta.tp_size or 0) or self.tp_size
        port = int(meta.remote_handshake_port) + get_port_offset(
            int(meta.remote_dp_rank or 0), self.tp_rank, remote_tp_size
        )
        request = {
            "transfer_id": self.request_id_to_transfer_id.get(req_id, meta.transfer_id),
            "decode_req_id": req_id,
            "decode_host": self.local_ip,
            "decode_handshake_port": self.base_handshake_port,
            "decode_dp_rank": self.dp_rank,
            "decode_tp_size": self.tp_size,
            "dst_block_ids": dst,
            "src_block_ids": src[: len(dst)],
        }
        try:
            self.moriio_wrapper.send_message(
                [MoRIIOConstants.WRITE_REQ, msgpack.dumps(request)],
                meta.remote_host,
                port,
            )
        except Exception:
            # Same reasoning as the read path: this runs on the busy_loop.
            logger.exception("req %s: could not ask the producer to push KV", req_id)
            self._verify_rows.pop(req_id, None)
            self._verify_before.pop(req_id, None)
            self._failed_before_read.add(req_id)
            return
        self._awaiting_write[req_id] = time.monotonic()

    def merge_contiguous_blocks(
        self,
        offsets_local: list[int],
        offsets_remote: list[int],
        sizes: list[int],
        assume_sorted: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        n = len(offsets_local)
        if n == 0:
            return [], [], []
        if not (n == len(offsets_remote) == len(sizes)):
            raise ValueError("Input list lengths mismatch")
        local_arr = np.fromiter(offsets_local, dtype=np.int64, count=n)
        remote_arr = np.fromiter(offsets_remote, dtype=np.int64, count=n)
        sizes_arr = np.fromiter(sizes, dtype=np.int64, count=n)

        if assume_sorted:
            local_sorted = local_arr
            remote_sorted = remote_arr
            sizes_sorted = sizes_arr
        else:
            if np.all(local_arr[:-1] <= local_arr[1:]):
                local_sorted = local_arr
                remote_sorted = remote_arr
                sizes_sorted = sizes_arr
            else:
                sort_idx = np.argsort(local_arr, kind="stable")
                local_sorted = local_arr[sort_idx]
                remote_sorted = remote_arr[sort_idx]
                sizes_sorted = sizes_arr[sort_idx]

        if n == 1:
            return (
                [int(local_sorted[0])],
                [int(remote_sorted[0])],
                [int(sizes_sorted[0])],
            )

        diff_local = local_sorted[1:] - local_sorted[:-1]
        diff_remote = remote_sorted[1:] - remote_sorted[:-1]
        prev_size = sizes_sorted[:-1]

        contiguous = (diff_local == prev_size) & (diff_remote == prev_size)

        if not contiguous.any():
            return local_sorted.tolist(), remote_sorted.tolist(), sizes_sorted.tolist()

        if contiguous.all():
            total_size = int(sizes_sorted.sum())
            return [int(local_sorted[0])], [int(remote_sorted[0])], [total_size]

        break_positions = np.flatnonzero(~contiguous) + 1
        segment_starts = np.concatenate(([0], break_positions))
        segment_ends = np.concatenate((break_positions, [n]))

        seg_count = len(segment_starts)
        merged_local = [0] * seg_count
        merged_remote = [0] * seg_count
        merged_sizes = [0] * seg_count

        for si in range(seg_count):
            s = segment_starts[si]
            e = segment_ends[si]
            merged_local[si] = int(local_sorted[s])
            merged_remote[si] = int(remote_sorted[s])

            merged_sizes[si] = int(
                local_sorted[e - 1] + sizes_sorted[e - 1] - local_sorted[s]
            )

        return merged_local, merged_remote, merged_sizes

    def _compute_block_transfer_offsets(
        self,
        layer_name: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_moriio_meta: MoRIIOAgentMetadata,
    ) -> tuple[list[int], list[int], list[int]]:
        """Compute per-block byte offsets within a single registered region.

        With chunked registration every region's base address corresponds
        to block 0 of that chunk.  The byte offset of block ``b`` is
        ``b * per_block_bytes``.  For cross-chunk transfers, callers
        must convert to chunk-relative block IDs first (see
        ``_read_blocks``).

        This method is kept for test compatibility; the hot path in
        ``_read_blocks`` does its own chunk-aware grouping.
        """
        del remote_moriio_meta
        assert self.kv_cache_shape is not None, "KV caches shape not initialized"
        is_mla = len(self.kv_cache_shape) == 3
        cache_tensor = self.kv_caches[layer_name].k_cache
        sz = cache_tensor.element_size()
        if is_mla:
            per_block_bytes = self.kv_cache_block_size * cache_tensor.stride(0) * sz
        else:
            per_block_bytes = cache_tensor.stride(0) * sz

        n = len(local_block_ids)
        offset_local = [lb * per_block_bytes for lb in local_block_ids]
        offset_remote = [rb * per_block_bytes for rb in remote_block_ids]
        sizes = [per_block_bytes] * n
        return offset_local, offset_remote, sizes

    def _get_or_build_sessions(
        self, remote_engine_id: str
    ) -> tuple[list[tuple[dict, dict]], MoRIIOAgentMetadata]:
        """Return cached RDMA sessions for the remote engine, building if needed.

        With chunked registration, local block ``b`` and remote block ``r``
        may reside in different chunks, so we build an NxN session grid
        for K chunks and another NxN grid for V chunks.

        Returns ``(per_layer_sessions, remote_meta)`` where each layer
        entry is ``(k_sessions_dict, v_sessions_dict)`` keyed by
        ``(local_chunk_idx, remote_chunk_idx)``.
        """
        if remote_engine_id not in self._built_sessions:
            nk = self.num_k_chunks
            per_layer_sessions: list[tuple[dict, dict]] = []
            for ln, local_metas in self.layer_name_to_local_kv_cache_metadata.items():
                remote_metas = self.layer_name_to_remote_kv_cache_metadata[
                    remote_engine_id
                ][ln]
                assert len(local_metas) == len(remote_metas), (
                    f"layer {ln}: local has {len(local_metas)} descs, "
                    f"remote has {len(remote_metas)} — chunk count mismatch"
                )
                # Packed descriptors carry the owning engine's key, so equal
                # bytes mean the handshake handed back our own. Every transfer
                # would then copy this rank onto itself and still report
                # success.
                if local_metas == remote_metas:
                    logger.error(
                        "layer %s: %s advertised descriptors byte-identical to "
                        "ours; transfers will copy this rank's own blocks and "
                        "the KV will never arrive",
                        ln,
                        remote_engine_id,
                    )

                def _unpack(packed):
                    return self.moriio_wrapper.get_unpack_memory_metadata(packed)

                # K sessions: NxN grid over first nk entries
                k_sessions: dict[tuple[int, int], Any] = {}
                for lci in range(nk):
                    for rci in range(nk):
                        local_md = _unpack(local_metas[lci])
                        remote_md = _unpack(remote_metas[rci])
                        k_sessions[(lci, rci)] = self.moriio_wrapper.build_session(
                            local_md, remote_md
                        )

                # V sessions: NxN grid over entries [nk:]
                v_sessions: dict[tuple[int, int], Any] = {}
                nv = len(local_metas) - nk
                for lci in range(nv):
                    for rci in range(nv):
                        local_md = _unpack(local_metas[nk + lci])
                        remote_md = _unpack(remote_metas[nk + rci])
                        v_sessions[(lci, rci)] = self.moriio_wrapper.build_session(
                            local_md, remote_md
                        )

                per_layer_sessions.append((k_sessions, v_sessions))

            logger.info(
                "Built %d K sessions + %d V sessions per layer for %s",
                len(k_sessions),
                len(v_sessions),
                remote_engine_id,
            )
            self._built_sessions[remote_engine_id] = per_layer_sessions

        return (
            self._built_sessions[remote_engine_id],
            self.remote_moriio_metadata[remote_engine_id],
        )

    def _verify_rows_for(self, local_block_ids: list[int], is_mla: bool) -> Any:
        """Row indices into dim 0 covered by the given blocks.

        MLA lays dim 0 out as tokens, so a block spans ``block_size`` rows;
        every other layout indexes blocks directly.
        """
        import torch

        first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
        device = self.kv_caches[first_layer].k_cache.device
        idx = torch.as_tensor(local_block_ids, dtype=torch.int64, device=device)
        if not is_mla:
            return idx
        offsets = torch.arange(
            self.kv_cache_block_size, dtype=torch.int64, device=device
        )
        return (idx[:, None] * self.kv_cache_block_size + offsets[None, :]).reshape(-1)

    def _verify_digest(self, rows: Any) -> dict[str, int]:
        """Position-weighted byte digest of each layer's destination rows.

        Read as uint8 so the digest is dtype-agnostic, and stacked so all
        layers cost one device sync rather than one each. Weighted because a
        reused block still holds an earlier request's KV, and under a plain
        byte sum new KV with the same total looks like KV that never landed.
        """
        import torch

        names = list(self.layer_name_to_local_kv_cache_metadata)
        sample = self.kv_caches[names[0]].k_cache
        row_bytes = sample[0].numel() * sample.element_size()
        # Weights stay below 2**16, so even a full 32k-token prompt sums far
        # inside int64.
        weights = (
            torch.arange(
                rows.numel() * row_bytes, dtype=torch.int64, device=sample.device
            )
            % 65521
            + 1
        )
        sums = [
            torch.sum(
                self.kv_caches[name]
                .k_cache.index_select(0, rows)
                .view(torch.uint8)
                .reshape(-1)
                .to(torch.int64)
                * weights
            )
            for name in names
        ]
        return dict(zip(names, torch.stack(sums).tolist()))

    def _verify_landed(self, req_id: ReqId) -> None:
        """Report layers whose destination never changed across the transfer."""
        rows = self._verify_rows.pop(req_id, None)
        before = self._verify_before.pop(req_id, None)
        if rows is None or before is None:
            return
        after = self._verify_digest(rows)
        unchanged = [name for name, value in before.items() if after[name] == value]
        if unchanged:
            # A digest of 0 on both sides means the destination is still the
            # zeroed buffer; a non-zero one that did not move means an earlier
            # request's bytes were there and the transfer left them alone.
            logger.error(
                "[moriio-verify] req %s: %d of %d layers unchanged after the "
                "transfer reported done (first %s, digest %d -> %d, rows=%d) -- "
                "decode will attend over KV that never arrived",
                req_id,
                len(unchanged),
                len(before),
                unchanged[0],
                before[unchanged[0]],
                after[unchanged[0]],
                rows.numel(),
            )
        else:
            logger.info(
                "[moriio-verify] req %s: all %d layers changed",
                req_id,
                len(before),
            )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_dp_rank: int = 0,
        remote_tp_size: int = 0,
    ) -> None:
        """Issue RDMA reads for all layers of a single request.

        Block pairs are grouped by ``(local_chunk_idx, remote_chunk_idx)``
        and chunk-relative offsets are computed.  Each group issues a
        separate RDMA batch read using the matching session from the NxN
        chunk grid.

        Transfer statuses are stored for later polling in
        :meth:`_pop_done_transfers`.
        """

        # Checked before anything else, because _group_offsets pairs what it
        # can and drops the rest silently. The two lists are legitimately
        # different lengths: the producer's block table also covers the token
        # it generated, which the consumer gets as first_token_id and does not
        # allocate for. Extra producer blocks are therefore the normal case and
        # the prefix pairing is what we want. Fewer is not: the consumer would
        # be left attending over blocks nobody filled.
        if len(remote_block_ids) < len(local_block_ids):
            raise ValueError(
                f"req {request_id}: {len(local_block_ids)} local blocks but "
                f"only {len(remote_block_ids)} remote blocks; the tail of the "
                "prompt's KV has no source"
            )
        if len(remote_block_ids) > len(local_block_ids):
            logger.debug(
                "req %s: producer offered %d blocks for %d local; pairing the "
                "first %d",
                request_id,
                len(remote_block_ids),
                len(local_block_ids),
                len(local_block_ids),
            )

        logger.debug(
            "Reading %d blocks for req %s from %s (tp_rank=%d, remote_dp_rank=%d)",
            len(local_block_ids),
            request_id,
            dst_engine_id,
            self.tp_rank,
            remote_dp_rank,
        )

        dp_engine_id = self._engine_name_with_dp(dst_engine_id, remote_dp_rank)
        sessions, _remote_meta = self._get_or_build_sessions(dp_engine_id)

        is_mla, per_block_bytes = self._kv_layout()
        groups = self._group_offsets(local_block_ids, remote_block_ids, per_block_bytes)

        # Notify port = base handshake port + offset(remote_dp_rank, local_tp_rank).
        # Strided by the *remote* TP size, since the port belongs to the peer.
        notify_port = remote_handshake_port + get_port_offset(
            remote_dp_rank, self.tp_rank, remote_tp_size or self.tp_size
        )

        layer_names = list(self.layer_name_to_local_kv_cache_metadata.keys())

        if self._verify_kv:
            rows = self._verify_rows_for(local_block_ids, is_mla)
            self._verify_rows[request_id] = rows
            self._verify_before[request_id] = self._verify_digest(rows)

        for layer_idx, layer_name in enumerate(layer_names):
            k_sessions, v_sessions = sessions[layer_idx]

            for (lci, rci), (l_offs, r_offs, szs) in groups.items():
                # K read
                status = self.moriio_wrapper.read_remote_data(
                    szs, l_offs, r_offs, k_sessions[(lci, rci)]
                )
                with self.moriio_wrapper.lock:
                    self._recving_transfers[request_id].append(status)
                    self._recving_transfers_callback_addr[request_id] = (
                        remote_host,
                        str(notify_port),
                    )

                # V read (same chunk-relative offsets)
                if v_sessions:
                    status = self.moriio_wrapper.read_remote_data(
                        szs, l_offs, r_offs, v_sessions[(lci, rci)]
                    )
                    with self.moriio_wrapper.lock:
                        self._recving_transfers[request_id].append(status)

        logger.debug(
            "RDMA read issued for req %s (%d layers, %d chunk groups) "
            "from %s (dp_rank=%d, notify_port=%d)",
            request_id,
            len(layer_names),
            len(groups),
            dst_engine_id,
            remote_dp_rank,
            notify_port,
        )

    def _kv_layout(self) -> tuple[bool, int]:
        """``(is_mla, per_block_bytes)`` for the registered KV cache.

        MLA lays dim 0 out as tokens, so a block spans ``block_size`` rows of
        it; every other layout indexes blocks directly.
        """
        first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
        cache_tensor = self.kv_caches[first_layer].k_cache
        is_mla = self.kv_caches[first_layer].v_cache is None
        rows_per_block = self.kv_cache_block_size if is_mla else 1
        per_block_bytes = (
            rows_per_block * cache_tensor.stride(0) * cache_tensor.element_size()
        )

        # Every layer moves with offsets derived from the first layer's stride,
        # so a layer shaped differently would be transferred at the wrong
        # addresses without ever failing a transfer.
        for name in self.layer_name_to_local_kv_cache_metadata:
            other = self.kv_caches[name].k_cache
            if other.shape != cache_tensor.shape or other.stride() != (
                cache_tensor.stride()
            ):
                raise ValueError(
                    f"layer {name} is shaped {tuple(other.shape)}/"
                    f"{other.stride()} but offsets come from {first_layer} "
                    f"shaped {tuple(cache_tensor.shape)}/{cache_tensor.stride()}"
                )

        if not self._logged_layout:
            self._logged_layout = True
            logger.info(
                "[moriio-layout] is_mla=%s layers=%d block_size=%d "
                "per_block_bytes=%d blocks_per_chunk=%s k_chunks=%d shape=%s",
                is_mla,
                len(self.layer_name_to_local_kv_cache_metadata),
                self.kv_cache_block_size,
                per_block_bytes,
                self.blocks_per_chunk,
                self.num_k_chunks,
                tuple(cache_tensor.shape),
            )
        return is_mla, per_block_bytes

    def _group_offsets(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        per_block_bytes: int,
    ) -> dict[tuple[int, int], tuple[list[int], list[int], list[int]]]:
        """Chunk-relative byte offsets of each block pair, grouped by chunk pair.

        Pairs positionally and stops at the shorter list; callers check the
        lengths first. Each ``(local_chunk, remote_chunk)`` group is one batch
        on the matching session of the NxN grid. Shared by both directions, so
        a read and a write of the same pair always touch the same bytes.
        """
        bpc = self.blocks_per_chunk
        groups: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
        for lb, rb in zip(local_block_ids, remote_block_ids):
            key = (lb // bpc, rb // bpc)
            if key not in groups:
                groups[key] = ([], [], [])
            groups[key][0].append((lb % bpc) * per_block_bytes)
            groups[key][1].append((rb % bpc) * per_block_bytes)
            groups[key][2].append(per_block_bytes)
        return groups

    def _write_blocks(self, request: dict[str, Any], engine_id: str) -> list[Any]:
        """Push our source blocks into the consumer's; return the statuses.

        The producer half of serial WRITE mode. Local here is our source and
        remote the consumer's destination, the same positional pairing the
        read path uses, issued from the other end.
        """
        dst = [int(b) for b in request["dst_block_ids"]]
        src = [int(b) for b in request["src_block_ids"]]
        if len(src) < len(dst):
            raise ValueError(
                f"transfer {request['transfer_id']}: {len(dst)} destination "
                f"blocks but only {len(src)} source blocks"
            )
        sessions, _remote_meta = self._get_or_build_sessions(engine_id)
        _is_mla, per_block_bytes = self._kv_layout()
        groups = self._group_offsets(src[: len(dst)], dst, per_block_bytes)

        statuses: list[Any] = []
        for k_sessions, v_sessions in sessions:
            for (lci, rci), (l_offs, r_offs, szs) in groups.items():
                statuses.append(
                    self.moriio_wrapper.write_remote_data_status(
                        szs, l_offs, r_offs, k_sessions[(lci, rci)]
                    )
                )
                if v_sessions:
                    statuses.append(
                        self.moriio_wrapper.write_remote_data_status(
                            szs, l_offs, r_offs, v_sessions[(lci, rci)]
                        )
                    )
        logger.debug(
            "RDMA write issued for transfer %s: %d blocks, %d layers, %d chunk "
            "groups to %s",
            request["transfer_id"],
            len(dst),
            len(sessions),
            len(groups),
            engine_id,
        )
        return statuses

    def _service_write_requests(self) -> None:
        """Start queued pushes and report the ones that have finished.

        Runs on the worker thread from ``get_finished``, which the engine keeps
        calling while finished prefills hold deferred blocks, so a request is
        served even when no batch is running.
        """
        while True:
            try:
                request = self._write_requests.get_nowait()
            except queue.Empty:
                break
            if self._is_valid_write_request(request):
                self._pending_writes.append((request, time.monotonic()))

        waiting: list[tuple[dict[str, Any], float]] = []
        for request, queued_at in self._pending_writes:
            ready = self._write_target_ready(request)
            if ready is None:
                if time.monotonic() - queued_at > self._handshake_timeout_s:
                    logger.error(
                        "transfer %s: no handshake with the consumer after %.0fs",
                        request["transfer_id"],
                        self._handshake_timeout_s,
                    )
                    self._finish_write(request, ok=False)
                else:
                    waiting.append((request, queued_at))
                continue
            if not ready:
                self._finish_write(request, ok=False)
                continue
            try:
                statuses = self._write_blocks(request, self._write_engine_id(request))
            except Exception:
                logger.exception(
                    "transfer %s: could not push KV to the consumer",
                    request["transfer_id"],
                )
                self._finish_write(request, ok=False)
                continue
            self._sending_writes[int(request["transfer_id"])] = (statuses, request)
        self._pending_writes = waiting

        for transfer_id, (statuses, request) in list(self._sending_writes.items()):
            failed = [status for status in statuses if status.Failed()]
            if failed:
                logger.error(
                    "RDMA write failed for transfer %s: %d of %d transfers "
                    "failed, first: %s (code %s)",
                    transfer_id,
                    len(failed),
                    len(statuses),
                    failed[0].Message(),
                    failed[0].Code(),
                )
                self._finish_write(request, ok=False)
            elif all(status.Succeeded() for status in statuses):
                self._finish_write(request, ok=True)

    @staticmethod
    def _is_valid_write_request(request: Any) -> bool:
        """Whether a WRITE_REQ has every field the push and its report need.

        Checked on dequeue because the fields are indexed and cast on the
        worker thread, where one bad request would take down the rank. A
        malformed one names nothing that could be released or answered, so it
        is dropped.
        """
        try:
            int(request["transfer_id"])
            int(request["decode_handshake_port"])
            int(request.get("decode_dp_rank", 0))
            int(request.get("decode_tp_size") or 0)
            valid = (
                "decode_req_id" in request
                and isinstance(request["decode_host"], str)
                and isinstance(request["dst_block_ids"], list)
                and isinstance(request["src_block_ids"], list)
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            logger.error(
                "Dropping malformed WRITE_REQ with fields %s",
                sorted(request) if isinstance(request, dict) else type(request),
            )
        return valid

    @classmethod
    def _write_engine_id(cls, request: dict[str, Any]) -> str:
        """The dp-suffixed id the consumer's sessions are cached under."""
        return cls._engine_name_with_dp(
            f"{request['decode_host']}:{request['decode_handshake_port']}",
            int(request.get("decode_dp_rank", 0)),
        )

    def _write_target_ready(self, request: dict[str, Any]) -> bool | None:
        """Whether we can write to this consumer yet: True, False, or None.

        None means the handshake is still running. It runs on the handshake
        executor rather than here, so an unreachable consumer cannot freeze
        this rank's worker loop.
        """
        engine_id = self._write_engine_id(request)
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return True
            future = self._write_handshakes.get(engine_id)
            if future is None:
                future = self._handshake_executor.submit(
                    self._execute_handshake,
                    request["decode_host"],
                    int(request["decode_handshake_port"]),
                    int(request.get("decode_tp_size") or self.tp_size),
                    engine_id,
                    int(request.get("decode_dp_rank", 0)),
                )
                self._write_handshakes[engine_id] = future
            if not future.done():
                return None
            # Dropped either way: success is recorded in _remote_agents, and a
            # failure should be retried by the next request, not cached.
            del self._write_handshakes[engine_id]
            try:
                self._remote_agents[engine_id] = future.result()
            except Exception:
                logger.exception("Handshake with consumer %s failed", engine_id)
                return False
            return True

    def _finish_write(self, request: dict[str, Any], ok: bool) -> None:
        """Release the source blocks and tell the consumer how the push went."""
        transfer_id = int(request["transfer_id"])
        self._sending_writes.pop(transfer_id, None)
        # Released whatever happened: a consumer told the push failed has no
        # reason to come back for these blocks.
        self.done_sending.add(transfer_id)
        port = int(request["decode_handshake_port"]) + get_port_offset(
            int(request.get("decode_dp_rank", 0)),
            self.tp_rank,
            int(request.get("decode_tp_size") or self.tp_size),
        )
        report = {"decode_req_id": request["decode_req_id"], "ok": ok}
        try:
            self.moriio_wrapper.send_message(
                [MoRIIOConstants.WRITE_DONE, msgpack.dumps(report)],
                request["decode_host"],
                port,
            )
        except Exception:
            logger.exception(
                "transfer %s: could not report the push to the consumer", transfer_id
            )

    def _collect_write_reports(
        self, done_recving: set[ReqId], failed_recving: set[ReqId]
    ) -> None:
        """Move producer reports, and pushes that never got one, into the sets."""
        with self._write_reports_lock:
            reports, self._write_reports = self._write_reports, {}
        for req_id, ok in reports.items():
            if self._awaiting_write.pop(req_id, None) is None:
                logger.warning("Write report for req %s, which is not waiting", req_id)
                continue
            (done_recving if ok else failed_recving).add(req_id)

        now = time.monotonic()
        for req_id, asked_at in list(self._awaiting_write.items()):
            if now - asked_at > self._write_timeout_s:
                logger.error(
                    "req %s: no report from the producer %.0fs after asking it "
                    "to push KV",
                    req_id,
                    self._write_timeout_s,
                )
                del self._awaiting_write[req_id]
                failed_recving.add(req_id)

    def _handshake_listener(
        self,
        metadata: MoRIIOAgentMetadata,
        ready_event: threading.Event,
        base_port: int,
        tp_rank: int,
        dp_rank: int,
        layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]],
    ) -> None:
        """Background thread that serves metadata to incoming handshake requests.

        Handles four message types:
        - ``GET_META_MSG``: Responds with engine + per-layer KV cache metadata.
        - ``POP_DONE_RECV``: Records that the consumer finished reading the request.
        - ``WRITE_REQ`` (producer): queues a consumer's request for a push.
        - ``WRITE_DONE`` (consumer): records the producer's report on a push.
        """
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        logger.info("Handshake listener ready (%d bytes metadata)", len(encoded_data))

        path = make_zmq_path("tcp", "*", base_port)
        logger.info("Handshake listener bound to %s", path)

        with _zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                parts = sock.recv_multipart()
                identity, msg = parts[0], parts[1]

                if msg == MoRIIOConstants.GET_META_MSG:
                    # Phase 1: send engine metadata
                    sock.send_multipart((identity, b"", encoded_data))
                    logger.debug("Handshake: sent engine metadata to peer")
                    # Phase 2: send per-layer KV cache metadata
                    buf = msgpack.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))

                elif msg == MoRIIOConstants.POP_DONE_RECV:
                    if len(parts) < 3:
                        raise ValueError("POP_DONE_RECV missing request ID")
                    req_id = int(parts[2])
                    self.done_sending.add(req_id)
                    logger.debug(
                        "Handshake listener: consumer finished reading req %d", req_id
                    )

                # A bad write frame is logged and dropped rather than raised:
                # this thread also serves every other peer's handshake.
                elif msg == MoRIIOConstants.WRITE_REQ:
                    # Only queued; the worker thread owns the RDMA engine.
                    try:
                        self._write_requests.put(msgpack.loads(parts[2]))
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_REQ")

                elif msg == MoRIIOConstants.WRITE_DONE:
                    try:
                        report = msgpack.loads(parts[2])
                        req_id, ok = report["decode_req_id"], bool(report["ok"])
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_DONE")
                        continue
                    with self._write_reports_lock:
                        self._write_reports[req_id] = ok

                else:
                    logger.error("Unexpected handshake message type: %s", msg)
                    raise ValueError(f"Unexpected handshake message: {msg!r}")

    def _execute_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dp_rank: int = 0,
    ) -> set[str]:
        """Perform a MoRIIO handshake with a remote engine instance.

        Connects to the remote handshake listener, exchanges engine and
        memory metadata, and registers the remote engine for RDMA ops.

        Returns:
            Set containing the remote agent name.
        """
        start_time = time.perf_counter()

        # Each (dp, tp) rank uses a unique port offset
        port_offset = get_port_offset(
            remote_dp_rank, self.tp_rank, remote_tp_size or self.tp_size
        )
        path = make_zmq_path("tcp", host, port + port_offset)
        logger.info("Initiating handshake on %s", path)

        with _zmq_ctx(zmq.DEALER, path) as sock:
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                raise ValueError(f"Unexpected frame! {received_frame = }")

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.info(
                "MoRIIO handshake: get metadata took: %s",
                got_metadata_time - start_time,
            )

            self.moriio_wrapper.remote_engine_ip = host
            remote_agent_name = self.moriio_wrapper.register_remote_engine(
                metadata.agent_metadata
            )

            logger.info(
                "MoRIIO handshake: registered"
                "remote agent %s for engine ID %s, path = %s",
                remote_agent_name,
                expected_engine_id,
                path,
            )

            if len(self.local_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.local_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.local_kv_cache_metadata),
                )
                self.local_kv_cache_metadata = []
            if len(self.remote_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.remote_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.remote_kv_cache_metadata),
                )
                self.remote_kv_cache_metadata = []

            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                raise ValueError(f"unexpected frame! {received_frame = }")
            buf = received_frame[1]
            self.layer_name_to_remote_kv_cache_metadata[expected_engine_id] = (
                msgpack.loads(buf)
            )
            self.remote_moriio_metadata[expected_engine_id] = metadata
            setup_agent_time = time.perf_counter()
            logger.debug(
                "MoRIIO handshake: add agent took: %s",
                setup_agent_time - got_metadata_time,
            )

        return {remote_agent_name}

    def _initiate_background_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ) -> None:
        """Start asynchronous handshake(s) with a remote engine.

        For multi-DP setups, initiates handshakes with all remote DP ranks
        in parallel via the single-threaded executor (to maintain MoRIIO
        thread safety).  Once all complete, the request is placed on
        ``_ready_requests`` for RDMA reads.
        """
        logger.debug(
            "Initiating background handshake for req %s -> %s",
            req_id,
            remote_engine_id,
        )

        host = meta.remote_host
        port = int(meta.remote_handshake_port)
        tp_size = int(meta.tp_size)
        remote_dp_size = int(meta.remote_dp_size)

        def _on_all_done(_f: Future[Any], entry=(req_id, meta)):
            logger.debug("All handshakes completed for req %s", req_id)
            self._ready_requests.put(entry)

        futures: list[Future[set[str]]] = []

        # In dp(prefill)<->dp(decode) communication, all-to-all handshake is required.
        for cur_dp_rank in range(remote_dp_size):
            dp_engine_id = self._engine_name_with_dp(remote_engine_id, cur_dp_rank)
            future = self._handshake_executor.submit(
                self._execute_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
            )
            futures.append(future)

            def _on_single_done(f: Future[set[str]], eid=dp_engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            future.add_done_callback(_on_single_done)
            self._handshake_futures[dp_engine_id] = future

        def _wait_all():
            for f in futures:
                f.result()
            return True

        all_done_future = self._handshake_executor.submit(_wait_all)
        all_done_future.add_done_callback(_on_all_done)

    def _pop_done_transfers(self) -> tuple[set[str], set[str]]:
        """Return the (done, failed) receive request IDs.

        ``_read_blocks`` issues one ``batch_read`` per (layer, chunk group) and
        each carries its own transfer uid, so they complete in any order. A
        request is done only once every one of them has succeeded; the last
        one issued finishing says nothing about the others.
        """
        done_req_ids: set[str] = set()
        failed_req_ids: set[str] = set()
        with self.moriio_wrapper.lock:
            to_remove = []
            for req_id, status_list in self._recving_transfers.items():
                failed = [status for status in status_list if status.Failed()]
                if failed:
                    logger.error(
                        "RDMA read failed for req %s: %d of %d transfers "
                        "failed, first: %s (code %s)",
                        req_id,
                        len(failed),
                        len(status_list),
                        failed[0].Message(),
                        failed[0].Code(),
                    )
                    failed_req_ids.add(req_id)
                elif not all(status.Succeeded() for status in status_list):
                    continue
                else:
                    done_req_ids.add(req_id)
                # Notify either way: the producer holds the source blocks until
                # it hears back, so a dropped notify leaks them for the run.
                # the Decode req_id(request_id) ,Prefill req_id(transfer_id)
                # so we need to use transfer_id to send notify
                self.moriio_wrapper.send_notify(
                    self.request_id_to_transfer_id[req_id],
                    self._recving_transfers_callback_addr[req_id][0],
                    self._recving_transfers_callback_addr[req_id][1],
                )
                to_remove.append(req_id)
            for req_id in to_remove:
                del self._recving_transfers[req_id]
                del self._recving_transfers_callback_addr[req_id]

            return done_req_ids, failed_req_ids

    def get_finished(self) -> KVConnectorOutput:
        """Return the finished sending, receiving, and failed request IDs.

        Called by the worker each step via ``async_proc_aggregation``.
        """
        if self.is_producer:
            # Before done_sending is read below, so a push that finishes this
            # step releases its blocks this step.
            self._service_write_requests()
        done_recving, failed_recving = self._pop_done_transfers()
        if self._awaiting_write or self._write_reports:
            self._collect_write_reports(done_recving, failed_recving)
        if self._failed_before_read:
            failed_recving |= self._failed_before_read
            self._failed_before_read = set()
        if self._verify_kv:
            # Outside _pop_done_transfers so the digests stay off the path
            # that holds the wrapper lock.
            for req_id in done_recving:
                self._verify_landed(req_id)
            for req_id in failed_recving:
                self._verify_rows.pop(req_id, None)
                self._verify_before.pop(req_id, None)
        if self.is_producer:
            done_sending = self.done_sending.copy()
            self.done_sending.clear()
        else:
            if self.done_sending:
                logger.warning(
                    "Consumer received %d stale done_sending notifications "
                    "(single-machine port collision?) — discarding: %s",
                    len(self.done_sending),
                    self.done_sending,
                )
                self.done_sending.clear()
            done_sending = set()
        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )


# ===================================================================
# MoRIIOConnectorScheduler — scheduler-side connector
# ===================================================================


class MoRIIOConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side KV connector that tracks transfer lifecycle.

    Runs in the scheduler process (not in TP workers).  Responsible for:

    1. Detecting when a request needs remote KV loading.
    2. Building :class:`ConnectorMetadata` to pass to workers.
    3. Populating the response with KV transfer output metadata so the
       proxy can coordinate between prefill and decode instances.
    4. Managing transfer_id <-> request_id mappings.
    """

    def __init__(self, config: Config) -> None:
        kv_transfer_config = config.kv_transfer_config
        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.handshake_port = get_open_port()
        self.base_handshake_port = kv_transfer_config.get(
            "handshake_port", MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
        )
        self.engine_id = "None"
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.parallel_config.data_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.host_ip = get_ip()

        # Pending receive requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Source-block ownership -- see `MooncakeConnectorScheduler` for why
        # the claim is taken in `request_finished` rather than at alloc.
        self._awaiting_send: set[str] = set()

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        """Check if this sequence needs remote KV prefill.

        Returns:
            ``(num_tokens, needs_async_load)`` where ``needs_async_load``
            is True if the scheduler should defer until KV transfer completes.
        """
        params = seq.kv_transfer_params or {}

        if params.get("do_remote_prefill") and not hasattr(seq, "kv_async_tagged"):
            seq.kv_async_tagged = True
            return len(seq.prompt_token_ids), True

        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        """Build a metadata snapshot of pending receive requests.

        The returned object is passed to the worker-side connector
        for RDMA operations.  The internal pending queue is cleared.
        """
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        logger.debug(
            "Built connector metadata with %d recv requests: %s",
            len(self._reqs_need_recv),
            list(self._reqs_need_recv.keys()),
        )
        self._reqs_need_recv.clear()
        return meta

    def update_state_after_alloc(self, seq: Sequence) -> None:
        """Update internal state after the scheduler allocates blocks for a sequence.

        For the decode (consumer) side, this records the transfer_id <->
        request_id mapping and queues the request for KV loading.
        """
        params = seq.kv_transfer_params or {}

        if not self.is_producer:
            transfer_id = params.get("transfer_id")
            if transfer_id is not None:
                self.transfer_id_to_request_id[transfer_id] = seq.id
                self.request_id_to_transfer_id[seq.id] = transfer_id

        # Decode side: queue for remote KV loading
        if params.get("do_remote_prefill"):
            assert (
                not self.is_producer
            ), "Only the decode (consumer) side handles do_remote_prefill"
            self._reqs_need_recv[seq.id] = (seq, list(seq.block_table))
            params["do_remote_prefill"] = False
            logger.debug(
                "Queued req %s for remote KV loading (%d blocks)",
                seq.id,
                len(seq.block_table),
            )

    def request_finished(self, seq: Sequence) -> None:
        """Populate KV transfer output metadata when a request completes.

        On the producer side this allows the proxy to forward block info
        to the decode instance.  On the consumer side this cleans up
        the transfer_id mapping.
        """
        if self.is_producer and getattr(seq, "leave_reason", None) == "aborted":
            # No send claim protects an abort's blocks from reuse. Never
            # advertise their addresses, including any previous metadata.
            seq.kv_transfer_params_output = None
            return

        # Claim the source blocks -- the metadata below hands the peer their
        # addresses. Gated on `is_producer` alone: unlike mooncake this backend
        # never reads `do_remote_decode` and sends everything it prefills.
        if self.is_producer:
            self._awaiting_send.add(str(seq.id))

        # Attach output metadata for the proxy to relay
        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        drafts = getattr(seq, "spec_token_ids", None)
        draft_token_ids = (
            [int(x) for x in drafts] if drafts is not None and len(drafts) else []
        )
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": list(seq.block_table),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "remote_handshake_port": self.base_handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "draft_token_ids": draft_token_ids,
            "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
        }

        # Clean up transfer ID mapping on the consumer side
        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)

    def should_defer_free(self, seq: Sequence) -> bool:
        return str(seq.id) in self._awaiting_send

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))

    def source_blocks_released(self, seq: Sequence) -> None:
        """No block-lifetime state remains after the send claim is retired."""


def _zmq_ctx(socket_type: int, addr: str):
    """Context manager for a ZMQ socket with role-appropriate bind semantics.

    ROUTER sockets bind; DEALER/REQ sockets connect.
    """
    return zmq_socket_ctx(addr, socket_type, bind=(socket_type == zmq.ROUTER))
