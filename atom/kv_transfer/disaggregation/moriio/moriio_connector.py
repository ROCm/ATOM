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
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgpack
import msgspec
import numpy as np
import zmq

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    MoRIIOWriteDone,
    MoRIIOWriteRegion,
    MoRIIOWriteRequest,
    TransferMode,
    _MORIIO_AVAILABLE,
    MAX_RDMA_CHUNK_BYTES,
    get_port_offset,
)
from atom.kv_transfer.disaggregation.utils import chunk_tensor_for_rdma
from atom.kv_transfer.disaggregation.moriio.moriio_engine import MoRIIOWrapper
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    EngineId,
    KVConnectorOutput,
    KVTransferRegion,
    KVTransferTensors,
    ReqId,
    ReqMeta,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_port,
    make_zmq_path,
    zmq_socket_ctx,
)
from atom.utils.network import get_ip
from aiter.dist.parallel_state import get_dp_group, get_tp_group

if _MORIIO_AVAILABLE:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        PollCqMode,
        RdmaBackendConfig,
        StatusCode,
    )

logger = logging.getLogger("atom")

if os.environ.get("MORIIO_TRACE", "0") == "1" and _MORIIO_AVAILABLE:
    try:
        from mori.io import set_log_level as _set_mori_log_level

        _set_mori_log_level("trace")
    except Exception:
        pass


@dataclass(frozen=True)
class _WriteTask:
    request: MoRIIOWriteRequest
    remote_engine_key: str


# ===================================================================
# MoRIIOConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MoRIIOConnector(KVConnectorBase):
    """Worker-side KV cache connector for disaggregated P/D inference.

    Each tensor-parallel worker instantiates one ``MoRIIOConnector``.  It is
    responsible for:

    1. Registering local KV cache tensors for RDMA access.
    2. Performing handshakes with remote engines to exchange memory metadata.
    3. Issuing RDMA read operations to pull KV cache blocks from the producer.
    4. Tracking transfer completion and notifying the producer when done.
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
        self.transfer_mode = TransferMode(
            kv_transfer_config.get("transfer_mode", TransferMode.WRITE_PUSH.value)
        )
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.base_handshake_port = kv_transfer_config.get(
            "handshake_port", MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
        )

        # Compute unique side-channel port for this (dp, tp) rank
        handshake_port = self.base_handshake_port
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
        kv_role = "producer" if self.is_producer else "consumer"
        self.moriio_engine_key = (
            f"atom-{kv_role}-dp{self.dp_rank}-tp{self.tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}-{uuid.uuid4().hex[:8]}"
        )
        self.moriio_engine = IOEngine(
            self.moriio_engine_key,
            IOEngineConfig(host=str(self.local_ip), port=0),
        )
        self.moriio_wrapper = MoRIIOWrapper(moriio_engine=self.moriio_engine)

        qp_per_transfer = kv_transfer_config.get("qp_per_transfer", 4)
        post_batch_size = kv_transfer_config.get("post_batch_size", -1)
        num_worker_threads = kv_transfer_config.get("num_worker_threads", 4)
        write_transfer_workers = kv_transfer_config.get(
            "write_transfer_workers", num_worker_threads
        )
        poll_mode = PollCqMode.POLLING
        enable_notification = kv_transfer_config.get("enable_notification", False)
        self._write_prefill_timeout_s = float(
            kv_transfer_config.get(
                "write_prefill_timeout", MoRIIOConstants.ABORT_REQUEST_TIMEOUT
            )
        )
        self._write_prefill_orphan_timeout_s = float(
            kv_transfer_config.get(
                "write_prefill_orphan_timeout", self._write_prefill_timeout_s
            )
        )
        self._write_recv_timeout_s = float(
            kv_transfer_config.get(
                "write_recv_timeout", MoRIIOConstants.ABORT_REQUEST_TIMEOUT
            )
        )
        self._write_transfer_timeout_ms = int(
            kv_transfer_config.get(
                "write_transfer_timeout_ms",
                MoRIIOConstants.ABORT_REQUEST_TIMEOUT * 1000,
            )
        )

        # Backend selection: FABRIC (UALink scale-up) when requested, else RDMA.
        # Fabric needs KV in fabric-exportable VMM memory (see model_runner
        # _moriio_fabric_alloc_ctx) and auto-loads the copy kernel on create.
        self._moriio_use_fabric = (
            str(kv_transfer_config.get("moriio_backend", "")).lower() == "fabric"
            or os.environ.get("ATOM_MORIIO_FABRIC", "0") == "1"
        )
        if self._moriio_use_fabric:
            # Lazy: fabric backend is absent on older mori-io builds.
            from mori.io import FabricBackendConfig

            fabric_cfg = FabricBackendConfig()
            fabric_cfg.num_streams = kv_transfer_config.get("fabric_num_streams", 4)
            fabric_cfg.num_events = kv_transfer_config.get("fabric_num_events", 16)
            logger.info(
                "MoRIIO using FABRIC backend (UALink scale-up): num_streams=%d, "
                "num_events=%d",
                fabric_cfg.num_streams,
                fabric_cfg.num_events,
            )
            self.moriio_wrapper.set_backend_type(BackendType.FABRIC, fabric_cfg)
        else:
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
                "poll_mode=%s, notification=%s",
                qp_per_transfer,
                num_worker_threads,
                poll_mode.name,
                enable_notification,
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
        self.load_ready_flag: dict[str, bool] = {}
        self.write_ready_flags: dict[str, bool] = {}
        self._handshake_lock = threading.RLock()
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

        # Completed send-side transfers (populated by handshake listener)
        self.done_sending: set[ReqId] = set()

        # Transfer ID mapping (worker side)
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # Write-mode local region metadata and transfer state.
        self._write_local_regions: list[MoRIIOWriteRegion] = []
        self._write_gather_slot = None
        self._write_scatter_slot = None
        self._write_has_slot_regions = False
        self._write_has_staging_region = False
        self._write_session_cache: dict[tuple[str, int, int, int], Any] = {}
        self._write_session_lock = threading.Lock()
        self._write_remote_engine_keys: set[str] = set()
        self._write_remote_engine_lock = threading.Lock()

        self._completed_prefills: dict[ReqId, dict[str, Any]] = {}
        self._completed_prefill_deadlines: dict[ReqId, float] = {}
        self._inflight_write_transfers: set[ReqId] = set()
        self._completed_prefills_lock = threading.Lock()
        self._completed_prefills_cv = threading.Condition(self._completed_prefills_lock)
        self._terminal_write_transfers: set[ReqId] = set()
        self._terminal_write_transfer_order: deque[ReqId] = deque()

        self.done_recving: set[ReqId] = set()
        self.failed_recving: set[ReqId] = set()
        self._deferred_write_recvs: dict[ReqId, ReqMeta] = {}
        self._deferred_write_recv_deadlines: dict[ReqId, float] = {}
        self._pending_write_recv: set[ReqId] = set()
        self._pending_write_recv_blocks: dict[ReqId, list[int]] = {}
        self._pending_write_recv_slots: dict[ReqId, tuple[int, int]] = {}
        self._pending_write_recv_deadlines: dict[ReqId, float] = {}
        self._blocks_pending_fence: list[int] = []
        self._fence_lock = threading.Lock()

        self._staging_free: list[int] = []
        self._staging_cv = threading.Condition(threading.Lock())

        self._write_request_sockets: dict[str, zmq.Socket] = {}
        self._write_request_sockets_lock = threading.Lock()
        self._write_notify_queue: queue.Queue[tuple[str, int, MoRIIOWriteDone]] = (
            queue.Queue()
        )
        self._write_notify_thread: threading.Thread | None = None
        self._write_executor: ThreadPoolExecutor | None = None
        if self.transfer_mode == TransferMode.WRITE_PUSH and self.is_producer:
            self._write_executor = ThreadPoolExecutor(
                max_workers=max(1, int(write_transfer_workers)),
                thread_name_prefix="atom-moriio-write-worker",
            )
            self._write_notify_thread = threading.Thread(
                target=self._write_notify_loop,
                daemon=True,
                name="moriio-write-notify",
            )
            self._write_notify_thread.start()

    def _chunk_kv_tensor(self, tensor, block_size_in_dim0):
        """Split a KV tensor for registration.

        RDMA must stay under the ~2 GiB ibv_reg_mr limit (chunk_tensor_for_rdma).
        FABRIC has no such limit and multi-chunk fabric import of one VMM
        allocation stalls the transfer, so register each region as a single desc.
        """
        if not self._moriio_use_fabric:
            return chunk_tensor_for_rdma(tensor, block_size_in_dim0)
        per_block_bytes = block_size_in_dim0 * tensor.stride(0) * tensor.element_size()
        total_blocks = tensor.shape[0] // block_size_in_dim0
        return [(tensor.data_ptr(), total_blocks * per_block_bytes)], total_blocks

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

        if self.transfer_mode == TransferMode.WRITE_PUSH:
            if transfer_tensors is None:
                raise RuntimeError(
                    "MoRIIO transfer_mode='write' requires KVTransferTensors from "
                    "the attention backend."
                )
            self._register_write_regions(transfer_tensors)
            metadata = MoRIIOAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.moriio_wrapper.get_agent_metadata(),
                kv_caches_base_addr=None,
                num_blocks=self.num_blocks,
                block_len=self.block_len,
                attn_backend_name="aiter",
            )
            self._start_handshake_listener(metadata, {})
            return

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
            k_chunks, bpc = self._chunk_kv_tensor(cache_tensor, bsd0)

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
                v_chunks, _ = self._chunk_kv_tensor(v_cache, 1)
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
        self._start_handshake_listener(
            metadata, self.layer_name_to_local_kv_cache_metadata
        )

    def _start_handshake_listener(
        self,
        metadata: MoRIIOAgentMetadata,
        layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]],
    ) -> None:
        ready_event = threading.Event()
        self._handshake_listener_thread = threading.Thread(
            target=self._handshake_listener,
            args=(
                metadata,
                ready_event,
                self.side_channel_port,
                self.tp_rank,
                self.dp_rank,
                layer_name_to_local_kv_cache_metadata,
            ),
            daemon=True,
            name="moriio-handshake-listener",
        )
        self._handshake_listener_thread.start()

    def _current_gpu_device_id(self) -> int:
        """Return the current CUDA/HIP device id, or -1 when no GPU is active."""
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:
            logger.debug("Unable to query current GPU device", exc_info=True)
        return -1

    def _assert_region_on_device(
        self, region: KVTransferRegion, device_id: int
    ) -> None:
        """Best-effort validation that a transfer region belongs to device_id."""
        if device_id < 0:
            return
        try:
            import torch

            cudart = torch.cuda.cudart()
            result = cudart.cudaPointerGetAttributes(region.base_addr)
            attrs = result[1] if isinstance(result, tuple) else result
            pointer_device = getattr(attrs, "device", None)
            if pointer_device is not None and int(pointer_device) != device_id:
                raise AssertionError(
                    f"KVTransferRegion pointer 0x{region.base_addr:x} is on "
                    f"device {pointer_device}, expected {device_id}"
                )
        except AssertionError:
            raise
        except Exception:
            # Some HIP/PyTorch builds do not expose pointer attributes through
            # cudart. Registration still uses current_device(); log only to avoid
            # false negatives on supported deployments.
            logger.debug(
                "Could not validate device for KVTransferRegion at 0x%x",
                region.base_addr,
                exc_info=True,
            )

    def _register_write_regions(self, transfer_tensors: Any) -> None:
        """Register unit-addressed KVTransferTensors for write-mode RDMA."""
        tt: KVTransferTensors = transfer_tensors
        device_id = self._current_gpu_device_id()

        self.num_blocks = tt.num_blocks
        self.block_len = self.kv_cache_block_size
        self._write_gather_slot = tt.gather_slot
        self._write_scatter_slot = tt.scatter_slot
        self._write_has_slot_regions = bool(tt.slot_regions)
        self._write_has_staging_region = tt.staging_region is not None

        self._write_local_regions = []
        for region in tt.block_regions:
            packed = self._pack_write_region("block", region, device_id)
            if packed is not None:
                self._write_local_regions.append(packed)
        for region in tt.slot_regions:
            packed = self._pack_write_region("slot", region, device_id)
            if packed is not None:
                self._write_local_regions.append(packed)
        if tt.staging_region is not None:
            packed = self._pack_write_region("staging", tt.staging_region, device_id)
            if packed is not None:
                self._write_local_regions.append(packed)
                self._staging_free = list(range(packed.total_units))

        logger.info(
            "MoRIIO write-mode registration complete: role=%s, regions=%d "
            "(block=%d, slot=%d, staging=%s), num_blocks=%d, device=%d",
            "PRODUCER" if self.is_producer else "CONSUMER",
            len(self._write_local_regions),
            len(tt.block_regions),
            len(tt.slot_regions),
            tt.staging_region is not None,
            tt.num_blocks,
            device_id,
        )

    def _pack_write_region(
        self,
        kind: str,
        region: KVTransferRegion,
        device_id: int,
    ) -> MoRIIOWriteRegion | None:
        if region.total_bytes == 0:
            return None
        if region.unit_bytes <= 0:
            raise ValueError(
                f"{kind} region has invalid unit_bytes={region.unit_bytes}"
            )
        if region.total_bytes % region.unit_bytes != 0:
            raise ValueError(
                f"{kind} region total_bytes={region.total_bytes} is not a "
                f"multiple of unit_bytes={region.unit_bytes}"
            )
        # RDMA-only: a unit must fit in one <2 GiB ibv_reg_mr chunk.
        if not self._moriio_use_fabric and region.unit_bytes > MAX_RDMA_CHUNK_BYTES:
            raise ValueError(
                f"{kind} region unit_bytes={region.unit_bytes} exceeds "
                f"MAX_RDMA_CHUNK_BYTES={MAX_RDMA_CHUNK_BYTES}"
            )

        self._assert_region_on_device(region, device_id)

        total_units = region.total_bytes // region.unit_bytes
        # Fabric registers each region as a single desc (see _chunk_kv_tensor).
        if self._moriio_use_fabric:
            units_per_chunk = total_units
        else:
            units_per_chunk = max(1, MAX_RDMA_CHUNK_BYTES // region.unit_bytes)
        chunks: list[bytes] = []
        for unit_start in range(0, total_units, units_per_chunk):
            units = min(units_per_chunk, total_units - unit_start)
            chunk_bytes = units * region.unit_bytes
            ptr = region.base_addr + unit_start * region.unit_bytes
            chunks.append(
                self.moriio_wrapper.register_local_buffer(ptr, chunk_bytes, device_id)
            )

        return MoRIIOWriteRegion(
            kind=kind,
            chunks=chunks,
            unit_bytes=region.unit_bytes,
            units_per_chunk=units_per_chunk,
            total_units=total_units,
        )

    def _acquire_staging_slot(self) -> int:
        with self._staging_cv:
            ready = self._staging_cv.wait_for(
                lambda: bool(self._staging_free),
                timeout=self._write_prefill_timeout_s,
            )
            if not ready:
                raise TimeoutError("Timed out waiting for a MoRIIO staging slot")
            return self._staging_free.pop()

    def _try_acquire_staging_slot(self) -> int | None:
        with self._staging_cv:
            if not self._staging_free:
                return None
            return self._staging_free.pop()

    def _release_staging_slot(self, idx: int) -> None:
        if idx < 0:
            return
        with self._staging_cv:
            self._staging_free.append(idx)
            self._staging_cv.notify()

    def _write_notify_loop(self) -> None:
        """Single-threaded WRITE_DONE sender; owns all notification sockets."""
        encoder = msgspec.msgpack.Encoder()
        sockets: dict[str, zmq.Socket] = {}
        while True:
            host, port, done = self._write_notify_queue.get()
            path = make_zmq_path("tcp", host, port)
            try:
                sock = sockets.get(path)
                if sock is None:
                    sock = self.zmq_context.socket(zmq.DEALER)
                    sock.setsockopt(zmq.LINGER, 5000)
                    sock.setsockopt(zmq.SNDHWM, 0)
                    sock.connect(path)
                    sockets[path] = sock
                payload = encoder.encode(done)
                sock.send_multipart([MoRIIOConstants.WRITE_DONE, payload])
            except Exception:
                logger.exception(
                    "Failed to send MoRIIO WRITE_DONE for req %s to %s",
                    done.decode_req_id,
                    path,
                )
                bad_sock = sockets.pop(path, None)
                if bad_sock is not None:
                    bad_sock.close(linger=0)

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
        if self.transfer_mode == TransferMode.WRITE_PUSH:
            if metadata is None:
                metadata = ConnectorMetadata()
            self._start_write_mode(metadata)
            return

        if metadata is None:
            return

        if self.is_producer:
            return

        if metadata is not None and metadata.reqs_to_recv:
            logger.debug("Starting KV load for %d requests", len(metadata.reqs_to_recv))

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        remote_engine_id: str | None = None
        need_handshake = False

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = f"{meta.remote_host}:{meta.remote_handshake_port}"
            meta.remote_engine_id = remote_engine_id
            dp0_id = self._engine_name_with_dp(remote_engine_id, 0)

            if dp0_id not in self._remote_agents:
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._initiate_background_handshake(
                            req_id, remote_engine_id, meta
                        )
                        need_handshake = True
                        continue

            self._issue_read_for_req(req_id, meta)

        # If a handshake was needed, spin until it completes then read.
        while need_handshake:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
            ):
                continue
            elif (
                not self._ready_requests.empty()
                and remote_engine_id in self.load_ready_flag
            ):
                self._issue_read_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

    def _start_write_mode(self, metadata: ConnectorMetadata) -> None:
        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        if self.is_producer:
            for req_id, meta in metadata.reqs_to_save.items():
                transfer_id: ReqId = (
                    req_id if meta.transfer_id is None else meta.transfer_id
                )
                with self.moriio_wrapper.lock:
                    if transfer_id in self._terminal_write_transfers:
                        continue
                with self._completed_prefills_cv:
                    self._completed_prefills[transfer_id] = {
                        "block_ids": list(meta.local_block_ids),
                        "slot_index": meta.local_slot_index,
                    }
                    self._completed_prefill_deadlines[transfer_id] = (
                        time.monotonic() + self._write_prefill_orphan_timeout_s
                    )
                    self._completed_prefills_cv.notify_all()
                logger.debug(
                    "MoRIIO write producer cached prefill: transfer_id=%s "
                    "blocks=%d slot=%d",
                    transfer_id,
                    len(meta.local_block_ids),
                    meta.local_slot_index,
                )
            return

        for req_id, meta in metadata.reqs_to_recv.items():
            if req_id in self._pending_write_recv:
                continue
            self._deferred_write_recvs[req_id] = meta
            self._deferred_write_recv_deadlines.setdefault(
                req_id, time.monotonic() + self._write_recv_timeout_s
            )

        self._retry_deferred_write_recvs()

    def _retry_deferred_write_recvs(self) -> None:
        if not self._deferred_write_recvs:
            return

        self._sweep_expired_write_recvs()
        for req_id, meta in list(self._deferred_write_recvs.items()):
            sent_or_terminal = self._send_write_request_for_req(req_id, meta)
            if sent_or_terminal:
                self._deferred_write_recvs.pop(req_id, None)
                self._deferred_write_recv_deadlines.pop(req_id, None)

    def _send_write_request_for_req(self, req_id: ReqId, meta: ReqMeta) -> bool:
        if not self._write_local_regions:
            raise RuntimeError("MoRIIO write mode has no registered local regions")

        with self.moriio_wrapper.lock:
            if req_id in self._pending_write_recv:
                return True

        remote_tp_size = int(meta.tp_size or 1)
        remote_tp_rank = self.tp_rank % remote_tp_size
        remote_port = int(meta.remote_handshake_port) + get_port_offset(
            int(meta.remote_dp_rank), remote_tp_rank, remote_tp_size
        )
        remote_addr = make_zmq_path("tcp", meta.remote_host, remote_port)

        staging_pool_idx = -1
        if self._write_has_staging_region and meta.local_slot_index >= 0:
            staging_pool_idx = self._try_acquire_staging_slot()
            if staging_pool_idx is None:
                logger.debug(
                    "MoRIIO write request deferred: req=%s has no free staging slot",
                    req_id,
                )
                return False

        try:
            request = MoRIIOWriteRequest(
                decode_req_id=req_id,
                transfer_id=req_id if meta.transfer_id is None else meta.transfer_id,
                consumer_engine_desc=self.moriio_wrapper.get_agent_metadata(),
                consumer_regions=self._write_local_regions,
                dst_block_ids=list(meta.local_block_ids),
                dst_slot_index=meta.local_slot_index,
                dst_staging_pool_idx=staging_pool_idx,
                notify_host=self.local_ip,
                notify_port=self.side_channel_port,
                consumer_tp_size=self.tp_size,
                consumer_dp_rank=self.dp_rank,
            )
        except Exception:
            self._release_staging_slot(staging_pool_idx)
            with self.moriio_wrapper.lock:
                self.failed_recving.add(req_id)
            logger.exception("Failed to build MoRIIO WRITE_REQUEST for req %s", req_id)
            return True

        with self.moriio_wrapper.lock:
            self._pending_write_recv.add(req_id)
            self._pending_write_recv_blocks[req_id] = list(meta.local_block_ids)
            self._pending_write_recv_deadlines[req_id] = (
                time.monotonic() + self._write_recv_timeout_s
            )
            if meta.local_slot_index >= 0:
                self._pending_write_recv_slots[req_id] = (
                    meta.local_slot_index,
                    staging_pool_idx,
                )

        encoder = msgspec.msgpack.Encoder()
        try:
            with self._write_request_sockets_lock:
                sock = self._write_request_sockets.get(remote_addr)
                if sock is None:
                    sock = self.zmq_context.socket(zmq.DEALER)
                    sock.setsockopt(zmq.LINGER, 5000)
                    sock.setsockopt(zmq.SNDHWM, 0)
                    sock.connect(remote_addr)
                    self._write_request_sockets[remote_addr] = sock
                sock.send_multipart(
                    [MoRIIOConstants.WRITE_REQUEST, encoder.encode(request)]
                )
        except Exception:
            logger.exception("Failed to send MoRIIO WRITE_REQUEST to %s", remote_addr)
            self._handle_write_done(
                MoRIIOWriteDone(
                    decode_req_id=req_id,
                    status="failed",
                    reason=f"failed to send WRITE_REQUEST to {remote_addr}",
                )
            )
            return True

        logger.info(
            "MoRIIO write request sent: req=%s transfer_id=%s dst_blocks=%d "
            "slot=%d staging=%d remote=%s",
            req_id,
            request.transfer_id,
            len(request.dst_block_ids),
            request.dst_slot_index,
            request.dst_staging_pool_idx,
            remote_addr,
        )
        return True

    def _issue_read_for_req(self, req_id: str, meta: ReqMeta) -> None:
        """Issue RDMA reads for a single request."""
        logger.debug(
            "Issuing RDMA read for req %s from engine %s (tp_rank=%d, remote_dp_rank=%d)",
            req_id,
            meta.remote_engine_id,
            self.tp_rank,
            meta.remote_dp_rank,
        )
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_handshake_port=meta.remote_handshake_port,
            remote_dp_rank=meta.remote_dp_rank,
            remote_tp_size=meta.tp_size,
        )

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

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_dp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> None:
        """Issue RDMA reads for all layers of a single request.

        Block pairs are grouped by ``(local_chunk_idx, remote_chunk_idx)``
        and chunk-relative offsets are computed.  Each group issues a
        separate RDMA batch read using the matching session from the NxN
        chunk grid.

        Transfer statuses are stored for later polling in
        :meth:`_pop_done_transfers`.
        """

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

        bpc = self.blocks_per_chunk
        first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
        cache_tensor = self.kv_caches[first_layer].k_cache
        is_mla = self.kv_caches[first_layer].v_cache is None
        sz = cache_tensor.element_size()

        if is_mla:
            per_block_bytes = self.kv_cache_block_size * cache_tensor.stride(0) * sz
        else:
            per_block_bytes = cache_tensor.stride(0) * sz

        # Group block pairs by (local_chunk, remote_chunk)
        groups: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
        for lb, rb in zip(local_block_ids, remote_block_ids):
            lci = lb // bpc
            rci = rb // bpc
            key = (lci, rci)
            if key not in groups:
                groups[key] = ([], [], [])
            groups[key][0].append((lb % bpc) * per_block_bytes)
            groups[key][1].append((rb % bpc) * per_block_bytes)
            groups[key][2].append(per_block_bytes)

        # Notify port = base handshake port + offset(remote_dp_rank, local_tp_rank)
        remote_tp_rank = self.tp_rank % int(remote_tp_size or 1)
        notify_port = remote_handshake_port + get_port_offset(
            remote_dp_rank, remote_tp_rank, int(remote_tp_size or 1)
        )

        layer_names = list(self.layer_name_to_local_kv_cache_metadata.keys())
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

        Handles two message types:
        - ``GET_META_MSG``: Responds with engine + per-layer KV cache metadata.
        - ``POP_DONE_RECV``: Records that the consumer finished reading the request.
        - ``WRITE_REQUEST``: Enqueues a write-mode transfer task.
        - ``WRITE_DONE``: Records a write-mode receive completion.
        """
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        write_request_decoder = msgspec.msgpack.Decoder(MoRIIOWriteRequest)
        write_done_decoder = msgspec.msgpack.Decoder(MoRIIOWriteDone)
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
                    logger.info("Handshake: sent engine metadata to peer")
                    # Phase 2: send per-layer KV cache metadata
                    buf = msgpack.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))

                elif msg == MoRIIOConstants.POP_DONE_RECV:
                    if len(parts) < 3:
                        raise ValueError("POP_DONE_RECV missing request ID")
                    req_text = parts[2].decode("utf-8")
                    try:
                        req_id: ReqId = int(req_text)
                    except ValueError:
                        req_id = req_text
                    with self.moriio_wrapper.lock:
                        self.done_sending.add(req_id)
                    logger.debug(
                        "Handshake listener: consumer finished reading req %s", req_id
                    )

                elif msg == MoRIIOConstants.WRITE_REQUEST:
                    if len(parts) < 3:
                        raise ValueError("WRITE_REQUEST missing payload")
                    request = write_request_decoder.decode(parts[2])
                    if not self.is_producer:
                        logger.error(
                            "Consumer received unexpected WRITE_REQUEST for req %s",
                            request.decode_req_id,
                        )
                        continue
                    try:
                        remote_engine_key = self._register_write_remote_engine(
                            request.consumer_engine_desc
                        )
                        if self._write_executor is None:
                            raise RuntimeError("MoRIIO write executor is not running")
                        self._write_executor.submit(
                            self._execute_write_task,
                            _WriteTask(request, remote_engine_key),
                        )
                    except Exception as exc:
                        logger.exception(
                            "Failed to enqueue MoRIIO WRITE_REQUEST for req %s",
                            request.decode_req_id,
                        )
                        self._enqueue_write_done(
                            request,
                            "failed",
                            f"failed to enqueue write request: {exc}",
                        )

                elif msg == MoRIIOConstants.WRITE_DONE:
                    if len(parts) < 3:
                        raise ValueError("WRITE_DONE missing payload")
                    self._handle_write_done(write_done_decoder.decode(parts[2]))

                else:
                    logger.error("Unexpected handshake message type: %s", msg)
                    raise ValueError(f"Unexpected handshake message: {msg!r}")

    def _register_write_remote_engine(self, packed_engine_desc: bytes) -> str:
        with self._write_remote_engine_lock:
            remote_key: str | None = None
            try:
                remote_key = EngineDesc.unpack(packed_engine_desc).key
            except Exception:
                logger.debug(
                    "Could not pre-decode remote EngineDesc key", exc_info=True
                )

            if remote_key is not None and remote_key in self._write_remote_engine_keys:
                return remote_key

            registered_key = self.moriio_wrapper.register_remote_engine(
                packed_engine_desc
            )
            self._write_remote_engine_keys.add(registered_key)
            return registered_key

    def _enqueue_write_done(
        self,
        request: MoRIIOWriteRequest,
        status: str,
        reason: str | None = None,
    ) -> None:
        self._write_notify_queue.put(
            (
                request.notify_host,
                request.notify_port,
                MoRIIOWriteDone(
                    decode_req_id=request.decode_req_id,
                    status=status,
                    reason=reason,
                ),
            )
        )

    def _handle_write_done(self, done: MoRIIOWriteDone) -> None:
        req_id = done.decode_req_id
        ok = done.status == "ok"

        with self.moriio_wrapper.lock:
            was_pending = req_id in self._pending_write_recv
            if not was_pending:
                logger.info("Ignoring stale MoRIIO WRITE_DONE for req %s", req_id)
                return
            slot_info = self._pending_write_recv_slots.pop(req_id, None)
            block_ids = self._pending_write_recv_blocks.pop(req_id, [])
            self._pending_write_recv_deadlines.pop(req_id, None)
            self._pending_write_recv.discard(req_id)

        if slot_info is not None:
            compute_slot, pool_idx = slot_info
            try:
                if ok and pool_idx >= 0 and self._write_scatter_slot is not None:
                    self._write_scatter_slot(compute_slot, pool_idx)
            finally:
                self._release_staging_slot(pool_idx)

        if ok:
            if block_ids:
                with self._fence_lock:
                    self._blocks_pending_fence.extend(block_ids)
            with self.moriio_wrapper.lock:
                self.done_recving.add(req_id)
            logger.info("MoRIIO WRITE_DONE ok for req %s", req_id)
        else:
            with self.moriio_wrapper.lock:
                self.failed_recving.add(req_id)
            logger.warning(
                "MoRIIO WRITE_DONE failed for req %s: %s", req_id, done.reason
            )

    def _sweep_expired_write_recvs(self) -> None:
        if self.is_producer:
            return

        now = time.monotonic()
        expired: list[ReqId] = []
        slots_to_release: list[int] = []

        with self.moriio_wrapper.lock:
            for req_id, deadline in list(self._pending_write_recv_deadlines.items()):
                if deadline > now:
                    continue
                self._pending_write_recv_deadlines.pop(req_id, None)
                self._pending_write_recv.discard(req_id)
                self._pending_write_recv_blocks.pop(req_id, None)
                slot_info = self._pending_write_recv_slots.pop(req_id, None)
                if slot_info is not None:
                    _, pool_idx = slot_info
                    if pool_idx >= 0:
                        slots_to_release.append(pool_idx)
                expired.append(req_id)

            for req_id, deadline in list(self._deferred_write_recv_deadlines.items()):
                if deadline > now:
                    continue
                self._deferred_write_recv_deadlines.pop(req_id, None)
                self._deferred_write_recvs.pop(req_id, None)
                expired.append(req_id)

            if expired:
                self.failed_recving.update(expired)

        for pool_idx in slots_to_release:
            self._release_staging_slot(pool_idx)

        if expired:
            logger.warning(
                "MoRIIO write receive timed out for %d request(s): %s",
                len(expired),
                expired,
            )

    def _sweep_orphaned_completed_prefills(self) -> None:
        if not self.is_producer:
            return

        now = time.monotonic()
        expired: list[ReqId] = []
        with self._completed_prefills_cv:
            for transfer_id, deadline in list(
                self._completed_prefill_deadlines.items()
            ):
                if deadline > now or transfer_id in self._inflight_write_transfers:
                    continue
                self._completed_prefill_deadlines.pop(transfer_id, None)
                self._completed_prefills.pop(transfer_id, None)
                expired.append(transfer_id)
            if expired:
                self._completed_prefills_cv.notify_all()

        for transfer_id in expired:
            self._mark_write_transfer_terminal(transfer_id)

        if expired:
            logger.warning(
                "MoRIIO producer expired %d orphaned prefill transfer(s): %s",
                len(expired),
                expired,
            )

    def _execute_write_task(self, task: _WriteTask) -> None:
        request = task.request
        status = "failed"
        reason: str | None = None
        prefill_data: dict[str, Any] | None = None
        try:
            if request.consumer_tp_size != self.tp_size:
                raise ValueError(
                    f"TP mismatch: producer_tp_size={self.tp_size}, "
                    f"consumer_tp_size={request.consumer_tp_size}"
                )

            prefill_data = self._wait_for_prefill_data(request.transfer_id)
            if prefill_data is None:
                raise TimeoutError(
                    f"Timed out waiting for prefill data for transfer_id="
                    f"{request.transfer_id}"
                )

            transfer_ok = self._write_blocks(
                request=request,
                remote_engine_key=task.remote_engine_key,
                src_block_ids=prefill_data["block_ids"],
                src_slot_index=prefill_data["slot_index"],
            )
            if not transfer_ok:
                raise RuntimeError("RDMA write did not complete successfully")
            status = "ok"
        except Exception as exc:
            reason = str(exc)
            logger.exception(
                "MoRIIO write transfer failed: req=%s transfer_id=%s",
                request.decode_req_id,
                request.transfer_id,
            )
        finally:
            self._enqueue_write_done(request, status, reason)
            self._mark_write_transfer_terminal(request.transfer_id)

    def _mark_write_transfer_terminal(self, transfer_id: ReqId) -> None:
        with self.moriio_wrapper.lock:
            if transfer_id in self._terminal_write_transfers:
                return
            self._terminal_write_transfers.add(transfer_id)
            self._terminal_write_transfer_order.append(transfer_id)
            while len(self._terminal_write_transfer_order) > 65536:
                old_transfer_id = self._terminal_write_transfer_order.popleft()
                self._terminal_write_transfers.discard(old_transfer_id)
            self.done_sending.add(transfer_id)

        with self._completed_prefills_cv:
            self._completed_prefills.pop(transfer_id, None)
            self._completed_prefill_deadlines.pop(transfer_id, None)
            self._inflight_write_transfers.discard(transfer_id)
            self._completed_prefills_cv.notify_all()

    def _wait_for_prefill_data(self, transfer_id: ReqId) -> dict[str, Any] | None:
        with self._completed_prefills_cv:
            ready = self._completed_prefills_cv.wait_for(
                lambda: transfer_id in self._completed_prefills,
                timeout=self._write_prefill_timeout_s,
            )
            if not ready:
                return None
            self._inflight_write_transfers.add(transfer_id)
            return self._completed_prefills[transfer_id]

    def _write_blocks(
        self,
        request: MoRIIOWriteRequest,
        remote_engine_key: str,
        src_block_ids: list[int],
        src_slot_index: int,
    ) -> bool:
        if len(src_block_ids) != len(request.dst_block_ids):
            raise ValueError(
                f"src/dst block count mismatch: {len(src_block_ids)} vs "
                f"{len(request.dst_block_ids)}"
            )
        if len(self._write_local_regions) != len(request.consumer_regions):
            raise ValueError(
                "write region count mismatch: "
                f"local={len(self._write_local_regions)}, "
                f"remote={len(request.consumer_regions)}"
            )

        block_statuses = self._submit_write_region_units(
            remote_engine_key,
            request.consumer_regions,
            kind="block",
            src_units=src_block_ids,
            dst_units=request.dst_block_ids,
        )
        if not self._wait_for_write_statuses(block_statuses):
            return False

        if src_slot_index < 0 or request.dst_slot_index < 0:
            return True

        slot_statuses = self._submit_write_region_units(
            remote_engine_key,
            request.consumer_regions,
            kind="slot",
            src_units=[src_slot_index],
            dst_units=[request.dst_slot_index],
        )

        producer_staging_pool_idx = -1
        try:
            if (
                self._write_gather_slot is not None
                and self._write_has_staging_region
                and request.dst_staging_pool_idx >= 0
            ):
                producer_staging_pool_idx = self._acquire_staging_slot()
                self._write_gather_slot(src_slot_index, producer_staging_pool_idx)
                import torch

                torch.cuda.current_stream().synchronize()
                slot_statuses.extend(
                    self._submit_write_region_units(
                        remote_engine_key,
                        request.consumer_regions,
                        kind="staging",
                        src_units=[producer_staging_pool_idx],
                        dst_units=[request.dst_staging_pool_idx],
                    )
                )

            return self._wait_for_write_statuses(slot_statuses)
        finally:
            self._release_staging_slot(producer_staging_pool_idx)

    def _submit_write_region_units(
        self,
        remote_engine_key: str,
        remote_regions: list[MoRIIOWriteRegion],
        kind: str,
        src_units: list[int],
        dst_units: list[int],
    ) -> list[Any]:
        statuses: list[Any] = []
        if not src_units:
            return statuses
        if len(src_units) != len(dst_units):
            raise ValueError(f"{kind} src/dst unit count mismatch")

        for region_idx, local_region in enumerate(self._write_local_regions):
            if local_region.kind != kind:
                continue
            remote_region = remote_regions[region_idx]
            self._validate_write_region_pair(region_idx, local_region, remote_region)

            groups: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
            for src_unit, dst_unit in zip(src_units, dst_units):
                l_chunk, l_off = self._unit_chunk_offset(local_region, src_unit)
                r_chunk, r_off = self._unit_chunk_offset(remote_region, dst_unit)
                key = (l_chunk, r_chunk)
                if key not in groups:
                    groups[key] = ([], [], [])
                groups[key][0].append(l_off)
                groups[key][1].append(r_off)
                groups[key][2].append(local_region.unit_bytes)

            for (l_chunk, r_chunk), (
                local_offsets,
                remote_offsets,
                sizes,
            ) in groups.items():
                session = self._get_or_build_write_session(
                    remote_engine_key,
                    region_idx,
                    l_chunk,
                    r_chunk,
                    local_region,
                    remote_region,
                )
                merged_local, merged_remote, merged_sizes = (
                    self.merge_contiguous_blocks(local_offsets, remote_offsets, sizes)
                )
                status = session.batch_write(
                    merged_local,
                    merged_remote,
                    merged_sizes,
                    self.moriio_engine.allocate_transfer_uid(),
                )
                statuses.append(status)

        return statuses

    def _validate_write_region_pair(
        self,
        region_idx: int,
        local_region: MoRIIOWriteRegion,
        remote_region: MoRIIOWriteRegion,
    ) -> None:
        if local_region.kind != remote_region.kind:
            raise ValueError(
                f"region {region_idx} kind mismatch: local={local_region.kind}, "
                f"remote={remote_region.kind}"
            )
        if local_region.unit_bytes != remote_region.unit_bytes:
            raise ValueError(
                f"region {region_idx} unit size mismatch: "
                f"local={local_region.unit_bytes}, remote={remote_region.unit_bytes}"
            )

    def _unit_chunk_offset(
        self,
        region: MoRIIOWriteRegion,
        unit_id: int,
    ) -> tuple[int, int]:
        if unit_id < 0 or unit_id >= region.total_units:
            raise ValueError(
                f"{region.kind} unit {unit_id} out of bounds "
                f"(total_units={region.total_units})"
            )
        chunk_idx = unit_id // region.units_per_chunk
        offset = (unit_id % region.units_per_chunk) * region.unit_bytes
        if chunk_idx >= len(region.chunks):
            raise ValueError(
                f"{region.kind} chunk {chunk_idx} out of bounds "
                f"(chunks={len(region.chunks)})"
            )
        return chunk_idx, offset

    def _get_or_build_write_session(
        self,
        remote_engine_key: str,
        region_idx: int,
        local_chunk_idx: int,
        remote_chunk_idx: int,
        local_region: MoRIIOWriteRegion,
        remote_region: MoRIIOWriteRegion,
    ) -> Any:
        cache_key = (
            remote_engine_key,
            region_idx,
            local_chunk_idx,
            remote_chunk_idx,
        )
        session = self._write_session_cache.get(cache_key)
        if session is not None:
            return session

        with self._write_session_lock:
            session = self._write_session_cache.get(cache_key)
            if session is None:
                local_md = self.moriio_wrapper.get_unpack_memory_metadata(
                    local_region.chunks[local_chunk_idx]
                )
                remote_md = self.moriio_wrapper.get_unpack_memory_metadata(
                    remote_region.chunks[remote_chunk_idx]
                )
                session = self.moriio_wrapper.build_session(local_md, remote_md)
                self._write_session_cache[cache_key] = session
            return session

    @staticmethod
    def _write_status_ok(status: Any) -> bool:
        """Return whether one transfer status succeeded, logging on failure.

        A status that does not expose ``Succeeded`` is treated as success
        (matching the RDMA fast path, which relies on the aggregate wait_all
        result for such objects).
        """
        succeeded = getattr(status, "Succeeded", None)
        if succeeded is None or succeeded():
            return True
        message = getattr(status, "Message", lambda: "<unknown>")()
        code = getattr(status, "Code", lambda: "<unknown>")()
        logger.error("MoRIIO write status failed: %s (code=%s)", message, code)
        return False

    def _wait_for_write_statuses(self, statuses: list[Any]) -> bool:
        if not statuses:
            return True

        # FABRIC completion is driven by the status wait-callback
        # (hipEventSynchronize). A bounded WaitAll timeout takes mori's
        # cv_.wait_until path, which is never notified for fabric and blocks the
        # full timeout. Use per-status blocking Wait() (as the mori benchmark
        # does) so the callback fires and the transfer finalizes immediately.
        if self._moriio_use_fabric:
            ok = True
            for status in statuses:
                status.Wait()
                # Evaluate first so every failing status is logged.
                ok = self._write_status_ok(status) and ok
            return ok

        try:
            result = self.moriio_engine.wait_all(
                statuses, timeout_ms=self._write_transfer_timeout_ms
            )
        except TypeError:
            result = self.moriio_engine.wait_all(statuses)
        except Exception:
            logger.exception("MoRIIO wait_all failed")
            return False

        if not self._wait_result_succeeded(result):
            logger.error("MoRIIO wait_all returned non-success status: %s", result)
            return False

        return all(self._write_status_ok(status) for status in statuses)

    def _wait_result_succeeded(self, result: Any) -> bool:
        if _MORIIO_AVAILABLE:
            try:
                return result == StatusCode.SUCCESS
            except Exception:
                pass
        succeeded = getattr(result, "Succeeded", None)
        if succeeded is not None:
            return bool(succeeded())
        return result in (0, True, None)

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
        remote_tp_rank = self.tp_rank % int(remote_tp_size or 1)
        port_offset = get_port_offset(
            remote_dp_rank, remote_tp_rank, int(remote_tp_size or 1)
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
        logger.info(
            "Initiating background handshake for req %s -> %s",
            req_id,
            remote_engine_id,
        )

        host = meta.remote_host
        port = int(meta.remote_handshake_port)
        tp_size = int(meta.tp_size)
        remote_dp_size = int(meta.remote_dp_size)

        def _on_all_done(_f: Future[Any], entry=(req_id, meta)):
            logger.info("All handshakes completed for req %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True

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

    def _pop_done_transfers(self) -> set[str]:
        done_req_ids: set[str] = set()
        with self.moriio_wrapper.lock:
            to_remove = []
            for req_id, status_list in self._recving_transfers.items():
                if status_list[-1].Succeeded():
                    done_req_ids.add(req_id)
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

            return done_req_ids

    def get_finished(self) -> tuple[set[int], set[str]] | KVConnectorOutput:
        """Return the sets of finished sending and receiving request IDs.

        Called by the worker each step via ``async_proc_aggregation``.

        Returns:
            ``(done_sending, done_recving)`` tuple.
        """
        if self.transfer_mode == TransferMode.WRITE_PUSH:
            if self.is_producer:
                self._sweep_orphaned_completed_prefills()
            else:
                self._sweep_expired_write_recvs()
            with self.moriio_wrapper.lock:
                done_sending = set(self.done_sending)
                done_recving = set(self.done_recving)
                failed_recving = set(self.failed_recving)
                self.done_sending.clear()
                self.done_recving.clear()
                self.failed_recving.clear()
            return KVConnectorOutput(
                finished_sending=done_sending,
                finished_recving=done_recving,
                failed_recving=failed_recving,
            )

        done_recving = self._pop_done_transfers()
        if self.is_producer:
            with self.moriio_wrapper.lock:
                done_sending = self.done_sending.copy()
                self.done_sending.clear()
        else:
            with self.moriio_wrapper.lock:
                if self.done_sending:
                    logger.warning(
                        "Consumer received %d stale done_sending notifications "
                        "(single-machine port collision?) — discarding: %s",
                        len(self.done_sending),
                        self.done_sending,
                    )
                    self.done_sending.clear()
            done_sending = set()
        return done_sending, done_recving

    def get_finished_recv_blocks(self) -> list[int]:
        with self._fence_lock:
            blocks = self._blocks_pending_fence
            self._blocks_pending_fence = []
        return blocks


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
        self.transfer_mode = TransferMode(
            kv_transfer_config.get("transfer_mode", TransferMode.WRITE_PUSH.value)
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

        # Pending requests: req_id -> (Sequence, block_table, local_slot_index)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int], int]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int], int]] = {}

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

        for req_id, (req, block_ids, slot_idx) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            req.kv_transfer_params["local_slot_index"] = slot_idx
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.transfer_mode == TransferMode.WRITE_PUSH:
            for req_id, (req, block_ids, slot_idx) in self._reqs_need_save.items():
                assert req.kv_transfer_params is not None
                req.kv_transfer_params["local_slot_index"] = slot_idx
                if req.kv_transfer_params.get("transfer_id") is None:
                    req.kv_transfer_params["transfer_id"] = req_id
                meta.add_new_req_to_save(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
        logger.debug(
            "Built connector metadata with %d recv and %d save requests: %s",
            len(self._reqs_need_recv),
            len(self._reqs_need_save),
            list(self._reqs_need_recv.keys()),
        )
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
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

        slot_index = getattr(seq, "per_req_cache_group", -1)

        # Decode side: queue for remote KV loading
        if params.get("do_remote_prefill"):
            assert (
                not self.is_producer
            ), "Only the decode (consumer) side handles do_remote_prefill"
            self._reqs_need_recv[seq.id] = (seq, seq.block_table, slot_index)
            params["do_remote_prefill"] = False
            params["local_slot_index"] = slot_index
            logger.debug(
                "Queued req %s for remote KV loading (%d blocks, slot=%d)",
                seq.id,
                len(seq.block_table),
                slot_index,
            )

        if self.transfer_mode == TransferMode.WRITE_PUSH and params.get(
            "do_remote_decode"
        ):
            assert self.is_producer, "Only the producer side handles do_remote_decode"
            if params.get("transfer_id") is None:
                params["transfer_id"] = seq.id
            params["local_slot_index"] = slot_index
            self._reqs_need_save[seq.id] = (seq, seq.block_table, slot_index)
            logger.debug(
                "Queued req %s for MoRIIO write save (%d blocks, slot=%d)",
                seq.id,
                len(seq.block_table),
                slot_index,
            )

    def request_finished(self, seq: Sequence) -> None:
        """Populate KV transfer output metadata when a request completes.

        On the producer side this allows the proxy to forward block info
        to the decode instance.  On the consumer side this cleans up
        the transfer_id mapping.
        """
        # Attach output metadata for the proxy to relay
        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        drafts = getattr(seq, "spec_token_ids", None)
        draft_token_ids = (
            [int(x) for x in drafts] if drafts is not None and len(drafts) else []
        )
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": seq.block_table.copy(),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "remote_handshake_port": self.base_handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "draft_token_ids": draft_token_ids,
            "local_slot_index": getattr(seq, "per_req_cache_group", -1),
        }

        # Clean up transfer ID mapping on the consumer side
        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)


def _zmq_ctx(socket_type: int, addr: str):
    """Context manager for a ZMQ socket with role-appropriate bind semantics.

    ROUTER sockets bind; DEALER/REQ sockets connect.
    """
    return zmq_socket_ctx(addr, socket_type, bind=(socket_type == zmq.ROUTER))
