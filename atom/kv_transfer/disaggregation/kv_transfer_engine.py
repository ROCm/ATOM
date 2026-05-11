# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV Cache Connector for Disaggregated Prefill-Decode (P/D) Architecture.

This module implements the KV cache transfer mechanism for disaggregated
inference, where prefill and decode stages run on separate GPU instances.
It uses RDMA-based zero-copy transfers via the MoRIIO library for efficient
KV cache migration between producer (prefill) and consumer (decode) nodes.

Key Components:
    - KVConnector: Worker-side connector managing RDMA transfers and handshakes.
    - KVConnectorScheduler: Scheduler-side connector coordinating transfer state.
    - MoRIIOWrapper: Abstraction over the MoRIIO RDMA engine.
    - ConnectorMetadata: Transfer metadata exchanged between scheduler and workers.

Transfer Modes:
    - Read mode: The decode instance reads KV cache directly from prefill memory.
      Prefill completes first, then decode reads the blocks via RDMA.

Architecture::

    Proxy (service discovery) <-> Prefill Instance <--RDMA--> Decode Instance
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import msgpack
import msgspec
import numpy as np
import zmq

if TYPE_CHECKING:
    import torch

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    EngineId,
    RemoteAllocInfo,
    ReqId,
    ReqMeta,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_port,
    make_zmq_path,
    make_zmq_socket,
    zmq_socket_ctx,
)
from atom.utils.network import get_ip
from aiter.dist.parallel_state import get_dp_group, get_tp_group

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# MoRIIO availability check
# ---------------------------------------------------------------------------

_MORIIO_AVAILABLE = False
try:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        MemoryLocationType,
        PollCqMode,
        RdmaBackendConfig,
    )

    _MORIIO_AVAILABLE = True
    logger.info("MoRIIO RDMA library loaded successfully")
except ImportError:
    logger.warning(
        "MoRIIO is not available — KV cache disaggregation will not work. "
        "Install the mori package to enable RDMA transfers."
    )


# ---------------------------------------------------------------------------
# Msgspec metadata structs
# ---------------------------------------------------------------------------


class MoRIIOAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Serializable metadata exchanged during the RDMA handshake."""

    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: Optional[list[int]] = None
    num_blocks: int = 0
    block_len: int = 0
    attn_backend_name: str = "aiter"


# ---------------------------------------------------------------------------
# Enums & role management
# ---------------------------------------------------------------------------


class Role(Enum):
    """Role of the current engine instance in the P/D architecture."""

    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOT_INITIALIZED = "not_initialized"


def convert_virtual_to_physical_pages(
    virtual_pages: list[int],
    virtual_block_size: int = 16,
    physical_block_size: int = 1,
) -> list[int]:
    """Expand virtual (coarse) block IDs into physical (fine-grained) page IDs.

    In paged-attention the scheduler works with *virtual* blocks of
    ``virtual_block_size`` tokens, but the RDMA transfer operates at
    ``physical_block_size`` granularity.

    Args:
        virtual_pages: List of virtual block IDs.
        virtual_block_size: Tokens per virtual block.
        physical_block_size: Tokens per physical block.

    Returns:
        Expanded list of physical page IDs.
    """
    block_ratio = virtual_block_size // physical_block_size
    physical_pages: list[int] = []
    for vp in virtual_pages:
        start = vp * block_ratio
        physical_pages.extend(range(start, start + block_ratio))
    return physical_pages


class _RoleManager:
    """Thread-safe singleton that tracks the P/D role of this process.

    Use the module-level :func:`get_role` / :func:`set_role` helpers
    instead of accessing this class directly.
    """

    _instance: Optional[_RoleManager] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._role: Role = Role.NOT_INITIALIZED

    @classmethod
    def get_instance(cls) -> _RoleManager:
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = object.__new__(cls)
                    instance.__init__()
                    cls._instance = instance
        return cls._instance

    def set_role(self, role: Role) -> None:
        with self._lock:
            self._role = role

    @property
    def role(self) -> Role:
        return self._role


def set_role(role: Role) -> None:
    """Set the global P/D role for this process."""
    _RoleManager.get_instance().set_role(role)


def get_role() -> Role:
    """Get the global P/D role for this process."""
    return _RoleManager.get_instance().role


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class MoRIIOConstants:
    """Protocol constants for the MoRIIO-based KV connector."""

    # ZMQ handshake message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"

    # Service discovery
    PING_INTERVAL_SECONDS = 5
    MAX_PING_RETRIES = 100

    # Networking
    DEFAULT_HANDSHAKE_PORT = 6301
    DEFAULT_NOTIFY_PORT = "61005"

    # Timeouts
    ABORT_REQUEST_TIMEOUT = 3600


def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return (dp_rank) * tp_size + tp_rank


MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024  # just under 2 GiB


def _chunk_tensor_for_rdma(
    tensor: torch.Tensor, block_size_in_dim0: int = 1
) -> tuple[list[tuple[int, int]], int]:
    """Split a tensor into <2 GiB RDMA-registrable chunks along dim 0.

    Args:
        tensor: contiguous torch.Tensor whose dim-0 is the block (or
            token) axis.
        block_size_in_dim0: elements per logical block in dim 0.
            Non-MLA: 1 (dim 0 = num_blocks).
            MLA: block_size (dim 0 = num_blocks * block_size).

    Returns:
        ``(chunks, blocks_per_chunk)`` where *chunks* is a list of
        ``(data_ptr, size_bytes)`` pairs and *blocks_per_chunk* is
        the number of logical blocks in each full chunk.
    """
    elem_sz = tensor.element_size()
    per_block_bytes = block_size_in_dim0 * tensor.stride(0) * elem_sz
    total_blocks = tensor.shape[0] // block_size_in_dim0
    bpc = max(1, MAX_RDMA_CHUNK_BYTES // per_block_bytes)
    chunks: list[tuple[int, int]] = []
    base = tensor.data_ptr()
    for start in range(0, total_blocks, bpc):
        end = min(start + bpc, total_blocks)
        chunks.append((base + start * per_block_bytes, (end - start) * per_block_bytes))
    return chunks, bpc


# ===================================================================
# MoRIIOWrapper — thin abstraction over the RDMA engine
# ===================================================================


class MoRIIOWrapper:
    """Low-level wrapper around a MoRIIO ``IOEngine``.

    Provides helper methods for memory registration, session management,
    and asynchronous RDMA read/write operations.  Both producer and
    consumer code paths share this wrapper.

    Thread-safety:
        ``transfer_status``, ``done_req_ids``, and ``done_write_cache_req_ids``
        are guarded by ``self.lock``.  The ZMQ socket cache ``self._sockets``
        is *not* thread-safe — callers must ensure ``send_notify`` is invoked
        from a single thread.

    Args:
        moriio_engine: MoRIIO IOEngine instance.
        tp_rank: Tensor parallel rank.
        dp_rank: Data parallel rank.
    """

    def __init__(
        self,
        moriio_engine: Any = None,
        tp_rank: int = 0,
        dp_rank: int = 0,
    ):
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moriio_engine = moriio_engine

        self.remote_memory_metadata: Any = None
        self.local_memory_registered: bool = False
        # Raw-pointer registration produces multiple MemoryDesc per layer
        # (e.g. K + V on MHA) to keep each ibv_reg_mr below the AINIC ~2 GiB
        # limit. We retain handles here purely so they aren't GC'd; nothing
        # in the active session/transfer path indexes into this list.
        self.local_memory_descs: list[Any] = []
        self.transfer_status: list[Any] = []
        self.remote_engine_ip: str | None = None
        self.notify_port: int | None = None

        self.lock = threading.Lock()
        self.done_req_ids: list[str] = []
        self.done_write_cache_req_ids: list[str] = []
        self.notify_thread: threading.Thread | None = None

        # ZMQ socket cache keyed by endpoint path
        self._sockets: dict[str, zmq.Socket] = {}

    def set_moriio_engine(self, moriio_engine: Any) -> None:
        """Assign the MoRIIO engine (must not be None)."""
        if moriio_engine is None:
            raise ValueError("Cannot assign a None MoRIIO engine")
        self.moriio_engine = moriio_engine

    def set_backend_type(self, backend_type, backend_config=None):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        if backend_config is not None:
            self.moriio_engine.create_backend(backend_type, backend_config)
        else:
            self.moriio_engine.create_backend(backend_type)

    def get_agent_metadata(self):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        engine_metadata = self.moriio_engine.get_engine_desc()
        engine_metadata_packed = engine_metadata.pack()
        return engine_metadata_packed

    def register_remote_engine(self, remote_packed_engine_metadata):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        consumer_engine_metadata = EngineDesc.unpack(remote_packed_engine_metadata)
        self.moriio_engine.register_remote_engine(consumer_engine_metadata)
        logger.info(
            "Registered remote engine with key: %s", consumer_engine_metadata.key
        )
        return consumer_engine_metadata.key

    def register_local_buffer(self, ptr: int, size: int, device_id: int) -> bytes:
        """Register one raw GPU memory region with MoRIIO.

        Using ``register_memory(ptr, size, ...)`` directly (instead of
        ``register_torch_tensor``) lets callers split a single tensor into
        multiple smaller regions, which is required to stay under the
        AINIC ~2 GiB ``ibv_reg_mr`` limit.
        """
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        try:
            desc = self.moriio_engine.register_memory(
                ptr, size, device_id, MemoryLocationType.GPU
            )
            assert desc is not None, "register_memory returned None"
            packed = desc.pack()
        except Exception as e:
            raise ValueError(f"Failed to register local memory: {e}") from e
        self.local_memory_descs.append(desc)
        self.local_memory_registered = True
        return packed

    def register_local_tensor(self, tensor):
        """Back-compat helper: register an entire contiguous torch tensor.

        Prefer :meth:`register_local_buffer` from new code so that callers
        explicitly choose how to chunk large tensors before registration.
        """
        if not tensor.is_contiguous():
            raise RuntimeError("input tensor must be contiguous")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        device_id = tensor.device.index if tensor.device.index is not None else -1
        return self.register_local_buffer(ptr, size, device_id)

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        return MemoryDesc.unpack(packed_memory_metadata)

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        tmp = self.moriio_engine.create_session(
            local_memory_metadata, remote_memory_metadata
        )

        return tmp

    def read_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        transfer_status = session.batch_read(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )
        return transfer_status

    def write_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        write_uid = self.moriio_engine.allocate_transfer_uid()

        transfer_status = session.batch_write(
            local_offset, remote_offset, transfer_size_byte, write_uid
        )
        with self.lock:
            self.transfer_status.append(transfer_status)

    def write_remote_data_single(
        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        transfer_status = self.sessions[sess_idx].write(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )
        with self.lock:
            self.transfer_status.append(transfer_status)

    def waiting_for_transfer_complete(self):
        if not self.transfer_status:
            return

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        "Transfer failed: %s, Code: %s", status.Message(), status.Code()
                    )
                    raise ValueError("MoRIIO transfer failed!")
            except Exception as e:
                logger.error("Transfer %s failed: %s", status, e)
                raise

    def async_wait_reqid(self):
        assert self.notify_port is not None, "Notify port cannot be None"

        if self.notify_thread is not None:
            return

        def _async_wait():
            host = "*"
            path = make_zmq_path("tcp", host, self.notify_port)
            logger.info("Node starting to listen notify from path = %s", path)

            with _zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._dispatch_message(msg)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        raise ValueError(f"Error processing message: {e}") from e

        self.notify_thread = threading.Thread(
            target=_async_wait, daemon=True, name="moriio-notify-listener"
        )
        self.notify_thread.start()

    def _dispatch_message(self, msg: bytes) -> None:
        """Route an incoming ZMQ message to the appropriate handler.

        Message formats:
            - msgpack dict with ``req_id``: remote block allocation (producer only)
            - UTF-8 string prefixed with ``cmpl``: transfer completion signal
        """
        # Try msgpack structured message first (block allocation from decode)
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict) and "req_id" in data:
                self._handle_block_alloc_message(data)
                return
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException):
            pass

        # Fall back to string-encoded completion message
        try:
            msg_str = msg.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Received non-decodable message of %d bytes", len(msg))
            return

        if msg_str.startswith(MoRIIOConstants.COMPLETION_PREFIX):
            self._handle_completion_message(msg_str)
        else:
            raise ValueError(f"Unrecognized message format: {msg_str!r}")

    def _handle_block_alloc_message(self, data: dict) -> None:
        """Process a remote block allocation notification (producer side)."""
        assert (
            get_role() == Role.PRODUCER
        ), "Only producer should receive block alloc messages"
        req_id = data["req_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        assert (
            len(block_notify_list) > 0
        ), "block_notify_list cannot be empty in remote allocate message"

        with self.lock:
            self.done_remote_allocate_req_dict[req_id] = RemoteAllocInfo(
                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank
            )

    def _handle_completion_message(self, msg: str) -> None:
        """Record a transfer completion notification."""
        with self.lock:
            if get_role() == Role.PRODUCER:
                self.done_req_ids.append(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def send_notify(
        self,
        req_ids: str | int | list[str | int],
        remote_ip: str,
        remote_port: str | int,
    ) -> None:
        """Notify a remote engine that transfer(s) have completed."""
        if not remote_ip or not remote_port:
            logger.warning("Cannot send notification: missing remote_ip or remote_port")
            return

        path = make_zmq_path("tcp", remote_ip, int(remote_port))

        if path not in self._sockets:
            ctx = zmq.Context.instance()
            self._sockets[path] = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )

        id_list = req_ids if isinstance(req_ids, list) else [req_ids]
        sock = self._sockets[path]
        try:
            for rid in id_list:
                rid_str = str(rid) if isinstance(rid, int) else rid
                if not isinstance(rid_str, str):
                    logger.warning("Skipping non-string req_id of type %s", type(rid))
                    continue
                sock.send_multipart(
                    [MoRIIOConstants.POP_DONE_RECV, rid_str.encode("utf-8")]
                )
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self._sockets.pop(path, None)
            raise

    def pop_finished_req_ids(self) -> set[str]:
        """Return and clear the set of completed send-side request IDs."""
        with self.lock:
            result = set(self.done_req_ids)
            self.done_req_ids.clear()
        return result

    def pop_finished_write_req_ids(self) -> set[str]:
        """Return and clear the set of completed write-side request IDs."""
        with self.lock:
            result = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids.clear()
        return result

    def shutdown(self) -> None:
        """Close all cached ZMQ sockets and release resources."""
        logger.debug(
            "Shutting down MoRIIOWrapper, closing %d sockets", len(self._sockets)
        )
        for path, sock in self._sockets.items():
            try:
                sock.close(linger=0)
            except Exception as e:
                logger.warning("Error closing socket for %s: %s", path, e)
        self._sockets.clear()


# ===================================================================
# KVConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class KVConnector(KVConnectorBase):
    """Worker-side KV cache connector for disaggregated P/D inference.

    Each tensor-parallel worker instantiates one ``KVConnector``.  It is
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
        self._local_ping_port = get_open_port()

        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.proxy_ping_port = kv_transfer_config.get("proxy_ping_port", 36367)
        self.proxy_ip = kv_transfer_config.get("proxy_ip")
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.base_handshake_port = kv_transfer_config.get(
            "handshake_port", MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
        )

        # Compute unique side-channel port for this (dp, tp) rank
        handshake_port = self.base_handshake_port
        self.side_channel_port = handshake_port + get_port_offset(
            self.dp_rank, self.tp_rank
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
        self.done_sending: set[int] = set()

        # Transfer ID mapping (worker side)
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # Start service-discovery ping (only on rank 0)
        if self.tp_rank == 0 and self.dp_rank == 0:
            self._ping_thread = threading.Thread(
                target=self._service_discovery_ping,
                args=(self.zmq_context,),
                daemon=True,
                name="kv-connector-ping",
            )
            self._ping_thread.start()

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register all KV cache tensors for RDMA and start the handshake listener.

        Must be called after model loading and KV cache allocation, before any
        transfers can occur.

        Each K (and V, when present) tensor is split into block-aligned
        chunks of < 2 GiB via :func:`_chunk_tensor_for_rdma` and each
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
            k_chunks, bpc = _chunk_tensor_for_rdma(cache_tensor, bsd0)

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
                v_chunks, _ = _chunk_tensor_for_rdma(v_cache, 1)
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
        notify_port = remote_handshake_port + get_port_offset(
            remote_dp_rank, self.tp_rank
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

    def _service_discovery_ping(self, zmq_context: zmq.Context) -> None:
        """Periodically register with the proxy for service discovery (rank 0 only)."""
        http_endpoint = f"http://{self.request_address}/v1/completions"
        role_code = "P" if self.is_producer else "D"
        retry_count = 0
        msg_index = 1
        proxy_path = f"tcp://{self.proxy_ip}:{self.proxy_ping_port}"

        with zmq_context.socket(zmq.DEALER) as sock:
            sock.connect(proxy_path)

            while True:
                try:
                    registration_data = {
                        "type": "register",
                        "role": role_code,
                        "index": str(msg_index),
                        "request_address": http_endpoint,
                        "handshake_port": self.base_handshake_port,
                        "dp_size": self.dp_size,
                        "tp_size": self.tp_size,
                        "transfer_mode": "read",
                    }
                    sock.send(msgpack.dumps(registration_data))
                    logger.debug(
                        "Ping #%d sent to %s (role=%s)",
                        msg_index,
                        proxy_path,
                        role_code,
                    )
                    retry_count = 0

                except ConnectionRefusedError:
                    logger.info(
                        "Proxy connection refused: %s:%s -> %s",
                        self.local_ip,
                        self._local_ping_port,
                        proxy_path,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error during ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected ping error: %s", e)
                    retry_count += 1
                    if retry_count >= MoRIIOConstants.MAX_PING_RETRIES:
                        logger.error(
                            "Ping failed after %d retries, aborting",
                            MoRIIOConstants.MAX_PING_RETRIES,
                        )
                        raise RuntimeError(
                            f"Service discovery ping failed after {retry_count} retries"
                        ) from e

                finally:
                    time.sleep(MoRIIOConstants.PING_INTERVAL_SECONDS)
                    msg_index += 1

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
                    logger.info("Handshake: sent engine metadata to peer")
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
        port_offset = get_port_offset(remote_dp_rank, self.tp_rank)
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

    def get_finished(self) -> tuple[set[int], set[str]]:
        """Return the sets of finished sending and receiving request IDs.

        Called by the worker each step via ``async_proc_aggregation``.

        Returns:
            ``(done_sending, done_recving)`` tuple.
        """
        done_recving = self._pop_done_transfers()
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
        return done_sending, done_recving


# ===================================================================
# KVConnectorScheduler — scheduler-side connector
# ===================================================================


class KVConnectorScheduler(KVConnectorSchedulerBase):
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
        self.engine_id = "None"
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.parallel_config.data_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.host_ip = get_ip()

        # Pending receive requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

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
            self._reqs_need_recv[seq.id] = (seq, seq.block_table)
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
        # Attach output metadata for the proxy to relay
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": seq.block_table.copy(),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "transfer_id": seq.id,
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
