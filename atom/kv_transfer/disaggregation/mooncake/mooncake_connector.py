# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses Mooncake TransferEngine for RDMA-based push (WRITE) transfers of
KV cache data from producer (prefill) to consumer (decode) nodes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import msgpack
import msgspec
import torch
import zmq

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    ReqId,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.utils import get_open_port, make_zmq_path, zmq_socket_ctx
from atom.utils.network import get_ip
from aiter.dist.parallel_state import get_dp_group, get_tp_group

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# Mooncake availability check
# ---------------------------------------------------------------------------

_MOONCAKE_AVAILABLE = False
try:
    from mooncake.engine import TransferEngine

    _MOONCAKE_AVAILABLE = True
    logger.info("Mooncake TransferEngine loaded successfully")
except ImportError:
    logger.warning(
        "Mooncake is not available — KV cache disaggregation via mooncake "
        "will not work. Install the mooncake package to enable push-mode "
        "RDMA transfers."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOONCAKE_PING_INTERVAL_SECONDS = 5
MOONCAKE_MAX_PING_RETRIES = 100
MOONCAKE_DEFAULT_PROTOCOL = "rdma"
PREFILL_LOOKUP_TIMEOUT = 60
PREFILL_LOOKUP_POLL_INTERVAL = 0.01

# ZMQ side-channel message types
MSG_WRITE_REQUEST = b"write_request"
MSG_WRITE_DONE = b"write_done"
MSG_GET_META = b"get_meta"


# ---------------------------------------------------------------------------
# Metadata struct for bootstrap handshake
# ---------------------------------------------------------------------------


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Serializable metadata exchanged during the mooncake bootstrap."""

    engine_id: str
    rpc_port: int
    kv_caches_base_addr: list[int] | None = None
    num_blocks: int = 0
    block_len: int = 0


def _port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return dp_rank * tp_size + tp_rank


# ===================================================================
# MooncakeConnectorScheduler — scheduler-side connector
# ===================================================================


class MooncakeConnectorScheduler(KVConnectorSchedulerBase):
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

        # Pending requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        params = seq.kv_transfer_params or {}

        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True

        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Producer side: pass completed prefill block_ids to worker
        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self._reqs_need_recv or self._reqs_need_save:
            logger.info(
                "[SCHEDULER] build_connector_meta: %d recv, %d save, " "id_map=%s",
                len(self._reqs_need_recv),
                len(self._reqs_need_save),
                meta.request_id_to_transfer_id,
            )
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        return meta

    def update_state_after_alloc(self, seq: Sequence) -> None:
        params = seq.kv_transfer_params or {}

        if not self.is_producer:
            transfer_id = params.get("transfer_id")
            if transfer_id is not None:
                self.transfer_id_to_request_id[transfer_id] = seq.id
                self.request_id_to_transfer_id[seq.id] = transfer_id

        # Consumer side: queue for remote KV loading
        if params.get("do_remote_prefill"):
            assert (
                not self.is_producer
            ), "Only the decode (consumer) side handles do_remote_prefill"
            self._reqs_need_recv[seq.id] = (seq, seq.block_table)
            params["do_remote_prefill"] = False
            logger.info(
                "[SCHEDULER-CONSUMER] Queued req %s for remote KV recv "
                "(%d blocks), transfer_id=%s, remote_host=%s, "
                "remote_handshake_port=%s",
                seq.id,
                len(seq.block_table),
                params.get("transfer_id"),
                params.get("remote_host"),
                params.get("remote_handshake_port"),
            )

        # Producer side: queue block_ids for the write listener to look up
        if params.get("do_remote_decode"):
            assert self.is_producer, "Only the producer side handles do_remote_decode"
            self._reqs_need_save[seq.id] = (seq, seq.block_table)
            logger.debug(
                "Queued req %s for KV save (%d blocks)",
                seq.id,
                len(seq.block_table),
            )

    def request_finished(self, seq: Sequence) -> None:
        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
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
            "first_token_id": first_token_id,
        }

        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)


# ===================================================================
# MooncakeConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MooncakeConnector(KVConnectorBase):
    """Worker-side KV cache connector using Mooncake push-mode RDMA.

    Mooncake uses a push/WRITE model: the prefill (producer) node writes
    KV cache data directly into the decode (consumer) node's registered
    GPU memory via ``batch_transfer_sync_write``.
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
        self.is_consumer = not self.is_producer

        # Networking / service discovery config
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.proxy_ping_port = kv_transfer_config.get("proxy_ping_port", 36367)
        self.proxy_ip = kv_transfer_config.get("proxy_ip")
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.protocol = kv_transfer_config.get("protocol", MOONCAKE_DEFAULT_PROTOCOL)

        # Side channel port (ZMQ) — deterministic from config for proxy relay
        self.base_handshake_port = kv_transfer_config.get("handshake_port", 6301)
        self._side_channel_port = self.base_handshake_port + _port_offset(
            self.dp_rank, self.tp_rank, self.tp_size
        )

        # --- Mooncake TransferEngine initialization ---
        if not _MOONCAKE_AVAILABLE:
            raise RuntimeError(
                "Mooncake is not installed but kv_connector='mooncake' was requested. "
                "Install the mooncake package to use push-mode RDMA transfers."
            )

        # Determine which RDMA device this TP rank should use.
        # On AMD MI300X, each GPU has a paired RDMA NIC: GPU N → rdmaN.
        # Registering GPU memory with a non-local RDMA NIC fails with
        # EINVAL.  Pass the device name as a filter so Mooncake only
        # creates a context for the local NIC.
        ib_device = kv_transfer_config.get("ib_device", "")
        if not ib_device:
            ib_device = os.environ.get("ATOM_MOONCAKE_IB_DEVICE", "")
        if not ib_device:
            gpu_idx = torch.cuda.current_device()
            ib_device = f"rdma{gpu_idx}"
            logger.info(
                "Auto-selecting RDMA device %s for GPU %d (tp_rank=%d)",
                ib_device,
                gpu_idx,
                self.tp_rank,
            )

        self.transfer_engine = TransferEngine()
        ret = self.transfer_engine.initialize(
            self.local_ip,
            "P2PHANDSHAKE",
            self.protocol,
            ib_device,
        )
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine.initialize() failed (ret={ret}) "
                f"on ip={self.local_ip}, protocol={self.protocol}, "
                f"ib_device={ib_device}"
            )
        self.rpc_port = self.transfer_engine.get_rpc_port()
        self.engine_id = f"{self.local_ip}:{self.rpc_port}"
        logger.info(
            "Mooncake TransferEngine initialized: ip=%s, protocol=%s, "
            "ib_device=%s, rpc_port=%d",
            self.local_ip,
            self.protocol,
            ib_device,
            self.rpc_port,
        )

        # --- KV cache state (populated in register_kv_caches) ---
        self.kv_caches: dict[str, Any] | None = None
        self.kv_caches_base_addr: list[int] = []
        self._per_block_bytes_list: list[int] = []
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.block_len: int = config.kv_cache_block_size
        self.num_blocks: int = 0
        self._per_block_bytes: int = 0

        # --- Producer: completed prefill block_ids cache ---
        # Populated from ConnectorMetadata.reqs_to_save each step.
        # The write listener looks up block_ids here when consumer requests a write.
        self._completed_prefills: dict[ReqId, list[int]] = {}
        self._completed_prefills_lock = threading.Lock()
        self._completed_prefills_cv = threading.Condition(self._completed_prefills_lock)
        self._transfer_refcount: dict[ReqId, int] = {}
        self._transfer_refcount_lock = threading.Lock()

        # --- Consumer: pending receive tracking ---
        self._pending_recv: set[ReqId] = set()
        self._pending_recv_blocks: dict[ReqId, list[int]] = {}
        self._notification_port = get_open_port()

        # --- Completion tracking ---
        self.done_sending: set[str] = set()
        self.done_recving: set[str] = set()
        self._completion_lock = threading.Lock()

        # --- GPU memory fence: blocks pending coherence enforcement ---
        self._blocks_pending_fence: list[int] = []
        self._fence_lock = threading.Lock()

        # --- Transfer ID mapping (worker side) ---
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # --- Producer: thread pool for RDMA writes ---
        if self.is_producer:
            self._send_executor = ThreadPoolExecutor(
                max_workers=kv_transfer_config.get("num_worker_threads", 16),
                thread_name_prefix="mooncake-send-worker",
            )

        # --- ZMQ for metadata exchange ---
        self.zmq_context = zmq.Context()

        # --- Producer: persistent socket cache for write-done notifications ---
        self._notify_sockets: dict[str, zmq.Socket] = {}
        self._notify_sockets_lock = threading.Lock()

        # --- Msgspec encoder/decoder for bootstrap metadata ---
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

        # --- Service discovery ping (rank 0 only) ---
        if self.tp_rank == 0 and self.dp_rank == 0:
            self._ping_thread = threading.Thread(
                target=self._service_discovery_ping,
                args=(self.zmq_context,),
                daemon=True,
                name="mooncake-ping",
            )
            self._ping_thread.start()

    # -----------------------------------------------------------------
    # Service discovery
    # -----------------------------------------------------------------

    def _service_discovery_ping(self, zmq_context: zmq.Context) -> None:
        """Periodically register with the proxy (rank 0 only)."""
        grpc_endpoint = f"http://{self.request_address}/v1/completions"
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
                        "request_address": grpc_endpoint,
                        "rpc_port": self.rpc_port,
                        "handshake_port": self.base_handshake_port,
                        "dp_size": self.dp_size,
                        "tp_size": self.tp_size,
                        "transfer_mode": "write",
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
                        "Proxy connection refused: %s -> %s",
                        self.local_ip,
                        proxy_path,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error during ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected ping error: %s", e)
                    retry_count += 1
                    if retry_count >= MOONCAKE_MAX_PING_RETRIES:
                        logger.error(
                            "Ping failed after %d retries, aborting",
                            MOONCAKE_MAX_PING_RETRIES,
                        )
                        raise RuntimeError(
                            f"Service discovery ping failed after "
                            f"{retry_count} retries"
                        ) from e

                finally:
                    time.sleep(MOONCAKE_PING_INTERVAL_SECONDS)
                    msg_index += 1

    # -----------------------------------------------------------------
    # KVConnectorBase: register_kv_caches
    # -----------------------------------------------------------------

    # RDMA MR size limit — must stay under device max_mr_size (2 GiB on
    # AMD ROCm RDMA).  Mooncake truncates silently, leaving addresses
    # beyond the limit unregistered.
    _MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register all KV cache tensors with the Mooncake TransferEngine.

        Must be called once after model loading and KV cache allocation.
        Starts the side-channel listener threads for write coordination.

        Large tensors (> 2 GiB) are split into chunks for RDMA memory
        registration to stay within the device max_mr_size.  The logical
        per-tensor base addresses (used for transfer offset computation)
        remain unchanged.

        For fp8 KV caches, k_scale and v_scale tensors are also registered
        so that quantization scale factors are transferred alongside data.
        """
        self.kv_caches = kv_caches

        # Logical per-tensor info (for transfer address computation)
        tensor_ptrs: list[int] = []
        tensor_sizes: list[int] = []

        # Chunked regions (for RDMA registration)
        reg_ptrs: list[int] = []
        reg_sizes: list[int] = []

        def _register_tensor(tensor):
            self.kv_caches_base_addr.append(tensor.data_ptr())
            sz = tensor.element_size()
            if is_mla:
                bpb = self.block_len * tensor.stride(0) * sz
            else:
                bpb = tensor.stride(0) * sz
            self._per_block_bytes_list.append(bpb)

            base = tensor.data_ptr()
            total = tensor.numel() * sz
            tensor_ptrs.append(base)
            tensor_sizes.append(total)

            # Chunk for RDMA registration
            offset = 0
            while offset < total:
                chunk = min(self._MAX_RDMA_CHUNK_BYTES, total - offset)
                reg_ptrs.append(base + offset)
                reg_sizes.append(chunk)
                offset += chunk

        for layer_name, kv_cache in kv_caches.items():
            k_cache = kv_cache.k_cache
            v_cache = kv_cache.v_cache
            is_mla = v_cache is None

            if self.kv_cache_shape is None:
                self.kv_cache_shape = k_cache.shape

            _register_tensor(k_cache)

            if not is_mla:
                _register_tensor(v_cache)

            if kv_cache.k_scale is not None:
                _register_tensor(kv_cache.k_scale)
            if kv_cache.v_scale is not None:
                _register_tensor(kv_cache.v_scale)

        logger.info(
            "Registering %d tensors as %d RDMA chunks " "(max_chunk=%.2f GiB)",
            len(tensor_ptrs),
            len(reg_ptrs),
            self._MAX_RDMA_CHUNK_BYTES / (1024**3),
        )
        for i, (ptr, sz_bytes) in enumerate(zip(reg_ptrs, reg_sizes)):
            logger.info(
                "  reg_chunk[%d] ptr=0x%x  size=%d (%.2f GiB)",
                i,
                ptr,
                sz_bytes,
                sz_bytes / (1024**3),
            )

        ret = self.transfer_engine.batch_register_memory(reg_ptrs, reg_sizes)
        if ret != 0:
            logger.error(
                "batch_register_memory FAILED (ret=%d). "
                "Trying individual registration as fallback...",
                ret,
            )
            for i, (ptr, sz_bytes) in enumerate(zip(reg_ptrs, reg_sizes)):
                r = self.transfer_engine.register_memory(ptr, sz_bytes)
                logger.info(
                    "  register_memory[%d] ptr=0x%x size=%d (%.2f GiB) " "=> ret=%d",
                    i,
                    ptr,
                    sz_bytes,
                    sz_bytes / (1024**3),
                    r,
                )
        else:
            logger.info(
                "batch_register_memory OK (%d chunks from %d tensors)",
                len(reg_ptrs),
                len(tensor_ptrs),
            )

        # Block geometry from last layer
        last_kv = list(kv_caches.values())[-1]
        k = last_kv.k_cache
        is_mla = last_kv.v_cache is None
        sz = k.element_size()

        if is_mla:
            self.num_blocks = k.shape[0] // self.block_len
            self._per_block_bytes = self.block_len * k.stride(0) * sz
        else:
            self.num_blocks = k.shape[0]
            self._per_block_bytes = k.stride(0) * sz

        logger.info(
            "Mooncake registered %d tensors (%d RDMA chunks, %d blocks, "
            "per_block_bytes=%s, block_len=%d, is_mla=%s, "
            "k_shape=%s, k_stride=%s, elem_sz=%d, "
            "tensor_sizes_MB=%s, chunk_sizes_MB=%s)",
            len(self.kv_caches_base_addr),
            len(reg_ptrs),
            self.num_blocks,
            sorted(set(self._per_block_bytes_list)),
            self.block_len,
            is_mla,
            k.shape,
            k.stride(),
            sz,
            [s // (1024 * 1024) for s in sorted(set(tensor_sizes))],
            [s // (1024 * 1024) for s in sorted(set(reg_sizes))],
        )

        logger.info(
            "Mooncake KV registration complete: role=%s, engine_id=%s, "
            "rpc_port=%d, num_regions=%d, num_blocks=%d, "
            "base_addrs=[%s]",
            "PRODUCER" if self.is_producer else "CONSUMER",
            self.engine_id,
            self.rpc_port,
            len(self.kv_caches_base_addr),
            self.num_blocks,
            ", ".join(f"0x{a:x}" for a in self.kv_caches_base_addr[:4]),
        )

        # Build metadata for bootstrap exchange
        self._local_metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            rpc_port=self.rpc_port,
            kv_caches_base_addr=self.kv_caches_base_addr,
            num_blocks=self.num_blocks,
            block_len=self.block_len,
        )

        # Start side channel threads
        if self.is_producer:
            self._write_listener_thread = threading.Thread(
                target=self._write_listener,
                daemon=True,
                name="mooncake-write-listener",
            )
            self._write_listener_thread.start()
        else:
            self._notification_listener_thread = threading.Thread(
                target=self._notification_listener,
                daemon=True,
                name="mooncake-notify-listener",
            )
            self._notification_listener_thread.start()

    # -----------------------------------------------------------------
    # KVConnectorBase: start_load_kv
    # -----------------------------------------------------------------

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate KV transfers for pending requests.

        **Producer side**: Cache completed prefill block_ids from
        ``metadata.reqs_to_save`` so the write listener can look them up.

        **Consumer side**: For each pending recv request, connect to the
        producer's ZMQ side channel and send a write request with our
        memory addresses and block allocation.
        """
        if metadata is None:
            return

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        # Producer: cache block_ids from completed prefills
        if self.is_producer:
            for req_id, meta in metadata.reqs_to_save.items():
                with self._completed_prefills_cv:
                    self._completed_prefills[req_id] = meta.local_block_ids
                    self._completed_prefills_cv.notify_all()
                logger.info(
                    "[PRODUCER] Cached %d prefill blocks for req %s",
                    len(meta.local_block_ids),
                    req_id,
                )
            return

        # Consumer: send write requests to producer
        if not metadata.reqs_to_recv:
            return

        logger.info(
            "[CONSUMER] start_load_kv: %d reqs_to_recv, id_map=%s",
            len(metadata.reqs_to_recv),
            metadata.request_id_to_transfer_id,
        )

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_tp_size = meta.tp_size
            if remote_tp_size != self.tp_size:
                remote_tp_rank = self.tp_rank % remote_tp_size
            else:
                remote_tp_rank = self.tp_rank
            remote_port = meta.remote_handshake_port + _port_offset(
                meta.remote_dp_rank, remote_tp_rank, remote_tp_size
            )
            remote_addr = make_zmq_path("tcp", meta.remote_host, remote_port)

            unique_bpb = sorted(set(self._per_block_bytes_list))
            logger.info(
                "[CONSUMER] Sending write_request for req %s (transfer_id=%s) "
                "to %s (handshake_port=%d, dp_rank=%d, "
                "local_tp=%d, remote_tp=%d/%d), "
                "dst_block_ids=%s, num_regions=%d, "
                "bytes/block=%s, num_blocks=%d",
                req_id,
                meta.transfer_id,
                remote_addr,
                meta.remote_handshake_port,
                meta.remote_dp_rank,
                self.tp_rank,
                remote_tp_rank,
                remote_tp_size,
                meta.local_block_ids[:10],
                len(self.kv_caches_base_addr),
                unique_bpb,
                self.num_blocks,
            )

            write_request = msgpack.dumps(
                {
                    "request_id": req_id,
                    "transfer_id": meta.transfer_id,
                    "consumer_host": self.local_ip,
                    "consumer_rpc_port": self.rpc_port,
                    "consumer_base_addrs": self.kv_caches_base_addr,
                    "dst_block_ids": meta.local_block_ids,
                    "notify_host": self.local_ip,
                    "notify_port": self._notification_port,
                    "consumer_tp_size": self.tp_size,
                }
            )

            with self._notify_sockets_lock:
                sock = self._notify_sockets.get(remote_addr)
                if sock is None:
                    sock = self.zmq_context.socket(zmq.DEALER)
                    sock.setsockopt(zmq.LINGER, 5000)
                    sock.setsockopt(zmq.SNDHWM, 0)
                    sock.connect(remote_addr)
                    self._notify_sockets[remote_addr] = sock
                sock.send_multipart([MSG_WRITE_REQUEST, write_request])

            self._pending_recv.add(req_id)
            self._pending_recv_blocks[req_id] = list(meta.local_block_ids)
            logger.info(
                "[CONSUMER] write_request sent for req %s to %s",
                req_id,
                remote_addr,
            )

    # -----------------------------------------------------------------
    # KVConnectorBase: get_finished
    # -----------------------------------------------------------------

    def get_finished(self) -> tuple[set, set]:
        """Return ``(done_sending, done_recving)`` and clear internal sets."""
        with self._completion_lock:
            ds = self.done_sending.copy()
            dr = self.done_recving.copy()
            self.done_sending.clear()
            self.done_recving.clear()
        if ds or dr:
            logger.info(
                "[%s] get_finished: sending=%s, recving=%s",
                "PRODUCER" if self.is_producer else "CONSUMER",
                ds,
                dr,
            )
        return ds, dr

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed RDMA receives."""
        with self._fence_lock:
            blocks = self._blocks_pending_fence
            self._blocks_pending_fence = []
        return blocks

    # -----------------------------------------------------------------
    # Producer: write listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _write_listener(self) -> None:
        """Accept write requests from consumers and dispatch RDMA writes."""
        path = make_zmq_path("tcp", "*", self._side_channel_port)
        logger.info("Mooncake write listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                identity, msg_type = parts[0], parts[1]

                if msg_type == MSG_GET_META:
                    encoded = self._encoder.encode(self._local_metadata)
                    sock.send_multipart([identity, b"", encoded])
                    logger.debug("Sent metadata to peer")

                elif msg_type == MSG_WRITE_REQUEST:
                    request_data = msgpack.loads(parts[2])
                    logger.info(
                        "[PRODUCER] Received write_request for req %s "
                        "(transfer_id=%s, consumer=%s:%s)",
                        request_data["request_id"],
                        request_data.get("transfer_id"),
                        request_data.get("consumer_host"),
                        request_data.get("consumer_rpc_port"),
                    )
                    self._send_executor.submit(self._execute_transfer, request_data)

                else:
                    logger.error("Unknown message type: %s", msg_type)

    # -----------------------------------------------------------------
    # Producer: execute RDMA write
    # -----------------------------------------------------------------

    def _execute_transfer(self, request_data: dict) -> None:
        """Compute offsets and perform RDMA write for a single request."""
        req_id = request_data["request_id"]
        transfer_id = request_data.get("transfer_id", req_id)
        consumer_host = request_data["consumer_host"]
        consumer_rpc_port = request_data["consumer_rpc_port"]
        consumer_base_addrs = request_data["consumer_base_addrs"]
        dst_block_ids = request_data["dst_block_ids"]
        notify_host = request_data["notify_host"]
        notify_port = request_data["notify_port"]
        consumer_tp_size = request_data.get("consumer_tp_size", self.tp_size)
        consumers_per_rank = max(1, consumer_tp_size // self.tp_size)

        logger.info(
            "[PRODUCER] _execute_transfer: req_id=%s, transfer_id=%s, "
            "consumer=%s:%s, dst_blocks=%d",
            req_id,
            transfer_id,
            consumer_host,
            consumer_rpc_port,
            len(dst_block_ids),
        )

        # Look up producer's block_ids by transfer_id (= prefill's seq.id),
        # not request_id (= decode's seq.id).
        src_block_ids = self._wait_for_prefill_blocks(transfer_id)
        if src_block_ids is None:
            logger.error(
                "[PRODUCER] Timed out waiting for prefill blocks for "
                "transfer_id=%s (req_id=%s). Available keys: %s",
                transfer_id,
                req_id,
                list(self._completed_prefills.keys()),
            )
            return

        target = f"{consumer_host}:{consumer_rpc_port}"

        # Verify the consumer's segment is discoverable via P2P handshake
        if hasattr(self.transfer_engine, "get_first_buffer_address"):
            remote_buf = self.transfer_engine.get_first_buffer_address(target)
            logger.info(
                "[PRODUCER] P2P segment probe for %s: " "get_first_buffer_address=0x%x",
                target,
                remote_buf,
            )
            if remote_buf == 0:
                logger.error(
                    "[PRODUCER] Consumer %s has NO registered buffers visible "
                    "via P2P handshake! Consumer batch_register_memory may "
                    "have failed, or segment descriptors haven't propagated.",
                    target,
                )

        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        sizes: list[int] = []

        num_regions = len(self.kv_caches_base_addr)
        if num_regions != len(consumer_base_addrs):
            logger.error(
                "[PRODUCER] REGION COUNT MISMATCH: producer has %d regions, "
                "consumer has %d regions",
                num_regions,
                len(consumer_base_addrs),
            )
        if len(src_block_ids) != len(dst_block_ids):
            logger.error(
                "[PRODUCER] BLOCK COUNT MISMATCH: src has %d blocks, "
                "dst has %d blocks",
                len(src_block_ids),
                len(dst_block_ids),
            )
        for region_idx in range(num_regions):
            src_base = self.kv_caches_base_addr[region_idx]
            dst_base = consumer_base_addrs[region_idx]
            bpb = self._per_block_bytes_list[region_idx]
            for sb, db in zip(src_block_ids, dst_block_ids):
                src_addrs.append(src_base + sb * bpb)
                dst_addrs.append(dst_base + db * bpb)
                sizes.append(bpb)

        unique_bpb = sorted(set(self._per_block_bytes_list))
        logger.info(
            "[PRODUCER] Starting RDMA write: req=%s, target=%s, "
            "%d regions × %d blocks, bytes/block=%s, "
            "src_block_ids=%s, dst_block_ids=%s, "
            "total_transfer_entries=%d, total_bytes=%d",
            req_id,
            target,
            num_regions,
            len(src_block_ids),
            unique_bpb,
            src_block_ids[:10],
            dst_block_ids[:10],
            len(src_addrs),
            sum(sizes),
        )

        if dst_addrs:
            logger.info(
                "[PRODUCER] dst_addr range: 0x%x -- 0x%x (first region), "
                "consumer_base_addrs[0]=0x%x",
                min(dst_addrs[: len(dst_block_ids)]),
                max(dst_addrs[: len(dst_block_ids)]) + sizes[0],
                consumer_base_addrs[0],
            )

        max_entries_per_batch = 4096
        total_entries = len(src_addrs)
        max_retries = 3

        for chunk_start in range(0, total_entries, max_entries_per_batch):
            chunk_end = min(chunk_start + max_entries_per_batch, total_entries)
            chunk_src = src_addrs[chunk_start:chunk_end]
            chunk_dst = dst_addrs[chunk_start:chunk_end]
            chunk_sizes = sizes[chunk_start:chunk_end]

            retry_delay = 2.0
            for attempt in range(max_retries):
                try:
                    ret = self.transfer_engine.batch_transfer_sync_write(
                        target, chunk_src, chunk_dst, chunk_sizes
                    )
                    if ret == 0:
                        break
                    logger.error(
                        "[PRODUCER] RDMA write chunk error %d for req %s → %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        ret,
                        req_id,
                        target,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                except Exception:
                    logger.exception(
                        "[PRODUCER] RDMA write chunk FAILED for req %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        req_id,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                    ret = -1
                if ret == 0:
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return

        logger.info(
            "[PRODUCER] RDMA write completed for req %s → %s (%d entries)",
            req_id,
            target,
            total_entries,
        )

        # Notify this consumer immediately — its data is written.
        self._send_write_done(notify_host, notify_port, req_id)

        # Track how many consumers still need this transfer_id's blocks.
        # Only mark done_sending (which triggers block deallocation on the
        # scheduler) after ALL consumers sharing this producer rank have
        # completed their transfers.
        all_done = False
        with self._transfer_refcount_lock:
            if transfer_id not in self._transfer_refcount:
                self._transfer_refcount[transfer_id] = consumers_per_rank
            self._transfer_refcount[transfer_id] -= 1
            if self._transfer_refcount[transfer_id] <= 0:
                self._transfer_refcount.pop(transfer_id)
                all_done = True

        if all_done:
            with self._completion_lock:
                self.done_sending.add(transfer_id)
            with self._completed_prefills_lock:
                self._completed_prefills.pop(transfer_id, None)
            logger.info(
                "[PRODUCER] All %d consumers served for transfer_id=%s, "
                "blocks released",
                consumers_per_rank,
                transfer_id,
            )

    def _wait_for_prefill_blocks(self, req_id: str) -> list[int] | None:
        """Wait until prefill block_ids are available for this request."""
        with self._completed_prefills_cv:
            ready = self._completed_prefills_cv.wait_for(
                lambda: req_id in self._completed_prefills,
                timeout=PREFILL_LOOKUP_TIMEOUT,
            )
            if ready:
                return self._completed_prefills[req_id]
            return None

    def _send_write_done(self, host: str, port: int, req_id: str) -> None:
        """Send write-done notification to consumer via persistent socket.

        Sends the notification multiple times for reliability — the consumer
        uses a set so duplicates are harmless.
        """
        path = make_zmq_path("tcp", host, port)
        notification = msgpack.dumps({"request_id": req_id})
        with self._notify_sockets_lock:
            sock = self._notify_sockets.get(path)
            if sock is None:
                sock = self.zmq_context.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 5000)
                sock.setsockopt(zmq.SNDHWM, 0)
                sock.connect(path)
                self._notify_sockets[path] = sock
            for _ in range(3):
                sock.send_multipart([MSG_WRITE_DONE, notification])
        logger.info("[PRODUCER] write-done sent for req %s", req_id)

    # -----------------------------------------------------------------
    # Consumer: notification listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _notification_listener(self) -> None:
        """Receive write-done notifications from producers."""
        path = make_zmq_path("tcp", "*", self._notification_port)
        logger.info("Mooncake notification listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                msg_type = parts[1]

                if msg_type == MSG_WRITE_DONE:
                    data = msgpack.loads(parts[2])
                    req_id = data["request_id"]
                    dst_blocks = self._pending_recv_blocks.pop(req_id, None)
                    if dst_blocks:
                        with self._fence_lock:
                            self._blocks_pending_fence.extend(dst_blocks)
                    with self._completion_lock:
                        self.done_recving.add(req_id)
                        self._pending_recv.discard(req_id)
                    logger.info(
                        "[CONSUMER] Write-done received for req %s, "
                        "done_recving now: %s",
                        req_id,
                        self.done_recving,
                    )
                else:
                    logger.error("Unknown notification type: %s", msg_type)
