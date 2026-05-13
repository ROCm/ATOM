# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Low-level RDMA wrapper around a MoRIIO ``IOEngine``.

Provides helper methods for memory registration, session management,
and asynchronous RDMA read/write operations.  Both producer and
consumer code paths share this wrapper.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import msgpack
import zmq

from atom.kv_transfer.disaggregation.moriio.moriio_common import (
    MoRIIOConstants,
    Role,
    _MORIIO_AVAILABLE,
    get_role,
)
from atom.kv_transfer.disaggregation.types import RemoteAllocInfo
from atom.utils import make_zmq_path, make_zmq_socket

if _MORIIO_AVAILABLE:
    from mori.io import EngineDesc, MemoryDesc, MemoryLocationType

logger = logging.getLogger("atom")


class MoRIIOWrapper:
    """Low-level wrapper around a MoRIIO ``IOEngine``.

    Provides helper methods for memory registration, session management,
    and asynchronous RDMA read/write operations.  Both producer and
    consumer code paths share this wrapper.

    Thread-safety:
        ``transfer_status``, ``done_req_ids``, ``done_write_cache_req_ids``,
        and ``done_remote_allocate_req_dict`` are guarded by ``self.lock``.  The ZMQ socket cache ``self._sockets``
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
        self.done_remote_allocate_req_dict: dict[str, Any] = {}
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
            from atom.kv_transfer.disaggregation.moriio.moriio_connector import (
                _zmq_ctx,
            )

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
