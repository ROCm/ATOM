# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import ipaddress
import logging
import queue
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional, List, Iterator
from enum import Enum
from urllib.parse import urlparse

import msgpack
import msgspec
import numpy as np
import torch
import torch.distributed as dist
import zmq
from dataclasses import dataclass
from atom.config import Config
from atom.model_engine.sequence import Sequence
from atom.utils.network import get_ip
from aiter.dist.parallel_state import get_dp_group, get_tp_group

logger = logging.getLogger("atom")

MoRIIO_enabled = False
try:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
    )
    logger.info("MoRIIO is available")
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")


Transfer = tuple[int, float]
EngineId = str
ReqId = str


class MoRIIOAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True, 
):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: Optional[List[int]] = None 
    num_blocks: int
    block_len: int
    attn_backend_name: str


@dataclass
class ReqMeta:
    """Metadata for a single request."""

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int
    # the seq.id between producer and consumer may be different, 
    # so we need to add transfer_id to the output params
    transfer_id:int


@dataclass
class RemoteAllocInfo:
    """Information about remote block allocation."""

    block_ids: list[int]
    writes_done: int = 0
    decode_dp_rank: int = 0
    transfer_offset: tuple[list[int], list[int], list[int]] | None = None
    
    
class ROLE(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOTINIT = "notinit"


def convert_virtual_to_physical_pages(
    virtual_pages, 
    virtual_block_size=16, 
    physical_block_size=1
):
    block_ratio = virtual_block_size // physical_block_size  
    physical_pages = []  
    
    for virtual_page in virtual_pages:  
        # Every virtual block pages corresponds to block_ratio*physical pages
        start_physical = virtual_page * block_ratio  
        end_physical = start_physical + block_ratio  
        physical_pages.extend(range(start_physical, end_physical))  
    
    return physical_pages  
    

class RoleManager:
    """Manages role state across the connector."""

    _instance: Optional["RoleManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._role: ROLE = ROLE.NOTINIT
        
    def __new__(cls) -> "RoleManager":
        raise RuntimeError("Use RoleManager.get_instance() instead of direct instantiation")

    @classmethod
    def get_instance(cls) -> "RoleManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_role(self, role: ROLE) -> None:
        """Set the current role."""
        with self._lock:
            self._role = role

    def get_role(self) -> ROLE:
        """Get the current role."""
        return self._role


def set_role(role: ROLE):
    """Set the global role."""
    RoleManager.get_instance().set_role(role)


def get_role() -> ROLE:
    """Get the global role."""
    return RoleManager.get_instance().get_role()


class MoRIIOConstants:
    """Constants for MoRIIO connector."""
    # ZMQ message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"

    PING_INTERVAL = 5
    MAX_PING_RETRIES = 100
    DEFAULT_HANDSHAKE_PORT = 6301
    DEFAULT_NOTIFY_PORT = "61005"

    VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT = 3600
    
    
def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False
    
def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return (dp_rank) * tp_size + tp_rank


@dataclass
class KVConnectorOutput:
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None

    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
        )


class MoRIIOWrapper:
    """Wrapper for MoRIIO engine operations.
    Handles both producer and consumer roles for KV cache transfers.
    Args:
        moriio_engine:  MoRIIO engine instance
        tp_rank: Tensor parallel rank
        dp_rank: Data parallel rank
    """

    def __init__(
        self,
        moriio_engine = None,
        tp_rank: int = 0,
        dp_rank: int = 0,
    ):  
        
      
        
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moriio_engine = moriio_engine
        self.remote_memory_metadata = None
        self.local_memory_registered = False
        self.local_memory_metadata = None
        self.transfer_status: list = []
        self.remote_engine_ip: str | None = None
        self.notify_port: int | None = None
        self.lock = threading.Lock()
        self.done_req_ids: list[str] = []
        self.done_write_cache_req_ids: list[str] = []
        self.notify_thread: threading.Thread | None = None
        self.paths = {}
    def set_moriio_engine(self, moriio_engine):
        assert moriio_engine is not None, (
            "You Cannot pass None engine to MoRIIOWrapper!"
        )
        self.moriio_engine = moriio_engine

    def set_backend_type(self, backend_type):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
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
        logger.info("Registered remote engine with key: %s", consumer_engine_metadata.key)
        return consumer_engine_metadata.key

    def register_local_tensor(self, tensor):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(
                tensor
            )
            assert self.local_memory_metadata is not None, (
                "register_torch_tensor returned None"
            )
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            raise ValueError(f"Failed to register local memory: {e}") from e
        self.local_memory_registered = True
        return local_memory_metadata_packed

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

            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._handle_message(msg)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        raise ValueError(f"Error processing message: {e}") from e

        self.notify_thread = threading.Thread(
            target=_async_wait, daemon=True, name="moriio-notify-listener"
        )
        self.notify_thread.start()

    def _handle_message(self, msg: bytes):
        """Handles incoming messages from remote nodes."""
        # Handles incoming remote messages:
        # Prefill Role:
        #   [write] mode: receives block information (allocation)
        #   [read]  mode: receives block release messages from decode side
        # Decode Role:
        #   [write] mode: receives KV cache write completion notifications
        handled = False
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict) and "req_id" in data:
                self._handle_structured_message(data)

                return
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException):
            logger.debug("Failed to decode msgpack message, will try as string")
            pass

        try:
            msg_str = msg.decode("UTF-8")
            if msg_str.startswith(MoRIIOConstants.COMPLETION_PREFIX):
                self._handle_completion_message(msg_str)
                handled = True
        except UnicodeDecodeError:
            logger.warning("Received non-UTF8 message: %s", msg_str)
        if not handled:
            raise ValueError(f"Unhandled message format: {msg_str}")

    def _handle_structured_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, "Only prefill can get block messages"
        req_id = data["req_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        assert len(block_notify_list) > 0, (
            "block_notify_list cannot be empty in remote allocate message"
        )

        with self.lock:
            self.done_remote_allocate_req_dict[req_id] = RemoteAllocInfo(
                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank
            )

    def _handle_completion_message(self, msg: str):
        with self.lock:
            if get_role() == ROLE.PRODUCER:
                self.done_req_ids.append(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def send_notify(self, req_ids, remote_ip, remote_port):
        if not remote_ip or not remote_port:
            logger.warning("Missing remote_ip or remote_port for notification")
            return

        path = make_zmq_path("tcp", remote_ip, remote_port)

        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        req_list = req_ids if isinstance(req_ids, list) else [req_ids]

        sock = self.paths[path]
        try:
            for req_id in req_list:
                if isinstance(req_id, int):
                    req_id= str(req_id)
                
                if not isinstance(req_id, str):
                    logger.warning(
                        "Invalid req_id type: %s, expected str", type(req_id)
                    )
                    continue
                sock.send_multipart([MoRIIOConstants.POP_DONE_RECV, req_id.encode("utf-8")])
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self.paths.pop(path, None)
            raise

    def pop_finished_req_ids(self):
        # producer invocation: get the set of completed requests at the decode
        with self.lock:
            done_send = set(self.done_req_ids)
            self.done_req_ids = []
        return done_send

    def pop_finished_write_req_ids(self):
        # Call the consumer in write mode to get the collection after write completion
        with self.lock:
            done_write_cache = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids = []
        return done_write_cache

    def shutdown(self):
        logger.debug("Closing MoRIIOWrapper and cleaning up ZMQ sockets")
        for path, sock in self.paths.items():
            try:
                sock.close(linger=0)
                logger.debug("Closed ZMQ socket for path: %s", path)
            except Exception as e:
                logger.warning("Error closing ZMQ socket for path %s: %s", path, e)
        self.paths.clear()
        
@dataclass
class RemoteMeta:
    block_ids: list[int]
    host: str
    port: int
    engine_id: str
    request_id: str

class ConnectorMetadata():
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        self.request_id_to_transfer_id: dict[ReqId, int] = {}

    def _add_new_req(
        self,
        req_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids= kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_handshake_port=kv_transfer_params["remote_handshake_port"],
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
            tp_size=kv_transfer_params.get("tp_size", 1),
            transfer_id=kv_transfer_params.get("transfer_id"),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            request_id,local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        req = self._add_new_req(request_id, local_block_ids, kv_transfer_params)
  
        self.reqs_to_recv[request_id] = req


class kvconnector():
    def __init__(self, config: Config):
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group  
        kv_transfer_config = config.kv_transfer_config
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size 
        self.local_ip = get_ip()
        self.local_ping_port =_get_open_port()
        
        self.is_producer = kv_transfer_config.get("kv_role","kv_producer")=="kv_producer"
        # self.is_producer = kv_transfer_config.get("kv_role", ROLE.PRODUCER.value) == ROLE.PRODUCER.value
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.proxy_ping_port = kv_transfer_config.get("proxy_ping_port", 36367)
        
        self.request_address = (
            f"{self.local_ip}:{self.http_port}"
        )
        
        self.base_handshake_port = MoRIIOConstants.DEFAULT_HANDSHAKE_PORT      
         
        self.layer_name_to_remote_kv_cache_metadata: dict[str, dict[str, list[Any]]] = (
            dict()
        )

        handshake_port = MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
        self.side_channel_port = int(handshake_port)+ get_port_offset(self.dp_rank, self.tp_rank)
        self.engine_id = (
            str(self.local_ip)
            + ":"
            + str(handshake_port)
        )

        self.zmq_context = zmq.Context()
        self.load_ready_flag: dict[str, bool] = {}
        self.write_ready_flags: dict[str, bool] = {}
        self._handshake_lock = threading.RLock()
        self.remote_moriio_metadata: dict[EngineId, MoRIIOAgentMetadata] = {}
        self._handshake_futures: dict[EngineId, Future[set[str]]] = {}
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
     
        
        self.moriio_engine = IOEngine(
        f"test+:ip:{self.local_ip}+tp:{self.tp_rank}+dp:{self.dp_rank}",
        IOEngineConfig(
        host=str(self.local_ip),
        port=0,
        ),
        )
        self.built_write_session: defaultdict[str, list] = defaultdict(list)
        self._remote_agents: dict[EngineId, set[str]] = {}
        # Background thread for initializing new MoRIIO handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # MoRIIO is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="atom-moriio-handshake-initiator",
        )
        self.moriio_wrapper = MoRIIOWrapper( moriio_engine=self.moriio_engine)
        self.layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]] = {}
        # logging.getLogger("aiter").disabled = True
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}
        self.local_kv_cache_metadata: list[bytes] = []
        self.remote_kv_cache_metadata: list[bytes] = []
        
        self.proxy_ip=kv_transfer_config.get("proxy_ip")

        if self.tp_rank == 0 and self.dp_rank==0: 
            
                self._ping_thread = threading.Thread(
                    target=self._ping, args=(self.zmq_context,), daemon=True
                )
                self._ping_thread.start()
        
        self.kv_cache_shape = None
        
        # In progress transfers.
        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}
        self.done_sending=set()
        self.moriio_wrapper.set_backend_type(BackendType.RDMA)
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}


    def register_kv_caches(self, kv_caches: dict):
        self.kv_caches=kv_caches
        for layer_name, kv_cache in kv_caches.items():
            the_cache=kv_cache.k_cache
            if self.kv_cache_shape is None:
                self.kv_cache_shape  = the_cache.shape
            if layer_name not in self.layer_name_to_local_kv_cache_metadata:
                self.layer_name_to_local_kv_cache_metadata[layer_name] = []

            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(the_cache)
            self.layer_name_to_local_kv_cache_metadata[layer_name].append(
                moriio_mem_metadata
            )

        if len(the_cache.shape)==5:
            self.num_blocks, num_kv_heads, _, self.block_len, _ = the_cache.shape
        else:
            self.num_blocks, self.block_len, hs = the_cache.shape

        self.num_blocks = the_cache.shape[0]
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=None,
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name="aiter",
        )
        ready_event = threading.Event()
        self._moriio_handshake_listener_t = threading.Thread(
            target=self._moriio_handshake_listener,
            args=(
                metadata,
                ready_event,
                self.side_channel_port,
                self.tp_rank,
                self.dp_rank,
                self.layer_name_to_local_kv_cache_metadata,
            ),
            daemon=True,
            name="moriio_handshake_listener",
        )
        self._moriio_handshake_listener_t.start()        
    
    def get_engine_name_with_dp(self, engine_name, dp_rank):
        return f"{engine_name}_dp{dp_rank}"

    def start_load_kv(self, metadata:ConnectorMetadata):
        if self.is_producer:
            return  
        if metadata is not None:
            if len(metadata.reqs_to_recv)>0:
                logger.debug(f"Start load kv cache process {metadata.reqs_to_recv=}")
                
                
        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        remote_engine_id = None
        wait_handshake_readd_req = False
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        wait_handshake_readd_req = True

                        continue
            self._read_blocks_for_req(req_id, meta)
                
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                continue
            elif (
                not self._ready_requests.empty()
                and remote_engine_id in self.load_ready_flag
            ):
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        
    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id,
            req_id,
        )
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_handshake_port=meta.remote_handshake_port
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
        """Compute transfer offsets for block data.
        Args:
            layer_name: Name of the layer to transfer
            local_block_ids: IDs of local blocks
            remote_block_ids: IDs of remote blocks
            remote_moriio_meta: Metadata of the remote MoRIIO agent
        Returns:
            Tuple of (local_offsets, remote_offsets, transfer_sizes)
        """
        assert self.kv_cache_shape is not None, "KV caches shape not initialized"
        is_mla = len(self.kv_cache_shape) == 3
        stride = self.kv_caches[layer_name].k_cache.stride()
        sz = self.kv_caches[layer_name].k_cache.element_size()
        if is_mla:
            blknum, blksize, hs = self.kv_cache_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = self.kv_cache_shape
            local_ktov_stride = stride[0]
            block_stride = stride[1]
            remote_ktov_stride = block_stride * remote_moriio_meta.num_blocks

        transfer_size_byte = blksize * hn * hs * sz
        per_block = 1 if is_mla else 2
        total = len(local_block_ids) * per_block
        offset_local = [0] * total
        offset_remote = [0] * total
        sizes = [transfer_size_byte] * total

        w = 0
        for i, lb in enumerate(local_block_ids):
            rb = remote_block_ids[i]
            # K
            offset_local[w] = sz * (lb * block_stride)
            offset_remote[w] = sz * (rb * block_stride)
            w += 1
            if not is_mla:
                # Handle num_block variations originating from PD (different kv strides)
                # TODO: address block_sz differences in heterogeneous TP scenarios
                # In MLA, we don't need to consider these two cases.
                offset_local[w] = sz * (1 * local_ktov_stride + lb * block_stride)
                offset_remote[w] = sz * (1 * remote_ktov_stride + rb * block_stride)
                w += 1

        merged_l, merged_r, merged_s = offset_local, offset_remote, sizes
        return merged_l, merged_r, merged_s

    def _get_built_session(self, remote_engine_id):
        if remote_engine_id not in self.built_write_session:
            cur_remote_engine_sessions = []
            for ln, local_meta in self.layer_name_to_local_kv_cache_metadata.items():
                unpacked_local_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(local_meta[0])
                )
                unpacked_remote_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(
                        self.layer_name_to_remote_kv_cache_metadata[remote_engine_id][
                            ln
                        ][0]
                    )
                )
                cur_remote_engine_sessions.append(
                    self.moriio_wrapper.build_session(
                        unpacked_local_memory_meta, unpacked_remote_memory_meta
                    )
                )
            self.built_write_session[remote_engine_id] = cur_remote_engine_sessions
        return self.built_write_session[remote_engine_id], self.remote_moriio_metadata[
            remote_engine_id
        ]

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        remote_host: str,
        remote_handshake_port:int
    ) -> None:
    
        local_block_ids=convert_virtual_to_physical_pages(local_block_ids)
        remote_block_ids=convert_virtual_to_physical_pages(remote_block_ids)
        logger.debug(
            f"Start reading blocks for req {request_id} from remote engine {dst_engine_id},{self.tp_rank=}"
        )
        dp0_engine_id = self.get_engine_name_with_dp(dst_engine_id, 0)
        sessions, remote_moriio_meta = self._get_built_session(dp0_engine_id)

        first_layer = list(self.layer_name_to_local_kv_cache_metadata.keys())[0]
        offs = self._compute_block_transfer_offsets(
            first_layer, local_block_ids, remote_block_ids, remote_moriio_meta
        )
     
        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            #TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )
        
            with self.moriio_wrapper.lock:
                self._recving_transfers[request_id].append(transfer_status)
                
                self._recving_transfers_callback_addr[request_id] = (
                    remote_host,
                    str(remote_handshake_port+self.tp_rank),   
                )

        logger.debug(
            f"Completed reading blocks for req {request_id} from remote engine {dst_engine_id},{self.tp_rank=}"
        )

    def _ping(self, zmq_context):
        import time
        http_request_address = f"http://{self.request_address}/v1/completions"
        role = "P" if self.is_producer else "D"
        retry_count = 0
        index = 1
        with zmq_context.socket(zmq.DEALER) as sock:
            proxy_path=f"tcp://{self.proxy_ip}:{self.proxy_ping_port}"
            sock.connect(proxy_path)

            while True:
                try:
                    data = {
                        "type": "register",
                        "role": role,
                        "index": str(index),
                        "request_address": http_request_address,
                        "handshake_port": self.base_handshake_port,
                        "dp_size": self.dp_size,
                        "tp_size": self.tp_size,
                        "transfer_mode": "read",
                    }

                    sock.send(msgpack.dumps(data))
                    logger.debug(f"Successfully sent ping message #{index} {proxy_path} {data}")
                    retry_count = 0

                except ConnectionRefusedError:
                    logger.info(
                        "Connection refused: %s:%s -> %s:%s",
                        self.local_ip,
                        self.local_ping_port,
                        self.proxy_ip,
                        self.proxy_ping_port,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error when sending ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected error when sending ping: %s", e)
                    retry_count += 1
                    if retry_count >= MoRIIOConstants.MAX_PING_RETRIES:
                        logger.error(
                            "Max retries (%s) exceeded. Stopping ping loop.",
                            MoRIIOConstants.MAX_PING_RETRIES,
                        )
                        raise RuntimeError(
                            f"Ping failed after {retry_count} retries"
                        ) from e

                finally:
                    time.sleep(MoRIIOConstants.PING_INTERVAL)
                    index += 1

    def _moriio_handshake_listener(
        self,
        metadata: MoRIIOAgentMetadata,
        ready_event: threading.Event,
        base_port: int,
        tp_rank: int,
        dp_rank: int,
        layer_name_to_local_kv_cache_metadata: dict,
    ):
        """Background thread for getting new MoRIIO handshakes."""
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.info(
            "Size of encoded MoRIIOAgentMetadata: %s bytes", str(size_in_bytes)
        )

        # Listen for new requests for metadata.
        host = "*"

        path = make_zmq_path("tcp", host, base_port)
        logger.info("mori handshake starting listening on path: %s", path)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                parts= sock.recv_multipart()
                identity, msg  =  parts[0], parts[1]
                if (
                    msg != MoRIIOConstants.GET_META_MSG
                    and msg != MoRIIOConstants.POP_DONE_RECV
                ):
                    logger.error("Connection listener got unexpected message")
                    raise ValueError("handshake failed, unexpected msg type")
                elif msg == MoRIIOConstants.GET_META_MSG:
                    sock.send_multipart(
                        (identity, b"", encoded_data)
                    )  # send local mori io engine meta data
                    logger.info("MoRIIO handshake listener sent metadata")
                    # now we send tensor meta data for each block
                    buf = msgpack.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))
                elif msg == MoRIIOConstants.POP_DONE_RECV:
                    if len(parts)<3:
                        logger.error("POP_DONE_RECV message missing req_id")
                        raise ValueError("POP_DONE_RECV message missing req_id")
                    req_id= int(parts[2])
                    
                    self.done_sending.add(req_id)
                    logger.debug(
                        f"MoRIIO handshake listener received done recv for req {req_id}",
                        )
                    
    def _moriio_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dp_rank: int = 0,
    ) -> set[str]:
        import time
        """Do a MoRIIO handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        port_offset = get_port_offset(remote_dp_rank, self.tp_rank)
        path = make_zmq_path("tcp", host, port + port_offset)
        logger.info("handshake Querying metadata on path: %s", path)

        # Send query for the request.
        with zmq_ctx(zmq.DEALER, path) as sock:
            logger.info("prepare send msg INSTAZNCE: %s", path)
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

    def _background_moriio_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        
        logger.info("Starting background MoRIIO handshake for request %s", req_id)
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            host = meta.remote_host
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            remote_dp_size = int(meta.remote_dp_size)

        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True

        fut_list = []

        # In dp(prefill)<->dp(decode) communication, we require an all-to-all handshake.

        for cur_dp_rank in range(remote_dp_size):
            dp_engine_id = self.get_engine_name_with_dp(remote_engine_id, cur_dp_rank)
            future = self._handshake_initiation_executor.submit(
                self._moriio_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
            )
            fut_list.append(future)

            def done_callback(f: Future[set[str]], eid=dp_engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            future.add_done_callback(done_callback)
            self._handshake_futures[dp_engine_id] = future

        # fut = fut_list
        def wait_all_dp():
            for future in fut_list:
                future.result()
            return True

        all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
        
        all_done_future.add_done_callback(request_ready)
    
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
    
    def get_finished(self):
        
        
        done_sending, done_recving = set(), set()
        done_recving = self._pop_done_transfers()
        done_sending= self.done_sending.copy()
        self.done_sending = set()
        return done_sending, done_recving
    
    
class kvconnector_scheduler():
    def __init__(self, config: Config):
        kv_transfer_config = config.kv_transfer_config
        self.is_producer = kv_transfer_config.get("kv_role","kv_producer")=="kv_producer"
        # self.is_producer = kv_transfer_config.get("kv_role", ROLE.PRODUCER.value) == ROLE.PRODUCER.value
        self.handshake_port = _get_open_port()
        self.engine_id = "None"
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.parallel_config.data_parallel_size
        self.host_ip = get_ip()         
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}
        self.request_id_to_transfer_id: dict[ReqId, int] = {}
        self.transfer_id_to_request_id: dict[int, ReqId] = {}
        
        
    def get_num_new_matched_tokens(self, seq:Sequence):
        params = seq.kv_transfer_params
        params = params or {}

        if  params.get("do_remote_prefill") and not hasattr(seq,"kv_async_tagged") :
            seq.kv_async_tagged=True
            return len(seq.prompt_token_ids), True
        
        return 0, False
  
    def build_connector_meta(self) -> ConnectorMetadata:
        meta=ConnectorMetadata()
        # meta.transfer_id_to_request_id = self.transfer_id_to_request_id
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id 
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        logger.debug(f"Scheduler build_connector_meta recv reqs: {list(self._reqs_need_recv.keys())}")
        
        self._reqs_need_recv.clear()
        return meta
    
    def update_state_after_alloc(self, seq:Sequence):
        params = seq.kv_transfer_params
        params = params or {}
        
        if not self.is_producer:
        
            transfer_id = params["transfer_id"]
            request_id = seq.id
            self.transfer_id_to_request_id[transfer_id] = request_id
            self.request_id_to_transfer_id[request_id] = transfer_id
        
        # Decode side
        if params.get("do_remote_prefill"):
            assert self.is_producer is False, "Only decode side can do remote prefill"
            
            self._reqs_need_recv[seq.id]=seq, seq.block_table
            params["do_remote_prefill"] = False
            logger.debug(f"Scheduler update_state_after_alloc recv reqs: {list(self._reqs_need_recv.keys())}")
    
            
    def request_finished(self, seq:Sequence):
        # Decode instance
        if self.is_producer==False:
            b=0
        
        
        # the seq.id between producer and consumer may be different, so we need to add transfer_id to the output params
        # this keep the producer can free block table after the decoderequest is finished
        seq.kv_transfer_params_output=dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=seq.block_table.copy(), #use copy to avoid mutation, or will get None
            remote_engine_id=self.engine_id,
            remote_host=self.host_ip,
            remote_port=self.handshake_port,
            tp_size=self.tp_size,
            transfer_id=seq.id,
        ) 
        
        
        # free the transfer_id and request_id after the request is finished
        if not self.is_producer:
            if seq.id in self.request_id_to_transfer_id:
                transfer_id = self.request_id_to_transfer_id[seq.id]
                del self.request_id_to_transfer_id[seq.id]
                if transfer_id in self.transfer_id_to_request_id:
                    del self.transfer_id_to_request_id[transfer_id]
        return None

    
def _get_open_port() -> int:
    import socket

    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
        
def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port

# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""
    import psutil

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
            