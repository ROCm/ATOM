# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses RDMA-based zero-copy transfers via the MoRIIO library for efficient
KV cache migration between producer (prefill) and consumer (decode) nodes.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
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
from atom.kv_transfer.disaggregation.moriio.fabric import (
    allocate_fabric_pool,
    block_runs,
    desc_fabric_info,
    is_fabric_backend,
    page_digest,
    page_offsets,
    page_region_digests,
    pool_span,
    repeated_blocks,
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
    KVTransferRegion,
    ReqId,
    ReqMeta,
    TransferId,
    produced_no_handoff,
)
from atom.kv_transfer.disaggregation.utils import (
    MAX_RDMA_CHUNK_BYTES,
    chunk_tensor_for_rdma,
)
from atom.model_engine.sequence import Sequence
from atom.utils import (
    envs,
    get_open_port,
    make_zmq_path,
    zmq_socket_ctx,
)
from atom.utils.network import get_ip

if _MORIIO_AVAILABLE:
    from mori.io import (
        BackendType,
        FabricBackendConfig,
        IOEngine,
        IOEngineConfig,
        PollCqMode,
        RdmaBackendConfig,
    )

logger = logging.getLogger("atom")

# State name of ReplaySSM's record cursor, one tensor shared by every layer.
REPLAY_CURSOR = "replayssm.write_pos"


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

       - ``read`` (default): the consumer pulls the KV with RDMA reads.
       - ``write``: the consumer sends the producer its block ids and the
         producer pushes the KV with RDMA writes, then reports back.

    4. Tracking transfer completion and notifying the peer when done.
    5. Periodically pinging the proxy for service discovery.
    """

    use_fabric = False
    _transfer_stats = False
    _write_service_thread: threading.Thread | None = None
    # How long the write service sleeps between polls while a push is waiting
    # or in flight, and at most between passes while idle.
    _WRITE_POLL_S = 50e-6
    _WRITE_IDLE_WAIT_S = 0.1
    # Finished prefills whose ready event aged out, remembered up to this many.
    _KV_READY_EXPIRED_MAX = 1 << 16

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
        # Sets `state_transfer_slots` (the runner, after sizing the pools).
        self._config = config
        self.transfer_mode = self._transfer_mode_from(kv_transfer_config)
        if (
            self.transfer_mode == "write"
            and not self.is_producer
            and envs.ATOM_PD_FAILED_WRITE_HOLD_S <= 0
        ):
            raise ValueError(
                "MoRIIO write mode needs ATOM_PD_FAILED_WRITE_HOLD_S > 0: a push "
                "started just before its deadline could land in freed blocks"
            )
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

        # RDMA engine and wrapper. The key is unique per process: MoRI keeps
        # the queue pairs it built for a key, so a peer restarted under the
        # same key could not be reached again.
        self.moriio_engine = IOEngine(
            f"atom:ip:{self.local_ip}+tp:{self.tp_rank}+dp:{self.dp_rank}"
            f"+{uuid.uuid4().hex[:8]}",
            IOEngineConfig(host=str(self.local_ip), port=0),
        )
        self.moriio_wrapper = MoRIIOWrapper(moriio_engine=self.moriio_engine)

        # FABRIC only, never RDMA next to it: mori would pick RDMA for any peer
        # FABRIC cannot reach and the transfer would quietly leave the fabric.
        self.use_fabric = is_fabric_backend(kv_transfer_config)
        if self.use_fabric:
            self._create_fabric_backend(kv_transfer_config)
        else:
            self._create_rdma_backend(kv_transfer_config)

        # PAGE layout (fabric): the one registered buffer's descriptor and,
        # per PAGE region, (offset in it, bytes per block, semantic role).
        self._page_desc: bytes | None = None
        self._page_regions: list[tuple[int, int, str]] = []
        self._page_views: list[Any] | None = None
        self._transfer_stats = envs.ATOM_MORIIO_TRANSFER_STATS
        # Read mode: (posted, bytes, blocks, segments) per request in flight.
        self._page_reads: dict[ReqId, tuple[float, int, int, int]] = {}
        # ATOM_MORIIO_VERIFY on the consumer: each pushed request's destination
        # blocks, and the producer's digest of its source blocks.
        self._verify_page_dst: dict[ReqId, list[int]] = {}
        self._write_page_digests: dict[ReqId, list[Any]] = {}

        # Per-layer local metadata (populated in register_kv_caches)
        self.layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]] = {}
        self.local_kv_cache_metadata: list[bytes] = []
        self.remote_kv_cache_metadata: list[bytes] = []
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.kv_caches: dict[str, Any] | None = None
        self.kv_cache_block_size: int = config.kv_cache_block_size
        self.blocks_per_chunk: int | None = None
        self.num_k_chunks: int = 0

        # Per-request state (`KVCacheTensor.per_request_state`), addressed by
        # state slot rather than block: name -> (descriptor, bytes per slot,
        # tensor). Never part of the block layout above.
        self._state_tensors: dict[str, tuple[bytes, int, Any]] = {}
        self._num_state_slots = 0
        self._built_state_sessions: dict[str, list[tuple[Any, int]]] = {}
        # Producer: slots from here up are transfer slots, and each staged
        # transfer's (slot, event after the copy that filled it).
        self._first_transfer_slot = 0
        self._staged_sources: dict[TransferId, tuple[int, Any]] = {}

        # Session cache: remote_engine_id -> list[session]
        self._built_sessions: defaultdict[str, list] = defaultdict(list)

        # Handshake management
        self.zmq_context = zmq.Context()
        self._handshake_lock = threading.RLock()
        self._handshake_timeout_s = envs.ATOM_MORIIO_HANDSHAKE_TIMEOUT_S
        self._remote_agents: dict[EngineId, set[str]] = {}
        # Read mode: the handshake with each producer DP rank, and the requests
        # waiting on one with their deadlines.
        self._read_handshakes: dict[EngineId, Future[set[str]]] = {}
        self._awaiting_handshake: dict[ReqId, tuple[float, ReqMeta]] = {}
        self._ready_requests: queue.Queue[
            tuple[ReqId, ReqMeta, BaseException | None]
        ] = queue.Queue()
        # MoRIIO is not guaranteed to be thread-safe, limit to 1 worker.
        self._handshake_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="atom-moriio-handshake-initiator",
        )
        # Sessions of peers dropped after a failed transfer; kept alive for
        # transfers still posted on them.
        self._retired_sessions: list[Any] = []

        # In-flight receive transfers
        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[
            ReqId, tuple[str, str, EngineId]
        ] = {}

        # Opt-in correctness probe. A transfer that reports success only says
        # the RDMA verbs retired, not that every layer's KV arrived, and decode
        # cannot tell the difference -- it just attends over whatever is in the
        # block and degenerates. Digesting each layer's destination before and
        # after the transfer turns that silent case into a log line.
        self._verify_kv = envs.ATOM_MORIIO_VERIFY
        self._verify_before: dict[ReqId, dict[str, int]] = {}
        self._verify_rows: dict[ReqId, Any] = {}
        self._verify_state_before: dict[ReqId, tuple[int, dict[str, int]]] = {}
        # Reused local prefix blocks, digested before the receive.
        self._verify_kept: dict[ReqId, tuple[Any, dict[str, int]]] = {}
        self._write_state_digests: dict[ReqId, dict[str, int]] = {}
        self._logged_layout = False

        # Requests rejected before a single transfer was issued. They have no
        # status to poll, so get_finished has to report them from here.
        self._failed_before_read: set[ReqId] = set()

        # Serial WRITE mode, producer side. The listener thread only queues
        # requests; the write service thread (or, without it, the worker
        # thread), which owns every transfer call, drains them, waits out each
        # consumer's handshake, and tracks the writes in flight.
        self._write_requests: queue.Queue[dict[str, Any]] = queue.Queue()
        self._write_wakeup = threading.Event()
        self._write_service_stop = threading.Event()
        self._pending_writes: list[tuple[dict[str, Any], float]] = []
        self._write_handshakes: dict[EngineId, Future[set[str]]] = {}
        self._sending_writes: dict[int, tuple[list[Any], dict[str, Any]]] = {}
        # Transfers asked for: None while queued or pushing, else when the push
        # was reported (monotonic). The source blocks go once it is, so a
        # second ask for the same transfer must not push again.
        self._write_ids: dict[int, float | None] = {}
        self._finished_writes: deque[tuple[float, int]] = deque()
        # When each finished prefill's KV writes are done on the GPU, recorded
        # by the runner after the forward of its final chunk. A push must not
        # read its source blocks before then. The consumer asks once the head
        # rank has returned the prefill's token, which can be before another
        # rank's forward has got that far, so a transfer with no event yet
        # waits for one. Events past their age go to the expired set, oldest
        # first (a dict as an ordered set), so a late ask still finds its
        # prefill complete.
        self._kv_ready_events: dict[int, Any] = {}
        self._kv_ready_batches: deque[tuple[float, Any, list[int]]] = deque()
        self._kv_ready_expired: dict[int, None] = {}
        self._kv_ready_lock = threading.Lock()
        # ATOM_MORIIO_DIAG_POST_READY: per transfer, when this rank ran a
        # forward over it after its KV-ready event (monotonic), oldest first.
        self._diag_post_ready = envs.ATOM_MORIIO_DIAG_POST_READY
        self._post_ready_forwards: dict[int, list[float]] = {}
        # Consumer: this step's write requests and an event after every forward
        # already queued. One of those may still be writing a block the
        # scheduler has just freed and handed to one of these requests, so the
        # producer is asked to push only once the event has completed.
        self._gated_writes: deque[tuple[Any, list[tuple[ReqId, ReqMeta]]]] = deque()

        # Serial WRITE mode, consumer side: requests waiting on the producer's
        # report with the push deadline each one sent, and the reports the
        # listener thread has collected.
        self._awaiting_write: dict[ReqId, tuple[float, ReqMeta]] = {}
        self._write_reports: dict[ReqId, bool] = {}
        self._write_reports_lock = threading.Lock()
        self._write_timeout_s = envs.ATOM_MORIIO_WRITE_TIMEOUT_S

        # Failed receives, held until the given time before they are reported.
        self._held_failures: dict[ReqId, float] = {}

        # Completed send-side transfers, added by the listener thread and by
        # _finish_write, so swapped out under the lock.
        self.done_sending: set[int] = set()
        self._done_sending_lock = threading.Lock()

        # Transfer ID mapping (worker side)
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # Region mode (backends that publish KVTransferTensors and no
        # per-layer block caches, e.g. DeepSeek-V4). None until registered.
        self._regions: list[_Region] | None = None
        self._region_sessions: dict[EngineId, dict[tuple[int, int, int], Any]] = {}
    def _create_rdma_backend(self, kv_transfer_config: dict) -> None:
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

    def _create_fabric_backend(self, kv_transfer_config: dict) -> None:
        import warnings

        num_streams = int(kv_transfer_config.get("fabric_num_streams", 64))
        num_events = int(kv_transfer_config.get("fabric_num_events", 64))
        # mori only warns when its fabric copy kernel fails to load, and without
        # it every copy of 4 MiB or more fails.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.moriio_wrapper.set_backend_type(
                BackendType.FABRIC, FabricBackendConfig(num_streams, num_events)
            )
        logger.info(
            "FabricBackendConfig: streams=%d, events=%d, transfer_mode=%s",
            num_streams,
            num_events,
            self.transfer_mode,
        )

    @classmethod
    def allocate_kv_pool(cls, config, nbytes: int, device):
        """moriio_backend fabric keeps the KV pool in fabric-exportable memory."""
        if not is_fabric_backend(config.kv_transfer_config or {}):
            return None
        return allocate_fabric_pool(nbytes, device)

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

        A backend with no per-layer block caches (DeepSeek-V4) publishes its
        storage only as ``transfer_tensors`` regions; those are registered
        instead, see :meth:`_register_regions`. With ``moriio_backend``
        ``fabric`` the PAGE regions are registered instead, see
        :meth:`_register_pages`.
        """
        self.kv_caches = kv_caches
        if self.use_fabric:
            self._register_pages(kv_caches, transfer_tensors, num_blocks)
            return
        if all(getattr(c, "per_request_state", False) for c in kv_caches.values()):
            self._start_handshake_listener(self._register_regions(transfer_tensors))
            return
        cache_tensor = None

        for layer_name, kv_cache in kv_caches.items():
            if getattr(kv_cache, "per_request_state", False):
                self._register_state(layer_name, kv_cache)
                continue
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
        if self._state_tensors:
            self._check_replay_cursor()
            self._first_transfer_slot = self._num_state_slots - int(
                getattr(self._config, "state_transfer_slots", 0) or 0
            )
            if not self.is_producer and self.transfer_mode == "read":
                raise NotImplementedError(
                    "moriio read mode moves block-addressed KV only, and this "
                    f"model keeps per-request state in {len(self._state_tensors)} "
                    "tensors; use moriio_mode=write"
                )
            logger.info(
                "Registered %d per-request state tensors for RDMA: %d slots, "
                "%d bytes per slot",
                len(self._state_tensors),
                self._num_state_slots,
                sum(slot_bytes for _, slot_bytes, _ in self._state_tensors.values()),
            )

        # The same layout test the transfers use. It also refuses layers shaped
        # unlike the first, so a mixed pool fails here rather than per request.
        is_mla, _per_block_bytes = self._kv_layout()
        self.block_len = self.kv_cache_block_size
        if is_mla:
            self.num_blocks = cache_tensor.shape[0] // self.block_len
        else:
            self.num_blocks = cache_tensor.shape[0]
        state = self._state_tensors
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=None,
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name="aiter",
            state_descs={n: d for n, (d, _, _) in state.items()} or None,
            state_slot_bytes={n: b for n, (_, b, _) in state.items()} or None,
            num_state_slots=self._num_state_slots,
        )
        self._start_handshake_listener(metadata)

    def _register_pages(
        self, kv_caches: dict[str, Any], transfer_tensors: Any, num_blocks: int | None
    ) -> None:
        """Register every PAGE region as one fabric buffer and start the listener.

        The regions all live in the paged KV pool (`allocate_kv_pool`), which
        is registered once; each region is then addressed by its offset in the
        pool and its bytes per block, and the peer gets that table in the
        handshake. mori maps a peer's whole allocation per registered
        descriptor, so one descriptor keeps it to one mapping per peer.
        """
        pages = list(getattr(transfer_tensors, "pages", None) or ())
        if not pages:
            raise NotImplementedError(
                "moriio_backend fabric moves PAGE regions, and this model "
                "publishes none"
            )
        if any(getattr(c, "per_request_state", False) for c in kv_caches.values()) or (
            getattr(transfer_tensors, "swa_block_regions", None)
        ):
            raise NotImplementedError(
                "moriio_backend fabric moves PAGE regions only, and this model "
                "also keeps per-request state"
            )
        regions = [page.region for page in pages]
        if any(region.reverse_indexed for region in regions):
            raise NotImplementedError(
                "moriio_backend fabric addresses PAGE blocks forward only"
            )
        base, size = pool_span([(r.base_addr, r.total_bytes) for r in regions])
        view = pages[0].view
        if view is not None and view.device.index is not None:
            device_id = view.device.index
        else:
            import torch

            device_id = torch.cuda.current_device()
        desc = self.moriio_wrapper.register_local_buffer(base, size, device_id)
        has_handle, vpod = desc_fabric_info(desc)
        if not has_handle:
            raise RuntimeError(
                f"the KV pool at {base:#x} ({size} B) has no fabric handle, so "
                "FABRIC cannot move it: it was not allocated by "
                "MoRIIOConnector.allocate_kv_pool, or the device cannot export "
                "fabric memory"
            )
        self._page_desc = desc
        self._page_regions = [
            (r.base_addr - base, r.unit_bytes, r.semantic_role or "") for r in regions
        ]
        views = [page.view for page in pages]
        self._page_views = views if all(v is not None for v in views) else None
        if self._verify_kv and self._page_views is None:
            logger.warning("ATOM_MORIIO_VERIFY: PAGE regions without views; off")
            self._verify_kv = False
        self.num_blocks = int(num_blocks or getattr(transfer_tensors, "num_blocks", 0))
        self.block_len = self.kv_cache_block_size
        logger.info(
            "MoRIIO fabric: registered %d PAGE regions (%d B per block) as one "
            "buffer of %.2f GiB at %#x, %d blocks, vPOD %d",
            len(regions),
            sum(unit for _, unit, _ in self._page_regions),
            size / (1 << 30),
            base,
            self.num_blocks,
            vpod,
        )
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name="aiter",
            page_desc=desc,
            page_regions=[list(region) for region in self._page_regions],
        )
        self._start_handshake_listener(metadata)

    def _start_handshake_listener(self, metadata: MoRIIOAgentMetadata) -> None:
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
        if (
            self.is_producer
            and self.transfer_mode == "write"
            and envs.ATOM_MORIIO_WRITE_THREAD
        ):
            self._start_write_service()

    def _start_write_service(self) -> None:
        import torch

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._write_service_thread = threading.Thread(
            target=self._write_service_loop,
            args=(device,),
            daemon=True,
            name="moriio-write-service",
        )
        self._write_service_thread.start()

    def _write_service_loop(self, device: int | None) -> None:
        """Serve pushes on their own thread instead of from ``get_finished``.

        A push starts as soon as its WRITE_REQ is queued and is reported as
        soon as it lands, rather than at the next engine step that reaches
        ``get_finished`` through the worker RPC. The consumer's blocks are
        released from here too, so this thread is the producer's only sender.
        """
        if device is not None:
            import torch

            torch.cuda.set_device(device)
        while not self._write_service_stop.is_set():
            if not (self._pending_writes or self._sending_writes):
                self._write_wakeup.wait(self._WRITE_IDLE_WAIT_S)
            # Cleared before the pass, so a request queued during it is
            # served by the next one rather than waiting out the idle timeout.
            self._write_wakeup.clear()
            try:
                self._service_write_requests()
            except Exception:
                logger.exception("MoRIIO write service: a pass failed")
            if self._pending_writes or self._sending_writes:
                time.sleep(self._WRITE_POLL_S)

    def _register_regions(self, transfer_tensors: Any) -> MoRIIOAgentMetadata:
        """Register a backend's PAGE and full-SLOT regions for RDMA.

        PAGE regions (``transfer_tensors.pages``) are indexed forward by
        scheduler block. Full per-request SLOT regions (``swa_block_regions``)
        are indexed by state slot and may be reverse-indexed. Each region is
        cut into unit-aligned chunks under the ``ibv_reg_mr`` limit, so no unit
        straddles two registrations. The compressor-only staging pool is not
        registered: a full SLOT already carries the compressor state.
        """
        tt = transfer_tensors
        if tt is None or not tt.block_regions:
            raise ValueError(
                "MoRIIO: the attention backend registered neither per-layer KV "
                "caches nor transfer regions, so there is nothing to transfer"
            )
        if not self.is_producer and self.transfer_mode != "write":
            raise NotImplementedError(
                "MoRIIO transfer regions are only moved in write mode; set "
                "moriio_mode=write (or ATOM_MORIIO_MODE=write) on the consumer"
            )
        if tt.slot_regions or self._verify_kv:
            raise NotImplementedError(
                "MoRIIO region mode moves PAGE and full SLOT regions only, "
                "without ATOM_MORIIO_VERIFY"
            )

        regions = [
            _Region.of(f"page.{i}.{p.region.semantic_role}", "page", p.region)
            for i, p in enumerate(tt.pages)
        ] + [
            _Region.of(f"slot.{i}.{r.semantic_role}", "slot", r)
            for i, r in enumerate(tt.swa_block_regions)
        ]
        device_id = tt.pages[0].view.device.index
        local_meta: dict[str, list[bytes]] = {}
        for region in regions:
            local_meta[region.name] = [
                self.moriio_wrapper.register_local_buffer(ptr, size, device_id)
                for ptr, size in region.chunks()
            ]

        self._regions = regions
        self.layer_name_to_local_kv_cache_metadata = local_meta
        return MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            region_layout=json.dumps([r.layout() for r in regions]),
        )

    def _remote_regions(self, engine_id: EngineId) -> list[_Region]:
        """The peer's regions, checked against ours."""
        meta = self.remote_moriio_metadata[engine_id]
        if not meta.region_layout:
            raise ValueError(
                f"{engine_id} registered per-layer KV caches but this rank "
                "registered transfer regions; P and D must run the same model"
            )
        remote = [_Region.from_layout(e) for e in json.loads(meta.region_layout)]
        if [r.shape() for r in remote] != [r.shape() for r in self._regions]:
            raise ValueError(
                f"transfer regions differ from {engine_id}: local "
                f"{[r.layout() for r in self._regions]} vs remote {meta.region_layout}"
            )
        return remote

    def _region_session(
        self, engine_id: EngineId, region_idx: int, local_chunk: int, remote_chunk: int
    ) -> Any:
        """The session for one (local chunk, remote chunk) pair, built on use."""
        sessions = self._region_sessions.setdefault(engine_id, {})
        key = (region_idx, local_chunk, remote_chunk)
        session = sessions.get(key)
        if session is None:
            name = self._regions[region_idx].name
            local_metas = self.layer_name_to_local_kv_cache_metadata[name]
            remote_metas = self.layer_name_to_remote_kv_cache_metadata[engine_id][name]
            if local_metas == remote_metas:
                raise ValueError(
                    f"region {name}: {engine_id} advertised descriptors "
                    "byte-identical to ours"
                )
            unpack = self.moriio_wrapper.get_unpack_memory_metadata
            session = self.moriio_wrapper.build_session(
                unpack(local_metas[local_chunk]), unpack(remote_metas[remote_chunk])
            )
            sessions[key] = session
        return session

    def _region_plan(
        self, request: dict[str, Any], engine_id: EngineId
    ) -> list[tuple[Any, list[int], list[int], list[int]]]:
        """Every ``(session, local_offs, remote_offs, sizes)`` batch of a push."""
        dst = [int(b) for b in request["dst_block_ids"]]
        src = [int(b) for b in request["src_block_ids"]]
        src_slot = int(request.get("src_slot_index", -1))
        dst_slot = int(request.get("dst_slot_index", -1))
        remote = self._remote_regions(engine_id)
        plan: list[tuple[Any, list[int], list[int], list[int]]] = []
        for ri, (local, peer) in enumerate(zip(self._regions, remote)):
            if local.kind == "page":
                pairs = list(zip(src, dst))
            else:
                pairs = [(src_slot, dst_slot)]
            groups: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
            for su, du in pairs:
                lc, lo = local.locate(su)
                rc, ro = peer.locate(du)
                group = groups.setdefault((lc, rc), ([], [], []))
                group[0].append(lo)
                group[1].append(ro)
                group[2].append(local.unit)
            for (lc, rc), (l_offs, r_offs, sizes) in groups.items():
                l_offs, r_offs, sizes = self.merge_contiguous_blocks(
                    l_offs, r_offs, sizes
                )
                plan.append(
                    (self._region_session(engine_id, ri, lc, rc), l_offs, r_offs, sizes)
                )
        return plan

    def _write_regions(self, request: dict[str, Any], engine_id: EngineId) -> list[Any]:
        """Push a request's PAGE blocks and SLOT units; return the statuses."""
        plan = self._region_plan(request, engine_id)
        statuses: list[Any] = []
        try:
            for session, l_offs, r_offs, sizes in plan:
                statuses.append(
                    self.moriio_wrapper.write_remote_data_status(
                        sizes, l_offs, r_offs, session
                    )
                )
        except Exception:
            if not statuses:
                raise
            # The queued write has to settle before the source blocks are released.
            logger.exception(
                "transfer %s: could not post every write", request["transfer_id"]
            )
            statuses.append(_POST_FAILED)
        return statuses

    def _register_state(self, layer_name: str, kv_cache: Any) -> None:
        """Register one layer's per-request state, one slot per dim-0 row.

        The KDA forward binds its conv state as ``k_cache``, its SSM state as
        ``v_cache`` and its ReplaySSM records as ``replay_buf_{k,u,g}``, each
        ``(num_slots, ...)`` and contiguous, so a slot is one run of
        ``stride(0)`` elements and moves as a single write. The records'
        cursor is one tensor shared by every layer, registered once.
        """
        parts = (
            ("k", kv_cache.k_cache),
            ("v", kv_cache.v_cache),
            ("replay_k", getattr(kv_cache, "replay_buf_k", None)),
            ("replay_u", getattr(kv_cache, "replay_buf_u", None)),
            ("replay_g", getattr(kv_cache, "replay_buf_g", None)),
        )
        for part, tensor in parts:
            if tensor is None or tensor.numel() == 0:
                continue
            self._register_state_tensor(f"{layer_name}.{part}", tensor)
        cursor = getattr(kv_cache, "replay_write_pos", None)
        if cursor is not None and REPLAY_CURSOR not in self._state_tensors:
            self._register_state_tensor(REPLAY_CURSOR, cursor)

    def _register_state_tensor(self, name: str, tensor: Any) -> None:
        """Register one ``(num_slots, ...)`` state tensor under ``name``."""
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name}: state shaped {tuple(tensor.shape)} with strides "
                f"{tensor.stride()} is not one run of bytes per slot"
            )
        num_slots = tensor.shape[0]
        if self._num_state_slots not in (0, num_slots):
            raise ValueError(
                f"{name} has {num_slots} slots, the state registered before "
                f"it {self._num_state_slots}"
            )
        slot_bytes = tensor.stride(0) * tensor.element_size()
        if num_slots * slot_bytes > MAX_RDMA_CHUNK_BYTES:
            raise ValueError(
                f"{name}: {num_slots} slots of {slot_bytes} bytes exceed one "
                "RDMA registration"
            )
        device_id = tensor.device.index if tensor.device.index is not None else -1
        desc = self.moriio_wrapper.register_local_buffer(
            tensor.data_ptr(), num_slots * slot_bytes, device_id
        )
        self._num_state_slots = num_slots
        self._state_tensors[name] = (desc, slot_bytes, tensor)

    def _check_replay_cursor(self) -> None:
        """Refuse ReplaySSM records registered without their cursor."""
        records = [name for name in self._state_tensors if ".replay_" in name]
        if records and REPLAY_CURSOR not in self._state_tensors:
            # Pushed alone, the records land under the consumer slot's stale
            # cursor, and its first decode folds the previous tenant's records.
            raise ValueError(
                f"{len(records)} ReplaySSM record tensors (first {records[0]}) "
                "came without the record cursor `replay_write_pos`"
            )

    @staticmethod
    def _transfer_mode_from(
        kv_transfer_config: dict, env_mode: str | None = None
    ) -> str:
        """``"read"`` or ``"write"``: which side posts the RDMA verbs.

        Only the consumer's mode matters: it either pulls with reads or asks
        for a push, and the producer serves both.
        """
        if env_mode is None:
            env_mode = envs.ATOM_MORIIO_MODE
        mode = env_mode or kv_transfer_config.get("moriio_mode", "read")
        mode = str(mode).lower()
        if mode not in ("read", "write"):
            raise ValueError(f"moriio_mode must be 'read' or 'write', got {mode!r}")
        return mode

    @staticmethod
    def _engine_name_with_dp(engine_name: str, dp_rank: int) -> str:
        """Build a unique engine identifier that includes the DP rank."""
        return f"{engine_name}_dp{dp_rank}"

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Start moving the KV of every pending receive request.

        Called by the worker process each step. In read mode, a request whose
        producer rank has not been handshaken yet waits on a background
        handshake, and ``get_finished`` issues its reads.
        """
        if self.is_producer:
            self._note_state_stages(getattr(metadata, "state_stages", None))
            return

        if metadata is not None and metadata.reqs_to_recv:
            logger.debug("Starting KV load for %d requests", len(metadata.reqs_to_recv))

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        if self.transfer_mode == "write":
            if metadata.reqs_to_recv:
                import torch

                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                self._gated_writes.append((event, list(metadata.reqs_to_recv.items())))
                self._request_gated_writes()
            return

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = f"{meta.remote_host}:{meta.remote_handshake_port}"
            meta.remote_engine_id = remote_engine_id
            # _remote_agents is keyed by the dp-suffixed id.
            peer = self._engine_name_with_dp(
                remote_engine_id, int(meta.remote_dp_rank or 0)
            )
            with self._handshake_lock:
                known = peer in self._remote_agents
            if known:
                self._issue_read_for_req(req_id, meta)
                continue
            # Not waited for here: this runs on the worker's busy loop.
            self._awaiting_handshake[req_id] = (
                time.monotonic() + self._handshake_timeout_s,
                meta,
            )
            self._initiate_background_handshake(req_id, remote_engine_id, meta)

    def _note_state_stages(self, stages: dict | None) -> None:
        """Mark the transfers whose state the runner just copied to a transfer slot.

        The copies were enqueued on this stream before this call; a push reads
        the slot only once the event recorded after them has fired.
        """
        if not stages:
            return
        import torch

        landed = torch.cuda.Event()
        landed.record()
        for transfer_id, (_src, dst) in stages.items():
            self._staged_sources[int(transfer_id)] = (int(dst), landed)

    def _state_source_ready(self, request: dict[str, Any]) -> bool | None:
        """Whether the push may read its state slot: True, False, or None.

        None while a transfer slot's copy has not landed. False when the slot
        is not the one the transfer was staged into.
        """
        src = int(request.get("src_slot_index", -1))
        if src < self._first_transfer_slot:
            return True
        staged = self._staged_sources.get(int(request["transfer_id"]))
        if staged is None:
            return None
        slot, landed = staged
        if slot != src:
            logger.error(
                "transfer %s: asked to push transfer slot %d, staged into %d",
                request["transfer_id"],
                src,
                slot,
            )
            return False
        return True if landed.query() else None

    def _issue_ready_reads(self) -> None:
        """Issue the reads whose handshake finished; fail those that cannot start."""
        while True:
            try:
                req_id, meta, error = self._ready_requests.get_nowait()
            except queue.Empty:
                break
            # Gone if it already timed out below.
            if self._awaiting_handshake.pop(req_id, None) is None:
                continue
            if error is not None:
                logger.error(
                    "req %s: handshake with %s failed (%s); failing it",
                    req_id,
                    meta.remote_engine_id,
                    error,
                )
                self._fail_before_transfer(req_id, meta)
                continue
            self._issue_read_for_req(req_id, meta)

        now = time.monotonic()
        for req_id, (deadline, meta) in list(self._awaiting_handshake.items()):
            if now >= deadline:
                logger.error(
                    "req %s: no handshake with its producer after %.0fs; failing it",
                    req_id,
                    self._handshake_timeout_s,
                )
                del self._awaiting_handshake[req_id]
                self._fail_before_transfer(req_id, meta)

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
            self._fail_before_transfer(req_id, meta)
            return
        kept = self._kept_prefix_blocks(meta)
        try:
            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote_engine_id,
                local_block_ids=list(meta.local_block_ids)[kept:],
                remote_block_ids=list(meta.remote_block_ids or [])[kept:],
                kept_block_ids=list(meta.local_block_ids)[:kept],
                remote_host=meta.remote_host,
                remote_handshake_port=meta.remote_handshake_port,
                remote_dp_rank=meta.remote_dp_rank,
                remote_tp_size=int(meta.tp_size or 0),
            )
        except Exception:
            # This runs on the worker's busy_loop, so letting this escape
            # takes down the rank and with it every other request on it.
            logger.exception("req %s: could not read KV", req_id)
            with self.moriio_wrapper.lock:
                posted = self._recving_transfers.get(req_id)
                if posted:
                    # Failed once the reads already posted have settled.
                    posted.append(_POST_FAILED)
            if not posted:
                self._fail_before_transfer(req_id, meta)

    def _fail_before_transfer(self, req_id: ReqId, meta: ReqMeta) -> None:
        """Fail a receive that moved nothing, and let the producer release it."""
        self._failed_before_read.add(req_id)
        self._notify_producer(req_id, meta)

    def _notify_producer(self, req_id: ReqId, meta: ReqMeta) -> None:
        """Best-effort done-notify; the producer holds the source blocks until
        it hears back. Repeats are harmless."""
        try:
            port = int(meta.remote_handshake_port) + get_port_offset(
                meta.remote_dp_rank,
                self.tp_rank,
                int(meta.tp_size or 0) or self.tp_size,
            )
            self.moriio_wrapper.send_notify(
                self.request_id_to_transfer_id.get(req_id, meta.transfer_id),
                meta.remote_host,
                port,
            )
        except Exception:
            logger.warning(
                "req %s: could not tell the producer to release its blocks",
                req_id,
                exc_info=True,
            )

    @staticmethod
    def _kept_prefix_blocks(meta: ReqMeta) -> int:
        """Leading blocks the consumer already holds; they are not moved."""
        return max(0, int(getattr(meta, "num_computed_blocks", 0) or 0))

    def _note_kept_prefix(self, req_id: ReqId, kept_ids: list[int], is_mla: bool):
        """Under verify, digest the reused prefix so a write into it is caught."""
        if kept_ids:
            rows = self._verify_rows_for(kept_ids, is_mla)
            self._verify_kept[req_id] = (rows, self._verify_digest(rows))

    @staticmethod
    def _rejects_transfer_subset(req_id: ReqId, meta: ReqMeta) -> bool:
        """Whether the request asks for a subset neither direction implements.

        Both paths move every advertised block, paired one to one. A request to
        skip already-computed blocks is safe to over-serve; a source stride is
        not, because the pairing would put blocks in the wrong place.
        """
        if getattr(meta, "src_block_skip_factor", 1) != 1:
            logger.error(
                "req %s: src_block_skip_factor=%s is not implemented by the "
                "MoRIIO connector; blocks would be paired one to one",
                req_id,
                meta.src_block_skip_factor,
            )
            return True
        return False

    def _request_gated_writes(self) -> None:
        """Ask for the pushes whose step's earlier forwards have finished."""
        while self._gated_writes and self._gated_writes[0][0].query():
            _, reqs = self._gated_writes.popleft()
            for req_id, meta in reqs:
                self._request_write(req_id, meta)

    def record_kv_cache_ready(self, req_ids: list[ReqId]) -> None:
        """Record when this prefill batch's KV writes are done on the GPU."""
        if not (self.is_producer and self.transfer_mode == "write") or not req_ids:
            return
        import torch

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        now = time.monotonic()
        ids = [int(req_id) for req_id in req_ids]
        # Past the write timeout and the hold no push for it can start, and
        # long before that its forward has finished.
        cutoff = now - self._write_timeout_s - envs.ATOM_PD_FAILED_WRITE_HOLD_S
        expired = self._kv_ready_expired
        with self._kv_ready_lock:
            for transfer_id in ids:
                self._kv_ready_events[transfer_id] = event
                expired.pop(transfer_id, None)
            self._kv_ready_batches.append((now, event, ids))
            while self._kv_ready_batches and self._kv_ready_batches[0][0] < cutoff:
                _, old, old_ids = self._kv_ready_batches.popleft()
                for transfer_id in old_ids:
                    if self._kv_ready_events.get(transfer_id) is old:
                        del self._kv_ready_events[transfer_id]
                        expired[transfer_id] = None
            while len(expired) > self._KV_READY_EXPIRED_MAX:
                del expired[next(iter(expired))]

    def note_forward(self, batch: Any, block_size: int, num_spec_tokens: int) -> None:
        """ATOM_MORIIO_DIAG_POST_READY: log each request this forward runs
        after its final prefill forward recorded the KV-ready event.

        Called before the forward, from the CPU-side batch only. The written
        positions are the target's window and the draft's `k - 1` lookahead
        past it (`draft_kv_lookahead`); a deferred decode row's window may sit
        `num_rejected` earlier.
        """
        if not (
            self._diag_post_ready and self.is_producer and self.transfer_mode == "write"
        ):
            return
        req_ids = getattr(batch, "req_ids", None) or []
        with self._kv_ready_lock:
            rows = [
                (row, int(req_id))
                for row, req_id in enumerate(req_ids)
                if int(req_id) in self._kv_ready_events
                or int(req_id) in self._kv_ready_expired
            ]
        if not rows:
            return
        now = time.monotonic()
        tables = batch.block_tables
        if len(tables) != len(req_ids):
            tables = None
        for row, transfer_id in rows:
            scheduled = int(batch.num_scheduled_tokens[row])
            end = int(batch.context_lens[row])
            first = end - scheduled
            last = end + max(num_spec_tokens - 2, 0)
            table = tables[row] if tables is not None else []
            indices = range(max(first, 0) // block_size, last // block_size + 1)
            touched = sorted({table[i] for i in indices if i < len(table)})
            push = self._push_request(transfer_id)
            src = (push or {}).get("src_block_ids") or []
            overlap = sorted(set(touched) & {int(b) for b in src})
            forwards = self._post_ready_forwards.setdefault(transfer_id, [])
            forwards.append(now)
            logger.warning(
                "[moriio-diag] transfer %d: forward after its KV-ready event "
                "(rows %d, prefill rows %d, dummy %s; this row: %d tokens at "
                "positions %d..%d, draft through %d, num_rejected %d, %d blocks "
                "in its table, writes blocks %s; push %s, its %d source blocks "
                "%s)",
                transfer_id,
                len(req_ids),
                int(getattr(batch, "total_seqs_num_prefill", 0)),
                bool(getattr(batch, "is_dummy_run", False)),
                scheduled,
                first,
                end - 1,
                last,
                int(batch.num_rejected[row]),
                len(table),
                touched,
                self._push_state(transfer_id, now),
                len(src),
                f"include {overlap}" if overlap else "are not among them",
            )
        while len(self._post_ready_forwards) > self._KV_READY_EXPIRED_MAX:
            del self._post_ready_forwards[next(iter(self._post_ready_forwards))]

    def _push_request(self, transfer_id: int) -> dict[str, Any] | None:
        """The WRITE_REQ of a push queued or in flight, if any."""
        sending = self._sending_writes.get(transfer_id)
        if sending is not None:
            return sending[1]
        for request, _ in list(self._pending_writes):
            if int(request["transfer_id"]) == transfer_id:
                return request
        return None

    def describe_blocks(self, blocks: list[int]) -> str:
        """ATOM_DRAFT_SLOT_CHECK: which transfers in flight hold these blocks.

        Read from another thread's bookkeeping without its lock, as a
        diagnostic: an entry may be a moment stale.
        """
        wanted = set(blocks)
        hits = []
        if self.is_producer:
            pushes = [
                (int(tid), request, "pushing")
                for tid, (_, request) in list(self._sending_writes.items())
            ] + [
                (int(request["transfer_id"]), request, "queued")
                for request, _ in list(self._pending_writes)
            ]
            for transfer_id, request, state in pushes:
                held = wanted & {int(b) for b in request.get("src_block_ids") or []}
                if held:
                    hits.append(
                        f"source of transfer {transfer_id} ({state}): {sorted(held)[:8]}"
                    )
        else:
            for req_id, (_, meta) in list(self._awaiting_write.items()):
                held = wanted & {int(b) for b in meta.local_block_ids}
                if held:
                    hits.append(f"target of req {req_id}'s push: {sorted(held)[:8]}")
        return "; ".join(hits) or "no transfer in flight holds them"

    def _push_state(self, transfer_id: int, now: float) -> str:
        """How far a transfer's push is, for the diagnostic log."""
        sending = self._sending_writes.get(transfer_id)
        if sending is not None:
            stats = sending[1].get("_page_stats")
            started = stats[0] if stats else now
            return f"in flight for {(now - started) * 1e3:.2f} ms"
        if transfer_id not in self._write_ids:
            return "not asked for yet"
        finished = self._write_ids[transfer_id]
        if finished is None:
            return "queued or starting"
        return f"finished {(now - finished) * 1e3:.2f} ms ago"

    def _source_ready(self, transfer_id: int) -> bool:
        """Whether the prefill that wrote this transfer's source has finished.

        Kept after a push starts: more than one consumer rank may ask for the
        same transfer.
        """
        with self._kv_ready_lock:
            event = self._kv_ready_events.get(transfer_id)
            if event is None:
                return transfer_id in self._kv_ready_expired
        return event.query()

    def _request_write(self, req_id: ReqId, meta: ReqMeta) -> None:
        """Ask the producer to push this request's KV into our blocks.

        The consumer half of serial WRITE mode. The producer has already run
        the prefill and holds the source blocks until it reports back, so all
        that is left is to tell it where to write.
        """
        if self._rejects_transfer_subset(req_id, meta):
            self._fail_before_transfer(req_id, meta)
            return
        kept = self._kept_prefix_blocks(meta)
        kept_ids = [int(b) for b in meta.local_block_ids[:kept]]
        dst = [int(b) for b in meta.local_block_ids[kept:]]
        src = [int(b) for b in (meta.remote_block_ids or [])[kept:]]
        try:
            self._source_covers(f"req {req_id}", dst, src)
        except ValueError as error:
            logger.error("%s", error)
            self._fail_before_transfer(req_id, meta)
            return
        src_slot = int(getattr(meta, "remote_slot_index", -1))
        dst_slot = int(getattr(meta, "local_slot_index", -1))
        remote_tp_size = int(meta.tp_size or 0) or self.tp_size
        if self._regions is not None:
            has_slots = any(r.kind == "slot" for r in self._regions)
            if remote_tp_size != self.tp_size or (
                has_slots and (src_slot < 0 or dst_slot < 0)
            ):
                logger.error(
                    "req %s: region transfer needs equal TP (producer %d, "
                    "consumer %d) and a state slot on both ends (src=%d, dst=%d)",
                    req_id,
                    remote_tp_size,
                    self.tp_size,
                    src_slot,
                    dst_slot,
                )
                self._fail_before_transfer(req_id, meta)
                return
        elif self._state_tensors and (src_slot < 0 or dst_slot < 0):
            # The KV alone would land, and decode would resume the recurrent
            # state from whatever its slot last held.
            logger.error(
                "req %s: this model keeps per-request state, but the transfer "
                "names state slot %d -> %d; refusing to move the KV without it",
                req_id,
                src_slot,
                dst_slot,
            )
            self._fail_before_transfer(req_id, meta)
            return

        if self._verify_kv and self.use_fabric:
            self._verify_page_dst[req_id] = dst
        elif self._verify_kv:
            first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
            is_mla = self.kv_caches[first_layer].v_cache is None
            rows = self._verify_rows_for(dst, is_mla)
            self._verify_rows[req_id] = rows
            self._verify_before[req_id] = self._verify_digest(rows)
            self._note_kept_prefix(req_id, kept_ids, is_mla)
            if self._state_tensors:
                self._verify_state_before[req_id] = (
                    dst_slot,
                    self._state_digest(dst_slot),
                )
        port = int(meta.remote_handshake_port) + get_port_offset(
            int(meta.remote_dp_rank or 0), self.tp_rank, remote_tp_size
        )
        # The producer starts no push after this. Wall clock, so P and D clocks
        # must agree to well within ATOM_PD_FAILED_WRITE_HOLD_S.
        deadline = time.time() + self._write_timeout_s
        request = {
            "transfer_id": self.request_id_to_transfer_id.get(req_id, meta.transfer_id),
            "decode_req_id": req_id,
            "decode_host": self.local_ip,
            "decode_handshake_port": self.base_handshake_port,
            "decode_dp_rank": self.dp_rank,
            "decode_tp_size": self.tp_size,
            "dst_block_ids": dst,
            "src_block_ids": src[: len(dst)],
            "deadline": deadline,
            "src_slot_index": src_slot,
            "dst_slot_index": dst_slot,
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
            if self.use_fabric:
                self._verify_page_dst.pop(req_id, None)
            self._verify_rows.pop(req_id, None)
            self._verify_before.pop(req_id, None)
            self._verify_kept.pop(req_id, None)
            self._verify_state_before.pop(req_id, None)
            self._fail_before_transfer(req_id, meta)
            return
        self._awaiting_write[req_id] = (deadline, meta)

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

    @staticmethod
    def _source_covers(label: str, dst: list[int], src: list[int]) -> None:
        """Raise unless every destination block has a source block.

        A longer source is normal: the producer's table also covers the token
        it generated, which the consumer never allocates for.
        """
        if len(src) < len(dst):
            raise ValueError(
                f"{label}: {len(dst)} destination blocks but only {len(src)} "
                "source blocks; the tail of the prompt's KV has no source"
            )

    def _get_or_build_sessions(
        self, remote_engine_id: str
    ) -> tuple[list[tuple[dict, dict]], MoRIIOAgentMetadata]:
        """Return cached RDMA sessions for the remote engine, building if needed.

        With chunked registration, local block ``b`` and remote block ``r``
        may reside in different chunks, so we build a local x remote session
        grid for K chunks and another for V chunks.

        Returns ``(per_layer_sessions, remote_meta)`` where each layer
        entry is ``(k_sessions_dict, v_sessions_dict)`` keyed by
        ``(local_chunk_idx, remote_chunk_idx)``.
        """
        if remote_engine_id not in self._built_sessions:
            nk = self.num_k_chunks
            remote_meta = self.remote_moriio_metadata[remote_engine_id]
            # Both ends cut a pool at the same blocks per chunk, which depends
            # only on the size of a block, but each registers as many chunks as
            # its own pool needs. P and D size their KV caches independently,
            # so one can cross a chunk boundary the other does not.
            remote_nk = -(-remote_meta.num_blocks // self.blocks_per_chunk)
            per_layer_sessions: list[tuple[dict, dict]] = []
            for ln, local_metas in self.layer_name_to_local_kv_cache_metadata.items():
                remote_metas = self.layer_name_to_remote_kv_cache_metadata[
                    remote_engine_id
                ][ln]
                has_v = len(local_metas) > nk
                if len(remote_metas) != remote_nk * (2 if has_v else 1):
                    raise ValueError(
                        f"layer {ln}: {remote_engine_id} advertised "
                        f"{len(remote_metas)} descriptors for "
                        f"{remote_meta.num_blocks} blocks, not {remote_nk} per "
                        f"cache at {self.blocks_per_chunk} blocks per chunk"
                    )
                # Packed descriptors carry the owning engine's key, so equal
                # bytes mean the handshake handed back our own. Every transfer
                # would then copy this rank onto itself and still report
                # success.
                if local_metas == remote_metas:
                    raise ValueError(
                        f"layer {ln}: {remote_engine_id} advertised descriptors "
                        "byte-identical to ours; every transfer would copy this "
                        "rank's own blocks and the KV would never arrive"
                    )

                def _unpack(packed):
                    return self.moriio_wrapper.get_unpack_memory_metadata(packed)

                # K sessions: grid over the first nk / remote_nk entries
                k_sessions: dict[tuple[int, int], Any] = {}
                for lci in range(nk):
                    for rci in range(remote_nk):
                        local_md = _unpack(local_metas[lci])
                        remote_md = _unpack(remote_metas[rci])
                        k_sessions[(lci, rci)] = self.moriio_wrapper.build_session(
                            local_md, remote_md
                        )

                # V sessions: the same grid over the entries after the K chunks
                v_sessions: dict[tuple[int, int], Any] = {}
                if has_v:
                    for lci in range(nk):
                        for rci in range(remote_nk):
                            local_md = _unpack(local_metas[nk + lci])
                            remote_md = _unpack(remote_metas[remote_nk + rci])
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

    def _get_or_build_state_sessions(self, engine_id: str) -> list[tuple[Any, int]]:
        """``(session, bytes_per_slot)`` per state tensor, in registration order.

        Built once per consumer, after checking it registered the same state
        tensors with the same slot size: any difference would land every
        request's state at the wrong offset without failing a write.
        """
        if engine_id not in self._built_state_sessions:
            remote = self.remote_moriio_metadata[engine_id]
            ours = {
                name: slot_bytes
                for name, (_, slot_bytes, _) in self._state_tensors.items()
            }
            theirs = remote.state_slot_bytes or {}
            if theirs != ours:
                differ = sorted(
                    name
                    for name in ours.keys() | theirs.keys()
                    if ours.get(name) != theirs.get(name)
                )
                # ReplaySSM and the draft length shape the state; both ends
                # must run the same ATOM_ENABLE_REPLAYSSM and DSpark size.
                raise ValueError(
                    f"{engine_id} registered {len(theirs)} per-request state "
                    f"tensors to our {len(ours)}; {len(differ)} differ, first "
                    f"{differ[0]}: ours {ours.get(differ[0])} B/slot, theirs "
                    f"{theirs.get(differ[0])} B/slot"
                )
            unpack = self.moriio_wrapper.get_unpack_memory_metadata
            sessions = []
            for name, (desc, slot_bytes, _) in self._state_tensors.items():
                remote_desc = remote.state_descs[name]
                if remote_desc == desc:
                    # Our own descriptor back: every write would copy this
                    # rank's slot onto itself and still succeed.
                    raise ValueError(f"{engine_id} advertised our own {name}")
                session = self.moriio_wrapper.build_session(
                    unpack(desc), unpack(remote_desc)
                )
                sessions.append((session, slot_bytes))
            self._built_state_sessions[engine_id] = sessions
        return self._built_state_sessions[engine_id]

    def _get_or_build_page_session(
        self, engine_id: str
    ) -> tuple[Any, list[tuple[int, int]]]:
        """The (local pool, remote pool) FABRIC session and the peer's regions.

        The peer's regions come back as ``(offset in its pool, bytes per
        block)``, paired with ours by position after checking that both ends
        published the same regions in the same order.
        """
        remote = self.remote_moriio_metadata[engine_id]
        if remote.page_desc is None or remote.page_regions is None:
            raise ValueError(
                f"{engine_id} advertised no PAGE layout; both ends need "
                "moriio_backend fabric"
            )
        theirs = [(int(o), int(u), str(r)) for o, u, r in remote.page_regions]
        if engine_id not in self._built_sessions:
            if remote.page_desc == self._page_desc:
                raise ValueError(
                    f"{engine_id} advertised our own PAGE descriptor; every "
                    "transfer would copy this rank onto itself"
                )
            ours = [(unit, role) for _, unit, role in self._page_regions]
            if ours != [(unit, role) for _, unit, role in theirs]:
                first = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(ours, theirs))
                        if a != (b[1], b[2])
                    ),
                    min(len(ours), len(theirs)),
                )
                raise ValueError(
                    f"{engine_id} published {len(theirs)} PAGE regions to our "
                    f"{len(ours)}; first difference at region {first}"
                )
            has_handle, vpod = desc_fabric_info(remote.page_desc)
            if not has_handle:
                raise RuntimeError(f"{engine_id}'s KV pool has no fabric handle")
            unpack = self.moriio_wrapper.get_unpack_memory_metadata
            session = self.moriio_wrapper.build_session(
                unpack(self._page_desc), unpack(remote.page_desc)
            )
            if session is None:
                raise RuntimeError(
                    f"no FABRIC session to {engine_id} (vPOD {vpod}, ours "
                    f"{desc_fabric_info(self._page_desc)[1]}): the peer is "
                    "outside this vPOD or its pool cannot be imported"
                )
            self._built_sessions[engine_id] = [session]
            logger.info(
                "MoRIIO fabric: session to %s over %d PAGE regions",
                engine_id,
                len(theirs),
            )
        return self._built_sessions[engine_id][0], [(o, u) for o, u, _ in theirs]

    def _page_batch(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_regions: list[tuple[int, int]],
        into_remote: bool,
    ) -> tuple[list[int], list[int], list[int]]:
        """Pool-relative (local offsets, remote offsets, sizes) of one request,
        read into the local blocks or written into the remote ones."""
        written = remote_block_ids if into_remote else local_block_ids
        repeated = repeated_blocks(written)
        if repeated:
            raise ValueError(
                f"{'remote' if into_remote else 'local'} blocks {repeated[:8]} "
                f"appear more than once among the {len(written)} a fabric batch "
                "writes; its copies run concurrently, so they must not overlap"
            )
        runs = block_runs(local_block_ids, remote_block_ids)
        local_regions = [(offset, unit) for offset, unit, _ in self._page_regions]
        return page_offsets(local_regions, remote_regions, runs)

    def _write_pages(
        self, request: dict[str, Any], engine_id: str, src: list[int], dst: list[int]
    ) -> list[Any]:
        """Push every PAGE region's source blocks into the consumer's, as one batch."""
        session, remote_regions = self._get_or_build_page_session(engine_id)
        l_offs, r_offs, sizes = self._page_batch(
            src, dst, remote_regions, into_remote=True
        )
        if self._verify_kv:
            request["page_digest"] = self._page_digests(src)
            request["_page_src"] = list(src)
        request["_page_stats"] = (time.monotonic(), sum(sizes), len(dst), len(sizes))
        return [
            self.moriio_wrapper.write_remote_data_status(sizes, l_offs, r_offs, session)
        ]

    def _log_transfer_stats(
        self,
        req: Any,
        mode: str,
        ok: bool,
        stats: tuple[float, int, int, int],
        waited_s: float,
    ) -> None:
        posted, nbytes, blocks, segments = stats
        write_ms = (time.monotonic() - posted) * 1e3
        logger.info(
            "MoRIIO transfer stats: req=%s protocol=fabric-%s ok=%s blocks=%d "
            "bytes=%d wait_ms=%.3f write_ms=%.3f gbps=%.1f segments=%d tp=%d",
            req,
            mode,
            ok,
            blocks,
            nbytes,
            waited_s * 1e3,
            write_ms,
            nbytes / max(write_ms, 1e-6) / 1e6,
            segments,
            self.tp_rank,
        )

    def _page_digests(self, block_ids: list[int]) -> list[Any]:
        """``[digest, digest of all but the last block, per-region digests]``
        of ``block_ids``; past the prompt the last block holds slots decode
        writes itself."""
        regions = page_region_digests(self._page_views, block_ids)
        full = sum(regions)
        if len(block_ids) < 2:
            return [full, full, regions]
        return [full, page_digest(self._page_views, block_ids[:-1]), regions]

    def _changed_regions(self, before: list[int], after: list[int]) -> str:
        """The regions whose digests differ, as a count and their roles."""
        changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
        roles = sorted({self._page_regions[i][2] or "?" for i in changed})
        return f"{len(changed)} of {len(before)} regions {changed[:12]} roles {roles}"

    def _recheck_page_source(self, request: dict[str, Any]) -> None:
        """Digest the source again once the push has landed: a change means
        something wrote those blocks while they were being pushed."""
        src = request.get("_page_src")
        if not src:
            return
        after = self._page_digests(src)
        before = request["page_digest"]
        if after[0] != before[0]:
            logger.error(
                "[moriio-verify] transfer %s: the %d source blocks changed during "
                "the push (all but the last block %s; %s)%s",
                request["transfer_id"],
                len(src),
                "unchanged" if after[1] == before[1] else "changed too",
                self._changed_regions(before[2], after[2]),
                self._post_ready_summary(request),
            )

    def _post_ready_summary(self, request: dict[str, Any]) -> str:
        """ATOM_MORIIO_DIAG_POST_READY: the forwards this rank ran over the
        transfer after its KV-ready event, timed from the push's start."""
        if not self._diag_post_ready:
            return ""
        forwards = self._post_ready_forwards.get(int(request["transfer_id"]), [])
        stats = request.get("_page_stats")
        if not forwards or not stats:
            return "; no forward ran it after its KV-ready event on this rank"
        offsets = ", ".join(f"{(t - stats[0]) * 1e3:+.2f}" for t in forwards[:8])
        return (
            f"; {len(forwards)} forward(s) ran it after its KV-ready event on "
            f"this rank, at {offsets} ms from the push start"
        )

    def _verify_page_landed(self, req_id: ReqId) -> None:
        """Compare the destination blocks with the producer's source digest."""
        dst = self._verify_page_dst.pop(req_id, None)
        with self._write_reports_lock:
            source = self._write_page_digests.pop(req_id, None)
        if dst is None:
            return
        if source is None:
            logger.error(
                "[moriio-verify] req %s: the producer sent no PAGE digest", req_id
            )
            return
        landed = self._page_digests(dst)
        if landed[0] != source[0]:
            import torch

            # Read again once the device is idle: a match then means the push
            # landed but was not yet visible here, not that it moved the
            # wrong bytes.
            torch.cuda.synchronize()
            time.sleep(0.05)
            again = self._page_digests(dst)[0]
            logger.error(
                "[moriio-verify] req %s: the %d PAGE regions of %d blocks differ "
                "from the producer's (digest %d, producer %d, 50 ms later %s; "
                "all but the last block %s; %s) -- decode will attend over KV "
                "that is not the prompt's",
                req_id,
                len(self._page_regions),
                len(dst),
                landed[0],
                source[0],
                "matches" if again == source[0] else f"still differs ({again})",
                "match" if landed[1] == source[1] else "differ too",
                self._changed_regions(source[2], landed[2]),
            )
        else:
            logger.info(
                "[moriio-verify] req %s: all %d PAGE regions of %d blocks match "
                "the producer",
                req_id,
                len(self._page_regions),
                len(dst),
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
                "[moriio-verify] req %s: all %d layers changed (%d rows received)",
                req_id,
                len(before),
                rows.numel(),
            )
        self._verify_kept_landed(req_id)

    def _verify_kept_landed(self, req_id: ReqId) -> None:
        """Report a reused prefix block the receive wrote into."""
        kept = self._verify_kept.pop(req_id, None)
        if kept is None:
            return
        rows, before = kept
        after = self._verify_digest(rows)
        changed = [name for name, value in before.items() if after[name] != value]
        if changed:
            logger.error(
                "[moriio-verify] req %s: %d of %d layers changed in the %d reused "
                "prefix rows (first %s) -- the receive wrote into blocks it kept",
                req_id,
                len(changed),
                len(before),
                rows.numel(),
                changed[0],
            )
        else:
            logger.info(
                "[moriio-verify] req %s: %d reused prefix rows unchanged",
                req_id,
                rows.numel(),
            )

    def _state_digest(self, slot: int) -> dict[str, int]:
        """Position-weighted byte digest of one slot in every state tensor.

        Weighted like `_verify_digest`, so a slot still holding an earlier
        request's state with the same byte sum reads as different.
        """
        import torch

        rows = [
            tensor[slot].reshape(-1).view(torch.uint8)
            for _, _, tensor in self._state_tensors.values()
        ]
        weights = (
            torch.arange(
                max(row.numel() for row in rows),
                dtype=torch.int64,
                device=rows[0].device,
            )
            % 65521
            + 1
        )
        sums = [torch.sum(row.to(torch.int64) * weights[: row.numel()]) for row in rows]
        return dict(zip(self._state_tensors, torch.stack(sums).tolist()))

    def _verify_state_landed(self, req_id: ReqId) -> None:
        """Compare the destination slot with the producer's source digest."""
        entry = self._verify_state_before.pop(req_id, None)
        with self._write_reports_lock:
            source = self._write_state_digests.pop(req_id, None)
        if entry is None:
            return
        slot, before = entry
        after = self._state_digest(slot)
        if source is None:
            logger.error(
                "[moriio-verify] req %s: the producer sent no state digest", req_id
            )
            return
        differ = [name for name, value in after.items() if source.get(name) != value]
        if differ:
            logger.error(
                "[moriio-verify] req %s: %d of %d state tensors differ from the "
                "producer's slot (first %s: %d before, %d after, producer %d) -- "
                "decode will resume from state the prompt never produced",
                req_id,
                len(differ),
                len(after),
                differ[0],
                before[differ[0]],
                after[differ[0]],
                source.get(differ[0], -1),
            )
        else:
            logger.info(
                "[moriio-verify] req %s: all %d state tensors match the producer",
                req_id,
                len(after),
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
        kept_block_ids: list[int] | None = None,
    ) -> None:
        """Issue RDMA reads for all layers of a single request.

        Block pairs are grouped by ``(local_chunk_idx, remote_chunk_idx)``
        and chunk-relative offsets are computed.  Each group issues a
        separate RDMA batch read using the matching session from the local x
        remote chunk grid.

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
        self._source_covers(f"req {request_id}", local_block_ids, remote_block_ids)
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

        # Notify port = base handshake port + offset(remote_dp_rank, local_tp_rank).
        # Strided by the *remote* TP size, since the port belongs to the peer.
        notify_port = remote_handshake_port + get_port_offset(
            remote_dp_rank, self.tp_rank, remote_tp_size or self.tp_size
        )

        if self.use_fabric:
            session, remote_regions = self._get_or_build_page_session(dp_engine_id)
            local = local_block_ids
            l_offs, r_offs, sizes = self._page_batch(
                local, remote_block_ids[: len(local)], remote_regions, into_remote=False
            )
            self._page_reads[request_id] = (
                time.monotonic(),
                sum(sizes),
                len(local),
                len(sizes),
            )
            status = self.moriio_wrapper.read_remote_data(
                sizes, l_offs, r_offs, session
            )
            with self.moriio_wrapper.lock:
                self._recving_transfers[request_id].append(status)
                self._recving_transfers_callback_addr[request_id] = (
                    remote_host,
                    str(notify_port),
                    dp_engine_id,
                )
            return

        sessions, _remote_meta = self._get_or_build_sessions(dp_engine_id)
        is_mla, per_block_bytes = self._kv_layout()
        groups = self._group_offsets(local_block_ids, remote_block_ids, per_block_bytes)

        layer_names = list(self.layer_name_to_local_kv_cache_metadata.keys())

        if self._verify_kv:
            rows = self._verify_rows_for(local_block_ids, is_mla)
            self._verify_rows[request_id] = rows
            self._verify_before[request_id] = self._verify_digest(rows)
            self._note_kept_prefix(request_id, kept_block_ids or [], is_mla)

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
                        dp_engine_id,
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
        on the matching session of the local x remote grid. Shared by both
        directions, so a read and a write of the same pair always touch the
        same bytes.
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
        self._source_covers(f"transfer {request['transfer_id']}", dst, src)
        if self.use_fabric:
            return self._write_pages(request, engine_id, src[: len(dst)], dst)
        if self._regions is not None:
            return self._write_regions(request, engine_id)
        sessions, _remote_meta = self._get_or_build_sessions(engine_id)
        # Before any block write: a request that cannot carry its state must not
        # leave a half-filled destination behind.
        state_writes = self._state_writes(request, engine_id)
        _is_mla, per_block_bytes = self._kv_layout()
        groups = self._group_offsets(src[: len(dst)], dst, per_block_bytes)

        statuses: list[Any] = []
        try:
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
            for session, size, local_offset, remote_offset in state_writes:
                statuses.append(
                    self.moriio_wrapper.write_remote_data_status(
                        [size], [local_offset], [remote_offset], session
                    )
                )
        except Exception:
            if not statuses:
                raise
            # Settled like any failed push, so the source blocks stay held
            # until the writes already posted can no longer read them.
            logger.exception(
                "transfer %s: could not post every write", request["transfer_id"]
            )
            statuses.append(_POST_FAILED)
        logger.debug(
            "RDMA write issued for transfer %s: %d blocks, %d layers, %d chunk "
            "groups, %d state tensors to %s",
            request["transfer_id"],
            len(dst),
            len(sessions),
            len(groups),
            len(state_writes),
            engine_id,
        )
        return statuses

    def _state_writes(
        self, request: dict[str, Any], engine_id: str
    ) -> list[tuple[Any, int, int, int]]:
        """``(session, size, local_offset, remote_offset)`` per state tensor.

        Empty for a model without per-request state. Raises when the request
        cannot carry the state, before anything has been written.
        """
        if not self._state_tensors:
            return []
        transfer_id = request["transfer_id"]
        src = int(request.get("src_slot_index", -1))
        dst = int(request.get("dst_slot_index", -1))
        remote_slots = self.remote_moriio_metadata[engine_id].num_state_slots
        if not (0 <= src < self._num_state_slots and 0 <= dst < remote_slots):
            raise ValueError(
                f"transfer {transfer_id}: state slot {src} -> {dst} is out of "
                f"range ({self._num_state_slots} -> {remote_slots} slots)"
            )
        decode_tp_size = int(request.get("decode_tp_size") or self.tp_size)
        if decode_tp_size != self.tp_size:
            # Each rank holds only its own heads' state, so rank i has to land
            # on rank i of a consumer sharded the same way.
            raise ValueError(
                f"transfer {transfer_id}: per-request state is sharded over TP "
                f"{self.tp_size}, the consumer runs TP {decode_tp_size}"
            )
        sessions = self._get_or_build_state_sessions(engine_id)
        if self._verify_kv:
            request["state_digest"] = self._state_digest(src)
        return [
            (session, slot_bytes, src * slot_bytes, dst * slot_bytes)
            for session, slot_bytes in sessions
        ]

    def _service_write_requests(self) -> None:
        """Start queued pushes and report the ones that have finished.

        Runs on the write service thread, or without it on the worker thread
        from ``get_finished``, which the engine keeps calling while finished
        prefills hold deferred blocks, so a request is served even when no
        batch is running.
        """
        while True:
            try:
                request = self._write_requests.get_nowait()
            except queue.Empty:
                break
            if self._is_valid_write_request(request):
                self._accept_write_request(request)
        self._forget_finished_writes()

        waiting: list[tuple[dict[str, Any], float]] = []
        now = time.time()
        for request, queued_at in self._pending_writes:
            deadline = request.get("deadline")
            if deadline is not None and now > float(deadline):
                logger.error(
                    "transfer %s: the consumer's push deadline passed before "
                    "the push could start",
                    request["transfer_id"],
                )
                self._finish_write(request, ok=False)
                continue
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
            staged = self._state_source_ready(request)
            if staged is None:
                if time.monotonic() - queued_at > self._handshake_timeout_s:
                    logger.error(
                        "transfer %s: its state never reached transfer slot %s",
                        request["transfer_id"],
                        request.get("src_slot_index"),
                    )
                    self._finish_write(request, ok=False)
                else:
                    waiting.append((request, queued_at))
                continue
            if not staged:
                self._finish_write(request, ok=False)
                continue
            if not self._source_ready(int(request["transfer_id"])):
                if time.monotonic() - queued_at > self._write_timeout_s:
                    logger.error(
                        "transfer %s: its prefill reported no finished forward "
                        "within %.0fs of the push request",
                        request["transfer_id"],
                        self._write_timeout_s,
                    )
                    self._finish_write(request, ok=False)
                else:
                    waiting.append((request, queued_at))
                continue
            try:
                statuses = self._write_blocks(request, self._write_engine_id(request))
            except Exception:
                logger.exception(
                    "transfer %s: could not push KV to the consumer",
                    request["transfer_id"],
                )
                self._forget_peer(self._write_engine_id(request))
                self._finish_write(request, ok=False)
                continue
            self._sending_writes[int(request["transfer_id"])] = (statuses, request)
        self._pending_writes = waiting

        for transfer_id, (statuses, request) in list(self._sending_writes.items()):
            # The consumer frees its blocks once told, so every write must
            # have settled first.
            if not all(status.Succeeded() or status.Failed() for status in statuses):
                continue
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
                self._forget_peer(self._write_engine_id(request))
                self._finish_write(request, ok=False)
            else:
                if self._verify_kv:
                    self._recheck_page_source(request)
                self._finish_write(request, ok=True)

    def _accept_write_request(self, request: dict[str, Any]) -> None:
        """Queue a push, unless it is a second ask for the same transfer."""
        transfer_id = int(request["transfer_id"])
        if transfer_id not in self._write_ids:
            self._write_ids[transfer_id] = None
            request["_accepted_at"] = time.monotonic()
            self._pending_writes.append((request, request["_accepted_at"]))
        elif self._write_ids[transfer_id] is None:
            logger.warning(
                "transfer %s: ignoring a second WRITE_REQ while the first is "
                "still being served",
                transfer_id,
            )
        else:
            logger.error(
                "transfer %s: WRITE_REQ for a push that already finished; its "
                "source blocks may belong to another request now",
                transfer_id,
            )
            self._finish_write(request, ok=False)

    def _forget_finished_writes(self) -> None:
        """Drop finished ids after the write timeout plus the hold; a second
        ask later than that is not caught."""
        cutoff = (
            time.monotonic() - self._write_timeout_s - envs.ATOM_PD_FAILED_WRITE_HOLD_S
        )
        while self._finished_writes and self._finished_writes[0][0] < cutoff:
            _, transfer_id = self._finished_writes.popleft()
            finished = self._write_ids.get(transfer_id)
            if finished is not None and finished < cutoff:
                del self._write_ids[transfer_id]

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
            float(request.get("deadline") or 0.0)
            int(request.get("src_slot_index", -1))
            int(request.get("dst_slot_index", -1))
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
        self._staged_sources.pop(transfer_id, None)
        stats = request.get("_page_stats")
        if self._transfer_stats and stats is not None:
            self._log_transfer_stats(
                transfer_id,
                "write",
                ok,
                stats,
                stats[0] - request.get("_accepted_at", stats[0]),
            )
        finished = time.monotonic()
        self._write_ids[transfer_id] = finished
        self._finished_writes.append((finished, transfer_id))
        # Released whatever happened: a consumer told the push failed has no
        # reason to come back for these blocks.
        with self._done_sending_lock:
            self.done_sending.add(transfer_id)
        port = int(request["decode_handshake_port"]) + get_port_offset(
            int(request.get("decode_dp_rank", 0)),
            self.tp_rank,
            int(request.get("decode_tp_size") or self.tp_size),
        )
        report = {"decode_req_id": request["decode_req_id"], "ok": ok}
        if ok and "state_digest" in request:
            report["state_digest"] = request["state_digest"]
        if ok and "page_digest" in request:
            report["page_digest"] = request["page_digest"]
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
        """Move producer reports, and pushes that never got one, into the sets.

        A push never reported fails a hold past its deadline: no push starts
        after the deadline, and one started before it has the hold to finish.
        """
        with self._write_reports_lock:
            reports, self._write_reports = self._write_reports, {}
        for req_id, ok in reports.items():
            if self._awaiting_write.pop(req_id, None) is None:
                logger.warning("Write report for req %s, which is not waiting", req_id)
                continue
            if ok:
                done_recving.add(req_id)
            else:
                self._hold_failure(req_id)

        hold_s = envs.ATOM_PD_FAILED_WRITE_HOLD_S
        now = time.time()
        for req_id, (deadline, meta) in list(self._awaiting_write.items()):
            if now > deadline + hold_s:
                logger.error(
                    "req %s: no report from the producer %.0fs past its push "
                    "deadline; failing it",
                    req_id,
                    hold_s,
                )
                del self._awaiting_write[req_id]
                failed_recving.add(req_id)
                # In case the WRITE_REQ never arrived and the producer still
                # holds the source blocks.
                self._notify_producer(req_id, meta)

    def _hold_failure(self, req_id: ReqId) -> None:
        """Hold a failed receive's blocks for ATOM_PD_FAILED_WRITE_HOLD_S.

        The scheduler frees the blocks as soon as the failure is reported, and
        MoRI fails a batch on its first failed work request while the rest of
        it may still be landing.
        """
        hold_s = envs.ATOM_PD_FAILED_WRITE_HOLD_S
        self._held_failures[req_id] = time.monotonic() + hold_s
        if hold_s > 0:
            logger.warning(
                "req %s failed; its blocks stay held %.0fs before the failure "
                "is reported",
                req_id,
                hold_s,
            )

    def _report_held_failures(self, failed_recving: set[ReqId]) -> None:
        """Move the failures whose hold has ended into ``failed_recving``."""
        now = time.monotonic()
        for req_id, until in list(self._held_failures.items()):
            if until <= now:
                del self._held_failures[req_id]
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
                    with self._done_sending_lock:
                        self.done_sending.add(req_id)
                    logger.debug(
                        "Handshake listener: consumer finished reading req %d", req_id
                    )

                # A bad write frame is logged and dropped rather than raised:
                # this thread also serves every other peer's handshake.
                elif msg == MoRIIOConstants.WRITE_REQ:
                    # Only queued; the write service owns the transfer engine.
                    try:
                        self._write_requests.put(msgpack.loads(parts[2]))
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_REQ")
                    self._write_wakeup.set()

                elif msg == MoRIIOConstants.WRITE_DONE:
                    try:
                        report = msgpack.loads(parts[2])
                        req_id, ok = report["decode_req_id"], bool(report["ok"])
                        state_digest = report.get("state_digest")
                        page_digest_ = report.get("page_digest")
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_DONE")
                        continue
                    with self._write_reports_lock:
                        self._write_reports[req_id] = ok
                        if state_digest is not None and self._verify_kv:
                            self._write_state_digests[req_id] = state_digest
                        if page_digest_ is not None and self._verify_kv:
                            self._write_page_digests[req_id] = page_digest_

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
        timeout_s = min(
            MoRIIOConstants.HANDSHAKE_RECV_TIMEOUT_S, self._handshake_timeout_s
        )

        with _zmq_ctx(zmq.DEALER, path) as sock:
            # Every handshake shares one thread, so a dead peer must not hold it.
            sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))

            def recv() -> list[bytes]:
                try:
                    return sock.recv_multipart()
                except zmq.Again:
                    raise TimeoutError(
                        f"no reply from {path} within {timeout_s:.0f}s"
                    ) from None

            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = recv()
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

            received_frame = recv()
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
        """Queue the request for its reads once its producer rank is handshaken.

        Requests for a rank whose handshake is already running share it. The
        outcome goes on ``_ready_requests`` for ``get_finished``.
        """
        logger.debug(
            "Initiating background handshake for req %s -> %s",
            req_id,
            remote_engine_id,
        )
        peer = self._engine_name_with_dp(
            remote_engine_id, int(meta.remote_dp_rank or 0)
        )
        future = self._read_handshakes.get(peer)
        if future is None or future.done():
            self._start_read_handshakes(remote_engine_id, meta)
            future = self._read_handshakes[peer]

        def _on_done(f: Future[set[str]], entry=(req_id, meta)):
            self._ready_requests.put((*entry, f.exception()))

        future.add_done_callback(_on_done)

    def _start_read_handshakes(self, remote_engine_id: EngineId, meta: ReqMeta) -> None:
        """Handshake with every producer DP rank not registered or in flight."""
        host = meta.remote_host
        port = int(meta.remote_handshake_port)
        tp_size = int(meta.tp_size)
        wanted = int(meta.remote_dp_rank or 0)

        # In dp(prefill)<->dp(decode) communication, all-to-all handshake is required.
        for cur_dp_rank in range(max(int(meta.remote_dp_size), wanted + 1)):
            dp_engine_id = self._engine_name_with_dp(remote_engine_id, cur_dp_rank)
            running = self._read_handshakes.get(dp_engine_id)
            if running is not None and not running.done():
                continue
            if cur_dp_rank != wanted:
                with self._handshake_lock:
                    if dp_engine_id in self._remote_agents:
                        continue
            future = self._handshake_executor.submit(
                self._execute_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
            )

            def _on_single_done(f: Future[set[str]], eid=dp_engine_id):
                with self._handshake_lock:
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            future.add_done_callback(_on_single_done)
            self._read_handshakes[dp_engine_id] = future

    def _pop_done_transfers(self) -> tuple[set[str], set[str]]:
        """Return the (done, failed) receive request IDs.

        ``_read_blocks`` issues one ``batch_read`` per (layer, chunk group) and
        each carries its own transfer uid, so they complete in any order. A
        request is settled only once none of them is still in flight, and done
        only if every one succeeded.
        """
        done_req_ids: set[str] = set()
        failed_req_ids: set[str] = set()
        failed_peers: set[str] = set()
        with self.moriio_wrapper.lock:
            to_remove = []
            for req_id, status_list in self._recving_transfers.items():
                if not all(s.Succeeded() or s.Failed() for s in status_list):
                    continue
                host, port, peer = self._recving_transfers_callback_addr[req_id]
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
                    failed_peers.add(peer)
                else:
                    done_req_ids.add(req_id)
                stats = self._page_reads.pop(req_id, None) if self.use_fabric else None
                if self._transfer_stats and stats is not None:
                    self._log_transfer_stats(req_id, "read", not failed, stats, 0.0)
                # Notify either way: the producer holds the source blocks until
                # it hears back, so a dropped notify leaks them for the run.
                # the Decode req_id(request_id) ,Prefill req_id(transfer_id)
                # so we need to use transfer_id to send notify
                self.moriio_wrapper.send_notify(
                    self.request_id_to_transfer_id[req_id], host, port
                )
                to_remove.append(req_id)
            for req_id in to_remove:
                del self._recving_transfers[req_id]
                del self._recving_transfers_callback_addr[req_id]

        for peer in failed_peers:
            self._forget_peer(peer)
        return done_req_ids, failed_req_ids

    def _forget_peer(self, engine_id: EngineId) -> None:
        """Drop a peer's cached handshake after a failed transfer to it.

        The next request handshakes again, so a peer that restarted on the
        same host and port is used with its new memory registrations.
        """
        with self._handshake_lock:
            if self._remote_agents.pop(engine_id, None) is None:
                return
            sessions = self._built_sessions.pop(engine_id, None)
            if sessions is not None:
                self._retired_sessions.append(sessions)
            region_sessions = self._region_sessions.pop(engine_id, None)
            if region_sessions is not None:
                self._retired_sessions.append(region_sessions)
            state_sessions = self._built_state_sessions.pop(engine_id, None)
            if state_sessions is not None:
                self._retired_sessions.append(state_sessions)
            self.remote_moriio_metadata.pop(engine_id, None)
            self.layer_name_to_remote_kv_cache_metadata.pop(engine_id, None)
        logger.warning(
            "Dropped the cached handshake with %s after a failed transfer",
            engine_id,
        )

    def get_finished(self) -> KVConnectorOutput:
        """Return the finished sending, receiving, and failed request IDs.

        Called by the worker each step via ``async_proc_aggregation``.
        """
        if self.is_producer:
            # Before done_sending is read below, so a push that finishes this
            # step releases its blocks this step.
            if self._write_service_thread is None:
                self._service_write_requests()
        else:
            if self._gated_writes:
                self._request_gated_writes()
            self._issue_ready_reads()
        done_recving, failed_reads = self._pop_done_transfers()
        for req_id in failed_reads:
            self._hold_failure(req_id)
        failed_recving: set[ReqId] = set()
        if self._awaiting_write or self._write_reports:
            self._collect_write_reports(done_recving, failed_recving)
        if self._held_failures:
            self._report_held_failures(failed_recving)
        if self._failed_before_read:
            failed_recving |= self._failed_before_read
            self._failed_before_read = set()
        if self._verify_kv:
            # Outside _pop_done_transfers so the digests stay off the path
            # that holds the wrapper lock.
            for req_id in done_recving:
                if self.use_fabric:
                    self._verify_page_landed(req_id)
                    continue
                self._verify_landed(req_id)
                self._verify_state_landed(req_id)
            for req_id in failed_recving:
                if self.use_fabric:
                    self._verify_page_dst.pop(req_id, None)
                    with self._write_reports_lock:
                        self._write_page_digests.pop(req_id, None)
                self._verify_rows.pop(req_id, None)
                self._verify_before.pop(req_id, None)
                self._verify_kept.pop(req_id, None)
                self._verify_state_before.pop(req_id, None)
                with self._write_reports_lock:
                    self._write_state_digests.pop(req_id, None)
        with self._done_sending_lock:
            done_sending, self.done_sending = self.done_sending, set()
        for transfer_id in done_sending:
            # Released without a push, e.g. the consumer failed before asking.
            self._staged_sources.pop(int(transfer_id), None)
        if not self.is_producer:
            if done_sending:
                logger.warning(
                    "Consumer received %d stale done_sending notifications "
                    "(single-machine port collision?) — discarding: %s",
                    len(done_sending),
                    done_sending,
                )
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
        # Tokens one block-table entry covers, the unit of the reused prefix.
        self.hash_block_size = (
            config.kv_cache_block_size * config.decode_context_parallel_size
        )

        # Pending receive requests: req_id -> (Sequence, block_table, state slot)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int], int]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Source-block ownership -- see `MooncakeConnectorScheduler` for why
        # the claim is taken in `request_finished` rather than at alloc.
        self._awaiting_send: set[str] = set()
        # Producer: state copies into transfer slots, until the next metadata.
        self._state_stages: dict[TransferId, tuple[int, int]] = {}

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    @property
    def receives_suffix_and_state(self) -> bool:
        """A consumer receives only the blocks past its local prefix hit, and
        the per-request state whole (`num_computed_blocks`)."""
        return not self.is_producer

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        """Check if this sequence needs remote KV prefill.

        Returns:
            ``(num_tokens, needs_async_load)`` where ``needs_async_load``
            is True if the scheduler should defer until KV transfer completes.
        """
        params = seq.kv_transfer_params or {}

        # Every time until `update_state_after_alloc` queues the receive: the
        # scheduler can hold a request back after a yes (a full running set).
        if params.get("do_remote_prefill"):
            return len(seq.prompt_token_ids), True

        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        """Build a metadata snapshot of pending receive requests.

        The returned object is passed to the worker-side connector
        for RDMA operations.  The internal pending queue is cleared.
        """
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id
        meta.state_stages, self._state_stages = self._state_stages, {}

        for req_id, (req, block_ids, slot) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # Ours; the producer's slot came in the params as `remote_slot_index`.
            req.kv_transfer_params["local_slot_index"] = slot
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
            self._reqs_need_recv[seq.id] = (
                seq,
                list(seq.block_table),
                getattr(seq, "state_slot", -1),
            )
            # The prefix this consumer already holds is not received again.
            params["num_computed_blocks"] = (
                seq.num_cached_tokens // self.hash_block_size
            )
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
        if self.is_producer and produced_no_handoff(seq):
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
            # The slot the push reads this request's state from, held with the
            # blocks until send_finished. -1 for a model without any.
            "remote_slot_index": getattr(seq, "state_slot", -1),
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

    def stage_state_for_send(self, seq: Sequence, src_slot: int, dst_slot: int):
        """Have the worker copy ``seq``'s state into its transfer slot.

        The scheduler already re-pointed ``seq`` at ``dst_slot``, which
        `request_finished` advertises; the push reads it once the copy landed.
        """
        if not self.is_producer:
            raise RuntimeError("only a producer stages state for a send")
        self._state_stages[seq.id] = (int(src_slot), int(dst_slot))

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))
        # An undispatched copy would land in a slot the release hands back.
        for key in [k for k in self._state_stages if str(k) == str(req_id)]:
            del self._state_stages[key]

    def source_blocks_released(self, seq: Sequence) -> None:
        """No block-lifetime state remains after the send claim is retired."""


@dataclass(frozen=True)
class _Region:
    """One registered transfer region, as both ends address it.

    ``kind`` is ``"page"`` (indexed by scheduler block) or ``"slot"`` (indexed
    by per-request state slot). ``chunk_bytes`` is a whole number of units, so
    a unit always sits inside a single registered chunk.
    """

    name: str
    kind: str
    base: int
    total: int
    unit: int
    reverse: bool
    chunk_bytes: int

    @classmethod
    def of(cls, name: str, kind: str, region: Any) -> _Region:
        unit = int(region.unit_bytes)
        total = int(region.total_bytes)
        if unit > MAX_RDMA_CHUNK_BYTES:
            raise ValueError(f"region {name}: a {unit} B unit exceeds one RDMA chunk")
        if total % unit:
            raise ValueError(f"region {name}: {total} B is not whole {unit} B units")
        return cls(
            name,
            kind,
            int(region.base_addr),
            total,
            unit,
            bool(region.reverse_indexed),
            (MAX_RDMA_CHUNK_BYTES // unit) * unit,
        )

    @classmethod
    def from_layout(cls, entry: list[Any]) -> _Region:
        name, kind, unit, total, reverse, chunk_bytes = entry
        return cls(name, kind, 0, total, unit, reverse, chunk_bytes)

    def layout(self) -> list[Any]:
        return [
            self.name,
            self.kind,
            self.unit,
            self.total,
            self.reverse,
            self.chunk_bytes,
        ]

    def shape(self) -> tuple[Any, ...]:
        """What the handshake compares. Pool totals may differ."""
        return (self.name, self.kind, self.unit, self.reverse, self.chunk_bytes)

    def chunks(self) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        offset = 0
        for size in KVTransferRegion.chunk_sizes(
            self.total, self.unit, self.chunk_bytes
        ):
            chunks.append((self.base + offset, size))
            offset += size
        return chunks

    def locate(self, index: int) -> tuple[int, int]:
        """``(chunk, offset within chunk)`` of unit ``index``."""
        if not 0 <= index < self.total // self.unit:
            raise ValueError(f"region {self.name}: unit {index} is out of range")
        offset = KVTransferRegion.unit_offset(
            index, self.total, self.unit, self.reverse
        )
        return offset // self.chunk_bytes, offset % self.chunk_bytes


class _PostFailed:
    """Stands in for the transfers a request could not post."""

    @staticmethod
    def Succeeded() -> bool:
        return False

    @staticmethod
    def Failed() -> bool:
        return True

    @staticmethod
    def Message() -> str:
        return "not every transfer could be posted"

    @staticmethod
    def Code() -> str:
        return "post-failed"


_POST_FAILED = _PostFailed()


def _zmq_ctx(socket_type: int, addr: str):
    """Context manager for a ZMQ socket with role-appropriate bind semantics.

    ROUTER sockets bind; DEALER/REQ sockets connect.
    """
    return zmq_socket_ctx(addr, socket_type, bind=(socket_type == zmq.ROUTER))
