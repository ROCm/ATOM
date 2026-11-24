# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses zero-copy transfers via the MoRIIO library for efficient KV cache
migration between producer (prefill) and consumer (decode) nodes: RDMA by
default, or the GPU scale-up fabric with ``moriio_backend`` ``fabric`` (see
``moriio/fabric.py``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
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
    check_page_table,
    desc_fabric_info,
    desc_size,
    is_fabric_backend,
    page_batch,
    page_table,
    shard_pairs,
    unit_batch,
)
from atom.kv_transfer.disaggregation.moriio.moriio_common import (
    _MORIIO_AVAILABLE,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    get_port_offset,
)
from atom.kv_transfer.disaggregation.moriio.moriio_engine import MoRIIOWrapper
from atom.kv_transfer.disaggregation.port_offset import side_channel_port_offset
from atom.kv_transfer.disaggregation.sharded_transfer import consumer_head_slices
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    EngineId,
    KVConnectorOutput,
    KVTransferRegion,
    PageUnitSpec,
    ReqId,
    ReqMeta,
    TransferId,
    pair_units,
    produced_no_handoff,
)
from atom.kv_transfer.disaggregation.utils import (
    MAX_RDMA_CHUNK_BYTES,
    chunk_tensor_for_rdma,
)
from atom.model_engine.sequence import Sequence
from atom.models.utils import get_pp_indices
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
        IOEngine,
        IOEngineConfig,
        PollCqMode,
        RdmaBackendConfig,
    )

logger = logging.getLogger("atom")

# PP producer, stage 0: released transfer ids remembered against a late repeat.
# Ids are request ids, never reused, and a repeat comes within seconds.
_RELEASED_TRANSFERS_KEPT = 65536


def _byte_digests(rows: list[Any]) -> list[int]:
    """Position-weighted byte digest of each tensor in ``rows``.

    Read as uint8 so the digest is dtype-agnostic, and stacked so all of them
    cost one device sync rather than one each. Weighted because a reused block
    still holds an earlier request's KV, and under a plain byte sum new KV with
    the same total looks like KV that never landed. Weights stay below 2**16,
    so even a full 32k-token prompt sums far inside int64.
    """
    import torch

    flat = [row.reshape(-1).view(torch.uint8) for row in rows]
    weights = (
        torch.arange(
            max(row.numel() for row in flat), dtype=torch.int64, device=flat[0].device
        )
        % 65521
        + 1
    )
    sums = [torch.sum(row.to(torch.int64) * weights[: row.numel()]) for row in flat]
    return [int(v) for v in torch.stack(sums).tolist()]


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

    With ``moriio_backend`` ``fabric`` the transfers go over the GPU scale-up
    fabric instead of RDMA, in write mode only (see ``moriio/fabric.py``).

    A PP producer (fabric only) is one connector per stage, each on a side
    channel of its own and holding only its stage's layers: a consumer asks
    every stage for its part, and a TP1 stage cuts each consumer rank its KV
    heads (`_page_plan`). Stage 0's scheduler owns the block table every stage
    reads, so it frees a request's blocks only once every consumer rank has
    heard from every stage and released it (`_record_release`).
    """

    use_fabric = False
    _transfer_stats = False
    # One PP stage unless `__init__` finds more: its rank and the target layers
    # it holds, `[first, end)`, end None for all past `first`.
    pp_size = 1
    pp_rank = 0
    _stage_layers: tuple[int, int | None] = (0, None)
    # ATOM_MORIIO_WRITE_THREAD (producer): the thread serving pushes, None
    # when get_finished serves them; how long it sleeps between passes while
    # a push waits or is in flight, and at most while idle.
    _push_thread: threading.Thread | None = None
    _PUSH_POLL_S = 1e-4
    _PUSH_IDLE_WAIT_S = 0.1
    # Finished prefills whose ready event aged out, remembered up to this many.
    _KV_READY_EXPIRED_MAX = 1 << 16

    def __init__(self, config: Config) -> None:
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size
        # Under PP every stage runs a connector of its own, on a side channel of
        # its own, holding only its stage's layers; a consumer asks each stage
        # for its part (`_request_write`).
        self.pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
        if self.pp_size > 1:
            self.pp_rank = int(config.parallel_config.pipeline_parallel_rank)
            hf_config = config.hf_config
            text_config = getattr(hf_config, "text_config", None) or hf_config
            first, end = get_pp_indices(
                text_config.num_hidden_layers, self.pp_rank, self.pp_size
            )
            # The last stage also holds the drafter's layers, numbered past
            # the target's, so its range is open (`pair_units`).
            last = self.pp_rank == self.pp_size - 1
            self._stage_layers = (first, None if last else end)

        kv_transfer_config = config.kv_transfer_config
        self.local_ip = get_ip()

        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.transfer_mode = self._transfer_mode_from(kv_transfer_config)
        self.use_fabric = self._fabric_backend_from(config)
        if self.pp_size > 1 and not self.use_fabric:
            raise ValueError(
                "MoRIIO moves a pipeline-parallel instance's KV with "
                "moriio_backend fabric only: each stage pushes its own layers"
            )
        if config.dcp_config.index_shard and not self.use_fabric:
            # The RDMA paths register and move whole regions (MiniMax-M3's
            # per-layer one moves no index cache at all).
            raise ValueError(
                "ATOM_M3_INDEX_SHARD=1 (dcp_config.index_shard) needs "
                "moriio_backend fabric, which moves a rank's share of the index "
                "cache; MoRIIO's RDMA paths move whole regions"
            )
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

        # Compute unique side-channel port for this (pp, dp, tp) rank
        handshake_port = self.base_handshake_port
        # tp_size must be passed: it strides the DP ranks apart, and the
        # default of 1 would put every DP rank on the same port set. A PP stage
        # strides past every (dp, tp) rank of the stages before it, so the
        # stages of one node listen apart; with one stage it is the (dp, tp)
        # offset.
        self.side_channel_port = handshake_port + side_channel_port_offset(
            self.dp_rank,
            self.tp_rank,
            self.tp_size,
            self.pp_rank,
            self.pp_size,
            self.dp_size,
        )
        self.engine_id = f"{self.local_ip}:{handshake_port}"
        if self.pp_size > 1:
            self.engine_id += f":pp{self.pp_rank}"

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
        if self.use_fabric:
            self._create_fabric_backend(kv_transfer_config)
        else:
            self._create_rdma_backend(kv_transfer_config)

        # PAGE layout (fabric): the one registered buffer's descriptor, its
        # size, and per PAGE region (offset in it, bytes per block, role, shard
        # rank, shard world: `fabric.page_table`).
        self._page_desc: bytes | None = None
        self._page_pool_bytes = 0
        self._page_regions: list[tuple[int, int, str, int, int]] = []
        # Per PAGE region, its unit's identity and head runs (`PageRegion`):
        # how a PP stage finds its regions among a whole-model consumer's, and
        # how a TP1 producer cuts each consumer rank its heads (`_page_plan`).
        self._page_units: list[tuple[PageUnitSpec | None, Any]] = []
        # Per (consumer rank's engine id, its TP size, its TP rank), the push
        # plan `_page_plan` made.
        self._page_plans: dict[tuple[EngineId, int, int], _PagePlan] = {}
        self._page_views: list[Any] | None = None
        self._transfer_stats = envs.ATOM_MORIIO_TRANSFER_STATS
        # ATOM_MORIIO_VERIFY on the consumer: each pushed request's destination
        # blocks, and the producer's digest of its source blocks.
        self._verify_page_dst: dict[ReqId, list[int]] = {}
        self._write_page_digests: dict[ReqId, list[Any]] = {}
        # Likewise for the leading blocks an incremental push leaves out (this
        # decode's own prefix-cache hit): held here, never pushed, and compared
        # with the producer's source blocks at the same positions.
        self._verify_held_dst: dict[ReqId, list[int]] = {}
        self._write_held_digests: dict[ReqId, Any] = {}

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
        self._write_state_digests: dict[ReqId, dict[str, int]] = {}
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
        self._sending_writes: dict[
            tuple[int, int, int], tuple[list[Any], dict[str, Any]]
        ] = {}
        # Pushes asked for, by (transfer, consumer DP rank, consumer TP rank):
        # None while queued or pushing, else when the push was reported
        # (monotonic). The source blocks go once every consumer rank that reads
        # this rank has had its push, so a second ask for the same push must not
        # push again.
        self._write_ids: dict[tuple[int, int, int], float | None] = {}
        self._finished_writes: deque[tuple[float, tuple[int, int, int]]] = deque()
        # Pushes finished per transfer, until each consumer rank has had one: a
        # rank at a smaller TP (a TP1 PP stage) serves several
        # (`_consumers_per_push`).
        self._pushes_done: dict[int, int] = {}
        # ATOM_MORIIO_WRITE_THREAD: set by the listener on each WRITE_REQ; the
        # push thread's stop flag; the exception that stopped it, raised on
        # the worker thread by the next get_finished.
        self._push_wakeup = threading.Event()
        self._push_stop = threading.Event()
        self._push_error: BaseException | None = None
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
        # A receive from a PP producer is one push per stage: how many reports
        # it waits for, and (stages reported so far, all succeeded) until it
        # has them.
        self._write_reports_expected: dict[ReqId, int] = {}
        self._write_report_counts: dict[ReqId, tuple[frozenset[int], bool]] = {}
        # ATOM_MORIIO_VERIFY: a PP receive's stage digests, merged by the region
        # index here until every stage has reported (`_merge_stage_digests`).
        self._stage_digests: dict[ReqId, dict[int, Any]] = {}
        self._stage_held_digests: dict[ReqId, dict[int, Any]] = {}
        self._write_timeout_s = envs.ATOM_MORIIO_WRITE_TIMEOUT_S

        # Failed receives, held until the given time before they are reported.
        self._held_failures: dict[ReqId, float] = {}

        # Completed send-side transfers, added by the listener thread and by
        # _finish_write, so swapped out under the lock.
        self.done_sending: set[int] = set()
        self._done_sending_lock = threading.Lock()
        # PP producer, stage 0: the consumer ranks that released each transfer
        # so far (`_record_release`), and the transfers already released, so a
        # late repeat cannot free blocks the scheduler has handed on.
        self._release_ranks: dict[TransferId, set[int]] = {}
        self._released_transfers: OrderedDict[TransferId, None] = OrderedDict()

        # Transfer ID mapping (worker side)
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # Region mode (backends that publish KVTransferTensors and no
        # per-layer block caches, e.g. DeepSeek-V4). None until registered.
        self._regions: list[_Region] | None = None
        self._region_sessions: dict[EngineId, dict[tuple[int, int, int], Any]] = {}

    @classmethod
    def _fabric_backend_from(cls, config: Any) -> bool:
        """Whether ``moriio_backend`` is ``fabric``; raises outside its scope.

        The fabric path only pushes: the read path posts its copies with no
        ordering against the consumer's queued forwards, so a fabric consumer
        in read mode is refused rather than run without that gate. Both ends
        are held to it, so one config names one mode. DCP and PCP, which split
        a request's blocks or tokens, are refused too. A PP producer pushes
        each stage's layers (`_page_plan`); a PP consumer is refused by
        `Config._check_pp_support`.
        """
        kv_transfer_config = config.kv_transfer_config or {}
        if not is_fabric_backend(kv_transfer_config):
            return False
        mode = cls._transfer_mode_from(kv_transfer_config)
        if mode != "write":
            raise ValueError(
                "moriio_backend fabric moves the KV by producer push only; set "
                "moriio_mode=write (or ATOM_MORIIO_MODE=write) on the prefill "
                f"and the decode, got {mode!r}"
            )
        for name in (
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        ):
            size = int(getattr(config, name, 1) or 1)
            if size != 1:
                raise ValueError(
                    f"moriio_backend fabric moves whole blocks per KV head; "
                    f"{name}={size} is not supported"
                )
        return True

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

        # Here, not at module level: a mori without the FABRIC backend still
        # serves RDMA.
        from mori.io import FabricBackendConfig

        num_streams = int(kv_transfer_config.get("fabric_num_streams", 64))
        num_events = int(kv_transfer_config.get("fabric_num_events", 64))
        # mori only warns when its fabric copy kernel fails to load, and without
        # it every copy of 4 MiB or more fails, so that warning is an error.
        # Recorded rather than raised: the filter is process-wide, and another
        # thread's RuntimeWarning in this window must not raise in that thread.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.moriio_wrapper.set_backend_type(
                BackendType.FABRIC, FabricBackendConfig(num_streams, num_events)
            )
        failed = None
        for warning in caught:
            from_mori = "mori" in str(warning.filename) or "fabric_copy" in str(
                warning.message
            )
            if (
                failed is None
                and issubclass(warning.category, RuntimeWarning)
                and from_mori
            ):
                failed = warning
            else:
                warnings.showwarning(
                    warning.message, warning.category, warning.filename, warning.lineno
                )
        if failed is not None:
            raise RuntimeError(f"moriio_backend fabric: {failed.message}")
        logger.info(
            "FabricBackendConfig: streams=%d, events=%d, transfer_mode=%s",
            num_streams,
            num_events,
            self.transfer_mode,
        )

    @classmethod
    def allocate_kv_pool(cls, config: Any, nbytes: int, device: Any) -> Any:
        """moriio_backend fabric keeps the KV pool in fabric-exportable memory."""
        if not cls._fabric_backend_from(config):
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

        With ``moriio_backend`` ``fabric`` the PAGE regions are registered
        instead, see :meth:`_register_pages`. Otherwise a backend with no
        per-layer block caches (DeepSeek-V4) publishes its storage only as
        ``transfer_tensors`` regions; those are registered instead, see
        :meth:`_register_regions`.
        """
        self.kv_caches = kv_caches
        if self.use_fabric:
            self._start_handshake_listener(
                self._register_pages(kv_caches, transfer_tensors, num_blocks)
            )
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
    ) -> MoRIIOAgentMetadata:
        """Register every PAGE region as one fabric buffer.

        The regions all live in the paged KV pool (`allocate_kv_pool`), which
        is registered once; each region is then addressed by its offset in the
        pool and its bytes per block, and the peer gets that table in the
        handshake. mori maps a peer's whole allocation per registered
        descriptor, so one descriptor keeps it to one mapping per peer. Every
        published PAGE region moves, the draft's and the indexer's included;
        `page_table` refuses one outside the pool or overlapping another.
        """
        pages = list(getattr(transfer_tensors, "pages", None) or ())
        if not pages:
            raise NotImplementedError(
                "moriio_backend fabric moves PAGE regions, and this model "
                "publishes none"
            )
        if any(getattr(c, "per_request_state", False) for c in kv_caches.values()) or (
            getattr(transfer_tensors, "swa_block_regions", None)
            or getattr(transfer_tensors, "slot_regions", None)
        ):
            raise NotImplementedError(
                "moriio_backend fabric moves PAGE regions only, and this model "
                "also keeps per-request state"
            )
        blocks = int(num_blocks or getattr(transfer_tensors, "num_blocks", 0))
        base, size, entries = page_table([page.region for page in pages], blocks)
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
                "FABRIC cannot move it: the device cannot export fabric memory "
                "or is not on a ready UALink vPOD"
            )
        self._page_desc = desc
        self._page_pool_bytes = size
        self._page_regions = entries
        self._page_units = [(page.unit_spec, page.head_major_runs) for page in pages]
        views = [page.view for page in pages]
        self._page_views = views if all(v is not None for v in views) else None
        if self._verify_kv and self._page_views is None:
            logger.warning("ATOM_MORIIO_VERIFY: PAGE regions without views; off")
            self._verify_kv = False
        self.num_blocks = blocks
        self.block_len = self.kv_cache_block_size
        logger.info(
            "MoRIIO fabric: registered %d PAGE regions (%d B per block, %d of "
            "them sharded) as one buffer of %.2f GiB at %#x, %d blocks, vPOD %d",
            len(entries),
            sum(entry[1] // entry[4] for entry in entries),
            sum(entry[4] > 1 for entry in entries),
            size / (1 << 30),
            base,
            self.num_blocks,
            vpod,
        )
        return MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name="aiter",
            page_desc=desc,
            page_regions=[list(entry) for entry in entries],
            page_units=(
                [
                    [*dataclasses.astuple(spec), None if runs is None else list(runs)]
                    for spec, runs in self._page_units
                ]
                if all(spec is not None for spec, _ in self._page_units)
                else None
            ),
        )

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
        # Whatever this producer's own moriio_mode: the consumer picks it.
        if self.is_producer and envs.ATOM_MORIIO_WRITE_THREAD:
            self._start_push_thread()

    def _start_push_thread(self) -> None:
        """ATOM_MORIIO_WRITE_THREAD: serve pushes on their own thread.

        From here on that thread is the only caller of
        `_service_write_requests` and so the only one to touch the push
        bookkeeping, post a push or report one; get_finished only drains
        ``done_sending``, which the thread fills once a push has settled or,
        when the consumers release the source (`_released_by_consumers`: PP,
        or one rank feeding several), the listener does on the last RELEASE.
        Either way the source blocks go back at a step after every push from
        them landed. Under PP each stage's connector runs a thread of its own.
        """
        import atexit

        import torch

        # mori launches the fabric copy on the calling thread's device.
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._push_thread = threading.Thread(
            target=self._push_loop,
            args=(device,),
            daemon=True,
            name="moriio-push",
        )
        self._push_thread.start()
        atexit.register(self.stop_push_thread)
        logger.info("MoRIIO: pushes are served on their own thread")

    def stop_push_thread(self, timeout: float = 5.0) -> None:
        """Stop the push thread after its current pass. Pushes still queued
        are left to the consumer's deadline and hold."""
        thread = self._push_thread
        if thread is None:
            return
        self._push_stop.set()
        self._push_wakeup.set()
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("MoRIIO: the push thread did not stop in %.0fs", timeout)

    def _push_loop(self, device: int | None) -> None:
        """Serve pushes as WRITE_REQs arrive and report each as it lands.

        Rather than at the engine step that next reaches get_finished: a
        WRITE_REQ waits up to a step to be taken up there, and a landed push
        up to another to be reported to the consumer. The KV-ready event still
        gates every push (`_source_ready`). An exception stops the thread and
        is raised on the worker thread by get_finished, as it would have been
        from the step.
        """
        if device is not None:
            import torch

            torch.cuda.set_device(device)
        try:
            while not self._push_stop.is_set():
                if not (self._pending_writes or self._sending_writes):
                    self._push_wakeup.wait(self._PUSH_IDLE_WAIT_S)
                # Cleared before the pass, so a WRITE_REQ queued during it is
                # served by the next pass rather than after the idle wait.
                self._push_wakeup.clear()
                self._service_write_requests()
                if self._pending_writes or self._sending_writes:
                    self._push_stop.wait(self._PUSH_POLL_S)
        except Exception as error:
            logger.exception("MoRIIO: the push thread failed")
            self._push_error = error

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

        The KDA forward binds its conv state as ``k_cache`` and its SSM state as
        ``v_cache``, each ``(num_slots, ...)`` and contiguous, so a slot is one
        run of ``stride(0)`` elements and moves as a single write.
        """
        if any(
            getattr(kv_cache, name, None) is not None
            for name in ("replay_buf_k", "replay_buf_u", "replay_buf_g")
        ):
            # Under ReplaySSM the records and the cursor are part of the state,
            # and this connector moves neither.
            raise NotImplementedError(
                f"{layer_name}: ReplaySSM record buffers are per-request state "
                "MoRIIO does not move; serve P/D with ATOM_ENABLE_REPLAYSSM=0"
            )
        for part, tensor in (("k", kv_cache.k_cache), ("v", kv_cache.v_cache)):
            if tensor is None or tensor.numel() == 0:
                continue
            name = f"{layer_name}.{part}"
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
            if int(getattr(meta, "remote_pp_size", 1) or 1) > 1:
                # One read covers one producer's layers; a PP producer pushes
                # each stage's in write mode.
                logger.error(
                    "req %s: the producer runs PP %s, which MoRIIO moves in write "
                    "mode only (moriio_mode 'write')",
                    req_id,
                    meta.remote_pp_size,
                )
                self._fail_before_transfer(req_id, meta)
                continue
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
        it hears back. Repeats are harmless.

        A PP producer hears it at stage 0, whose scheduler owns the block table
        every stage reads, as a RELEASE counted per consumer rank: only after
        every rank reading the request can stage 0 tell that no stage still
        reads its blocks (`_record_release`).
        """
        remote_tp_size = int(meta.tp_size or 0) or self.tp_size
        try:
            peer = self._peer_tp_rank(remote_tp_size)
        except ValueError:
            peer = self.tp_rank
        transfer_id = self.request_id_to_transfer_id.get(req_id, meta.transfer_id)
        stages = self._producer_stages(meta)
        try:
            port = self._producer_port(meta, 0, stages, peer, remote_tp_size)
            if not self._uses_release(meta):
                self.moriio_wrapper.send_notify(transfer_id, meta.remote_host, port)
                return
            release = {
                "transfer_id": int(transfer_id),
                "consumers": max(1, self.tp_size // remote_tp_size),
                "decode_tp_rank": self.tp_rank,
            }
            self.moriio_wrapper.send_message(
                [MoRIIOConstants.RELEASE, msgpack.dumps(release)],
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
    def _producer_stages(meta: ReqMeta) -> int:
        """How many PP stages push into a receive, one connector each."""
        return max(1, int(getattr(meta, "remote_pp_size", 1) or 1))

    def _uses_release(self, meta: ReqMeta) -> bool:
        """Whether the producer frees this receive's source on our RELEASE
        (`MoRIIOConnector._released_by_consumers`): under PP, or when its rank
        feeds several of ours."""
        remote_tp_size = int(meta.tp_size or 0) or self.tp_size
        return self._producer_stages(meta) > 1 or remote_tp_size != self.tp_size

    @staticmethod
    def _producer_port(
        meta: ReqMeta,
        stage: int,
        stages: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> int:
        """Side channel port of a producer rank: PP stage ``stage`` of
        ``stages``, TP rank ``remote_tp_rank`` of ``remote_tp_size``, as the
        producer lays its ports out (`side_channel_port_offset`). A PP
        producer runs on one node (``meta.remote_host``)."""
        return int(meta.remote_handshake_port) + side_channel_port_offset(
            int(meta.remote_dp_rank or 0),
            remote_tp_rank,
            remote_tp_size,
            stage,
            stages,
            int(getattr(meta, "remote_dp_size", 1) or 1),
        )

    def _peer_tp_rank(self, remote_tp_size: int) -> int:
        """The producer rank holding this rank's heads.

        This rank's own at equal TP. A producer at a smaller TP that divides
        this one (a TP1 PP stage) holds whole runs of heads, and the one whose
        contiguous block contains this rank's is ``tp_rank // (tp / remote)``.
        """
        if remote_tp_size == self.tp_size:
            return self.tp_rank
        if remote_tp_size > self.tp_size or self.tp_size % remote_tp_size:
            raise ValueError(
                f"a producer at TP {remote_tp_size} cannot feed a consumer at TP "
                f"{self.tp_size}: the producer's TP must equal or divide it"
            )
        return self.tp_rank // (self.tp_size // remote_tp_size)

    def _record_write_report(
        self,
        req_id: ReqId,
        ok: bool,
        state_digest: Any = None,
        page_digest: Any = None,
        stage: int = 0,
        held_digest: Any = None,
    ) -> None:
        """Count one producer stage's report on a push into this rank.

        A receive from a PP producer is one push per stage: it is reported
        once every stage has reported, and succeeds only if all did. A stage
        counts once, so a repeated report cannot complete a receive another
        stage is still writing. Under ATOM_MORIIO_VERIFY a single producer's
        digests are kept as sent; a PP producer's stages each cover their own
        layers, so theirs are merged (`_merge_stage_digests`) and kept once the
        last stage has reported. ``held_digest``, of the blocks an incremental
        push left out (`_verify_held_landed`), is kept the same way.
        """
        with self._write_reports_lock:
            expected = self._write_reports_expected.get(req_id, 1)
            stages, all_ok = self._write_report_counts.pop(req_id, (frozenset(), True))
            if stage in stages or not 0 <= stage < expected:
                logger.warning(
                    "req %s: a report from producer stage %d (of %d) after one "
                    "already counted, or from no stage asked; ignored",
                    req_id,
                    stage,
                    expected,
                )
                if stages:
                    self._write_report_counts[req_id] = (stages, all_ok)
                return
            stages, all_ok = stages | {stage}, all_ok and ok
            verify = self._verify_kv
            # Digests paired by unit name the regions they cover; every
            # stage's are merged. Positional ones cover the whole table.
            by_region = isinstance(page_digest, dict)
            if verify and by_region:
                self._merge_stage_digests(self._stage_digests, req_id, page_digest)
            if verify and isinstance(held_digest, dict):
                self._merge_stage_digests(self._stage_held_digests, req_id, held_digest)
            if len(stages) < expected:
                self._write_report_counts[req_id] = (stages, all_ok)
                return
            self._write_reports_expected.pop(req_id, None)
            self._write_reports[req_id] = all_ok
            merged = self._stage_digests.pop(req_id, None)
            merged_held = self._stage_held_digests.pop(req_id, None)
            if verify:
                if state_digest is not None:
                    self._write_state_digests[req_id] = state_digest
                for store, regions, digest in (
                    (self._write_page_digests, merged, page_digest),
                    (self._write_held_digests, merged_held, held_digest),
                ):
                    if regions is not None:
                        store[req_id] = {"regions": regions, "stages": expected}
                    elif digest is not None and not isinstance(digest, dict):
                        store[req_id] = digest

    @staticmethod
    def _merge_stage_digests(
        store: dict[ReqId, dict[int, Any]], req_id: ReqId, page_digest: Any
    ) -> None:
        """Fold one PP stage's PAGE digests into its receive's.

        The stages hold disjoint layers, so their regions never overlap: the
        digests merge by the region's index on this rank, which a stage sends
        with them (`_push_digests`). A stage that sends none cannot be placed,
        and its regions are reported as never digested.
        """
        merged = store.setdefault(req_id, {})
        if isinstance(page_digest, dict):
            merged.update(
                zip(
                    (int(r) for r in page_digest["regions"]),
                    zip(page_digest["body"], page_digest["tail"]),
                )
            )

    @staticmethod
    def _rejects_transfer_subset(
        req_id: ReqId, meta: ReqMeta, skips_computed: bool = False
    ) -> bool:
        """Whether the request asks for a subset this direction does not implement.

        Both paths pair blocks one to one. Write mode skips the blocks this
        decode already holds (``skips_computed``, `_request_write`); a read
        moves every block, which is safe to over-serve. A source stride is
        not, because the pairing would put blocks in the wrong place.
        """
        if getattr(meta, "num_computed_blocks", 0) and not skips_computed:
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

    def _request_gated_writes(self) -> None:
        """Ask for the pushes whose step's earlier forwards have finished.

        A request that cannot be asked for fails alone, before anything was
        sent for it: the rest of its step is still asked for, and the failure
        frees its blocks and lets the producer release its source.
        """
        while self._gated_writes and self._gated_writes[0][0].query():
            _, reqs = self._gated_writes.popleft()
            for req_id, meta in reqs:
                try:
                    self._request_write(req_id, meta)
                except Exception:
                    # On the worker's busy loop, like the read path.
                    logger.exception(
                        "req %s: could not ask the producer to push KV", req_id
                    )
                    self._fail_before_transfer(req_id, meta)

    def record_kv_cache_ready(self, req_ids: list[ReqId]) -> None:
        """Record when this prefill batch's KV writes are done on the GPU.

        On every producer, whatever its own ``moriio_mode``: the consumer
        picks the mode, every WRITE_REQ waits for this event
        (`_source_ready`), and a producer that never recorded it would let
        each push time out.
        """
        if not self.is_producer or not req_ids:
            return
        ids = []
        for req_id in req_ids:
            try:
                ids.append(int(req_id))
            except (TypeError, ValueError):
                continue
        if not ids:
            return
        import torch

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        now = time.monotonic()
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

    def _source_ready(self, transfer_id: int) -> bool:
        """Whether the prefill that wrote this transfer's source has finished."""
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
        if self._rejects_transfer_subset(req_id, meta, skips_computed=True):
            self._fail_before_transfer(req_id, meta)
            return
        dst = [int(b) for b in meta.local_block_ids]
        src = [int(b) for b in (meta.remote_block_ids or [])]
        try:
            self._source_covers(f"req {req_id}", dst, src)
        except ValueError as error:
            logger.error("%s", error)
            self._fail_before_transfer(req_id, meta)
            return
        # ATOM_MORIIO_INCREMENTAL_PUSH: the leading blocks are this decode's
        # own prefix-cache hit (`MoRIIOConnectorScheduler._local_prefix_blocks`),
        # shared blocks the push must not write; the producer pushes the rest,
        # still paired by position. At least the last block is pushed (a hit
        # never covers it), so every push and its report stay as they were.
        skip = max(0, min(int(meta.num_computed_blocks or 0), len(dst) - 1))
        held_src, held_dst = src[:skip], dst[:skip]
        src, dst = src[skip : len(dst)], dst[skip:]
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

        # A PP producer is one connector per stage, each on its own side
        # channel; each pushes its stage's layers. A producer at a smaller TP
        # (a TP1 stage) cuts this rank its heads (`_page_plan`).
        stages = self._producer_stages(meta)
        try:
            remote_tp_rank = self._peer_tp_rank(remote_tp_size)
        except ValueError as error:
            logger.error("req %s: %s", req_id, error)
            self._fail_before_transfer(req_id, meta)
            return

        if self._verify_kv and self.use_fabric:
            self._verify_page_dst[req_id] = dst
            if held_dst:
                self._verify_held_dst[req_id] = held_dst
        elif self._verify_kv:
            first_layer = next(iter(self.layer_name_to_local_kv_cache_metadata))
            is_mla = self.kv_caches[first_layer].v_cache is None
            rows = self._verify_rows_for(dst, is_mla)
            self._verify_rows[req_id] = rows
            self._verify_before[req_id] = self._verify_digest(rows)
            if self._state_tensors:
                self._verify_state_before[req_id] = (
                    dst_slot,
                    self._state_digest(dst_slot),
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
            "decode_tp_rank": self.tp_rank,
            "dst_block_ids": dst,
            "src_block_ids": src,
            # Leading blocks left out of both lists; for the producer's stats.
            "skipped_blocks": skip,
            "deadline": deadline,
            "src_slot_index": src_slot,
            "dst_slot_index": dst_slot,
        }
        if req_id in self._verify_held_dst:
            # ATOM_MORIIO_VERIFY: the producer digests the held blocks' source
            # too (`_held_source`).
            request["verify_held_src_block_ids"] = held_src
        if stages > 1:
            with self._write_reports_lock:
                self._write_reports_expected[req_id] = stages
        unasked = []
        for stage in range(stages):
            port = self._producer_port(
                meta, stage, stages, remote_tp_rank, remote_tp_size
            )
            try:
                self.moriio_wrapper.send_message(
                    [MoRIIOConstants.WRITE_REQ, msgpack.dumps(request)],
                    meta.remote_host,
                    port,
                )
            except Exception:
                logger.exception(
                    "req %s: could not ask producer stage %d at %s:%d to push KV",
                    req_id,
                    stage,
                    meta.remote_host,
                    port,
                )
                unasked.append(stage)
        if len(unasked) < stages:
            # The stages asked will push into our blocks, so wait for them; a
            # stage not asked counts as a failed report.
            for stage in unasked:
                self._record_write_report(req_id, False, stage=stage)
        else:
            # Same reasoning as the read path: this runs on the busy_loop.
            with self._write_reports_lock:
                self._write_reports_expected.pop(req_id, None)
                self._write_report_counts.pop(req_id, None)
            self._verify_page_dst.pop(req_id, None)
            self._verify_held_dst.pop(req_id, None)
            self._verify_rows.pop(req_id, None)
            self._verify_before.pop(req_id, None)
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
                raise ValueError(
                    f"{engine_id} registered {len(theirs)} per-request state "
                    f"tensors to our {len(ours)}, or with other slot sizes"
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
        self, engine_id: str, positional: bool = True
    ) -> tuple[Any, list[tuple[int, int]], int]:
        """The (local pool, remote pool) FABRIC session, the peer's regions as
        ``(offset in its pool, bytes per block)``, and its block count.

        ``positional``: the peer's table is paired with ours by position, after
        checking that both ends published the same regions in the same order.
        Otherwise a `_PagePlan` pairs them by unit. Either way each peer region
        must lie inside the peer's registered pool.
        """
        remote = self.remote_moriio_metadata[engine_id]
        if remote.page_desc is None or remote.page_regions is None:
            raise ValueError(
                f"{engine_id} advertised no PAGE layout; both ends need "
                "moriio_backend fabric"
            )
        # A peer from before region shards advertises (offset, unit, role).
        theirs = [
            (int(o), int(u), str(r), *(int(x) for x in (shard or (0, 1))))
            for o, u, r, *shard in remote.page_regions
        ]
        if engine_id not in self._built_sessions:
            if remote.page_desc == self._page_desc:
                raise ValueError(
                    f"{engine_id} advertised our own PAGE descriptor; every "
                    "transfer would copy this rank onto itself"
                )
            ours = [entry[1:] for entry in self._page_regions]
            if not positional and any(
                entry[4] > 1 for entry in [*self._page_regions, *theirs]
            ):
                # `_page_plan` pairs units across PP stages and TP sizes, and
                # moves whole units: it has no notion of a rank's share.
                raise ValueError(
                    f"{engine_id}: a sharded index cache (ATOM_M3_INDEX_SHARD=1) "
                    "moves only between one-stage ends of the same TP; run P "
                    "and D both without it"
                )
            if positional and ours != [entry[1:] for entry in theirs]:
                first = next(
                    (i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b[1:]),
                    min(len(ours), len(theirs)),
                )
                if first < min(len(ours), len(theirs)) and (
                    ours[first][:2] == theirs[first][1:3]
                ):
                    # Same region, other shard: one end keeps the whole index
                    # cache and the other a rank's share of it.
                    raise ValueError(
                        f"{engine_id} holds PAGE region {ours[first][1]!r} as "
                        f"shard {theirs[first][3:]} (rank, of) where this rank "
                        f"holds {ours[first][2:]}: run P and D both with or "
                        "both without ATOM_M3_INDEX_SHARD=1, at one TP"
                    )
                raise ValueError(
                    f"{engine_id} published {len(theirs)} PAGE regions to our "
                    f"{len(ours)}; first difference at region {first}"
                )
            check_page_table(
                theirs,
                int(remote.num_blocks),
                desc_size(remote.page_desc),
                f"{engine_id}'s",
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
        return (
            self._built_sessions[engine_id][0],
            [(entry[0], entry[1]) for entry in theirs],
            int(remote.num_blocks),
        )

    def _write_pages(
        self, request: dict[str, Any], engine_id: str, src: list[int], dst: list[int]
    ) -> list[Any]:
        """Push every PAGE region's source blocks into the consumer's, as one batch.

        Paired by position when both ends hold the same regions (one stage, the
        same TP); otherwise by unit (`_page_plan`): a PP stage writes the
        consumer's regions of its own layers, and a TP1 stage cuts each
        consumer rank its KV heads. One batch per consumer rank, and the
        stages' batches write disjoint regions.
        """
        decode_tp_size = int(request.get("decode_tp_size") or self.tp_size)
        local_regions = [entry[:2] for entry in self._page_regions]
        plan = None
        if self.pp_size > 1 or decode_tp_size != self.tp_size:
            plan = self._page_plan(
                engine_id, decode_tp_size, self._consumer_tp_rank(request)
            )
        session, remote_regions, remote_blocks = self._get_or_build_page_session(
            engine_id, positional=plan is None
        )
        if plan is None:
            # One batch: the whole regions' blocks, then each shard's own ones.
            l_offs, r_offs, sizes = [], [], []
            for (rank, world), positions in self._page_shard_groups().items():
                pairs = (src, dst) if world == 1 else shard_pairs(src, dst, rank, world)
                for into, part in zip(
                    (l_offs, r_offs, sizes),
                    page_batch(
                        [local_regions[i] for i in positions],
                        [remote_regions[i] for i in positions],
                        *pairs,
                        self.num_blocks // world,
                        remote_blocks // world,
                    ),
                ):
                    into.extend(part)
        else:
            l_offs, r_offs, sizes = unit_batch(
                local_regions,
                remote_regions,
                plan.targets,
                plan.cuts,
                src,
                dst,
                self.num_blocks,
                remote_blocks,
            )
        if self._verify_kv:
            request["_page_src_digest"] = self._page_digests(src)
            request["_page_src"] = list(src)
            request["page_digest"] = (
                request["_page_src_digest"]
                if plan is None
                else self._plan_digests(plan, src)
            )
            held = self._held_source(request)
            if held:
                request["held_digest"] = (
                    self._page_digests(held)
                    if plan is None
                    else self._plan_digests(plan, held)
                )
        request["_page_stats"] = (time.monotonic(), sum(sizes), len(dst), len(sizes))
        return [
            self.moriio_wrapper.write_remote_data_status(sizes, l_offs, r_offs, session)
        ]

    def _held_source(self, request: dict[str, Any]) -> list[int]:
        """ATOM_MORIIO_VERIFY: the source of the leading blocks the consumer
        holds and an incremental push leaves out. They are the head of this
        request's prompt KV, claimed with the rest until the push is released
        (`request_finished`), so resident at push time."""
        held = request.get("verify_held_src_block_ids")
        if not isinstance(held, list) or not held:
            return []
        held = [int(b) for b in held]
        if not all(0 <= b < self.num_blocks for b in held):
            logger.error(
                "[moriio-verify] transfer %s: held source blocks %s outside "
                "this rank's %d; not digested",
                request["transfer_id"],
                held[:8],
                self.num_blocks,
            )
            return []
        return held

    def _page_plan(
        self, engine_id: EngineId, consumer_tp: int, consumer_rank: int
    ) -> _PagePlan:
        """Where this rank's PAGE regions land on consumer rank ``consumer_rank``
        of ``consumer_tp``, and which bytes of each source block go there.

        Paired by unit identity (`pair_units`): every consumer unit of this
        stage's layers must be held here, the drafter's included on the last
        stage. At equal TP each unit moves whole; a TP1 producer cuts each
        consumer rank its KV heads (`consumer_head_slices`) and copies the
        replicated units (the index keys) whole to every rank. Anything else
        is refused before a byte moves.
        """
        key = (engine_id, consumer_tp, consumer_rank)
        plan = self._page_plans.get(key)
        if plan is not None:
            return plan
        remote = self.remote_moriio_metadata[engine_id]
        units = remote.page_units
        if units is None or len(units) != len(remote.page_regions or ()):
            raise ValueError(
                f"{engine_id} names no unit for each of its PAGE regions; a "
                "PP or TP1 producer pairs its regions with the consumer's by "
                "unit, so both ends need this ATOM"
            )
        if any(spec is None for spec, _ in self._page_units):
            raise ValueError(
                "this rank's PAGE regions name no units, so they cannot be "
                f"paired with {engine_id}'s across PP stages or TP sizes"
            )
        if consumer_tp != self.tp_size and self.tp_size != 1:
            raise ValueError(
                f"a TP{self.tp_size} producer cannot feed a TP{consumer_tp} "
                "consumer: KV heads are cut between TP sizes only from TP1"
            )
        specs = [PageUnitSpec(*unit[:5]) for unit in units]
        runs = [None if unit[5] is None else tuple(unit[5]) for unit in units]
        widths = [int(entry[1]) for entry in remote.page_regions]
        first, end = self._stage_layers
        targets = pair_units([spec for spec, _ in self._page_units], specs, first, end)
        cuts = []
        for i, c in enumerate(targets):
            src_runs, (_, src_bytes, role) = (
                self._page_units[i][1],
                self._page_regions[i][:3],
            )
            if consumer_tp == self.tp_size:
                if (src_runs, src_bytes) != (runs[c], widths[c]):
                    raise ValueError(
                        f"{specs[c]}: {src_runs} runs in {src_bytes} B here, "
                        f"{runs[c]} in {widths[c]} B on consumer rank "
                        f"{consumer_rank}"
                    )
                cuts.append(np.array([[0, 0, src_bytes]], dtype=np.int64))
                continue
            cuts.append(
                consumer_head_slices(
                    src_bytes,
                    widths[c],
                    src_runs,
                    runs[c],
                    consumer_rank,
                    consumer_tp,
                    f"{role} ({specs[c]})",
                )
            )
        plan = _PagePlan(tuple(targets), tuple(cuts))
        self._page_plans[key] = plan
        logger.info(
            "MoRIIO fabric: %d PAGE regions of PP stage %d (layers %d..%s) land "
            "in %d of %s's %d, TP%d -> rank %d of TP%d",
            len(targets),
            self.pp_rank,
            first,
            "end" if end is None else end - 1,
            len(set(targets)),
            engine_id,
            len(specs),
            self.tp_size,
            consumer_rank,
            consumer_tp,
        )
        return plan

    def _plan_digests(self, plan: _PagePlan, block_ids: list[int]) -> dict:
        """ATOM_MORIIO_VERIFY: the digests of exactly the bytes a plan sends,
        laid out as they land, keyed by the consumer's region index, so the
        consumer can merge every stage's and compare each region it holds."""
        import torch

        device = self._page_views[0].device
        body = torch.as_tensor(block_ids[:-1], dtype=torch.int64, device=device)
        tail = torch.as_tensor(block_ids[-1:], dtype=torch.int64, device=device)
        rows = []
        for ids in (body, tail):
            for view, cut in zip(self._page_views, plan.cuts):
                # flatten, not reshape(n, -1): an incremental push of only the last block
                # leaves the body empty, and reshape(0, -1) is ambiguous.
                blocks = view.index_select(0, ids).flatten(1)
                rows.append(
                    torch.cat([blocks[:, a : a + n] for a, _, n in cut.tolist()], 1)
                )
        sums = _byte_digests(rows)
        n = len(self._page_views)
        return {"regions": list(plan.targets), "body": sums[:n], "tail": sums[n:]}

    def _log_transfer_stats(
        self, transfer_id: int, ok: bool, request: dict[str, Any]
    ) -> None:
        """ATOM_MORIIO_TRANSFER_STATS: one line per fabric push.

        Three spans, all on this rank's monotonic clock: ``queue_ms`` from the
        WRITE_REQ's arrival on the listener to the step that took it up,
        ``wait_ms`` from there to posting the batch (the handshake and the
        prefill's KV-ready event), ``write_ms`` from posting to the step that
        saw it settle. ``skipped``: the decode's own prefix-cache blocks left
        out of the push (ATOM_MORIIO_INCREMENTAL_PUSH). Pushes are served from
        ``get_finished``, once per engine step, so the first and the last
        include up to a step each; with ATOM_MORIIO_WRITE_THREAD, by a thread
        that polls every ``_PUSH_POLL_S``.
        """
        posted, nbytes, blocks, segments = request["_page_stats"]
        accepted = request.get("_accepted_at", posted)
        received = request.get("_received_at", accepted)
        write_ms = (time.monotonic() - posted) * 1e3
        logger.info(
            "MoRIIO transfer stats: transfer=%s protocol=fabric-write ok=%s "
            "blocks=%d skipped=%d bytes=%d queue_ms=%.3f wait_ms=%.3f "
            "write_ms=%.3f GB_per_s=%.1f segments=%d tp=%d",
            transfer_id,
            ok,
            blocks,
            int(request.get("skipped_blocks") or 0),
            nbytes,
            (accepted - received) * 1e3,
            (posted - accepted) * 1e3,
            write_ms,
            nbytes / max(write_ms, 1e-6) / 1e6,
            segments,
            self.tp_rank,
        )

    def _page_shard_groups(self) -> dict[tuple[int, int], list[int]]:
        """The PAGE regions by ``(shard rank, shard world)``, in table order."""
        groups: dict[tuple[int, int], list[int]] = {}
        for i, entry in enumerate(self._page_regions):
            groups.setdefault((entry[3], entry[4]), []).append(i)
        return groups

    def _page_digests(self, block_ids: list[int]) -> list[list[int]]:
        """Per PAGE region, the digest of ``block_ids`` but the last, and of
        the last (a sharded region's: of those it holds). Past the prompt the
        last block holds slots that both ends write after the push (the
        producer's deferred-output decode, the consumer's own first decode),
        so it is compared on its own."""
        import torch

        device = self._page_views[0].device

        def units(blocks):
            """Per shard, the units of ``blocks`` its regions hold."""
            return {
                (rank, world): torch.as_tensor(
                    [b // world for b in blocks if b % world == rank],
                    dtype=torch.int64,
                    device=device,
                )
                for rank, world in self._page_shard_groups()
            }

        body, tail = units(block_ids[:-1]), units(block_ids[-1:])
        pairs = list(zip(self._page_views, self._page_regions))
        sums = _byte_digests(
            [view.index_select(0, body[entry[3:]]) for view, entry in pairs]
            + [view.index_select(0, tail[entry[3:]]) for view, entry in pairs]
        )
        n = len(self._page_views)
        return [sums[:n], sums[n:]]

    def _changed_regions(self, before: list[int], after: list[int]) -> str:
        """The regions whose digests differ, as a count and their roles."""
        changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
        roles = [self._page_regions[i][2] for i in changed[:8]]
        return f"{len(changed)} of {len(before)} regions, first {roles}"

    def _recheck_page_source(self, request: dict[str, Any]) -> None:
        """Digest the source again once the push has landed: a change before
        the last block means something wrote the prompt's KV during the push."""
        src = request.get("_page_src")
        if not src:
            return
        before = request["_page_src_digest"]
        after = self._page_digests(src)
        if after[0] != before[0]:
            logger.error(
                "[moriio-verify] transfer %s: the prompt's source blocks changed "
                "during the push (%s)",
                request["transfer_id"],
                self._changed_regions(before[0], after[0]),
            )

    def _verify_held_landed(self, req_id: ReqId) -> None:
        """Compare the leading blocks this decode holds (its own prefix-cache
        hit, left out of an incremental push) with the producer's source
        blocks at the same positions: decode attends over them as the prompt's.
        All are full prompt blocks, so the last of them counts like the rest.
        """
        held = self._verify_held_dst.pop(req_id, None)
        with self._write_reports_lock:
            source = self._write_held_digests.pop(req_id, None)
        if held is None:
            return
        if source is None:
            logger.error(
                "[moriio-verify] req %s: the producer sent no digest of the %d "
                "locally held prefix blocks",
                req_id,
                len(held),
            )
            return
        if isinstance(source, dict):
            regions, stages = source["regions"], source.get("stages", 1)
        else:
            regions, stages = dict(enumerate(zip(*source))), 1
        local = self._page_digests(held)
        missing = [i for i in range(len(self._page_regions)) if i not in regions]
        differ = [
            i for i, sums in regions.items() if (local[0][i], local[1][i]) != sums
        ]
        roles = [self._page_regions[i][2] for i in (missing or differ)[:8]]
        if missing:
            logger.error(
                "[moriio-verify] req %s: %d of %d PAGE regions of the %d locally "
                "held prefix blocks were digested by none of the %d producer "
                "stages (first %s)",
                req_id,
                len(missing),
                len(self._page_regions),
                len(held),
                stages,
                roles,
            )
        elif differ:
            logger.error(
                "[moriio-verify] req %s: %d of %d regions of the %d locally held "
                "prefix blocks differ from the producer's (first %s) -- decode "
                "will attend over a prefix-cache hit that is not the prompt's",
                req_id,
                len(differ),
                len(regions),
                len(held),
                roles,
            )
        else:
            logger.info(
                "[moriio-verify] req %s: all %d PAGE regions of the %d locally "
                "held prefix blocks match the producer's %d stage(s)",
                req_id,
                len(self._page_regions),
                len(held),
                stages,
            )

    def _verify_page_landed(self, req_id: ReqId) -> None:
        """Compare the destination blocks with the producer's source digest,
        and the blocks an incremental push left out (`_verify_held_landed`)."""
        self._verify_held_landed(req_id)
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
        if isinstance(source, dict):
            self._verify_regions_landed(req_id, len(dst), source, landed)
            return
        if landed[0] != source[0]:
            import torch

            # Read again once the device is idle: a match then means the push
            # landed but was not yet visible here, not that it moved the
            # wrong bytes.
            torch.cuda.synchronize()
            time.sleep(0.05)
            again = self._page_digests(dst)[0] == source[0]
            logger.error(
                "[moriio-verify] req %s: %d blocks differ from the producer's "
                "before the last block (%s; 50 ms later %s) -- decode will "
                "attend over KV that is not the prompt's",
                req_id,
                len(dst),
                self._changed_regions(source[0], landed[0]),
                "they match" if again else "still different",
            )
        elif landed[1] != source[1]:
            logger.warning(
                "[moriio-verify] req %s: only the last of %d blocks differs "
                "(%s): the producer wrote its slots past the prompt during "
                "the push, or the push lost the prompt's tail",
                req_id,
                len(dst),
                self._changed_regions(source[1], landed[1]),
            )
        else:
            logger.info(
                "[moriio-verify] req %s: all %d PAGE regions of %d blocks match "
                "the producer",
                req_id,
                len(self._page_regions),
                len(dst),
            )

    def _verify_regions_landed(
        self, req_id: ReqId, blocks: int, source: dict, landed: list[list[int]]
    ) -> None:
        """`_verify_page_landed` for pushes paired by unit: every PP stage sent
        the digests of the regions it wrote, merged by region here, so a region
        no stage wrote is an error too (layers decode would read unwritten)."""
        regions: dict[int, tuple[int, int]] = source["regions"]
        missing = [i for i in range(len(self._page_regions)) if i not in regions]
        body = [i for i, (b, _) in regions.items() if landed[0][i] != b]
        tail = [i for i, (_, t) in regions.items() if landed[1][i] != t]

        def roles(ids: list[int]) -> list[str]:
            return [self._page_regions[i][2] for i in ids[:8]]

        if missing:
            logger.error(
                "[moriio-verify] req %s: %d of %d PAGE regions were written by "
                "none of the %d producer stages (first %s) -- decode will "
                "attend over KV that never arrived",
                req_id,
                len(missing),
                len(self._page_regions),
                source.get("stages", 1),
                roles(missing),
            )
        if body:
            logger.error(
                "[moriio-verify] req %s: %d of %d regions differ from the "
                "producer's before the last of %d blocks (first %s) -- decode "
                "will attend over KV that is not the prompt's",
                req_id,
                len(body),
                len(regions),
                blocks,
                roles(body),
            )
        elif tail:
            logger.warning(
                "[moriio-verify] req %s: only the last of %d blocks differs in "
                "%d regions (first %s): the producer wrote its slots past the "
                "prompt during the push, or the push lost the prompt's tail",
                req_id,
                blocks,
                len(tail),
                roles(tail),
            )
        if not (missing or body or tail):
            logger.info(
                "[moriio-verify] req %s: all %d PAGE regions of %d blocks match "
                "the producer's %d stage(s)",
                req_id,
                len(self._page_regions),
                blocks,
                source.get("stages", 1),
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
        names = list(self.layer_name_to_local_kv_cache_metadata)
        sums = _byte_digests(
            [self.kv_caches[name].k_cache.index_select(0, rows) for name in names]
        )
        return dict(zip(names, sums))

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

    def _state_digest(self, slot: int) -> dict[str, int]:
        """Position-weighted byte digest of one slot in every state tensor.

        Weighted like `_verify_digest`, so a slot still holding an earlier
        request's state with the same byte sum reads as different.
        """
        rows = [tensor[slot] for _, _, tensor in self._state_tensors.values()]
        return dict(zip(self._state_tensors, _byte_digests(rows)))

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

        Runs on the worker thread from ``get_finished``, which the engine keeps
        calling while finished prefills hold deferred blocks, so a request is
        served even when no batch is running; with ATOM_MORIIO_WRITE_THREAD,
        on the push thread only (`_push_loop`).
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
            self._sending_writes[self._push_key(request)] = (statuses, request)
        self._pending_writes = waiting

        for (transfer_id, *_), (statuses, request) in list(
            self._sending_writes.items()
        ):
            # The consumer frees its blocks once told, so every write must
            # have settled first.
            if not all(status.Succeeded() or status.Failed() for status in statuses):
                continue
            failed = [status for status in statuses if status.Failed()]
            if failed:
                logger.error(
                    "%s write failed for transfer %s: %d of %d transfers "
                    "failed, first: %s (code %s)",
                    "FABRIC" if self.use_fabric else "RDMA",
                    transfer_id,
                    len(failed),
                    len(statuses),
                    failed[0].Message(),
                    failed[0].Code(),
                )
                self._forget_peer(self._write_engine_id(request))
                self._finish_write(request, ok=False)
            else:
                if self._verify_kv and self.use_fabric:
                    self._recheck_page_source(request)
                self._finish_write(request, ok=True)

    def _accept_write_request(self, request: dict[str, Any]) -> None:
        """Queue a push, unless it is a second ask for the same push."""
        key = self._push_key(request)
        transfer_id = key[0]
        if key not in self._write_ids:
            self._write_ids[key] = None
            request["_accepted_at"] = time.monotonic()
            self._pending_writes.append((request, request["_accepted_at"]))
        elif self._write_ids[key] is None:
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
            _, key = self._finished_writes.popleft()
            finished = self._write_ids.get(key)
            if finished is not None and finished < cutoff:
                del self._write_ids[key]
                # No push of this transfer starts past the deadline, so a count
                # some consumer rank never completed (it never asked) goes too.
                self._pushes_done.pop(key[0], None)

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
            int(request.get("decode_tp_rank", 0))
            float(request.get("deadline") or 0.0)
            int(request.get("src_slot_index", -1))
            int(request.get("dst_slot_index", -1))
            int(request.get("skipped_blocks") or 0)
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
        """The dp-suffixed id the consumer rank's sessions are cached under; a
        producer rank that feeds several consumer ranks tells them apart by
        their TP rank."""
        name = f"{request['decode_host']}:{request['decode_handshake_port']}"
        if "decode_tp_rank" in request:
            name += f":tp{int(request['decode_tp_rank'])}"
        return cls._engine_name_with_dp(name, int(request.get("decode_dp_rank", 0)))

    def _consumer_tp_rank(self, request: dict[str, Any]) -> int:
        """The consumer rank a push goes to; this rank's own at equal TP, from
        a consumer that predates naming it."""
        return int(request.get("decode_tp_rank", self.tp_rank))

    def _push_key(self, request: dict[str, Any]) -> tuple[int, int, int]:
        """One push: a transfer into one consumer rank."""
        return (
            int(request["transfer_id"]),
            int(request.get("decode_dp_rank", 0)),
            self._consumer_tp_rank(request),
        )

    def _consumers_per_push(self, request: dict[str, Any]) -> int:
        """How many consumer ranks read this rank's copy of a transfer: one at
        equal TP, the TP ratio when the consumer shards the heads further."""
        decode_tp_size = int(request.get("decode_tp_size") or self.tp_size)
        return max(1, decode_tp_size // self.tp_size)

    def _released_by_consumers(self, request: dict[str, Any]) -> bool:
        """Whether this transfer's source blocks wait for the consumers'
        RELEASE (`_record_release`) rather than for this rank's own pushes.

        Under PP stage 0's block table is what every stage reads, and stage 0
        finishing its pushes says nothing of the others'. A rank feeding several
        consumer ranks would otherwise free its source when one of them failed
        before asking. Either way each consumer rank releases once every push
        into it has been reported (`MoRIIOConnector._uses_release`).
        """
        return self.pp_size > 1 or self._consumers_per_push(request) > 1

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
                    self._consumer_tp_rank(request),
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
        key = self._push_key(request)
        self._sending_writes.pop(key, None)
        if self._transfer_stats and "_page_stats" in request:
            self._log_transfer_stats(transfer_id, ok, request)
        # A repeat of a push already reported is only answered: its source may
        # belong to another request by now.
        if self._write_ids.get(key) is None:
            finished = time.monotonic()
            self._write_ids[key] = finished
            self._finished_writes.append((finished, key))
            done = self._pushes_done.get(transfer_id, 0) + 1
            if done < self._consumers_per_push(request):
                # Another consumer rank still reads this rank's source.
                self._pushes_done[transfer_id] = done
            else:
                self._pushes_done.pop(transfer_id, None)
                # A second ask for this transfer is refused by `_write_ids`
                # from here on, so its ready entry has no reader left; one
                # arriving after `_forget_finished_writes` must not find its
                # source ready either.
                with self._kv_ready_lock:
                    self._kv_ready_events.pop(transfer_id, None)
                    self._kv_ready_expired.pop(transfer_id, None)
                # Released whatever happened: a consumer told the push failed
                # has no reason to come back for these blocks. Unless the
                # consumers release them (`_released_by_consumers`).
                if not self._released_by_consumers(request):
                    with self._done_sending_lock:
                        self.done_sending.add(transfer_id)
        port = int(request["decode_handshake_port"]) + get_port_offset(
            int(request.get("decode_dp_rank", 0)),
            self._consumer_tp_rank(request),
            int(request.get("decode_tp_size") or self.tp_size),
        )
        report = {"decode_req_id": request["decode_req_id"], "ok": ok}
        if self.pp_size > 1:
            report["pp_rank"] = self.pp_rank
        if ok and "state_digest" in request:
            report["state_digest"] = request["state_digest"]
        if ok and "page_digest" in request:
            report["page_digest"] = request["page_digest"]
        if ok and "held_digest" in request:
            report["held_digest"] = request["held_digest"]
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

    def _record_release(
        self, transfer_id: TransferId, consumers: int, rank: int
    ) -> None:
        """Count a consumer rank's release of a transfer (`_released_by_consumers`).

        A consumer rank releases a request once every push into it, from every
        stage, has been reported, so the transfer's blocks are free once each
        of the ``consumers`` ranks reading this rank has. A rank counts once,
        and a transfer is released once: a late repeat must not free blocks
        the scheduler has handed on.
        """
        with self._done_sending_lock:
            if transfer_id in self._released_transfers:
                return
            ranks = self._release_ranks.setdefault(transfer_id, set())
            ranks.add(rank)
            if len(ranks) < max(1, consumers):
                return
            del self._release_ranks[transfer_id]
            self._released_transfers[transfer_id] = None
            while len(self._released_transfers) > _RELEASED_TRANSFERS_KEPT:
                self._released_transfers.popitem(last=False)
            self.done_sending.add(transfer_id)

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
            awaited = self._awaiting_write.pop(req_id, None)
            if awaited is None:
                logger.warning("Write report for req %s, which is not waiting", req_id)
                continue
            if ok:
                done_recving.add(req_id)
            else:
                self._hold_failure(req_id)
            _, meta = awaited
            if self._uses_release(meta):
                # Every stage has reported, so none reads the request's blocks.
                self._notify_producer(req_id, meta)

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
                with self._write_reports_lock:
                    self._write_reports_expected.pop(req_id, None)
                    self._write_report_counts.pop(req_id, None)
                    self._stage_digests.pop(req_id, None)
                    self._stage_held_digests.pop(req_id, None)
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

        Handles five message types:
        - ``GET_META_MSG``: Responds with engine + per-layer KV cache metadata.
        - ``POP_DONE_RECV``: Records that the consumer finished reading the request.
        - ``WRITE_REQ`` (producer): queues a consumer's request for a push.
        - ``WRITE_DONE`` (consumer): records the producer's report on a push.
        - ``RELEASE`` (PP producer, stage 0): a consumer rank has every stage's
          report on a request.
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
                    # Only queued; the worker thread owns the RDMA engine.
                    try:
                        request = msgpack.loads(parts[2])
                        if isinstance(request, dict):
                            # For ATOM_MORIIO_TRANSFER_STATS' queue_ms.
                            request["_received_at"] = time.monotonic()
                        self._write_requests.put(request)
                        self._push_wakeup.set()
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_REQ")

                elif msg == MoRIIOConstants.WRITE_DONE:
                    try:
                        report = msgpack.loads(parts[2])
                        req_id, ok = report["decode_req_id"], bool(report["ok"])
                        state_digest = report.get("state_digest")
                        page_digest = report.get("page_digest")
                        held_digest = report.get("held_digest")
                        stage = int(report.get("pp_rank", 0))
                    except Exception:
                        logger.exception("Dropping undecodable WRITE_DONE")
                        continue
                    self._record_write_report(
                        req_id,
                        ok,
                        state_digest,
                        page_digest,
                        stage=stage,
                        held_digest=held_digest,
                    )

                elif msg == MoRIIOConstants.RELEASE:
                    try:
                        release = msgpack.loads(parts[2])
                        transfer_id = int(release["transfer_id"])
                        consumers = int(release["consumers"])
                        rank = int(release["decode_tp_rank"])
                    except Exception:
                        logger.exception("Dropping undecodable RELEASE")
                        continue
                    self._record_release(transfer_id, consumers, rank)

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
        remote_tp_rank: int | None = None,
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
            remote_dp_rank,
            self.tp_rank if remote_tp_rank is None else remote_tp_rank,
            remote_tp_size or self.tp_size,
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
            for key in [k for k in self._page_plans if k[0] == engine_id]:
                del self._page_plans[key]
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
            if self._push_thread is None:
                self._service_write_requests()
            elif self._push_error is not None:
                raise RuntimeError(
                    "MoRIIO: the push thread failed"
                ) from self._push_error
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
                self._verify_page_dst.pop(req_id, None)
                self._verify_held_dst.pop(req_id, None)
                with self._write_reports_lock:
                    self._write_page_digests.pop(req_id, None)
                    self._write_held_digests.pop(req_id, None)
                self._verify_rows.pop(req_id, None)
                self._verify_before.pop(req_id, None)
                self._verify_state_before.pop(req_id, None)
                with self._write_reports_lock:
                    self._write_state_digests.pop(req_id, None)
        with self._done_sending_lock:
            done_sending, self.done_sending = self.done_sending, set()
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

    pp_size = 1

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
        self.pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
        self.dp_size = config.parallel_config.data_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        # Tokens per block-table entry: `seq.num_cached_tokens` counts global
        # tokens, and under DCP an entry spans dcp_size physical blocks.
        self.hash_block_size = (
            config.kv_cache_block_size * config.decode_context_parallel_size
        )
        self.host_ip = get_ip()
        # A producer's pushes are served from the workers' `get_finished`, so
        # a downstream PP stage polls it at the idle cadence rather than every
        # 100 ms (`PPEngineCoreProc._downstream_busy_loop`); with
        # ATOM_MORIIO_WRITE_THREAD each stage's push thread serves them, and
        # under PP a stage's source is freed by stage 0's RELEASE, not its poll.
        self.pushes_from_worker_poll = (
            self.is_producer and not envs.ATOM_MORIIO_WRITE_THREAD
        )

        # Pending receive requests: req_id -> (Sequence, block_table, state slot)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int], int]] = {}
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
            params["do_remote_prefill"] = False
            params["num_computed_blocks"] = self._local_prefix_blocks(seq)
            logger.debug(
                "Queued req %s for remote KV loading (%d blocks, %d held here)",
                seq.id,
                len(seq.block_table),
                params["num_computed_blocks"],
            )

    def _local_prefix_blocks(self, seq: Sequence) -> int:
        """ATOM_MORIIO_INCREMENTAL_PUSH: the leading block-table entries whose
        KV this decode already holds -- its own prefix-cache hit, which
        `allocate` just shared into the table -- so the producer need not push
        them (`MoRIIOConnector._request_write`). As Mooncake's
        `num_computed_blocks`.

        0 with the flag off, and for a model with per-request state: the
        producer's state covers the whole prompt and a block-only delta does
        not carry it.
        """
        if not envs.ATOM_MORIIO_INCREMENTAL_PUSH or seq.has_per_req_cache:
            return 0
        return int(seq.num_cached_tokens) // self.hash_block_size

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
            # Each stage of a PP prefill pushes its own layers from a side
            # channel of its own (`MoRIIOConnector._producer_port`).
            "remote_pp_size": self.pp_size,
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


@dataclass(frozen=True)
class _PagePlan:
    """One producer rank's push into one consumer rank, paired by unit: local
    PAGE region ``i`` lands in the consumer's region ``targets[i]``, and
    ``cuts[i]`` is its ``[src offset, dst offset, bytes]`` rows of a block."""

    targets: tuple[int, ...]
    cuts: tuple[np.ndarray, ...]


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
