# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import hashlib
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import queue
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock, Thread, get_ident, local

import zmq
import zmq.asyncio
from zmq.utils.monitor import recv_monitor_message

from atom.config import Config
from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_CMD,
    DISCARD_WEIGHT_SYNC_CMD,
    FINISH_WEIGHT_SYNC_CMD,
    RpcResponseRouter,
    RpcResult,
    checked_timeout,
    engine_budget,
)
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.engine_utility import (
    FIRE_AND_FORGET_UTILITY_CMDS,
    WEIGHT_UPDATE_UTILITY_CMDS,
    completes_weight_sync,
)
from atom.model_engine.ipc.buffer_types import HostIdentity
from atom.model_engine.ipc.dto import CompletionResult
from atom.model_engine.ipc.manager_transport import ManagerTransport
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.observability import tracing
from atom.utils import (
    envs,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
)

logger = logging.getLogger("atom")
_WEIGHT_SYNC_FINISH_TIMEOUT_S = 30.0
# Only the sync that owns them sends these. A FINISH unfences every engine,
# and a DISCARD aborts whichever sync is between its buckets.
_WEIGHT_SYNC_OWNED_CMDS = WEIGHT_UPDATE_UTILITY_CMDS | {
    FINISH_WEIGHT_SYNC_CMD,
    DISCARD_WEIGHT_SYNC_CMD,
}

# Valid values for Config.dp_load_balance / --dp-load-balance, and the default.
# Single source of truth for argparse (choices + default) so the CLI flag and
# the Config field can never diverge.
DP_LB_STRATEGIES = ("round_robin", "least_requests", "least_tokens")
DP_LB_DEFAULT = "least_requests"

# Engine sockets for a multi-node run cannot use IPC paths, and the two ends
# must derive identical TCP ports without negotiating. Both compute this plan
# from the shared master port.
INTERNODE_DP_SOCKET_PORT_OFFSET = 100
INTERNODE_DP_PORTS_PER_ENGINE = 3


@dataclass(frozen=True)
class InternodeDPSocketPlan:
    """The three TCP ports one engine's sockets use, in a multi-node run.

    Single-node runs never build one of these: their engines are local, so
    CoreManager hands them IPC paths from ``get_open_zmq_ipc_path()`` instead.
    """

    rank: int
    input_port: int
    output_port: int
    control_port: int


def build_internode_dp_socket_plan(
    *, engine_count: int, master_port: int
) -> list[InternodeDPSocketPlan]:
    """Derive deterministic per-engine TCP ports from the DP master port.

    Multi-node only -- ``CoreManager.__init__`` calls this when
    ``is_multinode_dp`` and passes ``None`` otherwise, and that ``None`` is what
    selects the IPC path (and, with it, the connect/bind polarity) further down.

    Three ports per engine: requests, outputs, and control. Control is separate
    so the request socket keeps a single writer thread (see _send_request).
    """
    base = master_port + INTERNODE_DP_SOCKET_PORT_OFFSET
    return [
        InternodeDPSocketPlan(
            rank=rank,
            input_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE,
            output_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE + 1,
            control_port=base + rank * INTERNODE_DP_PORTS_PER_ENGINE + 2,
        )
        for rank in range(engine_count)
    ]


def iter_dp_rank_assignments(config) -> list[tuple[int, int]]:
    """The (global_dp_rank, local_dp_rank) pairs this node owns.

    The global rank identifies the rank within the whole DP group; the local
    rank indexes this node's GPUs. They coincide on a single node and diverge
    on every node after the first.

    Under DP-attention each TP rank becomes its own engine, so both the count
    and the global offset scale by tp_size.
    """
    pc = config.parallel_config
    tp_size = config.tensor_parallel_size
    local_dp_size = pc.data_parallel_size_local
    rank_offset = pc.data_parallel_rank if pc.is_multinode_dp else 0

    if config.enable_dp_attention:
        return [
            (
                (rank_offset + local_base) * tp_size + tp_rank,
                local_base * tp_size + tp_rank,
            )
            for local_base in range(local_dp_size)
            for tp_rank in range(tp_size)
        ]
    return [
        (rank_offset + local_rank, local_rank) for local_rank in range(local_dp_size)
    ]


def _resolve_dp_engine_count(config: Config, logical: int) -> int:
    """How many DP-attention EngineCores to launch for a `logical`-wide run.

    Normally `logical`, one per device. Under `--fake-eplb` on a box with fewer
    visible devices, launch only that many and record the width to keep sharding
    experts for, so each device owns the slice it would own in the full
    deployment; `FusedMoE` repeats the gathered tokens by the same ratio to
    match its token volume too.

    Gated on fake_eplb like `Config.tp_world_size`, and for the same reason.
    """
    import torch

    if not config.fake_eplb:
        return logical
    visible = torch.cuda.device_count()
    if not 0 < visible < logical:
        return logical
    if logical % visible:
        raise ValueError(
            f"Simulated DP-attention: {logical} ranks do not divide evenly over "
            f"{visible} visible device(s). Make the visible device count a "
            f"divisor of tp x dp, e.g. via HIP_VISIBLE_DEVICES."
        )
    config.dp_logical_size = logical
    logger.warning(
        "Simulated DP-attention: running %d of a %d-rank deployment. Experts "
        "shard %d ways and each rank repeats the gathered tokens %dx, so "
        "per-device MoE shapes match -- but collectives only cover the ranks "
        "that exist, so THE MODEL OUTPUT IS MEANINGLESS. Benchmarking only.",
        visible,
        logical,
        logical,
        logical // visible,
    )
    return visible


class CoreManager:
    # Why requests through this manager cannot be traced, if they cannot.
    _trace_unsupported: str | None = None

    def _init_shared_state(
        self,
        config: Config,
        *,
        label: str,
        local_engine_count: int,
        global_engine_count: int | None = None,
    ) -> None:
        """Every field the inherited methods touch, before any engine is spawned.

        Subclasses spawn their engines differently and so cannot run this
        class's ``__init__`` -- but they inherit its output threads, ``close()``
        and DP-load bookkeeping, all of which read the fields set here. This is
        the one place to add another such field.

        It exists because the alternative was tried: ``DisaggCoreManager`` used
        to hand-copy this block, and the copy drifted. ``_flush_stream_batch_fn``
        was added to the copy and to the API server that assigns it, but not to
        this class -- so the offline entrypoint, which is the one path that
        neither initialises nor assigns it, had its output thread die on the
        first streamed token and hung until CI timed out an hour later.

        ``local_engine_count`` is the engines spawned on THIS node;
        ``global_engine_count`` is the engines this manager routes to, which on
        a coordinator spans other nodes too. They are equal on a single node.
        Routing state sizes to the global count -- a short array would
        IndexError the moment the balancer picked a remote rank.
        """
        self.label = label
        self._config = config
        self._ipc = ManagerTransport(
            config, enable_shm=getattr(self, "is_coordinator", True)
        )
        self._startup_deadline = time.monotonic() + config.ipc_startup_timeout
        self._submission_lock = RLock()
        self._request_writer_id = None
        self._request_targets = {}
        self._committed_targets = {}
        self._submission_ids = {}
        self._cancelled = set()
        self._callback_only = set()
        self._peer_failures = {}
        self._closed = False  # Track whether already closed
        self.local_engine_count = local_engine_count
        self.global_engine_count = (
            local_engine_count if global_engine_count is None else global_engine_count
        )
        self.ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue[list[Sequence]]()
        # Utility replies are matched to their caller by request id, not by
        # position on a shared queue, where a late or unasked-for reply became
        # the next caller's answer.
        self._rpc_router = RpcResponseRouter()
        # Held across every bucket plus FINISH/abort. RLock lets the sender's
        # outer session call the same synchronous command method per bucket.
        self._weight_sync_lock = RLock()
        self._weight_sync_context = local()
        # Identity-only marker: unlike request_id, a caller cannot forge it by
        # choosing a string. Never put it on the wire.
        self._weight_sync_dispatch_token = object()
        # Runner processes behind each engine -- EngineCore's own sizing -- so
        # an engine that never answers still fails one result per rank. Read
        # with defaults: managers are also built from bare stand-in configs.
        self._rpc_ranks_per_engine = getattr(config, "tp_world_size", 1) * getattr(
            config, "prefill_context_parallel_size", 1
        )
        self._seq_id_to_callback = {}
        # Batched stream-flush hook, resolved lazily by the API server (avoids
        # an api_server <-> engine_core_mgr import cycle). Stays None on every
        # path that never streams, which the output thread checks for.
        self._flush_stream_batch_fn = None
        # Longest prompt the KV pool can hold, reported by each rank's READY.
        # None until the first one arrives, and thereafter the smallest across
        # ranks: a DP rank sizes its pool from its own free memory, so a prompt
        # is only safe to admit if it fits wherever the router sends it.
        self.max_pool_tokens: int | None = None
        self.engine_core_processes = []
        self._local_process_by_rank = {}
        self._distributed_stores = []
        self.input_sockets = []
        self.output_sockets = []
        self.engine_core_identities = []
        self.shutdown_paths = []
        self.output_threads = []
        # Fair-rotation cursor, advanced once per selection. round_robin picks the
        # rank directly (cursor % n); the load-aware strategies use it only to seed
        # the argmin start offset so fully-tied ranks rotate instead of always
        # resolving to rank 0.
        self._rank_rotation_cursor = 0

        # --- DP request load balancing (see _select_dp_rank_locked) ---
        # A subclass may fan out through its own add_request() and never charge
        # load at all, but the inherited output thread still calls
        # _release_seq_load() on every finished sequence, so these MUST exist.
        # Strategy: "round_robin" | "least_requests" | "least_tokens" (validated
        # at the CLI by argparse choices=DP_LB_STRATEGIES).
        self._dp_lb_strategy = config.dp_load_balance
        # Token-equivalent weight of one in-flight request for "least_tokens".
        # Read once here: this is a construction-time config value (CoreManager
        # is built after env/args are finalized), not a runtime-tunable knob.
        self._dp_lb_req_equiv = envs.ATOM_DP_LB_REQ_EQUIV
        self._dp_session_affinity_enabled = envs.ATOM_DP_SESSION_AFFINITY
        # Session id -> rank whose local prefix cache owns the session. Owners
        # are immutable for the lifetime of the process: moving one turn to a
        # light rank discards the dominant optimization in agentic workloads,
        # namely reuse of the accumulated conversation prefix.
        self._dp_session_owners: dict[str, int] = {}
        # Last prompt length observed for each sticky session.  A later turn on
        # the same owner normally reuses the old prompt, so only its positive
        # growth is new prefill debt.  Keeping this small scalar per session
        # avoids charging the complete cached conversation on every turn.
        self._dp_session_prompt_tokens: dict[str, int] = {}
        self._dp_route_counters = {
            "affinity_new_total": 0,
            "affinity_owner_hit_total": 0,
            # Kept as an explicit invariant/metric: strict affinity must leave
            # this at zero. It makes an accidental reintroduction of spill
            # visible in benchmark artifacts.
            "affinity_spill_total": 0,
            "affinity_parent_ignored_total": 0,
            "explicit_total": 0,
            "load_balanced_total": 0,
        }
        self._rank_routed_total = [0] * self.global_engine_count
        # Authoritative local load per rank. Request count is in-flight until
        # finish/abort; token count is queued/in-flight PREFILL work and is
        # released as soon as the first model output proves prefill completed.
        # Guarded by _lb_lock because dispatch runs on the request thread while
        # completion/release runs on the per-rank output threads.
        self._rank_reqs = [0] * self.global_engine_count
        self._rank_tokens = [0] * self.global_engine_count
        # seq_id -> (dp_rank, req_cost, tok_cost) so release subtracts exactly
        # what dispatch added, and only for ranks that were actually charged.
        self._seq_load = {}
        self._lb_lock = Lock()
        # Control traffic (utility commands, abort, shutdown) travels on its own
        # sockets so that input_sockets keeps a single writer -- see
        # _send_request. These have several writer threads and so do need
        # serializing, but none of them runs per request.
        self.control_sockets = []
        self.control_identities = []
        self._control_send_lock = Lock()
        # dp_rank -> newest metrics snapshot, refreshed by the output threads
        # from EngineCore's own periodic push. Read directly by the exporter, so
        # scraping costs no round trip and cannot time out.
        self.latest_metrics: dict[int, dict] = {}

    def _start_distributed_store(
        self, config: Config, *, multinode: bool = False
    ) -> None:
        """Bind the actual rendezvous server before publishing its port to workers."""
        from torch.distributed import TCPStore

        pc = config.parallel_config
        if multinode and pc.data_parallel_base_port == 0:
            raise ValueError(
                "Multi-node DP requires the same nonzero --data-parallel-base-port "
                "(or ATOM_DP_BASE_PORT) on every node."
            )
        if not multinode or pc.data_parallel_rank == 0:
            store = TCPStore(
                pc.data_parallel_master_ip,
                pc.data_parallel_base_port,
                is_master=True,
                wait_for_workers=False,
            )
            self._distributed_stores.append(store)
            pc.data_parallel_base_port = store.port
            logger.info(
                "%s: model-runner TCPStore listening on %s:%d",
                self.label,
                pc.data_parallel_master_ip,
                store.port,
            )
        pc._managed_distributed_store = True

    def __init__(self, config: Config):
        pp_size = config.pipeline_parallel_size
        self.pp_size = pp_size
        pc = config.parallel_config
        multinode = pc.is_multinode_dp
        if multinode and pp_size > 1:
            raise ValueError(
                "Multi-node data parallelism combined with pipeline "
                "parallelism (pipeline_parallel_size > 1) is not supported: "
                "the engine index space folds PP stages into DP ranks, and the "
                "two mappings have not been reconciled across nodes."
            )

        rank_assignments = iter_dp_rank_assignments(config)
        if config.enable_dp_attention:
            assert pp_size == 1, "Pipeline parallel + DP-attention is not supported yet"
            # One engine per assignment this node owns; on a single node that is
            # the whole tp x dp grid. --fake-eplb may shrink it to the visible
            # device count, taking the first N assignments.
            local_engine_count = _resolve_dp_engine_count(config, len(rank_assignments))
            logger.info(
                f"Enable dp attention, using {local_engine_count} data parallel ranks"
            )
            # Under DP-attention every TP rank becomes its own DP rank.
            config.parallel_config.data_parallel_size *= config.tensor_parallel_size
            config.parallel_config.data_parallel_size_local = local_engine_count
            if multinode:
                config.parallel_config.data_parallel_rank *= config.tensor_parallel_size
            config.tensor_parallel_size = 1
        else:
            dp_size = config.parallel_config.data_parallel_size
            assert not (
                pp_size > 1 and dp_size > 1
            ), "Pipeline parallel combined with data parallel is not supported yet."
            local_engine_count = len(rank_assignments) * pp_size

        global_engine_count = (
            config.parallel_config.data_parallel_size * pp_size
            if not config.enable_dp_attention
            else config.parallel_config.data_parallel_size
        )
        # Global DP rank 0's node owns the router; the others only host engines.
        self.is_coordinator = not multinode or pc.data_parallel_rank == 0
        socket_plan = (
            build_internode_dp_socket_plan(
                engine_count=global_engine_count,
                master_port=pc.data_parallel_master_port,
            )
            if multinode
            else None
        )

        # Inter-stage ZMQ channels (head<->downstream metadata, last->head
        # tokens), shared across the single dp group. PP+DP would need per-group
        # sets — deferred with the assertion above. Not shared state: only this
        # class's spawn loop reads them.
        self.pp_meta_addrs = []
        self.pp_token_addr = ""
        self.pp_kv_status_addr = ""
        if pp_size > 1:
            self.pp_meta_addrs = [get_open_zmq_ipc_path() for _ in range(pp_size)]
            self.pp_token_addr = get_open_zmq_ipc_path()
            self.pp_kv_status_addr = get_open_zmq_ipc_path()

        self._init_shared_state(
            config,
            label="Engine Core Mgr",
            local_engine_count=local_engine_count,
            global_engine_count=global_engine_count,
        )

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        processes_info = []
        local_dp_ranks = []

        try:
            self._start_distributed_store(config, multinode=multinode)
            for engine_index in range(self.local_engine_count):
                assignment_index = engine_index // self.pp_size
                dp_rank, local_dp_rank = rank_assignments[assignment_index]
                pp_rank = engine_index % self.pp_size
                logger.info(
                    f"{self.label}: Creating EngineCore engine {engine_index}"
                    f" (global dp={dp_rank}, local dp={local_dp_rank}, "
                    f"pp={pp_rank}) of {self.local_engine_count}"
                )

                # Create config for this (dp, pp) stage
                import copy

                rank_config = copy.deepcopy(config)
                rank_config.parallel_config.data_parallel_rank = dp_rank
                rank_config.parallel_config.data_parallel_rank_local = local_dp_rank
                rank_config.parallel_config.pipeline_parallel_rank = pp_rank
                if self.pp_size > 1:
                    rank_config.parallel_config.pp_meta_addrs = self.pp_meta_addrs
                    rank_config.parallel_config.pp_token_addr = self.pp_token_addr
                    rank_config.parallel_config.pp_kv_status_addr = (
                        self.pp_kv_status_addr
                    )

                if socket_plan is not None:
                    plan = socket_plan[dp_rank]
                    ip = config.parallel_config.data_parallel_master_ip
                    engine_addresses = {
                        "input_address": f"tcp://{ip}:{plan.input_port}",
                        "output_address": f"tcp://{ip}:{plan.output_port}",
                        "control_address": f"tcp://{ip}:{plan.control_port}",
                    }
                else:
                    engine_addresses = None

                engine_core_process, addresses, local_dp_rank = launch_engine_core(
                    rank_config, dp_rank, local_dp_rank, addresses=engine_addresses
                )

                processes_info.append(
                    {
                        "process": engine_core_process,
                        "addresses": addresses,
                        "dp_rank": dp_rank,
                        "config": rank_config,
                    }
                )
                local_dp_ranks.append(local_dp_rank)

            try:
                # No visible-device mask is published here. Device placement is
                # owned by ModelRunner._setup_device_and_distributed, which
                # selects an ABSOLUTE cuda:{local_dp_rank*tp_size+rank}. Masking
                # the child as well would renumber its devices and compound the
                # two offsets -- see set_device_control_env_var's docstring.
                for info, local_dp_rank in zip(processes_info, local_dp_ranks):
                    dp_rank = info["dp_rank"]
                    logger.info(
                        f"{self.label}: Starting EngineCore for DP rank "
                        f"{dp_rank}/{self.global_engine_count}"
                    )
                    info["process"].start()
                    self.engine_core_processes.append(info["process"])
                    wire_rank = (
                        dp_rank * self.pp_size
                        + info["config"].parallel_config.pipeline_parallel_rank
                    )
                    self._local_process_by_rank[wire_rank] = info["process"]

                if not self.is_coordinator:
                    # A worker node hosts engines but owns no router: the
                    # coordinator binds every socket. Wait for the children and
                    # return -- raising here would be caught by the enclosing
                    # handler and reported as a startup failure.
                    logger.info(
                        f"{self.label}: worker node for global DP ranks "
                        f"{[i['dp_rank'] for i in processes_info]}; "
                        f"coordinator at "
                        f"{config.parallel_config.data_parallel_master_ip}"
                    )
                    self._finalizer = weakref.finalize(self, self.close)
                    self.async_output_queue = None
                    self._output_handler_task = None
                    self._asyncio_mode = False
                    for proc in self.engine_core_processes:
                        proc.join()
                    return

                if socket_plan is not None:
                    bind_addresses = [
                        {
                            "input_address": f"tcp://0.0.0.0:{p.input_port}",
                            "output_address": f"tcp://0.0.0.0:{p.output_port}",
                            "control_address": f"tcp://0.0.0.0:{p.control_port}",
                        }
                        for p in socket_plan
                    ]
                else:
                    bind_addresses = [info["addresses"] for info in processes_info]

                for addresses in bind_addresses:
                    input_socket = make_zmq_socket(
                        self.ctx,
                        addresses["input_address"],
                        zmq.ROUTER,
                        bind=True,
                        hwm=config.ipc_max_pending_jobs,
                    )
                    identity = self._receive_identity(input_socket)
                    self.input_sockets.append(input_socket)
                    self.engine_core_identities.append(identity)

                    control_socket = make_zmq_socket(
                        self.ctx,
                        addresses["control_address"],
                        zmq.ROUTER,
                        bind=True,
                        hwm=config.ipc_max_pending_jobs,
                    )
                    control_identity = self._receive_identity(control_socket)
                    self.control_sockets.append(control_socket)
                    self.control_identities.append(control_identity)

                    # PULL always binds; the engine's PUSH always connects.
                    # True for ipc:// and tcp:// alike -- the transport
                    # difference is carried by the address (the engine gets
                    # tcp://<master_ip>:port, we bind tcp://0.0.0.0:port).
                    output_socket = make_zmq_socket(
                        self.ctx,
                        addresses["output_address"],
                        zmq.PULL,
                        bind=True,
                        hwm=config.ipc_max_pending_jobs,
                    )
                    self.output_sockets.append(output_socket)
                    self.shutdown_paths.append(get_open_zmq_inproc_path())

                self._wait_for_all_ready_signals()
                logger.info(
                    f"{self.label}: All EngineCores are fully initialized and ready"
                )

                for dp_rank in range(len(self.output_sockets)):
                    output_thread = self._create_output_thread(
                        dp_rank,
                        self.output_sockets[dp_rank],
                        self.shutdown_paths[dp_rank],
                    )
                    output_thread.start()
                    self.output_threads.append(output_thread)

            finally:
                # A worker node reaches this `finally` via its normal early
                # return above, with its children already joined and dead.
                # The liveness check does not apply to it -- dead children are
                # expected, not a startup failure.
                if self.is_coordinator and self.finished_procs():
                    logger.error(
                        f"{self.label}: Some processes failed to start, shutting down all"
                    )
                    self.close()
                    raise RuntimeError("Failed to start all EngineCore processes")

        except Exception as e:
            logger.error(
                f"{self.label}: Failed to initialize all EngineCores, cleaning up: {e}"
            )
            self.close()
            raise

        logger.info(
            f"{self.label}: All {len(self.output_sockets)} EngineCores initialized and ready"
        )
        self._finalizer = weakref.finalize(self, self.close)
        self.async_output_queue = asyncio.Queue() if config.asyncio_mode else None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    def _record_ready_payload(self, data) -> None:
        """Fold one rank's READY facts into the manager's view of capacity."""
        if not data:
            return
        reported = data.get("max_pool_tokens")
        if reported is None:
            return
        if self.max_pool_tokens is None:
            self.max_pool_tokens = reported
        else:
            self.max_pool_tokens = min(self.max_pool_tokens, reported)

    def _wait_for_all_ready_signals(self):
        pending = set(range(len(self.output_sockets)))
        poller = zmq.Poller()
        for socket in self.output_sockets:
            poller.register(socket, zmq.POLLIN)
        deadline = self._startup_deadline
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Engine IPC handshake timed out: ranks {sorted(pending)}"
                )
            if self.finished_procs():
                raise RuntimeError("Engine process exited during IPC handshake")
            for socket, _ in poller.poll(100):
                rank = self.output_sockets.index(socket)
                message = self._ipc.result.decode(socket.recv_multipart(copy=False))
                self._handle_handshake(rank, message)
                if self._ipc.peers.get(rank, {}).get("ready"):
                    pending.discard(rank)

    def _receive_identity(self, socket):
        while not socket.poll(100):
            if time.monotonic() >= self._startup_deadline or self.finished_procs():
                raise RuntimeError("engine failed to connect its IPC socket")
        frames = socket.recv_multipart()
        if len(frames) != 2 or frames[1] != b"":
            raise ValueError("invalid engine socket identity handshake")
        return frames[0]

    def _handle_handshake(self, rank, message):
        if message.kind == EngineCoreRequestType.READY:
            engine_id = message.metadata["engine_id"]
            if not engine_id or rank in self._ipc.peers:
                raise ValueError("missing or duplicate engine instance ID")
            self._record_ready_payload(message.payload)
            host_id = message.payload.get("host_id", "")
            self._ipc.peers[rank] = {
                "engine_id": engine_id,
                "shm": False,
                "ready": False,
                "host_id": host_id,
                "buffer_modes": message.payload.get("buffer_modes", []),
                "same_host": bool(host_id and host_id == HostIdentity.current()),
            }
            self.input_sockets[rank].setsockopt(zmq.ROUTER_MANDATORY, 1)
            self.control_sockets[rank].setsockopt(zmq.ROUTER_MANDATORY, 1)
            self._send_control(
                rank,
                self._ipc.control.encode(
                    "BUFFER_PROBE",
                    self._ipc.probe_payload(rank),
                    **self._ipc.context(rank),
                ),
            )
        elif message.kind == "BUFFER_PROBE_RESULT":
            peer = self._ipc.peers.get(rank, {})
            if message.metadata[
                "session_id"
            ] != self._ipc.session_id or message.metadata["engine_id"] != peer.get(
                "engine_id"
            ):
                raise ValueError("IPC probe belongs to another session")
            self._ipc.accept_probe(rank, message.payload)
        elif message.kind == EngineCoreRequestType.METRICS:
            self.latest_metrics[rank] = message.payload
        else:
            raise RuntimeError(f"Unexpected IPC handshake message: {message.kind}")

    def _create_output_thread(
        self, dp_rank: int, output_socket: zmq.Socket, shutdown_path: str
    ) -> Thread:
        def process_outputs_socket():
            assert isinstance(output_socket, zmq.Socket)
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            monitor = output_socket.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(output_socket, zmq.POLLIN)
                poller.register(monitor, zmq.POLLIN)
                logger.debug(f"{self.label} (DP {dp_rank}): output thread started")
                while True:
                    socks = dict(poller.poll(100))
                    self._ipc.reap()
                    if self._closed:
                        break
                    if monitor in socks:
                        recv_monitor_message(monitor)
                        raise ConnectionError(f"engine rank {dp_rank} disconnected")
                    process = self._local_process_by_rank.get(dp_rank)
                    if process is not None and not process.is_alive():
                        raise RuntimeError(f"engine rank {dp_rank} exited")
                    if not socks:
                        continue
                    if shutdown_socket in socks:
                        # shutdown signal, exit thread.
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive shutdown signal"
                        )
                        break
                    if output_socket not in socks:
                        continue

                    message = self._ipc.result.decode(
                        output_socket.recv_multipart(copy=False),
                        session_id=self._ipc.session_id,
                        engine_id=self._ipc.peers[dp_rank]["engine_id"],
                    )
                    request_type, data = message.kind, message.payload
                    self._ipc.reap()
                    if request_type == "BUFFER_RELEASE":
                        self._ipc.release_consumer(dp_rank, data)
                        continue
                    if request_type in ("ERROR", "STOPPED"):
                        self._fail_peer(dp_rank, RuntimeError(str(data)))
                        break
                    if request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive SHUTDOWN request"
                        )
                        self._shutdown_engine_core_rank(dp_rank)
                        break
                    elif request_type == EngineCoreRequestType.STREAM:
                        stream_outputs = data  # List of (seq_id, RequestOutput) tuples
                        logger.debug(
                            f"{self.label}: Received STREAM message with {len(stream_outputs)} outputs"
                        )
                        # Delivered only through the per-seq callbacks below.
                        # These also used to go onto stream_outputs_queue,
                        # which nothing ever read, so every RequestOutput
                        # stayed reachable for the life of the process and
                        # made each gen-2 GC pass progressively slower.
                        #
                        # The f-strings below are built by the caller before
                        # logger.debug() can drop them, so check the level
                        # once per step rather than twice per chunk.
                        dbg = logger.isEnabledFor(logging.DEBUG)
                        for seq_id, request_output in stream_outputs:
                            # The first emitted model output means this
                            # sequence's prompt prefill has completed. Keep the
                            # request count charged for decode pressure, but
                            # stop advertising its prompt as queued prefill.
                            self._mark_seq_prefill_complete(seq_id)
                            callback = self._seq_id_to_callback.get(seq_id)
                            if dbg:
                                logger.debug(
                                    f"{self.label}: seq_id={seq_id}, callback={'found' if callback is not None else 'NOT FOUND'}, tokens={request_output.output_tokens}"
                                )
                            if callback is not None:
                                try:
                                    callback(request_output)
                                    if dbg:
                                        logger.debug(
                                            f"{self.label}: Successfully called callback for seq_id={seq_id}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"Error calling stream_callback for sequence {seq_id}: {e}",
                                        exc_info=True,
                                    )
                            if request_output.finished:
                                self._seq_id_to_callback.pop(seq_id, None)
                                self._release_seq_load(seq_id)
                                if dbg:
                                    logger.debug(
                                        f"{self.label}: Cleaned up callback for finished sequence {seq_id}"
                                    )
                        # Batched stream dispatch: the per-seq callbacks only buffer
                        # their chunks into a thread-local; flush the whole step's
                        # buffer into the per-request stream collectors now (one
                        # call_soon_threadsafe per loop). Resolved lazily by the API
                        # server to avoid the api_server <-> engine_core_mgr import
                        # cycle. No-op when no streaming request is in flight.
                        if self._flush_stream_batch_fn is not None:
                            try:
                                self._flush_stream_batch_fn()
                            except Exception as e:
                                logger.warning(
                                    f"{self.label}: flush_stream_batch failed: {e}",
                                    exc_info=True,
                                )
                    elif request_type == EngineCoreRequestType.METRICS:
                        self.latest_metrics[dp_rank] = data
                    elif request_type == EngineCoreRequestType.UTILITY_RESPONSE:
                        self._route_utility_response(dp_rank, data)
                    elif request_type == EngineCoreRequestType.ADD:
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        with self._submission_lock:
                            seqs = [
                                CompletionResult(**value)
                                for value in data
                                if value["id"] in self._request_targets
                            ]
                        # Offline (non-streaming) completions arrive here as
                        # finished sequences; release their in-flight DP load.
                        #
                        # So do sequences the scheduler rejected before they
                        # ever ran (`_unschedulable_reason`, abort-while-waiting)
                        # — those never reach `postprocess`, so no STREAM chunk
                        # is ever built for them. An online client is still
                        # holding a callback and would wait forever, so the
                        # terminal output it is owed has to be raised here.
                        # Anything already delivered through STREAM has had its
                        # callback popped by then (STREAM is enqueued ahead of
                        # the finished-seq list and this socket is FIFO), so a
                        # normal completion finds nothing to do below.
                        delivered = False
                        for seq in seqs:
                            delivered |= self._deliver_terminal_output(seq)
                            self._release_seq_load(seq.id)
                            with self._submission_lock:
                                other_ranks = self._committed_targets.get(
                                    seq.id, set()
                                ) - {dp_rank}
                            for other_rank in other_ranks:
                                self._send_abort(other_rank, seq.id)
                        if delivered and self._flush_stream_batch_fn is not None:
                            # The callbacks above only buffer into a thread-local;
                            # without this flush the chunk never reaches the
                            # request's collector and the client still hangs.
                            try:
                                self._flush_stream_batch_fn()
                            except Exception as e:
                                logger.warning(
                                    f"{self.label}: flush_stream_batch failed: {e}",
                                    exc_info=True,
                                )
                        offline = []
                        with self._submission_lock:
                            for seq in seqs:
                                if seq.id not in self._callback_only:
                                    offline.append(seq)
                                self._callback_only.discard(seq.id)
                                self._request_targets.pop(seq.id, None)
                                self._committed_targets.pop(seq.id, None)
                                self._submission_ids.pop(seq.id, None)
                                self._cancelled.discard(seq.id)
                        if offline:
                            self.outputs_queue.put_nowait(offline)
            except Exception as exc:
                logger.exception(
                    "%s: output channel for rank %s failed", self.label, dp_rank
                )
                self._fail_peer(dp_rank, exc)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                output_socket.disable_monitor()
                monitor.close(linger=0)
                output_socket.close(linger=0)

        return Thread(
            target=process_outputs_socket,
            name=f"EngineCoreOutputThread-DP{dp_rank}",
            daemon=True,
        )

    def _ensure_output_handler_task(self):
        if self._asyncio_mode and self._output_handler_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._output_handler_task = loop.create_task(
                    self._async_output_handler()
                )
            except RuntimeError:
                # If no running event loop, try to get/create one
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._output_handler_task = loop.create_task(
                        self._async_output_handler()
                    )
                else:
                    raise RuntimeError(
                        "CoreManager with asyncio_mode requires a running event loop"
                    )

    async def _async_output_handler(self):
        loop = asyncio.get_event_loop()
        while True:
            # Use run_in_executor to avoid blocking event loop
            seqs = await loop.run_in_executor(None, self.outputs_queue.get)
            if isinstance(seqs, BaseException):
                await self.async_output_queue.put(seqs)
                break
            await self.async_output_queue.put(seqs)

    async def get_output_async(self) -> list[Sequence]:
        if not self.async_output_queue:
            raise RuntimeError("Engine async mode not enabled")

        # Ensure output handler task is started
        self._ensure_output_handler_task()

        seqs = await self.async_output_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    def close(self):
        if self._closed:
            return
        with self._submission_lock:
            self._closed = True
            self._ipc.closed = True
        self._ipc.executor.shutdown(wait=True, cancel_futures=True)
        for rank in range(len(self.input_sockets)):
            self._fail_peer(rank, RuntimeError("CoreManager closed"))

        logger.info(
            f"{self.label}: Shutting down {len(self.input_sockets)} EngineCores"
        )

        for dp_rank in range(len(self.input_sockets)):
            self._shutdown_engine_core_rank(dp_rank)

        for input_socket in self.input_sockets:
            if not input_socket.closed:
                input_socket.close(linger=0)

        for control_socket in self.control_sockets:
            if not control_socket.closed:
                control_socket.close(linger=0)

        for shutdown_path, thread in zip(self.shutdown_paths, self.output_threads):
            if shutdown_path and thread.is_alive():
                try:
                    with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                        shutdown_sender.setsockopt(zmq.LINGER, 0)
                        shutdown_sender.setsockopt(zmq.SNDTIMEO, 100)
                        shutdown_sender.connect(shutdown_path)
                        shutdown_sender.send(b"")
                except zmq.ZMQError as e:
                    logger.debug(f"{self.label}: Error sending shutdown signal: {e}")

        for thread in self.output_threads:
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

        for rank, output_socket in enumerate(self.output_sockets):
            if (
                rank >= len(self.output_threads)
                or not self.output_threads[rank].is_alive()
            ):
                output_socket.close(linger=0)

        # Wait for EngineCore processes to exit gracefully.
        # Use a single deadline so all processes share the grace period
        # instead of sequential per-process timeouts.  This prevents early
        # process exits from destroying the NCCL TCPStore while later
        # processes' HeartbeatMonitor threads still depend on it.
        import time

        deadline = time.monotonic() + envs.ATOM_SHUTDOWN_TIMEOUT_S
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                remaining = max(deadline - time.monotonic(), 0)
                proc.join(timeout=remaining)

        # Terminate any that are still alive.
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.terminate()
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.join(timeout=1)

        dead_engines = []
        # Final join + close to release sentinel semaphores
        for proc in self.engine_core_processes:
            if proc is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=1)
                if not proc.is_alive():
                    dead_engines.extend(
                        self._ipc.peers[rank]["engine_id"]
                        for rank, local in self._local_process_by_rank.items()
                        if local is proc and rank in self._ipc.peers
                    )
                try:
                    proc.close()
                except (ValueError, OSError):
                    pass

        if all(not thread.is_alive() for thread in self.output_threads):
            self.ctx.term()
        self._ipc.close(dead_engines)

        # Release the rendezvous server after stopping the local engine processes.
        self._distributed_stores.clear()
        logger.info(f"{self.label}: All EngineCores shut down")

    def _send_request(self, dp_rank, payload):
        """The request socket stays on the caller's original writer thread."""
        writer = get_ident()
        if self._request_writer_id is None:
            self._request_writer_id = writer
        elif self._request_writer_id != writer:
            raise RuntimeError("request socket used from a different writer thread")
        return self.input_sockets[dp_rank].send_multipart(
            [self.engine_core_identities[dp_rank], *payload.frames],
            flags=zmq.DONTWAIT,
            copy=False,
            track=True,
        )

    def _send_control(self, dp_rank, payload, *, deadline=None):
        deadline = min(
            deadline if deadline is not None else float("inf"),
            time.monotonic() + self._config.ipc_send_timeout,
        )
        if not self._control_send_lock.acquire(
            timeout=max(0, deadline - time.monotonic())
        ):
            raise TimeoutError("engine control writer is busy")
        try:
            while True:
                try:
                    tracker = self.control_sockets[dp_rank].send_multipart(
                        [self.control_identities[dp_rank], *payload.frames],
                        flags=zmq.DONTWAIT,
                        copy=False,
                        track=True,
                    )
                    self._ipc.retain(tracker, payload)
                    return
                except zmq.Again:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("engine control send timed out")
                    self.control_sockets[dp_rank].poll(10, zmq.POLLOUT)
        finally:
            self._control_send_lock.release()

    def _plan_requests(self, seqs):
        hints = self._resolve_and_validate_hints(seqs)
        buckets = [[] for _ in range(self._routable_engine_count)]
        with self._lb_lock:
            for seq, hint in zip(seqs, hints):
                rank = (
                    0
                    if self.pp_size > 1
                    else self._select_dp_rank_for_seq_locked(seq, hint)
                )
                if (
                    self._dp_lb_strategy != "round_robin"
                    or self._dp_session_affinity_enabled
                ):
                    self._charge_seq_load_locked(seq, rank)
                buckets[rank].append(seq)
        size = self._config.ipc_max_batch_size
        return [
            (rank, values[start : start + size])
            for rank, values in enumerate(buckets)
            for start in range(0, len(values), size)
        ]

    def _register_submission(self, seqs):
        with self._submission_lock:
            if self._closed:
                raise RuntimeError("CoreManager is closed")
            if len({seq.id for seq in seqs}) != len(seqs):
                raise ValueError("duplicate request IDs in admission")
            if any(seq.id in self._request_targets for seq in seqs):
                raise ValueError("request is already submitted")
            routes = self._plan_requests(seqs)
            sequence = self._ipc.next_id()
            if any(rank in self._peer_failures for rank, _ in routes):
                for seq in seqs:
                    self._release_seq_load(seq.id)
                raise RuntimeError("an admission target is unavailable")
            for rank, values in routes:
                for seq in values:
                    seq.submission_sequence = sequence
                    self._request_targets.setdefault(seq.id, set()).add(rank)
                    self._submission_ids[seq.id] = sequence
            for seq in seqs:
                if seq.stream_callback is not None:
                    self._seq_id_to_callback[seq.id] = seq.stream_callback
                if seq.callback_only:
                    self._callback_only.add(seq.id)
            return routes

    def _try_commit(self, entry):
        with self._submission_lock:
            if self._closed:
                raise RuntimeError("CoreManager closed during admission")
            if entry.rank in self._peer_failures:
                raise RuntimeError(f"engine rank {entry.rank} is unavailable")
            if all(seq.id in self._cancelled for seq in entry.requests):
                return True, None
            self._ipc.activate(entry)
            try:
                tracker = self._send_request(entry.rank, entry.message)
            except zmq.Again:
                return False, None
            entry.sent = True
            entry.tracker = tracker
            cancelled = []
            for seq in entry.requests:
                self._committed_targets.setdefault(seq.id, set()).add(entry.rank)
                if seq.id in self._cancelled:
                    cancelled.append(seq.id)
            self._ipc.retain(tracker, entry.message, entry.pins)
            entry.pins = []
        for req_id in cancelled:
            self._send_abort(entry.rank, req_id)
        return True, tracker

    def _finish_submission(self, routes, prepared, nbytes, trackers, failed=False):
        committed = {
            seq.id for entry in prepared if entry.sent for seq in entry.requests
        }
        all_ids = {seq.id for _, values in routes for seq in values}
        if failed:
            for req_id in committed:
                self.abort_request(req_id)
        for entry in prepared:
            if entry.sent:
                entry.release()
            else:
                self._ipc.forget_unsent(entry)
        with self._submission_lock:
            for req_id in all_ids - committed:
                self._release_seq_load(req_id)
                self._request_targets.pop(req_id, None)
                self._submission_ids.pop(req_id, None)
                self._seq_id_to_callback.pop(req_id, None)
                self._callback_only.discard(req_id)
                self._cancelled.discard(req_id)
        self._ipc.defer_unreserve(
            nbytes, [entry.tracker for entry in prepared if entry.sent]
        )

    def add_request(self, seqs):
        if not seqs:
            return
        tracing.bind_sequences(seqs, unsupported=self._trace_unsupported)
        nbytes = self._ipc.input_bytes(seqs)
        deadline = time.monotonic() + self._config.ipc_send_timeout
        while not self._ipc.reserve(nbytes):
            if time.monotonic() >= deadline:
                raise TimeoutError("IPC preparation queue is full")
            time.sleep(0.005)
        routes, prepared, trackers = [], [], []
        failed = True
        try:
            routes = self._register_submission(seqs)
            prepared = self._ipc.prepare(routes)
            for entry in prepared:
                while True:
                    done, tracker = self._try_commit(entry)
                    if done:
                        trackers.append(tracker)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("engine request send timed out")
                    self.input_sockets[entry.rank].poll(10, zmq.POLLOUT)
            failed = False
        finally:
            self._finish_submission(routes, prepared, nbytes, trackers, failed)

    async def add_request_async(self, seqs):
        if not seqs:
            return
        tracing.bind_sequences(seqs, unsupported=self._trace_unsupported)
        nbytes = self._ipc.input_bytes(seqs)
        deadline = time.monotonic() + self._config.ipc_send_timeout
        while not self._ipc.reserve(nbytes):
            if time.monotonic() >= deadline:
                raise TimeoutError("IPC preparation queue is full")
            await asyncio.sleep(0.005)
        routes, prepared, trackers = [], [], []
        failed = True
        deferred_cleanup = False
        try:
            routes = self._register_submission(seqs)
            future = self._ipc.executor.submit(self._ipc.prepare, routes)
            try:
                prepared = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                # The worker owns the buffers until sealing actually finishes.
                def discard(done, routes=routes, nbytes=nbytes):
                    try:
                        values = done.result()
                    except BaseException:
                        logger.debug(
                            "Cancelled admission preparation failed", exc_info=True
                        )
                        values = []
                    self._finish_submission(routes, values, nbytes, [], True)

                deferred_cleanup = True
                future.add_done_callback(discard)
                raise
            for entry in prepared:
                while True:
                    done, tracker = self._try_commit(entry)
                    if done:
                        trackers.append(tracker)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("engine request send timed out")
                    await asyncio.sleep(0.001)
            failed = False
        finally:
            if not deferred_cleanup:
                self._finish_submission(routes, prepared, nbytes, trackers, failed)

    @property
    def _routable_engine_count(self) -> int:
        """How many engine ranks this manager may route to.

        Equals ``global_engine_count``, which on a coordinator spans other
        nodes. Falls back to ``local_engine_count`` for callers that build the
        routing state by hand instead of through ``_init_shared_state`` --
        ``tests/test_dp_load_balance.py`` does exactly that, and single-node
        callers predate the global/local split, where the two are equal anyway.
        """
        return getattr(self, "global_engine_count", self.local_engine_count)

    def _resolve_and_validate_hints(self, seqs: list[Sequence]) -> list[int | None]:
        """Resolve every seq's explicit ``data_parallel_rank`` hint and validate
        the whole batch, once.

        Returns the per-seq resolved hint (an int rank, or None for a
        load-balanced seq) so the dispatch loop can reuse it instead of calling
        getattr/int a second time per seq.

        Validation runs BEFORE any load is charged so a bad hint in the middle
        of a batch cannot leave earlier siblings charged-but-undispatched (a
        permanent in-flight-load leak).
        """
        hints: list[int | None] = []
        engine_count = self._routable_engine_count
        for seq in seqs:
            raw = getattr(seq, "data_parallel_rank", None)
            hint = None if raw is None else int(raw)
            if hint is not None and not 0 <= hint < engine_count:
                raise ValueError(
                    f"Invalid data_parallel_rank={hint}; "
                    f"global_engine_count={engine_count}"
                )
            hints.append(hint)
        return hints

    def _dispatch_to_dp_ranks(self, seqs):
        self.add_request(seqs)

    def _select_dp_rank_locked(self) -> int:
        """Pick a DP engine rank for a new request. Caller must hold _lb_lock.

        - "round_robin": load-agnostic rotation.
        - "least_requests" (default): fewest in-flight requests, ties broken by
          the lighter in-flight prompt-token load. Request count keeps the
          lockstep DP ranks in phase; the token tie-break packs pending prefill
          work evenly across the equal-request ranks.
        - "least_tokens": lowest combined load ``tokens + req_equiv * reqs``
          (prefill pressure + decode-slot pressure).

        Fully-tied ranks are resolved by a rotating cursor so selection does not
        always fall on rank 0. See docs/distributed_guide.md for the rationale.
        """
        n = self._routable_engine_count
        if self._dp_lb_strategy == "round_robin":
            dp_rank = self._rank_rotation_cursor % n
            self._rank_rotation_cursor += 1
            return dp_rank

        # argmin over per-rank load, scanned from a rotating start offset so a run
        # of fully-equal ranks spreads evenly. Scores are computed inline (no
        # intermediate list) — the loop reads the counters directly.
        least_requests = self._dp_lb_strategy == "least_requests"
        best_rank = 0
        best_score = None
        offset = self._rank_rotation_cursor % n
        for i in range(n):
            r = (offset + i) % n
            if least_requests:
                # Lexicographic (request count, prompt-token load): tuples compare
                # element-wise, so tokens only decide among request-count ties.
                score = (self._rank_reqs[r], self._rank_tokens[r])
            else:  # "least_tokens"
                score = (
                    self._rank_tokens[r] + self._dp_lb_req_equiv * self._rank_reqs[r]
                )
            if best_score is None or score < best_score:
                best_score = score
                best_rank = r
        self._rank_rotation_cursor += 1
        return best_rank

    def _stable_session_rank(self, session_id: str) -> int:
        """Map a session to a stable rank with rendezvous hashing.

        Python's built-in ``hash`` is process-randomized, so it cannot define
        cache ownership. Rendezvous hashing is deterministic and, unlike a
        simple modulo, remaps only sessions owned by a rank that is added or
        removed. The DP width is tiny, and this runs only once per new session.
        """
        session_key = str(session_id).encode("utf-8")
        best_rank = 0
        best_score = -1
        for rank in range(self._routable_engine_count):
            digest = hashlib.blake2b(
                session_key + rank.to_bytes(4, "little"), digest_size=8
            ).digest()
            score = int.from_bytes(digest, "little")
            if score > best_score:
                best_rank = rank
                best_score = score
        return best_rank

    def _select_new_session_rank_locked(self, session_id: str) -> int:
        """Place a new sticky session on the lightest DP rank.

        Existing sessions never call this function: locality wins once an
        owner has cache state.  Before that first placement there is no cache
        to preserve, so use estimated outstanding prefill debt plus the
        configured token-equivalent decode pressure.  Rendezvous hashing is
        only the deterministic tie-breaker; it must not override real load.
        """
        session_key = str(session_id).encode("utf-8")
        best_rank = 0
        best_load = None
        best_hash = -1
        for rank in range(self._routable_engine_count):
            load = (
                self._rank_tokens[rank] + self._dp_lb_req_equiv * self._rank_reqs[rank]
            )
            tie_hash = int.from_bytes(
                hashlib.blake2b(
                    session_key + rank.to_bytes(4, "little"), digest_size=8
                ).digest(),
                "little",
            )
            if (
                best_load is None
                or load < best_load
                or (load == best_load and tie_hash > best_hash)
            ):
                best_rank = rank
                best_load = load
                best_hash = tie_hash
        return best_rank

    def _record_dp_route_locked(self, decision: str, rank: int) -> int:
        """Account for one routing decision and return its target rank."""
        self._dp_route_counters[decision] += 1
        self._rank_routed_total[rank] += 1
        return rank

    def _select_dp_rank_for_seq_locked(
        self, seq: Sequence, explicit_rank: int | None
    ) -> int:
        """Route one sequence using explicit hint, strict owner, then load.

        This intentionally matches SGLang Model Gateway's agentic routing
        semantics: a stable correlation/session id selects one DP rank, and
        every later turn stays there. Load cannot move an existing session;
        doing so turns a cheap cache hit into a potentially huge prefill.

        Parent lineage does not affect placement. Each child correlation id is
        its own sticky session, preventing sibling subagents from dogpiling the
        root's rank. Requests without a session retain normal load balancing.
        """
        if explicit_rank is not None:
            if self._dp_session_affinity_enabled:
                # An explicit placement is authoritative and becomes the
                # session's owner for subsequent unhinted requests.
                session_id = getattr(seq, "dp_session_id", None)
                if session_id:
                    old_owner = self._dp_session_owners.get(session_id)
                    if old_owner is not None and old_owner != explicit_rank:
                        # The new rank cannot be assumed to own the old prefix.
                        self._dp_session_prompt_tokens.pop(session_id, None)
                    self._dp_session_owners[session_id] = explicit_rank
            return self._record_dp_route_locked("explicit_total", explicit_rank)

        session_id = getattr(seq, "dp_session_id", None)
        parent_id = getattr(seq, "dp_parent_session_id", None)
        if not self._dp_session_affinity_enabled or not session_id:
            rank = self._select_dp_rank_locked()
            return self._record_dp_route_locked("load_balanced_total", rank)

        owner = self._dp_session_owners.get(session_id)
        if owner is None or not 0 <= owner < self._routable_engine_count:
            owner = self._select_new_session_rank_locked(session_id)
            self._dp_session_owners[session_id] = owner
            if parent_id:
                self._dp_route_counters["affinity_parent_ignored_total"] += 1
            logger.debug(
                "%s: DPA load-aware affinity new session=%s parent=%s owner=rank%d "
                "load=%d tokens=%d reqs=%d",
                self.label,
                session_id,
                parent_id,
                owner,
                self._rank_tokens[owner]
                + self._dp_lb_req_equiv * self._rank_reqs[owner],
                self._rank_tokens[owner],
                self._rank_reqs[owner],
            )
            return self._record_dp_route_locked("affinity_new_total", owner)

        return self._record_dp_route_locked("affinity_owner_hit_total", owner)

    def get_dp_router_statistics(self) -> dict:
        """Return a consistent, non-mutating snapshot of DP routing state."""
        with self._lb_lock:
            sessions_per_rank = [0] * self._routable_engine_count
            for rank in self._dp_session_owners.values():
                if 0 <= rank < self._routable_engine_count:
                    sessions_per_rank[rank] += 1
            return {
                "enabled": self._routable_engine_count > 1,
                **self._dp_route_counters,
                "requests_per_rank": list(self._rank_routed_total),
                "inflight_requests_per_rank": list(self._rank_reqs),
                "queued_prefill_tokens_per_rank": list(self._rank_tokens),
                "session_count_per_rank": sessions_per_rank,
            }

    def _charge_seq_load_locked(self, seq: Sequence, dp_rank: int) -> None:
        """Record a seq's in-flight load on dp_rank. Caller must hold _lb_lock."""
        req_cost = 1
        prompt_tokens = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        tok_cost = prompt_tokens
        session_id = getattr(seq, "dp_session_id", None)
        if self._dp_session_affinity_enabled and session_id:
            previous_prompt_tokens = self._dp_session_prompt_tokens.get(session_id)
            if previous_prompt_tokens is not None:
                # Agentic turns normally extend their previous prompt.  Charge
                # only that extension; request-equivalent load still accounts
                # for lookup/decode pressure when the delta is zero.
                tok_cost = max(0, prompt_tokens - previous_prompt_tokens)
            self._dp_session_prompt_tokens[session_id] = prompt_tokens
        self._rank_reqs[dp_rank] += req_cost
        self._rank_tokens[dp_rank] += tok_cost
        self._seq_load[seq.id] = (dp_rank, req_cost, tok_cost)

    def _deliver_terminal_output(self, seq) -> bool:
        """Give an online client the terminal output for a seq that never streamed.

        A sequence the scheduler rejected before it ran produced no tokens and
        no STREAM chunk, so nothing has answered the client yet. Synthesises the
        finished `RequestOutput` the normal path would have built in
        `postprocess`, carrying `leave_reason` as the finish reason so the
        response says why it ended rather than closing empty.

        Returns whether a callback was invoked, so the caller knows to flush the
        batch the callback buffered into. Offline sequences register no callback
        and normal completions have had theirs popped by the STREAM that
        delivered them, so both answer False and keep their existing path.
        """
        callback = self._seq_id_to_callback.pop(seq.id, None)
        if callback is None:
            return False
        # Never empty: an empty reason reaches the client as `finish_reason:
        # null`, which the OpenAI schema reserves for a choice still being
        # generated -- so a client would keep waiting on a finished stream.
        reason = seq.leave_reason or "rejected"
        try:
            callback(
                RequestOutput(
                    request_id=seq.id,
                    output_tokens=[],
                    finished=True,
                    finish_reason=reason,
                )
            )
        except Exception as e:
            logger.warning(
                f"Error delivering terminal output for sequence {seq.id}: {e}",
                exc_info=True,
            )
            return False
        logger.info(
            "%s: seq %s returned without running: %s", self.label, seq.id, reason
        )
        return True

    def _release_seq_load(self, seq_id) -> None:
        """Undo a seq's in-flight load when it finishes or is aborted.

        Idempotent: a seq is only charged once and released once, so a repeated
        call (e.g. finish followed by abort) is a no-op.
        """
        with self._lb_lock:
            entry = self._seq_load.pop(seq_id, None)
            if entry is None:
                return
            dp_rank, req_cost, tok_cost = entry
            self._rank_reqs[dp_rank] -= req_cost
            self._rank_tokens[dp_rank] -= tok_cost

    def _mark_seq_prefill_complete(self, seq_id) -> None:
        """Release only a sequence's prefill-token charge, once."""
        # Every decode token reaches this path. The prefill charge may be zero
        # at admission, or already released by the first output. Avoid taking
        # the router lock for a no-op; re-read under the lock before changing
        # counters because abort/finish may release the charge concurrently.
        entry = self._seq_load.get(seq_id)
        if entry is None:
            return
        _, _, tok_cost = entry
        if tok_cost == 0:
            return
        with self._lb_lock:
            entry = self._seq_load.get(seq_id)
            if entry is None:
                return
            dp_rank, req_cost, tok_cost = entry
            if tok_cost == 0:
                return
            self._rank_tokens[dp_rank] -= tok_cost
            self._seq_load[seq_id] = (dp_rank, req_cost, 0)

    def reset_dp_router(self) -> None:
        """Reset all DP routing state (rotation cursor + in-flight load).

        Called at the start of a fresh offline ``generate()`` batch so counts do
        not leak across independent batches and DP assignment is deterministic.

        Precondition: the previous batch has fully drained. If any request is
        still charged when this runs (e.g. this CoreManager is being shared with
        a concurrent streaming path), that request's later release becomes a
        no-op and the per-rank counters would drift — so we warn loudly instead
        of corrupting accounting silently.
        """
        with self._lb_lock:
            if self._seq_load:
                logger.warning(
                    "%s: reset_dp_router() called with %d request(s) still "
                    "charged in-flight; dropping their load. Expected only "
                    "between fully-drained offline batches — a shared/concurrent "
                    "CoreManager will see counters drift.",
                    self.label,
                    len(self._seq_load),
                )
            self._rank_rotation_cursor = 0
            self._rank_reqs = [0] * self._routable_engine_count
            self._rank_tokens = [0] * self._routable_engine_count
            self._seq_load.clear()
            self._dp_session_owners.clear()
            self._dp_session_prompt_tokens.clear()

    def _utility_message(self, rank, cmd, args):
        """*cmd* for *rank*. A ``request_id`` in *args* asks for a reply."""
        return self._ipc.control.encode(
            "UTILITY",
            (cmd, args),
            **self._ipc.context(rank),
            metadata={
                "expected_ranks": [rank],
                "same_host": self._ipc.peers[rank]["same_host"],
            },
        )

    def send_utility_command(self, cmd, dp_rank=None):
        self._refuse_unowned_weight_sync_cmd(cmd, None)
        ranks = range(len(self.control_sockets)) if dp_rank is None else [dp_rank]
        for rank in ranks:
            logger.debug(
                f"{self.label}: Send utility command '{cmd}' to DP rank {rank}"
            )
            self._send_control(rank, self._utility_message(rank, cmd, {}))

    def _send_abort(self, rank, req_id):
        self._send_control(
            rank,
            self._ipc.control.encode(
                "ABORT",
                {
                    "request_ids": [req_id],
                    "submission_sequence": self._submission_ids.get(req_id, 0),
                },
                **self._ipc.context(rank),
            ),
        )

    def abort_request(self, req_id):
        with self._submission_lock:
            if req_id not in self._request_targets:
                return
            self._cancelled.add(req_id)
            ranks = tuple(self._committed_targets.get(req_id, ()))
            self._seq_id_to_callback.pop(req_id, None)
        self._release_seq_load(req_id)
        for rank in ranks:
            try:
                self._send_abort(rank, req_id)
            except Exception as exc:
                logger.warning(
                    "Could not cancel request %s on rank %s",
                    req_id,
                    rank,
                    exc_info=True,
                )
                self._fail_peer(rank, exc)

    def _route_utility_response(self, dp_rank: int, data):
        """Hand a utility reply to the caller waiting on its request id.

        A reply nobody waits for is dropped: one with no usable id was asked
        for by a sender that never waits, and one with an id nobody holds is
        late, its caller having given up.
        """
        request_id = data.get("request_id") if isinstance(data, dict) else None
        if data.get("rank", dp_rank) != dp_rank:
            logger.warning(
                f"{self.label}: dropping utility reply {request_id} that rank "
                f"{data.get('rank')} sent on DP rank {dp_rank}'s socket"
            )
            return
        if not isinstance(request_id, str) or not request_id:
            # Routing an unhashable id would raise on this thread and end it.
            logger.debug(
                f"{self.label}: dropping a utility reply no caller can be "
                f"waiting for, from DP rank {dp_rank}"
            )
            return
        if not self._rpc_router.route(request_id, (dp_rank, data)):
            logger.warning(
                f"{self.label}: dropping late utility reply {request_id} "
                f"from DP rank {dp_rank}"
            )

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 300.0,
    ) -> list[RpcResult]:
        """Run *method* on every DP engine's every TP runner.

        Returns replies in DP-major then TP-rank order. Every rank is
        represented: a DP engine that did not answer in time contributes one
        failed :class:`RpcResult` per TP rank rather than silently shortening
        the list, because a caller checking coverage needs to know which ranks
        are missing.

        Never raises for a worker-side failure -- the failure travels in
        ``RpcResult.error`` so the ranks that did succeed are still reported.
        Raises up front for a timeout that is not a finite number of seconds,
        and for a manager whose engines are not one DP rank each, which the
        replies would otherwise be reported as.
        """
        timeout = checked_timeout(timeout)
        unsupported = self._collective_rpc_unsupported()
        if unsupported is not None:
            raise NotImplementedError(
                f"{self.label}: collective_rpc needs one engine per DP rank, "
                f"but {unsupported}"
            )
        request_id = uuid.uuid4().hex
        engine_count = len(self.control_sockets)
        deadline = time.monotonic() + timeout
        worker_deadline = time.time() + engine_budget(timeout)

        with self._rpc_router.register(request_id) as replies:
            self.broadcast_utility_command(
                COLLECTIVE_RPC_CMD,
                _send_deadline=deadline,
                request_id=request_id,
                method=method,
                args=tuple(args),
                kwargs=dict(kwargs or {}),
                barrier=bool(barrier),
                timeout=engine_budget(timeout),
                deadline=worker_deadline,
            )
            by_dp_rank: dict[int, dict] = {}
            while len(by_dp_rank) < engine_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    dp_rank, body = replies.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                if dp_rank in by_dp_rank:
                    logger.warning(
                        f"{self.label}: duplicate collective_rpc reply from "
                        f"DP rank {dp_rank}, ignoring"
                    )
                    continue
                by_dp_rank[dp_rank] = body

        return self._flatten_dp_replies(request_id, method, engine_count, by_dp_rank)

    def _collective_rpc_unsupported(self) -> str | None:
        """Why this manager's engines are not one DP rank each, if they are not."""
        pp_size = getattr(self, "pp_size", 1)
        if pp_size > 1:
            # One engine per stage, each holding a slice of the layers.
            return f"its {pp_size} engines are pipeline stages"
        return None

    def _flatten_dp_replies(
        self,
        request_id: str,
        method: str,
        engine_count: int,
        by_dp_rank: dict[int, dict],
    ) -> list[RpcResult]:
        """DP-major, TP-minor, with a placeholder for every absent rank."""
        ranks = self._rpc_ranks_per_engine
        flat: list[RpcResult] = []
        for dp_rank in range(engine_count):
            body = by_dp_rank.get(dp_rank)
            if body is None:
                error = (
                    f"DP rank {dp_rank} did not answer {method!r} before the deadline"
                )
                flat.extend(
                    RpcResult(request_id, tp, error=error) for tp in range(ranks)
                )
                continue
            if body.get("error"):
                # The engine failed before reaching its runners, so every one of
                # them shares the reason.
                flat.extend(
                    RpcResult(request_id, tp, error=body["error"])
                    for tp in range(ranks)
                )
                continue
            for entry in body.get("results", []):
                flat.append(
                    RpcResult(
                        request_id,
                        entry.get("tp_rank", -1),
                        value=entry.get("value"),
                        error=entry.get("error"),
                    )
                )
        return flat

    def _refuse_unowned_weight_sync_cmd(self, cmd: str, dispatch_token) -> None:
        if (
            cmd in _WEIGHT_SYNC_OWNED_CMDS
            and dispatch_token is not self._weight_sync_dispatch_token
        ):
            raise ValueError(
                f"{self.label}: {cmd!r} cannot be sent directly; weight syncs "
                f"go through broadcast_utility_command_sync, which alone "
                f"finishes or aborts them"
            )

    def broadcast_utility_command(
        self, cmd: str, *, _dispatch_token=None, _send_deadline=None, **kwargs
    ):
        """Send *cmd* to every engine; *_send_deadline* bounds the sends
        (monotonic), apart from the engine's own ``deadline`` in *kwargs*."""
        self._refuse_unowned_weight_sync_cmd(cmd, _dispatch_token)
        if cmd == "abort_request":
            # Cancellation is its own message, sent only where it was admitted.
            self.abort_request(kwargs["req_id"])
            return
        # Refused before the first send, so no engine runs a command its
        # failed peer never will.
        failed = sorted(
            set(self._peer_failures) & set(range(len(self.control_sockets)))
        )
        if failed:
            raise RuntimeError(f"engine rank(s) {failed} are unavailable")
        for rank in range(len(self.control_sockets)):
            logger.debug(
                f"{self.label}: Broadcast utility command '{cmd}' to DP rank {rank}"
            )
            # Encoded per rank: each message carries its own peer's session.
            self._send_control(
                rank,
                self._utility_message(rank, cmd, kwargs),
                deadline=_send_deadline,
            )

    @contextmanager
    def weight_sync_session(self):
        """Serialize one direct/SHM/IPC synchronization session.

        Its updates share one sync id until that sync finishes, and none can
        be sent once it has failed: the engines would otherwise take the rest
        of a failed sync's buckets for a new sync and commit them. A nested
        session joins the outer one's sync.
        """
        with self._weight_sync_lock:
            ctx = self._weight_sync_context
            outer = getattr(ctx, "active", False)
            if not outer:
                ctx.sync_id = uuid.uuid4().hex
                ctx.failure = None
            ctx.active = True
            try:
                yield
            finally:
                ctx.active = outer

    def _discard_failed_weight_sync(self, failed_cmd: str) -> None:
        """Fence every engine this reaches and drop the failed sync's scratch.

        Tried on every rank and never raises: the caller is already raising
        the failure that matters, and a rank whose socket fails here must not
        keep its peers, which may hold the failed update, from hearing it.
        """
        args = {"failed_cmd": failed_cmd}
        for rank in range(len(self.control_sockets)):
            try:
                self._send_control(
                    rank, self._utility_message(rank, DISCARD_WEIGHT_SYNC_CMD, args)
                )
            except Exception:
                logger.exception(
                    f"{self.label}: could not tell DP rank {rank} to discard "
                    f"the failed {failed_cmd!r}"
                )

    def broadcast_utility_command_sync(
        self,
        cmd: str,
        timeout: float = 300.0,
        **kwargs,
    ):
        """Run *cmd* on every engine and return their replies in DP order.

        Raises if an engine reports an error or does not answer in time.
        """
        owns_sync = getattr(self._weight_sync_context, "active", False)
        if cmd == FINISH_WEIGHT_SYNC_CMD:
            raise ValueError(
                f"{self.label}: {cmd!r} commits a weight sync; only the sync "
                f"whose complete update succeeded on every DP engine sends it"
            )
        if cmd == DISCARD_WEIGHT_SYNC_CMD:
            if not owns_sync:
                raise ValueError(
                    f"{self.label}: {cmd!r} aborts the weight sync in flight; "
                    f"send it from inside the `with weight_sync_session()` that "
                    f"owns it"
                )
            ctx = self._weight_sync_context
            ctx.failure = ctx.failure or "aborted by its sender"
        is_nonfinal_bucket = cmd in WEIGHT_UPDATE_UTILITY_CMDS and not (
            completes_weight_sync(cmd, kwargs)
        )
        if is_nonfinal_bucket and not owns_sync:
            raise ValueError(
                f"{self.label}: non-final {cmd!r} needs "
                f"`with weight_sync_session()` so another sync cannot interleave"
            )
        if cmd in FIRE_AND_FORGET_UTILITY_CMDS:
            raise ValueError(
                f"{self.label}: {cmd!r} is fire-and-forget; its handler never "
                f"answers, so waiting on it could only time out. Send it with "
                f"broadcast_utility_command instead."
            )
        if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
            # Re-entrant, so the weight_sync_session() a bucket belongs to
            # takes it again rather than waiting on itself.
            with self._weight_sync_lock:
                return self._sync_weight_update(cmd, timeout, owns_sync, kwargs)
        return self._broadcast_and_wait(cmd, timeout, **kwargs)

    def _sync_weight_update(
        self, cmd: str, timeout: float, in_session: bool, kwargs: dict
    ):
        """Send one weight update tagged with the sync it belongs to.

        Outside a session the update is a whole sync on its own. Inside one, a
        failure closes the session's sync and a finish starts its next one.
        """
        if not in_session:
            return self._broadcast_and_wait(
                cmd, timeout, sync_id=uuid.uuid4().hex, **kwargs
            )
        ctx = self._weight_sync_context
        if ctx.failure is not None:
            raise RuntimeError(
                f"{self.label}: this session's weight sync already failed "
                f"({ctx.failure}), and a failed sync cannot be completed; "
                f"retry it from a new weight_sync_session()"
            )
        try:
            responses = self._broadcast_and_wait(
                cmd, timeout, sync_id=ctx.sync_id, **kwargs
            )
        except BaseException as exc:
            ctx.failure = f"{cmd}: {type(exc).__name__}: {exc}"
            raise
        if completes_weight_sync(cmd, kwargs):
            ctx.sync_id = uuid.uuid4().hex
        return responses

    def _broadcast_and_wait(self, cmd: str, timeout: float, **kwargs):
        """The round trip behind broadcast_utility_command_sync, unchecked.

        That method has already refused what its caller may not send, and for
        an update has taken the session lock and named the update's sync. The
        only other caller is the FINISH below, which runs under that lock.
        """
        timeout = checked_timeout(timeout)
        request_id = uuid.uuid4().hex
        # The global engine count on a coordinator, as the broadcast reaches.
        engine_count = len(self.control_sockets)
        deadline = time.monotonic() + timeout
        worker_deadline = time.time() + engine_budget(timeout)
        by_dp_rank: dict[int, object] = {}
        with self._rpc_router.register(request_id) as replies:
            try:
                self.broadcast_utility_command(
                    cmd,
                    _dispatch_token=self._weight_sync_dispatch_token,
                    _send_deadline=deadline,
                    request_id=request_id,
                    timeout=engine_budget(timeout),
                    deadline=worker_deadline,
                    **kwargs,
                )
            except Exception:
                # The engines before the failed send already hold this update.
                if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                    self._discard_failed_weight_sync(cmd)
                raise
            while len(by_dp_rank) < engine_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    dp_rank, body = replies.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                by_dp_rank.setdefault(dp_rank, body)

        responses = [by_dp_rank[dp_rank] for dp_rank in sorted(by_dp_rank)]
        missing = sorted(set(range(engine_count)) - set(by_dp_rank))
        if missing:
            if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                self._discard_failed_weight_sync(cmd)
            raise TimeoutError(
                f"{self.label}: no reply to utility command {cmd!r} from DP "
                f"rank(s) {missing} within {timeout}s"
            )
        failed = [r for r in responses if isinstance(r, dict) and r.get("error")]
        if failed:
            if cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                # An engine whose local TP ranks all succeeded has no local
                # reason to discard its packed/expert/IPC scratch. Once any DP
                # engine failed, every engine belongs to the abandoned sync.
                self._discard_failed_weight_sync(cmd)
            # Callers read r["result"]; an error reply has none, and handing it
            # back would turn a named failure into a KeyError somewhere else.
            raise RuntimeError(
                f"{self.label}: utility command {cmd!r} failed on {len(failed)} "
                f"of {len(responses)} engine(s): "
                + "; ".join(str(r["error"]) for r in failed)
            )
        if completes_weight_sync(cmd, kwargs):
            try:
                self._broadcast_and_wait(
                    FINISH_WEIGHT_SYNC_CMD,
                    _WEIGHT_SYNC_FINISH_TIMEOUT_S,
                    sync_id=kwargs.get("sync_id"),
                    completed_cmd=cmd,
                    completed_request_id=request_id,
                )
            except Exception:
                self._discard_failed_weight_sync(cmd)
                raise
        return responses

    def _fail_peer(self, rank, error):
        with self._submission_lock:
            if rank in self._peer_failures:
                return
            self._peer_failures[rank] = error
            if rank in self._ipc.peers:
                self._ipc.peers[rank]["ready"] = False
            ids = [
                req_id
                for req_id, targets in self._request_targets.items()
                if rank in targets
            ]
            for req_id in ids:
                self._deliver_terminal_output(
                    CompletionResult(id=req_id, leave_reason="engine_error")
                )
                self._release_seq_load(req_id)
                self._request_targets.pop(req_id, None)
                self._committed_targets.pop(req_id, None)
                self._submission_ids.pop(req_id, None)
                self._callback_only.discard(req_id)
                self._cancelled.discard(req_id)
        # Every utility call goes to every engine, so each one waiting would
        # otherwise wait out its timeout for this rank's reply.
        self._rpc_router.route_all(
            (rank, {"error": f"engine rank {rank} failed: {error}"})
        )
        if self._flush_stream_batch_fn is not None:
            try:
                self._flush_stream_batch_fn()
            except Exception:
                logger.warning("Could not flush failed engine callbacks", exc_info=True)
        self.outputs_queue.put(error)
        if not self._closed:
            self._shutdown_engine_core_rank(rank)
            process = self._local_process_by_rank.get(rank)
            if (
                process is not None
                and not process.is_alive()
                and self._ipc.arena is not None
            ):
                self._ipc.arena.reclaim_dead_engine(self._ipc.peers[rank]["engine_id"])

    def _shutdown_engine_core_rank(self, dp_rank: int):
        # Determine whether this rank has a local process and/or a control socket.
        # On a coordinator with remote engines, dp_rank may exceed the local
        # process list -- but we still hold a live control socket for it, and
        # the remote engine must receive SHUTDOWN so its worker node unblocks.
        process = self._local_process_by_rank.get(dp_rank)
        has_local_process = process is not None
        has_local_alive = process is not None and process.is_alive()

        has_control_socket = (
            dp_rank < len(self.control_sockets)
            and not self.control_sockets[dp_rank].closed
        )

        # Send SHUTDOWN if:
        #   - local rank with a live process (original behavior), OR
        #   - remote rank that has an open control socket (new: coordinator → worker).
        should_send = has_local_alive or (not has_local_process and has_control_socket)
        if should_send:
            try:
                if has_control_socket:
                    self._send_control(
                        dp_rank,
                        self._ipc.control.encode(
                            "SHUTDOWN", None, **self._ipc.context(dp_rank)
                        ),
                    )
                    logger.debug(f"{self.label}: Sent shutdown to DP rank {dp_rank}")
                else:
                    logger.warning(
                        f"{self.label}: no usable control socket for DP rank "
                        f"{dp_rank}; shutdown not delivered"
                    )
            except Exception as e:  # noqa: BLE001 - teardown must not raise
                logger.debug(
                    f"{self.label}: Error sending shutdown to DP rank {dp_rank}: {e}"
                )

    def get_output(self) -> list[Sequence]:
        seqs = self.outputs_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    def is_rest(self):
        return not self.outputs_queue.empty()

    def is_alive(self):
        return any(
            proc is not None and proc.is_alive() for proc in self.engine_core_processes
        )

    def finished_procs(self):
        return any(
            proc is not None and not proc.is_alive()
            for proc in self.engine_core_processes
        )


def launch_engine_core(
    config: Config,
    dp_rank: int = 0,
    local_dp_rank: int | None = None,
    addresses: dict | None = None,
):
    if addresses is None:
        input_address = get_open_zmq_ipc_path()
        output_address = get_open_zmq_ipc_path()
        control_address = get_open_zmq_ipc_path()
    else:
        # Multi-node: TCP endpoints derived from the shared port plan. The
        # engine connects; the coordinator binds.
        input_address = addresses["input_address"]
        output_address = addresses["output_address"]
        control_address = addresses["control_address"]
    import torch

    # Imported here, not at module scope: EngineCore pulls the heavy
    # engine_core -> async_proc -> aiter chain. Spawning a worker is inherently a
    # GPU-side operation, so the cost belongs here and keeps CoreManager (routing
    # only) importable on a CPU-only runner.
    from atom.model_engine.engine_core import EngineCore

    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method("spawn", force=False)

    if local_dp_rank is None:
        local_dp_rank = dp_rank
    config.parallel_config.data_parallel_rank = dp_rank
    config.parallel_config.data_parallel_rank_local = local_dp_rank
    # Rides on the config rather than run_engine's signature, which every
    # EngineCore subclass would otherwise have to thread through.
    config.parallel_config.control_address = control_address

    # tp_world_size: the GPUs this DP rank really occupies.
    logger.info(
        f"Creating EngineCore process: global DP rank {dp_rank} "
        f"(local {local_dp_rank}), GPUs "
        f"{local_dp_rank * config.tp_world_size} to "
        f"{(local_dp_rank + 1) * config.tp_world_size - 1}"
    )

    process = multiprocessing.Process(
        target=EngineCore.run_engine,
        name=f"EngineCore-DP{dp_rank}",
        kwargs={
            "config": config,
            "input_address": input_address,
            "output_address": output_address,
        },
    )

    return (
        process,
        {
            "input_address": input_address,
            "output_address": output_address,
            "control_address": control_address,
        },
        local_dp_rank,
    )


class DisaggCoreManager(CoreManager):
    """CoreManager for intra-GPU prefill/decode disaggregation.

    Spawns two separate EngineCore processes on the same GPU(s):
      - PrefillEngineCore: runs prefill forward passes, writes KV cache.
      - DecodeEngineCore: owns BlockManager and KV cache, runs decode.

    add_request() fans out every new sequence to BOTH processes.
    Only DecodeEngineCore produces finished sequences back to LLMEngine.

    The two processes coordinate via direct ZMQ PUSH/PULL sockets whose
    addresses are established here before spawning and passed through config.
    """

    def __init__(self, config: Config):
        import copy

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        # Generate the inter-process ZMQ addresses before spawning.
        d2p_addr = get_open_zmq_ipc_path()  # decode → prefill (BlockAssignment)
        p2d_addr = get_open_zmq_ipc_path()  # prefill → decode (PrefillDone)
        # Bootstrap round 1: weight IPC handles (prefill → decode) + ACK (decode → prefill)
        weight_ipc_addr = get_open_zmq_ipc_path()
        weight_ack_addr = get_open_zmq_ipc_path()
        # Bootstrap round 2: kvcache handle + num_blocks (prefill → decode)
        kvcache_ipc_addr = get_open_zmq_ipc_path()

        # Shared memory for dynamic CU partitioning: 4 bytes (float32).
        # DecodeScheduler writes the chosen CU fraction; PrefillScheduler reads it.
        # 0.0 means no mask (None).
        # Only created in constrained mode; unconstrained mode runs prefill
        # and decode on plain separate streams with no CU coordination.
        if config.disagg_constrained:
            cu_shm_name = f"atom_cu_split_{os.getpid()}"
            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=cu_shm_name, create=True, size=4
            )
            self._cu_shm.buf[:4] = b"\x00" * 4
        else:
            cu_shm_name = ""
            self._cu_shm = None

        # Build per-process configs.
        prefill_config = copy.deepcopy(config)
        if config.disagg_prefill_max_num_seqs is not None:
            prefill_config.max_num_seqs = config.disagg_prefill_max_num_seqs
        prefill_config.enforce_eager = True
        prefill_config.disagg_d2p_addr = d2p_addr
        prefill_config.disagg_p2d_addr = p2d_addr
        prefill_config.disagg_weight_ipc_addr = weight_ipc_addr
        prefill_config.disagg_weight_ack_addr = weight_ack_addr
        prefill_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        prefill_config.disagg_cu_shm_name = cu_shm_name
        # Prefill gets its own server; decode honors the configured port.
        prefill_config.parallel_config.data_parallel_base_port = 0

        decode_config = copy.deepcopy(config)
        decode_config.disagg_d2p_addr = d2p_addr
        decode_config.disagg_p2d_addr = p2d_addr
        decode_config.disagg_weight_ipc_addr = weight_ipc_addr
        decode_config.disagg_weight_ack_addr = weight_ack_addr
        decode_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        decode_config.disagg_cu_shm_name = cu_shm_name
        # Decode allocates no GPU memory — kvcache and weights are imported from
        # prefill via CUDA IPC after prefill's READY signal.
        decode_config.disagg_is_decode = True

        if config.torch_profiler_dir:
            prefill_config.torch_profiler_dir = os.path.join(
                config.torch_profiler_dir, "prefill"
            )
            decode_config.torch_profiler_dir = os.path.join(
                config.torch_profiler_dir, "decode"
            )
            os.makedirs(prefill_config.torch_profiler_dir, exist_ok=True)
            os.makedirs(decode_config.torch_profiler_dir, exist_ok=True)

        # Addresses for the standard CoreManager input/output/control sockets.
        prefill_input_addr = get_open_zmq_ipc_path()
        prefill_output_addr = get_open_zmq_ipc_path()
        decode_input_addr = get_open_zmq_ipc_path()
        decode_output_addr = get_open_zmq_ipc_path()
        prefill_config.parallel_config.control_address = get_open_zmq_ipc_path()
        decode_config.parallel_config.control_address = get_open_zmq_ipc_path()

        from atom.model_engine.engine_core import DecodeEngineCore, PrefillEngineCore

        prefill_proc = multiprocessing.Process(
            target=PrefillEngineCore.run_engine,
            name="PrefillEngineCore",
            kwargs={
                "config": prefill_config,
                "input_address": prefill_input_addr,
                "output_address": prefill_output_addr,
            },
        )
        decode_proc = multiprocessing.Process(
            target=DecodeEngineCore.run_engine,
            name="DecodeEngineCore",
            kwargs={
                "config": decode_config,
                "input_address": decode_input_addr,
                "output_address": decode_output_addr,
            },
        )

        # Set up the inherited state without running CoreManager.__init__,
        # which would spawn its own engines the base way. This manager fans out
        # through its own add_request() and never charges DP load, but the
        # inherited output thread still releases it on every finished sequence.
        self._init_shared_state(
            config,
            label="DisaggCoreManager",
            local_engine_count=2,  # prefill + decode
        )

        import weakref

        def _connect_proc(proc, in_addr, out_addr, ctrl_addr, name):
            proc.start()
            self.engine_core_processes.append(proc)
            self._local_process_by_rank[len(self.engine_core_processes) - 1] = proc
            in_sock = make_zmq_socket(
                self.ctx,
                in_addr,
                zmq.ROUTER,
                bind=True,
                hwm=config.ipc_max_pending_jobs,
            )
            identity = self._receive_identity(in_sock)
            self.input_sockets.append(in_sock)
            self.engine_core_identities.append(identity)
            ctrl_sock = make_zmq_socket(
                self.ctx,
                ctrl_addr,
                zmq.ROUTER,
                bind=True,
                hwm=config.ipc_max_pending_jobs,
            )
            ctrl_identity = self._receive_identity(ctrl_sock)
            self.control_sockets.append(ctrl_sock)
            self.control_identities.append(ctrl_identity)
            out_sock = make_zmq_socket(
                self.ctx, out_addr, zmq.PULL, hwm=config.ipc_max_pending_jobs
            )
            self.output_sockets.append(out_sock)
            self.shutdown_paths.append(get_open_zmq_inproc_path())
            logger.info(f"{self.label}: {name} process started and connected")

        try:
            self._start_distributed_store(decode_config)
            self._start_distributed_store(prefill_config)
            # Start both processes simultaneously.  Prefill binds the bootstrap
            # PUSH socket and blocks on send() until decode connects and calls
            # recv() — they rendezvous naturally without any sequential ordering.
            _connect_proc(
                prefill_proc,
                prefill_input_addr,
                prefill_output_addr,
                prefill_config.parallel_config.control_address,
                "prefill",
            )
            _connect_proc(
                decode_proc,
                decode_input_addr,
                decode_output_addr,
                decode_config.parallel_config.control_address,
                "decode",
            )
            self._wait_for_single_ready(idx=0)
            self._wait_for_single_ready(idx=1)
            logger.info(f"{self.label}: both EngineCores ready")

            # Start output thread for decode only (index 1).
            # Prefill has a separate output thread just for READY/error monitoring.
            for idx, name in [(0, "prefill"), (1, "decode")]:
                t = self._create_output_thread(
                    idx, self.output_sockets[idx], self.shutdown_paths[idx]
                )
                t.start()
                self.output_threads.append(t)

            if self.finished_procs():
                raise RuntimeError("DisaggCoreManager: a process failed to start")

        except Exception:
            self.close()
            raise

        self._finalizer = weakref.finalize(self, self.close)
        self.async_output_queue = None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    def _wait_for_single_ready(self, idx):
        socket = self.output_sockets[idx]
        deadline = self._startup_deadline
        while not self._ipc.peers.get(idx, {}).get("ready"):
            if time.monotonic() >= deadline or self.finished_procs():
                raise RuntimeError(f"engine {idx} failed its IPC handshake")
            if socket.poll(100):
                self._handle_handshake(
                    idx, self._ipc.result.decode(socket.recv_multipart(copy=False))
                )

    # Tracing needs distinct span identities for the two halves first.
    _trace_unsupported = "pd"

    def _plan_requests(self, seqs):
        """Fan-out: every new sequence goes to BOTH prefill and decode."""
        from copy import copy
        from dataclasses import replace

        # For prefill: limit each sequence to 1 output token. Prefill discards
        # all sampled tokens (postprocess is a no-op), but max_tokens=1 ends
        # the forward after a single generate step. Decode admits and
        # allocates the blocks (`DecodeScheduler.allocate_waiting`), so it
        # alone gets the prompt hash chains.
        prefill = []
        for seq in seqs:
            sampling = copy(seq.sampling_params)
            sampling.max_tokens = 1
            prefill.append(
                replace(seq, sampling_params=sampling, prompt_hash_chain=None)
            )
        size = self._config.ipc_max_batch_size
        return [
            (rank, values[start : start + size])
            for rank, values in ((1, seqs), (0, prefill))
            for start in range(0, len(values), size)
        ]

    def _collective_rpc_unsupported(self) -> str | None:
        return "its two engines are the prefill and decode halves of one replica"

    def close(self):
        super().close()
        # Clean up dynamic CU partitioning shared memory (if created).
        if getattr(self, "_cu_shm", None) is not None:
            try:
                self._cu_shm.close()
                self._cu_shm.unlink()
            except (OSError, ValueError):
                logger.debug(
                    "Could not close CU partitioning shared memory", exc_info=True
                )
