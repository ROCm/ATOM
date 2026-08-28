# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import hashlib
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import pickle
import queue
import weakref
from dataclasses import dataclass
from threading import Lock, Thread

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.utils import (
    envs,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
)

logger = logging.getLogger("atom")

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
        self._closed = False  # Track whether already closed
        self.local_engine_count = local_engine_count
        self.global_engine_count = (
            local_engine_count if global_engine_count is None else global_engine_count
        )
        self.ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue[list[Sequence]]()
        self.utility_response_queue = queue.Queue()
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
                        self.ctx, addresses["input_address"], zmq.ROUTER, bind=True
                    )
                    identity, _ = input_socket.recv_multipart()
                    self.input_sockets.append(input_socket)
                    self.engine_core_identities.append(identity)

                    control_socket = make_zmq_socket(
                        self.ctx, addresses["control_address"], zmq.ROUTER, bind=True
                    )
                    control_identity, _ = control_socket.recv_multipart()
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
        """Wait for READY signals from all DP ranks in parallel (no timeout)."""
        poller = zmq.Poller()
        for dp_rank, output_socket in enumerate(self.output_sockets):
            poller.register(output_socket, zmq.POLLIN)

        engine_count = len(self.output_sockets)
        ready_received = [False] * engine_count
        remaining = engine_count

        while remaining > 0:
            socks = poller.poll()  # Wait indefinitely
            if not socks:
                continue

            for socket, _ in socks:
                # Find which DP rank this socket belongs to
                dp_rank = self.output_sockets.index(socket)
                if ready_received[dp_rank]:
                    continue

                obj = socket.recv(copy=False)
                request_type, data = pickle.loads(obj)

                if request_type == EngineCoreRequestType.READY:
                    logger.info(
                        f"{self.label}: DP rank {dp_rank} is fully initialized and ready"
                    )
                    self._record_ready_payload(data)
                    ready_received[dp_rank] = True
                    remaining -= 1
                elif request_type == EngineCoreRequestType.SHUTDOWN:
                    raise RuntimeError(
                        f"{self.label}: Received unexpected SHUTDOWN signal from DP rank {dp_rank} during initialization"
                    )
                else:
                    raise RuntimeError(
                        f"{self.label}: Expected READY signal from DP rank {dp_rank}, but got {request_type}"
                    )

    def _create_output_thread(
        self, dp_rank: int, output_socket: zmq.Socket, shutdown_path: str
    ) -> Thread:
        def process_outputs_socket():
            assert isinstance(output_socket, zmq.Socket)
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(output_socket, zmq.POLLIN)
                logger.debug(f"{self.label} (DP {dp_rank}): output thread started")
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive shutdown signal"
                        )
                        break

                    obj = output_socket.recv(copy=False)
                    request_type, data = pickle.loads(obj)
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
                        self.utility_response_queue.put_nowait(data)
                    elif request_type == EngineCoreRequestType.ADD:
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        seqs = data
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
                        self.outputs_queue.put_nowait(seqs)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
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
        self._closed = True

        logger.info(
            f"{self.label}: Shutting down {len(self.input_sockets)} EngineCores"
        )

        for dp_rank in range(len(self.input_sockets)):
            self._shutdown_engine_core_rank(dp_rank)

        for input_socket in self.input_sockets:
            if not input_socket.closed:
                input_socket.close()

        for control_socket in self.control_sockets:
            if not control_socket.closed:
                control_socket.close()

        for shutdown_path in self.shutdown_paths:
            if shutdown_path:
                try:
                    with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                        shutdown_sender.connect(shutdown_path)
                        shutdown_sender.send(b"")
                except Exception as e:
                    logger.debug(f"{self.label}: Error sending shutdown signal: {e}")

        for thread in self.output_threads:
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

        # Wait for EngineCore processes to exit gracefully.
        # Use a single deadline so all processes share the grace period
        # instead of sequential per-process timeouts.  This prevents early
        # process exits from destroying the NCCL TCPStore while later
        # processes' HeartbeatMonitor threads still depend on it.
        import time

        deadline = time.monotonic() + 5
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

        # Final join + close to release sentinel semaphores
        for proc in self.engine_core_processes:
            if proc is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=1)
                try:
                    proc.close()
                except (ValueError, OSError):
                    pass

        logger.info(f"{self.label}: All EngineCores shut down")

    def _send_request(self, dp_rank: int, payload: bytes) -> None:
        """Send one already-pickled request to an engine core. Hot path.

        Deliberately unsynchronized: ``input_sockets`` carries nothing but
        ``add_request``, which runs on a single thread (the API server's event
        loop online, the caller's thread offline). Everything that can be sent
        from another thread -- utility commands, abort, shutdown -- goes to
        :meth:`_send_control` on a separate socket instead.

        That separation is load-bearing, not stylistic. A ZMQ socket is not
        thread-safe: two unserialized ``send_multipart`` calls interleave their
        frames, the DEALER on the other end then reads a routing identity where
        a payload should be, and its input thread dies on ``UnpicklingError``.
        Nothing recovers from that -- the engine spins on a forever-empty input
        queue, the workers idle, and every client hangs with no error logged
        anywhere but that thread's own traceback. So: never send to
        ``input_sockets`` from anywhere but here.
        """
        self.input_sockets[dp_rank].send_multipart(
            [self.engine_core_identities[dp_rank], payload], copy=False
        )

    def _send_control(self, dp_rank: int, payload: bytes, copy: bool = False) -> None:
        """Send one already-pickled control message. Serialized, never hot.

        Writers here are the event loop (abort, the /debug/* endpoints) and the
        per-rank output threads (shutdown), so this does need a lock -- but none
        of them runs per request, so its cost never lands on admission.
        """
        with self._control_send_lock:
            self.control_sockets[dp_rank].send_multipart(
                [self.control_identities[dp_rank], payload], copy=copy
            )

    def add_request(self, seqs: list[Sequence]):
        logger.debug(
            f"{self.label}: Add request, sequence ids: {[seq.id for seq in seqs]}"
        )
        # Register callbacks before sending to engine core
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None
        if self.pp_size > 1:
            # Pipeline parallel (dp=1): requests enter only at stage 0, which
            # drives the pipeline downstream.
            logger.debug(f"{self.label}: Add {len(seqs)} requests to PP head 0")
            self._send_request(0, pickle.dumps((EngineCoreRequestType.ADD, seqs)))
        elif self._routable_engine_count == 1:
            # Single routable engine, send all requests
            logger.debug(f"{self.label}: Add {len(seqs)} requests to DP rank 0")
            self._send_request(0, pickle.dumps((EngineCoreRequestType.ADD, seqs)))
        else:
            self._dispatch_to_dp_ranks(seqs)

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

    def _dispatch_to_dp_ranks(self, seqs: list[Sequence]) -> None:
        """Route a batch across DP ranks and send each rank its sub-batch.

        Honors an explicit ``data_parallel_rank`` hint; otherwise picks a rank
        via ``_select_dp_rank_locked`` (load-aware by default). Selection and the
        in-flight-load charge happen atomically under ``_lb_lock`` so a burst of
        requests spreads across ranks instead of all landing on the current
        minimum.
        """
        # Resolve + validate all hints in one pass first — no charging until the
        # whole batch is known good, so a rejected batch never leaks partial
        # load. The resolved hints are reused in the loop below to avoid a second
        # getattr/int pass per seq.
        hints = self._resolve_and_validate_hints(seqs)

        # round_robin normally skips load bookkeeping. Session affinity still
        # needs queued-prefill counters even if the fallback strategy is RR.
        track_load = (
            self._dp_lb_strategy != "round_robin" or self._dp_session_affinity_enabled
        )
        engine_count = self._routable_engine_count
        dp_seqs = [[] for _ in range(engine_count)]
        reqs_snapshot = tokens_snapshot = None
        with self._lb_lock:
            for seq, hint in zip(seqs, hints):
                dp_rank = self._select_dp_rank_for_seq_locked(seq, hint)
                if track_load:
                    self._charge_seq_load_locked(seq, dp_rank)
                dp_seqs[dp_rank].append(seq)
            # Copy the counters under the lock so the snapshot log below is a
            # consistent instant, not a torn read racing _release_seq_load.
            if track_load:
                reqs_snapshot = list(self._rank_reqs)
                tokens_snapshot = list(self._rank_tokens)

        # Track which ranks were actually handed off, plus a compact per-rank
        # delta ("rankR:Nreq/Ttok") for the single summary log after the loop. If
        # a send fails partway, the seqs on the not-yet-dispatched ranks were
        # charged above but will never produce a finished output to release them,
        # so we roll back their in-flight load before propagating — otherwise
        # routing skews forever.
        dispatched = [False] * engine_count
        added = []
        try:
            for dp_rank, rank_seqs in enumerate(dp_seqs):
                if not rank_seqs:
                    continue
                self._send_request(
                    dp_rank, pickle.dumps((EngineCoreRequestType.ADD, rank_seqs))
                )
                dispatched[dp_rank] = True
                batch_prefill_tokens = sum(
                    int(getattr(seq, "num_prompt_tokens", 0) or 0) for seq in rank_seqs
                )
                added.append(
                    f"rank{dp_rank}: {len(rank_seqs)} req / {batch_prefill_tokens} tok"
                )
        except Exception:
            # Roll back only ranks we never handed off. _release_seq_load is
            # idempotent (pops from _seq_load), so even if a failing send had
            # already delivered its frames and the engine finished + released
            # those seqs on an output thread, this rollback cannot double-count:
            # whichever release runs first wins, the other is a no-op.
            if track_load:
                for dp_rank, rank_seqs in enumerate(dp_seqs):
                    if rank_seqs and not dispatched[dp_rank]:
                        for seq in rank_seqs:
                            self._release_seq_load(seq.id)
            raise

        # One line per add: the per-rank delta this add placed, plus (for the
        # load-aware strategies) the resulting in-flight distribution across all
        # ranks, so a single grep shows both what changed and how balanced it is.
        if reqs_snapshot is not None:
            logger.info(
                "%s: add %s | in-flight reqs=%s queued_prefill_tokens=%s",
                self.label,
                ", ".join(added),
                reqs_snapshot,
                tokens_snapshot,
            )
        else:
            logger.info("%s: add %s", self.label, ", ".join(added))

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

    def send_utility_command(self, cmd: str, dp_rank: int | None = None):
        if dp_rank is None:
            # Send to all DP ranks
            for rank in range(len(self.control_sockets)):
                logger.debug(
                    f"{self.label}: Send utility command '{cmd}' to DP rank {rank}"
                )
                self._send_control(
                    rank, pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd}))
                )
        else:
            logger.debug(
                f"{self.label}: Send utility command '{cmd}' to DP rank {dp_rank}"
            )
            self._send_control(
                dp_rank, pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd}))
            )

    def abort_request(self, req_id):
        """Tell the engine core(s) to drop a request (client disconnected).

        Broadcast to every DP rank (only the one holding ``req_id`` acts). The
        scheduler finishes the seq at its next step via the normal stop path,
        freeing its KV blocks. Fire-and-forget; safe if the seq already finished.
        """
        # Release DP load bookkeeping now: an aborted seq may never emit a
        # finished STREAM output, so relying on the finish path alone would leak
        # its in-flight count. _release_seq_load is idempotent.
        self._release_seq_load(req_id)
        try:
            self.broadcast_utility_command("abort_request", req_id=req_id)
        except Exception as e:
            logger.warning(f"{self.label}: abort_request({req_id}) failed: {e}")

    def broadcast_utility_command(self, cmd: str, **kwargs):
        payload = {"cmd": cmd, **kwargs}
        # Serialize once and reuse for all ranks (optimization: avoid repeated pickle.dumps)
        serialized_payload = pickle.dumps((EngineCoreRequestType.UTILITY, payload))
        for rank in range(len(self.control_sockets)):
            logger.debug(
                f"{self.label}: Broadcast utility command '{cmd}' to DP rank {rank}"
            )
            # copy=True: the same buffer is reused for every rank.
            self._send_control(rank, serialized_payload, copy=True)

    def broadcast_utility_command_sync(
        self, cmd: str, timeout: float = 300.0, **kwargs
    ):
        # Drain any stale responses that might be left over
        while not self.utility_response_queue.empty():
            try:
                self.utility_response_queue.get_nowait()
            except queue.Empty:
                break

        self.broadcast_utility_command(cmd, **kwargs)

        # Collect one response per routable engine (must match the broadcast count
        # len(self.control_sockets), which is the global engine count on a coordinator).
        responses = []
        for _ in range(len(self.control_sockets)):
            try:
                resp = self.utility_response_queue.get(timeout=timeout)
                responses.append(resp)
            except queue.Empty:
                raise TimeoutError(
                    f"{self.label}: Timed out waiting for UTILITY_RESPONSE "
                    f"for command '{cmd}' (timeout={timeout}s)"
                )
        return responses

    def _shutdown_engine_core_rank(self, dp_rank: int):
        # Determine whether this rank has a local process and/or a control socket.
        # On a coordinator with remote engines, dp_rank may exceed the local
        # process list -- but we still hold a live control socket for it, and
        # the remote engine must receive SHUTDOWN so its worker node unblocks.
        has_local_process = dp_rank < len(self.engine_core_processes)
        process = self.engine_core_processes[dp_rank] if has_local_process else None
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
                        dp_rank, pickle.dumps((EngineCoreRequestType.SHUTDOWN, None))
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
        from atom.utils import get_open_port as _get_open_port

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
        # Give prefill a distinct distributed rendezvous port so it doesn't
        # collide with decode's data_parallel_base_port (both deep-copy the
        # same port from config).
        prefill_config.parallel_config.data_parallel_base_port = _get_open_port()

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
            in_sock = make_zmq_socket(self.ctx, in_addr, zmq.ROUTER, bind=True)
            identity, _ = in_sock.recv_multipart()
            self.input_sockets.append(in_sock)
            self.engine_core_identities.append(identity)
            ctrl_sock = make_zmq_socket(self.ctx, ctrl_addr, zmq.ROUTER, bind=True)
            ctrl_identity, _ = ctrl_sock.recv_multipart()
            self.control_sockets.append(ctrl_sock)
            self.control_identities.append(ctrl_identity)
            out_sock = make_zmq_socket(self.ctx, out_addr, zmq.PULL)
            self.output_sockets.append(out_sock)
            self.shutdown_paths.append(get_open_zmq_inproc_path())
            logger.info(f"{self.label}: {name} process started and connected")

        try:
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

    def _wait_for_single_ready(self, idx: int):
        """Block until output_sockets[idx] sends a READY signal."""
        sock = self.output_sockets[idx]
        while True:
            obj = sock.recv(copy=False)
            request_type, data = pickle.loads(obj)
            if request_type == EngineCoreRequestType.READY:
                self._record_ready_payload(data)
                return
            if request_type == EngineCoreRequestType.SHUTDOWN:
                raise RuntimeError(
                    f"{self.label}: process {idx} sent SHUTDOWN during initialization"
                )

    def add_request(self, seqs: list[Sequence]):
        """Fan-out: send every new sequence to BOTH prefill and decode."""
        logger.debug(f"{self.label}: fan-out {len(seqs)} seqs to prefill and decode")
        # Register stream callbacks before sending (decode will produce output).
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None

        # Send decode payload as-is.
        decode_payload = pickle.dumps((EngineCoreRequestType.ADD, seqs))
        self._send_request(1, decode_payload)

        # For prefill: limit each sequence to 1 output token.  Prefill discards
        # all sampled tokens (postprocess is a no-op), but setting max_tokens=1
        # ensures the forward pass terminates after a single generate step and
        # that num_scheduled_tokens correctly reflects only the prompt tokens.
        import copy as _copy

        prefill_seqs = []
        for seq in seqs:
            ps = _copy.copy(seq)
            ps.max_tokens = 1
            prefill_seqs.append(ps)
        prefill_payload = pickle.dumps((EngineCoreRequestType.ADD, prefill_seqs))
        self._send_request(0, prefill_payload)

    def close(self):
        super().close()
        # Clean up dynamic CU partitioning shared memory (if created).
        if getattr(self, "_cu_shm", None) is not None:
            try:
                self._cu_shm.close()
                self._cu_shm.unlink()
            except Exception:
                pass
