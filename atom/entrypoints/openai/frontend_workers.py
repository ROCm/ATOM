# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Run the OpenAI front end in several processes (``--api-server-count N``).

One API process does everything for a request on one event loop and one GIL:
reading the body, JSON and pydantic, the chat template, tokenizing, handing
the prompt to the engine and streaming the reply. Agentic clients resend the
whole conversation every turn, so that work grows with the context; past a
request rate the loop stops reading sockets in time, requests queue in the
kernel before ATOM sees them and the engine runs short of work.

With N > 1 the process that built the engine (the hub) keeps everything that
must stay single: the engine cores, the DP router (load, session affinity) and
the engine metrics. It serves only the management endpoints, on a Unix socket.
N worker processes share the public port (``SO_REUSEPORT``) and do the
per-request work. A worker hands a tokenized request to the hub; the hub
routes it like any other and sends each step's outputs back to the worker
that owns the request.

    client --HTTP--> worker k --ADD/ABORT (ZMQ)--> hub --> CoreManager --> engines
    client <--SSE--- worker k <--outputs (ZMQ)---- hub <--------------------'

Sequence ids stay unique across processes: worker k numbers its requests in
``[(k + 1) << 24, (k + 2) << 24)``, wrapping inside that window, and the hub
from 0.

The hub waits for every worker to bind the port before it serves, fails the
launch if one exits first, and reports 503 on ``/health`` if one dies later.
Workers publish their stream-silence and GC readings to a shared board that
the hub's ``/metrics`` reads, so those series cover every process.
"""

from __future__ import annotations

import gc
import itertools
import logging
import os
import pickle
import socket
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

import zmq

from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.prefix_hash import attach_shipped_chains
from atom.model_engine.sequence import Sequence
from atom.structured_output import is_structured
from atom.utils import get_open_zmq_ipc_path, make_zmq_socket

logger = logging.getLogger("atom")

# Worker k numbers its sequences in [(k + 1) << _SEQ_ID_SHIFT, (k + 2) <<
# _SEQ_ID_SHIFT), wrapping inside it; the hub keeps [0, 1 << _SEQ_ID_SHIFT).
# An id comes back only after 16M newer requests on the same worker, and 127
# workers still fit an int32.
_SEQ_ID_SHIFT = 24
_SEQ_ID_MASK = (1 << _SEQ_ID_SHIFT) - 1

# How long the hub waits for a worker to load its tokenizer and bind the port.
_WORKER_READY_TIMEOUT_S = 600.0

# The receive loops wake this often to notice close(); a socket may only be
# closed by the thread that last used it once that thread has stopped.
_POLL_MS = 200

# Structured-output admission compiles the grammar on the engine workers; a
# cold schema can take tens of seconds (the engine's own preflight allows 60).
_STRUCTURED_TIMEOUT_S = 120.0

# worker -> hub
_MSG_ADD = b"A"
_MSG_ABORT = b"X"
_MSG_HELLO = b"H"
_MSG_STRUCTURED = b"S"
# A hub send to a worker that stopped reading is dropped after this, so no
# hub thread (engine output, structured replies, close) can block on it.
_SEND_TIMEOUT_MS = 5000
# The queue that timeout applies to: messages the hub buffers per worker before
# a send waits (one per engine step, ~minutes of steps). With no limit the
# timeout never fires and the queue grows without bound.
_SEND_HWM = 20000

# hub -> worker (first frame)
_OUT_STEP = b"O"
_OUT_REPLY = b"R"


def worker_seq_id_base(index: int) -> int:
    return (index + 1) << _SEQ_ID_SHIFT


class worker_seq_ids:
    """Worker ``index``'s sequence ids, wrapping inside its own window.

    ``next()`` is safe from several threads (preprocessing runs on a pool): it
    is one ``next`` on an ``itertools.count`` plus local arithmetic, where a
    generator would raise "generator already executing"."""

    def __init__(self, index: int, start: int = 0):
        self._base = worker_seq_id_base(index)
        self._count = itertools.count(start)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        return self._base | (next(self._count) & _SEQ_ID_MASK)


def bind_shared_listener(host: str, port: int) -> socket.socket:
    """A listening TCP socket on host:port that other processes can bind too;
    the kernel spreads new connections over them."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


# --------------------------------------------------------------------------
# Hub (the process that owns the engine)
# --------------------------------------------------------------------------
class FrontendHub:
    """Takes requests from the workers into ``core_mgr`` and returns each
    step's outputs to the worker that sent the request."""

    def __init__(self, core_mgr, num_workers: int, prepare_structured=None):
        self.core_mgr = core_mgr
        self.num_workers = num_workers
        # ``LLMEngine.prepare_structured_request``: structured-output admission
        # (option checks, grammar preflight) needs the engine, so workers ask
        # the hub. Off the request thread, which every worker's traffic uses.
        self._prepare_structured = prepare_structured
        self._structured_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="FrontendHubStructured"
        )
        self.ctx = zmq.Context.instance()
        self.request_address = get_open_zmq_ipc_path()
        self.output_addresses = [get_open_zmq_ipc_path() for _ in range(num_workers)]
        # A filesystem path for uvicorn, not a ZMQ address.
        self.admin_uds = (
            f"{tempfile.gettempdir()}/atom-fe-hub-{uuid.uuid4().hex[:12]}.sock"
        )
        self.stats = FrontendStatsBoard(num_workers)
        self._ready: set[int] = set()
        self._ready_cond = threading.Condition()
        self._processes: list = []

        self._request_socket = make_zmq_socket(
            self.ctx, self.request_address, zmq.PULL, bind=True
        )
        self._output_sockets = []
        self._output_locks = []
        for address in self.output_addresses:
            sock = make_zmq_socket(self.ctx, address, zmq.PUSH, bind=True)
            sock.setsockopt(zmq.SNDHWM, _SEND_HWM)
            sock.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
            self._output_sockets.append(sock)
            self._output_locks.append(threading.Lock())
        # The engine's output threads buffer here between a step's callbacks
        # and its flush: one list per worker, per thread.
        self._local = threading.local()
        self._callbacks = [self._make_callback(i) for i in range(num_workers)]
        self._closing = threading.Event()
        self._sockets_closed = False
        self._thread = threading.Thread(
            target=self._serve_requests, name="FrontendHub", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def connection_info(self, worker_index: int) -> dict:
        return {
            "index": worker_index,
            "request_address": self.request_address,
            "output_address": self.output_addresses[worker_index],
            "admin_uds": self.admin_uds,
            "max_pool_tokens": getattr(self.core_mgr, "max_pool_tokens", None),
            "stats": self.stats,
        }

    # worker lifecycle ------------------------------------------------------
    def wait_for_workers(
        self, processes, timeout: float = _WORKER_READY_TIMEOUT_S
    ) -> None:
        """Block until every worker has bound the port. Raises if one exits
        first or the timeout passes: a launch whose public port is not served
        must fail, not report that it started."""
        self._processes = list(processes)
        deadline = time.monotonic() + timeout
        with self._ready_cond:
            while len(self._ready) < len(self._processes):
                for index, proc in enumerate(self._processes):
                    if index not in self._ready and proc.exitcode is not None:
                        raise RuntimeError(
                            f"front-end worker {index} exited with code "
                            f"{proc.exitcode} before it was serving"
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(range(len(self._processes))) - self._ready)
                    raise RuntimeError(
                        f"front-end workers {missing} not serving after "
                        f"{timeout:.0f} s"
                    )
                self._ready_cond.wait(min(remaining, 1.0))
        dead = self.dead_workers()
        if dead:
            raise RuntimeError(f"front-end workers {dead} exited after starting")

    def dead_workers(self) -> list[int]:
        return [
            index
            for index, proc in enumerate(self._processes)
            if proc.exitcode is not None
        ]

    # engine output threads -------------------------------------------------
    def _make_callback(self, worker_index: int):
        local = self._local

        def forward(request_output):
            pending = getattr(local, "pending", None)
            if pending is None:
                pending = local.pending = defaultdict(list)
            pending[worker_index].append((request_output.request_id, request_output))

        return forward

    def flush(self) -> None:
        """Send what this output thread buffered during the step, one message
        per worker. Called by the engine output thread after each step."""
        pending = getattr(self._local, "pending", None)
        if not pending:
            return
        self._local.pending = None
        for worker_index, outputs in pending.items():
            frame = pickle.dumps(outputs, protocol=pickle.HIGHEST_PROTOCOL)
            self._send_to_worker(worker_index, _OUT_STEP, frame)

    def _send_to_worker(self, worker_index: int, *parts) -> None:
        with self._output_locks[worker_index]:
            # close() takes this lock before closing the socket; nothing is
            # sent after that, from any thread.
            if self._sockets_closed:
                return
            try:
                self._output_sockets[worker_index].send_multipart(parts, copy=False)
            except zmq.Again:
                logger.warning(
                    "Frontend hub: worker %d not reading for %d ms; message dropped",
                    worker_index,
                    _SEND_TIMEOUT_MS,
                )

    def _answer_structured(self, worker_index: int, call_id: bytes, frame) -> None:
        try:
            params, kv_transfer, multimodal = pickle.loads(frame)
            if self._prepare_structured is None:
                raise ValueError("Structured output requires engine admission")
            self._prepare_structured(params, kv_transfer, multimodal)
            error = None
        except Exception as exc:  # noqa: BLE001 - goes back to the request
            error = exc
        try:
            reply = pickle.dumps(error, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - an unpicklable error still answers
            reply = pickle.dumps(ValueError(str(error)))
        self._send_to_worker(worker_index, _OUT_REPLY, call_id, reply)

    # worker requests -------------------------------------------------------
    def _serve_requests(self) -> None:
        sock = self._request_socket
        while not self._closing.is_set():
            if not sock.poll(_POLL_MS):
                continue
            parts = sock.recv_multipart(copy=False)
            kind = bytes(parts[0].buffer)
            try:
                if kind == _MSG_ADD:
                    worker_index = int(bytes(parts[1].buffer))
                    _, seqs, *rest = pickle.loads(parts[2].buffer)
                    attach_shipped_chains(seqs, rest[0] if rest else None)
                    callback = self._callbacks[worker_index]
                    for seq in seqs:
                        seq.stream_callback = callback
                    try:
                        self.core_mgr.add_request(seqs)
                    except Exception:
                        # The worker's client is waiting on these ids: end them.
                        logger.exception("Frontend hub: request rejected")
                        self._reject(worker_index, seqs)
                elif kind == _MSG_ABORT:
                    self.core_mgr.abort_request(int(bytes(parts[1].buffer)))
                elif kind == _MSG_STRUCTURED:
                    self._structured_pool.submit(
                        self._answer_structured,
                        int(bytes(parts[1].buffer)),
                        bytes(parts[2].buffer),
                        bytes(parts[3].buffer),
                    )
                elif kind == _MSG_HELLO:
                    worker_index = int(bytes(parts[1].buffer))
                    logger.info(
                        "Frontend worker %d (pid %s) serving",
                        worker_index,
                        bytes(parts[2].buffer).decode(),
                    )
                    with self._ready_cond:
                        self._ready.add(worker_index)
                        self._ready_cond.notify_all()
            except Exception:
                # One bad request must not take every worker's traffic with it.
                logger.exception("Frontend hub: failed to handle a %r message", kind)

    def _reject(self, worker_index: int, seqs) -> None:
        from atom.model_engine.request import RequestOutput

        # CoreManager registers callbacks before it validates the batch.
        registered = getattr(self.core_mgr, "_seq_id_to_callback", {})
        outputs = []
        for seq in seqs:
            registered.pop(seq.id, None)
            outputs.append(
                (
                    seq.id,
                    RequestOutput(
                        request_id=seq.id,
                        output_tokens=[],
                        finished=True,
                        finish_reason="abort",
                    ),
                )
            )
        frame = pickle.dumps(outputs, protocol=pickle.HIGHEST_PROTOCOL)
        self._send_to_worker(worker_index, _OUT_STEP, frame)

    def close(self) -> None:
        """Stop taking requests, then the structured-admission pool, then the
        workers, then the sockets. Call after the engine has stopped calling
        ``flush``. Never blocks on a worker: a reply still being prepared is
        dropped once the sockets are closed."""
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join()
        self._structured_pool.shutdown(wait=False, cancel_futures=True)
        for proc in self._processes:
            if proc.exitcode is None:
                proc.terminate()
        for lock in self._output_locks:
            lock.acquire()
        try:
            self._sockets_closed = True
            for sock in (self._request_socket, *self._output_sockets):
                sock.close(linger=0)
        finally:
            for lock in self._output_locks:
                lock.release()


# --------------------------------------------------------------------------
# Worker (one of the processes serving the public port)
# --------------------------------------------------------------------------
class RemoteCoreManager:
    """Stands in for ``CoreManager`` in a worker: requests go to the hub, and
    the hub's outputs reach the same per-request callbacks."""

    def __init__(self, info: dict):
        self.index = info["index"]
        self.max_pool_tokens = info.get("max_pool_tokens")
        self.ctx = zmq.Context.instance()
        self._send_socket = make_zmq_socket(
            self.ctx, info["request_address"], zmq.PUSH, bind=False
        )
        self._send_lock = threading.Lock()
        self._output_socket = make_zmq_socket(
            self.ctx, info["output_address"], zmq.PULL, bind=False
        )
        self._seq_id_to_callback: dict = {}
        self._flush_stream_batch_fn = None
        self._index_bytes = str(self.index).encode()
        self._call_ids = itertools.count()
        self._calls: dict[bytes, Future] = {}
        self._closing = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_outputs,
            name=f"FrontendWorker{self.index}Outputs",
            daemon=True,
        )
        self._thread.start()

    def announce_ready(self) -> None:
        """Tell the hub this worker has bound the public port."""
        self._send(_MSG_HELLO, self._index_bytes, str(os.getpid()).encode())

    def _send(self, *parts) -> None:
        with self._send_lock:
            self._send_socket.send_multipart(parts, copy=False)

    def add_request(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None
        self._send(_MSG_ADD, self._index_bytes, _add_frame(seqs))

    def abort_request(self, req_id) -> None:
        self._seq_id_to_callback.pop(req_id, None)
        self._send(_MSG_ABORT, str(int(req_id)).encode())

    def prepare_structured_request(self, params, kv_transfer=None, multimodal=False):
        """``LLMEngine.prepare_structured_request``, answered by the hub's
        engine; its error, if any, is raised here as the engine raised it."""
        if not is_structured(params):
            return
        call_id = str(next(self._call_ids)).encode()
        reply: Future = Future()
        self._calls[call_id] = reply
        try:
            self._send(
                _MSG_STRUCTURED,
                self._index_bytes,
                call_id,
                pickle.dumps(
                    (params, kv_transfer, multimodal),
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )
            error = reply.result(timeout=_STRUCTURED_TIMEOUT_S)
        finally:
            self._calls.pop(call_id, None)
        if error is not None:
            raise error

    def is_healthy(self) -> bool:
        return self._thread.is_alive()

    def _receive_outputs(self) -> None:
        sock = self._output_socket
        while not self._closing.is_set():
            if not sock.poll(_POLL_MS):
                continue
            parts = sock.recv_multipart(copy=False)
            try:
                if bytes(parts[0].buffer) == _OUT_REPLY:
                    reply = self._calls.get(bytes(parts[1].buffer))
                    if reply is not None:
                        reply.set_result(pickle.loads(parts[2].buffer))
                    continue
                outputs = pickle.loads(parts[1].buffer)
            except Exception:
                logger.exception("Frontend worker %d: undecodable output", self.index)
                continue
            for seq_id, request_output in outputs:
                callback = (
                    self._seq_id_to_callback.pop(seq_id, None)
                    if request_output.finished
                    else self._seq_id_to_callback.get(seq_id)
                )
                if callback is None:
                    continue
                try:
                    callback(request_output)
                except Exception as e:
                    logger.warning(
                        f"Error calling stream_callback for sequence {seq_id}: {e}",
                        exc_info=True,
                    )
            if self._flush_stream_batch_fn is not None:
                try:
                    self._flush_stream_batch_fn()
                except Exception as e:
                    logger.warning(f"flush_stream_batch failed: {e}", exc_info=True)

    def close(self) -> None:
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join()
        with self._send_lock:
            self._send_socket.close(linger=0)
        self._output_socket.close(linger=0)


class FrontendEngine:
    """The parts of ``LLMEngine`` a worker's request path uses: the hub's
    config, a local tokenizer/preprocessor, and a ``RemoteCoreManager``."""

    def __init__(self, config, tokenizer, info: dict):
        from atom.model_engine.llm_engine import InputOutputProcessor

        self.config = config
        self.tokenizer = tokenizer
        self.io_processor = InputOutputProcessor(
            config, tokenizer, config.kv_cache_block_size
        )
        self.core_mgr = RemoteCoreManager(info)
        # Structured output is admitted by the hub's engine, here and in
        # preprocess (``InputOutputProcessor.structured_admission``).
        self.prepare_structured_request = self.core_mgr.prepare_structured_request
        self.io_processor.structured_admission = self.prepare_structured_request
        Sequence.counter = worker_seq_ids(info["index"])

    def close(self) -> None:
        self.core_mgr.close()


class FrontendStatsBoard:
    """Per-worker readings that the hub's ``/metrics`` exports for every
    process: when the oldest waiting stream started waiting, and the GC
    counters.

    Shared memory, written by each worker on its metrics cadence. The wait
    start is a ``time.monotonic()`` stamp (CLOCK_MONOTONIC, one clock for
    every process on the host), turned into a silence at scrape time, so a
    worker whose loop is stuck still shows a growing silence.
    """

    _GENERATIONS = 3
    _GC_KEYS = ("collections", "collected", "uncollectable")
    # waiting_since, then gc stats (generation x key), then thresholds
    _SLOTS = 1 + _GENERATIONS * len(_GC_KEYS) + _GENERATIONS

    def __init__(self, num_workers: int):
        import multiprocessing

        self.num_workers = num_workers
        self._values = multiprocessing.get_context("spawn").RawArray(
            "d", num_workers * self._SLOTS
        )

    def publish(self, index: int, waiting_since: float | None) -> None:
        values = [0.0 if waiting_since is None else waiting_since]
        stats = gc.get_stats()
        for gen in range(self._GENERATIONS):
            values += [float(stats[gen].get(key, 0)) for key in self._GC_KEYS]
        values += [float(v) for v in gc.get_threshold()[: self._GENERATIONS]]
        row = index * self._SLOTS
        self._values[row : row + self._SLOTS] = values

    def longest_silence(self, now: float | None = None) -> float:
        starts = [
            self._values[i * self._SLOTS]
            for i in range(self.num_workers)
            if self._values[i * self._SLOTS] > 0
        ]
        if not starts:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, now - min(starts))

    def gc_readings(self) -> list[tuple[str, list[dict], tuple[float, ...]]]:
        """``(process, gc.get_stats()-shaped list, thresholds)`` per worker."""
        n_keys = len(self._GC_KEYS)
        rows = []
        for i in range(self.num_workers):
            row = self._values[i * self._SLOTS : (i + 1) * self._SLOTS]
            stats = [
                dict(zip(self._GC_KEYS, row[1 + g * n_keys : 1 + (g + 1) * n_keys]))
                for g in range(self._GENERATIONS)
            ]
            rows.append((f"worker{i}", stats, tuple(row[-self._GENERATIONS :])))
        return rows


def watch_parent(parent_pid: int, interval_s: float = 2.0) -> None:
    """Exit this worker if the hub that spawned it is gone."""

    def watch():
        import time

        while True:
            time.sleep(interval_s)
            if os.getppid() != parent_pid:
                logger.warning("Frontend hub %d exited; worker exiting", parent_pid)
                os._exit(1)

    threading.Thread(target=watch, name="FrontendParentWatch", daemon=True).start()
