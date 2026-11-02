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

Sequence ids stay unique across processes: worker k numbers its requests from
``(k + 1) << 24``, the hub from 0.
"""

from __future__ import annotations

import itertools
import logging
import os
import pickle
import socket
import tempfile
import threading
import uuid
from collections import defaultdict

import zmq

from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.prefix_hash import attach_shipped_chains
from atom.model_engine.sequence import Sequence

logger = logging.getLogger("atom")

# Worker k numbers its sequences from (k + 1) << _SEQ_ID_SHIFT; the hub keeps
# [0, 1 << _SEQ_ID_SHIFT). 16M requests per process, and 127 workers still fit
# an int32.
_SEQ_ID_SHIFT = 24

# The receive loops wake this often to notice close(); a socket may only be
# closed by the thread that last used it once that thread has stopped.
_POLL_MS = 200

# worker -> hub
_MSG_ADD = b"A"
_MSG_ABORT = b"X"
_MSG_HELLO = b"H"


def worker_seq_id_base(index: int) -> int:
    return (index + 1) << _SEQ_ID_SHIFT


def _ipc_path(tag: str) -> str:
    return f"ipc://{tempfile.gettempdir()}/atom-fe-{tag}-{uuid.uuid4().hex[:12]}"


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

    def __init__(self, core_mgr, num_workers: int):
        self.core_mgr = core_mgr
        self.num_workers = num_workers
        self.ctx = zmq.Context.instance()
        self.request_address = _ipc_path("req")
        self.output_addresses = [_ipc_path(f"out{i}") for i in range(num_workers)]
        self.admin_uds = (
            f"{tempfile.gettempdir()}/atom-fe-hub-{uuid.uuid4().hex[:12]}.sock"
        )

        self._request_socket = self.ctx.socket(zmq.PULL)
        self._request_socket.setsockopt(zmq.RCVHWM, 0)
        self._request_socket.bind(self.request_address)
        self._output_sockets = []
        self._output_locks = []
        for address in self.output_addresses:
            sock = self.ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.bind(address)
            self._output_sockets.append(sock)
            self._output_locks.append(threading.Lock())
        # The engine's output threads buffer here between a step's callbacks
        # and its flush: one list per worker, per thread.
        self._local = threading.local()
        self._callbacks = [self._make_callback(i) for i in range(num_workers)]
        self._closing = threading.Event()
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
        }

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
            with self._output_locks[worker_index]:
                self._output_sockets[worker_index].send(frame, copy=False)

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
                elif kind == _MSG_HELLO:
                    logger.info(
                        "Frontend worker %s connected", bytes(parts[1].buffer).decode()
                    )
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
        with self._output_locks[worker_index]:
            self._output_sockets[worker_index].send(frame, copy=False)

    def close(self) -> None:
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join()
        for sock in (self._request_socket, *self._output_sockets):
            sock.close(linger=0)


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
        self._send_socket = self.ctx.socket(zmq.PUSH)
        self._send_socket.setsockopt(zmq.SNDHWM, 0)
        self._send_socket.connect(info["request_address"])
        self._send_lock = threading.Lock()
        self._output_socket = self.ctx.socket(zmq.PULL)
        self._output_socket.setsockopt(zmq.RCVHWM, 0)
        self._output_socket.connect(info["output_address"])
        self._seq_id_to_callback: dict = {}
        self._flush_stream_batch_fn = None
        self._index_bytes = str(self.index).encode()
        self._closing = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_outputs,
            name=f"FrontendWorker{self.index}Outputs",
            daemon=True,
        )
        self._thread.start()
        self._send(_MSG_HELLO, str(os.getpid()).encode())

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

    def is_healthy(self) -> bool:
        return self._thread.is_alive()

    def _receive_outputs(self) -> None:
        sock = self._output_socket
        while not self._closing.is_set():
            if not sock.poll(_POLL_MS):
                continue
            frame = sock.recv(copy=False)
            try:
                outputs = pickle.loads(frame.buffer)
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
        Sequence.counter = itertools.count(worker_seq_id_base(info["index"]))

    def close(self) -> None:
        self.core_mgr.close()


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
