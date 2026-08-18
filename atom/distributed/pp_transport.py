# SPDX-License-Identifier: MIT
# Pipeline-parallel inter-stage CPU transport.
#
# Each PP stage runs as its own EngineCore process. Three control channels run
# over ZMQ (CPU); the hidden-state tensors themselves go GPU-to-GPU over NCCL
# (see pp_comm.py), never here.
#
#   metadata   (head -> every downstream stage): the scheduled batch to run.
#   tokens     (last stage -> head):             sampled ScheduledBatchOutput,
#                                                fed back so the head owns the
#                                                request lifecycle / next step.
#   kv_status  (every downstream stage -> head): KV offload load/save completion
#                                                status, so the head can
#                                                aggregate across all PP stages.
#
# The objects moved here (ScheduledBatch, ScheduledBatchOutput, KVConnectorOutput)
# are the same ones already pickled to broadcast to workers, so no bespoke wire
# format is needed — pickle round-trips them verbatim.
#
# Bind/connect convention: the RECEIVER binds, the SENDER connects (PUSH/PULL,
# so connect-before-bind is fine; ZMQ queues at the sender).

import logging
import pickle
from typing import Any

import zmq

logger = logging.getLogger("atom")


class PPStageTransport:
    """Per-stage ZMQ endpoints for the head<->downstream metadata/token/kv_status channels.

    Args:
        pp_rank: this stage's index (0 = head).
        pp_size: number of pipeline stages.
        meta_addrs: length pp_size; meta_addrs[s] is the endpoint on which stage
            s RECEIVES metadata from the head (index 0 is unused). The head
            connects a PUSH socket to each downstream endpoint; each downstream
            stage binds a PULL socket to its own endpoint.
        token_addr: endpoint on which the head RECEIVES tokens from the last
            stage. The head binds a PULL socket; the last stage connects PUSH.
        kv_status_addr: endpoint on which the head RECEIVES KV offload status
            from ALL downstream stages. The head binds a PULL socket; every
            downstream stage connects a PUSH socket. Empty string = disabled.
        ctx: optional shared zmq.Context (one is created if omitted).
    """

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        meta_addrs: list[str],
        token_addr: str,
        kv_status_addr: str = "",
        ctx: zmq.Context | None = None,
    ):
        assert pp_size >= 2, "PPStageTransport is only used when pp_size >= 2"
        assert len(meta_addrs) == pp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_head = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1
        self._ctx = ctx if ctx is not None else zmq.Context()
        self._owns_ctx = ctx is None

        self._meta_send: list[zmq.Socket] = []
        self._meta_recv: zmq.Socket | None = None
        self._token_recv: zmq.Socket | None = None
        self._token_send: zmq.Socket | None = None
        self._kv_status_recv: zmq.Socket | None = None
        self._kv_status_send: zmq.Socket | None = None

        if self.is_head:
            # One PUSH per downstream stage (metadata fan-out).
            for s in range(1, pp_size):
                sock = self._ctx.socket(zmq.PUSH)
                sock.connect(meta_addrs[s])
                self._meta_send.append(sock)
            # Receive sampled tokens back from the last stage.
            self._token_recv = self._ctx.socket(zmq.PULL)
            self._token_recv.bind(token_addr)
            # Receive KV offload status from all downstream stages.
            if kv_status_addr:
                self._kv_status_recv = self._ctx.socket(zmq.PULL)
                self._kv_status_recv.bind(kv_status_addr)
        else:
            # Receive the scheduled batch from the head.
            self._meta_recv = self._ctx.socket(zmq.PULL)
            self._meta_recv.bind(meta_addrs[pp_rank])
            if self.is_last:
                self._token_send = self._ctx.socket(zmq.PUSH)
                self._token_send.connect(token_addr)
            # Send KV offload status back to the head.
            if kv_status_addr:
                self._kv_status_send = self._ctx.socket(zmq.PUSH)
                self._kv_status_send.connect(kv_status_addr)

    # ---- head side ----------------------------------------------------------
    def send_metadata(self, batch: Any) -> None:
        """Head: broadcast the scheduled batch to every downstream stage."""
        payload = pickle.dumps(batch)
        for sock in self._meta_send:
            sock.send(payload, copy=False)

    def recv_tokens(self, timeout_ms: int | None = None) -> Any:
        """Head: block for the last stage's sampled ScheduledBatchOutput.

        Returns None on timeout.
        """
        if timeout_ms is not None and not self._token_recv.poll(timeout_ms):
            return None
        return pickle.loads(self._token_recv.recv())

    def recv_kv_status(self, timeout_ms: int = 0) -> list[tuple[int, Any]]:
        """Head: drain all pending KV offload status messages.

        Returns a (possibly empty) list of ``(pp_rank, KVConnectorOutput)``
        tuples.  Non-blocking by default (``timeout_ms=0``).
        """
        if self._kv_status_recv is None:
            return []
        results: list[tuple[int, Any]] = []
        while self._kv_status_recv.poll(timeout_ms):
            pp_rank, output = pickle.loads(self._kv_status_recv.recv())
            results.append((pp_rank, output))
            timeout_ms = 0
        return results

    def recv_completion(self, timeout_ms: int | None = None) -> Any:
        """Head: receive ``(req_ids, output)`` after the last stage finishes."""
        return self.recv_tokens(timeout_ms)

    # ---- downstream / last side --------------------------------------------
    def recv_metadata(self, timeout_ms: int | None = None) -> Any:
        """Downstream: block for the head's scheduled batch."""
        if timeout_ms is not None and not self._meta_recv.poll(timeout_ms):
            return None
        return pickle.loads(self._meta_recv.recv())

    def send_tokens(self, out: Any) -> None:
        """Last stage: feed the sampled output back to the head."""
        assert self._token_send is not None, "send_tokens only valid on last stage"
        self._token_send.send(pickle.dumps(out), copy=False)

    def send_kv_status(self, output: Any) -> None:
        """Downstream: send KV offload completion status to the head."""
        if self._kv_status_send is None:
            return
        self._kv_status_send.send(pickle.dumps((self.pp_rank, output)), copy=False)

    def send_completion(self, req_ids: Any, out: Any = None) -> None:
        """Last stage: acknowledge completion, with sampled output if present."""
        self.send_tokens((tuple(req_ids), out))

    def close(self) -> None:
        for sock in self._meta_send:
            sock.close(linger=0)
        for sock in (
            self._meta_recv,
            self._token_recv,
            self._token_send,
            self._kv_status_recv,
            self._kv_status_send,
        ):
            if sock is not None:
                sock.close(linger=0)
        if self._owns_ctx:
            self._ctx.term()
