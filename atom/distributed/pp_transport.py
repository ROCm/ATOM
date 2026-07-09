# SPDX-License-Identifier: MIT
# Pipeline-parallel inter-stage CPU transport (CPP A·P1b, 方案②).
#
# Each PP stage runs as its own EngineCore process. Two control channels run
# over ZMQ (CPU); the hidden-state tensors themselves go GPU-to-GPU over NCCL
# (see pp_comm.py), never here.
#
#   metadata  (head -> every downstream stage): the scheduled batch to run.
#   tokens    (last stage -> head):             sampled ScheduledBatchOutput,
#                                               fed back so the head owns the
#                                               request lifecycle / next step.
#
# The objects moved here (ScheduledBatch, ScheduledBatchOutput) are the same
# ones already pickled to broadcast to workers, so no bespoke wire format is
# needed — pickle round-trips them verbatim.
#
# Bind/connect convention: the RECEIVER binds, the SENDER connects (PUSH/PULL,
# so connect-before-bind is fine; ZMQ queues at the sender).

import logging
import pickle
from typing import Any, List, Optional

import zmq

logger = logging.getLogger("atom")


class PPStageTransport:
    """Per-stage ZMQ endpoints for the head<->downstream metadata/token channels.

    Args:
        pp_rank: this stage's index (0 = head).
        pp_size: number of pipeline stages.
        meta_addrs: length pp_size; meta_addrs[s] is the endpoint on which stage
            s RECEIVES metadata from the head (index 0 is unused). The head
            connects a PUSH socket to each downstream endpoint; each downstream
            stage binds a PULL socket to its own endpoint.
        token_addr: endpoint on which the head RECEIVES tokens from the last
            stage. The head binds a PULL socket; the last stage connects PUSH.
        ctx: optional shared zmq.Context (one is created if omitted).
    """

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        meta_addrs: List[str],
        token_addr: str,
        ctx: Optional[zmq.Context] = None,
    ):
        assert pp_size >= 2, "PPStageTransport is only used when pp_size >= 2"
        assert len(meta_addrs) == pp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_head = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1
        self._ctx = ctx or zmq.Context.instance()
        self._owns_ctx = ctx is None

        self._meta_send: List[zmq.Socket] = []
        self._meta_recv: Optional[zmq.Socket] = None
        self._token_recv: Optional[zmq.Socket] = None
        self._token_send: Optional[zmq.Socket] = None

        if self.is_head:
            # One PUSH per downstream stage (metadata fan-out).
            for s in range(1, pp_size):
                sock = self._ctx.socket(zmq.PUSH)
                sock.connect(meta_addrs[s])
                self._meta_send.append(sock)
            # Receive sampled tokens back from the last stage.
            self._token_recv = self._ctx.socket(zmq.PULL)
            self._token_recv.bind(token_addr)
        else:
            # Receive the scheduled batch from the head.
            self._meta_recv = self._ctx.socket(zmq.PULL)
            self._meta_recv.bind(meta_addrs[pp_rank])
            if self.is_last:
                self._token_send = self._ctx.socket(zmq.PUSH)
                self._token_send.connect(token_addr)

    # ---- head side ----------------------------------------------------------
    def send_metadata(self, batch: Any) -> None:
        """Head: broadcast the scheduled batch to every downstream stage."""
        payload = pickle.dumps(batch)
        for sock in self._meta_send:
            sock.send(payload, copy=False)

    def recv_tokens(self, timeout_ms: Optional[int] = None) -> Any:
        """Head: block for the last stage's sampled ScheduledBatchOutput.

        Returns None on timeout (used as a deadlock guard in tests/bring-up).
        """
        if timeout_ms is not None:
            if not self._token_recv.poll(timeout_ms):
                return None
        return pickle.loads(self._token_recv.recv())

    # ---- downstream / last side --------------------------------------------
    def recv_metadata(self, timeout_ms: Optional[int] = None) -> Any:
        """Downstream: block for the head's scheduled batch."""
        if timeout_ms is not None:
            if not self._meta_recv.poll(timeout_ms):
                return None
        return pickle.loads(self._meta_recv.recv())

    def send_tokens(self, out: Any) -> None:
        """Last stage: feed the sampled output back to the head."""
        assert self._token_send is not None, "send_tokens only valid on last stage"
        self._token_send.send(pickle.dumps(out), copy=False)

    def close(self) -> None:
        for sock in self._meta_send:
            sock.close(linger=0)
        for sock in (self._meta_recv, self._token_recv, self._token_send):
            if sock is not None:
                sock.close(linger=0)
        if self._owns_ctx:
            self._ctx.term()
