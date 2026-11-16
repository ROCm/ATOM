# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm/receive_server.py`` (``ReceiveServer``) and the
# decode side of ``tilert/pd_vllm/transport.py`` (``MooncakeTransport``), for the DeepSeek-V4.1 wire of ``wire``.
"""The decode host's receive path: the TCP control plane a prefill ("P") rank talks to, its one receive slot, and the
RDMA-registered buffer P writes the KV image into.

Per connection (``wire``'s framing; TileRT ``receive_server.py:221-263``):

1. D sends ``hello`` (``wire.hello_msg``: magic, protocol 2, layout 2041, transport, ``max_seq_len``, an advisory
   ``busy``, the transport's ``session_id``, ``senders`` / ``dsv41_base`` / ``dsv41_wire``);
2. P sends ``{rid, rank, seq_len, last_prompt_token, sampling, admission_window_s[, prompt_token_ids][, kv_scales]}``;
   D answers ``accept`` (``{accepted, rid, rank, generation}``) or ``reject`` (``error``: ``seq_len exceeds
   max_seq_len`` / ``cancelling`` / ``duplicate_rank`` / ``busy``; P retries ``busy`` and ``cancelling``);
3. P RDMA-writes its image into the buffer at ``dsv41_base`` (the transport; D does nothing), then sends ``done``;
   once every sender rank (``wire.SENDER_RANKS``: rank 0) is done the request is ``COMPLETE`` and queued on
   ``completed``, which the decode call drains (``server``).

The slot (one request at a time): ``RESERVED`` -> ``TRANSFERRING`` (a writer admitted) -> ``COMPLETE``; ``release``
frees it (``CANCELLING`` while a writer is still connected) and tombstones the rid for ``request_timeout`` s, so a
late P of an abandoned request is refused (``cancelling``, ``detail: no_consumer``) instead of overwriting the next
request's KV; ``expect`` (the decode call arriving) lifts the tombstone. A request's ``generation`` (a counter over
admitted requests) tags its ``accept`` and P echoes it in ``done``: a ``done`` for an older generation is ignored.

Differences from TileRT's class: the buffer and transport are made apart (``alloc_receive_buffer``,
``make_transport``, ``open_receive_server``) and passed in, so the control plane runs without a GPU or RDMA (tests);
``port=0`` binds an ephemeral port (``ReceiveServer.port``); a P ``kv_scales`` field is ignored (the DeepSeek-V4.1
profile has none and TileRT's would fail on it: ``dsv41_rocm.py`` has no ``set_kv_scales``). NIXL is not ported.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time

from . import wire as W

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------
# transport (transport.py): D only registers its buffer and publishes its session id; P writes


class MooncakeTransport:
    """Mooncake's transfer engine on D: ``init`` (P2P handshake over RDMA), ``register`` the receive buffer,
    ``local_meta`` the hello's ``session_id``."""

    name = "mooncake"

    def init(self, host: str) -> None:
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        ret = self.engine.initialize(host, "P2PHANDSHAKE", "rdma", "")
        if ret != 0:
            raise RuntimeError(f"Mooncake engine init failed: {ret}")
        self.session_id = f"{host}:{self.engine.get_rpc_port()}"

    def register(
        self, ptr: int, nbytes: int, location: str | None = None, host: bool = False
    ) -> None:
        if host:
            location = None
        args = ([ptr], [nbytes]) if location is None else ([ptr], [nbytes], location)
        ret = self.engine.batch_register_memory(*args)
        if ret != 0:
            raise RuntimeError(f"Mooncake register failed: {ret}")

    def local_meta(self) -> dict:
        return {"session_id": self.session_id}


class PublishedTransport:
    """A transport another process owns (the rank-0 worker of ``fanout``): only what the hello publishes of it."""

    def __init__(self, name: str, meta: dict):
        self.name, self._meta = name, dict(meta)

    def local_meta(self) -> dict:
        return dict(self._meta)


def make_transport(name: str | None):
    key = (name or "mooncake").lower()
    if key == "mooncake":
        return MooncakeTransport()
    raise NotImplementedError(
        f"transport {name!r}: only mooncake is ported (TileRT's nixl backend is not)"
    )


def alloc_receive_buffer(max_seq_len: int, device: str = "cuda:0"):
    """The receive buffer: ``wire.wire_bytes(max_seq_len)`` bytes, uint8, zeroed, on ``device`` (TileRT's
    ``TILERT_PD_SHARDS=1``, ``--pd-buffer-device cuda``: one unsharded buffer on cuda:0).
    """
    import torch

    return torch.zeros(W.wire_bytes(max_seq_len), dtype=torch.uint8, device=device)


# ---------------------------------------------------------------------------------------------------------------
# the control plane (receive_server.py)


class ReceiveServer:
    """One receive slot and its TCP control plane (module docstring). ``base_ptr``: the buffer's address (the hello's
    ``dsv41_base``); ``transport``: an object with ``name`` and ``local_meta()``. Starts listening at once.
    """

    def __init__(
        self,
        max_seq_len: int,
        base_ptr: int,
        transport,
        port: int = 5556,
        host: str = "::",
        request_timeout: float = 120.0,
        buffer=None,
    ):
        self.max_seq_len = int(max_seq_len)
        self.base_ptr = int(base_ptr)
        self.buffer = buffer
        self.request_timeout = request_timeout
        self.sender_ranks = W.SENDER_RANKS
        self._transport_name = transport.name
        self._transport_meta = transport.local_meta()
        self._hello_layout = W.hello_layout(self.base_ptr, self.max_seq_len)
        self._lock = threading.Lock()
        self._current: W.ReceivedRequest | None = None
        self._cancelled: dict[str, float] = {}
        self._generation = 0
        self.completed: queue.Queue[W.ReceivedRequest] = queue.Queue()
        self._srv = _listen(host, port)
        self.port = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop, name="pd-recv-accept", daemon=True
        )
        self._thread.start()
        logger.info("control plane listening on :%d", self.port)

    # ---- the decode call's side
    def expect(self, rid: str) -> None:
        """A decode call for ``rid`` arrived: lift its tombstone (a re-announced request)."""
        with self._lock:
            if self._cancelled.pop(rid, None) is not None:
                logger.info("request %s re-announced; its tombstone is dropped", rid)

    def release(self, rid: str) -> None:
        """``rid`` is done with (decoded, refused or abandoned): tombstone it and free the slot if it holds it."""
        with self._lock:
            self._tombstone(rid)
            cur = self._current
            if cur is None:
                return
            if cur.rid != rid:
                logger.info("release(%s) ignored: the slot now holds %s", rid, cur.rid)
                return
            if cur.has_live_writer and cur.state != W.COMPLETE:
                cur.state = W.CANCELLING
                logger.warning(
                    "release(%s) with %d writer(s) still active: slot stays claimed (cancelling) until they stop",
                    cur.rid,
                    cur.active_writers,
                )
                return
            self._current = None

    def state_snapshot(self) -> dict:
        with self._lock:
            cur = self._current
            if cur is None:
                return {
                    "state": W.FREE,
                    "rid": None,
                    "generation": None,
                    "active_writers": 0,
                }
            return {
                "state": cur.state,
                "rid": cur.rid,
                "generation": cur.generation,
                "active_writers": cur.active_writers,
            }

    def close(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    # ---- slot bookkeeping (under self._lock)
    def _tombstone(self, rid: str) -> None:
        now = time.time()
        self._cancelled = {r: t for r, t in self._cancelled.items() if t > now}
        self._cancelled[rid] = now + self.request_timeout

    def _extend_tombstone(self, rid: str, window_s) -> None:
        try:
            window = float(window_s)
        except (TypeError, ValueError):
            return
        if window <= 0:
            return
        deadline = time.time() + window
        if deadline > self._cancelled.get(rid, 0.0):
            self._cancelled[rid] = deadline

    def _is_tombstoned(self, rid: str) -> bool:
        deadline = self._cancelled.get(rid)
        if deadline is None:
            return False
        if deadline <= time.time():
            del self._cancelled[rid]
            return False
        return True

    def _reusable(self, cur: W.ReceivedRequest) -> bool:
        if cur.state == W.COMPLETE:
            return True
        if cur.has_live_writer:
            return False
        if cur.state in (W.FREE, W.CANCELLING):
            return True
        return time.time() - cur.t_first_conn >= self.request_timeout

    def _admit(self, req: dict, rid: str, rank: int) -> dict:
        if req.get("kv_scales"):
            logger.warning(
                "request %s rank %d sent kv_scales; DeepSeek-V4.1 has none, ignored",
                rid,
                rank,
            )
        with self._lock:
            if self._is_tombstoned(rid):
                self._extend_tombstone(rid, req.get("admission_window_s"))
                logger.warning(
                    "refusing %s rank %d for now: no consumer is waiting for this request",
                    rid,
                    rank,
                )
                return W.reject_msg("cancelling", rid=rid, detail="no_consumer")
            cur = self._current
            if cur is not None and cur.rid == rid:
                if cur.state == W.CANCELLING:
                    return W.reject_msg("cancelling", rid=rid)
                if rank in cur.done_ranks:
                    return W.reject_msg("duplicate_rank", rid=rid, rank=rank)
            elif cur is not None and not self._reusable(cur):
                logger.warning(
                    "rejecting %s rank %d (busy with %s, state=%s, writers=%d)",
                    rid,
                    rank,
                    cur.rid,
                    cur.state,
                    cur.active_writers,
                )
                return W.reject_msg("busy", busy_rid=cur.rid, busy_state=cur.state)
            else:
                self._generation += 1
                self._current = cur = W.ReceivedRequest(
                    rid=rid,
                    seq_len=int(req["seq_len"]),
                    last_prompt_token=int(req.get("last_prompt_token", 0)),
                    first_token_id=req.get("first_token_id"),
                    sampling=req.get("sampling"),
                    prompt_token_ids=list(req.get("prompt_token_ids") or []),
                    t_first_conn=time.time(),
                    generation=self._generation,
                    state=W.RESERVED,
                )
                logger.info(
                    "request %s: seq_len=%d (generation %d)",
                    rid,
                    cur.seq_len,
                    cur.generation,
                )
            if not cur.prompt_token_ids and req.get("prompt_token_ids"):
                cur.prompt_token_ids = list(req["prompt_token_ids"])
            cur.state = W.TRANSFERRING
            cur.active_writers += 1
            return W.accept_msg(rid, rank, cur.generation)

    def _writer_left(self, rid: str, generation: int) -> None:
        with self._lock:
            cur = self._current
            if cur is None or cur.generation != generation:
                return
            cur.active_writers = max(0, cur.active_writers - 1)
            if cur.state == W.CANCELLING and cur.active_writers == 0:
                logger.info("request %s drained; receive slot free", cur.rid)
                self._current = None

    # ---- connections
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn, addr), daemon=True
            ).start()

    def _handle(self, conn: socket.socket, addr) -> None:
        admitted: tuple[str, int] | None = None
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self.request_timeout)
            with self._lock:
                busy = self._current is not None and not self._reusable(self._current)
            W.send_msg(
                conn,
                W.hello_msg(
                    self._transport_name,
                    self._transport_meta,
                    self.max_seq_len,
                    W.LAYOUT_VERSION,
                    self._hello_layout,
                    busy=busy,
                ),
            )
            req = W.recv_msg(conn)
            rid, rank = req["rid"], int(req["rank"])
            if req.get("seq_len", 0) > self.max_seq_len:
                W.send_msg(
                    conn,
                    W.reject_msg(
                        "seq_len exceeds max_seq_len",
                        rid=rid,
                        max_seq_len=self.max_seq_len,
                    ),
                )
                return
            reply = self._admit(req, rid, rank)
            W.send_msg(conn, reply)
            if not reply.get("accepted"):
                return
            admitted = (rid, reply["generation"])
            done = W.recv_msg(conn)
            if not done.get("done"):
                logger.warning("rank %d sent non-done message: %s", rank, done)
                return
            with self._lock:
                cur = self._current
                if cur is None or cur.rid != rid or cur.generation != admitted[1]:
                    logger.warning(
                        "ignoring done from %s rank %d: admitted to generation %d and the slot has moved on",
                        rid,
                        rank,
                        admitted[1],
                    )
                    return
                if cur.state == W.CANCELLING:
                    logger.warning(
                        "ignoring done from %s rank %d: request was abandoned",
                        rid,
                        rank,
                    )
                    return
                cur.done_ranks.add(rank)
                if cur.done_ranks >= set(self.sender_ranks):
                    cur.state = W.COMPLETE
                    cur.t_complete = time.time()
                    self.completed.put(cur)
                    logger.info(
                        "request %s: all ranks done in %.1f ms",
                        rid,
                        1000 * (cur.t_complete - cur.t_first_conn),
                    )
        except Exception:
            logger.exception("connection from %s failed", addr)
        finally:
            if admitted is not None:
                self._writer_left(*admitted)
            conn.close()


def _listen(host: str, port: int) -> socket.socket:
    """A listening TCP socket on ``host:port``; ``"::"`` is dual-stack (IPv4 too), as TileRT binds it."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if fam == socket.AF_INET6:
        try:
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
    s.bind((host, port))
    s.listen(32)
    return s


def open_receive_server(
    max_seq_len: int,
    ctrl_port: int,
    transport: str = "mooncake",
    device: str = "cuda:0",
    hostname: str | None = None,
    request_timeout: float = 120.0,
) -> ReceiveServer:
    """TileRT's ``ReceiveServer.__init__``: the buffer on ``device``, the transport initialized on this host's
    address (``wire.local_ip``, or ``hostname``) with the buffer registered, then the control plane on
    ``ctrl_port``."""
    if (os.environ.get("TILERT_PD_SHARDS") or "1").strip() not in ("", "1"):
        raise SystemExit(
            "dsv41 lands into one unsharded buffer: set TILERT_PD_SHARDS=1"
        )
    buf = alloc_receive_buffer(max_seq_len, device)
    tr = make_transport(transport)
    tr.init(hostname or W.local_ip())
    tr.register(buf.data_ptr(), buf.numel(), host=buf.device.type == "cpu")
    logger.info(
        "transport=%s ready, buffer registered (%.2f GB)",
        tr.name,
        buf.numel() / 1024**3,
    )
    return ReceiveServer(
        max_seq_len,
        buf.data_ptr(),
        tr,
        port=ctrl_port,
        request_timeout=request_timeout,
        buffer=buf,
    )
