# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV cache events: wire-compatible with vLLM's `vllm.distributed.kv_events`,
plus ATOM extensions (`BlockTransferred`, CPU/DISK/REMOTE medium constants)."""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Any, Final

import msgspec

logger = logging.getLogger("atom")

# Where a block lives.
MEDIUM_GPU: Final[str] = "GPU"
MEDIUM_CPU: Final[str] = "CPU"
MEDIUM_DISK: Final[str] = "DISK"
MEDIUM_REMOTE: Final[str] = "REMOTE"

# Reserved seq frame that terminates a replay response. Its payload is a
# msgpack `[oldest_available_seq, latest_seq]` window so a consumer knows when
# the replay is complete and whether earlier events were already evicted
# (start_seq < oldest). 2**64-1 is reserved for this and never used as a data
# seq. (Replay requires a DEALER/ROUTER-style client that can read the
# multi-message reply; a REQ socket cannot.)
REPLAY_DONE: Final[bytes] = b"\xff" * 8


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    """Tagged-union base so subscribers can dispatch on event type."""


class BlockStored(KVCacheEvent):
    """A run of contiguous prefix-cacheable blocks just became resident."""

    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = MEDIUM_GPU
    lora_name: str | None = None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    # Reserved wire slots; emitted as None until hybrid-cache wiring lands.
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    # ATOM extension (trailing, so strict vLLM array_like consumers ignore it):
    # sequence position of the first token of the first block in this run.
    # With block_size, block i covers [token_offset + i*block_size, +block_size).
    token_offset: int | None = None


class BlockRemoved(KVCacheEvent):
    """One or more blocks were evicted from the given medium."""

    block_hashes: list[int]
    medium: str | None = MEDIUM_GPU
    group_idx: int | None = None


class AllBlocksCleared(KVCacheEvent):
    """Entire cache (or one medium, when set) was cleared."""

    medium: str | None = None


class BlockTransferred(KVCacheEvent):
    """A block moved between tiers without changing identity.

    Emitted on GPU↔CPU/DISK swap, REMOTE→GPU receive, and GPU→REMOTE send.
    ATOM-only — strict vLLM consumers should narrow the union to exclude it.
    """

    block_hashes: list[int]
    from_medium: str
    to_medium: str
    group_idx: int | None = None


# Union of all events. Subscribers that only know the vLLM-compatible subset
# should narrow this to `BlockStored | BlockRemoved | AllBlocksCleared`.
EventType = BlockStored | BlockRemoved | AllBlocksCleared | BlockTransferred


class EventBatch(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """A batch of events emitted at one publish tick. Field layout matches
    vLLM's EventBatch (ts as float seconds, events list, optional dp_rank)."""

    ts: float
    events: list[EventType]
    data_parallel_rank: int | None = None


# ----- Publisher ---------------------------------------------------------- #


class EventPublisher(ABC):
    """Strategy for delivering EventBatch off the hot path.

    Implementations must be safe to call from the scheduler thread; the actual
    network/IO must run in a background thread or be lock-free so the
    `publish()` call returns quickly.
    """

    @abstractmethod
    def publish(self, events: Iterable[KVCacheEvent]) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class NullEventPublisher(EventPublisher):
    """No-op. Default when KV events are disabled."""

    def publish(self, events: Iterable[KVCacheEvent]) -> None:
        return

    def shutdown(self) -> None:
        return


# Batches a single sender-loop iteration replays before it goes back to the
# live queue. Bounds how long a large replay (default retention 10k batches)
# can keep the live PUB stream waiting; it does not bound the replay itself.
REPLAY_CHUNK: Final = 64


def _bind(sock, endpoint: str, role: str, zmq_error_cls) -> None:
    """Bind, turning EADDRINUSE and friends into an error that names the
    endpoint rules, since the collision is usually a config mistake."""
    try:
        sock.bind(endpoint)
    except zmq_error_cls as e:
        raise RuntimeError(
            f"KV events: cannot bind {role} socket to {endpoint!r}: {e}. "
            "Each publisher binds its configured PUB and replay endpoints "
            "offset by its DP rank, so (a) the two configured tcp ports must "
            "be at least data_parallel_size apart, and (b) every other engine "
            "process on this host (a P/D peer, another deployment) needs its "
            "own ATOM_KV_EVENTS_ENDPOINT / ATOM_KV_EVENTS_REPLAY_ENDPOINT."
        ) from e


def offset_endpoint(endpoint: str, data_parallel_rank: int | None) -> str:
    """Make a configured ZMQ bind endpoint unique per data-parallel rank.

    Every DP rank runs its own Scheduler and therefore its own publisher, but
    they all read the same `KVEventsConfig`, so binding the configured address
    verbatim collides from rank 1 onward. `tcp://host:port` gets the rank added
    to the port; `ipc://` and `inproc://` get a `_dp{rank}` suffix. Rank 0 (or
    no DP) keeps the configured endpoint, so single-engine deployments and
    existing consumers see no change.
    """
    if not data_parallel_rank:
        return endpoint
    if endpoint.startswith("tcp://"):
        host, sep, port = endpoint.rpartition(":")
        if sep and port.isdigit():
            return f"{host}:{int(port) + data_parallel_rank}"
        raise ValueError(
            f"cannot offset KV event endpoint {endpoint!r} for dp_rank "
            f"{data_parallel_rank}: tcp endpoints need an explicit numeric port"
        )
    return f"{endpoint}_dp{data_parallel_rank}"


class ZmqEventPublisher(EventPublisher):
    """ZMQ PUB-socket publisher.

    Uses a background sender thread + bounded queue so the scheduler never
    blocks on network IO. If the queue fills (slow subscriber), the oldest
    batch is dropped — KV events are advisory and a missed eviction is
    cheaper than stalling inference.

    Every message is a three-frame multipart `[topic, seq, payload]`, where
    `topic` is the (possibly empty) subscription key, `seq` is a monotonic
    8-byte big-endian batch counter (wrapping at 2**64-1; the value 2**64-1 is
    reserved for the REPLAY_DONE terminal frame and never used as a data seq),
    and `payload` is the msgpack-encoded EventBatch. Consumers must use
    `recv_multipart()`.

    `seq` is assigned at enqueue time, so a batch that never reaches the wire
    still consumes a sequence number: the loss surfaces to subscribers as a
    gap in the seq stream rather than vanishing silently. Three loss cases:
      * transport drop (slow/late SUB) — detectable as a gap AND recoverable
        from the replay buffer (the batch was sent, so it is buffered);
      * queue-overflow drop (slow encoder/sender) — detectable as a gap but
        NOT recoverable (never sent, never buffered); also counted in
        `stats['dropped']`;
      * encode failure — same as overflow, counted in `stats['encode_errors']`.

    Sequence numbers start at 0 per publisher process and advance by one per
    batch, so the 2**64-1 modulus is a wire-format bound, not an operational
    one: at one batch per scheduler step the counter cannot wrap within the
    lifetime of a process. Replay ordering (`seq >= start_seq`) therefore
    assumes no wrap inside the retained window; see `_advance_replay`.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topic: str = "",
        hwm: int = 0,
        buffer_steps: int = 10_000,
        replay_endpoint: str = "",
        replay_buffer_steps: int = 10_000,
        data_parallel_rank: int | None = None,
        encoder: msgspec.msgpack.Encoder | None = None,
    ) -> None:
        if buffer_steps < 1:
            raise ValueError(
                f"buffer_steps must be >= 1 to keep the drop-on-overflow "
                f"backpressure intact; got {buffer_steps}"
            )
        if replay_endpoint and replay_buffer_steps < 1:
            raise ValueError(
                f"replay_buffer_steps must be >= 1 when replay is enabled "
                f"(maxlen=0 would silently disable replay retention); "
                f"got {replay_buffer_steps}"
            )
        # Local import: keep pyzmq an optional runtime dep. BlockManager imports
        # this module unconditionally, but only the zmq publisher path needs pyzmq.
        import zmq

        self._dp_rank = data_parallel_rank
        self._topic_bytes = topic.encode("utf-8")
        self._encoder = encoder or msgspec.msgpack.Encoder()
        # Queue items are (seq, payload) tuples; None is the shutdown sentinel.
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(
            maxsize=buffer_steps
        )

        # Effective bind addresses, after the per-DP-rank offset.
        self.endpoint = offset_endpoint(endpoint, data_parallel_rank)
        self.replay_endpoint = (
            offset_endpoint(replay_endpoint, data_parallel_rank)
            if replay_endpoint
            else ""
        )

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.set_hwm(hwm)
        _bind(self._socket, self.endpoint, "PUB", zmq.ZMQError)
        # Captured so the sender thread never re-imports zmq.
        self._zmq_error_cls = zmq.ZMQError
        self._zmq_again_cls = zmq.Again
        self._zmq_noblock = zmq.NOBLOCK

        # Optional replay: a ROUTER socket + ring buffer of recently-sent
        # batches. A subscriber that detects a seq gap can request everything
        # from a start sequence number and get the buffered batches back. The
        # ROUTER is created here but used only by the sender thread.
        # `replay_buffer_steps` is a distinct knob from `buffer_steps` (the
        # in-flight send queue): it bounds the long-lived retention of encoded
        # payloads (which can include large token_id lists) for the publisher's
        # lifetime, and only when replay is enabled.
        self._replay = None
        self._replay_buffer: deque[tuple[int, bytes, bytes]] | None = None
        if self.replay_endpoint:
            self._replay = ctx.socket(zmq.ROUTER)
            # Replay is serviced on the sender thread, so a replay client that
            # stops reading must never stall live publication. ROUTER_MANDATORY
            # turns a full per-peer pipe into an error instead of a silent drop,
            # and every replay send is NOBLOCK, so that error is EAGAIN: the
            # request is abandoned (see _service_replay) and live sends resume.
            self._replay.setsockopt(zmq.ROUTER_MANDATORY, 1)
            _bind(self._replay, self.replay_endpoint, "replay ROUTER", zmq.ZMQError)
            self._replay_buffer = deque(maxlen=replay_buffer_steps)
        # In-progress replay, if any: (routing prefix, next seq to send). Set
        # by _accept_replay_request, advanced a chunk at a time by
        # _advance_replay so live sends interleave with a long replay.
        self._pending_replay: tuple[list[bytes], int] | None = None
        self._replay_chunk = REPLAY_CHUNK

        self._seq_gen = itertools.count()
        self._drops = 0
        self._sent = 0
        self._replayed = 0
        self._replay_aborted = 0
        self._encode_errors = 0
        self._closing = False
        self._lock = threading.Lock()
        self._sender = threading.Thread(
            target=self._run, name="atom-kv-event-sender", daemon=True
        )
        self._sender.start()

    def publish(self, events: Iterable[KVCacheEvent]) -> None:
        if self._closing:
            return
        evt_list = list(events)
        if not evt_list:
            return
        batch = EventBatch(
            ts=time.time(),
            events=evt_list,
            data_parallel_rank=self._dp_rank,
        )
        # Assign the sequence number here (at enqueue), before encoding and
        # not at send: a batch lost to an encode failure or dropped on overflow
        # below still consumes a seq, so every loss is visible to subscribers
        # as a gap instead of vanishing silently.
        # Keep seq in [0, 2**64-2] so the wire frame, the replay-buffer key, and
        # the start_seq comparison in _service_replay all use the same value,
        # AND never collide with the reserved all-0xFF REPLAY_DONE terminal
        # (modulo, not mask).
        seq = next(self._seq_gen) % 0xFFFFFFFFFFFFFFFF

        try:
            payload = self._encoder.encode(batch)
        except Exception:
            # Surface via stats.encode_errors. Log the first occurrence with
            # traceback so the root cause is discoverable; further failures are
            # tracked via the counter only to avoid log spam.
            with self._lock:
                first_failure = self._encode_errors == 0
                self._encode_errors += 1
            if first_failure:
                logger.exception(
                    "KV event encode failed; subsequent failures will be "
                    "tracked via stats['encode_errors']"
                )
            return

        # Non-blocking enqueue; drop oldest on overflow.
        while True:
            try:
                self._queue.put_nowait((seq, payload))
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._drops += 1
                except queue.Empty:  # pragma: no cover - race window
                    pass

    def shutdown(self) -> None:
        self._closing = True  # publish() will return early from here on
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._drops += 1
                except queue.Empty:
                    pass
        self._sender.join(timeout=2.0)
        linger = 0 if self._sender.is_alive() else 1000
        # Best-effort close on shutdown: a socket already torn down by the
        # sender thread must not turn shutdown into an error.
        with contextlib.suppress(Exception):  # pragma: no cover
            self._socket.close(linger=linger)
        if self._replay is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._replay.close(linger=0)

    # --- internal ---
    def _run(self) -> None:
        # Poll the replay socket between sends. When replay is disabled the
        # queue.get() blocks (timeout=None); when enabled it wakes periodically
        # so replay requests are serviced even while no events are flowing.
        # While a replay is in progress the loop alternates: one chunk of
        # replay, then whatever is on the live queue (without waiting), so a
        # 10k-batch replay to a fast client cannot starve live publication.
        idle_timeout = 0.05 if self._replay is not None else None
        while True:
            if self._replay is not None:
                try:
                    if self._pending_replay is None and self._replay.poll(0):
                        self._accept_replay_request()
                    if self._pending_replay is not None:
                        self._advance_replay()
                except self._zmq_error_cls:  # pragma: no cover - closed on shutdown
                    return
                except Exception:  # pragma: no cover - replay is non-critical
                    logger.exception("KV event replay request failed")
                    self._pending_replay = None
            get_timeout = 0 if self._pending_replay is not None else idle_timeout
            try:
                item = self._queue.get(timeout=get_timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            seq, payload = item
            try:
                # seq is already masked to uint64 at enqueue.
                seq_bytes = seq.to_bytes(8, "big")
                self._socket.send_multipart([self._topic_bytes, seq_bytes, payload])
                if self._replay_buffer is not None:
                    self._replay_buffer.append((seq, seq_bytes, payload))
                with self._lock:
                    self._sent += 1
            except self._zmq_error_cls:  # pragma: no cover - socket closed
                return

    def _accept_replay_request(self) -> None:
        """Take one replay request off the ROUTER and make it the pending
        replay. Request frame is `[client_id, (delim,) start_seq]`; the routing
        prefix is echoed back on every reply frame."""
        frames = self._replay.recv_multipart()
        # The wire contract is exactly one 8-byte big-endian start_seq after
        # the routing prefix. int.from_bytes would happily read b"" as 0 and
        # a longer frame as a huge value, so a garbled request could trigger a
        # full replay; reject anything but the exact width.
        if len(frames) < 2 or len(frames[-1]) != 8:
            logger.warning("KV event replay: malformed request %r", frames)
            return
        start_seq = int.from_bytes(frames[-1], "big")
        self._pending_replay = (frames[:-1], start_seq)

    def _advance_replay(self) -> None:
        """Send the next chunk of the pending replay: up to `_replay_chunk`
        buffered batches with seq >= the request's start sequence, in order.
        When the buffer is exhausted, send a terminal frame so the consumer
        knows the reply is complete, and clear the pending replay.

        The terminal frame is `[*prefix, REPLAY_DONE, [oldest, latest]]`:
        REPLAY_DONE distinguishes it from data frames, and the msgpack window
        lets the consumer see whether events before `start_seq` were already
        evicted (start_seq < oldest) and terminate without a timeout even on a
        zero-match request.

        Ordering is plain integer comparison. A window straddling the 2**64-1
        wrap would misorder, but the counter starts at 0 per process and steps
        by one per batch, so reaching the wrap is not achievable in practice
        (~1.8e19 batches). Not handled by design; see the class docstring."""
        assert self._pending_replay is not None
        prefix, next_seq = self._pending_replay
        # Safe to iterate the deque directly: the sender thread is the only
        # mutator and it is the same thread running this method. Batches
        # appended by live sends between chunks simply extend the replay,
        # which is what a consumer catching up wants.
        buf = self._replay_buffer or ()
        sent = 0
        for seq, seq_bytes, payload in buf:
            if seq < next_seq:
                continue
            if not self._replay_send([*prefix, seq_bytes, payload]):
                self._pending_replay = None
                return
            with self._lock:
                self._replayed += 1
            next_seq = seq + 1
            sent += 1
            if sent >= self._replay_chunk:
                self._pending_replay = (prefix, next_seq)
                return
        # Terminal frame with the available window. Encode with the module
        # helper (fresh encoder) rather than self._encoder, which the scheduler
        # thread uses concurrently in publish().
        oldest = buf[0][0] if buf else None
        latest = buf[-1][0] if buf else None
        window = msgspec.msgpack.encode([oldest, latest])
        self._replay_send([*prefix, REPLAY_DONE, window])
        self._pending_replay = None

    def _replay_send(self, frames: list[bytes]) -> bool:
        """Non-blocking send on the replay ROUTER. Returns False when the
        client's pipe is full (or the client is gone), in which case the caller
        abandons the rest of this replay: the consumer is not draining, and
        blocking here would stall the live PUB stream that shares this thread.
        The consumer sees a missing REPLAY_DONE and can re-request."""
        try:
            self._replay.send_multipart(frames, flags=self._zmq_noblock)
            return True
        except self._zmq_again_cls:
            reason = "client not draining"
        except self._zmq_error_cls as e:  # EHOSTUNREACH: peer disconnected
            reason = f"{e.__class__.__name__}: {e}"
        with self._lock:
            self._replay_aborted += 1
            first = self._replay_aborted == 1
        if first:
            logger.warning(
                "KV event replay abandoned (%s); further aborts are counted in "
                "stats['replay_aborted']",
                reason,
            )
        return False

    # Test/diagnostic hooks.
    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sent": self._sent,
                "dropped": self._drops,
                "replayed": self._replayed,
                "replay_aborted": self._replay_aborted,
                "encode_errors": self._encode_errors,
            }


def make_publisher(
    enabled: bool,
    publisher_kind: str,
    endpoint: str,
    *,
    topic: str = "",
    hwm: int = 0,
    buffer_steps: int = 10_000,
    replay_endpoint: str = "",
    replay_buffer_steps: int = 10_000,
    data_parallel_rank: int | None = None,
) -> EventPublisher:
    """Construct a publisher from plain-config args. Returns `NullEventPublisher`
    when disabled, so callers can always call `publish()` without checking."""
    if not enabled or publisher_kind == "null":
        return NullEventPublisher()
    if publisher_kind == "zmq":
        return ZmqEventPublisher(
            endpoint=endpoint,
            topic=topic,
            hwm=hwm,
            buffer_steps=buffer_steps,
            replay_endpoint=replay_endpoint,
            replay_buffer_steps=replay_buffer_steps,
            data_parallel_rank=data_parallel_rank,
        )
    raise ValueError(f"unknown KV event publisher: {publisher_kind!r}")
