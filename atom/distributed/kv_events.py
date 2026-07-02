# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV cache events: wire-compatible with vLLM's `vllm.distributed.kv_events`,
plus ATOM extensions (`BlockTransferred`, CPU/DISK/REMOTE medium constants)."""

from __future__ import annotations

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


class ZmqEventPublisher(EventPublisher):
    """ZMQ PUB-socket publisher.

    Uses a background sender thread + bounded queue so the scheduler never
    blocks on network IO. If the queue fills (slow subscriber), the oldest
    batch is dropped — KV events are advisory and a missed eviction is
    cheaper than stalling inference.

    Every message is a three-frame multipart `[topic, seq, payload]`, where
    `topic` is the (possibly empty) subscription key, `seq` is a monotonic
    8-byte big-endian batch counter, and `payload` is the msgpack-encoded
    EventBatch. Consumers must use `recv_multipart()`. The sequence number
    lets a subscriber detect batches it missed (e.g. a slow/late SUB that the
    PUB socket dropped at the transport level) and, when a `replay_endpoint`
    is configured, request them back from the in-memory replay buffer.

    Note: `seq` is assigned at send time, so it counts only batches that left
    the sender. Batches dropped from the internal queue on overflow (slow
    encoder) are advisory losses tracked in `stats['dropped']`; they consume
    no sequence number and are not replayable.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topic: str = "",
        hwm: int = 0,
        buffer_steps: int = 10_000,
        replay_endpoint: str = "",
        data_parallel_rank: int | None = None,
        encoder: msgspec.msgpack.Encoder | None = None,
    ) -> None:
        if buffer_steps < 1:
            raise ValueError(
                f"buffer_steps must be >= 1 to keep the drop-on-overflow "
                f"backpressure intact; got {buffer_steps}"
            )
        # Local import: keep pyzmq an optional runtime dep. BlockManager imports
        # this module unconditionally, but only the zmq publisher path needs pyzmq.
        import zmq

        self._dp_rank = data_parallel_rank
        self._topic_bytes = topic.encode("utf-8")
        self._encoder = encoder or msgspec.msgpack.Encoder()
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=buffer_steps)

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.set_hwm(hwm)
        self._socket.bind(endpoint)
        self._zmq_error_cls = zmq.ZMQError  # captured so _run doesn't re-import

        # Optional replay: a ROUTER socket + ring buffer of recently-sent
        # batches. A subscriber that detects a seq gap can request everything
        # from a start sequence number and get the buffered batches back. The
        # ROUTER is created here but used only by the sender thread.
        self._replay = None
        self._replay_buffer: deque[tuple[int, bytes, bytes]] | None = None
        if replay_endpoint:
            self._replay = ctx.socket(zmq.ROUTER)
            self._replay.bind(replay_endpoint)
            self._replay_buffer = deque(maxlen=buffer_steps)

        self._seq_gen = itertools.count()
        self._drops = 0
        self._sent = 0
        self._replayed = 0
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
                self._queue.put_nowait(payload)
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
        try:
            self._socket.close(linger=linger)
        except Exception:  # pragma: no cover
            pass
        if self._replay is not None:
            try:
                self._replay.close(linger=0)
            except Exception:  # pragma: no cover
                pass

    # --- internal ---
    def _run(self) -> None:
        # Poll the replay socket between sends. When replay is disabled the
        # queue.get() blocks (timeout=None); when enabled it wakes periodically
        # so replay requests are serviced even while no events are flowing.
        get_timeout = 0.05 if self._replay is not None else None
        while True:
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception:  # pragma: no cover - replay is non-critical
                    logger.exception("KV event replay request failed")
            try:
                item = self._queue.get(timeout=get_timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                seq = next(self._seq_gen)
                seq_bytes = seq.to_bytes(8, "big")
                self._socket.send_multipart([self._topic_bytes, seq_bytes, item])
                if self._replay_buffer is not None:
                    self._replay_buffer.append((seq, seq_bytes, item))
                with self._lock:
                    self._sent += 1
            except self._zmq_error_cls:  # pragma: no cover - socket closed
                return

    def _service_replay(self) -> None:
        """Answer a pending replay request: resend every buffered batch with
        seq >= the requested start sequence. Request frame is
        `[client_id, (delim,) start_seq]`; we echo the routing prefix back."""
        frames = self._replay.recv_multipart()
        if len(frames) < 2:
            logger.warning("KV event replay: malformed request %r", frames)
            return
        try:
            start_seq = int.from_bytes(frames[-1], "big")
        except Exception:
            logger.warning("KV event replay: bad start_seq %r", frames[-1])
            return
        prefix = frames[:-1]  # [client_id] or [client_id, empty_delim]
        for seq, seq_bytes, payload in list(self._replay_buffer or ()):
            if seq >= start_seq:
                self._replay.send_multipart([*prefix, seq_bytes, payload])
                with self._lock:
                    self._replayed += 1

    # Test/diagnostic hooks.
    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sent": self._sent,
                "dropped": self._drops,
                "replayed": self._replayed,
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
            data_parallel_rank=data_parallel_rank,
        )
    raise ValueError(f"unknown KV event publisher: {publisher_kind!r}")
