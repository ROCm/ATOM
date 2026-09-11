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

    Every message is a three-frame multipart `[topic, seq, payload]`, the
    layout vLLM's `ZmqEventPublisher` emits: `topic` is the (possibly empty)
    subscription key, `seq` is a monotonic uint64 big-endian batch counter,
    and `payload` is the msgpack-encoded `EventBatch`. Consumers must use
    `recv_multipart()`. `seq` is assigned at enqueue, so a batch dropped on
    queue overflow shows up as a gap in the stream instead of vanishing
    silently. The all-0xFF value is never used as a data seq; vLLM reserves
    it as the replay terminator.

    With `data_parallel_rank` set, the bound endpoint is moved to that rank
    (see `offset_endpoint_port`) so co-located DP ranks do not collide on one
    socket, and subscribers reach rank N at base port + N.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topic: str = "",
        hwm: int = 0,
        buffer_steps: int = 10_000,
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
        # Queue items are (seq, payload); None is the shutdown sentinel.
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(
            maxsize=buffer_steps
        )
        self._seq_gen = itertools.count()

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.set_hwm(hwm)
        self._socket.bind(offset_endpoint_port(endpoint, data_parallel_rank or 0))
        self._zmq_error_cls = zmq.ZMQError  # captured so _run doesn't re-import

        self._drops = 0
        self._sent = 0
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

        # Assigned at enqueue rather than at send: a batch dropped on overflow
        # below still consumes a seq, so subscribers see the loss as a gap.
        # Modulo keeps the counter in [0, 2**64-2], clear of vLLM's all-0xFF
        # replay terminator.
        seq = next(self._seq_gen) % 0xFFFFFFFFFFFFFFFF

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
        try:
            self._socket.close(linger=linger)
        except Exception:  # pragma: no cover
            pass

    # --- internal ---
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            seq, payload = item
            try:
                self._socket.send_multipart(
                    [self._topic_bytes, seq.to_bytes(8, "big"), payload]
                )
                with self._lock:
                    self._sent += 1
            except self._zmq_error_cls:  # pragma: no cover - socket closed
                return

    # Test/diagnostic hooks.
    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sent": self._sent,
                "dropped": self._drops,
                "encode_errors": self._encode_errors,
            }


def offset_endpoint_port(endpoint: str, data_parallel_rank: int) -> str:
    """Return `endpoint` moved to the given data-parallel rank: tcp ports are
    offset by the rank, other transports (inproc, ipc) get a `_dp<rank>`
    suffix. Rank 0 returns the endpoint unchanged. Mirrors vLLM's
    `ZmqEventPublisher.offset_endpoint_port`, so a subscriber that dials
    base port + rank works against either engine."""
    if data_parallel_rank <= 0:
        return endpoint
    if endpoint.startswith("tcp://"):
        base, sep, port = endpoint.rpartition(":")
        if not sep or not port.isdigit():
            raise ValueError(f"tcp endpoint without a numeric port: {endpoint!r}")
        return f"{base}:{int(port) + data_parallel_rank}"
    return f"{endpoint}_dp{data_parallel_rank}"


def make_publisher(
    enabled: bool,
    publisher_kind: str,
    endpoint: str,
    *,
    topic: str = "",
    hwm: int = 0,
    buffer_steps: int = 10_000,
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
            data_parallel_rank=data_parallel_rank,
        )
    raise ValueError(f"unknown KV event publisher: {publisher_kind!r}")
