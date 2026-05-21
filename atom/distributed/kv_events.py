# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV cache events: BlockStored, BlockRemoved, AllBlocksCleared, BlockTransferred.

Wire-compatible with vLLM's `vllm.distributed.kv_events` for `BlockStored` /
`BlockRemoved` / `AllBlocksCleared` so existing subscribers (LMCache, Mooncake
event listener, etc.) can consume ATOM events unchanged. ATOM extends this
with:

  * Medium constants: GPU / CPU / DISK / REMOTE (vLLM has only GPU)
  * `BlockTransferred` event for cross-tier moves (HMA tracking, P/D)
  * Direct emission API for KV transfer connectors (Mooncake/MoriIO)

Design notes
------------

The block-hash field is a 64-bit integer (ATOM uses xxhash64). vLLM's
`ExternalBlockHash` is also typed as an integer, so the wire format matches.
ATOM does not currently model multi-cache-groups, multi-LoRA, or multimodal
extra-keys; the corresponding fields are emitted as default values (group_idx=0,
lora_id=None, etc.) so the schema stays vLLM-compatible without paying the
runtime cost of populating fields ATOM never uses.

`BlockTransferred` is ATOM-only and not understood by vLLM consumers. The
tagged-union encoding means strict subscribers will fail decoding it; opt-in
consumers can declare the wider union including this tag.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

import msgspec

if TYPE_CHECKING:
    pass


# ----- Medium constants --------------------------------------------------- #
# Where a block lives. String values (not enum) for vLLM wire compatibility:
# vLLM uses `medium: str | None` on its events.

MEDIUM_GPU: Final[str] = "GPU"
MEDIUM_CPU: Final[str] = "CPU"  # HMA host-memory tier
MEDIUM_DISK: Final[str] = "DISK"  # HMA disk tier
MEDIUM_REMOTE: Final[str] = "REMOTE"  # transferred via Mooncake/MoriIO/peer


# ----- Event schema ------------------------------------------------------- #


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    """Base class for all KV cache events. The `tag=True` enables msgspec's
    tagged-union encoding so subscribers can dispatch on event type."""


class BlockStored(KVCacheEvent):
    """A run of contiguous prefix-cacheable blocks just became resident.

    The fields match vLLM's `BlockStored` so existing subscribers work. The
    only practical difference is that ATOM emits `medium=MEDIUM_GPU` by
    default; connectors may emit with `MEDIUM_REMOTE` to indicate a block
    received from a remote producer.
    """

    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    # ATOM does not (yet) serve LoRA adapters; emitted as None for compat.
    lora_id: int | None = None
    medium: str | None = MEDIUM_GPU
    lora_name: str | None = None
    # ATOM does not (yet) support multimodal extra keys; emitted as None.
    extra_keys: list[tuple[Any, ...] | None] | None = None
    # ATOM uses a single cache group; emitted as 0 for compat.
    group_idx: int | None = 0
    # Store events carry cache-spec metadata so consumers can classify and
    # filter groups as they are learned. Remove events only need
    # group_idx+hash. Layout matches vLLM PR vllm-project/vllm#40984.
    # ATOM emits these as None until hybrid-cache wiring lands; the wire slots
    # are reserved here so consumers built against the vLLM schema decode
    # ATOM frames unchanged.
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None


class BlockRemoved(KVCacheEvent):
    """One or more blocks were evicted from the given medium."""

    block_hashes: list[int]
    medium: str | None = MEDIUM_GPU
    group_idx: int | None = 0


class AllBlocksCleared(KVCacheEvent):
    """Entire cache (optionally restricted to one medium) was cleared.

    `medium=None` means all tiers cleared. vLLM's variant has no medium field
    so subscribers reading via vLLM's struct will ignore this attribute, which
    matches the intended "everything" semantics.
    """

    medium: str | None = None


class BlockTransferred(KVCacheEvent):
    """ATOM extension: a block moved between tiers without changing identity.

    Emitted on:
      * GPU → CPU/DISK swap (when HMA offload is wired up)
      * REMOTE → GPU receive (Mooncake/MoriIO pull on decode side)
      * GPU → REMOTE send (Mooncake/MoriIO push on prefill side)

    vLLM has no equivalent. Subscribers that don't understand this tag will
    fail to decode it — opt-in only.
    """

    block_hashes: list[int]
    from_medium: str
    to_medium: str
    group_idx: int | None = 0


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

    Topic prefix can be set so consumers can subscribe selectively
    (e.g. one topic per DP rank).
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
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=buffer_steps)

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUB)
        self._socket.set_hwm(hwm)
        self._socket.bind(endpoint)
        self._zmq_error_cls = zmq.ZMQError  # captured so _run doesn't re-import

        self._drops = 0
        self._sent = 0
        self._encode_errors = 0
        self._lock = threading.Lock()
        self._sender = threading.Thread(
            target=self._run, name="atom-kv-event-sender", daemon=True
        )
        self._sender.start()

    def publish(self, events: Iterable[KVCacheEvent]) -> None:
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
            # Encoding shouldn't fail in practice; if it does, bump a counter
            # so operators can see broken serialization in `stats` instead of
            # silently losing events.
            with self._lock:
                self._encode_errors += 1
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
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Force-poison: clear and inject.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._sender.join(timeout=2.0)
        try:
            self._socket.close(linger=0)
        except Exception:  # pragma: no cover
            pass

    # --- internal ---
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if self._topic_bytes:
                    self._socket.send_multipart([self._topic_bytes, item])
                else:
                    self._socket.send(item)
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
