# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The store tier of the KV events: the chunks of this engine's namespace the
Mooncake Store holds, relayed from the KV events of its pools' masters
(Mooncake#2214, run with ``-enable_kv_events -kv_events_emit_object_key``).

A chunk is held once the object of every rank of it is stored, in any medium,
and dropped once one of them is removed from every medium; the master evicts
a chunk's group whole. It is named by the first 8 bytes of its digest,
big-endian, as the Atomesh router names it. The relay's thread follows the
masters, and the scheduler thread publishes what it saw (``drain``) among its
own cache's KV events, with medium STORE. The Store's taint marker, or
messages a master's stream lost, clear the tier: what was stored before is
not known again.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import msgspec
import zmq

from atom.distributed.kv_events import (
    MEDIUM_STORE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from atom.kv_transfer.offload.mooncake_store.keys import DIGEST_BYTES, STORE_TAINT_KEY

logger = logging.getLogger("atom")

_POLL_MS = 100


def chunk_name(digest_hex: str) -> int:
    """The store tier's name of the chunk whose digest is ``digest_hex``."""
    return int(digest_hex[:16], 16)


class StoreTier:
    """What one namespace's masters say the Store holds, as KV events."""

    def __init__(self, namespace: str, world: int, chunk_tokens: int) -> None:
        self._prefix = f"{namespace}/w"
        self._world_suffix = f"of{world}"
        self._world = world
        self._chunk_tokens = chunk_tokens
        # Object key -> the media holding it.
        self._objects: dict[str, set[str]] = {}
        # Digest -> the ranks whose object of the chunk is held.
        self._ranks: dict[str, int] = {}
        self._tainted = False
        self._events: list[KVCacheEvent] = []

    def apply(self, event: Any) -> None:
        """An event of a master's batch: a map with ``event_type``."""
        if not isinstance(event, dict):
            return
        kind = event.get("event_type")
        if kind == "cleared":
            self.clear()
            return
        key, medium = event.get("object_key"), str(event.get("medium"))
        if kind not in ("stored", "removed") or not isinstance(key, str):
            return
        if key == STORE_TAINT_KEY:
            self._taint(kind == "stored")
            return
        rank, _, digest = key.removeprefix(self._prefix).partition("/")
        if (
            not key.startswith(self._prefix)
            or not rank.endswith(self._world_suffix)
            or len(digest) != 2 * DIGEST_BYTES
        ):
            return
        media = self._objects.get(key)
        if kind == "stored":
            if media is None:
                media = self._objects[key] = set()
                self._ranks[digest] = self._ranks.get(digest, 0) + 1
                if self._ranks[digest] == self._world:
                    self._emit(BlockStored, digest)
            media.add(medium)
        elif media is not None and medium in media:
            media.discard(medium)
            if media:
                return
            del self._objects[key]
            if self._ranks[digest] == self._world:
                self._emit(BlockRemoved, digest)
            self._ranks[digest] -= 1
            if not self._ranks[digest]:
                del self._ranks[digest]

    def clear(self) -> None:
        """Nothing is known to be held: what a master said before is lost,
        whether the Store is tainted too."""
        self._objects.clear()
        self._ranks.clear()
        self._tainted = False
        self._events = [AllBlocksCleared(medium=MEDIUM_STORE)]

    def take(self) -> list[KVCacheEvent]:
        events, self._events = self._events, []
        return events

    def _taint(self, tainted: bool) -> None:
        # A tainted Store loads nothing, so it holds nothing for the router.
        if tainted == self._tainted:
            return
        self._tainted = tainted
        if tainted:
            self._events = [AllBlocksCleared(medium=MEDIUM_STORE)]
            return
        for digest, ranks in self._ranks.items():
            if ranks == self._world:
                self._emit(BlockStored, digest)

    def _emit(self, kind: type, digest: str) -> None:
        if self._tainted:
            return
        name = chunk_name(digest)
        last = self._events[-1] if self._events else None
        if type(last) is kind:
            last.block_hashes.append(name)
        elif kind is BlockStored:
            self._events.append(
                BlockStored(
                    block_hashes=[name],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=self._chunk_tokens,
                    lora_id=None,
                    medium=MEDIUM_STORE,
                    lora_name=None,
                )
            )
        else:
            self._events.append(BlockRemoved(block_hashes=[name], medium=MEDIUM_STORE))


class StoreTierRelay:
    """Follows the masters' KV event endpoints into ``tier`` on a thread."""

    def __init__(self, endpoints: list[str], tier: StoreTier) -> None:
        self._tier = tier
        self._out: deque[KVCacheEvent] = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(list(endpoints),), name="store-tier", daemon=True
        )
        self._thread.start()

    def drain(self) -> list[KVCacheEvent]:
        """The tier's events since the last call, in order."""
        events = []
        while self._out:
            events.append(self._out.popleft())
        return events

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self, endpoints: list[str]) -> None:
        context = zmq.Context()
        poller = zmq.Poller()
        last_seq: dict[zmq.Socket, int | None] = {}
        try:
            for endpoint in endpoints:
                socket = context.socket(zmq.SUB)
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(endpoint)
                poller.register(socket, zmq.POLLIN)
                last_seq[socket] = None
            while not self._stop.is_set():
                for socket, _ in poller.poll(_POLL_MS):
                    frames = socket.recv_multipart()
                    seq = int.from_bytes(frames[1], "big") if len(frames) == 3 else None
                    last, last_seq[socket] = last_seq[socket], seq
                    if seq is None or (last is not None and seq != last + 1):
                        logger.warning(
                            "Mooncake Store offload: a master's KV events lost "
                            "messages (seq %s after %s); clearing the store tier",
                            seq,
                            last,
                        )
                        self._tier.clear()
                    if seq is not None:
                        self._apply(frames[2])
                self._out.extend(self._tier.take())
        except Exception:
            logger.exception("Mooncake Store offload: the store tier's relay stopped")
            self._tier.clear()
            self._out.extend(self._tier.take())
        finally:
            context.destroy(linger=0)

    def _apply(self, payload: bytes) -> None:
        """A master's batch: ``[timestamp, [event, ...], dp_rank]``."""
        try:
            batch = msgspec.msgpack.decode(payload)
        except msgspec.DecodeError:
            batch = None
        if (
            not isinstance(batch, list)
            or len(batch) < 2
            or not isinstance(batch[1], list)
        ):
            logger.warning("Mooncake Store offload: an unreadable KV event batch")
            self._tier.clear()
            return
        for event in batch[1]:
            self._tier.apply(event)
