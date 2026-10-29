# SPDX-License-Identifier: MIT
"""Scheduler-local prepare leases and source-side execution arbitration.

All methods run on the scheduler thread. Shadows never enter waiting/running
before commit. Source release and source allocation cannot interleave.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrepareMessage:
    kind: str
    request_id: object
    generation: int
    seq: Any = None
    chain: Any = None
    minimum_hit: int = 0
    ttl: float = 0.0
    boundary: int = 0
    hbm_hit: int = 0
    forward_start: int | None = None


@dataclass
class _Target:
    seq: Any
    deadline: float
    minimum_hit: int
    boundary: int | None = None


class CachePreparation:
    def __init__(self, host):
        self.host = host
        self.emit: Callable[[PrepareMessage], None] | None = None
        self.sources: dict[object, Any] = {}
        self.targets: dict[object, _Target] = {}
        self._last_target_generation = 0

    def _emit(self, kind, seq, **values):
        if self.emit is not None:
            self.emit(
                PrepareMessage(kind, seq.id, seq.dp_cache.preparation_id, **values)
            )

    def register_source(self, seq):
        state = seq.dp_cache
        if state is not None and state.preparation_role == "source":
            self.sources[seq.id] = seq
            state.prepare_deadline = time.monotonic() + state.prepare_timeout
            self._emit("queued", seq)

    def source_started(self, seq):
        if self.sources.get(seq.id) is not seq:
            return
        self.sources.pop(seq.id)
        self._emit("source_kept", seq)
        seq.dp_cache.preparation_role = ""
        seq.dp_cache.migration_id = 0
        seq.dp_cache.minimum_hit = 0

    def _drop_target(self, request_id):
        entry = self.targets.pop(request_id, None)
        if entry is not None:
            self.host.kv_connector.cancel_recovery(entry.seq)
        return entry

    def handle(self, message: PrepareMessage):
        sid, generation = message.request_id, message.generation
        if message.kind in {"release_source", "keep_source"}:
            seq = self.sources.get(sid)
            if seq is None or seq.dp_cache.preparation_id != generation:
                return
            if message.kind == "keep_source":
                self.source_started(seq)
            elif seq in self.host.waiting and not seq.block_table:
                # No KV allocation, forward or DMA was allowed before this
                # decision. Existing request cleanup releases any prefetch pin.
                self.host.waiting.remove(seq)
                self.sources.pop(sid)
                callback = getattr(self.host.kv_connector, "request_finished", None)
                if callable(callback):
                    callback(seq)
                self.host.block_manager.deallocate(seq)
                self._emit("source_released", seq)
            else:
                self.source_started(seq)
            return

        if message.kind == "prepare":
            # A monotonically increasing coordinator ticket prevents a delayed
            # duplicate PREPARE+COMMIT from reviving an already finished request.
            # Out-of-order older offers may safely lose to their source.
            if generation <= self._last_target_generation or sid in self.targets:
                return
            self._last_target_generation = generation
            seq = message.seq
            seq.prompt_hash_chain = message.chain
            ttl_fn = getattr(self.host.kv_connector, "recovery_prepare_ttl", None)
            ttl = min(message.ttl, ttl_fn()) if callable(ttl_fn) else 0.0
            if ttl <= 0 or self.emit is None:
                self._emit("failed", seq)
                return
            self.targets[sid] = _Target(
                seq, time.monotonic() + ttl, message.minimum_hit
            )
            return

        entry = self.targets.get(sid)
        if entry is None or entry.seq.dp_cache.preparation_id != generation:
            return
        if message.kind == "cancel":
            self._drop_target(sid)
        elif message.kind == "commit":
            now = time.monotonic()
            if (
                entry.boundary is None
                or now >= entry.deadline
                or not self.host.kv_connector.recovery_prepared(
                    entry.seq, entry.boundary
                )
            ):
                self._drop_target(sid)
                self._emit("failed", entry.seq)
                return
            # Transfer the SAME Sequence and its live pin to ordinary admission.
            # Expiry from now on uses the existing DMA-safe migration abort.
            self.targets.pop(sid)
            state = entry.seq.dp_cache
            state.prepare_timeout = entry.deadline - now
            state.minimum_hit = entry.minimum_hit
            state.preparation_role = ""
            self.host.add(entry.seq)
            self._emit("committed", entry.seq)

    def advance(self):
        if not self.sources and not self.targets:
            return
        now = time.monotonic()
        for seq in list(self.sources.values()):
            if seq not in self.host.waiting:
                self.sources.pop(seq.id, None)
            elif now >= seq.dp_cache.prepare_deadline:
                # Timeout never relinquishes the source's execution authority.
                self.source_started(seq)
        for sid, entry in list(self.targets.items()):
            if now >= entry.deadline:
                self._drop_target(sid)
                self._emit("failed", entry.seq)
                continue
            if entry.boundary is not None:
                continue
            boundary = self.host.kv_connector.prepare_recovery(entry.seq)
            if boundary is None:
                continue
            if boundary < entry.minimum_hit:
                self._drop_target(sid)
                self._emit("short", entry.seq, boundary=boundary)
                continue
            entry.boundary = boundary
            # HBM is a cost hint only; the credential pins the ENTIRE promised
            # tier prefix, so HBM eviction cannot invalidate this credential.
            probe = self.host._probe_hbm_prefix(entry.seq)
            hbm = (
                max(0, probe.num_cached_blocks)
                * self.host.block_manager.hash_block_size
            )
            # Use the scheduler's replay rule, including encoder replay and
            # minimum final-prefill rows. A PAGE lease is not restored STATE.
            seq = entry.seq
            was_replay = seq.replays_hit
            try:
                seq.replays_hit = True
                forward_start = self.host.block_manager.replay_start(
                    seq, boundary, hit=True
                )
            finally:
                seq.replays_hit = was_replay
            entry.minimum_hit = min(entry.minimum_hit, forward_start)
            self._emit(
                "ready",
                entry.seq,
                boundary=boundary,
                hbm_hit=hbm,
                forward_start=forward_start,
                ttl=max(0.0, entry.deadline - time.monotonic()),
            )

    def close(self):
        for sid in list(self.targets):
            self._drop_target(sid)
        self.sources.clear()
