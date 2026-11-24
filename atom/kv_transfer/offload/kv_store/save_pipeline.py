# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in save pipeline: one sender thread makes a save's native calls.

The save thread packs the next batch while the sender puts the previous one;
at most ``CREDITS`` batches are preparing or in flight. A save reports its
terminal only after :meth:`Sender.drain_request` saw every put return.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.offload.kv_store.client import CallRefused, KVStoreClient

if TYPE_CHECKING:
    from atom.kv_transfer.offload.kv_store.pool import Slot, TransferSlotPool
    from atom.kv_transfer.offload.kv_store.worker import KVStoreOffloadConnector
    from atom.kv_transfer.offload.metadata import LMCacheReqMeta

MAX_BATCH_CHUNKS = 64
CREDITS = 2


class StreamStopped(BaseException):
    """The sender is closed, or an earlier put of this save failed.

    Not an ``Exception``, so the serial code's ``except Exception`` handlers
    (a failed lookup is put whole) cannot absorb it.
    """


class SettlementTransaction:
    """The releases and quarantines a settlement hook asks for, applied together.

    The slots stay leased until the hook has returned and ``commit`` applies
    every action; a hook that raises changes nothing.
    """

    def __init__(self, pool: TransferSlotPool) -> None:
        self.pool = pool
        self.actions: list[tuple[Callable[..., None], list[Slot], dict[str, str]]] = []

    def release(self, slots: list[Slot]) -> None:
        self.actions.append((self.pool.release, list(slots), {}))

    def quarantine(self, slots: list[Slot], **kwargs: str) -> None:
        self.actions.append((self.pool.quarantine, list(slots), kwargs))

    def commit(self) -> None:
        for apply, slots, kwargs in self.actions:
            apply(slots, **kwargs)


@dataclass
class Ticket:
    request: LMCacheReqMeta
    run: range
    keys: list[str]
    slots: list[Slot]
    problem: str | None = None
    kind: str = "put_failed"
    native_ms: float = 0.0
    stored_chunks: int = 0
    returned_chunks: int = 0


@dataclass
class RequestResult:
    outstanding: int = 0
    returned_chunks: int = 0
    stored_chunks: int = 0
    native_ms: float = 0.0
    problem: str | None = None
    kind: str = "put_failed"
    read_from: int | None = None  # the save's lowest chunk read or found held


class Lookahead:
    """The ``client`` of ``_save_segments`` under the pipeline.

    Its ``exists`` queues the next window's lookup before it waits for this
    one's, so the lookup runs ahead of the last put. Once the save has failed
    (or the sender closed) it stops the save; only a lookup it already queued
    still runs.
    """

    def __init__(
        self,
        sender: Sender,
        key: Any,
        keys_of: Callable[[int], list[str]],
        count: int,
    ) -> None:
        self.sender = sender
        self.key = key
        self.keys_of = keys_of
        self.count = count
        self.index = 0
        self.ahead: Future | None = None

    def exists(self, keys: list[str]) -> list[int]:
        reason = self.sender._stop_reason(self.key)
        if reason is not None:
            raise StreamStopped(reason)
        future = self.ahead or self.sender.start_lookup(keys)
        self.index += 1
        self.ahead = None
        if self.index < self.count:
            self.ahead = self.sender.start_lookup(self.keys_of(self.index))
        return self.sender.finish_lookup(future)


class Sender:
    def __init__(self, owner: KVStoreOffloadConnector) -> None:
        self.owner = owner
        self.credits = threading.BoundedSemaphore(CREDITS)
        self.jobs: queue.Queue[Ticket | tuple[Future, list[str]] | None] = queue.Queue()
        # Only the one save thread reserves, submits and releases.
        self.preparing = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.pending = 0
        self.closed = False
        self.stop_sent = False
        self.results: dict[Any, RequestResult] = {}
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="store-save-sender"
        )
        self.thread.start()

    @staticmethod
    def _key(req: LMCacheReqMeta) -> Any:
        return req.save_operation

    def _stop_reason(self, key: Any) -> str | None:
        """Why no further batch of ``key`` is prepared."""
        with self.lock:
            return "closed" if self.closed else self.results[key].problem

    def begin_request(self, req: LMCacheReqMeta) -> None:
        with self.lock:
            self.results[self._key(req)] = RequestResult()

    def start_lookup(self, keys: list[str]) -> Future:
        """Queue ``keys``' lookup behind earlier jobs (it takes no credit)."""
        future: Future = Future()
        with self.lock:
            if self.closed:
                raise StreamStopped("closed")
            self.jobs.put((future, list(keys)))
        return future

    @staticmethod
    def finish_lookup(future: Future) -> list[int]:
        return future.result()

    def lookahead(self, req: LMCacheReqMeta, windows: list[range]) -> Lookahead:
        keys = self.owner._window_keys
        return Lookahead(
            self, self._key(req), lambda i: keys(req, windows[i]), len(windows)
        )

    def reserve(self, req: LMCacheReqMeta) -> None:
        """Take one credit for a batch the calling thread is about to prepare."""
        # Credits are held only by this save's own batches, each returned once
        # its put returns, so this wait ends; the stop is checked after it.
        self.credits.acquire()
        with self.lock:
            self.preparing = True
            self.pending += 1
        reason = self._stop_reason(self._key(req))
        if reason is not None:
            self.release_preparing()
            raise StreamStopped(reason)

    def release_preparing(self) -> None:
        """Give back the credit of a batch that will not be submitted."""
        if self.preparing:
            self.preparing = False
            self._release_credit()

    def _release_credit(self) -> None:
        with self.lock:
            self.pending -= 1
        self.credits.release()

    def submit(
        self, req: LMCacheReqMeta, run: range, keys: list[str], slots: list[Slot]
    ) -> None:
        """Queue a packed batch; its GPU copy has returned."""
        assert self.preparing and len(run) == len(slots)
        ticket = Ticket(req, run, keys, slots)
        with self.lock:
            self.results[self._key(req)].outstanding += 1
            self.preparing = False
            self.jobs.put(ticket)

    def _safe_distrust(self, why: str, ticket: Ticket) -> None:
        # Whatever `_distrust_store` raises must not stop the sender draining.
        try:
            self.owner._distrust_store(why)
        except BaseException as exc:  # noqa: BLE001 - preserve native drain
            ticket.problem = f"{ticket.problem or why}; distrust failed: {exc!r}"

    def _run(self) -> None:
        while True:
            item = self.jobs.get()
            try:
                if item is None:
                    return
                if isinstance(item, tuple):
                    future, keys = item
                    try:
                        future.set_result(self.owner._registered()[1].exists(keys))
                    except BaseException as exc:  # noqa: BLE001 - to the waiter
                        future.set_exception(exc)
                else:
                    self._put(item)
            finally:
                self.jobs.task_done()

    def _put(self, t: Ticket) -> None:
        pool = None
        try:
            pool, client, _ = self.owner._registered()
            with self.lock:
                reason = self.results[self._key(t.request)].problem
            if reason is not None:
                # An earlier put failed; nothing reads these slots any more.
                settlement = SettlementTransaction(pool)
                settlement.release(t.slots)
                self._settle(t, settlement)
            else:
                self._put_native(t, pool, client)
        except BaseException as exc:  # noqa: BLE001 - retain unsettled slots
            # Registration, a release, the settlement hook or its commit, or
            # the quarantine and warning after a native put raised.
            t.problem = t.problem or f"put handling raised {exc!r}"
            if pool is not None:
                self._quarantine_remaining(t, pool)
            self._safe_distrust(f"put handling raised {exc!r}", t)
        finally:
            with self.condition:
                state = self.results[self._key(t.request)]
                state.outstanding -= 1
                state.returned_chunks += t.returned_chunks
                state.stored_chunks += t.stored_chunks
                state.native_ms += t.native_ms
                if state.problem is None:
                    state.problem, state.kind = t.problem, t.kind
                self.condition.notify_all()
            self._release_credit()

    def _put_native(
        self, t: Ticket, pool: TransferSlotPool, client: KVStoreClient
    ) -> None:
        call_start = time.perf_counter()
        try:
            answer = self.owner._put_chunks(client, t.request, t.run, t.keys, t.slots)
        except CallRefused as exc:
            t.problem, t.kind = f"refused: {exc}", "put_refused"
            settlement = SettlementTransaction(pool)
            settlement.release(t.slots)
            self._settle(t, settlement)
            return
        except BaseException as exc:  # noqa: BLE001 - the NIC may still use the buffers
            t.problem, t.kind = f"native exception: {exc!r}", "put_raised"
            settlement = SettlementTransaction(pool)
            settlement.quarantine(t.slots, reason="native put raised")
            self._settle(t, settlement)
            self.owner._warn_rate_limited(
                "save:native_put_exception",
                "%s: a native put raised %r, req=%s",
                self.owner.LABEL,
                exc,
                t.request.req_id,
                exc_info=True,
            )
            self._safe_distrust(f"a put raised {exc!r}", t)
            return
        finally:
            t.native_ms = (time.perf_counter() - call_start) * 1000
        t.returned_chunks = len(t.slots)
        settlement = SettlementTransaction(pool)
        t.problem = self.owner._settle_put(settlement, t.slots, answer)
        self._settle(t, settlement)
        if t.problem is None:
            t.stored_chunks = len(t.slots)

    def _settle(self, t: Ticket, settlement: SettlementTransaction) -> None:
        """Hand ``t``'s slots back as ``settlement`` says, atomically.

        Under the pool's lock: if an action raises after an earlier one
        released a slot, the save thread cannot lease that slot for its next
        batch before the recovery runs, so the recovery quarantines only what
        ``t`` still holds. The failure is then the ticket's, as in ``_put``.
        """
        with settlement.pool.holding():
            try:
                settlement.commit()
                return
            except BaseException as exc:  # noqa: BLE001 - recover, then report
                failure = exc
                self._quarantine_remaining(t, settlement.pool)
        t.problem = t.problem or f"put handling raised {failure!r}"
        self._safe_distrust(f"put handling raised {failure!r}", t)

    def _quarantine_remaining(self, t: Ticket, pool: TransferSlotPool) -> None:
        # The put's handling failed: do not infer that the NIC is idle.
        remaining = [slot for slot in t.slots if slot.leased]
        try:
            pool.quarantine(remaining, reason="put handling raised")
        except BaseException as exc:  # noqa: BLE001
            t.problem = f"{t.problem}; quarantine recovery failed: {exc!r}"
            with self.lock:
                self.closed = True

    def drain_request(self, req: LMCacheReqMeta) -> RequestResult:
        """Wait until every put of ``req`` has returned and been settled."""
        with self.condition:
            state = self.results[self._key(req)]
            while state.outstanding:
                self.condition.wait()
            return state

    def note_read_from(self, req: LMCacheReqMeta, read_from: int) -> None:
        with self.lock:
            self.results[self._key(req)].read_from = read_from

    def read_from(self, req: LMCacheReqMeta) -> int | None:
        with self.lock:
            return self.results[self._key(req)].read_from

    def forget_request(self, req: LMCacheReqMeta) -> None:
        with self.lock:
            key = self._key(req)
            self.results.pop(key, None)

    def close(self, timeout: float = 2) -> bool:
        """Stop the thread once its queue is empty and no batch is preparing.

        False if that takes longer than ``timeout`` (the thread keeps running;
        keep the pool and client open). Once drained nothing can queue behind
        the stop sentinel, so the exit is awaited without a deadline: True.
        """
        with self.lock:
            self.closed = True
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.pending == 0 and self.jobs.unfinished_tasks == 0:
                    if not self.stop_sent:
                        self.jobs.put(None)
                        self.stop_sent = True
                    break
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        self.thread.join()
        return True
