# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side transfers for the Kimi-K3 recurrent-state offload tier.

Two directions, and they are deliberately asymmetric.

The **load** is synchronous here and has no queue, no executor and no report of
its own: it runs inline inside the connector's own load task, immediately after
that task's KV leg, so one task produces one completion for both legs. That is
the whole design -- the previous implementation gave the state leg a second
completion channel and then needed a park to reconcile the two, and every
load-path defect it shipped lived in that reconciliation.

The **store** cannot ride a request, because a checkpoint outlives the request
that produced it. So it keeps an executor and a completion channel, and reports
two phases of one operation: the PAGE units are released as soon as the device
has stopped reading them, while whether the CPU put succeeded is known only
afterwards.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger("atom")


class StateOffloadTier:
    """Moves one recurrent-state image between an Active Slot and LMCache.

    Reports, never applies: the index this feeds lives in the engine process,
    and the engine applies these outcomes when it drains the connector output.
    """

    def __init__(self, codec, *, max_workers: int = 1) -> None:
        self.codec = codec
        # Stores only. A load runs on the calling load task's own thread, so it
        # cannot queue behind a store here -- which is what a separate store
        # executor buys: a ~55 MiB gather+D2H in front of a load is TTFT.
        self._store_executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="lmc-state-store"
        )
        self._inflight: set[Future] = set()
        # Terminal store events, drained by the connector into completions.
        # `(op_id, released?)` is one record in two phases; see the module
        # docstring.
        self._source_released: set = set()
        self._stored: set = set()
        self._store_failed: set = set()
        # Hashes whose `get` itself missed, as opposed to a load that failed for
        # some other reason. Only this may retract a hash from the engine's
        # index: a fused verdict of "failed" may mean the KV leg, and retracting
        # on that would permanently deny state bytes that are still present.
        self._missed: set[int] = set()

    # -------------------------------- load --------------------------------- #
    def load_state(self, prefix_hash: int, slot: int) -> bool:
        """Fetch `prefix_hash` into `slot`. Synchronous, on the caller's thread.

        A miss is a normal path, not an error: LMCache's LRU can drop bytes
        under a hash the engine's index still advertises. Retracting that claim
        is the engine's job; `take_missed_hashes` is how it learns of it.
        """
        try:
            ok = bool(self.codec.get(int(prefix_hash), int(slot)))
        except Exception:
            # Deliberately blind: `codec.get` reaches into LMCache, whose
            # failure modes are its own. A load that cannot happen must cost one
            # request a recompute, not this worker thread -- whose death would
            # strand every request parked on a later load.
            logger.warning(
                "state offload: load of hash %d failed", prefix_hash, exc_info=True
            )
            ok = False
        if not ok:
            self._missed.add(int(prefix_hash))
        return ok

    def take_missed_hashes(self) -> set[int]:
        missed, self._missed = self._missed, set()
        return missed

    # -------------------------------- store -------------------------------- #
    def submit_store(self, op_id, prefix_hash: int, unit_ids) -> None:
        future = self._store_executor.submit(
            self._do_store, op_id, int(prefix_hash), tuple(unit_ids)
        )
        self._inflight.add(future)
        future.add_done_callback(self._inflight.discard)

    def _do_store(self, op_id, prefix_hash: int, unit_ids) -> None:
        released = False

        def _source_released() -> None:
            # Fires from `codec.put` once the gather and its D2H have drained,
            # before the CPU put: the device has stopped reading the units, so
            # hand them back now rather than hold a whole image out of the KV
            # pool across the put. Guarded so the backstop below cannot emit a
            # second release -- that would double-unpin engine-side.
            nonlocal released
            if released:
                return
            released = True
            self._source_released.add(op_id)

        stored = False
        try:
            stored = bool(
                self.codec.put(
                    prefix_hash, unit_ids, on_source_released=_source_released
                )
            )
        except Exception:
            # Blind for the same reason as the load: a store that cannot happen
            # costs one checkpoint's CPU copy, never the worker.
            logger.warning(
                "state offload: store of hash %d failed", prefix_hash, exc_info=True
            )
        # Backstop, not the primary path. On success the callback has already
        # published the release. This fires only where the callback was never
        # reached -- a refused allocation never read the units, and a throwing
        # gather has already fenced its streams -- and withholding it there would
        # hold an image out of the pool until the reclaimer noticed.
        if not released:
            released = True
            self._source_released.add(op_id)
        (self._stored if stored else self._store_failed).add(op_id)

    def take_store_reports(self) -> tuple[set, set, set]:
        """`(source released, stored, store failed)` since the last call."""
        src, self._source_released = self._source_released, set()
        ok, self._stored = self._stored, set()
        bad, self._store_failed = self._store_failed, set()
        return src, ok, bad

    # ------------------------------ lifecycle ------------------------------ #
    def has_inflight(self) -> bool:
        return bool(self._inflight) or bool(
            self._source_released or self._stored or self._store_failed
        )

    def drain(self) -> None:
        for future in list(self._inflight):
            try:
                future.result()
            except Exception:
                logger.warning(
                    "state offload: store failed during drain", exc_info=True
                )

    def shutdown(self) -> None:
        self.drain()
        self._store_executor.shutdown(wait=True)
