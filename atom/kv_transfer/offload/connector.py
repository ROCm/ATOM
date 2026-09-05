# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public ``lmcache_offload`` connector selection and delegation shell.

The public connector name stays stable while the implementation is selected on
both the scheduler and worker from configuration alone:

* ``dense`` stores ordinary token-indexed KV chunks;
* ``hybrid`` stores DSV4 compressed PAGE chunks plus complete SLOT sidecars;
* ``kimi_k3`` stores dense MLA KV plus a KDA per-request state tier.

Keeping selection config-only is important because the scheduler process does
not have access to the worker's transfer tensors.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.offload._offload_common import StateOffloadFace
from atom.kv_transfer.offload.config import select_offload_layout

logger = logging.getLogger("atom")


def select_variant(config) -> str:
    """Return the offload family selected for ``config``."""

    return select_offload_layout(config)


def _build_worker(config):
    variant = select_variant(config)
    logger.info("lmcache_offload: worker family=%s", variant)
    if variant == "hybrid":
        from atom.kv_transfer.offload.hybrid.dsv4.connector import (
            DSV4OffloadConnector,
        )

        return DSV4OffloadConnector(config)

    if variant == "kimi_k3":
        from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
            KimiK3OffloadConnector,
        )

        return KimiK3OffloadConnector(config)

    from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

    return DenseOffloadConnector(config)


def _build_scheduler(config):
    variant = select_variant(config)
    logger.info("lmcache_offload: scheduler family=%s", variant)
    if variant == "hybrid":
        from atom.kv_transfer.offload.hybrid.dsv4.connector import (
            DSV4OffloadScheduler,
        )

        return DSV4OffloadScheduler(config)

    if variant == "kimi_k3":
        from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
            KimiK3OffloadScheduler,
        )

        return KimiK3OffloadScheduler(config)

    from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

    return DenseOffloadScheduler(config)


class LMCacheOffloadConnector(KVConnectorBase):
    """Worker-side shell delegating to the selected implementation."""

    is_producer = False

    def __init__(self, config) -> None:
        self._impl = _build_worker(config)

    @property
    def _state_tier(self):
        """Expose the implementation's KDA state tier through the shell.

        `MultiConnector._refuse_two_state_tiers` probes `_state_tier` on each
        sub-connector to find the one offload backend that built a state tier.
        Under the `multi` shell the sub IS this object, not the `kimi_k3` impl
        behind `_impl`, so without this forwarder the probe reads None on every
        member and the composite never adopts the tier that was actually built
        -- state spills silently go nowhere. The same "every probed member
        appears on the shell" contract enforced on
        `LMCacheOffloadConnectorScheduler`.
        """
        return getattr(self._impl, "_state_tier", None)

    def register_kv_caches(
        self, kv_caches, transfer_tensors=None, num_blocks=None
    ) -> None:
        self._impl.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata) -> None:
        self._impl.start_load_kv(metadata)

    def get_finished(self):
        return self._impl.get_finished()

    def get_finished_recv_blocks(self):
        return self._impl.get_finished_recv_blocks()

    def close(self) -> None:
        """Join the impl's save/load executors at worker teardown.

        `ModelRunner.exit()` resolves this through
        `getattr(connector, "close", None)` on `self.kv_connector` -- which is
        this shell. Without the forwarder that getattr returns None, so
        `OffloadWorkerMixin.close()` / `KimiK3OffloadConnector.close()` never
        run: `destroy_dist_env()` then releases the KV pool while non-daemon
        `lmc-state-store` / `lmc-state-load` / `offload-save` threads are still
        copying out of it -- use-after-free on the KV tensors, or the
        interpreter-shutdown hang `close()` exists to prevent. Every `_impl`
        defines `close` (`OffloadWorkerMixin`), so this is a plain forward, the
        same "every probed member appears on the shell" contract as `_state_tier`.
        """
        self._impl.close()


class LMCacheOffloadConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side shell delegating to the selected implementation.

    **Every member `Scheduler` reads off `self.kv_connector` has to appear here**,
    because `self.kv_connector` IS this object -- the implementation behind
    `_impl` is invisible from outside. A member the scheduler probes with
    `getattr(..., default)` and this class does not define does not raise; it
    silently takes the default forever, which is how `enqueue_state_stores`
    refused every state store from the day it was written. `tests/
    test_lmcache_offload_connector.py` sweeps for the gap rather than trusting
    this comment.
    """

    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._impl = _build_scheduler(config)

    @property
    def has_state_tier(self) -> bool:
        """True when the selected impl actually hosts the KDA state tier.

        This shell defines the whole state face (`enqueue_state_stores`,
        `take_state_reports`, `take_state_source_releases`) unconditionally,
        delegating through
        `getattr` to `_impl` and returning the no-tier default when the impl
        does not implement a method. So method *presence on the shell* cannot
        tell a tier-hosting `kimi_k3` impl from a `dense`/`hybrid` impl that has
        none -- which is exactly how `MultiConnector._first_with`, selecting by
        attribute presence, routed every state call to a dense offload shell
        whose `_impl` silently returns the no-tier defaults. Ask the impl's
        *type* instead: only `KimiK3OffloadScheduler` inherits `StateOffloadFace`
        (see there), so the check cannot be fooled by the shell's own forwards.
        """
        return isinstance(self._impl, StateOffloadFace)

    def get_num_new_matched_tokens(self, seq):
        return self._impl.get_num_new_matched_tokens(seq)

    def update_state_after_alloc(self, seq) -> None:
        self._impl.update_state_after_alloc(seq)

    def build_connector_meta(self):
        return self._impl.build_connector_meta()

    def request_finished(self, seq) -> None:
        self._impl.request_finished(seq)

    def should_park_for_load_after_alloc(self, seq) -> bool:
        return self._impl.should_park_for_load_after_alloc(seq)

    def should_defer_free(self, seq) -> bool:
        return self._impl.should_defer_free(seq)

    def release_stalled_save(self, seq) -> None:
        # Plain forward, not getattr-guarded: OffloadSchedulerMixin declares the
        # save/load lifecycle abstract, so every _impl defines all six methods or
        # fails at construction. A guard here would silently swallow a genuinely
        # missing forwarder -- the exact failure mode the abstract contract exists
        # to make loud.
        self._impl.release_stalled_save(seq)

    def has_pending_work(self) -> bool:
        return self._impl.has_pending_work()

    def save_finished(self, req_id) -> None:
        self._impl.save_finished(req_id)

    def abandon_save(self, req_id) -> None:
        self._impl.abandon_save(req_id)

    def load_failed(self, req_id):
        return self._impl.load_failed(req_id)

    def adjust_prefill_chunk_after_alloc(self, seq, chunk):
        callback = getattr(self._impl, "adjust_prefill_chunk_after_alloc", None)
        return callback(seq, chunk) if callback is not None else chunk

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        callback = getattr(self._impl, "should_park_partial_prefill_for_load", None)
        return callback(seq) if callback is not None else False

    def cancel_pending_load(self, seq) -> None:
        # Plain forward: the lifecycle contract (see release_stalled_save)
        # guarantees every _impl defines this.
        self._impl.cancel_pending_load(seq)

    def load_finished(self, req_id):
        # Plain forward: guaranteed by the abstract lifecycle contract.
        return self._impl.load_finished(req_id)

    def process_completions(self, output):
        return self._impl.process_completions(output)

    def enqueue_state_stores(self, stores) -> bool:
        callback = getattr(self._impl, "enqueue_state_stores", None)
        return bool(callback(stores)) if callback is not None else False

    def take_state_reports(self):
        # Contract is a 2-tuple (indexed, failed) -- matched by the kimi_k3 impl
        # and by the sole caller `indexed, failed = take()` in
        # scheduler._update_from_kv_xfer_finished. The no-tier fallback returned
        # a 3-tuple, so any offload variant whose _impl lacks take_state_reports
        # (e.g. dense/hybrid non-k3) but still exposes state_offload crashed the
        # engine mid-run with "too many values to unpack (expected 2)", wedging
        # all TP workers on the next collective. Return the 2-tuple empty set.
        callback = getattr(self._impl, "take_state_reports", None)
        return callback() if callback is not None else (set(), set())

    def take_missed_state_hashes(self) -> set:
        """Hashes whose state `get` missed, and which the index must forget.

        `Scheduler._update_from_kv_xfer_finished` reads this off the SHELL, so
        without the forwarder the `getattr(..., None)` default wins and a hash
        LMCache has dropped stays advertised -- parking every later request over
        that prefix against a `get` that must miss. An impl with no state tier
        misses nothing.
        """
        callback = getattr(self._impl, "take_missed_state_hashes", None)
        return callback() if callback is not None else set()

    def take_state_source_releases(self) -> set:
        """Stores whose PAGE units the GPU has finished reading.

        Its own forwarder rather than a third element of `take_state_reports`,
        for the reason recorded above: that tuple's arity is a contract with
        the caller, and widening it is exactly the change that wedged the pool
        once already. An impl without a state tier releases nothing.
        """
        callback = getattr(self._impl, "take_state_source_releases", None)
        return callback() if callback is not None else set()

    def save_abandon_timeout_s(self) -> float:
        # Plain forward: the abstract lifecycle contract guarantees every _impl
        # defines this (concrete on OffloadSchedulerMixin). The scheduler sources
        # the reclaim window from the connector because it is LMCache knowledge.
        return self._impl.save_abandon_timeout_s()

    @property
    def max_pending_saves(self) -> int | None:
        """The state leg's running-plus-queued save bound, read off the impl.

        `Scheduler._state_store_pending_cap` reads this off `self.kv_connector`
        -- which is this shell -- so it must appear here rather than only on the
        `_impl`, or the scheduler falls back to the bare env default of 2 and
        the two legs pin different amounts of the same pool.
        """
        return getattr(self._impl, "max_pending_saves", None)

    def get_statistics(self) -> dict[str, int]:
        return self._impl.get_statistics()


__all__ = [
    "LMCacheOffloadConnector",
    "LMCacheOffloadConnectorScheduler",
    "select_variant",
]
