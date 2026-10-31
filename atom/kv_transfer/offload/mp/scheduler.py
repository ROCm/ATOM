# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side LMCache MP connector for generic PAGE offload."""

from __future__ import annotations

import logging
import os
import time
import weakref
from typing import Any

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.chunked_scheduler import (
    ChunkedOffloadSchedulerBase,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    SaveSpec,
)
from atom.kv_transfer.offload.mp.deployment import (
    _extra_config,
    _make_scheduler_adapter,
    _mp_session_id,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.lookup import _MPLookupClient
from atom.kv_transfer.offload.mp.transfer import (
    _SCHEDULER_DEADLINE_MARGIN_S,
    LMCacheTransferUnprovable,
    _enforce_transfer_deadline,
    _transfer_deadline_s,
)

logger = logging.getLogger("atom")


class _StrongRef:
    """`weakref.ref`'s call interface for objects that refuse weak references."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


class LMCacheMPConnectorScheduler(ChunkedOffloadSchedulerBase):
    """Scheduler-side LMCache MP connector for generic PAGE offload."""

    _supports_early_block_release = True

    def __init__(
        self, config: Any, *, checkpoint_spec: Any = None, images: bool = True
    ) -> None:
        _validate_mp_config(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        offcfg._strict_integer(
            "LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        adapter = _make_scheduler_adapter(
            config, checkpoint_spec=checkpoint_spec, images=images
        )
        try:
            extra = _extra_config(config)
            timeout = float(extra.get("lmcache.mp.lookup_timeout", 30.0))
            poll_interval = float(extra.get("lmcache.mp.lookup_poll_interval", 0.01))
            lookup_client = _MPLookupClient(
                adapter,
                config=config,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            self._mp_adapter = adapter
            self._scheduler_deadline_s = (
                _transfer_deadline_s(config) + _SCHEDULER_DEADLINE_MARGIN_S
            )
            self._transfer_seen_at: dict[Any, float] = {}
            super().__init__(
                config,
                chunk_size=int(adapter.lmcache_tokens_per_chunk),
                lookup_client=lookup_client,
            )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        matched = super().get_num_new_matched_tokens(seq)
        sid = str(seq.id)
        num_prompt = int(seq.num_prompt_tokens)
        # The base's remembered hit, not `hit_tokens`: a step answered from the
        # memo runs no lookup, so there is no client-side state to read back.
        hit = self._last_tier_hit(seq, sid)
        if hit != num_prompt or num_prompt % self.chunk_size:
            return matched

        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            return matched
        if self.block_size == 1:
            # Allocating prompt-1 one-token blocks cannot provide a destination
            # for the complete final LMCache chunk. Recompute instead.
            self._clear_pending_load(sid)
            return 0, False

        load_spec.transfer_end_tokens = num_prompt
        self._hit_save_floors[sid] = num_prompt
        return matched

    def _new_load_operation(self, seq: Any) -> LoadOperationId:
        operation = super()._new_load_operation(seq)
        # The worker restores PAGE KV straight into allocated blocks, which
        # never pass through `hash_blocks()`. Naming where the load starts lets
        # `Scheduler._mark_offload_load_ready` publish the loaded prefix once
        # the load succeeds; without it the suffix prefill finds its parent
        # block unhashed and cannot register its own blocks either.
        # `build_connector_meta` dispatches only chunk-aligned loads from the
        # post-allocate HBM frontier, so this is always a hash-block boundary.
        seq.offload_load_start_tokens = int(seq.num_cached_tokens)
        return operation

    # A save names its LMCache chunks by the request's token prefix. Long
    # prompts are saved a chunk at a time, so shipping the whole prefix with
    # every save is quadratic in the prompt. Workers keep each request's
    # tokens (`LMCacheMPConnector._resolve_shipped_tokens`); only tokens past
    # what they already hold travel. ATOM_MP_INCREMENTAL_TOKENS=0 ships whole
    # prefixes again.
    _INCREMENTAL_TOKENS = os.environ.get("ATOM_MP_INCREMENTAL_TOKENS", "1") == "1"

    def _ship_tokens(self, seq: Any, end: int) -> tuple[int, list[int]]:
        """(worker-held prefix length, tokens to append) for prefix ``[0, end)``."""
        shipped = self.__dict__.setdefault("_shipped_tokens", {})
        entry = shipped.get(seq.id)
        base = 0
        if entry is not None:
            if entry[0]() is seq:
                base = entry[1]
            else:
                self._drop_shipped_tokens(seq.id)
        new = max(base, end)
        delta = list(seq.token_ids[base:new]) if new > base else []
        try:
            ref = weakref.ref(seq)
        except TypeError:  # test doubles without weakref support
            ref = _StrongRef(seq)
        shipped[seq.id] = (ref, new)
        return base, delta

    def _drop_shipped_tokens(self, req_id: Any) -> None:
        shipped = self.__dict__.setdefault("_shipped_tokens", {})
        if shipped.pop(req_id, None) is not None:
            self.__dict__.setdefault("_shipped_drops", []).append(req_id)

    def _build_save_request(
        self, seq, saved, aligned, operation, block_ids, is_last_prefill
    ):
        if not self._INCREMENTAL_TOKENS:
            return super()._build_save_request(
                seq, saved, aligned, operation, block_ids, is_last_prefill
            )
        base, delta = self._ship_tokens(seq, aligned)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=delta,
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
            token_base=base,
            token_end=aligned,
        )

    def _build_load_request(self, seq, load_spec, load_operation, transfer_end):
        if not self._INCREMENTAL_TOKENS:
            return super()._build_load_request(
                seq, load_spec, load_operation, transfer_end
            )
        base, delta = self._ship_tokens(seq, transfer_end)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=delta,
            block_ids=list(seq.block_table),
            load_spec=load_spec,
            load_operation=load_operation,
            token_base=base,
            token_end=transfer_end,
        )

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        metadata = super().build_connector_meta()
        drops = self.__dict__.get("_shipped_drops")
        if drops and metadata.requests:
            # Only metadata with requests is sure to reach the workers; an
            # empty snapshot may be dropped, so the drops wait for one.
            metadata.token_cache_drops = drops
            self._shipped_drops = []
        for req in metadata.requests:
            if req.load_spec is not None:
                end = req.load_spec.transfer_end_tokens
                if end is None:
                    end = req.load_spec.lmcache_cached_tokens
                self._lookup_client.prepare_retrieve(
                    str(req.req_id),
                    int(req.load_spec.hbm_cached_tokens),
                    int(end),
                )
        # Also catches retirements that bypass _finish_retired_request, such as
        # a late save with nothing left to store.
        self._end_finished_sessions()
        return metadata

    def load_finished(self, req_id: Any) -> bool:
        finished = super().load_finished(req_id)
        if finished:
            raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=True,
                )
            except Exception:
                logger.warning(
                    "LMCache MP successful-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return finished

    def load_failed(self, req_id: Any) -> bool:
        raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
        active = self._active_load_operations.get(str(raw_id))
        if isinstance(req_id, LoadOperationId):
            is_current = active is not None and active[1] == req_id
        else:
            is_current = active is None
        if is_current:
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=False,
                )
            except Exception:
                logger.warning(
                    "LMCache MP failed-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return super().load_failed(req_id)

    def waits_for_transfer_report(self, seq: Any) -> bool:
        """MP releases memory only on a report; see `_enforce_transfer_deadlines`.

        The engine's clock-based reclaim cannot prove the MP server stopped
        reading or writing, so it leaves MP requests alone instead of logging
        them as abandoned or wedged; a report that never comes stops the engine.
        """
        del seq
        return True

    def abandon_save(self, req_id: Any) -> None:
        # Reached only through a composite connector, which still abandons its
        # other legs: an MP save keeps its lease until a terminal report.
        logger.debug(
            "LMCache MP keeps the source of request %s leased until its save "
            "reports (deadline %.0fs)",
            req_id,
            self._scheduler_deadline_s,
        )

    def reclaim_stale_leases(self, timeout_s: float) -> list[frozenset]:
        del timeout_s  # See abandon_save: no release without a report.
        return []

    def _live_transfers(self) -> set[Any]:
        """Dispatched operations still waiting for a terminal worker report."""
        live: set[Any] = set(self._save_inflight.values())
        live.update(self._save_operation_owner)
        live.update(operation for _, operation in self._active_load_operations.values())
        return live

    def _enforce_transfer_deadlines(self) -> None:
        """Fail-stop when a dispatched transfer never reports.

        The worker bounds what it submitted; this catches what it cannot see,
        such as a lost completion or a TP rank that never reports.
        """
        live = self._live_transfers()
        seen = self._transfer_seen_at
        for operation in [op for op in seen if op not in live]:
            del seen[operation]
        now = time.monotonic()
        for operation in live:
            _enforce_transfer_deadline(
                operation, seen.setdefault(operation, now), self._scheduler_deadline_s
            )

    def process_completions(self, output: KVConnectorOutput) -> KVConnectorOutput:
        output = super().process_completions(output)
        self._enforce_transfer_deadlines()
        return output

    def request_finished(self, seq: Any) -> None:
        super().request_finished(seq)
        self._drop_shipped_tokens(seq.id)
        self._pending_session_ends()[str(seq.id)] = seq
        self._end_finished_sessions()

    def _finish_retired_request(self, sid: str) -> None:
        super()._finish_retired_request(sid)
        self._end_finished_sessions()

    def has_pending_work(self) -> bool:
        """Also report pending session ends and unreleased discarded lookups.

        A detached lookup can outlive its request. If no other request is
        running, nothing else steps the connector, so without this the lookup
        is never pumped and its session never ends. Both are bounded: a
        discarded lookup is given up after `_ORPHAN_TIMEOUT_S`, and a session
        waits only for its saves and its lookup.
        """
        orphans = getattr(self._lookup_client, "has_orphans", None)
        return (
            super().has_pending_work()
            or bool(self._pending_session_ends())
            or bool(callable(orphans) and orphans())
        )

    def _pending_session_ends(self) -> dict[str, Any]:
        """Finished requests whose MP session must outlive their last save.

        A late save is emitted, and submitted under the request's session,
        after request_finished, so the session ends only once none can follow.
        """
        return self.__dict__.setdefault("_sessions_to_end", {})

    def _end_finished_sessions(self) -> None:
        """End each finished request's session once no save or lookup remains.

        A detached lookup (`detach_lookup`) can outlive its request. If
        `end_session` ran first, the later `free_lookup_locks` would find no
        prefetch record in a recreated session and release only the
        full-attention groups; the state groups' read locks would pin their
        L1 objects until the read TTL. So the session ends only after its
        lookup has answered and released its locks.
        """
        # Pumped on every pass: a detached lookup must be released even while
        # its request is still transferring and the waiting queue, which
        # drives `prefetch_lookups`, is empty.
        client = self._lookup_client
        pump = getattr(client, "pump", None)
        if callable(pump):
            pump()
        self.reconcile_detached_lookups()
        pending = self._pending_session_ends()
        in_flight = getattr(client, "is_pending", None)
        for sid, seq in list(pending.items()):
            entry = self._save_tracker.get(sid)
            if (entry is not None and entry[0] is seq) or sid in self._save_inflight:
                continue
            if callable(in_flight) and in_flight(sid):
                continue
            del pending[sid]
            try:
                self._mp_adapter.end_session(_mp_session_id(self._config, seq.id))
            except Exception:
                logger.warning(
                    "LMCache MP end_session failed for request %s",
                    seq.id,
                    exc_info=True,
                )


__all__ = ["LMCacheMPConnectorScheduler", "LMCacheTransferUnprovable"]
