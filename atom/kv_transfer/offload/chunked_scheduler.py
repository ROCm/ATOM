# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout-neutral scheduler policy for chunked KV-cache offload."""

from __future__ import annotations

import logging
import os
import time

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveCompletionId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    OffloadSchedulerMixin,
    validated_kv_role,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)

logger = logging.getLogger("atom")

DENSE_PAGE_SOURCE_SAFE_CHANNEL = "dense.page.source_safe"
DENSE_PAGE_STORE_CHANNEL = "dense.page.store"


class ChunkedOffloadSchedulerBase(OffloadSchedulerMixin, KVConnectorSchedulerBase):
    """Transport- and layout-neutral policy for chunk-aligned KV offload."""

    # Consumer semantics: finished_recving wakes parked seqs (the engine asserts
    # `not is_producer` on that path). Offload never uses finished_sending.
    is_producer = False
    # Opt the scheduler into offload-wake (suffix prefill) instead of the P/D
    # decode-jump in Scheduler.schedule(); see Scheduler._is_offload_connector.
    is_offload = True
    # Only transports that publish source-safe completion groups may opt in.
    _supports_early_block_release = False

    def __init__(
        self,
        config,
        *,
        chunk_size: int,
        lookup_client,
    ) -> None:
        """Initialize layout-independent chunked scheduling state.

        The standalone connector supplies LMCache's legacy lookup client. The
        multiprocess connector supplies a small adapter with the same public
        ``lookup``/``clear_lookup_status`` contract, so both transports retain
        one scheduling and exact-completion implementation.
        """
        self._init_offload_statistics()
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = offcfg._strict_integer(
            "Offload block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        self.virtual_block_size = self.block_size * int(
            getattr(config, "decode_context_parallel_size", 1) or 1
        )
        self.chunk_size = offcfg._strict_integer(
            "LMCache chunk size",
            chunk_size,
            minimum=1,
        )
        self._lookup_client = lookup_client

        # req_id -> LoadSpec (pending load decided at match time)
        self._load_specs: dict[str, LoadSpec] = {}
        # req_id -> Sequence (queued to recv this step)
        self._reqs_need_recv: dict[str, object] = {}
        # req_id -> HBM chunk frontier for an emitted load. If the load fails,
        # lower the save frontier to this value so recomputed chunks can be
        # stored again.
        self._load_save_floors: dict[str, int] = {}
        # req_id -> LMCache chunk frontier observed by lookup. The scheduler
        # should not re-save this already-persisted prefix unless a later load
        # actually fails.
        self._hit_save_floors: dict[str, int] = {}
        # Persistent save tracker: sid -> [seq, saved_offset]. A seq's prompt
        # prefix is stored to LMCache once prefill computes it
        # (seq.prefix_hashes_published flips True), chunk by chunk.
        self._save_tracker: dict[str, list] = {}
        # Round-robin cursor over `_save_tracker`: the last sid that emitted a
        # save. Subclasses may bound the number of outstanding saves through
        # `_may_emit_save`; resuming after this sid prevents starvation.
        self._save_rr_last: str | None = None
        # sid -> exact save generation.  Exact matching prevents a delayed TP
        # notification for an older request lifecycle from releasing the
        # current request's deferred blocks.
        self._save_inflight: dict[str, SaveCompletionId] = {}
        # Early block-release bookkeeping. Operation maps retain the exact
        # token-index -> block-id relationship frozen at save emission; the
        # per-request lease set exists only after request teardown transfers a
        # refcount share from the request to the save. Source-safe and
        # store-terminal are separate connector completions.
        self._early_release = bool(self._supports_early_block_release)
        self._save_operation_blocks: dict[SaveOperationId, dict[int, int]] = {}
        self._save_operation_safe: dict[SaveOperationId, set[int]] = {}
        self._save_operation_owner: dict[SaveOperationId, object] = {}
        self._save_lease_blocks: dict[int, set[int]] = {}
        self._save_lease_at: dict[int, float] = {}
        self._save_lease_owner: dict[int, object] = {}
        self._pending_source_safe_releases: list[frozenset] = []
        self._source_safe_waiting_for_store: dict[SaveOperationId, set[int]] = {}
        self._save_nonce = 0
        self._load_nonce = 0
        self._load_lifecycles: dict[str, object] = {}
        self._active_load_operations: dict[str, tuple[object, LoadOperationId]] = {}
        self._lookup_in_step: list[str] = []
        self._lookup_results: dict[str, tuple[object, int]] = {}
        self._handoff_loads: set[str] = set()
        # Unaligned handoff is always on: when the HBM prefix-cache hit is not
        # chunk-aligned, recompute the misaligned head up to the next chunk
        # boundary, then load the aligned remainder from CPU. (Previously gated
        # by the OFFLOAD_UNALIGNED_HANDOFF env var; now unconditional.)
        try:
            self._min_load_tokens = max(
                0, int(os.environ.get("OFFLOAD_MIN_LOAD_TOKENS", "8192"))
            )
        except ValueError:
            logger.warning(
                "LMCache offload scheduler: invalid OFFLOAD_MIN_LOAD_TOKENS=%r; "
                "using 8192",
                os.environ.get("OFFLOAD_MIN_LOAD_TOKENS"),
            )
            self._min_load_tokens = 8192

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def _begin_load_lifecycle(self, seq) -> None:
        sid = str(seq.id)
        previous = self._load_lifecycles.get(sid)
        if previous is not None and previous is not seq:
            self._clear_pending_load(sid)
            self._active_load_operations.pop(sid, None)
        self._load_lifecycles[sid] = seq

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if not self._do_load or self._lookup_client is None:
            return 0, False
        self._begin_load_lifecycle(seq)
        num_prompt = seq.num_prompt_tokens
        token_ids = list(seq.token_ids[:num_prompt])
        sid = str(seq.id)
        pending = self._lookup_results.get(sid)
        if pending is not None and pending[0] is not seq:
            # An older lifecycle still owns this worker-side pin. Its cleanup
            # must be dispatched before the ID can acquire a new lease.
            return 0, False
        try:
            if pending is None:
                if sid not in self._lookup_in_step:
                    self._lookup_in_step.append(sid)
                self._lookup_results[sid] = (seq, 0)
                hit = self._lookup_client.lookup(token_ids, lookup_id=sid)
                if hit is None:
                    self._lookup_results.pop(sid, None)
                else:
                    self._lookup_results[sid] = (seq, int(hit))
            else:
                hit = pending[1]
        except Exception:
            logger.exception("LMCache offload lookup failed for seq %s", seq.id)
            return 0, False
        if logger.isEnabledFor(logging.DEBUG):
            _lh = None
            try:
                tdb = getattr(self._lookup_client, "token_database", None)
                if tdb is not None:
                    _lh = [
                        k
                        for (_s, _e, k) in list(
                            tdb.process_tokens(token_ids, make_key=False)
                        )[:3]
                    ]
            except Exception as e:  # noqa: BLE001  # debug-only introspection
                _lh = f"err:{e}"
            logger.debug(
                "[OFFLOAD-LOOKUP] seq=%s num_prompt=%d hbm_cached=%d hit=%s lookuphash3=%s",
                seq.id,
                num_prompt,
                int(seq.num_cached_tokens),
                hit,
                _lh,
            )
        if not hit:
            return 0, False
        hit = int(hit)
        if hit == num_prompt:  # full-prompt hit → recompute last token
            hit -= 1
        self._hit_save_floors[sid] = self._chunk_floor(hit)
        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            self._clear_pending_load(sid)
            self._hit_save_floors[sid] = self._chunk_floor(hit)
            return 0, False
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=hit,
            can_load=False,
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        self._begin_load_lifecycle(seq)
        sid = str(seq.id)
        ls = self._load_specs.get(sid) if self._do_load else None
        logger.debug(
            "[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
            seq.id,
            ls is not None,
            int(getattr(seq, "num_cached_tokens", -1)),
        )
        if ls is not None:
            ls.can_load = True
            self._reqs_need_recv[sid] = seq
        # Track for save; build_connector_meta stores chunks once the scheduler's
        # computed frontier (seq.num_cached_tokens) has advanced past them.
        #
        # If LMCache lookup already found a prefix for this request, do not save
        # that prefix again. This covers both direct loads and the
        # hbm_satisfies_after_alloc case where HBM prefix cache already covers
        # the lookup hit. Only suffix chunks computed by this request should be
        # stored.
        initial_saved = max(
            self._lmcache_hit_save_floor(ls),
            int(self._hit_save_floors.get(sid, 0)),
        )
        if self._do_save:
            entry = self._save_tracker.get(sid)
            if entry is None or entry[0] is not seq:
                self._save_tracker[sid] = [seq, initial_saved]
            else:
                entry[1] = max(int(entry[1]), initial_saved)

    def _clear_pending_load(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.discard(sid)
        self._load_save_floors.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        # clear_lookup_status only clears the client's memo; it does not
        # release worker pins. Keep the ID for the next metadata dispatch.
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:
                logger.debug(
                    "LMCache offload: lookup status cleanup failed for req=%s",
                    sid,
                    exc_info=True,
                )

    def _decide_load_after_alloc(
        self, seq, ls: LoadSpec
    ) -> tuple[bool, str, int, int, int, int]:
        hbm = int(getattr(seq, "num_cached_tokens", ls.hbm_cached_tokens))
        lmc = int(ls.lmcache_cached_tokens)
        ls.hbm_cached_tokens = hbm
        chunk = int(self.chunk_size or 256)
        need = lmc - hbm
        if lmc <= hbm:
            return False, "hbm_satisfies_after_alloc", hbm, lmc, need, chunk
        if hbm % chunk != 0:
            return False, "unaligned_hbm_prefill", hbm, lmc, need, chunk
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if need < min_load:
            return False, "too_small", hbm, lmc, need, chunk
        return True, "aligned_large_hit", hbm, lmc, need, chunk

    def adjust_prefill_chunk_after_alloc(self, seq, chunk: int) -> int:
        sid = str(seq.id)
        if sid not in self._handoff_loads:
            return chunk
        boundary = getattr(seq, "offload_handoff_boundary_tokens", None)
        if boundary is None:
            return chunk
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        limit = int(boundary) - hbm
        if limit <= 0:
            return chunk
        adjusted = min(int(chunk), limit)
        return max(1, adjusted)

    def _may_emit_save(self) -> bool:
        """Return whether another save may be emitted this scheduler step."""
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()

        # Loads
        logger.debug("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        loading_sids: set[str] = set()
        load_items = list(self._reqs_need_recv.items()) if self._do_load else []
        for sid, seq in load_items:
            ls = self._load_specs.pop(sid, None)
            if ls is None or not ls.can_load:
                logger.debug(
                    "[OFFLOAD-LOAD-SKIP] seq=%s ls=%s can_load=%s",
                    sid,
                    ls is not None,
                    getattr(ls, "can_load", None),
                )
                continue
            # ★ Use the REAL HBM-cached count as the load floor.
            # get_num_new_matched_tokens runs BEFORE the prefix-cache match in
            # block_manager.allocate, so seq.num_cached_tokens was stale (often
            # 0) when the LoadSpec was recorded. By now (post-allocate) it is the
            # true HBM hit. Loading below this floor would overwrite HBM
            # prefix-cache blocks (possibly shared with other seqs) -> output
            # corruption. So load only [hbm_cached, offload_hit).
            should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
                seq, ls
            )
            if not should_load:
                self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
                self._clear_pending_load(sid)
                continue
            # num_cached after load = max(HBM, offload); never drop below HBM.
            seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.debug(
                "[OFFLOAD-LOAD-EMIT] seq=%s hbm_cached=%d lmc_cached=%d "
                "offload_loaded=%d need=%d min_load=%d nblocks=%d reason=aligned_large_hit",
                seq.id,
                hbm,
                lmc,
                seq.offload_loaded_tokens,
                need,
                int(getattr(self, "_min_load_tokens", 8192)),
                len(list(seq.block_table)),
            )
            loading_sids.add(sid)
            self._load_save_floors[sid] = self._chunk_floor(hbm)
            load_operation = LoadOperationId(seq.id, self._load_nonce)
            self._load_nonce += 1
            seq._load_operation = load_operation
            self._active_load_operations[sid] = (seq, load_operation)
            self._track_load_statistics(load_operation, lmc - hbm)
            transfer_end = (
                lmc if ls.transfer_end_tokens is None else int(ls.transfer_end_tokens)
            )
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:transfer_end]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                    load_operation=load_operation,
                )
            )
        meta.lookup_requests_in_step = [
            sid
            for sid in self._lookup_in_step
            if sid in loading_sids or sid not in self._load_specs
        ]
        # Saves: store fully computed prompt chunks. Under scheduler-side
        # chunked prefill, seq.num_cached_tokens advances after each prefill
        # chunk's forward has completed; use it as the D2H-safe frontier.
        chunk = self.chunk_size or 256
        tracker_sids = list(self._save_tracker.keys())
        if tracker_sids and self._save_rr_last in self._save_tracker:
            start = (tracker_sids.index(self._save_rr_last) + 1) % len(tracker_sids)
            tracker_sids = tracker_sids[start:] + tracker_sids[:start]
        for sid in tracker_sids:
            entry = self._save_tracker[sid]
            if not self._do_save:
                continue
            if not self._may_emit_save():
                break
            seq, saved = entry
            if sid in self._reqs_need_recv or sid in loading_sids:
                continue  # loading this step; defer its save
            if sid in self._save_inflight:
                continue  # keep at most one save per request in flight
            computed = min(
                int(
                    getattr(
                        seq,
                        "_offload_finished_cached_tokens",
                        getattr(seq, "num_cached_tokens", 0),
                    )
                ),
                int(seq.num_prompt_tokens),
            )
            is_last_prefill = computed >= int(seq.num_prompt_tokens)
            aligned = (computed // chunk) * chunk
            if aligned <= saved:
                continue
            logger.debug(
                "[OFFLOAD-SAVE-EMIT] seq=%s computed=%d num_prompt=%d aligned=%d saved=%d",
                seq.id,
                computed,
                int(seq.num_prompt_tokens),
                aligned,
                saved,
            )
            save_operation = SaveOperationId(seq.id, self._save_nonce)
            self._save_nonce += 1
            self._track_save_statistics(save_operation, aligned - saved)
            block_ids = list(
                getattr(seq, "_offload_finished_block_ids", seq.block_table)
            )
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:aligned]),
                    block_ids=block_ids,
                    save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                    is_last_prefill=is_last_prefill,
                    save_operation=save_operation,
                )
            )
            entry[1] = aligned
            self._save_inflight[sid] = save_operation
            self._save_rr_last = sid
            if getattr(self, "_early_release", False):
                # Freeze the exact token-index -> block-id mapping before a
                # finished request clears its block table. The lease itself is
                # activated only by `activate_block_leases` at teardown.
                source_block_size = getattr(self, "virtual_block_size", self.block_size)
                start_block = saved // source_block_size
                end_block = -(-aligned // source_block_size)  # ceil div
                self._save_operation_blocks[save_operation] = {
                    index: block_ids[index]
                    for index in range(start_block, min(end_block, len(block_ids)))
                }
                self._save_operation_safe[save_operation] = set()
                self._save_operation_owner[save_operation] = seq
        dispatched = set(meta.lookup_requests_in_step)
        for sid in dispatched:
            self._lookup_results.pop(sid, None)
        self._lookup_in_step = [
            sid for sid in self._lookup_in_step if sid not in dispatched
        ]
        self._reqs_need_recv.clear()
        return meta

    def should_defer_free(self, seq) -> bool:
        if self._has_active_load(seq):
            return True
        if not self._do_save:
            return False
        sid = str(seq.id)
        operation_blocks = getattr(self, "_save_operation_blocks", {})
        operation_safe = getattr(self, "_save_operation_safe", {})
        operation_owner = getattr(self, "_save_operation_owner", {})
        unsafe_retired_operation = any(
            owner is seq
            and not set(operation_blocks.get(operation, {}).values()).issubset(
                operation_safe.get(operation, set())
            )
            for operation, owner in operation_owner.items()
        )
        return (
            sid in self._save_inflight
            or self._has_pending_save(seq)
            or unsafe_retired_operation
        )

    def protected_block_ids(self, seq) -> frozenset | None:
        """Exact pending/in-flight source blocks not yet known source-safe.

        None means "this connector cannot narrow the protection" (the layout
        does not support exact leases, or a load is in flight, since a load also
        touches HBM blocks this connector does not track per-range) -- the
        scheduler falls back to deferring the whole request. This includes a
        final save that has not been emitted yet: request teardown freezes its
        full block table and computed frontier so the next metadata build can
        still dispatch that save after unrelated blocks have been released.
        """
        if not self._early_release or self._has_active_load(seq):
            return None
        sid = str(seq.id)
        table = list(getattr(seq, "_offload_finished_block_ids", seq.block_table))
        if not hasattr(seq, "_offload_finished_block_ids"):
            seq._offload_finished_block_ids = table
        seq._offload_finished_cached_tokens = min(
            int(getattr(seq, "num_cached_tokens", 0)), int(seq.num_prompt_tokens)
        )

        protected: set[int] = set()
        for operation, blocks in self._save_operation_blocks.items():
            if self._save_operation_owner.get(operation) is not seq:
                continue
            safe = self._save_operation_safe.get(operation, set())
            protected.update(
                block_id for block_id in blocks.values() if block_id not in safe
            )

        entry = self._save_tracker.get(sid)
        if entry is not None and entry[0] is seq:
            saved = int(entry[1])
            aligned = self._save_frontier(seq)
            if aligned > saved:
                start_block = saved // self.virtual_block_size
                end_block = -(-aligned // self.virtual_block_size)
                protected.update(table[start_block:end_block])
        return frozenset(protected)

    def activate_block_leases(self, seq, block_ids: frozenset[int]) -> None:
        """Record the refcount shares transferred at request deallocation."""

        if not self._early_release or not block_ids:
            return
        lease_key = id(seq)
        leased = self._save_lease_blocks.setdefault(lease_key, set())
        self._save_lease_owner[lease_key] = seq
        added = set(block_ids) - leased
        leased.update(added)
        if added:
            self._save_lease_at.setdefault(lease_key, time.monotonic())
            self.total_leased_source_blocks += len(added)

    def take_source_safe_releases(self) -> list[frozenset]:
        """Drain block-ID sets whose lease became source-safe since the last poll."""
        out = self._pending_source_safe_releases
        self._pending_source_safe_releases = []
        return out

    def reclaim_stale_leases(self, timeout_s: float) -> list[frozenset]:
        """Force-release leases whose save never reported, past ``timeout_s``."""
        if timeout_s <= 0 or not self._save_lease_at:
            return []
        now = time.monotonic()
        stale_keys = [
            lease_key
            for lease_key, at in self._save_lease_at.items()
            if now - at >= timeout_s
        ]
        released: list[frozenset] = []
        for lease_key in stale_keys:
            self._save_lease_at.pop(lease_key, None)
            blocks = self._save_lease_blocks.pop(lease_key, None)
            owner = self._save_lease_owner.pop(lease_key, None)
            owned_operations = [
                op
                for op, operation_owner in self._save_operation_owner.items()
                if id(operation_owner) == lease_key
            ]
            sid = str(owner.id) if owner is not None else None
            operation = self._save_inflight.get(sid) if sid is not None else None
            if operation in owned_operations:
                self._save_inflight.pop(sid, None)
                self._cancel_save_statistics(operation)
            for candidate in [
                op for op in self._save_operation_blocks if op in owned_operations
            ]:
                self._save_operation_blocks.pop(candidate, None)
                self._save_operation_safe.pop(candidate, None)
                self._save_operation_owner.pop(candidate, None)
                self._source_safe_waiting_for_store.pop(candidate, None)
            if sid is not None:
                entry = self._save_tracker.get(sid)
                if entry is not None and entry[0] is owner:
                    self._save_tracker.pop(sid, None)
            if blocks:
                released.append(frozenset(blocks))
                self.total_abnormal_lease_reclaims += len(blocks)
        return released

    def release_stalled_save(self, seq) -> None:
        """Hook for layouts that allow the scheduler to reclaim stalled saves."""

    def has_pending_work(self) -> bool:
        """True while a load/cleanup is dispatchable or a save is unreported.

        Feeds ``EngineCore.has_pending_kv_work()``, so it reads only state
        that clears itself: ``_reqs_need_recv`` is emptied by every
        ``build_connector_meta`` and ``_save_inflight`` by ``save_finished``
        (or ``abandon_save`` when the scheduler reclaims a stalled save).
        With early release, a finished request's final not-yet-emitted save is
        no longer represented by ``deferred_free_blocks``; its frozen tracker
        entry must therefore keep the engine polling until it is dispatched.

        A pre-allocation lookup belongs to a waiting request. Keep its pin,
        but do not advertise idle work until allocation or cancellation makes
        a load or cleanup dispatchable in ``build_connector_meta``.
        """
        pending_finished_save = getattr(self, "_early_release", False) and any(
            hasattr(entry[0], "_offload_finished_block_ids")
            and self._has_pending_save(entry[0])
            for entry in self._save_tracker.values()
        )
        return (
            bool(self._reqs_need_recv)
            or bool(self._save_inflight)
            or bool(getattr(self, "_save_lease_blocks", {}))
            or pending_finished_save
            or any(sid not in self._load_specs for sid in self._lookup_in_step)
        )

    def save_finished(self, req_id) -> None:
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        active = self._save_inflight.get(sid)
        if isinstance(req_id, SaveOperationId):
            if active != req_id:
                return
        elif isinstance(active, SaveOperationId):
            # Once this lifecycle has an exact identity, a raw request ID
            # cannot complete it.  Raw IDs still clear explicitly legacy
            # entries should one be restored from older scheduler state.
            return
        self._save_inflight.pop(sid, None)
        if getattr(self, "_early_release", False):
            # The dedicated connector completion reports store success/failure.
            # This legacy terminal remains for MultiConnector save pairing.
            self._finish_retired_request(sid)
            return
        self._finish_save_statistics(req_id)
        self._release_operation_lease(req_id)
        self._finish_retired_request(sid)

    def connector_completion(self, completion: ConnectorCompletion) -> bool | None:
        """Apply TP/PP-quorumed source-safe and store-terminal reports."""

        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL:
            identity = completion.operation_id
            if not isinstance(identity, SaveSourceGroupId):
                return False
            self._source_group_finished(identity)
            return None
        if completion.channel != DENSE_PAGE_STORE_CHANNEL:
            return False
        operation = completion.operation_id
        if not isinstance(operation, SaveOperationId):
            return False
        self._store_finished(operation, succeeded=completion.succeeded)
        return True

    def _source_group_finished(self, identity: SaveSourceGroupId) -> None:
        operation = identity.save_operation
        block_map = self._save_operation_blocks.get(operation)
        if block_map is None:
            return
        source_blocks: set[int] = set()
        for start, end in identity.ranges:
            start_block = start // self.virtual_block_size
            end_block = -(-end // self.virtual_block_size)
            source_blocks.update(
                block_map[index]
                for index in range(start_block, end_block)
                if index in block_map
            )
        safe = self._save_operation_safe.setdefault(operation, set())
        newly_safe = source_blocks - safe
        safe.update(newly_safe)
        if not newly_safe:
            return
        sid = str(operation.req_id)
        owner = self._save_operation_owner.get(operation)
        lease_key = id(owner) if owner is not None else None
        leased = self._save_lease_blocks.get(lease_key)
        releasable = newly_safe & leased if leased is not None else set()
        if releasable:
            leased.difference_update(releasable)
            self._pending_source_safe_releases.append(frozenset(releasable))
            self.total_source_safe_released_blocks += len(releasable)
            if not leased:
                self._save_lease_blocks.pop(lease_key, None)
                self._save_lease_at.pop(lease_key, None)
                self._save_lease_owner.pop(lease_key, None)
        if self._save_inflight.get(sid) == operation:
            self._source_safe_waiting_for_store.setdefault(operation, set()).update(
                newly_safe
            )
        elif set(block_map.values()).issubset(safe):
            self._save_operation_blocks.pop(operation, None)
            self._save_operation_safe.pop(operation, None)
            self._save_operation_owner.pop(operation, None)

    def _store_finished(self, operation: SaveOperationId, *, succeeded: bool) -> None:
        sid = str(operation.req_id)
        active = self._save_inflight.get(sid)
        if (
            active != operation
            and operation not in self._save_operation_blocks
            and operation not in self._save_inflight_tokens
        ):
            return
        if active == operation:
            self._save_inflight.pop(sid, None)
        self._source_safe_waiting_for_store.pop(operation, None)
        if succeeded:
            self._finish_save_statistics(operation)
            # Store completion is also a source-safety fence for cache hits.
            self._release_operation_lease(operation)
        else:
            self._cancel_save_statistics(operation)
            # Failed stores keep unsafe ranges leased until abandon timeout.
        self._finish_retired_request(sid)

    def _release_operation_lease(self, operation) -> None:
        if not isinstance(operation, SaveOperationId):
            return
        block_map = self._save_operation_blocks.pop(operation, {})
        self._save_operation_safe.pop(operation, None)
        owner = self._save_operation_owner.pop(operation, None)
        lease_key = id(owner) if owner is not None else None
        leased = self._save_lease_blocks.get(lease_key)
        releasable = set(block_map.values()) & leased if leased is not None else set()
        if releasable:
            leased.difference_update(releasable)
            self._pending_source_safe_releases.append(frozenset(releasable))
            self.total_source_safe_released_blocks += len(releasable)
            if not leased:
                self._save_lease_blocks.pop(lease_key, None)
                self._save_lease_at.pop(lease_key, None)
                self._save_lease_owner.pop(lease_key, None)

    def _finish_retired_request(self, sid: str) -> None:
        entry = self._save_tracker.get(sid)
        if entry is None:
            return
        seq = entry[0]
        if hasattr(seq, "_offload_finished_block_ids") and not self._has_pending_save(
            seq
        ):
            self._save_tracker.pop(sid, None)

    def blocks_waiting_for_store(self) -> int:
        return sum(
            len(blocks) for blocks in self._source_safe_waiting_for_store.values()
        )

    def abandon_save(self, req_id) -> None:
        """Drop a save reclaimed after the backend failed to report it."""
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        operation = self._save_inflight.pop(sid, None)
        if operation is not None:
            self._cancel_save_statistics(operation)
        self._save_tracker.pop(sid, None)
        owner = self._save_operation_owner.pop(operation, None)
        lease_key = id(owner) if owner is not None else None
        self._save_lease_at.pop(lease_key, None)
        blocks = self._save_lease_blocks.pop(lease_key, None)
        self._save_lease_owner.pop(lease_key, None)
        if isinstance(operation, SaveOperationId):
            self._save_operation_blocks.pop(operation, None)
            self._save_operation_safe.pop(operation, None)
            self._source_safe_waiting_for_store.pop(operation, None)
        if blocks:
            self._pending_source_safe_releases.append(frozenset(blocks))
            self.total_abnormal_lease_reclaims += len(blocks)

    def load_failed(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            # Once this lifecycle has an exact generation, a legacy raw request
            # ID cannot complete it (including after request-ID reuse).
            return False
        self._finish_load_statistics(req_id, succeeded=False)
        floor = self._load_save_floors.get(sid)
        entry = self._save_tracker.get(sid)
        if floor is not None and entry is not None:
            # The LMCache hit was not actually loaded. Let the recomputed
            # [HBM, LMC) chunks be saved again instead of permanently treating
            # them as already persisted.
            entry[1] = self._chunk_floor(floor)
        self._clear_pending_load(sid)
        return True

    def load_finished(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            return False
        self._finish_load_statistics(req_id, succeeded=True)
        self._load_save_floors.pop(sid, None)
        return True

    def cancel_pending_load(self, seq) -> None:
        sid = str(seq.id)
        if self._load_lifecycles.get(sid) is not seq:
            return
        self._clear_pending_load(sid)
        active = self._active_load_operations.get(sid)
        if active is not None and active[0] is seq:
            self._active_load_operations.pop(sid, None)
            operation = active[1]
            self._cancel_load_statistics(operation)
            if getattr(seq, "_load_operation", None) == operation:
                delattr(seq, "_load_operation")

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        if self._load_lifecycles.get(sid) is seq:
            self._clear_pending_load(sid)
            active = self._active_load_operations.get(sid)
            if active is not None and active[0] is seq:
                self._active_load_operations.pop(sid, None)
                self._cancel_load_statistics(active[1])
            self._load_lifecycles.pop(sid, None)
        entry = self._save_tracker.get(sid)
        if entry is not None and entry[0] is seq:
            if self._early_release:
                # Freeze the final computed frontier before BlockManager
                # clears it during partial deallocation. Keep the tracker when
                # a final chunk still needs emission; a later metadata build
                # uses the frozen block table recorded by
                # `protected_block_ids`.
                seq._offload_finished_cached_tokens = min(
                    int(getattr(seq, "num_cached_tokens", 0)),
                    int(seq.num_prompt_tokens),
                )
                if not self.should_defer_free(seq):
                    self._save_tracker.pop(sid, None)
            elif not self.should_defer_free(seq):
                self._save_tracker.pop(sid, None)
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")


__all__ = [
    "DENSE_PAGE_SOURCE_SAFE_CHANNEL",
    "DENSE_PAGE_STORE_CHANNEL",
    "ChunkedOffloadSchedulerBase",
]
