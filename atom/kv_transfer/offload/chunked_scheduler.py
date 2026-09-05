# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout-neutral scheduler policy for chunked KV-cache offload."""

from __future__ import annotations

import logging
import os

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.types import (
    LoadOperationId,
    SaveCompletionId,
    SaveOperationId,
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


class ChunkedOffloadSchedulerBase(OffloadSchedulerMixin, KVConnectorSchedulerBase):
    """Transport- and layout-neutral policy for chunk-aligned KV offload."""

    # Consumer semantics: finished_recving wakes parked seqs (the engine asserts
    # `not is_producer` on that path). Offload never uses finished_sending.
    is_producer = False
    # Opt the scheduler into offload-wake (suffix prefill) instead of the P/D
    # decode-jump in Scheduler.schedule(); see Scheduler._is_offload_connector.
    is_offload = True

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
        self._save_nonce = 0
        self._load_nonce = 0
        self._load_lifecycles: dict[str, object] = {}
        self._active_load_operations: dict[str, tuple[object, LoadOperationId]] = {}
        self._lookup_in_step: list[str] = []
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
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=str(seq.id))
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
        sid = str(seq.id)
        hit = int(hit)
        if hit == num_prompt:  # full-prompt hit → recompute last token
            hit -= 1
        self._hit_save_floors[sid] = self._chunk_floor(hit)
        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            if self._lookup_client is not None:
                try:
                    self._lookup_client.clear_lookup_status(sid)
                except Exception:
                    logger.debug(
                        "LMCache offload: lookup status cleanup failed for req=%s",
                        sid,
                        exc_info=True,
                    )
            return 0, False
        self._lookup_in_step.append(sid)
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
        self._lookup_in_step = [
            req_id for req_id in self._lookup_in_step if req_id != sid
        ]
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
        meta.lookup_requests_in_step = list(self._lookup_in_step)
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
                int(getattr(seq, "num_cached_tokens", 0)),
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
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:aligned]),
                    block_ids=list(seq.block_table),
                    save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                    is_last_prefill=is_last_prefill,
                    save_operation=save_operation,
                )
            )
            entry[1] = aligned
            self._save_inflight[sid] = save_operation
            self._save_rr_last = sid
        dispatched = set(meta.lookup_requests_in_step)
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
        return sid in self._save_inflight or self._has_pending_save(seq)

    def release_stalled_save(self, seq) -> None:
        """Hook for layouts that allow the scheduler to reclaim stalled saves."""

    def has_pending_work(self) -> bool:
        """True while a load still needs dispatch or a save is unreported.

        Feeds ``EngineCore.has_pending_kv_work()``, so it reads only state
        that clears itself: ``_reqs_need_recv`` is emptied by every
        ``build_connector_meta`` and ``_save_inflight`` by ``save_finished``
        (or ``abandon_save`` when the scheduler reclaims a stalled save).
        Saves that are queued but not yet dispatched are covered there by the
        scheduler's ``deferred_free_blocks``, which ``should_defer_free``
        keeps populated for exactly those requests.
        """
        return bool(self._reqs_need_recv) or bool(self._save_inflight)

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
        self._finish_save_statistics(req_id)

    def abandon_save(self, req_id) -> None:
        """Drop a save reclaimed after the backend failed to report it."""
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        operation = self._save_inflight.pop(sid, None)
        if operation is not None:
            self._cancel_save_statistics(operation)
        self._save_tracker.pop(sid, None)

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
        if entry is not None and entry[0] is seq and not self.should_defer_free(seq):
            self._save_tracker.pop(sid, None)
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")


__all__ = ["ChunkedOffloadSchedulerBase"]
