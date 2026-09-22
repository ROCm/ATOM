# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout-neutral scheduler policy for chunked KV-cache offload."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from math import ceil

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
from atom.kv_transfer.offload.save_admission import (
    PrefixDemandKey,
    PrefixDemandTracker,
    SaveBlockReservation,
    load_save_admission_config,
)

logger = logging.getLogger("atom")

DENSE_PAGE_SOURCE_SAFE_CHANNEL = "dense.page.source_safe"
DENSE_PAGE_STORE_CHANNEL = "dense.page.store"


@dataclass(frozen=True)
class _ChunkedSaveCandidate:
    sid: str
    seq: object
    generation: int
    saved: int
    aligned: int
    observed_count: int
    reusable_tokens: int
    enqueued_at: float
    finished: bool
    held_blocks: int

    @property
    def dirty_tokens(self) -> int:
        return max(0, self.aligned - self.saved)


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
        save_admission = load_save_admission_config()
        self._save_min_observed_count = save_admission.min_observed_count
        self._save_aging_weight = save_admission.aging_weight
        self._save_release_weight = save_admission.release_weight
        self._save_max_pinned_ratio = save_admission.max_pinned_ratio
        self._save_max_pinned_blocks = save_admission.max_pinned_blocks
        self._save_pin_total_blocks = 0
        self._save_pin_budget_blocks: int | None = None
        self._save_block_reservations: dict[
            str | SaveOperationId, SaveBlockReservation
        ] = {}
        self._prefix_demand = PrefixDemandTracker(
            block_tokens=save_admission.demand_block_tokens,
            max_entries=save_admission.demand_max_entries,
            ttl_seconds=save_admission.demand_ttl_seconds,
        )
        self._save_demand_keys: dict[
            str, tuple[object, tuple[PrefixDemandKey, ...]]
        ] = {}
        self._save_candidate_since: dict[str, tuple[object, float]] = {}
        self._save_candidate_generation: dict[str, tuple[object, int]] = {}
        self._save_candidate_nonce = 0
        self._save_committed: dict[str, object] = {}
        self._finished_save_requests: dict[str, object] = {}
        self._finished_save_failed: set[str] = set()
        self._save_inflight_since: dict[SaveOperationId, float] = {}
        self.total_save_admitted = 0
        self.total_save_budget_rejected = 0
        self.total_save_budget_rejected_blocks = 0
        self.total_save_budget_evicted = 0
        self.total_save_budget_evicted_blocks = 0
        self.total_save_oversized = 0
        self._save_drop_totals = {
            "capacity": 0,
            "low_value": 0,
            "terminal_failure": 0,
            "stale": 0,
        }
        self._save_drop_token_totals = dict.fromkeys(self._save_drop_totals, 0)
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
        self._block_manager = None
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
        try:
            self._min_save_tokens = max(
                0, int(os.environ.get("OFFLOAD_MIN_SAVE_TOKENS", "8192"))
            )
        except ValueError:
            logger.warning(
                "LMCache offload scheduler: invalid OFFLOAD_MIN_SAVE_TOKENS=%r; "
                "using 8192",
                os.environ.get("OFFLOAD_MIN_SAVE_TOKENS"),
            )
            self._min_save_tokens = 8192

    def bind_block_manager(self, block_manager) -> None:
        if self._block_manager is not None and self._block_manager is not block_manager:
            raise RuntimeError("offload scheduler is already bound to a block manager")
        self._block_manager = block_manager
        total_blocks = int(block_manager.total_allocatable_kv_blocks)
        if total_blocks <= 0:
            raise ValueError("BlockManager must expose a positive KV block capacity")
        ratio_budget = int(total_blocks * self._save_max_pinned_ratio)
        absolute = self._save_max_pinned_blocks
        self._save_pin_total_blocks = total_blocks
        self._save_pin_budget_blocks = (
            ratio_budget if absolute is None else min(ratio_budget, absolute)
        )

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def _begin_load_lifecycle(self, seq) -> None:
        sid = str(seq.id)
        previous = self._load_lifecycles.get(sid)
        if previous is not None and previous is not seq:
            self._clear_pending_load(sid)
            self._active_load_operations.pop(sid, None)
        self._load_lifecycles[sid] = seq

    def _lookup_token_ids(self, seq) -> list[int]:
        """The prompt extent this layout can safely ask the tier to resume."""
        return list(seq.token_ids[: seq.num_prompt_tokens])

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if not self._do_load or self._lookup_client is None:
            return 0, False
        self._begin_load_lifecycle(seq)
        num_prompt = seq.num_prompt_tokens
        token_ids = self._lookup_token_ids(seq)
        if not token_ids:
            return 0, False
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
                now = time.monotonic()
                demand_keys = self._prefix_demand.observe(
                    seq.token_ids,
                    int(seq.num_prompt_tokens),
                    now,
                )
                self._save_demand_keys[sid] = (seq, demand_keys)
                self._save_candidate_generation[sid] = (
                    seq,
                    self._save_candidate_nonce,
                )
                self._save_candidate_nonce += 1
                previous_committed = self._save_committed.pop(sid, None)
                if previous_committed is not None:
                    self._release_save_reservation(sid, owner=previous_committed)
                self._finished_save_requests.pop(sid, None)
                self._finished_save_failed.discard(sid)
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

    def _ensure_save_admission_state(self) -> None:
        """Supply compatibility defaults for lightweight/manual schedulers."""

        if not hasattr(self, "_save_min_observed_count"):
            self._save_min_observed_count = 2
        if not hasattr(self, "_save_aging_weight"):
            self._save_aging_weight = 0.01
        if not hasattr(self, "_save_release_weight"):
            self._save_release_weight = 1.0
        if not hasattr(self, "_save_max_pinned_ratio"):
            self._save_max_pinned_ratio = 0.20
        if not hasattr(self, "_save_max_pinned_blocks"):
            self._save_max_pinned_blocks = None
        if not hasattr(self, "_save_pin_total_blocks"):
            self._save_pin_total_blocks = 0
        if not hasattr(self, "_save_pin_budget_blocks"):
            self._save_pin_budget_blocks = None
        if not hasattr(self, "_save_block_reservations"):
            self._save_block_reservations = {}
        if not hasattr(self, "_save_lease_blocks"):
            self._save_lease_blocks = {}
        if not hasattr(self, "_prefix_demand"):
            self._prefix_demand = PrefixDemandTracker()
        for name in (
            "_save_demand_keys",
            "_save_candidate_since",
            "_save_candidate_generation",
            "_save_committed",
            "_finished_save_requests",
            "_save_inflight_since",
        ):
            if not hasattr(self, name):
                setattr(self, name, {})
        if not hasattr(self, "_save_candidate_nonce"):
            self._save_candidate_nonce = 0
        if not hasattr(self, "_finished_save_failed"):
            self._finished_save_failed = set()
        if not hasattr(self, "total_save_admitted"):
            self.total_save_admitted = 0
        for name in (
            "total_save_budget_rejected",
            "total_save_budget_rejected_blocks",
            "total_save_budget_evicted",
            "total_save_budget_evicted_blocks",
            "total_save_oversized",
        ):
            if not hasattr(self, name):
                setattr(self, name, 0)
        if not hasattr(self, "_save_drop_totals"):
            self._save_drop_totals = {
                "capacity": 0,
                "low_value": 0,
                "terminal_failure": 0,
                "stale": 0,
            }
        if not hasattr(self, "_save_drop_token_totals"):
            self._save_drop_token_totals = dict.fromkeys(self._save_drop_totals, 0)

    def _reserved_save_resources_allow(self, committed_after: int) -> bool:
        """Layout-specific admission hook evaluated before any mutation."""

        del committed_after
        return True

    def _save_count_allows(self, committed_after: int) -> bool:
        limit = getattr(self, "_max_pending_saves", None)
        return limit is None or len(self._save_inflight) + committed_after <= limit

    def _pinned_save_block_ids(self) -> set[int]:
        pinned: set[int] = set()
        for blocks in self._save_lease_blocks.values():
            pinned.update(blocks)
        return pinned

    def _save_block_usage(
        self,
        reservations: dict[str | SaveOperationId, SaveBlockReservation] | None = None,
    ) -> tuple[int, int, int]:
        """Return ``(reserved, pinned, total_budget_usage)`` without double count."""

        current = (
            self._save_block_reservations if reservations is None else reservations
        )
        pinned_ids = self._pinned_save_block_ids()
        reserved_ids: set[int] = set()
        estimated = 0
        for reservation in current.values():
            reserved_ids.update(reservation.block_ids)
            estimated += int(reservation.estimated_blocks)
        reserved = len(reserved_ids - pinned_ids) + estimated
        return reserved, len(pinned_ids), len(pinned_ids | reserved_ids) + estimated

    def _candidate_block_reservation(
        self,
        candidate: _ChunkedSaveCandidate,
        *,
        priority_score: float,
    ) -> SaveBlockReservation:
        source_block_size = int(getattr(self, "virtual_block_size", self.block_size))
        start_block = candidate.saved // source_block_size
        end_block = ceil(candidate.aligned / source_block_size)
        required = max(0, end_block - start_block)
        table = list(getattr(candidate.seq, "block_table", ()))
        # A frozen table records old identities only. Once early deallocation
        # clears the live table those IDs may already belong to another request,
        # so retain a conservative count rather than a false exact reservation.
        if candidate.finished and not table:
            block_ids = frozenset()
            estimated = required
        else:
            available = table[start_block : min(end_block, len(table))]
            block_ids = frozenset(
                physical_id
                for block_id in available
                if (physical_id := int(block_id)) >= 0
            )
            estimated = max(0, required - len(available))
        return SaveBlockReservation(
            sid=candidate.sid,
            seq=candidate.seq,
            generation=candidate.generation,
            block_ids=block_ids,
            estimated_blocks=estimated,
            priority_score=float(priority_score),
            committed=True,
        )

    @staticmethod
    def _reservation_footprint(reservation: SaveBlockReservation) -> int:
        return len(reservation.block_ids) + int(reservation.estimated_blocks)

    def _reservation_fits(
        self,
        reservation: SaveBlockReservation,
        *,
        victim_keys: tuple[str, ...] = (),
    ) -> bool:
        projected = dict(self._save_block_reservations)
        for key in victim_keys:
            projected.pop(key, None)
        projected[reservation.sid] = reservation
        committed_after = sum(item.committed for item in projected.values())
        if not self._save_count_allows(committed_after):
            return False
        if not self._reserved_save_resources_allow(committed_after):
            return False
        budget = self._save_pin_budget_blocks
        return budget is None or self._save_block_usage(projected)[2] <= budget

    def _try_reserve_save_candidate(
        self,
        candidate: _ChunkedSaveCandidate,
        *,
        allow_eviction: bool,
        now: float | None = None,
    ) -> bool:
        """Atomically reserve PAGE/count/state budget for one candidate.

        Victims are simulated first and are restricted to lower-priority,
        undispatched committed saves owned by this same scheduler. Inflight
        operation reservations are never considered.
        """

        existing = self._save_block_reservations.get(candidate.sid)
        if (
            existing is not None
            and existing.seq is candidate.seq
            and existing.generation == candidate.generation
            and existing.committed
        ):
            return True

        score = self._candidate_priority(
            candidate,
            time.monotonic() if now is None else now,
        )
        reservation = self._candidate_block_reservation(candidate, priority_score=score)
        victims: list[SaveBlockReservation] = []
        if not self._reservation_fits(reservation):
            if allow_eviction:
                eligible = sorted(
                    (
                        item
                        for key, item in self._save_block_reservations.items()
                        if isinstance(key, str)
                        and item.committed
                        and item.priority_score < reservation.priority_score
                        and self._save_committed.get(key) is item.seq
                        and key not in self._save_inflight
                        and (entry := self._save_tracker.get(key)) is not None
                        and entry[0] is item.seq
                    ),
                    key=lambda item: (
                        item.priority_score,
                        item.sid,
                        item.generation,
                    ),
                )
                for victim in eligible:
                    victims.append(victim)
                    if self._reservation_fits(
                        reservation,
                        victim_keys=tuple(item.sid for item in victims),
                    ):
                        break
                else:
                    victims = []
            if not victims and not self._reservation_fits(reservation):
                footprint = self._reservation_footprint(reservation)
                self.total_save_budget_rejected += 1
                self.total_save_budget_rejected_blocks += footprint
                budget = self._save_pin_budget_blocks
                if budget is not None and footprint > budget:
                    self.total_save_oversized += 1
                return False

        # Every condition was checked against the same simulated state. Apply
        # the replacement only now so a candidate that still cannot fit never
        # destroys useful committed work.
        for victim in victims:
            self.total_save_budget_evicted += 1
            self.total_save_budget_evicted_blocks += self._reservation_footprint(victim)
            self.drop_unadmitted_save(victim.seq, reason="capacity")
        self._save_committed[candidate.sid] = candidate.seq
        self._save_block_reservations[candidate.sid] = reservation
        self.total_save_admitted += 1
        return True

    def _release_save_reservation(
        self, key: str | SaveOperationId, *, owner=None
    ) -> SaveBlockReservation | None:
        reservation = self._save_block_reservations.get(key)
        if reservation is None or (owner is not None and reservation.seq is not owner):
            return None
        return self._save_block_reservations.pop(key)

    def _candidate_generation_for(self, sid: str, seq) -> int:
        record = self._save_candidate_generation.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        generation = self._save_candidate_nonce
        self._save_candidate_nonce += 1
        self._save_candidate_generation[sid] = (seq, generation)
        return generation

    def _candidate_since_for(self, sid: str, seq, now: float) -> float:
        record = self._save_candidate_since.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        self._save_candidate_since[sid] = (seq, now)
        return now

    def _build_save_candidate(
        self, sid: str, *, now: float | None = None
    ) -> _ChunkedSaveCandidate | None:
        self._ensure_save_admission_state()
        entry = self._save_tracker.get(sid)
        if entry is None:
            return None
        seq, saved = entry
        aligned = self._save_frontier(seq)
        if aligned <= int(saved):
            return None
        candidate_now = time.monotonic() if now is None else float(now)
        demand = self._save_demand_keys.get(sid)
        observed_count, reusable_tokens = (0, 0)
        if demand is not None and demand[0] is seq:
            observed_count, reusable_tokens = self._prefix_demand.heat(
                demand[1],
                max_tokens=aligned,
                now=candidate_now,
            )
        finished = self._finished_save_requests.get(sid) is seq
        return _ChunkedSaveCandidate(
            sid=sid,
            seq=seq,
            generation=self._candidate_generation_for(sid, seq),
            saved=int(saved),
            aligned=aligned,
            observed_count=observed_count,
            reusable_tokens=reusable_tokens,
            enqueued_at=self._candidate_since_for(sid, seq, candidate_now),
            finished=finished,
            held_blocks=(
                len(
                    getattr(
                        seq,
                        "_offload_finished_block_ids",
                        getattr(seq, "block_table", ()),
                    )
                )
                if finished
                else 0
            ),
        )

    def _candidate_priority(
        self, candidate: _ChunkedSaveCandidate, now: float
    ) -> float:
        chunk = max(1, int(self.chunk_size or 256))
        cost_units = max(1, ceil(candidate.dirty_tokens / chunk))
        reusable = max(candidate.reusable_tokens, chunk)
        benefit = candidate.observed_count * reusable
        release_bonus = (
            self._save_release_weight
            * candidate.held_blocks
            * max(1, int(getattr(self, "virtual_block_size", self.block_size)))
        )
        age = max(0.0, now - candidate.enqueued_at)
        return (benefit + release_bonus) / cost_units + self._save_aging_weight * age

    def _candidate_sort_key(
        self, candidate: _ChunkedSaveCandidate, now: float
    ) -> tuple[float, int, float, str, int]:
        return (
            -self._candidate_priority(candidate, now),
            candidate.dirty_tokens,
            candidate.enqueued_at,
            candidate.sid,
            candidate.generation,
        )

    def _meets_save_value_threshold(self, candidate: _ChunkedSaveCandidate) -> bool:
        return candidate.observed_count >= self._save_min_observed_count

    def _commit_finished_save(self, candidate: _ChunkedSaveCandidate) -> bool:
        if not candidate.finished or not self._meets_save_value_threshold(candidate):
            return False
        entry = self._save_tracker.get(candidate.sid)
        if entry is None or entry[0] is not candidate.seq:
            return False
        return self._try_reserve_save_candidate(
            candidate,
            allow_eviction=True,
        )

    def _forget_save_candidate(self, sid: str, seq) -> None:
        for mapping in (
            self._save_demand_keys,
            self._save_candidate_since,
            self._save_candidate_generation,
        ):
            record = mapping.get(sid)
            if record is not None and record[0] is seq:
                mapping.pop(sid, None)

    def drop_unadmitted_save(self, seq, *, reason: str) -> bool:
        """Atomically remove a candidate that no worker can be reading."""

        sid = str(seq.id)
        if sid in self._save_inflight:
            return False
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return False
        candidate = self._build_save_candidate(sid)
        dropped_tokens = 0 if candidate is None else candidate.dirty_tokens
        self._save_tracker.pop(sid, None)
        if self._save_committed.get(sid) is seq:
            self._save_committed.pop(sid, None)
        self._release_save_reservation(sid, owner=seq)
        if self._finished_save_requests.get(sid) is seq:
            self._finished_save_requests.pop(sid, None)
        self._finished_save_failed.discard(sid)
        self._forget_save_candidate(sid, seq)
        if reason not in self._save_drop_totals:
            reason = "stale"
        self._save_drop_totals[reason] += 1
        self._save_drop_token_totals[reason] += dropped_tokens
        logger.info(
            "LMCache offload: dropped unadmitted save req=%s reason=%s tokens=%d",
            sid,
            reason,
            dropped_tokens,
        )
        return True

    def _settle_finished_save(self, sid: str, *, failed: bool = False) -> None:
        seq = self._finished_save_requests.get(sid)
        if seq is None or sid in self._save_inflight or sid in self._save_committed:
            return
        if failed:
            self._finished_save_failed.add(sid)
        if sid in self._finished_save_failed:
            self.drop_unadmitted_save(seq, reason="terminal_failure")
            return
        candidate = self._build_save_candidate(sid)
        if candidate is None:
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is seq:
                self._save_tracker.pop(sid, None)
            self._finished_save_requests.pop(sid, None)
            self._finished_save_failed.discard(sid)
            self._forget_save_candidate(sid, seq)
            return
        if self._commit_finished_save(candidate):
            return
        reason = (
            "low_value"
            if not self._meets_save_value_threshold(candidate)
            else "capacity"
        )
        self.drop_unadmitted_save(seq, reason=reason)

    def _new_load_operation(self, seq) -> LoadOperationId:
        operation = LoadOperationId(seq.id, self._load_nonce)
        self._load_nonce += 1
        return operation

    def _build_save_request(
        self,
        seq,
        saved: int,
        aligned: int,
        operation: SaveOperationId,
        block_ids: list[int],
        is_last_prefill: bool,
    ) -> LMCacheReqMeta | None:
        """Admit any layout-specific sources before advancing the watermark."""
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=list(seq.token_ids[:aligned]),
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
        )

    def _late_save_frontier(self, seq, saved: int, available: int) -> int:
        """Return the largest layout-valid boundary for a late-acquired source."""
        del seq, saved
        return self._chunk_floor(available)

    def _late_save_source(
        self, seq, saved: int, aligned: int
    ) -> tuple[int, list[int], frozenset[int]] | None:
        """Reacquire a finished request's still-resident prefix at admission."""
        if self._block_manager is None:
            raise RuntimeError("late offload save requires a bound block manager")
        block_ids, available, claimed = self._block_manager.acquire_offload_prefix(
            seq, saved, aligned
        )
        target = self._late_save_frontier(seq, saved, available)
        source_block_size = int(getattr(self, "virtual_block_size", self.block_size))
        keep_count = max(0, (target - saved) // source_block_size)
        keep = frozenset(list(claimed)[:keep_count])
        drop = set(claimed) - set(keep)
        if drop:
            self._block_manager.free_leased_blocks(drop)
        if target - saved < self._min_save_tokens:
            if keep:
                self._block_manager.free_leased_blocks(keep)
            return None
        return target, block_ids, keep

    def _emit_save_candidate(
        self,
        meta: LMCacheOffloadMetadata,
        candidate: _ChunkedSaveCandidate,
    ) -> bool:
        sid = candidate.sid
        seq = candidate.seq
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return False
        committed_reservation = self._save_block_reservations.get(sid)
        if (
            committed_reservation is None
            or committed_reservation.seq is not seq
            or committed_reservation.generation != candidate.generation
            or not committed_reservation.committed
        ):
            return False
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
        aligned = candidate.aligned
        save_operation = SaveOperationId(seq.id, self._save_nonce)
        self._save_nonce += 1
        late_acquired = frozenset()
        if hasattr(seq, "_offload_finished_block_ids") and not seq.block_table:
            late_source = self._late_save_source(seq, candidate.saved, aligned)
            if late_source is None:
                if candidate.finished:
                    self.drop_unadmitted_save(seq, reason="stale")
                else:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    self._save_tracker.pop(sid, None)
                    self._forget_save_candidate(sid, seq)
                return False
            aligned, block_ids, late_acquired = late_source
        else:
            block_ids = list(seq.block_table)
        try:
            request = self._build_save_request(
                seq,
                candidate.saved,
                aligned,
                save_operation,
                block_ids,
                is_last_prefill,
            )
        except Exception:
            if late_acquired:
                self._block_manager.free_leased_blocks(late_acquired)
            raise
        if request is None:
            if late_acquired:
                self._block_manager.free_leased_blocks(late_acquired)
            self._save_committed.pop(sid, None)
            self._release_save_reservation(sid, owner=seq)
            if candidate.finished:
                self.drop_unadmitted_save(seq, reason="capacity")
            return False

        logger.debug(
            "[OFFLOAD-SAVE-EMIT] seq=%s computed=%d num_prompt=%d "
            "aligned=%d saved=%d observed=%d",
            seq.id,
            computed,
            int(seq.num_prompt_tokens),
            aligned,
            candidate.saved,
            candidate.observed_count,
        )
        self._track_save_statistics(save_operation, aligned - candidate.saved)
        self._save_inflight_since[save_operation] = time.monotonic()
        meta.add_request(request)
        entry[1] = aligned
        self._save_committed.pop(sid, None)
        self._release_save_reservation(sid, owner=seq)
        self._save_inflight[sid] = save_operation
        if getattr(self, "_early_release", False):
            # Freeze the exact token-index -> block-id mapping before a
            # finished request clears its block table. The lease itself is
            # activated only by `activate_block_leases` at teardown.
            source_block_size = getattr(self, "virtual_block_size", self.block_size)
            start_block = candidate.saved // source_block_size
            end_block = -(-aligned // source_block_size)  # ceil div
            block_map = {
                index: block_ids[index]
                for index in range(start_block, min(end_block, len(block_ids)))
            }
            self._save_operation_blocks[save_operation] = block_map
            self._save_operation_safe[save_operation] = set()
            self._save_operation_owner[save_operation] = seq
            self._save_block_reservations[save_operation] = replace(
                committed_reservation,
                block_ids=frozenset(
                    int(block_id)
                    for block_id in block_map.values()
                    if int(block_id) >= 0
                ),
                estimated_blocks=max(0, end_block - start_block - len(block_map)),
                committed=False,
            )
            if late_acquired:
                self.activate_block_leases(seq, late_acquired)
        else:
            self._save_block_reservations[save_operation] = replace(
                committed_reservation, committed=False
            )
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        self._ensure_save_admission_state()
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
            load_operation = self._new_load_operation(seq)
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
        # Saves. Admit/replace the complete candidate set before
        # dispatch so a newly hotter candidate can still evict lower-value
        # committed work. Dispatching first would make that work inflight and
        # therefore correctly, but prematurely, non-evictable.
        if self._do_save:
            now = time.monotonic()
            for sid, seq in list(self._save_committed.items()):
                entry = self._save_tracker.get(sid)
                if entry is None or entry[0] is not seq:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    continue
                candidate = self._build_save_candidate(sid, now=now)
                if candidate is None:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    self._settle_finished_save(sid)

            candidates = [
                candidate
                for sid in self._save_tracker
                if (candidate := self._build_save_candidate(sid, now=now)) is not None
                and self._meets_save_value_threshold(candidate)
            ]
            candidates.sort(
                key=lambda candidate: self._candidate_sort_key(candidate, now)
            )

            for candidate in candidates:
                sid = candidate.sid
                if sid in self._save_committed or sid in self._save_inflight:
                    continue
                if sid in self._reqs_need_recv or sid in loading_sids:
                    continue
                if not self._try_reserve_save_candidate(
                    candidate,
                    allow_eviction=True,
                    now=now,
                ):
                    self.drop_unadmitted_save(candidate.seq, reason="capacity")

            committed = []
            for sid, seq in list(self._save_committed.items()):
                candidate = self._build_save_candidate(sid, now=now)
                if candidate is None or candidate.seq is not seq:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    continue
                committed.append(candidate)
            committed.sort(
                key=lambda candidate: self._candidate_sort_key(candidate, now)
            )
            for candidate in committed:
                sid = candidate.sid
                if sid in self._reqs_need_recv or sid in loading_sids:
                    continue
                if sid in self._save_inflight:
                    continue
                self._emit_save_candidate(meta, candidate)
        dispatched = set(meta.lookup_requests_in_step)
        for sid in dispatched:
            self._lookup_results.pop(sid, None)
        self._lookup_in_step = [
            sid for sid in self._lookup_in_step if sid not in dispatched
        ]
        self._reqs_need_recv.clear()
        return meta

    def should_defer_free(self, seq) -> bool:
        self._ensure_save_admission_state()
        if self._has_active_load(seq):
            return True
        if not self._do_save:
            return False
        sid = str(seq.id)
        operation_blocks = getattr(self, "_save_operation_blocks", {})
        operation_safe = getattr(self, "_save_operation_safe", {})
        operation_owner = getattr(self, "_save_operation_owner", {})
        active_operation = self._save_inflight.get(sid)
        active_owner = operation_owner.get(active_operation)
        active_entry = self._save_tracker.get(sid)
        active_save = active_operation is not None and (
            active_owner is seq
            or (
                active_owner is None
                and (active_entry is None or active_entry[0] is seq)
            )
        )
        unsafe_retired_operation = any(
            owner is seq
            and not set(operation_blocks.get(operation, {}).values()).issubset(
                operation_safe.get(operation, set())
            )
            for operation, owner in operation_owner.items()
        )
        return (
            active_save
            or self._save_committed.get(sid) is seq
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
        token identity and computed frontier, but protects no physical PAGE
        until admission reacquires the still-canonical prefix.
        """
        if not self._early_release or self._has_active_load(seq):
            return None
        table = list(getattr(seq, "_offload_finished_block_ids", seq.block_table))
        if not hasattr(seq, "_offload_finished_block_ids"):
            seq._offload_finished_block_ids = table
        if not hasattr(seq, "_offload_finished_cached_tokens"):
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

        return frozenset(protected)

    def activate_block_leases(self, seq, block_ids: frozenset[int]) -> None:
        """Record the refcount shares transferred at request deallocation."""

        if not self._early_release:
            return
        sid = str(seq.id)
        committed = self._save_block_reservations.get(sid)
        if committed is not None and committed.seq is seq and committed.committed:
            # The scheduler is about to clear the request's live block table.
            # Its old physical IDs can be reused, so a not-yet-dispatched save
            # keeps only a conservative count reservation from this point on.
            self._save_block_reservations[sid] = replace(
                committed,
                block_ids=frozenset(),
                estimated_blocks=self._reservation_footprint(committed),
            )
        for operation, reservation in list(self._save_block_reservations.items()):
            if (
                isinstance(operation, SaveOperationId)
                and reservation.seq is seq
                and not reservation.committed
            ):
                # The future reservation becomes an actual save-owned physical
                # lease. Removing it before adding the lease avoids double
                # charging the same IDs during the ownership transition.
                self._save_block_reservations.pop(operation, None)
        if not block_ids:
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
                self._save_inflight_since.pop(operation, None)
                self._cancel_save_statistics(operation)
            for candidate in [
                op for op in self._save_operation_blocks if op in owned_operations
            ]:
                self._save_operation_blocks.pop(candidate, None)
                self._save_operation_safe.pop(candidate, None)
                self._save_operation_owner.pop(candidate, None)
                self._source_safe_waiting_for_store.pop(candidate, None)
                self._release_save_reservation(candidate, owner=owner)
            if sid is not None:
                entry = self._save_tracker.get(sid)
                if entry is not None and entry[0] is owner:
                    self._save_tracker.pop(sid, None)
                if self._save_committed.get(sid) is owner:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=owner)
                if self._finished_save_requests.get(sid) is owner:
                    self._finished_save_requests.pop(sid, None)
                self._finished_save_failed.discard(sid)
                if owner is not None:
                    self._forget_save_candidate(sid, owner)
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
            or bool(self._save_committed)
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

    def save_finished_by_request(self, req_id) -> None:
        """Complete a save when only the plain request id is available.

        `save_finished` refuses a raw id once the lifecycle has an exact
        `SaveOperationId`, so a delayed report cannot complete a newer
        lifecycle. A vLLM-plugin scheduler cannot satisfy that: vLLM's
        `KVConnectorOutput` carries request ids as plain strings, so the exact
        identity never survives the trip back from the worker.

        Resolving the parked identity here keeps the guard meaningful instead of
        weakening `save_finished` -- and without it the entry never clears, so
        `_save_inflight` grows for the life of the process and
        `has_pending_work()` never goes quiet.
        """
        sid = str(req_id)
        active = self._save_inflight.get(sid)
        self.save_finished(active if active is not None else sid)

    def load_finished_by_request(self, req_id) -> bool:
        """`load_finished` for a caller that has only the plain request id.

        Same reason as `save_finished_by_request`.
        """
        sid = str(req_id)
        entry = self._active_load_operations.get(sid)
        return self.load_finished(entry[1] if entry is not None else sid)

    def load_failed_by_request(self, req_id) -> bool:
        """`load_failed` for a caller that has only the plain request id.

        Same reason as `save_finished_by_request`: vLLM's `KVConnectorOutput`
        carries request ids as plain strings, so the exact `LoadOperationId`
        never survives the trip back from the worker half. Routing a failure
        through `load_finished` instead would pop `_load_save_floors`, which is
        the record that the `[HBM, LMCache)` range is NOT persisted -- the
        recomputed chunks would then never be saved.
        """

        sid = str(req_id)
        entry = self._active_load_operations.get(sid)
        return self.load_failed(entry[1] if entry is not None else sid)

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
        reservation = self._save_block_reservations.get(operation)
        if reservation is not None:
            remaining = reservation.block_ids - newly_safe
            self._save_block_reservations[operation] = replace(
                reservation, block_ids=frozenset(remaining)
            )
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
            self._release_save_reservation(operation)

    def _store_finished(self, operation: SaveOperationId, *, succeeded: bool) -> None:
        sid = str(operation.req_id)
        active = self._save_inflight.get(sid)
        owner = self._save_operation_owner.get(operation)
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
            block_map = self._save_operation_blocks.get(operation, {})
            safe = self._save_operation_safe.get(operation, set())
            if set(block_map.values()).issubset(safe):
                # A terminal failure can arrive after every source range became
                # safe. Retire that exact operation now; otherwise native MP,
                # whose terminal is also its final source fence, leaks owner and
                # block-map bookkeeping forever.
                self._release_operation_lease(operation)
        self._save_inflight_since.pop(operation, None)
        if owner is not None and self._finished_save_requests.get(sid) is owner:
            self._settle_finished_save(sid, failed=not succeeded)
        self._finish_retired_request(sid)

    def _release_operation_lease(self, operation) -> None:
        if not isinstance(operation, SaveOperationId):
            return
        self._release_save_reservation(operation)
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
            if self._save_committed.get(sid) is seq:
                self._save_committed.pop(sid, None)
            self._release_save_reservation(sid, owner=seq)
            if self._finished_save_requests.get(sid) is seq:
                self._finished_save_requests.pop(sid, None)
            self._finished_save_failed.discard(sid)
            self._forget_save_candidate(sid, seq)

    def blocks_waiting_for_store(self) -> int:
        return sum(
            len(blocks) for blocks in self._source_safe_waiting_for_store.values()
        )

    def abandon_save(self, req_id) -> None:
        """Drop a save reclaimed after the backend failed to report it."""
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        operation = self._save_inflight.pop(sid, None)
        if operation is not None:
            self._save_inflight_since.pop(operation, None)
            self._cancel_save_statistics(operation)
        tracker = self._save_tracker.pop(sid, None)
        owner = self._save_operation_owner.pop(operation, None)
        lease_key = id(owner) if owner is not None else None
        self._save_lease_at.pop(lease_key, None)
        blocks = self._save_lease_blocks.pop(lease_key, None)
        self._save_lease_owner.pop(lease_key, None)
        if isinstance(operation, SaveOperationId):
            self._save_operation_blocks.pop(operation, None)
            self._save_operation_safe.pop(operation, None)
            self._source_safe_waiting_for_store.pop(operation, None)
            self._release_save_reservation(operation, owner=owner)
        committed = self._save_committed.pop(sid, None)
        if committed is not None:
            self._release_save_reservation(sid, owner=committed)
        finished = self._finished_save_requests.pop(sid, None)
        seq = finished if finished is not None else committed
        if seq is None and tracker is not None:
            seq = tracker[0]
        self._finished_save_failed.discard(sid)
        if seq is not None:
            self._forget_save_candidate(sid, seq)
        if blocks:
            self._pending_source_safe_releases.append(frozenset(blocks))
            self.total_abnormal_lease_reclaims += len(blocks)

    def get_statistics(self) -> dict[str, int | float]:
        """Return save-admission gauges in addition to transfer counters."""

        self._ensure_save_admission_state()
        statistics = super().get_statistics()
        now = time.monotonic()
        candidates = []
        for sid, entry in self._save_tracker.items():
            seq = entry[0]
            operation = self._save_inflight.get(sid)
            if (
                self._save_operation_owner.get(operation) is seq
                or self._save_committed.get(sid) is seq
            ):
                continue
            candidate = self._build_save_candidate(sid, now=now)
            if candidate is not None:
                candidates.append(candidate)

        finished = [candidate for candidate in candidates if candidate.finished]
        candidate_scores = [
            self._candidate_priority(candidate, now) for candidate in candidates
        ]
        reserved_blocks, pinned_blocks, budget_used = self._save_block_usage()
        budget = self._save_pin_budget_blocks
        total_blocks = self._save_pin_total_blocks
        statistics.update(
            save_candidates=len(candidates),
            save_candidates_finished=len(finished),
            save_committed=len(self._save_committed),
            save_admitted=self.total_save_admitted,
            save_candidate_wait_seconds=max(
                (max(0.0, now - candidate.enqueued_at) for candidate in candidates),
                default=0.0,
            ),
            save_priority_score=max(candidate_scores, default=0.0),
            save_inflight_wait_seconds=max(
                (
                    max(0.0, now - started)
                    for started in self._save_inflight_since.values()
                ),
                default=0.0,
            ),
            save_pinned_blocks=pinned_blocks,
            save_pinned_tokens=pinned_blocks * int(self.virtual_block_size),
            save_pin_budget_blocks=0 if budget is None else budget,
            save_reserved_blocks=reserved_blocks,
            save_pinned_ratio=(budget_used / total_blocks if total_blocks > 0 else 0.0),
            save_budget_available_blocks=(
                0 if budget is None else max(0, budget - budget_used)
            ),
            save_budget_rejected=self.total_save_budget_rejected,
            save_budget_rejected_blocks=self.total_save_budget_rejected_blocks,
            save_budget_evicted=self.total_save_budget_evicted,
            save_budget_evicted_blocks=self.total_save_budget_evicted_blocks,
            save_oversized=self.total_save_oversized,
            deferred_free_requests=sum(
                bool(blocks) for blocks in self._save_lease_blocks.values()
            ),
        )
        for reason, total in self._save_drop_totals.items():
            statistics[f"save_dropped_{reason}"] = total
            statistics[f"save_dropped_tokens_{reason}"] = self._save_drop_token_totals[
                reason
            ]
        return statistics

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
            self._finished_save_requests[sid] = seq
            # Freeze the final source identity before request teardown. A
            # reservation dispatches on a later metadata build and reacquires
            # only the still-canonical prefix blocks.
            if self._early_release:
                seq._offload_finished_cached_tokens = min(
                    int(getattr(seq, "num_cached_tokens", 0)),
                    int(seq.num_prompt_tokens),
                )
            if sid in self._save_inflight or self._save_committed.get(sid) is seq:
                pass
            else:
                candidate = self._build_save_candidate(sid)
                if candidate is None:
                    self._settle_finished_save(sid)
                elif not self._meets_save_value_threshold(candidate):
                    self.drop_unadmitted_save(seq, reason="low_value")
                elif not self._commit_finished_save(candidate):
                    self.drop_unadmitted_save(seq, reason="capacity")
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")


__all__ = [
    "DENSE_PAGE_SOURCE_SAFE_CHANNEL",
    "DENSE_PAGE_STORE_CHANNEL",
    "ChunkedOffloadSchedulerBase",
]
