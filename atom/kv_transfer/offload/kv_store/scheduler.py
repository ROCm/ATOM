# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of a KV-store offload, whatever the store.

The policy -- what to save and when, whether a hit is worth loading, early
block release, exact operation completions -- is all
:class:`ChunkedOffloadSchedulerBase`'s. :class:`KVStoreOffloadScheduler` adds
what every store backend shares: requests that carry the prompt's chunk
digests, which name the store's objects, instead of its token ids; paced
retries of failed saves; and the save abandon timeout of the connector's own
settings. A backend supplies the lookup client, which answers the one question
the base asks a transport -- how much of a prompt the store holds -- built on
:class:`StoreLookupBase`. Under a draft window the lookup also cuts a hit where
the store lacks a draft object its read window needs, and saves and loads name
the draft objects they move (``LMCacheReqMeta.draft_chunks``).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload._offload_common import max_pending_saves
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.kv_store.keys import (
    DIGEST_BYTES,
    chain_seed,
    chunk_digest,
    chunk_hash_chain,
    draft_namespace,
    rank_key_prefix,
)
from atom.kv_transfer.offload.lookup_client import LookupClient
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.spec_decode.draft_window_geometry import DraftReadWindow
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds between two `[OFFLOAD-LOOKUP-STATS]` lines.
_LOOKUP_STATS_INTERVAL_S = 60.0
# After a failed save the base retries, no save is emitted for this long,
# doubled for each further such failure in a row up to `_SAVE_RETRY_MAX_S`. A
# stored save ends the pause and starts the doubling over, whenever it reports.
# The base retries a failed save on the next step, and a failure that repeats
# -- a full store, a store that is down, a stage whose save slots are all
# quarantined -- would otherwise re-pack the request's unsaved range on every
# PP stage every step.
_SAVE_RETRY_BASE_S = 1.0
_SAVE_RETRY_MAX_S = 30.0
# Names the boot a CLOCK_MONOTONIC reading is from; see `monotonic_clock_id`.
_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
# Uncut hits kept for loads not decided yet (`StoreLookupBase.take_uncut_hit`);
# those of requests that never reach the decision go oldest first.
_MAX_UNCUT_HITS = 4096


@functools.cache
def monotonic_clock_id() -> str | None:
    """Which CLOCK_MONOTONIC ``time.monotonic()`` reads here: this boot's ID.

    Every process of one boot of a host reads the same CLOCK_MONOTONIC, which no
    wall-clock step moves; another host's counts from another origin. None when
    the kernel does not say (`require_monotonic_clock_id`).
    """
    try:
        with open(_BOOT_ID_PATH) as boot_id:
            return boot_id.read().strip() or None
    except OSError:
        return None


def require_monotonic_clock_id(who: str = "the KV store offload") -> str:
    """`monotonic_clock_id`, or a refusal to start without one.

    A worker reads a save's dispatch stamp only on the clock the stamp names,
    and treats any other as one it cannot place (the save then reads nothing):
    without a boot ID every save would fail on its deadline, with no other sign
    than a rate-limited warning. ``who`` names the connector in the refusal.

    Raises:
        RuntimeError: The kernel does not name this boot.
    """
    clock = monotonic_clock_id()
    if clock is None:
        raise RuntimeError(
            f"{who} cannot read the boot ID in {_BOOT_ID_PATH}, "
            "which names the clock its saves are timed on; without it every save "
            "would store nothing"
        )
    return clock


def prompt_chunk_hashes(seq: Any, chunk_tokens: int) -> bytes:
    """Digests of every full chunk of ``seq``'s prompt, hashed once per sequence.

    Kept on the sequence (``_kv_store_chunk_hashes``): its prompt never
    changes, and lookups, saves and loads of it all need the same chain.
    """
    num_prompt = int(seq.num_prompt_tokens)
    expected = (num_prompt // int(chunk_tokens)) * DIGEST_BYTES
    cached = getattr(seq, "_kv_store_chunk_hashes", None)
    if isinstance(cached, bytes) and len(cached) == expected:
        return cached
    hashes = chunk_hash_chain(
        seq.token_ids,
        chunk_tokens,
        num_tokens=num_prompt,
        seed=chain_seed(getattr(seq, "cache_seed", None)),
    )
    seq._kv_store_chunk_hashes = hashes
    return hashes


def first_draft_chunk(window: DraftReadWindow, num_prompt: int, chunk: int) -> int:
    """d0: the first chunk holding draft rows any reader of a `num_prompt`-token
    prompt reads; a draft window stores draft objects from it on."""
    return window.first_read_token(num_prompt) // int(chunk)


class StoreLookupBase(LookupClient):
    """What every backend's lookup client shares.

    A hit is the prefix every rank holds: per rank, the run of chunks present
    from chunk 0, and the minimum over ranks, since a load is all-or-nothing
    across the PP stages and TP ranks (`rank_prefixes`). Under a draft window
    it is cut where its read window lacks a draft object (`_gate_draft_rows`).
    A subclass asks its store and implements ``lookup`` and
    ``clear_lookup_status``.
    """

    def __init__(
        self,
        *,
        namespace: str,
        world: int,
        chunk_tokens: int,
        draft_window: DraftReadWindow | None = None,
        draft_ranks: range = range(0),
    ) -> None:
        self._namespace = namespace
        self._world = int(world)
        self._chunk_tokens = int(chunk_tokens)
        # The read window draft objects are kept for, and the ranks they belong
        # to; None keeps the draft rows in the target objects.
        self.draft_window = draft_window
        if draft_window is not None and not (
            draft_ranks and set(draft_ranks) <= set(range(self._world))
        ):
            raise ValueError(
                f"a draft window needs draft ranks among the {self._world} ranks, "
                f"got {draft_ranks!r}"
            )
        self.draft_ranks = draft_ranks if draft_window is not None else range(0)
        # Draft keys asked; chunks whose draft object some rank lacks inside a
        # hit's read window; lookups that window cut at d0 or at a hole past it.
        self.draft_keys = 0
        self.draft_missing = 0
        self.draft_cuts_at_d0 = 0
        self.draft_cuts_above_d0 = 0
        # Tokens the cuts made requests prefill, counted as their loads are
        # decided (`KVStoreOffloadScheduler._count_draft_cut_cost`).
        self.draft_cut_recompute_tokens = 0
        # Lookup id -> the hit the window cut, until its load is decided.
        self._uncut_hits: OrderedDict[str, int] = OrderedDict()
        self._uncut_lock = threading.Lock()
        # Answered lookups; those where the ranks held prefixes of different
        # lengths; and the rank objects past the shared prefix they found,
        # present but unusable while another rank lacks the chunk -- evicted,
        # failed, or still being written by a slower stage.
        self.lookups = 0
        self.uneven_lookups = 0
        self.stranded_objects = 0
        self._next_stats_at = time.monotonic() + _LOOKUP_STATS_INTERVAL_S

    def _rank_keys(self, hashes: bytes, first: int, last: int) -> list[str]:
        """Every rank's keys of chunks ``[first, last)``, rank by rank."""
        chunks = range(first, last)
        return self._keys_by_rank(self._namespace, range(self._world), hashes, chunks)

    def _keys_by_rank(
        self, namespace: str, ranks: range, hashes: bytes, chunks: range
    ) -> list[str]:
        """`ranks`' keys of `chunks` under `namespace`, rank by rank. The ranks
        share the chunks' digests: each is hex-encoded once, and none for no rank."""
        if not ranks:
            return []
        digests = [chunk_digest(hashes, index).hex() for index in chunks]
        keys = []
        for rank in ranks:
            prefix = rank_key_prefix(namespace, rank, self._world)
            keys += [prefix + digest for digest in digests]
        return keys

    def _rank_prefixes(
        self, answers: list[Any], chunks: int
    ) -> tuple[list[int], list[int]]:
        """Per rank, the run of chunks from chunk 0 the store holds.

        ``answers`` is per key of `_rank_keys` over ``chunks`` chunks: 1 (or
        True) present, 0 (or False) absent, anything else unanswered. Returns
        the runs counting an unanswered key as absent (``held``) and as
        present (``bound``); they are equal when every key was answered.
        """
        held, bound = [], []
        for rank in range(self._world):
            shard = answers[rank * chunks : (rank + 1) * chunks]
            held.append(next((i for i, code in enumerate(shard) if code != 1), chunks))
            bound.append(next((i for i, code in enumerate(shard) if code == 0), chunks))
        return held, bound

    def _draft_range(self, seq: Any, chunks: int) -> range:
        """Chunks `[d0, d1)` whose draft objects a load of `seq` may read: no load
        takes the prompt's last token (`load_math.loadable_hit`)."""
        if self.draft_window is None:
            return range(0)
        num_prompt = int(seq.num_prompt_tokens)
        first = first_draft_chunk(self.draft_window, num_prompt, self._chunk_tokens)
        return range(first, min(chunks, (num_prompt - 1) // self._chunk_tokens))

    def _draft_keys(self, hashes: bytes, chunks: range) -> list[str]:
        """Every draft rank's keys of `chunks`, rank by rank; none, and no
        digest encoded, without a draft window."""
        namespace = draft_namespace(self._namespace)
        return self._keys_by_rank(namespace, self.draft_ranks, hashes, chunks)

    def _draft_cut(self, draft: range, hit_chunks: int, codes: list[Any]) -> int | None:
        """`hit_chunks` cut at the first chunk of `[d0, hit)` whose draft object
        some draft rank lacks; None when keys no store answered decide the cut."""
        held = bound = hit_chunks
        missing = 0
        for index in range(draft.start, min(hit_chunks, draft.stop)):
            column = codes[index - draft.start :: len(draft)]
            absent = 0 in column
            missing += absent
            if held == hit_chunks and any(code != 1 for code in column):
                held = index
            if absent and bound == hit_chunks:
                bound = index
        if held != bound:
            return None
        self.draft_missing += missing
        if held == draft.start < hit_chunks:
            self.draft_cuts_at_d0 += 1
        elif held < hit_chunks:
            self.draft_cuts_above_d0 += 1
        return held

    def _gate_draft_rows(
        self, lookup_id: str | None, draft: range, hit_chunks: int, codes: list[Any]
    ) -> int | None:
        """`hit_chunks` cut by the draft objects of `draft`, answered in `codes` in
        `_draft_keys` order; None where unanswered keys decide the cut."""
        if not draft:
            return hit_chunks
        cut = self._draft_cut(draft, hit_chunks, codes)
        if cut is None:
            logger.debug(
                "[OFFLOAD-LOOKUP] lookup_id=%s: draft keys no store answered "
                "decide the hit; no answer",
                lookup_id,
            )
            return None
        self.draft_keys += len(draft) * len(self.draft_ranks)
        if cut < hit_chunks:
            logger.debug(
                "[OFFLOAD-LOOKUP] lookup_id=%s draft window cut: d0=%d cut=%d "
                "hit=%d (chunks)",
                lookup_id,
                draft.start,
                cut,
                hit_chunks,
            )
        self._remember_uncut_hit(
            lookup_id, hit_chunks * self._chunk_tokens if cut < hit_chunks else None
        )
        return cut

    def _remember_uncut_hit(self, lookup_id: str | None, hit: int | None) -> None:
        """Keep the hit the window cut from `lookup_id`'s answer; None forgets
        the one an earlier answer of it left."""
        if lookup_id is None:
            return
        with self._uncut_lock:
            if hit is None:
                self._uncut_hits.pop(lookup_id, None)
                return
            self._uncut_hits[lookup_id] = int(hit)
            self._uncut_hits.move_to_end(lookup_id)
            while len(self._uncut_hits) > _MAX_UNCUT_HITS:
                self._uncut_hits.popitem(last=False)

    def take_uncut_hit(self, lookup_id: str) -> int | None:
        """The hit the window cut from `lookup_id`'s last answer, once; None
        when it cut nothing."""
        with self._uncut_lock:
            return self._uncut_hits.pop(lookup_id, None)

    def _count_lookup(self, stranded: int) -> None:
        """Count an answered lookup, logging the totals once a minute."""
        self.lookups += 1
        if stranded:
            self.uneven_lookups += 1
            self.stranded_objects += stranded
        now = time.monotonic()
        if now < self._next_stats_at:
            return
        self._next_stats_at = now + _LOOKUP_STATS_INTERVAL_S
        draft = (
            ""
            if self.draft_window is None
            else f" draft_keys={self.draft_keys} draft_missing={self.draft_missing}"
            f" draft_cuts_at_d0={self.draft_cuts_at_d0}"
            f" draft_cuts_above_d0={self.draft_cuts_above_d0}"
            f" draft_cut_recompute_tokens={self.draft_cut_recompute_tokens}"
        )
        logger.info(
            "[OFFLOAD-LOOKUP-STATS] lookups=%d uneven_lookups=%d stranded_objects=%d"
            + draft,
            self.lookups,
            self.uneven_lookups,
            self.stranded_objects,
        )


class KVStoreOffloadScheduler(ChunkedOffloadSchedulerBase):
    """Chunked offload scheduling with an external KV store as the tier."""

    # Workers publish one source-safe group per chunk once its GPU pack is
    # done (the block GPU connector), separately from the store terminal.
    _supports_early_block_release = True
    # How logs name the backend.
    LABEL = "KV store offload"

    def __init__(
        self,
        config: Any,
        *,
        store_cfg: Any,
        namespace: str,
        world: int,
        lookup_client: Any,
        endpoint: str,
        store_lookup: StoreLookupBase | None = None,
    ) -> None:
        """``store_cfg`` is the backend's parsed settings, ``endpoint`` how the
        startup log names the store it asks (``"server=..."``). ``store_lookup``
        is the lookup behind ``lookup_client``, whose draft window saves and loads
        follow; None keeps the draft rows in the target objects."""
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self._store_cfg = store_cfg
        self._namespace = namespace
        self._store_lookup = store_lookup
        self._draft_window = None if store_lookup is None else store_lookup.draft_window
        # Failed saves since the last stored one, and when saves may resume.
        self._failed_saves_in_a_row = 0
        self._saves_paused_until = 0.0
        super().__init__(
            config,
            chunk_size=store_cfg.chunk_tokens,
            lookup_client=lookup_client,
        )
        extra = kvc.get("kv_connector_extra_config", kvc) or {}
        # Unbounded unless asked for, in the connector config or through
        # OFFLOAD_MAX_PENDING_SAVES: one save per request is in flight at most
        # either way.
        asked = (
            "max_pending_saves" in extra or envs.OFFLOAD_MAX_PENDING_SAVES is not None
        )
        self._max_pending_saves = (
            max_pending_saves(kvc, envs.OFFLOAD_COPY_WORKERS) if asked else None
        )
        logger.info(
            "%s scheduler: namespace=%s world=%d chunk=%d %s "
            "max_pending_saves=%s abandon_timeout=%.0fs%s",
            self.LABEL,
            namespace,
            world,
            store_cfg.chunk_tokens,
            endpoint,
            self._max_pending_saves,
            store_cfg.save_abandon_timeout_s,
            (
                ""
                if self._draft_window is None
                else f" draft_window={self._draft_window.window}"
                f" draft_ranks={list(store_lookup.draft_ranks)}"
            ),
        )

    def _lookup_token_ids(self, seq: Any) -> Any:
        """The sequence itself: the lookup hashes its prompt once per sequence.

        Copying a long prompt into a list on every lookup would cost more than
        the lookup (`prompt_chunk_hashes`).
        """
        return seq

    def _tier_lookup_kwargs(self, seq: Any) -> dict:
        # Store keys chain from `Sequence.cache_seed`, the source of the media
        # salt (`prompt_chunk_hashes`), so the lookup takes no salt argument.
        return {}

    def _lookup_wanted(self, seq: Any) -> bool:
        # A prompt too short to load from the store, whatever it holds, is not
        # asked about: no hit is longer than the prompt less a token, floored
        # to a chunk (`_loadable_hit`), and no shorter one reaches the load
        # threshold. A prompt of at most one chunk can load nothing, even when
        # the threshold is 0. The query would only hold the lookup, and park
        # the request while it ran.
        num_prompt = int(seq.num_prompt_tokens)
        hit = self._loadable_hit(num_prompt, num_prompt)
        return hit > 0 and hit >= self._min_load_tokens

    def _fresh_tier_lookup(self, seq: Any, sid: str) -> int | None:
        if not self._lookup_wanted(seq):
            return 0
        self._lookup_starting(sid)
        return super()._fresh_tier_lookup(seq, sid)

    def _lookup_starting(self, sid: str) -> None:
        """Called right before the store is asked about request ``sid``."""

    def _ensure_lookup_pin(self, seq: Any, sid: str, spec: LoadSpec) -> bool:
        """Always dispatch the load: a store lookup pins nothing to re-take.

        The engine parked the request when it admitted it, before this
        dispatch, and only the load's report wakes it: a load dropped here
        would leave it parked for good, holding its KV blocks and a
        ``max_num_seqs`` slot. A prefix evicted since the lookup fails the
        worker's all-or-nothing get instead, and the request prefills once
        that failure reports (``load_failed``).
        """
        return True

    def _may_emit_save(self) -> bool:
        if time.monotonic() < self._saves_paused_until:
            return False
        cap = self._max_pending_saves
        return cap is None or len(self._save_inflight) < cap

    def _store_finished(self, operation: SaveOperationId, *, succeeded: bool) -> None:
        if succeeded:
            # Saves run concurrently, so this one may report after another
            # failed and paused them: the store takes saves again.
            self._failed_saves_in_a_row = 0
            self._saves_paused_until = 0.0
        super()._store_finished(operation, succeeded=succeeded)

    def _retry_or_retire_failed_save(self, operation: SaveOperationId) -> bool:
        """Pace the retry the base sets up; see `_SAVE_RETRY_BASE_S`.

        A save the base retires instead (its request finished, or its ID was
        reused) has nothing to pace, and pauses no other request's saves.
        """
        if not super()._retry_or_retire_failed_save(operation):
            return False
        self._failed_saves_in_a_row += 1
        pause = min(
            _SAVE_RETRY_MAX_S,
            _SAVE_RETRY_BASE_S * 2 ** min(self._failed_saves_in_a_row - 1, 16),
        )
        self._saves_paused_until = max(
            self._saves_paused_until, time.monotonic() + pause
        )
        return True

    def _build_save_request(
        self,
        seq: Any,
        saved: int,
        aligned: int,
        operation: SaveOperationId,
        block_ids: list[int],
        is_last_prefill: bool,
    ) -> LMCacheReqMeta:
        """Save chunks ``[saved, aligned)``: the digests of ``[0, aligned)``.

        No token ids: every PP stage receives each request, and the digests
        are all the worker needs to name its objects. The dispatch time starts
        the workers' source read deadline where the reclaim clock starts, on
        the same CLOCK_MONOTONIC (the reclaim stamps the save just after).
        """
        hashes = prompt_chunk_hashes(seq, self.chunk_size)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
            chunk_hashes=hashes[: (aligned // self.chunk_size) * DIGEST_BYTES],
            dispatched_at=time.monotonic(),
            dispatch_clock=monotonic_clock_id(),
            draft_chunks=self._save_draft_chunks(seq, saved, aligned, block_ids),
        )

    def _save_draft_chunks(
        self, seq: Any, saved: int, aligned: int, block_ids: list[int]
    ) -> tuple[int, int] | None:
        """The draft objects a save of `[saved, aligned)` puts: the highest run of
        the prompt's read window `[d0, C)` it covers whose blocks have draft rows."""
        if self._draft_window is None:
            return None
        chunk = self.chunk_size
        first = max(
            saved // chunk,
            first_draft_chunk(self._draft_window, seq.num_prompt_tokens, chunk),
        )
        end = start = aligned // chunk
        while start > first and not self._draft_rows_missing_in(block_ids, start - 1):
            start -= 1
        return (start, end) if start < end else None

    def _draft_rows_missing_in(self, block_ids: list[int], index: int) -> bool:
        """Whether a block of chunk `index` lacks draft rows; True for a block the
        scheduler cannot see (no block manager, or past the table)."""
        manager = self._block_manager
        per_chunk = self.chunk_size // self.virtual_block_size
        blocks = block_ids[index * per_chunk : (index + 1) * per_chunk]
        return (
            manager is None
            or len(blocks) < per_chunk
            or any(manager.draft_rows_missing(int(block)) for block in blocks)
        )

    def _build_load_request(
        self,
        seq: Any,
        load_spec: LoadSpec,
        load_operation: LoadOperationId,
        transfer_end: int,
    ) -> LMCacheReqMeta:
        """Load ``[hbm, transfer_end)``: the digests of ``[0, transfer_end)``, and
        under a draft window the draft objects of ``[max(hbm, d0), transfer_end)``."""
        hashes = prompt_chunk_hashes(seq, self.chunk_size)
        request = LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=list(seq.block_table),
            load_spec=load_spec,
            load_operation=load_operation,
            chunk_hashes=hashes[: (transfer_end // self.chunk_size) * DIGEST_BYTES],
        )
        if self._draft_window is not None:
            first = self._load_draft_start(
                seq, load_spec.hbm_cached_tokens, transfer_end
            )
            end = int(transfer_end) // self.chunk_size
            if first < end:
                request.draft_chunks = (first, end)
            # The blocks below get no draft rows: `publish_loaded_prefix` marks them.
            seq.offload_draft_start_tokens = first * self.chunk_size
            self._count_draft_cut_cost(
                seq, load_spec.hbm_cached_tokens, load_spec.lmcache_cached_tokens
            )
        return request

    def _load_draft_start(self, seq: Any, hbm: int, end: int) -> int:
        """The first draft object a load of `[hbm, end)` reads, in chunks:
        `max(hbm, d0)`, at most `end`."""
        chunk = self.chunk_size
        d0 = first_draft_chunk(self._draft_window, seq.num_prompt_tokens, chunk)
        return min(max(int(hbm) // chunk, d0), int(end) // chunk)

    def _maybe_start_unaligned_handoff(self, seq, load_spec, hbm, lmc, chunk) -> bool:
        started = super()._maybe_start_unaligned_handoff(
            seq, load_spec, hbm, lmc, chunk
        )
        manager = self._block_manager
        if started and self._draft_window is not None and manager is not None:
            # Marked before the partial prefill: a P/D producer sends V with its
            # first forward, and this load runs only after it.
            boundary = int(load_spec.hbm_cached_tokens)
            first = self._load_draft_start(seq, boundary, lmc)
            manager.mark_draft_rows_missing(seq, boundary, first * self.chunk_size)
        return started

    def update_state_after_alloc(self, seq: Any) -> None:
        super().update_state_after_alloc(seq)
        if str(seq.id) not in self._load_specs:
            # No load follows (a hit cut to 0, or under HBM): decided here.
            self._count_draft_cut_cost(seq, int(seq.num_cached_tokens), 0)

    def _mark_load_skip(
        self, seq: Any, reason: str, hbm: int, lmc: int, need: int, chunk: int
    ) -> None:
        super()._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
        self._count_draft_cut_cost(seq, hbm, lmc)

    def _count_draft_cut_cost(self, seq: Any, hbm: int, hit: int) -> None:
        """Once `seq`'s load is decided at `hit` over an HBM prefix of `hbm`: the
        tokens it prefills that the hit the window cut would have loaded."""
        if self._draft_window is None:
            return
        uncut = self._store_lookup.take_uncut_hit(str(seq.id))
        if uncut is None:
            return
        uncut = self._loadable_hit(uncut, int(seq.num_prompt_tokens))
        self._store_lookup.draft_cut_recompute_tokens += max(
            0, uncut - max(int(hbm), int(hit))
        )

    def save_abandon_timeout_s(self) -> float:
        """Seconds a deferred save may sit before the engine reclaims it.

        A setting of the connector (``<prefix>save_abandon_timeout_s``) rather
        than a pin timeout: the workers stop reading a save's source blocks
        well before it runs out (see the worker's source read deadline).
        """
        return self._store_cfg.save_abandon_timeout_s


__all__ = [
    "KVStoreOffloadScheduler",
    "StoreLookupBase",
    "first_draft_chunk",
    "monotonic_clock_id",
    "prompt_chunk_hashes",
    "require_monotonic_clock_id",
]
