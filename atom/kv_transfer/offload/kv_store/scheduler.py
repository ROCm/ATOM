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
:class:`StoreLookupBase`.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload._offload_common import max_pending_saves
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.kv_store.keys import (
    DIGEST_BYTES,
    chain_seed,
    chunk_digest,
    chunk_hash_chain,
    rank_key_prefix,
)
from atom.kv_transfer.offload.lookup_client import LookupClient
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
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

# Seconds lookups do not ask a store the scheduler could not reach.
# Connecting blocks the scheduler thread, so a missing store must not cost
# that on every step.
LOOKUP_RECONNECT_INTERVAL_S = 30.0
# The same for a store whose lookup failed.
LOOKUP_BACKOFF_S = 10.0
# Either pause lasts at least this many times as long as the failed call
# blocked the scheduler thread (`pause_lookups`).
LOOKUP_BACKOFF_FACTOR = 10.0
# Seconds between two warnings about one store's failing lookups.
LOOKUP_WARNING_INTERVAL_S = 60.0


def pause_lookups(target: Any, floor_s: float, since: float) -> float:
    """Pause lookups to ``target`` after a failed call that started at ``since``.

    Sets ``target.paused_until``: for ``floor_s`` or `LOOKUP_BACKOFF_FACTOR`
    times as long as the call blocked, whichever is longer, counted from now.
    A failed setup can return minutes after it started, and a pause counted
    from before it would already be over. Returns the seconds paused.
    """
    now = time.monotonic()
    pause = max(floor_s, LOOKUP_BACKOFF_FACTOR * (now - since))
    target.paused_until = now + pause
    return pause


class LookupWarnings:
    """At most one warning per store and `LOOKUP_WARNING_INTERVAL_S`; the
    rest go to the debug log. ``label`` prefixes every message."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._next_at: dict[str, float] = {}

    def warn(self, store: str, message: str, *args: Any, exc_info: Any = None) -> None:
        now = time.monotonic()
        if now < self._next_at.get(store, 0.0):
            logger.debug(f"{self._label}: " + message, *args)
            return
        self._next_at[store] = now + LOOKUP_WARNING_INTERVAL_S
        logger.warning(f"{self._label}: " + message, *args, exc_info=exc_info)


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


class StoreLookupBase(LookupClient):
    """What every backend's lookup client shares.

    A hit is the prefix every rank holds: per rank, the run of chunks present
    from chunk 0, and the minimum over ranks, since a load is all-or-nothing
    across the PP stages and TP ranks (`rank_prefixes`). A subclass asks its
    store and implements ``lookup`` and ``clear_lookup_status``.
    """

    def __init__(self, *, namespace: str, world: int, chunk_tokens: int) -> None:
        self._namespace = namespace
        self._world = int(world)
        self._chunk_tokens = int(chunk_tokens)
        # Answered lookups; those where the ranks held prefixes of different
        # lengths; and the rank objects past the shared prefix they found,
        # present but unusable while another rank lacks the chunk -- evicted,
        # failed, or still being written by a slower stage.
        self.lookups = 0
        self.uneven_lookups = 0
        self.stranded_objects = 0
        self._next_stats_at = time.monotonic() + _LOOKUP_STATS_INTERVAL_S

    def _rank_keys(self, hashes: bytes, first: int, last: int) -> list[str]:
        """Every rank's keys of chunks ``[first, last)``, rank by rank.

        Every rank's keys share the chunks' digests: each is hex-encoded once.
        """
        digests = [chunk_digest(hashes, index).hex() for index in range(first, last)]
        keys = []
        for rank in range(self._world):
            prefix = rank_key_prefix(self._namespace, rank, self._world)
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
        logger.info(
            "[OFFLOAD-LOOKUP-STATS] lookups=%d uneven_lookups=%d stranded_objects=%d",
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
    ) -> None:
        """``store_cfg`` is the backend's parsed settings, ``endpoint`` how the
        startup log names the store it asks (``"server=..."``)."""
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self._store_cfg = store_cfg
        self._namespace = namespace
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
            "max_pending_saves=%s abandon_timeout=%.0fs",
            self.LABEL,
            namespace,
            world,
            store_cfg.chunk_tokens,
            endpoint,
            self._max_pending_saves,
            store_cfg.save_abandon_timeout_s,
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
        )

    def _build_load_request(
        self,
        seq: Any,
        load_spec: LoadSpec,
        load_operation: LoadOperationId,
        transfer_end: int,
    ) -> LMCacheReqMeta:
        """Load ``[hbm, transfer_end)``: the digests of ``[0, transfer_end)``."""
        hashes = prompt_chunk_hashes(seq, self.chunk_size)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=list(seq.block_table),
            load_spec=load_spec,
            load_operation=load_operation,
            chunk_hashes=hashes[: (transfer_end // self.chunk_size) * DIGEST_BYTES],
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
    "monotonic_clock_id",
    "prompt_chunk_hashes",
    "require_monotonic_clock_id",
]
