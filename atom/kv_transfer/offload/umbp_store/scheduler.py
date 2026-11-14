# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of the UMBP offload.

Copied from the Mooncake Store offload's scheduler and rewritten for UMBP; a
shared layer is planned. The policy -- what to save and when, whether a hit
is worth loading, early block release, exact operation completions -- is all
:class:`ChunkedOffloadSchedulerBase`'s. This module answers the one question
it asks a transport, how much of a prompt the server holds, and shapes the
requests the workers receive: each carries the prompt's chunk digests, which
name the server's objects, instead of the prompt's token ids.

Differences from the Mooncake Store scheduler: one server instead of pools,
no read leases (a UMBP lookup holds nothing, so there is nothing to renew
before a load), and no cross-client taint marker (see :mod:`.taint`).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    max_pending_saves,
    validated_kv_role,
)
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.client import UMBPStoreClient
from atom.kv_transfer.offload.umbp_store.config import (
    UMBPStoreOffloadConfig,
    check_engine_compatibility,
    parse_umbp_store_config,
)
from atom.kv_transfer.offload.umbp_store.keys import (
    DIGEST_BYTES,
    chain_seed,
    chunk_digest,
    chunk_hash_chain,
    rank_key_prefix,
    store_namespace,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds lookups do not ask a server the scheduler could not reach.
# Connecting blocks the scheduler thread for up to the startup timeout, so a
# missing server must not cost that on every step.
_RECONNECT_INTERVAL_S = 30.0
# The same for a server whose lookup raised.
_LOOKUP_BACKOFF_S = 10.0
# Either pause lasts at least this many times as long as the failed call
# blocked the scheduler thread (`_ServerLookup._pause`).
_LOOKUP_BACKOFF_FACTOR = 10.0
# Seconds between two warnings about failing lookups.
_LOOKUP_WARNING_INTERVAL_S = 60.0
# Seconds between two `[OFFLOAD-LOOKUP-STATS]` lines.
_LOOKUP_STATS_INTERVAL_S = 60.0
# After a failed save the base retries, no save is emitted for this long,
# doubled for each further such failure in a row up to `_SAVE_RETRY_MAX_S`. A
# stored save ends the pause and starts the doubling over, whenever it reports.
# A failure that repeats -- a full server, a server that is down, a stage
# whose save slots are all quarantined -- would otherwise re-pack the
# request's unsaved range on every PP stage every step.
_SAVE_RETRY_BASE_S = 1.0
_SAVE_RETRY_MAX_S = 30.0
# Names the boot a CLOCK_MONOTONIC reading is from; see `monotonic_clock_id`.
_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


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


def require_monotonic_clock_id() -> str:
    """`monotonic_clock_id`, or a refusal to start without one.

    A worker reads a save's dispatch stamp only on the clock the stamp names,
    and treats any other as one it cannot place (the save then reads nothing):
    without a boot ID every save would fail on its deadline, with no other sign
    than a rate-limited warning.

    Raises:
        RuntimeError: The kernel does not name this boot.
    """
    clock = monotonic_clock_id()
    if clock is None:
        raise RuntimeError(
            f"the UMBP offload cannot read the boot ID in {_BOOT_ID_PATH}, which "
            "names the clock its saves are timed on; without it every save would "
            "store nothing"
        )
    return clock


def prompt_chunk_hashes(seq: Any, chunk_tokens: int) -> bytes:
    """Digests of every full chunk of ``seq``'s prompt, hashed once per sequence.

    Kept on the sequence (``_umbp_store_chunk_hashes``): its prompt never
    changes, and lookups, saves and loads of it all need the same chain.
    """
    num_prompt = int(seq.num_prompt_tokens)
    expected = (num_prompt // int(chunk_tokens)) * DIGEST_BYTES
    cached = getattr(seq, "_umbp_store_chunk_hashes", None)
    if isinstance(cached, bytes) and len(cached) == expected:
        return cached
    hashes = chunk_hash_chain(
        seq.token_ids,
        chunk_tokens,
        num_tokens=num_prompt,
        seed=chain_seed(getattr(seq, "cache_seed", None)),
    )
    seq._umbp_store_chunk_hashes = hashes
    return hashes


class _ServerLookup:
    """The chunked scheduler's lookup client, answered by the UMBP server.

    A hit is the prefix every rank holds: per rank, the run of chunks present
    from chunk 0, and the minimum over ranks, since a load is all-or-nothing
    across the PP stages and TP ranks. One ``batch_exists`` per call (split
    into batches by the client); it grants nothing, so a chunk found may be
    evicted before its load, whose all-or-nothing get then fails.

    UMBP answers a failed RPC as every key absent, so a lookup cannot tell a
    server that did not answer from one that holds nothing: that costs a miss,
    never a wrong hit. Only a lookup that raised answers None.

    A server this process has marked (:mod:`.taint`) holds nothing for the
    lookups. The mark is this process's only; the workers, which place it,
    refuse their own loads either way.

    The client is opened on the first lookup, not at construction: every PP
    stage builds a scheduler, and only the head ever schedules. A lookup
    blocks the scheduler thread, so a server the scheduler cannot reach, or
    whose lookup raised, is not asked for a while (``_RECONNECT_INTERVAL_S``,
    ``_LOOKUP_BACKOFF_S``).
    """

    def __init__(
        self,
        cfg: UMBPStoreOffloadConfig,
        *,
        namespace: str,
        world: int,
        chunk_tokens: int,
    ) -> None:
        self._cfg = cfg
        self._namespace = namespace
        self._world = int(world)
        self._chunk_tokens = int(chunk_tokens)
        self.client: UMBPStoreClient | None = None
        self.paused_until = 0.0
        self._next_warning_at = 0.0
        # Answered lookups; those where the ranks held prefixes of different
        # lengths; and the rank objects past the shared prefix they found,
        # present but unusable while another rank lacks the chunk -- evicted,
        # failed, or still being written by a slower stage.
        self.lookups = 0
        self.uneven_lookups = 0
        self.stranded_objects = 0
        self._next_stats_at = time.monotonic() + _LOOKUP_STATS_INTERVAL_S

    def lookup(self, seq: Any, lookup_id: str | None = None) -> int | None:
        """Tokens of ``seq``'s prompt prefix the server holds on every rank.

        None, not 0, when the server could not be asked: the scheduler retries
        a non-answer later, where a 0 would be remembered as a miss.
        """
        started = time.perf_counter()
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        chunks = len(hashes) // DIGEST_BYTES
        if not chunks:
            return 0
        if taint.reason(self._cfg.address) is not None:
            return 0
        client = self._client_to_ask()
        if client is None:
            return None
        # Every rank's keys share the chunks' digests: each is hex-encoded once.
        digests = [chunk_digest(hashes, index).hex() for index in range(chunks)]
        keys = []
        for rank in range(self._world):
            prefix = rank_key_prefix(self._namespace, rank, self._world)
            keys += [prefix + digest for digest in digests]
        asked_at = time.monotonic()
        try:
            present = client.exists(keys)
        except Exception as exc:  # noqa: BLE001  # the server is not asked a while
            self._warn(
                "the UMBP server at %s failed a lookup of %d keys; it is not "
                "asked for %.0fs",
                self._cfg.address,
                len(keys),
                self._pause(_LOOKUP_BACKOFF_S, asked_at),
                exc_info=exc,
            )
            return None
        # Per rank, the run of chunks from chunk 0 the server holds.
        held = []
        for rank in range(self._world):
            shard = present[rank * chunks : (rank + 1) * chunks]
            held.append(next((i for i, ok in enumerate(shard) if not ok), chunks))
        hit_chunks = min(held)
        self._count_lookup(sum(n - hit_chunks for n in held))
        hit = hit_chunks * self._chunk_tokens
        logger.debug(
            "[OFFLOAD-LOOKUP] lookup_id=%s chunks=%d keys=%d hit=%d elapsed_ms=%.2f",
            lookup_id,
            chunks,
            len(keys),
            hit,
            (time.perf_counter() - started) * 1000,
        )
        return hit

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Nothing to release: a UMBP lookup holds nothing."""

    def _client_to_ask(self) -> UMBPStoreClient | None:
        """The client, connecting it if needed; None while paused."""
        if time.monotonic() < self.paused_until:
            return None
        if self.client is not None:
            return self.client
        started = time.monotonic()
        try:
            # Lookups only: no memory is registered.
            self.client = UMBPStoreClient(
                address=self._cfg.address,
                startup_timeout_ms=self._cfg.startup_timeout_ms,
            )
        except Exception:
            pause = self._pause(_RECONNECT_INTERVAL_S, started)
            logger.warning(
                "UMBP offload: the scheduler cannot reach the server at %s; "
                "lookups do not ask it for the next %.0fs",
                self._cfg.address,
                pause,
                exc_info=True,
            )
            return None
        logger.info("UMBP offload: scheduler lookups use %s", self._cfg.address)
        return self.client

    def _pause(self, floor_s: float, since: float) -> float:
        """Pause lookups after a failed call that started at ``since``.

        For ``floor_s`` or `_LOOKUP_BACKOFF_FACTOR` times as long as the call
        blocked, whichever is longer, counted from now. Returns the seconds
        paused.
        """
        now = time.monotonic()
        pause = max(floor_s, _LOOKUP_BACKOFF_FACTOR * (now - since))
        self.paused_until = now + pause
        return pause

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

    def _warn(self, message: str, *args: Any, exc_info: Any = None) -> None:
        """At most one warning per `_LOOKUP_WARNING_INTERVAL_S`."""
        now = time.monotonic()
        if now < self._next_warning_at:
            logger.debug("UMBP offload: " + message, *args)
            return
        self._next_warning_at = now + _LOOKUP_WARNING_INTERVAL_S
        logger.warning("UMBP offload: " + message, *args, exc_info=exc_info)


class UMBPStoreOffloadScheduler(ChunkedOffloadSchedulerBase):
    """Chunked offload scheduling with a UMBP standalone server as the tier."""

    # Workers publish one source-safe group per chunk once its GPU pack is
    # done (the block GPU connector), separately from the store terminal.
    _supports_early_block_release = True

    def __init__(self, config: Any) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        cfg = parse_umbp_store_config(kvc)
        check_engine_compatibility(cfg, config)
        require_monotonic_clock_id()
        world = offcfg.lmcache_replica_world_size(config)
        namespace = store_namespace(config, cfg.chunk_tokens)
        self._store_cfg = cfg
        self._namespace = namespace
        # Failed saves since the last stored one, and when saves may resume.
        self._failed_saves_in_a_row = 0
        self._saves_paused_until = 0.0
        super().__init__(
            config,
            chunk_size=cfg.chunk_tokens,
            lookup_client=_ServerLookup(
                cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
            ),
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
            "UMBP offload scheduler: namespace=%s world=%d chunk=%d server=%s "
            "max_pending_saves=%s abandon_timeout=%.0fs",
            namespace,
            world,
            cfg.chunk_tokens,
            cfg.address,
            self._max_pending_saves,
            cfg.save_abandon_timeout_s,
        )

    def _lookup_token_ids(self, seq: Any) -> Any:
        """The sequence itself: the lookup hashes its prompt once per sequence.

        Copying a long prompt into a list on every lookup would cost more than
        the lookup (`prompt_chunk_hashes`).
        """
        return seq

    def _fresh_tier_lookup(self, seq: Any, sid: str) -> int | None:
        # A prompt too short to load from the server, whatever it holds, is
        # not asked about: no hit is longer than the prompt less a token,
        # floored to a chunk (`_loadable_hit`), and no shorter one reaches the
        # load threshold. The lookup would block the scheduler thread for
        # nothing.
        num_prompt = int(seq.num_prompt_tokens)
        if self._loadable_hit(num_prompt, num_prompt) < self._min_load_tokens:
            return 0
        return super()._fresh_tier_lookup(seq, sid)

    def _ensure_lookup_pin(self, seq: Any, sid: str, spec: LoadSpec) -> bool:
        """Always dispatch the load: a UMBP lookup holds nothing to re-take.

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
            # failed and paused them: the server takes saves again.
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

        A setting of this connector (``umbp_store.save_abandon_timeout_s``)
        rather than a pin timeout: the workers stop reading a save's source
        blocks well before it runs out (see the worker's source read deadline).
        """
        return self._store_cfg.save_abandon_timeout_s


__all__ = [
    "UMBPStoreOffloadScheduler",
    "monotonic_clock_id",
    "prompt_chunk_hashes",
    "require_monotonic_clock_id",
]
