# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of the UMBP offload.

The policy, the digest-carrying requests and the paced save retries are the
store-agnostic :class:`KVStoreOffloadScheduler`'s. This module answers the one
question it asks a transport, how much of a prompt the server holds. Unlike a
Mooncake Store: one server instead of pools, no read leases (a UMBP lookup
holds nothing, so nothing is renewed before a load; the base dispatches it as
it is), and no cross-client taint marker (see :mod:`.taint`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.kv_store import scheduler as _store_scheduler
from atom.kv_transfer.offload.kv_store.keys import DIGEST_BYTES
from atom.kv_transfer.offload.kv_store.scheduler import (
    KVStoreOffloadScheduler,
    StoreLookupBase,
    prompt_chunk_hashes,
)
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.client import UMBPStoreClient
from atom.kv_transfer.offload.umbp_store.config import (
    UMBPStoreOffloadConfig,
    check_engine_compatibility,
    parse_umbp_store_config,
    store_namespace,
)

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


def require_monotonic_clock_id() -> str:
    """The boot ID the saves are timed on, or a refusal to start without one."""
    return _store_scheduler.require_monotonic_clock_id("the UMBP offload")


class _ServerLookup(StoreLookupBase):
    """The chunked scheduler's lookup client, answered by the UMBP server.

    One ``batch_exists`` per call (split into batches by the client); it grants
    nothing, so a chunk found may be evicted before its load, whose
    all-or-nothing get then fails.

    UMBP answers a failed RPC as every key absent, so a lookup cannot tell a
    server that did not answer from one that holds nothing: that costs a miss,
    never a wrong hit. A lookup answers None, which the scheduler retries
    later, only when it could not ask: the call raised, or the server is not
    asked for a while (``_client_to_ask``) because it could not be reached or
    its last lookup raised.

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
        super().__init__(namespace=namespace, world=world, chunk_tokens=chunk_tokens)
        self._cfg = cfg
        self.client: UMBPStoreClient | None = None
        self.paused_until = 0.0
        self._next_warning_at = 0.0

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
        keys = self._rank_keys(hashes, 0, chunks)
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
        # Per rank, the run of chunks from chunk 0 the server holds. Every key
        # is answered, so `held` is the whole answer.
        held, _ = self._rank_prefixes(present, chunks)
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
        """The client, connecting it if needed; None while paused or when it
        cannot connect (which pauses it)."""
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

    def _warn(self, message: str, *args: Any, exc_info: Any = None) -> None:
        """At most one warning per `_LOOKUP_WARNING_INTERVAL_S`."""
        now = time.monotonic()
        if now < self._next_warning_at:
            logger.debug("UMBP offload: " + message, *args)
            return
        self._next_warning_at = now + _LOOKUP_WARNING_INTERVAL_S
        logger.warning("UMBP offload: " + message, *args, exc_info=exc_info)


class UMBPStoreOffloadScheduler(KVStoreOffloadScheduler):
    """Chunked offload scheduling with a UMBP standalone server as the tier."""

    LABEL = "UMBP offload"

    def __init__(self, config: Any) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        cfg = parse_umbp_store_config(kvc)
        check_engine_compatibility(cfg, config)
        require_monotonic_clock_id()
        world = offcfg.lmcache_replica_world_size(config)
        namespace = store_namespace(config, cfg.chunk_tokens)
        super().__init__(
            config,
            store_cfg=cfg,
            namespace=namespace,
            world=world,
            lookup_client=_ServerLookup(
                cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
            ),
            endpoint=f"server={cfg.address}",
        )


__all__ = ["UMBPStoreOffloadScheduler", "require_monotonic_clock_id"]
