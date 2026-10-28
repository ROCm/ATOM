# SPDX-License-Identifier: MIT
"""CoreManager integration for cache-affinity routing and safe retry."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from atom.model_engine.dp_cache.balance import CacheBalanceConfig, CacheBalancer
from atom.model_engine.dp_cache.state import (
    CacheAdmissionEvent,
    CacheWorkEvent,
    DPCacheState,
)
from atom.utils import envs

if TYPE_CHECKING:
    from atom.config import Config
    from atom.model_engine.sequence import Sequence

logger = logging.getLogger("atom")


class DPCacheRouter:
    """Placement and event reconciliation under the manager's load lock.

    Load containers are shared explicitly with CoreManager and reset in place.
    The manager supplies a retry sender that owns framing and socket locking.
    """

    def __init__(
        self,
        config: Config,
        *,
        rank_requests: list[int],
        rank_tokens: list[int],
        request_loads: dict[object, tuple[int, int, int]],
        load_lock: AbstractContextManager,
        send_retry: Callable[[int, Sequence], None],
    ):
        self._rank_reqs = rank_requests
        self._rank_tokens = rank_tokens
        self._seq_load = request_loads
        self._lb_lock = load_lock
        self._send_retry = send_retry
        self._retry_payloads: dict[object, tuple[Sequence, object]] = {}
        if not config.enable_prefix_caching:
            raise ValueError("cache_affinity requires prefix caching")
        if getattr(config, "pipeline_parallel_size", 1) != 1:
            raise ValueError("cache_affinity currently requires PP=1")
        if getattr(config, "disagg_is_decode", False) or getattr(
            config, "disagg_is_prefill", False
        ):
            raise ValueError(
                "cache_affinity routes complete DP engines, not PD-only workers"
            )

        def has_mp(kvc):
            return isinstance(kvc, dict) and (
                kvc.get("kv_connector") == "lmcache_mp"
                or any(has_mp(c) for c in kvc.get("connectors", ()))
            )

        if not has_mp(getattr(config, "kv_transfer_config", None)):
            raise ValueError(
                "cache_affinity requires lmcache_mp and a shared cache tier; "
                "V4.1 must use the native PAGE/STATE connector"
            )
        hf = getattr(config, "hf_config", None)
        hf = getattr(hf, "text_config", hf)
        if getattr(hf, "model_type", None) == "deepseek_v41" and getattr(
            config, "enable_encoder_swa_bounded_replay", True
        ):
            raise ValueError(
                "V4.1 cache_affinity requires --no-encoder-swa-bounded-replay "
                "to retain native PAGE/STATE checkpoints for lmcache_mp"
            )
        values = json.loads(envs.ATOM_DP_CACHE_BALANCE_CONFIG)
        if isinstance(values, dict):
            values.setdefault(
                "checkpoint_tokens",
                getattr(config, "state_checkpoint_interval_tokens", 8192),
            )
        options = CacheBalanceConfig.parse(values)
        self.balancer = CacheBalancer(len(self._rank_reqs), options)

    def select_locked(self, seq: Sequence, hint: int | None) -> int:
        balancer = self.balancer
        chain = seq.prompt_hash_chain
        placement = balancer.place(
            seq.id,
            seq.dp_session_id,
            seq.num_prompt_tokens,
            chain,
            self._rank_reqs,
            self._rank_tokens,
            time.monotonic(),
            explicit_rank=hint,
            can_migrate=True,
        )
        seq.dp_cache = DPCacheState(
            migration_id=placement.migration,
            minimum_hit=placement.minimum_hit,
            prepare_timeout=balancer.config.prepare_timeout_seconds,
        )
        if placement.migration:
            # The frontend Sequence is never modified by the target process.
            # ADD detaches its hash chain, so retain that separately for retry.
            self._retry_payloads[seq.id] = (seq, chain)
        return placement.rank

    def admitted(self, rank: int, event: CacheAdmissionEvent) -> None:
        balancer = self.balancer
        sid, migration, hit = event
        with self._lb_lock:
            load = self._seq_load.get(sid)
            if load is None or load[0] != rank:
                return
            placement = balancer.requests.get(sid)
            if placement is None:
                return
            hit = max(0, min(int(hit), placement.prompt_tokens))
            if not balancer.admitted(sid, rank, migration, hit, time.monotonic()):
                return
            remaining = placement.prompt_tokens - hit
            self._rank_tokens[rank] += remaining - load[2]
            self._seq_load[sid] = (rank, load[1], remaining)
            self._retry_payloads.pop(sid, None)

    def retry(self, rank: int, returned_seq: Sequence) -> bool:
        """Consume target rejections, including stale replies after cancellation.

        Retries use the locked control socket: this is an output thread, not
        the single writer that owns the normal request socket.
        """
        state = returned_seq.dp_cache
        if state is None or state.retry_reason is None:
            return False

        balancer = self.balancer
        with self._lb_lock:
            sid = returned_seq.id
            payload = self._retry_payloads.get(sid)
            load = self._seq_load.get(sid)
            if payload is None or load is None or load[0] != rank:
                return True
            source = balancer.retry(
                sid,
                rank,
                state.migration_id,
                time.monotonic(),
                reason=state.retry_reason,
                recoverable_hit=state.recoverable_hit,
            )
            if source is None:
                return True
            original, chain = self._retry_payloads.pop(sid)
            original.prompt_hash_chain = chain
            original.dp_cache = DPCacheState(
                prepare_timeout=balancer.config.prepare_timeout_seconds
            )
            self._rank_reqs[rank] -= load[1]
            self._rank_tokens[rank] -= load[2]
            self._rank_reqs[source] += load[1]
            self._rank_tokens[source] += original.num_prompt_tokens
            self._seq_load[sid] = (source, load[1], original.num_prompt_tokens)
            # Keep abort's load removal ordered after this send. The control
            # socket has its own lock and sending needs no response callback.
            try:
                self._send_retry(source, original)
            except Exception:
                # Let ADD's normal terminal path answer the client and release
                # the transferred load charge. One failed retry must not end
                # this output thread and strand every other request on it.
                returned_seq.leave_reason = "cache_migration_retry_failed"
                balancer.counters["migration_retry_send_failed"] += 1
                logger.exception("Cache migration fallback could not be sent")
                return False
        return True

    def update_work(self, rank: int, events: Iterable[CacheWorkEvent]) -> None:
        balancer = self.balancer
        with self._lb_lock:
            for sid, migration, revision, remaining in events:
                load = self._seq_load.get(sid)
                placement = balancer.requests.get(sid)
                if (
                    load is None
                    or load[0] != rank
                    or placement is None
                    or not placement.admitted
                    or placement.work_migration != migration
                    or revision <= placement.work_revision
                ):
                    continue
                placement.work_revision = revision
                remaining = max(0, int(remaining))
                self._rank_tokens[rank] += remaining - load[2]
                self._seq_load[sid] = (rank, load[1], remaining)
                balancer.counters["prefill_work_updates"] += 1

    def finish_locked(self, sid: object, succeeded: bool) -> None:
        self.balancer.finish(sid, time.monotonic(), succeeded=succeeded)
        self._retry_payloads.pop(sid, None)

    def invalidate(self) -> None:
        with self._lb_lock:
            self.balancer.invalidate()

    def reset_locked(self) -> None:
        self.balancer = CacheBalancer(len(self._rank_reqs), self.balancer.config)
        self._retry_payloads.clear()
