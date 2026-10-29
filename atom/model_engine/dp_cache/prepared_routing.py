# SPDX-License-Identifier: MIT
"""Coordinator for source-owned requests with a prepared target.

Source allocation and release are serialized by its scheduler. A target COMMIT
is sent only after SOURCE_RELEASED, never on the strength of a timeout. Methods
run under the router load lock; control-socket order fences cancellation.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

from atom.model_engine.dp_cache.balance import PrefixHint
from atom.model_engine.dp_cache.preparation import PrepareMessage
from atom.model_engine.dp_cache.state import DPCacheState


@dataclass
class _Attempt:
    seq: object
    chain: object
    source: int
    target: int
    generation: int
    phase: str = "queued"
    deadline: float = 0.0
    hbm_hit: int = 0
    boundary: int = 0
    forward_start: int = 0
    cancelled: bool = False


class PreparedRouting:
    def __init__(self, router, send):
        self.router = router
        self.send = send
        self.attempts: dict[object, _Attempt] = {}

    def _send(self, attempt, rank, kind, **values):
        self.send(
            rank, PrepareMessage(kind, attempt.seq.id, attempt.generation, **values)
        )

    def begin(self, seq, chain, placement):
        if (
            not self.router.balancer.config.prepare_migrations
            or not placement.migration
        ):
            return
        if self.send is None:
            raise RuntimeError(
                "prepare_migrations requires the ordered control transport"
            )
        a = _Attempt(seq, chain, placement.source, placement.rank, placement.migration)
        self.attempts[seq.id] = a
        placement.prepare_target = a.target
        placement.rank = a.source
        placement.estimated_hit = placement.source_hint.tokens
        placement.work_prefix = placement.estimated_hit
        seq.dp_cache = DPCacheState(
            preparation_id=a.generation,
            preparation_role="source",
            prepare_timeout=self.router.balancer.config.prepare_timeout_seconds,
        )
        counts = self.router.balancer.counters
        counts["migration_started"] -= 1
        counts["migration_prepare_started"] += 1

    def _move_load(self, sid, target, hit):
        router = self.router
        source, count, remaining = router._seq_load[sid]
        new_remaining = max(0, router.balancer.requests[sid].prompt_tokens - hit)
        router._rank_reqs[source] -= count
        router._rank_tokens[source] -= remaining
        router._rank_reqs[target] += count
        router._rank_tokens[target] += new_remaining
        router._seq_load[sid] = (target, count, new_remaining)

    def _source(self, a, *, requeue=False):
        router = self.router
        p = router.balancer.requests[a.seq.id]
        self._send(a, a.target, "cancel")
        p.rank, p.prepare_target, p.source = a.source, None, None
        p.migration = p.minimum_hit = 0
        p.transfer_seconds = 0.0
        p.estimated_hit = p.work_prefix = 0 if requeue else p.work_prefix
        session = router.balancer.sessions.get(p.session_id)
        if session is not None and session.pending == a.seq.id:
            session.pending = None
            session.last_migration = time.monotonic()
        self.attempts.pop(a.seq.id)
        router._retry_payloads.pop(a.seq.id, None)
        if requeue:
            self._move_load(a.seq.id, a.source, 0)
            a.seq.prompt_hash_chain = a.chain
            a.seq.dp_cache = DPCacheState()
            router._send_retry(a.source, a.seq)
        else:
            self._send(a, a.source, "keep_source")

    def _worthwhile(self, a):
        router = self.router
        balancer = router.balancer
        p = balancer.requests[a.seq.id]
        session = balancer.sessions.get(p.session_id)
        now = time.monotonic()
        if (
            a.cancelled
            or now >= a.deadline
            or session is None
            or session.version != p.session_version
            or p.source_hint is None
            or not p.source_hint.matches(a.chain, now, balancer.config.hint_ttl_seconds)
        ):
            return False
        requests = list(router._rank_reqs)
        work = router.prefill_work_locked()
        rank, count, remaining = router._seq_load[a.seq.id]
        requests[rank] -= count
        work[rank] = max(0.0, work[rank] - balancer.cost(p.work_prefix, remaining))
        pending = [
            other
            for other in balancer.requests.values()
            if other is not p and other.migration
        ]
        waits, transfers = balancer._estimate_waits(requests, work, pending)
        # The incoming request is removed from both decode dimensions before
        # evaluating where it should be added. No counterfactual double charge.
        current = balancer.requests.pop(a.seq.id)
        try:
            scores = balancer._decode_placement_scores(p.prompt_tokens, requests)
        finally:
            balancer.requests[a.seq.id] = current
        hits = [0] * balancer.ranks
        hits[a.source] = p.minimum_hit
        hits[a.target] = min(a.hbm_hit, p.prompt_tokens - 1)
        candidate = balancer._best_migration(
            session,
            a.source,
            p.prompt_tokens,
            a.chain,
            hits,
            now,
            waits=waits,
            transfers=transfers,
            decode_scores=scores,
            pending=pending,
            only_target=a.target,
            target_prefix=a.forward_start,
            transfer_boundary=a.boundary,
        )
        if candidate is None:
            return False
        p.transfer_seconds = candidate[2]
        return True

    def event(self, rank, event):
        a = self.attempts.get(event.request_id)
        if a is None or event.generation != a.generation:
            return
        router = self.router
        p = router.balancer.requests.get(event.request_id)
        if p is None or event.request_id not in router._seq_load:
            return
        counts = router.balancer.counters
        if rank == a.source and event.kind == "queued" and a.phase == "queued":
            a.phase = "preparing"
            shadow = pickle.loads(pickle.dumps(a.seq))
            shadow.dp_cache = DPCacheState(
                preparation_id=a.generation,
                preparation_role="target",
                migration_id=a.generation,
                minimum_hit=p.minimum_hit,
            )
            self._send(
                a,
                a.target,
                "prepare",
                seq=shadow,
                chain=a.chain,
                minimum_hit=p.minimum_hit,
                ttl=router.balancer.config.prepare_timeout_seconds,
            )
        elif (
            rank == a.source and event.kind == "source_kept" and a.phase != "committing"
        ):
            counts["migration_prepare_source_won"] += 1
            self._source(a)
        elif rank == a.target and event.kind in {"failed", "short"}:
            counts["migration_prepare_" + event.kind] += 1
            if event.kind == "short" and event.boundary < p.minimum_hit:
                session = router.balancer.sessions.get(p.session_id)
                hint = p.source_hint
                if session is not None and hint is not None:
                    session.recovery_misses[a.source, a.target] = PrefixHint(
                        hint.tokens,
                        hint.block_size,
                        hint.seed,
                        hint.digest,
                        time.monotonic(),
                    )
            if a.phase == "releasing":
                # A may already have removed the request. Wait for its ACK;
                # never resurrect it based on target timeout alone.
                a.cancelled = True
            else:
                self._source(a, requeue=a.phase == "committing")
        elif rank == a.target and event.kind == "ready" and a.phase == "preparing":
            if event.boundary < p.minimum_hit or event.ttl <= 0:
                self._source(a)
                return
            # Different hosts have different clocks. Deadline is only a cost
            # hint here; target rechecks its own original clock at commit.
            a.deadline = time.monotonic() + event.ttl
            a.hbm_hit = max(0, event.hbm_hit)
            a.boundary = event.boundary
            a.forward_start = (
                event.boundary if event.forward_start is None else event.forward_start
            )
            if not 0 <= a.forward_start <= a.boundary < p.prompt_tokens:
                self._source(a)
                return
            if not self._worthwhile(a):
                counts["migration_prepare_gain_disappeared"] += 1
                self._source(a)
                return
            a.phase = "releasing"
            self._send(a, a.source, "release_source")
        elif (
            rank == a.source
            and event.kind == "source_released"
            and a.phase == "releasing"
        ):
            if not self._worthwhile(a):
                counts["migration_prepare_gain_disappeared"] += 1
                self._source(a, requeue=True)
                return
            a.phase = "committing"
            # Cost/accounting and admission use the actual replay start;
            # transport still loads through the pinned PAGE boundary.
            p.minimum_hit = min(p.minimum_hit, a.forward_start)
            self._move_load(a.seq.id, a.target, a.forward_start)
            p.rank, p.prepare_target = a.target, None
            p.work_prefix = p.estimated_hit = a.forward_start
            counts["migration_started"] += 1
            self._send(a, a.target, "commit")
        elif rank == a.target and event.kind == "committed" and a.phase == "committing":
            counts["migration_prepare_committed"] += 1
            self.attempts.pop(a.seq.id)
            # Ordinary post-load admission / DMA-safe retry now owns it.

    def forget(self, sid):
        a = self.attempts.pop(sid, None)
        if a is not None:
            self._send(a, a.target, "cancel")
            self._send(a, a.source, "keep_source")

    def invalidate(self):
        for a in list(self.attempts.values()):
            if a.phase == "releasing":
                a.cancelled = True
                self._send(a, a.target, "cancel")
            elif a.phase != "committing":
                self._source(a)
