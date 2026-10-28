# SPDX-License-Identifier: MIT
"""Session placement and transactional migration, independent of GPU transport.

Prefix hints predict cost; only a scheduler admission acknowledges residency.
All methods run under CoreManager's load lock. Clocks are coordinator-local;
only a relative preparation budget crosses the network.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field, fields
from math import isfinite


@dataclass(frozen=True)
class CacheBalanceConfig:
    # Initial estimates, not a V4.1 calibration. Operators can replace them
    # with rates measured on their model and parallel configuration.
    prefill_tokens_per_second: float = 32768.0
    transfer_tokens_per_second: float = 2000000.0
    transfer_fixed_seconds: float = 0.01
    decode_request_seconds: float = 0.002
    min_gain_seconds: float = 0.05
    min_gain_ratio: float = 0.15
    cooldown_seconds: float = 30.0
    prepare_timeout_seconds: float = 5.0
    hint_ttl_seconds: float = 600.0
    recovery_miss_ttl_seconds: float = 120.0
    max_sessions: int = 4096
    max_migrations: int = 8
    max_migrations_per_rank: int = 2
    checkpoint_tokens: int = 8192

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict):
            # Report malformed JSON option values through one validation API.
            raise ValueError(  # noqa: TRY004
                "DP cache balance config must be a JSON object"
            )
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown DP cache balance options: {sorted(unknown)}")
        result = cls(**values)
        for f in fields(cls):
            value = getattr(result, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{f.name} must be numeric")  # noqa: TRY004
            if not isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
            if (
                f.name
                in {
                    "prefill_tokens_per_second",
                    "transfer_tokens_per_second",
                    "prepare_timeout_seconds",
                    "hint_ttl_seconds",
                    "max_sessions",
                    "max_migrations",
                    "max_migrations_per_rank",
                    "checkpoint_tokens",
                }
                and value <= 0
            ):
                raise ValueError(f"{f.name} must be positive")
            if (f.name.startswith("max_") or f.name == "checkpoint_tokens") and type(
                value
            ) is not int:
                raise ValueError(f"{f.name} must be an integer")
        if result.min_gain_ratio >= 1:
            raise ValueError("min_gain_ratio must be less than one")
        return result


@dataclass(frozen=True)
class PrefixHint:
    tokens: int
    block_size: int
    seed: int
    digest: int
    observed_at: float

    def matches(self, chain, now, ttl):
        return bool(
            chain is not None
            and 0 <= now - self.observed_at <= ttl
            and chain.hash_block_size == self.block_size
            and chain.cache_seed == self.seed
            and chain.num_prompt_tokens > self.tokens
            and len(chain.hashes) >= self.tokens // self.block_size
            and chain.hashes[self.tokens // self.block_size - 1] == self.digest
        )


@dataclass
class Session:
    owner: int
    hints: dict[int, PrefixHint] = field(default_factory=dict)
    active: set = field(default_factory=set)
    last_migration: float = float("-inf")
    pending: object = None
    version: int = 0
    # Failed joint PAGE/STATE restoration, scoped to source, target and the
    # exact promised prefix. A newer completion is not a tier-save ACK.
    recovery_misses: dict[tuple[int, int], PrefixHint] = field(default_factory=dict)


@dataclass
class Placement:
    rank: int
    prompt_tokens: int
    session_id: str | None
    hint: PrefixHint | None
    estimated_hit: int = 0
    source: int | None = None
    migration: int = 0
    minimum_hit: int = 0
    session_version: int = 0
    admitted: bool = False
    source_hint: PrefixHint | None = None
    transfer_seconds: float = 0.0
    work_migration: int = 0
    work_revision: int = 0


class CacheBalancer:
    def __init__(self, ranks: int, config: CacheBalanceConfig):
        self.ranks = ranks
        self.config = config
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.requests: dict[object, Placement] = {}
        self.counters = Counter()
        self._generation = 0
        self._cursor = 0

    def _snapshot(self, chain, prompt, now):
        if chain is None or chain.num_prompt_tokens != prompt:
            return None
        block = chain.hash_block_size
        if block <= 0 or self.config.checkpoint_tokens % block:
            return None
        boundary = (prompt - 1) // self.config.checkpoint_tokens
        boundary *= self.config.checkpoint_tokens
        if boundary <= 0 or len(chain.hashes) < boundary // block:
            return None
        return PrefixHint(
            boundary, block, chain.cache_seed, chain.hashes[boundary // block - 1], now
        )

    def _session(self, name, owner):
        if not name:
            return None
        if name in self.sessions:
            self.sessions.move_to_end(name)
            return self.sessions[name]
        while len(self.sessions) >= self.config.max_sessions:
            victim = next((k for k, s in self.sessions.items() if not s.active), None)
            if victim is None:
                self.counters["session_table_full"] += 1
                return None
            del self.sessions[victim]
        session = Session(owner)
        self.sessions[name] = session
        return session

    def _decode_placement_scores(self, prompt, requests):
        """Increment in normalized request-count/context imbalance, up to a constant.

        Initial prompt lengths proxy each in-flight request's decode context;
        this needs no decode-step messages. For Phi=sum((n/mean_n)^2 +
        (context/mean_context)^2), the rank-dependent increment for adding a
        request of length q is proportional to n + q*context/mean_length^2,
        where mean_length includes the incoming request. Both candidate
        placements have the same post-placement totals and normalizers.
        These scores describe balance, not calibrated execution seconds.
        """
        count = sum(max(0, n) for n in requests)
        contexts = [0] * self.ranks
        for placement in self.requests.values():
            contexts[placement.rank] += placement.prompt_tokens
        if count <= 0 or not sum(contexts):
            return None
        mean_length = (sum(contexts) + prompt) / (count + 1)
        weight = prompt / (mean_length * mean_length)
        return [max(0, n) + weight * c for n, c in zip(requests, contexts)]

    def _estimate_waits(self, requests, tokens, pending):
        cfg = self.config
        transfers = [0.0] * self.ranks
        for p in pending:
            transfers[p.rank] += p.transfer_seconds
        waits = [
            max(max(0, t) / cfg.prefill_tokens_per_second, transfers[r])
            + max(0, n) * cfg.decode_request_seconds
            for r, (n, t) in enumerate(zip(requests, tokens, strict=True))
        ]
        return waits, transfers

    def _pick_lightest(self, waits, decode_scores):
        cfg = self.config
        lightest = min(
            range(self.ranks), key=lambda r: (waits[r], (r - self._cursor) % self.ranks)
        )
        if decode_scores is not None:
            # Inside the existing minimum useful TTFT-gain margin, prefer a
            # placement that balances both decode batch size and context work.
            candidates = [
                r
                for r in range(self.ranks)
                if waits[r] - waits[lightest] <= cfg.min_gain_seconds
            ]
            lightest = min(
                candidates,
                key=lambda r: (
                    decode_scores[r],
                    waits[r],
                    (r - self._cursor) % self.ranks,
                ),
            )
        self._cursor = (self._cursor + 1) % self.ranks
        return lightest

    def _cached_hits(self, session, chain, now):
        if session is None:
            return [0] * self.ranks
        cfg = self.config
        return [
            (
                h.tokens
                if (h := session.hints.get(r)) is not None
                and h.matches(chain, now, cfg.hint_ttl_seconds)
                else 0
            )
            for r in range(self.ranks)
        ]

    def _best_migration(
        self,
        session,
        rank,
        prompt,
        chain,
        hits,
        now,
        *,
        waits,
        transfers,
        decode_scores,
        pending,
    ):
        """Evaluate candidates; leave placement/session ownership unmodified."""
        cfg = self.config
        stay = waits[rank] + (prompt - hits[rank]) / cfg.prefill_tokens_per_second
        candidates = []
        for target in range(self.ranks):
            if (
                target == rank
                or sum(p.rank == target for p in pending) >= cfg.max_migrations_per_rank
            ):
                continue
            missing = max(0, hits[rank] - hits[target])
            miss = session.recovery_misses.get((rank, target))
            if (
                missing
                and miss is not None
                and miss.tokens == hits[rank]
                and miss.matches(chain, now, cfg.recovery_miss_ttl_seconds)
            ):
                self.counters["migration_recovery_miss_avoided"] += 1
                continue
            copy_time = (
                (cfg.transfer_fixed_seconds + missing / cfg.transfer_tokens_per_second)
                if missing
                else 0
            )
            finish = (
                max(waits[target], transfers[target] + copy_time)
                + (prompt - max(hits[rank], hits[target]))
                / cfg.prefill_tokens_per_second
            )
            gain = stay - finish
            # Compare completion estimates directly: even a lightly
            # loaded owner can benefit from moving. A zero-gain move
            # is never useful, including when both margins are zero.
            if gain > 0 and gain >= max(
                cfg.min_gain_seconds, stay * cfg.min_gain_ratio
            ):
                # Moving prefill off the owner must not increase joint
                # decode imbalance across the ranks sharing MoE work.
                if (
                    decode_scores is not None
                    and decode_scores[target] > decode_scores[rank]
                ):
                    self.counters["migration_decode_balance_rejected"] += 1
                    continue
                candidates.append((finish, target, copy_time))
        return min(candidates) if candidates else None

    def place(
        self,
        request_id,
        session_id,
        prompt,
        chain,
        requests,
        tokens,
        now,
        *,
        explicit_rank=None,
        can_migrate=False,
    ):
        if request_id in self.requests:
            raise ValueError("request already has a DP cache placement")
        if len(requests) != self.ranks or len(tokens) != self.ranks:
            raise ValueError("DP cache load snapshot has the wrong rank count")
        cfg = self.config
        pending = [p for p in self.requests.values() if p.migration]
        waits, transfers = self._estimate_waits(requests, tokens, pending)
        decode_scores = self._decode_placement_scores(prompt, requests)
        lightest = self._pick_lightest(waits, decode_scores)
        session = self._session(session_id, lightest)
        rank = session.owner if session else lightest
        if explicit_rank is not None:
            rank = explicit_rank
            if session:
                session.owner = rank
                session.version += 1
        placement = Placement(
            rank,
            prompt,
            session_id if session else None,
            self._snapshot(chain, prompt, now),
        )
        hits = self._cached_hits(session, chain, now)
        if session:
            # A concurrent turn must not let an older in-flight migration
            # overwrite the owner's newer routing decision.
            if session.pending is not None:
                session.version += 1
            eligible = (
                can_migrate
                and explicit_rank is None
                and not session.active
                and hits[rank] > 0
                and now - session.last_migration >= cfg.cooldown_seconds
                and len(pending) < cfg.max_migrations
            )
            if eligible:
                candidate = self._best_migration(
                    session,
                    rank,
                    prompt,
                    chain,
                    hits,
                    now,
                    waits=waits,
                    transfers=transfers,
                    decode_scores=decode_scores,
                    pending=pending,
                )
                if candidate is not None:
                    _, target, copy_time = candidate
                    self._generation += 1
                    placement.source = rank
                    placement.rank = target
                    placement.migration = self._generation
                    placement.minimum_hit = hits[rank]
                    placement.source_hint = session.hints[rank]
                    placement.transfer_seconds = copy_time
                    placement.session_version = session.version
                    session.pending = request_id
                    self.counters["migration_started"] += 1
            session.active.add(request_id)
        # Charge the same GPU work that won the cost comparison. Outstanding
        # restore work has its own queue above; it is not free or GPU prefill.
        placement.estimated_hit = max(hits[placement.rank], placement.minimum_hit)
        self.requests[request_id] = placement
        return placement

    def admitted(self, request_id, rank, migration, hit, now):
        placement = self.requests.get(request_id)
        if (
            placement is None
            or placement.rank != rank
            or placement.migration != migration
        ):
            return False
        if placement.admitted or hit < placement.minimum_hit:
            return False
        placement.admitted = True
        placement.work_migration = migration
        placement.transfer_seconds = 0.0
        self.counters["admitted"] += 1
        if migration:
            session = self.sessions.get(placement.session_id)
            if session and session.pending == request_id:
                session.recovery_misses.pop((placement.source, rank), None)
                session.pending = None
                session.last_migration = now
                if session.version == placement.session_version:
                    session.owner = rank
                    session.version += 1
                    self.counters["migration_committed"] += 1
                else:
                    self.counters["migration_owner_superseded"] += 1
            placement.migration = 0
        return True

    def retry(
        self, request_id, rank, migration, now, *, reason=None, recoverable_hit=None
    ):
        placement = self.requests.get(request_id)
        if (
            placement is None
            or placement.rank != rank
            or not migration
            or placement.migration != migration
            or placement.admitted
        ):
            return None
        source = placement.source
        session = self.sessions.get(placement.session_id)
        if (
            session is not None
            and reason == "recoverable_prefix_too_short"
            and type(recoverable_hit) is int
            and 0 <= recoverable_hit < placement.minimum_hit
            and (hint := placement.source_hint) is not None
        ):
            session.recovery_misses[source, rank] = PrefixHint(
                hint.tokens, hint.block_size, hint.seed, hint.digest, now
            )
            self.counters["migration_recovery_miss_recorded"] += 1
        if session and session.pending == request_id:
            session.pending = None
            session.last_migration = now
        placement.rank = source
        placement.source = None
        placement.migration = 0
        placement.minimum_hit = 0
        placement.transfer_seconds = 0.0
        # A failed destination is no proof of residency at the source either.
        placement.estimated_hit = 0
        self.counters["migration_fallback"] += 1
        # Fixed categories keep metrics bounded and identify which preparation
        # stage needs attention without per-request tracing on the hot path.
        if reason not in {
            "preparation_timeout",
            "recoverable_prefix_too_short",
            "alignment_load_unavailable",
            "alignment_chunk_unavailable",
            "admission_callback_unavailable",
        }:
            reason = "other"
        self.counters["migration_fallback_" + reason] += 1
        return source

    def finish(self, request_id, now, *, succeeded=False):
        placement = self.requests.pop(request_id, None)
        if placement is None:
            return
        session = self.sessions.get(placement.session_id)
        if session is None:
            return
        session.active.discard(request_id)
        if session.pending == request_id:
            session.pending = None
            session.last_migration = now
        if succeeded and placement.admitted and placement.hint:
            hint = placement.hint
            session.hints[placement.rank] = PrefixHint(
                hint.tokens, hint.block_size, hint.seed, hint.digest, now
            )

    def invalidate(self):
        """Cache clear / weight change: discard predictions and fence old commits."""
        for session in self.sessions.values():
            session.hints.clear()
            session.recovery_misses.clear()
            session.version += 1
        for placement in self.requests.values():
            placement.hint = None
            placement.source_hint = None

    def statistics(self):
        transfers = [0.0] * self.ranks
        for placement in self.requests.values():
            transfers[placement.rank] += placement.transfer_seconds
        return {
            **self.counters,
            "sessions": len(self.sessions),
            "pending_migrations": sum(
                bool(p.migration) for p in self.requests.values()
            ),
            "pending_restore_seconds_per_rank": transfers,
        }
