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
    source_probe_timeout_seconds: float = 0.25
    hint_ttl_seconds: float = 600.0
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
                    "source_probe_timeout_seconds",
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
    reconsidering: bool = False
    reconsidered: bool = False


@dataclass(frozen=True)
class RecoveryChoice:
    rank: int
    source_hit: int
    target_hit: int
    source_seconds: float
    target_seconds: float


class CacheBalancer:
    def __init__(self, ranks: int, config: CacheBalanceConfig):
        self.ranks = ranks
        self.config = config
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.requests: dict[object, Placement] = {}
        self.counters = Counter()
        self._generation = 0
        self._cursor = 0
        self.cache_epoch = 0

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
        cfg = self.config
        waits = [
            max(0, t) / cfg.prefill_tokens_per_second
            + max(0, n) * cfg.decode_request_seconds
            for n, t in zip(requests, tokens, strict=True)
        ]
        if len(waits) != self.ranks:
            raise ValueError("DP cache load snapshot has the wrong rank count")
        lightest = min(
            range(self.ranks), key=lambda r: (waits[r], (r - self._cursor) % self.ranks)
        )
        self._cursor = (self._cursor + 1) % self.ranks
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
        hits = [0] * self.ranks
        if session:
            hits = [
                (
                    h.tokens
                    if (h := session.hints.get(r)) is not None
                    and h.matches(chain, now, cfg.hint_ttl_seconds)
                    else 0
                )
                for r in range(self.ranks)
            ]
            # A concurrent turn must not let an older in-flight migration
            # overwrite the owner's newer routing decision.
            if session.pending is not None:
                session.version += 1
            pending = [p for p in self.requests.values() if p.migration]
            eligible = (
                can_migrate
                and explicit_rank is None
                and not session.active
                and hits[rank] > 0
                and now - session.last_migration >= cfg.cooldown_seconds
                and len(pending) < cfg.max_migrations
            )
            if eligible:
                stay = (
                    waits[rank] + (prompt - hits[rank]) / cfg.prefill_tokens_per_second
                )
                candidates = []
                for target in range(self.ranks):
                    if (
                        target == rank
                        or sum(p.rank == target for p in pending)
                        >= cfg.max_migrations_per_rank
                    ):
                        continue
                    missing = max(0, hits[rank] - hits[target])
                    copy_time = (
                        (
                            cfg.transfer_fixed_seconds
                            + missing / cfg.transfer_tokens_per_second
                        )
                        if missing
                        else 0
                    )
                    finish = (
                        max(waits[target], copy_time)
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
                        candidates.append((finish, target))
                if candidates:
                    target = min(candidates)[1]
                    self._generation += 1
                    placement.source = rank
                    placement.rank = target
                    placement.migration = self._generation
                    placement.minimum_hit = hits[rank]
                    placement.session_version = session.version
                    session.pending = request_id
                    self.counters["migration_started"] += 1
            session.active.add(request_id)
        placement.estimated_hit = hits[placement.rank]
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
        if placement.admitted or placement.reconsidering or hit < placement.minimum_hit:
            return False
        placement.admitted = True
        self.counters["admitted"] += 1
        if migration:
            session = self.sessions.get(placement.session_id)
            if session and session.pending == request_id:
                session.pending = None
                session.last_migration = now
                if session.version == placement.session_version:
                    session.owner = rank
                    session.version += 1
                    self.counters["migration_committed"] += 1
                    if placement.reconsidered:
                        self.counters["migration_partial_committed"] += 1
                else:
                    self.counters["migration_owner_superseded"] += 1
            placement.migration = 0
        return True

    def begin_reconsider(self, request_id, rank, migration):
        placement = self.requests.get(request_id)
        if (
            placement is None
            or placement.rank != rank
            or not migration
            or placement.migration != migration
            or placement.admitted
            or placement.reconsidering
            or placement.reconsidered
        ):
            return False
        placement.reconsidering = True
        return True

    def reconsider(
        self, request_id, rank, migration, source_hit, target_hit, requests, tokens, now
    ):
        """Choose remaining work after a short restore; loads exclude this request.

        Hits are fresh snapshots, never authority to skip prefill. An unknown
        source gets no reuse credit. Preparation already spent is sunk cost;
        neither candidate is charged for another promised full-prefix copy.
        """
        p = self.requests.get(request_id)
        if (
            p is None
            or p.rank != rank
            or p.migration != migration
            or not p.reconsidering
            or p.admitted
        ):
            return None
        cfg = self.config
        source = p.source
        source_known = source_hit is not None
        source_hit = max(0, min(source_hit or 0, p.prompt_tokens))
        target_hit = max(0, min(target_hit, p.prompt_tokens))

        def remaining(r, hit):
            return (
                max(0, tokens[r]) + p.prompt_tokens - hit
            ) / cfg.prefill_tokens_per_second + max(
                0, requests[r]
            ) * cfg.decode_request_seconds

        stay = remaining(rank, target_hit)
        back = remaining(source, source_hit)
        # Returning to A is another move. Require the same configured gain
        # margin; ties/small differences keep the request at B.
        gain = stay - back
        go_back = gain > 0 and gain >= max(
            cfg.min_gain_seconds, stay * cfg.min_gain_ratio
        )
        choice = RecoveryChoice(
            source if go_back else rank, source_hit, target_hit, back, stay
        )
        self.counters["migration_reconsidered"] += 1
        if not source_known:
            self.counters["migration_source_unknown"] += 1
        session = self.sessions.get(p.session_id)
        if session:
            for r, hit, known in (
                (source, source_hit, source_known),
                (rank, target_hit, True),
            ):
                hint = session.hints.get(r)
                if known and hint and hint.tokens > hit:
                    del session.hints[r]
        p.reconsidering = False
        if go_back:
            self.counters["migration_reconsider_source"] += 1
            self.retry(
                request_id, rank, migration, now, reason="recoverable_prefix_too_short"
            )
            p.estimated_hit = source_hit
        else:
            self.counters["migration_reconsider_target"] += 1
            # Fence duplicate rejections/admissions from B's original attempt.
            self._generation += 1
            p.migration = self._generation
            p.minimum_hit = 0
            p.estimated_hit = target_hit
            p.reconsidered = True
        return choice

    def retry(self, request_id, rank, migration, now, *, reason=None):
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
        if session and session.pending == request_id:
            session.pending = None
            session.last_migration = now
        placement.rank = source
        placement.source = None
        placement.migration = 0
        placement.minimum_hit = 0
        placement.reconsidering = False
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
        self.cache_epoch += 1
        for session in self.sessions.values():
            session.hints.clear()
            session.version += 1
        for placement in self.requests.values():
            placement.hint = None

    def statistics(self):
        return {
            **self.counters,
            "sessions": len(self.sessions),
            "pending_migrations": sum(
                bool(p.migration) for p in self.requests.values()
            ),
        }
