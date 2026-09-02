# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unified engine statistics.

Holds :class:`EngineStats`, which folds in the former ``SpecStats`` and
``CacheStats`` and adds a throughput section alongside them. Kept out of
``scheduler.py`` because it is engine-level rather than scheduler-level: every
scheduler owns one, and ``engine_utility`` reads the one on the scheduler it
was handed to answer ``/debug/mtp_stats``, ``/debug/cache_stats`` and
``/metrics``.

Both P/D processes replace the scheduler that ``EngineCore.__init__`` built
the handler around, so each rewires ``utility_handler.scheduler`` to the one
it actually runs; the prefill side's snapshot then carries no ``kv_blocks_*``
keys, because the decode process owns the blocks.
"""

import logging
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("atom")


class EngineStats:
    """Unified engine statistics: the former ``SpecStats`` and ``CacheStats``
    folded together, with a throughput section added beside them.

    Three independent sections, each gated by its own enable flag and each
    emitting at its **own pace**:

    - **spec** — speculative-decoding acceptance. Logs every
      ``spec_log_interval * mtp_k`` draft tokens (count-based).
    - **cache** — prefix-caching hit statistics. Logs every
      ``cache_log_interval`` requests (count-based).
    - **throughput** — engine-status summary line (running/waiting requests,
      KV usage, prefix-cache hit rate, prompt/generation throughput),
      mirroring vLLM's ``LoggingStatLogger``. Logs every
      ``throughput_log_interval_s`` wall-clock seconds (time-based)::

          Engine 000: Avg prompt throughput: 254.4 tokens/s, Avg generation
          throughput: 0.3 tokens/s, Running: 6 reqs, Waiting: 0 reqs, GPU KV
          cache usage: 0.0%, Prefix cache hit rate: 0.0%

    A disabled section's ``update_*`` / ``maybe_log_*`` entry points are
    no-ops, so callers need not guard on the feature flag.
    """

    __slots__ = (
        "_cache_interval_cached_tokens",
        "_cache_interval_compressed_tokens",
        "_cache_interval_evicted_base",
        "_cache_interval_full_tokens",
        "_cache_interval_requests",
        "_cache_interval_reusable_tokens",
        "_cache_interval_wanted_tokens",
        "_cache_log_interval",
        "_pool_pressure",
        "_recent_cached_tokens",
        "_recent_hits",
        "_recent_reusable_tokens",
        "_recent_window",
        "_spec_interval_distribution",
        "_spec_interval_draft_tokens",
        "_spec_log_interval",
        "_throughput_last_log_time",
        "block_manager",
        "cache_enabled",
        "distribution",
        "engine_index",
        "label",
        "mtp_k",
        "num_generation_tokens",
        "num_prompt_tokens",
        "spec_enabled",
        "throughput_enabled",
        "throughput_log_interval_s",
        "total_cached_tokens",
        "total_compressed_tokens",
        "total_draft_tokens",
        "total_full_tokens",
        "total_requests",
        "total_reusable_tokens",
        "total_wanted_tokens",
    )

    def __init__(
        self,
        *,
        engine_index: int = 0,
        label: str = "",
        use_spec: bool = False,
        mtp_k: int = 0,
        enable_prefix_caching: bool = False,
        enable_log_stats: bool = False,
        spec_log_interval: int = 1000,
        cache_log_interval: int = 100,
        cache_hit_rate_window: int = 1000,
        pool_pressure: Callable[[], dict[str, int]] | None = None,
        throughput_log_interval_s: float = 10.0,
    ):
        self.spec_enabled = use_spec
        self.cache_enabled = enable_prefix_caching
        self.throughput_enabled = enable_log_stats

        # ── spec section ─────────────────────────────────────────────────
        self.mtp_k = mtp_k
        # Log every spec_log_interval decode steps (in terms of draft tokens)
        self._spec_log_interval = spec_log_interval * mtp_k
        self.total_draft_tokens: int = 0
        self.distribution: dict[int, int] = {k: 0 for k in range(mtp_k + 1)}
        # Per-interval tracking
        self._spec_interval_draft_tokens: int = 0
        self._spec_interval_distribution: dict[int, int] = {
            k: 0 for k in range(mtp_k + 1)
        }

        # ── cache section ────────────────────────────────────────────────
        self._cache_log_interval = cache_log_interval
        # Read at log time rather than passed per update: the free-list scan
        # behind it is O(free blocks), which is ~10k here and would be paid
        # once per request for a line printed once per `cache_log_interval`.
        self._pool_pressure = pool_pressure
        self.total_requests: int = 0
        self.total_cached_tokens: int = 0
        self.total_full_tokens: int = 0
        # The reuse ceiling, and the only honest denominator for a hit rate.
        #
        # `full` is not reachable: `BlockManager.can_allocate` matches over
        # `range(n_hash_blocks - 1)`, because prefill must forward at least one
        # block to produce sampler logits, so a request's own trailing block
        # never comes from cache. Dividing by `full` therefore charges both
        # pools for a block neither was ever offered, and reports a ceiling of
        # 100% that no run can reach.
        #
        # It also silently rescales with sequence length -- the unreachable
        # block is a fixed `hash_block_size`, so it is ~13% of a 1k prompt and
        # ~0.05% of a 275k one. A hit rate divided by `full` thus moves with
        # the length mix even when both pools behave identically, which is
        # exactly the confound that makes two runs incomparable.
        self.total_reusable_tokens: int = 0
        # Pre-gate compressed-prefix hit tokens, and the boundary between the
        # two pools: everything below it is the paged pool's doing, everything
        # between it and `cached` is the state gates'. `reusable - compressed`
        # is reuse the paged pool could not offer; `compressed - cached` is
        # reuse it offered and the Pool.STATE gates declined.
        self.total_compressed_tokens: int = 0
        # Where the gates would have landed with every state ladder dense. It
        # sits between cached and compressed and splits the declined reuse in
        # two: below it a checkpoint was missing, above it nothing would have
        # helped. Without the split "declined" is one number and whether
        # demand-driven checkpointing applies to a workload is unfalsifiable.
        self.total_wanted_tokens: int = 0
        self._cache_interval_requests: int = 0
        self._cache_interval_cached_tokens: int = 0
        self._cache_interval_full_tokens: int = 0
        self._cache_interval_compressed_tokens: int = 0
        self._cache_interval_wanted_tokens: int = 0
        self._cache_interval_reusable_tokens: int = 0
        # Sliding window behind `recent_cache_hit_rate`, the figure the
        # engine-status line reports. Mirrors vLLM's PrefixCachingMetrics: the
        # last N requests, aggregated incrementally so `observe` stays O(1).
        if cache_hit_rate_window <= 0:
            raise ValueError(
                f"cache_hit_rate_window must be > 0, got {cache_hit_rate_window}"
            )
        self._recent_window = cache_hit_rate_window
        self._recent_hits: deque[tuple[int, int]] = deque()
        self._recent_reusable_tokens: int = 0
        self._recent_cached_tokens: int = 0
        # Set by Scheduler for pool occupancy logging.
        self.block_manager = None
        self._cache_interval_evicted_base: int = 0

        # ── throughput section ───────────────────────────────────────────
        # A real exception, not an assert: `python -O` strips asserts, and the
        # only thing between a non-positive interval and the division in
        # `maybe_log_throughput` is this check. Stripped, the engine would die
        # of ZeroDivisionError inside the scheduler loop.
        if throughput_log_interval_s <= 0:
            raise ValueError(
                "throughput_log_interval_s must be > 0, got "
                f"{throughput_log_interval_s}"
            )
        self.engine_index = engine_index
        # Distinguishes the P/D processes, which both run as engine index 0 and
        # usually log to the same place. "" for the aggregated engine.
        self.label = label
        self.throughput_log_interval_s = throughput_log_interval_s
        self._throughput_last_log_time = time.monotonic()
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0

    # ══ spec section ═════════════════════════════════════════════════════

    def update_spec(self, num_accepted_tokens: int) -> None:
        """Record acceptance result for one sequence in one decode step."""
        if not self.spec_enabled:
            return
        self.total_draft_tokens += self.mtp_k
        self._spec_interval_draft_tokens += self.mtp_k
        num_bonus = num_accepted_tokens - 1
        self.distribution[num_bonus] += 1
        self._spec_interval_distribution[num_bonus] += 1

        if self.total_draft_tokens % self._spec_log_interval == 0:
            self.log_spec()
            self._reset_spec_interval()

    @property
    def total_accepted(self) -> int:
        """Total number of accepted bonus tokens across all steps."""
        return sum(k * v for k, v in self.distribution.items())

    @property
    def total_steps(self) -> int:
        """Total number of decode steps recorded."""
        return sum(self.distribution.values())

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_draft_tokens

    def spec_statistics(self) -> dict:
        """Return a summary dict compatible with engine_core reporting."""
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "distribution": dict(self.distribution),
        }

    def reset_spec(self) -> None:
        self.total_draft_tokens = 0
        self.distribution = {k: 0 for k in range(self.mtp_k + 1)}
        self._reset_spec_interval()

    def _reset_spec_interval(self) -> None:
        self._spec_interval_draft_tokens = 0
        self._spec_interval_distribution = {k: 0 for k in range(self.mtp_k + 1)}

    def log_spec(self) -> None:
        ts = self.total_steps
        if ts == 0:
            return
        # Interval stats
        iv_steps = sum(self._spec_interval_distribution.values())
        if iv_steps == 0:
            self._reset_spec_interval()
            return
        iv_accepted = sum(k * v for k, v in self._spec_interval_distribution.items())
        iv_rate = (
            iv_accepted / self._spec_interval_draft_tokens
            if self._spec_interval_draft_tokens > 0
            else 0.0
        )
        logger.info(
            f"[MTP Stats Interval] Average toks/fwd: {1 + iv_accepted / iv_steps:.2f}, "
            f"Accepted/Total Draft tokens: {iv_accepted}/{self._spec_interval_draft_tokens}, "
            f"Acceptance rate: {iv_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / iv_steps:.2%}' for k, v in self._spec_interval_distribution.items()} }"
        )
        logger.info(
            f"[MTP Stats         ] Average toks/fwd: {1 + self.total_accepted / ts:.2f}, "
            f"Accepted/Total Draft tokens: {self.total_accepted}/{self.total_draft_tokens}, "
            f"Acceptance rate: {self.acceptance_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / ts:.2%}' for k, v in self.distribution.items()} }"
        )

    # ══ cache section ════════════════════════════════════════════════════

    def update_cache(
        self,
        num_cached_tokens: int,
        num_full_tokens: int,
        num_compressed_tokens: int,
        num_wanted_tokens: int,
        num_reusable_tokens: int,
    ) -> None:
        """Record cache stats for one prefill sequence.

        All five are required because the reported rates are differences
        between them: `cached <= wanted <= compressed <= reusable <= full`. A
        defaulted argument would silently report a negative rate rather than a
        missing one.

        `reusable` is the caller's, not this class's, because the gap between
        it and `full` is a `BlockManager` matching detail (the trailing block
        has no stable hash, so it is never a reuse candidate). Recomputing it
        here would mean duplicating that rule in a second place and letting the
        two drift.
        """
        if not self.cache_enabled:
            return
        ordered = (
            num_cached_tokens
            <= num_wanted_tokens
            <= num_compressed_tokens
            <= num_reusable_tokens
            <= num_full_tokens
        )
        if not ordered:
            # Warned and clamped, not asserted. These are logging counters, and
            # `num_cached_tokens` has four independent writers -- the CPU-offload
            # wake at `_wake_offloaded_seq` sets it without touching the two
            # hit-block counters `can_allocate` derives the rest from, so an
            # LMCache resume that loads more prefix than the GPU index held
            # produces `cached > wanted` legitimately. Aborting `schedule()`
            # over it would take the engine down to protect a log line, and
            # would do so only in builds without `-O`, so the two would differ
            # in behaviour. Clamping keeps the rates monotone and the run alive.
            logger.warning(
                "Cache stats ordering violated, clamping: cached=%d wanted=%d "
                "compressed=%d reusable=%d full=%d",
                num_cached_tokens,
                num_wanted_tokens,
                num_compressed_tokens,
                num_reusable_tokens,
                num_full_tokens,
            )
            num_full_tokens = max(num_full_tokens, 0)
            num_reusable_tokens = min(max(num_reusable_tokens, 0), num_full_tokens)
            num_compressed_tokens = min(
                max(num_compressed_tokens, 0), num_reusable_tokens
            )
            num_wanted_tokens = min(max(num_wanted_tokens, 0), num_compressed_tokens)
            num_cached_tokens = min(max(num_cached_tokens, 0), num_wanted_tokens)
        self.total_requests += 1
        self.total_cached_tokens += num_cached_tokens
        self.total_full_tokens += num_full_tokens
        self.total_reusable_tokens += num_reusable_tokens
        self.total_compressed_tokens += num_compressed_tokens
        self.total_wanted_tokens += num_wanted_tokens
        self._cache_interval_requests += 1
        self._cache_interval_cached_tokens += num_cached_tokens
        self._cache_interval_full_tokens += num_full_tokens
        self._cache_interval_reusable_tokens += num_reusable_tokens
        self._cache_interval_compressed_tokens += num_compressed_tokens
        self._cache_interval_wanted_tokens += num_wanted_tokens

        # Slide the recent-requests window. One call is one request here, so
        # the deque length is the request count directly. Denominated in
        # `reusable`, like every rate below it — see `total_reusable_tokens`.
        self._recent_hits.append((num_reusable_tokens, num_cached_tokens))
        self._recent_reusable_tokens += num_reusable_tokens
        self._recent_cached_tokens += num_cached_tokens
        if len(self._recent_hits) > self._recent_window:
            old_reusable, old_cached = self._recent_hits.popleft()
            self._recent_reusable_tokens -= old_reusable
            self._recent_cached_tokens -= old_cached

        if self.total_requests % self._cache_log_interval == 0:
            self._log_cache()
            self._reset_cache_interval()

    @property
    def cache_hit_rate(self) -> float:
        """End-to-end reuse, against what was reusable at all.

        Lifetime figure: feeds `/debug/cache_stats`, `/metrics` and the
        cumulative row of the `[Cache Stats]` line, all cumulative by design.

        Not against `total_full_tokens`: that denominator includes each
        request's trailing block, which no cache is ever allowed to serve, so
        it reports a ceiling nothing can reach and drifts with the prompt
        length mix. `paged_hit_rate * state_hit_rate == hit_rate` exactly.
        """
        return self._rate(self.total_cached_tokens, self.total_reusable_tokens)

    @property
    def paged_hit_rate(self) -> float:
        """The paged KV pool's own hit rate, with the state cache factored out.

        Denominator is what the paged pool was asked for (`reusable`);
        numerator is what it had (`compressed`). The state gates run strictly
        after this and cannot change either term, so this number is unaffected
        by checkpoint policy -- change `--state-checkpoint-*` and this should
        not move. It answers "is the prefix still in KV?" and nothing else.

        What it charges the pool for: eviction, capacity, and genuinely novel
        prefixes. That last one is a workload property, not a defect, so this
        rate has a ceiling below 100% set by how much of the traffic is new
        text -- compare it against the dataset's theoretical prefix hit, not
        against 100%.
        """
        return self._rate(self.total_compressed_tokens, self.total_reusable_tokens)

    @property
    def state_hit_rate(self) -> float:
        """The state cache's own hit rate, with the paged pool factored out.

        Denominator is what the paged pool actually offered (`compressed`),
        NOT `reusable` -- the state gates never see a prefix the paged pool
        already lost, and charging them for it would mean a KV eviction shows
        up as a state-cache failure and sends tuning at the wrong pool. This
        is the conditional probability: given the prefix was there, did a
        checkpoint let us resume from it?

        Unlike `paged_hit_rate`, 100% is genuinely reachable here: it means
        every boundary the paged pool offered had a resumable checkpoint. The
        gap decomposes into `state_recoverable_loss_rate` (a checkpoint would
        have fixed it) and the remainder (nothing would have).
        """
        return self._rate(self.total_cached_tokens, self.total_compressed_tokens)

    @property
    def state_recoverable_loss_rate(self) -> float:
        """The part of the state cache's miss that checkpoint placement owns.

        Same denominator as `state_hit_rate`, so the two compose:
        `state_hit_rate + state_recoverable_loss_rate` is the rate the state
        cache would reach with a dense ladder. The distance from that to 1.0
        is the part no checkpoint can buy, and so the honest cap on what any
        amount of checkpoint capacity is worth.
        """
        return self._rate(
            self.total_wanted_tokens - self.total_cached_tokens,
            self.total_compressed_tokens,
        )

    @property
    def recent_cache_hit_rate(self) -> float | None:
        """`cache_hit_rate` over the last `cache_hit_rate_window` requests.

        What the engine-status line reports, because every other field on that
        line is windowed: a lifetime figure there stops tracking the workload,
        and on a long run it is dominated by traffic hours old. vLLM's
        `LoggingStatLogger` uses a 1000-request window for the same reason.
        Same `reusable` denominator as the lifetime rate, so the two are
        comparable.

        None when the window holds nothing, which the line renders `n/a`
        rather than as a measured 0%.
        """
        if not self._recent_reusable_tokens:
            return None
        return self._recent_cached_tokens / self._recent_reusable_tokens

    def cache_statistics(self) -> dict:
        """Counters, not rates — the caller derives those.

        Every rate this class reports is a ratio of two of these totals, and a
        rate cannot be aggregated across DP ranks that saw different token
        counts. Handing back the counts keeps the merge a sum.
        """
        return {
            "requests": self.total_requests,
            "cached_tokens": self.total_cached_tokens,
            "compressed_tokens": self.total_compressed_tokens,
            "wanted_tokens": self.total_wanted_tokens,
            # The denominator for every rate here. `full_tokens` is reported
            # too, but only so a consumer can see the unreachable gap; it is
            # not a hit-rate denominator -- see `total_reusable_tokens`.
            "reusable_tokens": self.total_reusable_tokens,
            "full_tokens": self.total_full_tokens,
        }

    def _reset_cache_interval(self) -> None:
        self._cache_interval_requests = 0
        self._cache_interval_cached_tokens = 0
        self._cache_interval_full_tokens = 0
        self._cache_interval_compressed_tokens = 0
        self._cache_interval_wanted_tokens = 0
        self._cache_interval_reusable_tokens = 0

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    def _log_cache(self) -> None:
        # compressed = pre-gate prefix hit; cached = post-gate (admitted); the
        # two differ by what the Pool.STATE gates declined, and `wanted` splits
        # that difference where it matters:
        #   Lost-to-checkpoint  wanted - cached, reuse a checkpoint at that
        #                       boundary would have delivered. What the demand
        #                       rung goes after; expected to fall toward 0 on a
        #                       workload with a genuinely shared prefix.
        #   Lost-unrecoverable  compressed - wanted, declined for a reason no
        #                       checkpoint touches: the SWA tail is gone, or the
        #                       boundary is too near the prompt's end to fork.
        # (reusable - compressed) is the rest: compressed eviction, or no reuse.
        self._cache_log_line(
            "Interval",
            self._cache_interval_requests,
            self._cache_interval_cached_tokens,
            self._cache_interval_compressed_tokens,
            self._cache_interval_wanted_tokens,
            self._cache_interval_reusable_tokens,
            self._cache_interval_full_tokens,
        )
        self._cache_log_line(
            "        ",
            self.total_requests,
            self.total_cached_tokens,
            self.total_compressed_tokens,
            self.total_wanted_tokens,
            self.total_reusable_tokens,
            self.total_full_tokens,
        )
        if self.block_manager is not None:
            occ = self.block_manager.pool_occupancy()
            evicted_iv = occ["evicted_total"] - self._cache_interval_evicted_base
            self._cache_interval_evicted_base = occ["evicted_total"]
            total = occ["total"] or 1
            logger.info(
                f"[Cache Pool          ] "
                f"used {occ['used']} ({occ['used'] / total:.0%}), "
                f"free {occ['free']} ({occ['free'] / total:.0%}), "
                f"retained-cache {occ['retained']}, "
                f"evicted this interval {evicted_iv} "
                f"(total {occ['evicted_total']})"
            )
        self._log_pools()
        if self._pool_pressure is not None:
            self._log_pressure(self._pool_pressure())

    def _log_pools(self) -> None:
        """Each pool's hit rate against its own denominator.

        The `[Cache Stats]` line reports one end-to-end rate, which cannot say
        which pool to fix: the same 85% is a KV pool that lost the prefix or a
        state cache that refused to resume from it, and those want opposite
        changes. Splitting needs two denominators, because the pools are in
        series and the second only ever sees what the first passed on:

            paged = compressed / reusable      "was the prefix still in KV?"
            state = cached     / compressed    "given it was, could we resume?"
            paged * state = cached / reusable = the end-to-end rate

        So the product is exact, and the smaller factor is the bottleneck --
        that comparison is the whole point of the line. Reading `state`
        against `reusable` instead would fold KV evictions into the state
        cache's score and point tuning at the wrong pool.

        `+ckpt` is where `state` would land if every ladder were dense. It is
        the ceiling on what checkpoint placement or more slots can buy; if it
        sits near `state`, the state cache is already doing all it can and the
        remaining loss is the paged pool's.
        """
        paged = self.paged_hit_rate
        state = self.state_hit_rate
        # Which factor is further from 1.0 loses more reuse, since the rates
        # multiply. Named here rather than left to the reader because the
        # comparison is against each other, not against 100%.
        worse = "paged" if paged <= state else "state"
        logger.info(
            "[Cache Pools] "
            f"paged-hit: {paged:.2%} "
            f"({self.total_compressed_tokens}/{self.total_reusable_tokens}), "
            f"state-hit: {state:.2%} "
            f"({self.total_cached_tokens}/{self.total_compressed_tokens}), "
            f"state-hit+ckpt: {state + self.state_recoverable_loss_rate:.2%}, "
            f"combined: {self.cache_hit_rate:.2%}, "
            f"binding: {worse}"
        )

    @staticmethod
    def _log_pressure(p: dict[str, int]) -> None:
        """The two pools' own account of what they destroyed.

        `full - compressed` in the line above is reuse the paged pool did not
        have, but it cannot say why — a prompt with no shared prefix and a
        prefix evicted an hour ago read identically. These counters separate
        them, and are the only evidence that eviction happened at all:
        `blocks_evicted == 0` at the end of a run means every miss above was
        absence of reuse, not loss of it.

        Vacant is called out because it is the leading indicator. Evictions
        can only begin once it reaches 0, so a run that ends with vacant
        blocks to spare never had paged pressure whatever its hit rate.
        """
        logger.info(
            "[Pool Pressure] "
            f"paged: {p['blocks_used']}/{p['blocks_total']} used, "
            f"{p['blocks_free_reusable']} reusable-free, "
            f"{p['blocks_free'] - p['blocks_free_reusable']} vacant, "
            f"{p['blocks_indexed']} indexed | "
            f"evicted: {p['blocks_evicted']}, retired: {p['blocks_retired']} | "
            f"state: {p['slots_used']}/{p['slots_total']} used, "
            f"{p['slots_held']} checkpointed, {p['slots_vacant']} vacant"
        )
        # The state pool's own losses, which `blocks_evicted` cannot express:
        # a checkpoint can die without any block dying (`evicted`, the pool ran
        # out of slots) or *because* a block died (`orphaned`, the prefix it
        # was filed under left the KV index first). The pair says which pool to
        # grow — see `StateSlotPool.__init__` for why they are kept apart.
        logger.info(
            "[Checkpoint Fates] "
            f"kept: {p['checkpoints_kept']}, "
            f"dropped: {p['checkpoints_dropped']}, "
            f"evicted: {p['checkpoints_evicted']}, "
            f"orphaned: {p['checkpoints_orphaned']}"
        )

    @classmethod
    def _cache_log_line(
        cls,
        label: str,
        reqs: int,
        cached: int,
        compressed: int,
        wanted: int,
        reusable: int,
        full: int,
    ) -> None:
        """Every rate here is over `reusable`; `full` is shown, not divided by.

        `Unreachable` is the gap between them -- the trailing block of each
        request, which prefill must always compute. It is reported so the
        older `full`-denominated numbers in past logs remain translatable, and
        because a large value is itself a signal: it means short prompts
        dominate, and a run whose length mix differs this much is not
        comparable to another on hit rate alone.
        """
        logger.info(
            f"[Cache Stats {label}] Reqs: {reqs}, "
            f"Cached/Reusable: {cached}/{reusable}, "
            f"Hit: {cls._rate(cached, reusable):.2%}, "
            f"Compressed-hit: {cls._rate(compressed, reusable):.2%}, "
            f"Lost-to-checkpoint: {cls._rate(wanted - cached, reusable):.2%}, "
            f"Lost-unrecoverable: {cls._rate(compressed - wanted, reusable):.2%}, "
            f"Unreachable: {cls._rate(full - reusable, full):.2%} of {full}"
        )

    # ══ throughput section ═══════════════════════════════════════════════

    def update_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        if not self.throughput_enabled:
            return
        self.num_prompt_tokens += num_prompt_tokens
        self.num_generation_tokens += num_generation_tokens

    def window_expired(self, now: float) -> bool:
        """Whether the throughput window is due to close.

        Split out so `heartbeat_throughput` can answer it without touching the
        queues or the KV pool. The busy loops call that on *every* pass, busy
        or idle, so this runs at loop frequency and says no all but once per
        interval; it takes the `now` the loop already read.
        """
        return (
            self.throughput_enabled
            and now - self._throughput_last_log_time >= self.throughput_log_interval_s
        )

    def maybe_log_throughput(
        self,
        num_running_reqs: int,
        num_waiting_reqs: int,
        kv_usage: float | None,
    ) -> None:
        if not self.throughput_enabled:
            return
        now = time.monotonic()
        elapsed = now - self._throughput_last_log_time
        # `elapsed <= 0` is unreachable while the interval holds the positive
        # value `__init__` validated, but the interval is a public attribute
        # and the division is two lines below; one compare keeps that division
        # safe whatever the attribute has been set to since.
        if elapsed < self.throughput_log_interval_s or elapsed <= 0:
            return
        if (
            self.num_prompt_tokens == 0
            and self.num_generation_tokens == 0
            and num_running_reqs == 0
            and num_waiting_reqs == 0
        ):
            self._throughput_last_log_time = now
            return
        prompt_throughput = self.num_prompt_tokens / elapsed
        generation_throughput = self.num_generation_tokens / elapsed
        # The prefix-cache hit rate is owned by this same object now; pull it
        # from the cache section rather than have the caller thread it in.
        # Windowed, like every other field on this line, and None — rendered
        # `n/a` — when nothing has been measured. "0.0%" is a claim about
        # reuse, and a P/D decode engine never reaches `update_cache` at all.
        prefix_cache_hit_rate = (
            self.recent_cache_hit_rate if self.cache_enabled else None
        )
        logger.info(
            "%sEngine %03d: Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, Running: %d reqs, "
            "Waiting: %d reqs, GPU KV cache usage: %s, "
            "Prefix cache hit rate: %s",
            self.label,
            self.engine_index,
            prompt_throughput,
            generation_throughput,
            num_running_reqs,
            num_waiting_reqs,
            "n/a" if kv_usage is None else f"{kv_usage * 100:.1f}%",
            (
                "n/a"
                if prefix_cache_hit_rate is None
                else f"{prefix_cache_hit_rate * 100:.1f}%"
            ),
        )
        self._throughput_last_log_time = now
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0
