"""
PrefillDelayer — local TP or cross-DP prefill coalescing for ATOM.

Purpose
-------
Under DP-attention each rank schedules independently. Left alone, ranks fire
many prefill forwards that each carry only a handful of tokens (a short fresh
prompt, or the small tail chunk of a chunked prefill). Every prefill forward
has ~fixed cost (kernel launch, pad-to-shape, the lockstep MoE all-to-all), so a
forward carrying 500 of a 16384-token budget wastes ~97% of that forward.

The delayer has two related jobs: **hold back prefill admission until the
accumulated prefill is worth a forward, then release** — Nagle's algorithm for
prefill — and optionally protect a fixed number of decode passes after every
prefill. While it holds, decode keeps running. Coalescing hold episodes are
bounded; the decode interval, backlog, and execution add to observed TTFT.

Single-rank / TP-only mode
--------------------------
With ``is_local`` (``dp_size=1`` and ``cpu_group=None``) the delayer runs the same coalescer
locally and skips the cross-rank ``all_reduce`` — a single scheduler drives all
TP workers, so there is no cross-rank phase to align. Only the batching value
remains (fill / stall / ttft / kv bounds decide FIRE/HOLD from this rank's own
counts); the ``n_prefillable < dp_size`` alignment gate is vacuous.
EngineCore opts TP/DCP into this mode when DP=1, PP=1, the master switch is
on, and ``ATOM_PREFILL_DECODE_INTERVAL > 0``. During local decode protection,
``protects_decode`` lets the scheduler skip cache probes and queue-age scans.
The interval precedes all coalescing bounds, including ``max_queue_ms``; those
bounds do not impose a hard end-to-end TTFT limit.

It is NOT about mixing prefill+decode in one forward. It DOES preserve cross-DP
phase alignment: it only releases when every rank is prefill-ready (so all ranks
enter prefill together and the MoE collective stays aligned), except when a
must-fire bound forces release regardless.

Decision (evaluated every tick, on every rank, in lockstep)
-----------------------------------------------------------
Each rank reports local state; a single cpu ``all_reduce(SUM)`` reduces it; then
every rank computes the SAME FIRE/HOLD from the reduced values:

  n_prefillable   = #ranks with admittable prefill (SUM of a 0/1 flag)
  G_pending       = total pending prefill tokens across ranks (fresh + partial
                    remaining, each rank capped at the token budget)
  G_running_dec   = total decode seqs across ranks
  any_kv_high/low = any prefillable rank at/above / below a KV watermark
  any_partial     = any rank mid-chunked-prefill
  any_prefill_ran = any rank completed a prefill forward on the previous tick

  if any_prefill_ran:                              arm decode interval
  if post-prefill decode interval is active and decode exists: HOLD
  if n_prefillable == 0:                          FIRE   # nothing to do (vacuous)
  # -- must-fire bounds (release even if unaligned / underfilled) --
  if G_running_dec == 0:                          FIRE   # no decode to hide the wait behind
  if any_kv_high or any_kv_low:                   FIRE   # KV pressure / starvation
  if any_queue_hot:                             FIRE   # queue age bound
  if hold_ticks >= ttft_max_ticks:                FIRE   # TTFT bound
  if any_partial and hold_ticks >= partial_max_ticks: FIRE  # partial holds KV — tight bound
  # -- alignment gate: never fire while some rank lacks prefill (anti-skew) --
  if n_prefillable < dp_size:                     HOLD
  # -- goal: fire once the aggregate fills a worthwhile forward --
  fill = G_pending / (n_prefillable * budget)
  if fill >= target_fill:                         FIRE
  # -- adaptive give-up: queue stopped growing, waiting longer is futile --
  if G_pending <= prev_G_pending:  stall += 1  else  stall = 0
  if stall >= stall_ticks:                        FIRE
  HOLD

Why SUM (not MAX)
-----------------
A MAX ("fire as soon as the busiest rank is ready") drags quiet ranks into
firing tiny prefills. A SUM/aggregate-fill target holds until the forward is
collectively worthwhile, so quiet ranks keep accumulating during the hold and
fire larger when release finally happens. Under-balanced load a quiet rank still
fires smaller — that is a request-routing problem, out of scope here.

Cross-DP comms
--------------
Single ``all_reduce(SUM)`` over an 8-int64 cpu buffer (gloo-safe). Booleans are
encoded as 0/1 and read back as ``sum > 0`` (logical OR). All timing is
tick-based (``hold_ticks``), which is deterministic across ranks — no per-rank
wall-clock, so ranks never diverge on a timeout boundary. Fail-open on a
collective error (a peer dying should not crash schedule()); the guarded
DP-state barrier drives shutdown.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ATOM_PREFILL_DELAYER_DEBUG", "0") == "1"
_DEBUG_GROUP = os.environ.get("ATOM_PREFILL_GROUP_DELAYER_DEBUG", "0") == "1"


class PrefillDelayer:
    __slots__ = (
        "_cost_slots",
        "_decode_interval_remaining",
        "_first",
        "_hold_ticks",
        "_prefill_executed_since_last_decision",
        "_prev_pending",
        # buffer / episode state
        "_rank",
        "_reduce_buf",
        "_stall_count",
        # stats
        "_stat_fire_fill",
        "_stat_fire_kv",
        "_stat_fire_nodecode",
        "_stat_fire_partial",
        "_stat_fire_queue_ms",
        "_stat_fire_stall",
        "_stat_fire_ttft",
        "_stat_fire_vacuous",
        "_stat_hold",
        "_stat_hold_decode_interval",
        "_stat_log_every",
        "cpu_group",
        "dp_size",
        "kv_high_watermark",
        "max_num_batched_tokens",
        "max_queue_ms",
        "partial_max_ticks",
        "prefill_decode_interval",
        "rank_prefill_costs",
        "stall_ticks",
        "target_fill",
        "token_usage_low_watermark",
        "ttft_max_ticks",
    )

    def __init__(
        self,
        dp_size: int,
        cpu_group,
        max_num_batched_tokens: int,
        target_fill: float = 0.9,
        ttft_max_ticks: int = 200,
        partial_max_ticks: int = 100,
        stall_ticks: int = 10,
        kv_high_watermark: float = 0.9,
        token_usage_low_watermark: float | None = None,
        max_queue_ms: float | None = None,
        prefill_decode_interval: int = 0,
        exchange_prefill_costs: bool = False,
    ):
        if dp_size > 1 and cpu_group is None:
            raise ValueError("Cross-DP prefill coalescing requires a CPU group")
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.max_num_batched_tokens = max_num_batched_tokens

        # target_fill is the fraction of a full prefill batch that must be
        # accumulated (across prefillable ranks) before we release. Must be in
        # (0, 1]: <= 0 would fire every tick (no coalescing); > 1 is unreachable
        # (fill is capped at 1) and would degrade to stall/timeout-only. Clamp.
        if target_fill <= 0.0:
            logger.warning(
                f"target_fill={target_fill} <= 0 disables coalescing "
                "(fires every tick); clamping to a small positive 0.05."
            )
            target_fill = 0.05
        elif target_fill > 1.0:
            logger.warning(
                f"target_fill={target_fill} > 1.0 is unreachable (fill is capped "
                "at 1.0); clamping to 1.0."
            )
            target_fill = 1.0
        self.target_fill = target_fill

        # Tick bounds must be >= 1. 0 would make the corresponding `>=` fire on
        # the first tick, defeating the bound.
        self.ttft_max_ticks = self._clamp_ticks("ttft_max_ticks", ttft_max_ticks)
        self.partial_max_ticks = self._clamp_ticks(
            "partial_max_ticks", partial_max_ticks
        )
        self.stall_ticks = self._clamp_ticks("stall_ticks", stall_ticks)
        self.kv_high_watermark = kv_high_watermark
        self.token_usage_low_watermark = token_usage_low_watermark
        # TTFT SLA guard: if any rank's oldest schedulable waiting-prefill has
        # queued (since arrival) >= this many ms, force release regardless of the
        # fill target. None disables it. Wall-clock, but lockstep-safe: each rank
        # compares its OWN requests' ages to the threshold locally (disjoint
        # request sets across ranks), and only the OR of those per-rank booleans
        # crosses the collective — so all ranks act on the identical result the
        # same tick. (Contrast the removed per-rank hold-clock E1, which compared
        # each rank's own clock to a threshold for a GLOBAL decision → skew.)
        self.max_queue_ms = max_queue_ms
        if prefill_decode_interval < 0:
            raise ValueError(
                "prefill_decode_interval must be non-negative; "
                f"got {prefill_decode_interval}"
            )
        self.prefill_decode_interval = prefill_decode_interval

        # 8-slot SUM-reduce buffer (gloo-safe). Encoding:
        #   slot 0 = prefillable        (SUM → n_prefillable)
        #   slot 1 = pending_tokens     (SUM → G_pending; fresh + partial remain)
        #   slot 2 = running_decode     (SUM → G_running_dec)
        #   slot 3 = kv_high flag       (SUM>0 → any prefillable rank KV-high)
        #   slot 4 = kv_low flag        (SUM>0 → any prefillable rank KV-low)
        #   slot 5 = has_partial flag   (SUM>0 → any rank mid-chunked-prefill)
        #   slot 6 = queue_hot flag     (SUM>0 → any rank's oldest waiting prefill
        #                                aged past max_queue_ms; TTFT SLA guard)
        #   slot 7 = prefill_ran flag   (SUM>0 → a rank completed a prefill
        #                                forward on the previous scheduler tick)
        # With exchange_prefill_costs (DP prefill cost balancing), dp_size more:
        #   slot 8 + r = rank r's prefill chunk cost (one-hot, SUM → every
        #                rank's cost), read back as rank_prefill_costs
        self._cost_slots = dp_size if exchange_prefill_costs else 0
        # The DP group is stateless (no default process group): ask it directly.
        self._rank = (
            cpu_group.rank() if self._cost_slots and cpu_group is not None else 0
        )
        self._reduce_buf = torch.zeros(
            8 + self._cost_slots, dtype=torch.int64, device="cpu"
        )
        self.rank_prefill_costs: list[float] = []

        # Episode state. All ticks decide FIRE/HOLD in lockstep, so these evolve
        # identically on every rank (deterministic).
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        # First call fires immediately to seed the initial decode batch build-up
        # (mirrors SGL PR #19836's skip_first).
        self._first = True
        self._decode_interval_remaining = 0
        self._prefill_executed_since_last_decision = False

        # Per-exit fire counters + hold counter for periodic logging. Each exit
        # is counted separately so the log shows WHICH reason released prefill —
        # a high fire_stall/fire_ttft vs fire_fill means the queue rarely reaches
        # target and the coalescer is mostly giving up rather than batching.
        self._stat_fire_fill = 0
        self._stat_fire_stall = 0
        self._stat_fire_ttft = 0
        self._stat_fire_kv = 0
        self._stat_fire_partial = 0
        self._stat_fire_nodecode = 0
        self._stat_fire_queue_ms = 0
        self._stat_fire_vacuous = 0
        self._stat_hold = 0
        self._stat_hold_decode_interval = 0
        self._stat_log_every = int(
            os.environ.get("ATOM_PREFILL_DELAYER_LOG_EVERY", "1000")
        )

        logger.info(
            f"PrefillDelayer initialized: dp_size={dp_size} "
            f"max_num_batched_tokens={max_num_batched_tokens} "
            f"target_fill={self.target_fill} "
            f"ttft_max_ticks={self.ttft_max_ticks} "
            f"partial_max_ticks={self.partial_max_ticks} "
            f"stall_ticks={self.stall_ticks} "
            f"kv_high_watermark={kv_high_watermark} "
            f"token_usage_low_watermark={token_usage_low_watermark} "
            f"max_queue_ms={max_queue_ms} "
            f"prefill_decode_interval={prefill_decode_interval}"
        )

    @staticmethod
    def _clamp_ticks(name: str, value: int) -> int:
        if value < 1:
            logger.warning(
                f"{name}={value} < 1 would fire on the first tick (bound "
                "disabled); clamping to 1."
            )
            return 1
        return value

    @property
    def is_local(self) -> bool:
        return self.dp_size == 1 and self.cpu_group is None

    def protects_decode(self, running_decode_batch: int) -> bool:
        """Whether the local decision can skip cache probes during protection."""
        return (
            self.is_local
            and not self._first
            and running_decode_batch > 0
            and (
                self._decode_interval_remaining > 0
                or (
                    self._prefill_executed_since_last_decision
                    and self.prefill_decode_interval > 0
                )
            )
        )

    def should_allow_prefill(
        self,
        prefillable: bool,
        pending_tokens: int,
        running_decode_batch: int = 0,
        kv_usage: float = 0.0,
        has_partial: bool = False,
        oldest_waiting_age_ms: float = 0.0,
        prefill_cost: float = 0.0,
    ) -> bool:
        """Return True iff this rank may admit new prefills this tick (FIRE).

        MUST be called every tick on every DP rank (it runs a cross-DP
        all_reduce) so ranks stay in lockstep.

        Args:
            prefillable: this rank has admittable prefill work (fresh head that
                can allocate, or a resumable partial). Only prefillable ranks
                count toward the fill target and the alignment gate. During
                local decode protection, the caller uses work existence alone:
                fit cannot change the hard interval decision.
            pending_tokens: this rank's accumulated prefill tokens — fresh
                waiting new-tokens PLUS the remaining tokens of resumable
                partials — already capped at max_num_batched_tokens by the
                caller. The local scheduler bounds its scan and uses chunk
                limits, so this signal can omit work beyond that scan.
            running_decode_batch: decode seqs running on this rank (NOT counting
                mid-chunked-prefill seqs). If no rank has decode, holding wastes
                GPU → fire.
            kv_usage: fraction of KV cache blocks in use ∈ [0, 1]. Drives the
                KV-high (can't accumulate more) and KV-low (GPU starving) bounds.
            has_partial: this rank has a mid-chunked-prefill seq in flight. Its
                remaining tokens are in pending_tokens; a partial only forces
                release once held for partial_max_ticks (it holds KV), counted
                after the decode-protection interval.
            oldest_waiting_age_ms: age (ms since arrival) of this rank's oldest
                schedulable waiting prefill. If it reaches max_queue_ms, this
                rank flags the age guard; all ranks release after decode
                protection. No waiting prefill / max_queue_ms=None → inactive.
            prefill_cost: this rank's estimated prefill chunk cost (0 if none).
                With exchange_prefill_costs every rank's value is in
                ``rank_prefill_costs`` after the call.
        """
        # KV + queue-age bounds are gated on this rank actually having prefill to
        # push (firing when this rank can't admit anything would be a no-op).
        low = self.token_usage_low_watermark
        kv_high = prefillable and kv_usage >= self.kv_high_watermark
        kv_low = prefillable and low is not None and kv_usage < low
        queue_hot = (
            prefillable
            and self.max_queue_ms is not None
            and oldest_waiting_age_ms >= self.max_queue_ms
        )

        self._reduce_buf[0] = 1 if prefillable else 0
        self._reduce_buf[1] = int(pending_tokens)
        self._reduce_buf[2] = int(running_decode_batch)
        self._reduce_buf[3] = 1 if kv_high else 0
        self._reduce_buf[4] = 1 if kv_low else 0
        self._reduce_buf[5] = 1 if has_partial else 0
        self._reduce_buf[6] = 1 if queue_hot else 0
        self._reduce_buf[7] = 1 if self._prefill_executed_since_last_decision else 0
        if self._cost_slots:
            self._reduce_buf[8:] = 0
            self._reduce_buf[8 + self._rank] = int(prefill_cost)
        if self.cpu_group is not None:
            try:
                torch.distributed.all_reduce(
                    self._reduce_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.cpu_group,
                )
            except RuntimeError:
                logger.warning(
                    "PrefillDelayer all_reduce failed (peer down?); admitting "
                    "prefill and deferring to the DP-state shutdown barrier."
                )
                self._reset()
                self.rank_prefill_costs = []
                return True

        # One readback for all slots (a single .tolist() beats per-slot .item()
        # boundary crossings on this per-tick hot path).
        reduced = self._reduce_buf.tolist()
        (
            n_prefillable,
            g_pending,
            g_running_dec,
            kv_high_n,
            kv_low_n,
            partial_n,
            queue_hot_n,
            prefill_ran_n,
        ) = reduced[:8]
        self.rank_prefill_costs = [float(c) for c in reduced[8:]]
        # This local execution report has now participated in the global
        # decision. A later successful prefill forward sets it again.
        self._prefill_executed_since_last_decision = False
        any_kv_high = kv_high_n > 0
        any_kv_low = kv_low_n > 0
        any_partial = partial_n > 0
        any_queue_hot = queue_hot_n > 0

        # A prefill that actually ran on the previous tick arms the same
        # countdown on every rank. Delayer FIRE is only permission to try:
        # allocation, remote-KV handling, or queue changes may still produce a
        # decode/empty batch, which must not start a decode-protection window.
        if prefill_ran_n > 0:
            self._decode_interval_remaining = self.prefill_decode_interval

        # First call fires unconditionally (one-time warmup seed).
        if self._first:
            self._first = False
            self._reset()
            return True

        # Hard post-prefill decode protection. The countdown is identical on
        # all ranks because it is armed from the globally reduced execution
        # signal and schedule() calls this method in lockstep. Do not advance the
        # coalescer's own hold/TTFT episode: SGLang applies the interval before
        # invoking PrefillDelayer, so its delay budget starts afterwards.
        if self._decode_interval_remaining > 0:
            self._decode_interval_remaining -= 1
            if n_prefillable > 0 and g_running_dec > 0:
                self._stat_hold += 1
                self._stat_hold_decode_interval += 1
                self._maybe_log()
                return False

        # Nothing to prefill anywhere → allow (vacuous), reset the episode.
        if n_prefillable == 0:
            self._reset()
            self._stat_fire_vacuous += 1
            self._maybe_log()
            return True

        # ---- must-fire bounds (release even if unaligned / underfilled) ----
        if g_running_dec == 0:
            return self._fire("nodecode", g_pending)
        if any_kv_high or any_kv_low:
            return self._fire("kv", g_pending)
        # Queue age includes backlog and prior holds. This release applies
        # after decode protection; it is not an end-to-end TTFT guarantee.
        if any_queue_hot:
            return self._fire("queue_ms", g_pending)
        if self._hold_ticks >= self.ttft_max_ticks:
            return self._fire("ttft", g_pending)
        if any_partial and self._hold_ticks >= self.partial_max_ticks:
            return self._fire("partial", g_pending)

        # ---- alignment gate: never fire while a rank lacks prefill (anti-skew).
        # Hold to let stragglers catch up; the ttft bound above bounds the wait.
        if n_prefillable < self.dp_size:
            return self._hold(g_pending)

        # ---- goal: fire once the aggregate fills a worthwhile forward ----
        budget = n_prefillable * self.max_num_batched_tokens
        fill = g_pending / budget if budget > 0 else 1.0
        if fill >= self.target_fill:
            return self._fire("fill", g_pending)

        # ---- adaptive give-up: queue stopped growing → waiting is futile ----
        if g_pending <= self._prev_pending:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._prev_pending = g_pending
        if self._stall_count >= self.stall_ticks:
            return self._fire("stall", g_pending)

        return self._hold(g_pending)

    def _fire(self, reason: str, g_pending: int) -> bool:
        # reason ∈ {fill, stall, ttft, kv, partial, nodecode, vacuous}; each maps
        # to a dedicated _stat_fire_<reason> slot so the log shows what released.
        attr = f"_stat_fire_{reason}"
        setattr(self, attr, getattr(self, attr) + 1)
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] FIRE ({reason}): g_pending={g_pending} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count}"
            )
        self._reset()
        self._maybe_log()
        return True

    def notify_prefill_executed(self) -> None:
        """Report a completed local prefill forward.

        The next lockstep decision folds this flag into the existing SUM
        reduction, so every DP rank arms the interval together without adding
        a second collective to the scheduler hot path.
        """
        self._prefill_executed_since_last_decision = True

    def _hold(self, g_pending: int) -> bool:
        self._hold_ticks += 1
        self._stat_hold += 1
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] HOLD: g_pending={g_pending} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count} "
                f"target_fill={self.target_fill}"
            )
        self._maybe_log()
        return False

    def _reset(self) -> None:
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1

    def _maybe_log(self) -> None:
        # Cheap guard first — skip the counter sum entirely when logging is
        # disabled (this runs on every FIRE/HOLD tick).
        if self._stat_log_every <= 0:
            return
        total = (
            self._stat_fire_fill
            + self._stat_fire_stall
            + self._stat_fire_ttft
            + self._stat_fire_kv
            + self._stat_fire_partial
            + self._stat_fire_nodecode
            + self._stat_fire_queue_ms
            + self._stat_fire_vacuous
            + self._stat_hold
        )
        if total == 0:
            return
        # Also log the FIRST real decision, not only every Nth. The counter is
        # per PROCESS, and with N prefill ranks each one sees roughly
        # total_prefills/N decisions — a run with a few hundred prefills spread
        # over 8 ranks never reaches the default 1000, so the delayer looks dead
        # when it is simply quiet. One line up front proves it is live.
        if total == 1 or total % self._stat_log_every == 0:
            logger.info(
                f"[PrefillDelayer stats] total={total} "
                f"fire_fill={self._stat_fire_fill} "
                f"fire_stall={self._stat_fire_stall} "
                f"fire_ttft={self._stat_fire_ttft} "
                f"fire_kv={self._stat_fire_kv} "
                f"fire_partial={self._stat_fire_partial} "
                f"fire_nodecode={self._stat_fire_nodecode} "
                f"fire_queue_ms={self._stat_fire_queue_ms} "
                f"fire_vacuous={self._stat_fire_vacuous} "
                f"hold_decode_interval={self._stat_hold_decode_interval} "
                f"hold={self._stat_hold} "
                f"(hold_rate={self._stat_hold / total:.2%})"
            )


class PrefillGroupDelayer:
    """Per-rank MIN barrier on PrefillScheduler itself: fire together, or not.

    A different problem from `PrefillDelayer` above, which gates admission --
    when a NEW request becomes visible to prefill at all
    (`DecodeScheduler.allocate_waiting`). Once prefill holds a landed
    `BlockAssignment`, nothing delays running it; `PrefillScheduler._schedule()`
    fires on the very next tick. This class gates THAT: whether prefill runs a
    forward this tick for sequences it already has, so that every rank's
    forward carries at least `min_tokens` instead of firing the moment any
    single sequence lands.

    And the release rule is a per-rank MIN, not `PrefillDelayer`'s cross-rank
    SUM/average `fill`. That average can score a lopsided case (one rank at
    the floor, the rest at zero) the same as a balanced one (every rank at a
    fraction of it) -- no `target_fill` expresses "every rank individually
    clears the floor." Hold every rank until each one does, release together,
    then require the floor to be cleared again -- which happens for free: once
    a release drains `ready`, next tick's pending count starts back near zero
    on every rank, so reaccumulating to `min_tokens` on all of them is exactly
    "don't fire again until the condition is met again," no separate latch.

    No KV watermarks (prefill has no `BlockManager`, so no KV-usage concept to
    watch), no decode-protection interval (no decode batches here to hide a
    wait behind), no partial-chunk exemption (a rank mid-chunked-prefill waits
    like any other -- its remaining tokens simply count toward its own
    pending total). Those three account for most of `PrefillDelayer`'s size;
    without them this is a 3-slot reduction instead of an 8-slot one.

    Same proven primitives: one small CPU `all_reduce(SUM)`, deterministic
    tick counters (no wall-clock, so ranks cannot diverge on a timeout
    boundary), fail-open on a collective error.
    """

    __slots__ = (
        "_first",
        "_hold_ticks",
        "_prev_pending",
        "_reduce_buf",
        "_stall_count",
        "_stat_fire_min_tokens",
        "_stat_fire_stall",
        "_stat_fire_ttft",
        "_stat_fire_vacuous",
        "_stat_hold",
        "_stat_log_every",
        "cpu_group",
        "dp_size",
        "min_tokens",
        "stall_ticks",
        "ttft_max_ticks",
    )

    def __init__(
        self,
        dp_size: int,
        cpu_group,
        min_tokens: int,
        ttft_max_ticks: int = 200,
        stall_ticks: int = 20,
    ):
        if dp_size > 1 and cpu_group is None:
            raise ValueError("Cross-DP prefill batching requires a CPU group")
        if min_tokens <= 0:
            raise ValueError(f"min_tokens must be positive; got {min_tokens}")
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.min_tokens = min_tokens
        self.ttft_max_ticks = self._clamp_ticks("ttft_max_ticks", ttft_max_ticks)
        self.stall_ticks = self._clamp_ticks("stall_ticks", stall_ticks)

        # 3-slot SUM-reduce buffer (gloo-safe):
        #   slot 0 = any_ready flag   (SUM → n_any,   ranks with ANY ready work)
        #   slot 1 = cleared flag     (SUM → n_ready, ranks AT/OVER min_tokens)
        #   slot 2 = pending_tokens   (SUM → g_pending, for the stall bound)
        self._reduce_buf = torch.zeros(3, dtype=torch.int64, device="cpu")

        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        # First call fires immediately, same reason PrefillDelayer's does: seed
        # the initial batch build-up rather than holding before anything exists
        # to hold for.
        self._first = True

        self._stat_fire_min_tokens = 0
        self._stat_fire_stall = 0
        self._stat_fire_ttft = 0
        self._stat_fire_vacuous = 0
        self._stat_hold = 0
        self._stat_log_every = int(
            os.environ.get("ATOM_PREFILL_GROUP_DELAYER_LOG_EVERY", "1000")
        )

        logger.info(
            f"PrefillGroupDelayer initialized: dp_size={dp_size} "
            f"min_tokens={min_tokens} ttft_max_ticks={self.ttft_max_ticks} "
            f"stall_ticks={self.stall_ticks}"
        )

    @staticmethod
    def _clamp_ticks(name: str, value: int) -> int:
        if value < 1:
            logger.warning(
                f"{name}={value} < 1 would fire on the first tick (bound "
                "disabled); clamping to 1."
            )
            return 1
        return value

    @property
    def is_local(self) -> bool:
        return self.dp_size == 1 and self.cpu_group is None

    def should_allow_prefill(self, any_ready: bool, pending_tokens: int) -> bool:
        """Return True iff every rank's ready-pending has cleared `min_tokens`.

        MUST be called every tick this rank reaches `PrefillEngineCore.
        _process_engine_step` (every tick, whether or not THIS rank has
        anything ready) -- it runs a cross-DP all_reduce, and `busy_loop`'s own
        `_sync_dp_state` guarantees that step is reached on every prefill rank
        or none, every tick. A rank that reached it but skipped this call would
        mismatch the collective and hang its peers.

        Args:
            any_ready: this rank has at least one sequence whose
                `BlockAssignment` has landed and could run this tick.
            pending_tokens: the sum of remaining new-tokens across this rank's
                ready sequences. Uncapped -- unlike `PrefillDelayer`'s averaged
                fill, a per-rank floor has no reason to clip a rank already
                carrying more than it.
        """
        cleared = any_ready and pending_tokens >= self.min_tokens
        self._reduce_buf[0] = 1 if any_ready else 0
        self._reduce_buf[1] = 1 if cleared else 0
        self._reduce_buf[2] = int(pending_tokens)
        if self.cpu_group is not None:
            try:
                torch.distributed.all_reduce(
                    self._reduce_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.cpu_group,
                )
            except RuntimeError:
                logger.warning(
                    "PrefillGroupDelayer all_reduce failed (peer down?); "
                    "firing and deferring to the DP-state shutdown barrier."
                )
                self._reset()
                return True

        n_any, n_ready, g_pending = self._reduce_buf.tolist()

        if self._first:
            self._first = False
            self._reset()
            return True

        if n_any == 0:
            return self._fire("vacuous", g_pending)

        # ---- must-fire bounds (release even if some rank is still short) ----
        if self._hold_ticks >= self.ttft_max_ticks:
            return self._fire("ttft", g_pending)
        if g_pending <= self._prev_pending:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._prev_pending = g_pending
        if self._stall_count >= self.stall_ticks:
            return self._fire("stall", g_pending)

        # ---- alignment gate: literally every one of dp_size ranks ----
        # Same anti-skew idiom as PrefillDelayer's `n_prefillable < dp_size`:
        # hold while any rank has nothing ready yet, so a straggler about to
        # receive its BlockAssignment this very tick is waited for rather than
        # left out of an uneven release. This is NOT a permanent deadlock risk
        # for a rank that never gets routed anything -- the must-fire bounds
        # above are checked first and are blind to WHY hold_ticks is climbing,
        # so they rescue this case exactly as they rescue a rank that has work
        # but never clears min_tokens.
        if n_any < self.dp_size:
            return self._hold(g_pending, n_any, n_ready)
        if n_ready < n_any:
            return self._hold(g_pending, n_any, n_ready)

        return self._fire("min_tokens", g_pending, n_any, n_ready)

    def _fire(
        self, reason: str, g_pending: int, n_any: int = 0, n_ready: int = 0
    ) -> bool:
        # reason ∈ {min_tokens, ttft, stall, vacuous}; each maps to a dedicated
        # _stat_fire_<reason> slot so the log shows what released.
        attr = f"_stat_fire_{reason}"
        setattr(self, attr, getattr(self, attr) + 1)
        if _DEBUG_GROUP:
            logger.info(
                f"[PrefillGroupDelayer] FIRE ({reason}): g_pending={g_pending} "
                f"n_ready={n_ready}/{n_any} (dp_size={self.dp_size}) "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count} "
                f"min_tokens={self.min_tokens}"
            )
        self._reset()
        self._maybe_log()
        return True

    def _hold(self, g_pending: int, n_any: int = 0, n_ready: int = 0) -> bool:
        self._hold_ticks += 1
        self._stat_hold += 1
        if _DEBUG_GROUP:
            logger.info(
                f"[PrefillGroupDelayer] HOLD: g_pending={g_pending} "
                f"n_ready={n_ready}/{n_any} (dp_size={self.dp_size}) "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count} "
                f"min_tokens={self.min_tokens}"
            )
        self._maybe_log()
        return False

    def _reset(self) -> None:
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1

    def _maybe_log(self) -> None:
        if self._stat_log_every <= 0:
            return
        total = (
            self._stat_fire_min_tokens
            + self._stat_fire_stall
            + self._stat_fire_ttft
            + self._stat_fire_vacuous
            + self._stat_hold
        )
        if total == 0:
            return
        if total == 1 or total % self._stat_log_every == 0:
            logger.info(
                f"[PrefillGroupDelayer stats] total={total} "
                f"fire_min_tokens={self._stat_fire_min_tokens} "
                f"fire_stall={self._stat_fire_stall} "
                f"fire_ttft={self._stat_fire_ttft} "
                f"fire_vacuous={self._stat_fire_vacuous} "
                f"hold={self._stat_hold} "
                f"(hold_rate={self._stat_hold / total:.2%})"
            )
