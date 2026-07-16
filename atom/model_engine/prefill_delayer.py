"""
PrefillDelayer — a cross-DP-rank prefill *coalescer* for ATOM.

Purpose
-------
Under DP-attention each rank schedules independently. Left alone, ranks fire
many prefill forwards that each carry only a handful of tokens (a short fresh
prompt, or the small tail chunk of a chunked prefill). Every prefill forward
has ~fixed cost (kernel launch, pad-to-shape, the lockstep MoE all-to-all), so a
forward carrying 500 of a 16384-token budget wastes ~97% of that forward.

The delayer's single job: **hold back prefill admission until the accumulated
prefill is worth a forward, then release** — Nagle's algorithm for prefill.
While it holds, decode keeps running (nothing is wasted); TTFT is bounded so a
held request never starves.

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

  if n_prefillable == 0:                          FIRE   # nothing to do (vacuous)
  # -- must-fire bounds (release even if unaligned / underfilled) --
  if G_running_dec == 0:                          FIRE   # no decode to hide the wait behind
  if any_kv_high or any_kv_low:                   FIRE   # KV pressure / starvation
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
Single ``all_reduce(SUM)`` over a 6-int64 cpu buffer (gloo-safe). Booleans are
encoded as 0/1 and read back as ``sum > 0`` (logical OR). All timing is
tick-based (``hold_ticks``), which is deterministic across ranks — no per-rank
wall-clock, so ranks never diverge on a timeout boundary. Fail-open on a
collective error (a peer dying should not crash schedule()); the guarded
DP-state barrier drives shutdown.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ATOM_PREFILL_DELAYER_DEBUG", "0") == "1"


class PrefillDelayer:
    __slots__ = (
        "dp_size",
        "cpu_group",
        "max_num_batched_tokens",
        "target_fill",
        "ttft_max_ticks",
        "partial_max_ticks",
        "stall_ticks",
        "kv_high_watermark",
        "token_usage_low_watermark",
        "max_queue_ms",
        # buffer / episode state
        "_reduce_buf",
        "_hold_ticks",
        "_stall_count",
        "_prev_pending",
        "_first",
        # stats
        "_stat_fire_fill",
        "_stat_fire_stall",
        "_stat_fire_ttft",
        "_stat_fire_kv",
        "_stat_fire_partial",
        "_stat_fire_nodecode",
        "_stat_fire_queue_ms",
        "_stat_fire_vacuous",
        "_stat_hold",
        "_stat_log_every",
    )

    def __init__(
        self,
        dp_size: int,
        cpu_group,
        max_num_batched_tokens: int,
        target_fill: float = 0.7,
        ttft_max_ticks: int = 30,
        partial_max_ticks: int = 8,
        stall_ticks: int = 3,
        kv_high_watermark: float = 0.9,
        token_usage_low_watermark: Optional[float] = None,
        max_queue_ms: Optional[float] = None,
    ):
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

        # 7-slot SUM-reduce buffer (gloo-safe). Encoding:
        #   slot 0 = prefillable        (SUM → n_prefillable)
        #   slot 1 = pending_tokens     (SUM → G_pending; fresh + partial remain)
        #   slot 2 = running_decode     (SUM → G_running_dec)
        #   slot 3 = kv_high flag       (SUM>0 → any prefillable rank KV-high)
        #   slot 4 = kv_low flag        (SUM>0 → any prefillable rank KV-low)
        #   slot 5 = has_partial flag   (SUM>0 → any rank mid-chunked-prefill)
        #   slot 6 = queue_hot flag     (SUM>0 → any rank's oldest waiting prefill
        #                                aged past max_queue_ms; TTFT SLA guard)
        self._reduce_buf = torch.zeros(7, dtype=torch.int64, device="cpu")

        # Episode state. All ticks decide FIRE/HOLD in lockstep, so these evolve
        # identically on every rank (deterministic).
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        # First call fires immediately to seed the initial decode batch build-up
        # (mirrors SGL PR #19836's skip_first).
        self._first = True

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
            f"max_queue_ms={max_queue_ms}"
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

    def should_allow_prefill(
        self,
        prefillable: bool,
        pending_tokens: int,
        running_decode_batch: int = 0,
        kv_usage: float = 0.0,
        has_partial: bool = False,
        oldest_waiting_age_ms: float = 0.0,
    ) -> bool:
        """Return True iff this rank may admit new prefills this tick (FIRE).

        MUST be called every tick on every DP rank (it runs a cross-DP
        all_reduce) so ranks stay in lockstep.

        Args:
            prefillable: this rank has admittable prefill work (fresh head that
                can allocate, or a resumable partial). Only prefillable ranks
                count toward the fill target and the alignment gate.
            pending_tokens: this rank's accumulated prefill tokens — fresh
                waiting new-tokens PLUS the remaining tokens of resumable
                partials — already capped at max_num_batched_tokens by the
                caller. The coalescer's fill signal.
            running_decode_batch: decode seqs running on this rank (NOT counting
                mid-chunked-prefill seqs). If no rank has decode, holding wastes
                GPU → fire.
            kv_usage: fraction of KV cache blocks in use ∈ [0, 1]. Drives the
                KV-high (can't accumulate more) and KV-low (GPU starving) bounds.
            has_partial: this rank has a mid-chunked-prefill seq in flight. Its
                remaining tokens are in pending_tokens; a partial only forces
                release once held for partial_max_ticks (it holds KV).
            oldest_waiting_age_ms: age (ms since arrival) of this rank's oldest
                schedulable waiting prefill. If it reaches max_queue_ms, this
                rank flags the TTFT SLA guard and all ranks release. 0 / no
                waiting prefill / max_queue_ms=None → guard inactive.
        """
        # First call fires unconditionally (one-time warmup seed); not counted in
        # the per-exit stats so `fire_vacuous` stays exactly "n_prefillable == 0".
        if self._first:
            self._first = False
            self._reset()
            return True

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
        try:
            torch.distributed.all_reduce(
                self._reduce_buf,
                op=torch.distributed.ReduceOp.SUM,
                group=self.cpu_group,
            )
        except RuntimeError:
            logger.warning(
                "PrefillDelayer all_reduce failed (peer down?); admitting prefill "
                "and deferring to the DP-state shutdown barrier."
            )
            self._reset()
            return True

        # One host<-device readback for all 7 slots (a single .tolist() beats
        # seven .item() boundary crossings on this per-tick hot path).
        (
            n_prefillable,
            g_pending,
            g_running_dec,
            kv_high_n,
            kv_low_n,
            partial_n,
            queue_hot_n,
        ) = self._reduce_buf.tolist()
        any_kv_high = kv_high_n > 0
        any_kv_low = kv_low_n > 0
        any_partial = partial_n > 0
        any_queue_hot = queue_hot_n > 0

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
        # TTFT SLA guard: a real request has queued (since arrival) past the
        # threshold — release now regardless of fill/alignment. This is the
        # end-to-end wait (includes backlog + coalescer holds), unlike the
        # tick-based ttft bound below which only caps a single hold episode.
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
        if total % self._stat_log_every == 0:
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
                f"hold={self._stat_hold} "
                f"(hold_rate={self._stat_hold / total:.2%})"
            )
