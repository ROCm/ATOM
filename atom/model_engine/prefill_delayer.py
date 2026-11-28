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
prefill-only forward. While it holds, decode keeps running. Coalescing hold episodes are
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
enter prefill together and the MoE collective stays aligned) or blocked (it has
waiting work whose head cannot be admitted, which no wait changes; it runs a
dummy prefill beside its peers and adds no pending tokens), except when a
must-fire bound forces release regardless, the group's work stops changing
(stall), or the cost gate measures that waiting longer no longer pays
off. Pending-token probes are capped per rank, so a flat pending sum alone is
not "no more work": the stall count also restarts when a rank becomes (or stops
being) prefillable or a request arrives anywhere in the group. Nor does stall
release an unaligned group while every rank without prefill will finish a
decode, and so free a slot for its next request, within the TTFT bound: under a
closed loop that rank aligns then, however quiet the group is meanwhile.

Decision (evaluated every tick, on every rank, in lockstep)
-----------------------------------------------------------
Each rank reports local state; a single cpu ``all_reduce(SUM)`` reduces it; then
every rank computes the SAME FIRE/HOLD from the reduced values:

  n_prefillable   = #ranks with admittable prefill (SUM of a 0/1 flag)
  n_blocked       = #ranks with waiting work whose head cannot be admitted
                    (no KV for it, or its remote KV still loading)
  n_unreachable   = #ranks with neither, whose next decode completion (the
                    slot a new request would fill) lies past the hold's TTFT
                    bound, or that run no decode
  G_pending       = total pending prefill tokens across prefillable ranks
                    (fresh + partial remaining, each capped at the budget)
  G_running_dec   = total decode seqs across ranks
  G_arrived       = requests added across ranks since the previous tick
  any_kv_high     = any prefillable rank at/above the KV watermark
  any_kv_low      = any prefillable rank below the low KV watermark
  any_partial     = any rank mid-chunked-prefill
  any_prefill_ran = any rank completed a prefill forward on the previous tick
  any_prefill_only_ran = any such forward carried no decode rows

  if any_prefill_only_ran:                         arm decode interval
  if post-prefill decode interval is active and decode exists:
      stall = 0 if the group's work changed;      HOLD
  if n_prefillable == 0:                          FIRE   # nothing to do (vacuous)
  # -- must-fire bounds (release even if unaligned / underfilled) --
  if G_running_dec == 0:                          FIRE   # no decode to hide the wait behind
  if any_kv_high or any_kv_low:                   FIRE   # KV pressure / starvation
  if any_queue_hot:                             FIRE   # queue age bound
  if hold_ticks >= ttft_max_ticks:                FIRE   # TTFT bound
  if any_partial and hold_ticks >= partial_max_ticks: FIRE  # partial holds KV — tight bound
  # -- adaptive give-up: the group's work stopped changing --
  if G_pending > prev_G_pending or n_prefillable != prev_n_prefillable
     or G_arrived > 0:                            stall = 0
  else:                                           stall += 1
  # -- alignment gate: hold while some rank lacks prefill and could take
  #    some (anti-skew); a blocked rank waits on its own decode, not on us --
  if n_prefillable + n_blocked < dp_size:
      if n_unreachable > 0:  # waiting may not align the group in time
          if stall >= stall_ticks:                FIRE
          stall_us = G_running_dec * max(rank cost) * t_prefill  # unaligned
          if cost_gate and (arrival rate on ranks without prefill * stall_us
                            < G_waiting or hold_cost >= stall_us):
                                                  FIRE   # wait can't pay off
          hold_cost += G_waiting * t_decode
      HOLD
  # -- goal: fire once the aggregate fills a worthwhile forward --
  G_prefill_budget = sum(prefill budgets on prefillable ranks)
  fill = G_pending / G_prefill_budget if G_prefill_budget > 0 else 1.0
  if fill >= target_fill:                         FIRE
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
Single ``all_reduce(SUM)`` over a 13-int64 cpu buffer (gloo-safe). Booleans are
encoded as 0/1 and read back as ``sum > 0`` (logical OR). All timing is
tick-based (``hold_ticks``), which is deterministic across ranks — no per-rank
wall-clock, so ranks never diverge on a timeout boundary. The cost gate's step
times (t_decode a decode step, t_prefill a prefill step per unit of chunk cost)
are per-rank wall-clock averages, but they cross the collective and every rank
decides on their identical means. Fail-open on a
collective error (a peer dying should not crash schedule()); the guarded
DP-state barrier drives shutdown.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ATOM_PREFILL_DELAYER_DEBUG", "0").lower() in ("1", "true")
# weight of a new sample in the cost gate's step-time moving averages
_STEP_EMA_WEIGHT = 0.2
# The reduce buffer: these fixed slots, each rank's prefill cost, then the cost
# gate's `_Gate` (`PrefillDelayer.__init__`).
_FIXED_SLOTS = 13


class _Gate(NamedTuple):
    """The cost gate's slots, each SUM-reduced across the group."""

    waiting_requests: int
    decode_tick_us: int
    prefill_ps_per_cost: int
    timed_ranks: int
    arrivals_per_ks: int


def _ema(avg: float, sample: float) -> float:
    return sample if avg <= 0 else avg + _STEP_EMA_WEIGHT * (sample - avg)


# --------------------------------------------------------------------------
# --prefill-admit-policy adaptive  (default "interval" keeps the legacy behavior)
#
# Replaces the fixed post-prefill decode interval (ATOM_PREFILL_DECODE_INTERVAL)
# and the fill/stall coalescer with a rent-or-buy (EOQ) rule derived from
# measured step costs:
#
#   T0 = a, the decode step's part that  (a request held one more decision
#        does not grow with the batch     waits for the step that decision runs,
#                                         and its peers skip the time it would
#                                         add to that step: T_d - k*B)
#   C  = mean cost of one prefill pass   (prefill step + its aftermath, below)
#   B  = decode seqs of the group        (every one stalls while a pass runs)
#
# A hold makes each of the W admittable requests wait one decode step T_d;
# a FIRE stalls each of the B decoding requests for the pass. In the closed
# loop (a fixed population of clients) a request admitted one decision later
# also leaves one decode seq out of that step, which every one of the B
# decoding requests would otherwise pay k (the step's slope in the batch) for:
# admitting earlier does not merely move that time, it raises the batch every
# later step runs at. So a held request costs T_d - k*B = a, the step's part
# that does not grow with the batch.
#
# The two costs are weighed as serving is judged: the product
# of throughput and interactivity, each a relative change. A held request's
# wait lengthens its whole round trip Z (wait + decode + the client's turn),
# and in the closed loop throughput goes as 1/Z, so W*T0 counts as W*T0/Z;
# a pass's stall lengthens the decode time D of each of the B decoding
# requests, and interactivity goes as 1/D, so C*B counts as C*B/D. By
# Little's law, over the requests the group holds, Z/D >= (B + W)/B (the
# clients' own turns, which the group does not see, would only add to it).
# So the rule fires once the accumulated holding cost reaches the pass cost
# over the requests it holds -- sum over held ticks of W_t*T0 >= C*(B + W) --
# the EOQ optimum of that product for a steadily growing queue and
# 2-competitive otherwise. In a backlog (W comparable to B) this waits
# longer than C*B would, which keeps the batch from overshooting while the
# queue is long; with a short queue it is the plain rent-or-buy rule. No
# interval and no weight to tune: a cheaper pass, fewer decoding or waiting
# seqs, or more waiting requests a held tick delays, fire sooner.
#
# Sustained backlog: when admission capacity binds, the product rule spaces
# the passes too far apart (its threshold C*(B + W) shrinks slower than the
# holding cost as the queue drains), the queue drains slower and closed-loop
# throughput drops. A rank has a backlog when more fresh requests wait than
# arrived since the last FIRE, i.e. the last pass could not take its work;
# that 0/1 rides the SUM all_reduce. When the FIREs that open and close the
# current cycle both saw a backlog somewhere in the group (one leftover can be
# a burst or one long prompt), the rule switches to the throughput optimum:
# FIRE once sum over held ticks of W_t*T_d >= C*B (a hold costs each waiting
# request a whole decode step; a pass stalls each decoding request for C).
# This never admits less often than the fixed interval rule during a backlog.
# Floor: no FIRE before the _FLOOR_TICKS-th decision of a cycle, so the rule
# never admits more often than the fixed interval rule either. Together, a
# sustained backlog runs at the fixed cadence (throughput unchanged) and the
# product rule only stretches the cycle where that buys interactivity.
#
# Lockstep: every term of the rule is a reduced (group) value, so every rank
# decides alike; no rank-local estimate enters the decision (one that did
# would let ranks fire apart: a pass per rank group instead of one aligned
# pass).
#
# T0 = T_d - k*B is measured from the plain decode steps (outside a pass's
# aftermath): T_d and B are their mean step and mean batch, which follow the
# load within a few samples; k is the step's slope in the batch it ran, a
# regression on the batch's level over a memory of its own (below), which
# spans the batch's spread across cycles. A step a request finishes in also
# carries that request's completion work; that work goes with a change of the
# batch between successive steps, not with its level, so it adds to the
# noise, not to the slope (a fit on successive differences, or one that sees
# the batch move only by completions, would take it for the slope and drive
# T0 to 0). Until the batch has varied there is no slope and T0 = T_d; k is
# kept within [0, T_d / B].
#
# C is measured, not extrapolated: a prefill pass also slows the steps after
# it -- its GPU work spills into the next step, and ranks still finishing it
# reach the DP sync of the step after that late -- so that aftermath is
# charged to the pass: every following decode step's excess over the decode
# mean, while the step stays above the mean and shorter than the one before
# (the aftermath decays; a step at or below the mean, or no shorter than the
# previous one, is back to plain decoding). Steps in a pass's aftermath stay
# out of the decode mean. A pass sample weighs as the decisions of the cycle
# it ends: the cycle a decision falls in is drawn by length, so this is the
# expected cost of the pass a decision is waiting for, and a burst of
# back-to-back passes (warm-up flood, a load spike) does not outweigh the
# cycles that follow it by its count of passes.
#
# Memory: T_d, k and C are `_AdaptiveMean`s, which size their memory from
# their own samples instead of a half-life. The rule places a FIRE in a cycle
# of N decisions to one decision only if T0 and C are known to about 1/N (a
# relative error e moves the FIRE by e*n in the backlog regime, by half that
# in the growth regime), so an estimate keeps the fewest samples that give it
# that standard error: n_eff = CV^2 * N^2 for a mean (CV^2 = noise variance /
# mean^2, the noise taken from successive samples -- residuals about the line
# for k -- blind to a slow drift), and for k, whose share k*B of T0 must be
# resolved, n_eff = CV^2 * (mean B^2 / var B) * N^2 with CV taken on T0
# (unbounded, i.e. growing a sample a sample, while the batch has not
# varied). While the
# samples drift -- the innovations, in units of what that noise and the
# mean's own error predict, average away from 0 by more than their standard
# error (squared ratio > 1) -- the memory shrinks by that ratio, so a load
# change is followed within a few samples. The memory grows by at most one
# sample per sample (no mean rests on more samples than it has seen since it
# last forgot), so after an outlier or a shift it regrows as a running mean.
# N is the measured mean cycle (the mean weight of the pass samples). Nothing
# to tune; a mean is used from its first sample.
#
# No capacity cap: once one pass cannot take the waiting work, the leftover
# keeps W -- and so the regret -- high from the first decision of the next
# cycle, and fewer requests decode (smaller B), so EOQ fires sooner without
# one. The longest hold is the ttft_max_ticks bound checked before this rule.
# No cold-start cadence: until T0 and C are measured the rule holds (the
# must-fire bounds still apply). The first pass is forced by the no-decode
# bound and times C; holding runs the decode steps that time T0; so both are
# measured within a few decisions of the first request. Should the first pass
# go untimed, ttft_max_ticks releases the next one.
#
# Lockstep: every rank measures T0 and C from its own step clock; the means,
# the per-tick admittable counts and each rank's executed prefill tokens ride
# the existing SUM all_reduce, and every rank decides from the identical
# reduced values.
# --------------------------------------------------------------------------
# --prefill-admit-policy values; Config and the CLI take them from here.
PREFILL_ADMIT_POLICIES = ("interval", "adaptive")
# fixed slots: W_eff, T0 (us), C (us), measured ranks, ranks with a backlog,
# T_d (us); then dp_size one-hot slots of each rank's prefill tokens executed
# on the previous tick
_ADAPT_FIXED = 6
# Throughput guard: the fewest decisions per cycle. The adaptive rule never
# admits more often than the fixed interval rule's cadence (40 decisions, the
# value the DEP serving recipes ran with), so its decode cadence is never
# choppier than that baseline; the backlog rule below keeps it from admitting
# less often while a backlog lasts, which pins a sustained backlog to this
# cadence.
_FLOOR_TICKS = 40
# one line per FIRE (rank 0): its n, reason and costs
_ADAPT_LOG_FIRES = _DEBUG


class _AdaptiveMean:
    """Exponentially weighted mean of y -- and of x, with the line y = a + k*x
    through them -- that sizes its own memory (see the module-level comment):
    after every sample, `memory` becomes the fewest samples whose standard
    error resolves one decision of a cycle of `cycle` decisions, shortened
    while the samples drift. What must be resolved: the mean of y; with
    fit=True, the slope's share k * mean x of the intercept (the decode
    step's slope in the batch, whose part T0 removes from the mean step)."""

    __slots__ = (
        "memory",
        "mx",
        "my",
        "n",
        "prev",
        "prev_x",
        "sc",
        "sk",
        "skk",
        "sr",
        "su",
        "sw",
        "sww",
        "vxx",
        "vxy",
    )

    def __init__(self):
        self.n = 0
        self.memory = 1.0
        # weights and squared weights; weighted means of x and y and the
        # weighted centered sums of x*x and x*y (updated in place, so an x that
        # never changes leaves vxx exactly 0)
        self.sw = self.sww = 0.0
        self.mx = self.my = self.vxx = self.vxy = 0.0
        # unweighted: samples; successive pairs (and squared, for their n_eff),
        # their half squared residual differences (noise), standardized
        # innovations
        self.sc = self.sk = self.skk = self.sr = self.su = 0.0
        self.prev = self.prev_x = 0.0

    def mean(self) -> float:
        return self.my if self.sw > 0 else 0.0

    def mean_weight(self) -> float:
        """Mean weight of the remembered samples (0 before any)."""
        return self.sw / self.sc if self.sc > 0 else 0.0

    def slope(self) -> float:
        """k of y = a + k*x over the remembered samples: 0 until x has varied
        (no slope to resolve), and within [0, mean y / mean x] -- y does not
        fall as x grows, and its part that does not grow with x is not
        negative."""
        if self.vxx <= 0:
            return 0.0
        k = self.vxy / self.vxx
        return min(max(k, 0.0), self.my / self.mx) if self.mx > 0 else 0.0

    def intercept(self) -> float:
        """a = mean y - k * mean x: the part of y that does not grow with x
        (the mean when there is no x)."""
        return self.my - self.slope() * self.mx if self.sw > 0 else 0.0

    def add(
        self,
        y: float,
        w: float = 1.0,
        cycle: float | None = None,
        x: float = 0.0,
        fit: bool = False,
    ) -> None:
        """Fold sample y (at x) of weight w. cycle: the decisions per cycle the
        estimate must resolve to one; None: this mean's own mean weight (pass
        samples weigh as the decisions of their cycle). fit: size the memory
        for the slope (see the class docstring), not the mean."""
        d = 1.0 - 1.0 / self.memory
        if self.n:
            # the residual about the line (fit) or about the mean
            k = self.slope() if fit else 0.0
            noise = 0.5 * ((y - self.prev) - k * (x - self.prev_x)) ** 2
            r = self.sr / self.sk if self.sr > 0 else noise
            u = 0.0
            if r > 0:
                # innovation in units of what noise and the estimate's own
                # error at this x predict for it
                n_eff = self.sw * self.sw / self.sww
                lev = 1.0
                if fit and self.vxx > 0:
                    lev += (x - self.mx) ** 2 * self.sw / self.vxx
                pred = self.my + k * (x - self.mx)
                u = (y - pred) / (r * (1.0 + lev / n_eff)) ** 0.5
            self.sk = d * self.sk + 1.0
            self.skk = d * d * self.skk + 1.0
            self.sr = d * self.sr + noise
            self.su = d * self.su + u
        self.sc = d * self.sc + 1.0
        self.sw = d * self.sw + w
        # squared weights decay by d^2: sw^2 / sww is the estimate's n_eff
        # (2M - 1 for equal weights)
        self.sww = d * d * self.sww + w * w
        dx = x - self.mx
        dy = y - self.my
        self.mx += w * dx / self.sw
        self.my += w * dy / self.sw
        self.vxx = d * self.vxx + w * dx * (x - self.mx)
        self.vxy = d * self.vxy + w * dx * (y - self.my)
        self.prev = y
        self.prev_x = x
        self.n += 1
        a = self.intercept() if fit else self.mean()
        if self.sk <= 0 or a <= 0:
            return
        if cycle is None:
            cycle = self.mean_weight()
        cv2 = self.sr / self.sk / (a * a)
        # the variance of k * mean x over that of the mean: mean x^2 / var x
        # (unbounded while x has not varied: no slope is resolved yet, so keep
        # every sample until it does)
        vif = 1.0
        if fit and self.mx != 0:
            vif = (
                self.mx * self.mx * self.sw / self.vxx if self.vxx > 0 else float("inf")
            )
        # drift: the mean innovation over its standard error, squared (~1
        # without drift; grows with the square of the samples since a shift)
        u_mean = self.su / self.sk
        drift = max(1.0, u_mean * u_mean * self.sk * self.sk / self.skk)
        # an exponentially weighted mean of memory M has n_eff = 2M - 1. It
        # grows by at most one sample a sample: an estimate cannot rest on
        # more samples than it has seen since it last forgot (memory M+1 after
        # M is the running mean). This also keeps one outlier pair, which
        # inflates the noise until it is forgotten, from freezing the memory.
        self.memory = min(
            self.memory + 1.0,
            max(1.0, (cv2 * vif * cycle * cycle / drift + 1.0) / 2.0),
        )


class PrefillDecision(NamedTuple):
    """One lockstep decision, alike on every rank: whether prefill may be
    admitted this tick, and how many ranks of the group have prefill to admit."""

    allow: bool
    prefillable_ranks: int


class PrefillDelayer:
    __slots__ = (
        "_adapt_arr_since_fire",
        "_adapt_at",
        "_adapt_backlog_prev",
        "_adapt_dec",
        "_adapt_dec_clean",
        "_adapt_last_fire",
        "_adapt_last_tokens",
        "_adapt_pf",
        "_adapt_pf_open",
        "_adapt_prev_dec_per_rank",
        "_adapt_regret",
        "_adapt_regret_td",
        "_adapt_since_pf",
        "_adapt_slope",
        "_adapt_thr_last",
        "_arrival_gap_s",
        "_clock",
        "_cost_gate",
        "_cost_slots",
        "_decode_interval_remaining",
        "_decode_tick_us",
        "_fired_cost",
        "_first",
        "_gate_at",
        "_hold_cost_us",
        "_hold_ticks",
        "_last_arrival_s",
        "_last_call_s",
        "_prefill_executed_since_last_decision",
        "_prefill_only_since_last_decision",
        "_prefill_ps_per_cost",
        "_prev_pending",
        "_prev_prefillable",
        "_prev_running_dec",
        # buffer / episode state
        "_rank",
        "_reduce_buf",
        "_reduce_np",
        "_stall_count",
        "_stat_adapt_cycles",
        "_stat_adapt_hold_ticks",
        "_stat_fire_adaptive",
        "_stat_fire_backlog",
        # stats
        "_stat_fire_cost",
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
        "admit_policy",
        "cpu_group",
        "dp_size",
        "kv_high_watermark",
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
        cost_gate: bool = False,
        clock=time.perf_counter,
        admit_policy: str = "interval",
    ):
        if dp_size > 1 and cpu_group is None:
            raise ValueError("Cross-DP prefill coalescing requires a CPU group")
        self.dp_size = dp_size
        self.cpu_group = cpu_group

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
        if admit_policy not in PREFILL_ADMIT_POLICIES:
            raise ValueError(f"unknown admit_policy={admit_policy!r}")
        self.admit_policy = admit_policy
        if admit_policy == "adaptive" and prefill_decode_interval:
            logger.info(
                "prefill_admit_policy=adaptive: ignoring "
                f"prefill_decode_interval={prefill_decode_interval}"
            )
            prefill_decode_interval = 0
        self.prefill_decode_interval = prefill_decode_interval

        # `_FIXED_SLOTS`-slot SUM-reduce buffer (gloo-safe). Encoding:
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
        #   slot 8 = arrived_requests   (SUM → G_arrived; restarts the stall
        #                                count, see _decide)
        #   slot 9 = blocked flag       (SUM → n_blocked; waiting work but the
        #                                head cannot be admitted)
        #   slot 10 = unreachable flag  (SUM → n_unreachable; no prefill and no
        #                                decode completion within the hold)
        #   slot 11 = prefill_only flag (SUM>0 → that forward decoded nothing:
        #                                the step the decode-protection
        #                                interval answers; a mixed step decoded
        #                                beside its prefill and arms none)
        #   slot 12 = prefill_budget    (SUM over prefillable ranks → what one
        #                                step can admit; the fill target's
        #                                denominator. Below max_num_batched_tokens
        #                                when mixed batching reserves decode rows)
        # With exchange_prefill_costs (DP prefill cost balancing), dp_size more:
        #   slot `_FIXED_SLOTS` + r = rank r's prefill chunk cost (one-hot, SUM
        #                → every rank's cost), read back as rank_prefill_costs
        # With cost_gate (on top of the costs), a `_Gate` after them, from
        #   `_gate_at`: waiting requests, the rank's decode step in µs and its
        #   prefill time a cost unit in ps, ranks that timed both (the divisor
        #   of the two means), request arrivals per 1000 s on ranks without
        #   prefill
        self._cost_slots = dp_size if exchange_prefill_costs else 0
        # The cost gate judges only a wait for a rank that cannot align in
        # time (n_unreachable): at 8k/1k c128 DPA its releases for stragglers
        # about to complete a decode fragmented prefills (2578 vs 2810 tok/s).
        self._cost_gate = cost_gate and exchange_prefill_costs
        # The DP group is stateless (no default process group): ask it directly.
        self._rank = (
            cpu_group.rank() if self._cost_slots and cpu_group is not None else 0
        )
        self._gate_at = _FIXED_SLOTS + self._cost_slots
        self._adapt_at = self._gate_at + (len(_Gate._fields) if self._cost_gate else 0)
        if admit_policy == "adaptive" and cpu_group is not None:
            self._rank = cpu_group.rank()
        self._reduce_buf = torch.zeros(
            self._adapt_at
            + (_ADAPT_FIXED + dp_size if admit_policy == "adaptive" else 0),
            dtype=torch.int64,
            device="cpu",
        )
        self._reduce_np = self._reduce_buf.numpy()
        # adaptive admission state (see the module-level comment)
        # the mean decode step (and batch) and, over a memory of its own, the
        # step's slope in the batch
        self._adapt_dec = _AdaptiveMean()
        self._adapt_arr_since_fire = 0
        self._adapt_backlog_prev = False
        self._adapt_regret_td = 0.0
        self._adapt_slope = _AdaptiveMean()
        self._adapt_pf = _AdaptiveMean()
        self._adapt_last_tokens = 0
        self._adapt_prev_dec_per_rank = 0.0
        # the previous step was a plain decode step (timed, decode running,
        # outside a pass's aftermath): the next decode step is a T0 sample
        self._adapt_dec_clean = False
        self._adapt_regret = 0.0
        self._adapt_thr_last = (0.0, 0.0, 0.0, 0)
        # pass-cost sample still collecting its aftermath:
        # [cost us, weight (decisions), last step us]; None = no pass open
        self._adapt_pf_open = None
        # decisions since the previous prefill step (-1: none seen yet)
        self._adapt_since_pf = -1
        # last adaptive-policy FIRE: (reason, n, eoq_n, C_us, T0_us, B)
        self._adapt_last_fire = None
        self._stat_adapt_cycles = 0
        self._stat_adapt_hold_ticks = 0
        self._stat_fire_adaptive = 0
        self.rank_prefill_costs: list[float] = []
        # the alignment gate's waiting cost this hold episode (µs x requests)
        self._hold_cost_us = 0.0
        # Step timing for the cost gate: the interval between two decisions is
        # one scheduler step. Moving averages of a decode step and of a prefill
        # step's time per unit of the costliest chunk it ran; 0 = not yet timed.
        self._clock = clock
        self._last_call_s = None
        # this rank's request arrivals: moving average of their spacing (0 =
        # fewer than two seen) and when the last came
        self._arrival_gap_s = 0.0
        self._last_arrival_s = None
        self._prev_running_dec = 0
        self._fired_cost = 0.0
        self._decode_tick_us = 0.0
        self._prefill_ps_per_cost = 0.0

        # Episode state. All ticks decide FIRE/HOLD in lockstep, so these evolve
        # identically on every rank (deterministic).
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        self._prev_prefillable = -1
        # First call fires immediately to seed the initial decode batch build-up
        # (mirrors SGL PR #19836's skip_first).
        self._first = True
        self._decode_interval_remaining = 0
        self._prefill_executed_since_last_decision = False
        self._prefill_only_since_last_decision = False

        # Per-exit fire counters + hold counter for periodic logging. Each exit
        # is counted separately so the log shows WHICH reason released prefill —
        # a high fire_stall/fire_ttft vs fire_fill means the queue rarely reaches
        # target and the coalescer is mostly giving up rather than batching.
        self._stat_fire_cost = 0
        self._stat_fire_fill = 0
        self._stat_fire_backlog = 0
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
            f"prefill_decode_interval={prefill_decode_interval} "
            f"cost_gate={self._cost_gate} "
            f"admit_policy={admit_policy!r}"
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
                    self._prefill_only_since_last_decision
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
        waiting_requests: int = 0,
        arrived_requests: int = 0,
        next_completion_steps: int | None = None,
        admittable_requests: int | None = None,
        *,
        prefill_budget: int,
    ) -> PrefillDecision:
        """Whether this rank may admit new prefills this tick (FIRE), and how
        many ranks have prefill to admit.

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
                partials — already capped at ``prefill_budget`` by the
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
            waiting_requests: requests on this rank a hold keeps waiting
                (fresh prefills and resumable partials); read by the cost gate.
            arrived_requests: requests added to this rank since the previous
                call; the cost gate's arrival rate.
            next_completion_steps: decode steps until one of this rank's
                running decodes reaches max_tokens if every step accepts all
                its draft tokens (EOS can come sooner); None when it runs no
                decode. Read only when this rank has no prefill and no
                waiting work.
            admittable_requests: (prefill_admit_policy adaptive) requests a
                FIRE could put into this rank's decode batch now: resumable
                partials plus waiting heads that fit the free seq slots.
                None falls back to `waiting_requests`.
            prefill_budget: the prefill tokens one step can admit on this rank
                after reserving decode work. Mixed scheduling leaves a 1/8 token
                floor when decode pages fit, or zero under page pressure.
                Required even when this rank has no prefillable work.
        """
        # KV + queue-age bounds are gated on this rank actually having prefill to
        # push (firing when this rank can't admit anything would be a no-op).
        low = self.token_usage_low_watermark
        kv_high = prefillable and kv_usage >= self.kv_high_watermark
        # Waiting work this rank cannot admit (no KV for its head, or the
        # head's remote KV still loading): its own decode completions or the
        # load change that, not its peers waiting, so holding cannot align it.
        blocked = not prefillable and (waiting_requests > 0 or has_partial)
        # A rank with neither aligns once a decode completes and its slot takes
        # the next request. If that cannot happen before the TTFT bound
        # releases this hold anyway, waiting for it only delays its peers.
        # hold_ticks evolves in lockstep, so every rank compares to the same
        # remaining budget.
        unreachable = (
            not prefillable
            and not blocked
            and (
                next_completion_steps is None
                or next_completion_steps > self.ttft_max_ticks - self._hold_ticks
            )
        )
        kv_low = prefillable and low is not None and kv_usage < low
        queue_hot = (
            prefillable
            and self.max_queue_ms is not None
            and oldest_waiting_age_ms >= self.max_queue_ms
        )

        self._reduce_np[0] = 1 if prefillable else 0
        # A blocked rank counts toward alignment but can admit none of its
        # pending tokens, so they must not fill the forward.
        self._reduce_np[1] = int(pending_tokens) if prefillable else 0
        self._reduce_np[2] = int(running_decode_batch)
        self._reduce_np[3] = 1 if kv_high else 0
        self._reduce_np[4] = 1 if kv_low else 0
        self._reduce_np[5] = 1 if has_partial else 0
        self._reduce_np[6] = 1 if queue_hot else 0
        self._reduce_np[7] = 1 if self._prefill_executed_since_last_decision else 0
        self._reduce_np[8] = int(arrived_requests)
        self._reduce_np[9] = 1 if blocked else 0
        self._reduce_np[10] = 1 if unreachable else 0
        self._reduce_np[11] = 1 if self._prefill_only_since_last_decision else 0
        self._reduce_np[12] = int(prefill_budget) if prefillable else 0
        if self._cost_slots:
            self._reduce_np[_FIXED_SLOTS : self._gate_at] = 0
            self._reduce_np[_FIXED_SLOTS + self._rank] = int(prefill_cost)
        # one scheduler step since the previous decision (0: none yet, or untimed)
        step_s = 0.0
        if self._cost_gate:
            now = self._clock()
            if self._last_call_s is not None:
                step_s = now - self._last_call_s
            self._last_call_s = now
            self._count_arrivals(now, arrived_requests)
            timed = self._decode_tick_us > 0 and self._prefill_ps_per_cost > 0
            gate = _Gate(
                waiting_requests=int(waiting_requests),
                decode_tick_us=int(self._decode_tick_us) if timed else 0,
                prefill_ps_per_cost=int(self._prefill_ps_per_cost) if timed else 0,
                timed_ranks=1 if timed else 0,
                arrivals_per_ks=(
                    0 if prefillable else int(1000 * self._arrival_rate(now))
                ),
            )
            self._reduce_np[self._gate_at : self._gate_at + len(gate)] = gate
        if self.admit_policy == "adaptive":
            if not self._cost_gate:
                now_a = self._clock()
                if self._last_call_s is not None:
                    step_s = now_a - self._last_call_s
                self._last_call_s = now_a
            w_eff = (
                waiting_requests if admittable_requests is None else admittable_requests
            )
            t0_us, c_us = self._adapt_local_fits()
            at = self._adapt_at
            buf = self._reduce_np
            buf[at] = int(w_eff) if prefillable else 0
            # measured: a decode step and a pass timed. T0 itself may be a
            # measured 0 (the step is all per-seq cost) -- that rank still
            # counts, or the group mean would skip it and over-charge holds
            fitted = self._adapt_dec.n > 0 and c_us > 0
            buf[at + 1] = int(t0_us) if fitted else 0
            buf[at + 2] = int(c_us) if fitted else 0
            buf[at + 3] = 1 if fitted else 0
            # Backlog: a waiting request that was already here at the
            # last FIRE -- more fresh heads wait than arrived since -- i.e. the
            # pass could not take this rank's work. A resumable partial is not
            # counted: one long prompt spans passes without a queue behind it.
            self._adapt_arr_since_fire += int(arrived_requests)
            heads = int(waiting_requests) - (1 if has_partial else 0)
            buf[at + 4] = 1 if prefillable and heads > self._adapt_arr_since_fire else 0
            buf[at + 5] = int(self._adapt_dec.mean()) if fitted else 0
            buf[at + _ADAPT_FIXED : at + _ADAPT_FIXED + self.dp_size] = 0
            buf[at + _ADAPT_FIXED + self._rank] = int(self._adapt_last_tokens)
            self._adapt_last_tokens = 0
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
                return PrefillDecision(allow=True, prefillable_ranks=0)

        # One readback for all slots (a single .tolist() beats per-slot .item()
        # boundary crossings on this per-tick hot path).
        reduced = self._reduce_buf.tolist()
        self.rank_prefill_costs = [
            float(c) for c in reduced[_FIXED_SLOTS : self._gate_at]
        ]
        return PrefillDecision(
            allow=self._decide(reduced, step_s), prefillable_ranks=int(reduced[0])
        )

    def _decide(self, reduced: list[int], step_s: float) -> bool:
        allow = self._decide_inner(reduced, step_s)
        if allow and self.admit_policy == "adaptive":
            # every rank FIREs on the same decision: the arrival count restarts
            # and the backlog seen at this FIRE (left by the previous pass)
            # is remembered, identically on all ranks (a reduced value)
            self._adapt_arr_since_fire = 0
            self._adapt_backlog_prev = reduced[self._adapt_at + 4] > 0
        return allow

    def _decide_inner(self, reduced: list[int], step_s: float) -> bool:
        """FIRE (True) or HOLD, from the group's reduced signals."""
        (
            n_prefillable,
            g_pending,
            g_running_dec,
            kv_high_n,
            kv_low_n,
            partial_n,
            queue_hot_n,
            prefill_ran_n,
            g_arrived,
            n_blocked,
            n_unreachable,
            prefill_only_n,
            g_prefill_budget,
        ) = reduced[:_FIXED_SLOTS]
        if self._cost_gate:
            self._time_step(step_s, prefill_ran_n > 0)
            self._prev_running_dec = g_running_dec
        if self.admit_policy == "adaptive":
            self._adapt_observe(reduced, step_s, prefill_ran_n > 0, g_running_dec)
        # This local execution report has now participated in the global
        # decision. A later successful prefill forward sets it again.
        self._prefill_executed_since_last_decision = False
        self._prefill_only_since_last_decision = False
        any_kv_high = kv_high_n > 0
        any_kv_low = kv_low_n > 0
        any_partial = partial_n > 0
        any_queue_hot = queue_hot_n > 0

        # A prefill that actually ran on the previous tick arms the same
        # countdown on every rank. Delayer FIRE is only permission to try:
        # allocation, remote-KV handling, or queue changes may still produce a
        # decode/empty batch, which must not start a decode-protection window.
        # Nor does a mixed step: it decoded beside its prefill. The cost gate
        # above still times it as the prefill step it is.
        if prefill_only_n > 0:
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
                # Not a coalescer tick, so the stall count does not advance,
                # but work that arrives in the window still restarts it.
                if self._work_changed(g_pending, n_prefillable, g_arrived):
                    self._stall_count = 0
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
        if self.admit_policy == "adaptive":
            return self._adapt_decide(reduced, g_running_dec, g_pending)
        if any_partial and self._hold_ticks >= self.partial_max_ticks:
            return self._fire("partial", g_pending)

        # Stall: the group's work stopped changing, so more waiting will not
        # fill or align the forward. Pending tokens are capped per rank and can
        # stay flat while a decoding peer finishes a request and receives its
        # next one, so a rank turning (non-)prefillable or any arrival in the
        # group restarts the count. This also restarts it when the group
        # aligns, so ticks held unaligned never release an aligned forward.
        if self._work_changed(g_pending, n_prefillable, g_arrived):
            self._stall_count = 0
        else:
            self._stall_count += 1
        stalled = self._stall_count >= self.stall_ticks

        # ---- alignment gate: hold while a rank lacks prefill (anti-skew) for
        # stragglers to catch up, until the group's work stops changing while
        # some straggler cannot align within the TTFT bound (stall: e.g. idle
        # ranks at concurrency below dp_size) or (cost gate) the wait has cost
        # more than the unaligned prefill it avoids. A straggler that will
        # complete a decode in time is waited for even through quiet ticks: at
        # high concurrency its next request arrives with that completion. A
        # blocked rank is no straggler: waiting cannot make its head
        # admittable, so it keeps its peers company with a dummy prefill, as an
        # idle rank does once released.
        if n_prefillable + n_blocked < self.dp_size:
            if n_unreachable > 0:
                if stalled:
                    return self._fire("stall", g_pending)
                if self._cost_gate and self._alignment_wait_paid(
                    _Gate(*reduced[self._gate_at :]), g_running_dec
                ):
                    return self._fire("cost", g_pending)
            return self._hold(g_pending)

        # ---- goal: fire once the aggregate fills a worthwhile forward ----
        # What the prefillable ranks can admit this step, not the batch budget:
        # under mixed batching a rank holds decode rows back, and pending past
        # its prefill budget is work it cannot take in one step anyway.
        budget = g_prefill_budget
        fill = g_pending / budget if budget > 0 else 1.0
        if fill >= self.target_fill:
            return self._fire("fill", g_pending)

        if stalled:
            return self._fire("stall", g_pending)
        return self._hold(g_pending)

    def _work_changed(self, g_pending: int, n_prefillable: int, g_arrived: int) -> bool:
        """Whether the group's work moved since the previous coalescer tick
        (pending grew, a rank turned (non-)prefillable, or a request arrived),
        recording this tick's values for the next comparison."""
        changed = (
            g_pending > self._prev_pending
            or n_prefillable != self._prev_prefillable
            or g_arrived > 0
        )
        self._prev_pending = g_pending
        self._prev_prefillable = n_prefillable
        return changed

    def _alignment_wait_paid(self, gate: _Gate, g_running_dec: int) -> bool:
        """Whether waiting longer for alignment costs more than it can save.

        Releasing while a rank lacks prefill stalls every decode for one
        prefill step that a later aligned release would share -- if work
        reaches the ranks without prefill meanwhile. A held tick delays every
        waiting request by a decode step and saves that stall only with the
        chance of such an arrival in the tick: release once that expectation
        falls below the tick's cost, or once the wait has cost the whole stall
        (rent-or-buy). Untimed (no step measured on some rank yet): never
        paid, only the stall and must-fire bounds release."""
        timed = gate.timed_ranks
        if timed == 0:
            return False
        fire_us = (
            g_running_dec
            * max(self.rank_prefill_costs)
            * gate.prefill_ps_per_cost
            / timed
            / 1e6
        )
        if gate.arrivals_per_ks / 1e9 * fire_us < gate.waiting_requests:
            return True
        if self._hold_cost_us >= fire_us:
            return True
        self._hold_cost_us += gate.waiting_requests * gate.decode_tick_us / timed
        return False

    def _count_arrivals(self, now: float, arrived: int) -> None:
        if arrived <= 0:
            return
        if self._last_arrival_s is not None:
            gap = (now - self._last_arrival_s) / arrived
            self._arrival_gap_s = _ema(self._arrival_gap_s, gap)
        self._last_arrival_s = now

    def _arrival_rate(self, now: float) -> float:
        """Requests a second reaching this rank: the inverse of their average
        spacing, decaying while none has come for longer than that."""
        if self._arrival_gap_s <= 0:
            return 0.0
        return 1.0 / max(self._arrival_gap_s, now - self._last_arrival_s)

    def _time_step(self, step_s: float, prefill_ran: bool) -> None:
        """Fold the step run since the previous decision into the averages: a
        prefill step per unit of the costliest chunk reported when it fired, a
        decode step when decode ran and no prefill did."""
        if step_s <= 0:
            return
        if prefill_ran:
            if self._fired_cost > 0:
                self._prefill_ps_per_cost = _ema(
                    self._prefill_ps_per_cost, step_s * 1e12 / self._fired_cost
                )
            self._fired_cost = 0.0
        elif self._prev_running_dec > 0:
            self._decode_tick_us = _ema(self._decode_tick_us, step_s * 1e6)

    def _fire(self, reason: str, g_pending: int) -> bool:
        # reason ∈ {fill, stall, cost, ttft, kv, partial, nodecode, queue_ms};
        # each maps to a dedicated _stat_fire_<reason> slot so the log shows
        # what released.
        attr = f"_stat_fire_{reason}"
        setattr(self, attr, getattr(self, attr) + 1)
        self._fired_cost = max(self.rank_prefill_costs, default=0.0)
        if self.admit_policy == "adaptive":
            self._adapt_note_fire(reason)
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] FIRE ({reason}): g_pending={g_pending} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count}"
            )
        self._reset()
        self._maybe_log()
        return True

    # ---------------- prefill_admit_policy adaptive ----------------
    def _adapt_local_fits(self) -> tuple[float, float]:
        """This rank's (T0, C) in us; 0 = not measured yet (T0 may also be a
        measured 0: the step is all per-seq cost).
        T0 = the decode step's part that does not grow with the decode batch
        (intercept of the step on the batch); C = mean cost of a prefill pass."""
        dec = self._adapt_dec
        t_d = dec.mean()
        if dec.mx <= 0:
            return t_d, self._adapt_pf.mean()
        k = min(self._adapt_slope.slope(), t_d / dec.mx)
        return t_d - k * dec.mx, self._adapt_pf.mean()

    def _adapt_close_pass(self) -> None:
        """Fold the open pass-cost sample into C, weighted by the decisions of
        the cycle it ended."""
        cost_us, weight, _ = self._adapt_pf_open
        self._adapt_pf.add(cost_us, weight)
        self._adapt_pf_open = None

    def _adapt_observe(
        self, reduced: list[int], step_s: float, prefill_ran: bool, g_running_dec: int
    ) -> None:
        """Fold the step that ran since the previous decision into the
        estimates: a prefill step opens a pass-cost sample that its aftermath
        completes; a plain decode step after a plain decode step feeds the
        decode step's line in the batch it ran."""
        at = self._adapt_at
        if self._adapt_since_pf >= 0:
            self._adapt_since_pf += 1
        dec = self._adapt_dec
        dec_mean = dec.mean()
        open_ = self._adapt_pf_open
        timed = step_s > 0 and self._adapt_prev_dec_per_rank > 0
        if prefill_ran:
            if open_ is not None:
                self._adapt_close_pass()
            toks = max(reduced[at + _ADAPT_FIXED : at + _ADAPT_FIXED + self.dp_size])
            if toks > 0:
                # the cycle this pass ends: decisions since the previous one
                weight = float(max(self._adapt_since_pf, 1))
                self._adapt_since_pf = 0
                if step_s > 0:
                    self._adapt_pf_open = [step_s * 1e6, weight, 0.0]
            self._adapt_dec_clean = False
        elif open_ is not None:
            # A prefill pass also slows the steps after it (its GPU work spills
            # into the next one, late ranks hold up the DP sync of the one
            # after): while a step stays above the decode mean and shorter
            # than the one before, its excess is part of the pass. The first
            # step back at the mean (or no shorter, e.g. a wider batch) closes
            # the sample; picked by its time, it is no T0 sample either.
            y = step_s * 1e6
            if timed and 0 < dec_mean < y and (open_[2] == 0 or y < open_[2]):
                open_[0] += y - dec_mean
                open_[2] = y
            else:
                self._adapt_close_pass()
                self._adapt_dec_clean = timed
        else:
            if timed and self._adapt_dec_clean:
                # A plain decode step at the batch it ran (per rank). A sample
                # over twice the current mean -- more than one step's worth on
                # top of a step (warm-up, compile, a stall) -- is clipped to
                # that, so the estimate cannot run away. Its memory resolves
                # the intercept to the measured cycle (1 decision before the
                # first pass is measured).
                y = step_s * 1e6
                if dec.n:
                    y = min(y, 2.0 * dec_mean)
                cycle = max(self._adapt_pf.mean_weight(), 1.0)
                x = self._adapt_prev_dec_per_rank
                dec.add(y, 1.0, cycle, x=x)
                self._adapt_slope.add(y, 1.0, cycle, x=x, fit=True)
            self._adapt_dec_clean = timed
        self._adapt_prev_dec_per_rank = g_running_dec / max(self.dp_size, 1)

    def _adapt_decide(
        self, reduced: list[int], g_running_dec: int, g_pending: int
    ) -> bool:
        """Rent-or-buy (all reduced values; hold while T0 and C are not measured
        on any rank yet), never before the _FLOOR_TICKS-th decision:
        - sustained backlog (this pass and the one before left waiting
          requests behind): admission capacity binds, so FIRE once
          sum_t W_t*T_d >= C*B -- every held decision costs each waiting
          request a whole decode step (the throughput rule);
        - otherwise FIRE once sum_t W_t*T0 >= C*(B + W) (the product rule)."""
        at = self._adapt_at
        w_eff, t0_sum, c_sum, fitted, backlog_n, td_sum = reduced[
            at : at + _ADAPT_FIXED
        ]
        if w_eff <= 0:
            # nothing a FIRE could admit (slots full / KV-blocked): holding is free
            return self._adapt_hold(g_pending)
        if fitted <= 0:
            # Not measured yet: a few decisions after the first request (see
            # the module-level comment); the must-fire bounds cover them.
            self._adapt_thr_last = (0.0, 0.0, 0.0, g_running_dec)
            return self._adapt_hold(g_pending)
        t0 = t0_sum / fitted
        c = c_sum / fitted
        self._adapt_regret += w_eff * t0
        self._adapt_regret_td += w_eff * td_sum / fitted
        if self._hold_ticks + 1 < _FLOOR_TICKS:
            self._adapt_thr_last = (t0, c, 0.0, g_running_dec)
            return self._adapt_hold(g_pending)
        if backlog_n > 0 and self._adapt_backlog_prev:
            thr = c * g_running_dec
            self._adapt_thr_last = (t0, c, thr, g_running_dec)
            if self._adapt_regret_td >= thr:
                self._stat_adapt_cycles += 1
                return self._fire("backlog", g_pending)
            return self._adapt_hold(g_pending)
        # the pass cost over the requests the group holds (decoding + waiting)
        thr = c * (g_running_dec + w_eff)
        self._adapt_thr_last = (t0, c, thr, g_running_dec)
        if self._adapt_regret >= thr:
            self._stat_adapt_cycles += 1
            return self._fire("adaptive", g_pending)
        return self._adapt_hold(g_pending)

    def _adapt_note_fire(self, reason: str) -> None:
        """Record a FIRE under the adaptive policy: its decision count n, the
        EOQ decision count (n when EOQ fired; else extrapolated from the
        regret so far) and the costs it saw."""
        t0, c, thr, b = self._adapt_thr_last
        n = self._hold_ticks + 1
        regret = self._adapt_regret
        if reason in ("adaptive", "backlog"):
            eoq_n = float(n)
        else:
            eoq_n = n * thr / regret if regret > 0 else -1.0
        self._adapt_last_fire = (reason, n, eoq_n, c, t0, b)
        if _ADAPT_LOG_FIRES and self._rank == 0:
            logger.info(
                f"[PrefillDelayer adaptive fire] reason={reason} n={n} "
                f"eoq_n={eoq_n:.1f} C_us={c:.0f} T0_us={t0:.0f} "
                f"B={b} regret={regret:.0f} thr={thr:.0f} "
                f"pf_n={self._adapt_pf.n} pf_mem={self._adapt_pf.memory:.1f} "
                f"dec_n={self._adapt_dec.n} dec_mem={self._adapt_dec.memory:.1f} "
                f"Td_us={self._adapt_dec.mean():.0f} "
                f"k_ns={1e3 * self._adapt_slope.slope() / max(self.dp_size, 1):.0f} "
                f"k_mem={self._adapt_slope.memory:.1f}"
            )

    def _adapt_hold(self, g_pending: int) -> bool:
        self._stat_adapt_hold_ticks += 1
        return self._hold(g_pending)

    def notify_prefill_executed(
        self, tokens: int = 0, *, decoded: bool = False
    ) -> None:
        """Report a completed local prefill forward.

        The next lockstep decision folds both flags into the existing SUM
        reduction, so every DP rank arms the interval together without adding
        a second collective to the scheduler hot path. Every prefill forward is
        timed as one (the cost gate); only one that decoded nothing arms the
        decode-protection interval.
        """
        self._prefill_executed_since_last_decision = True
        self._adapt_last_tokens += int(tokens)
        if not decoded:
            self._prefill_only_since_last_decision = True

    def notify_idle(self) -> None:
        """Report a tick the DP group spent with no work, which schedules
        nothing: the next decision follows no step, so it times none."""
        self._last_call_s = None

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
        self._prev_prefillable = -1
        self._hold_cost_us = 0.0
        self._adapt_regret = 0.0
        self._adapt_regret_td = 0.0

    def _maybe_log(self) -> None:
        # Cheap guard first — skip the counter sum entirely when logging is
        # disabled (this runs on every FIRE/HOLD tick).
        if self._stat_log_every <= 0:
            return
        total = (
            self._stat_fire_fill
            + self._stat_fire_stall
            + self._stat_fire_cost
            + self._stat_fire_ttft
            + self._stat_fire_kv
            + self._stat_fire_partial
            + self._stat_fire_nodecode
            + self._stat_fire_queue_ms
            + self._stat_fire_vacuous
            + self._stat_fire_adaptive
            + self._stat_fire_backlog
            + self._stat_hold
        )
        if total == 0:
            return
        if total % self._stat_log_every == 0:
            logger.info(
                f"[PrefillDelayer stats] total={total} "
                f"fire_fill={self._stat_fire_fill} "
                f"fire_stall={self._stat_fire_stall} "
                f"fire_cost={self._stat_fire_cost} "
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
            if self.admit_policy == "adaptive":
                t0, c, thr, b = self._adapt_thr_last
                cyc = self._stat_adapt_cycles
                last = self._adapt_last_fire
                last_s = f"{last[0]}:n={last[1]}:eoq_n={last[2]:.1f}" if last else "-"
                logger.info(
                    "[PrefillDelayer adaptive] "
                    f"fire_adaptive={self._stat_fire_adaptive} "
                    f"T0_us={t0:.0f} C_us={c:.0f} B={b} thr={thr:.0f} "
                    f"dec_n={self._adapt_dec.n} dec_mem={self._adapt_dec.memory:.1f} "
                    f"pf_n={self._adapt_pf.n} pf_mem={self._adapt_pf.memory:.1f} "
                    f"hold_ticks/cycle={self._stat_adapt_hold_ticks / max(cyc, 1):.1f} "
                    f"last_fire={last_s}"
                )
