# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Work placement over the common DP prefix ownership and offload credit index."""

from __future__ import annotations

import math
from typing import ClassVar

from atom.model_engine.dp_routing.prefix_spread import PrefixSpreadRouter


def _smooth_max_rises(values, adds, temp: float) -> list[float]:
    """For each rank r, how much smooth-max(values) rises when values[r] grows
    by adds[r]; smooth-max(v) = temp * log(sum(exp(v / temp))), the max of v
    plus at most temp*log(n).

    Close to the rise of the plain max for the busiest rank, and small but not
    zero for a rank just under it: the chance that it becomes the busiest while
    the request lives there.

    One pass for all ranks: with p_r = exp(v_r / temp) / sum(exp(v / temp)),
    the rise is temp * log(1 + p_r * expm1(add / temp)). Sums are shifted by the
    max, and a large add is taken in log space, so big backlogs or increments
    neither overflow nor lose the rise to cancellation."""
    top = max(values)
    weights = [math.exp((v - top) / temp) for v in values]
    total = sum(weights)
    log_total = math.log(total)
    rises = []
    for value, weight, add in zip(values, weights, adds):
        if add <= 0:
            rises.append(0.0)
            continue
        grow = add / temp
        share = weight / total
        if grow <= 32.0:
            rises.append(temp * math.log1p(share * math.expm1(grow)))
            continue
        # log(1 - p + p * e^grow) = log(e^stay + e^moved)
        moved = (value - top) / temp - log_total + grow
        rest = total - weight
        if rest <= 0:
            rises.append(add)
            continue
        stay = math.log(rest) - log_total
        high, low = (moved, stay) if moved >= stay else (stay, moved)
        rises.append(temp * (high + math.log1p(math.exp(low - high))))
    return rises


class LockstepCostPlacement:
    """Place a request where it adds the least lockstep step time.

    DP ranks step together, so a step lasts as long as its busiest rank:
    decode steps grow with the widest rank's decode batch, a prefill pass with
    the rank that prefills the most tokens. A rank's in-flight requests are the
    decode batch it is about to run (decoding, plus queued requests that decode
    once prefilled). The score of rank r, in milliseconds of group step time:

      PF_MS_PER_TOKEN * rise of smooth-max(backlog) by this request's cost on r
      + DECODE_MS_PER_LANE * DECODE_STEPS * rise of smooth-max(in-flight) by 1
      + LOAD_MS_PER_TOKEN * tokens r loads from the tier / in-flight requests

    Both balance terms follow the busiest rank, so a placement is charged only
    when it raises (or may soon raise) the step everyone waits for; ranks well
    below it are not pulled level with each other. The last term is the
    request's own wait for its tier load, shared by the in-flight requests a
    lockstep step serves; it keeps a request on the rank that holds its prefix
    in HBM unless that rank is near the busiest. The constants were fitted on
    DeepSeek-V4.1-Flash DEP8 with an agentic workload: the decode step cost per
    lane of the widest rank, the prefill pass plus the step after it per token
    of the busiest rank, the mean decode steps of a request, and the tier load
    time per token. Only their ratios matter, and choices change little when
    any of them is scaled by 4x either way. Ties go to the prefix owner, then
    round-robin.
    """

    # Group step ms per prefill token of the busiest rank.
    PF_MS_PER_TOKEN = 0.0093
    # Group decode step ms per lane of the widest rank.
    DECODE_MS_PER_LANE = 0.33
    # Decode steps a request lives on its rank, on average.
    DECODE_STEPS = 276
    # Smooth-max temperatures: in-flight requests, and prefill tokens.
    LANE_TEMP = 2.0
    TOKEN_TEMP = 4096.0
    # Tier load ms per token: KV bytes per token / tier load bandwidth.
    LOAD_MS_PER_TOKEN = 1490 / 1.25e9 * 1e3

    def __init__(self) -> None:
        self.cursor = 0

    def scores(self, backlog, requests, costs, loads):
        decode_ms = self.DECODE_MS_PER_LANE * self.DECODE_STEPS
        sharing = max(1, sum(requests))
        prefill = _smooth_max_rises(backlog, costs, self.TOKEN_TEMP)
        decode = _smooth_max_rises(requests, [1] * len(requests), self.LANE_TEMP)
        return [
            self.PF_MS_PER_TOKEN * pf
            + decode_ms * dc
            + self.LOAD_MS_PER_TOKEN * load / sharing
            for pf, dc, load in zip(prefill, decode, loads)
        ]

    def __call__(self, backlog, requests, costs, owner, loads):
        width = len(backlog)
        cursor = self.cursor
        self.cursor += 1
        score = self.scores(backlog, requests, costs, loads)
        return min(
            range(width),
            key=lambda rank: (score[rank], rank != owner, (rank - cursor) % width),
        )


class SharedWorkPlacement:
    """Balance actual backlog plus this request's cost and decode slot pressure.

    The rule for P/D prefill and decode engines, where one engine does not run
    both the prefill and the decode of a request, so the lockstep costs above
    (fitted on engines that do both) do not describe its step."""

    def __init__(self, req_equiv: int) -> None:
        self.req_equiv = int(req_equiv)
        self.cursor = 0

    def __call__(self, backlog, requests, costs, owner, loads=None):
        width = len(backlog)
        cursor = self.cursor
        self.cursor += 1
        return min(
            range(width),
            key=lambda rank: (
                backlog[rank] + costs[rank] + self.req_equiv * requests[rank],
                rank != owner,
                (rank - cursor) % width,
            ),
        )


class ShareCacheRouter(PrefixSpreadRouter):
    """Shared-tier placement: the prefix router's index, balanced on lockstep cost.

    All prefix state belongs to `PrefixSpreadRouter`; this class only picks the
    placement rule and names the decision counters. `pd_req_equiv` is set on a
    P/D prefill or decode engine: it keeps `SharedWorkPlacement` with that
    per-request weight instead of `LockstepCostPlacement`.
    """

    DECISIONS: ClassVar[dict[str, str]] = {
        "share_cache_hit_total": "Requests placed on a prefix the shared tier holds.",
        "share_cache_source_rank_total": "Requests placed on the rank still prefilling their prefix.",
        "share_cache_cold_total": "Requests placed without a known prefix.",
        "share_cache_stale_total": "Actual reuse reports corrected stale prefix hints.",
    }

    def __init__(self, tier, pd_req_equiv: int | None = None) -> None:
        super().__init__(idle_tokens=0, tier=tier, max_seqs=0)
        self.placement = (
            LockstepCostPlacement()
            if pd_req_equiv is None
            else SharedWorkPlacement(pd_req_equiv)
        )

    def place(self, seq, rank, n_ranks):
        route = super().place(seq, rank, n_ranks)
        if route.credit:
            self._reuse_pending[seq.id] = (seq.prompt_hash_chain, route.credit, rank)
        return route

    def select(self, seq, rank_tokens, rank_reqs, rank_load, lightest):
        """`rank_tokens` is each rank's prefill backlog; the load callbacks of
        the prefix policy are not used."""
        return self.select_work(
            seq,
            rank_tokens,
            rank_reqs,
            self.placement,
            (
                "share_cache_source_rank_total",
                "share_cache_hit_total",
                "share_cache_cold_total",
            ),
        )
