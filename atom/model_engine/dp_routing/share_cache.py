# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Work placement over the common DP prefix ownership and offload credit index."""

from __future__ import annotations

from typing import ClassVar

from atom.model_engine.dp_routing.prefix_spread import PrefixSpreadRouter


class SharedWorkPlacement:
    """Balance actual backlog plus this request's cost and decode slot pressure."""

    def __init__(self, req_equiv: int) -> None:
        self.req_equiv = int(req_equiv)
        self.cursor = 0

    def __call__(self, backlog, requests, costs, owner):
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
    """Shared-tier placement: the prefix router's index, balanced on backlog.

    All prefix state belongs to `PrefixSpreadRouter`; this class only picks the
    placement rule and names the decision counters.
    """

    DECISIONS: ClassVar[dict[str, str]] = {
        "share_cache_hit_total": "Requests placed on a prefix the shared tier holds.",
        "share_cache_source_rank_total": "Requests placed on the rank still prefilling their prefix.",
        "share_cache_cold_total": "Requests placed without a known prefix.",
        "share_cache_stale_total": "Actual reuse reports corrected stale prefix hints.",
    }

    def __init__(self, tier, req_equiv: int) -> None:
        super().__init__(idle_tokens=0, tier=tier, max_seqs=0)
        self.placement = SharedWorkPlacement(req_equiv)

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
