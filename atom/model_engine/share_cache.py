# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Work placement over the common DP prefix ownership and offload credit index."""

from __future__ import annotations

from typing import ClassVar

from atom.kv_transfer.offload.tier import OffloadTierProfile
from atom.model_engine.dp_routing.prefix_spread import PrefixSpreadRouter

SHARED_TIER_CONNECTORS = frozenset({"mooncake_store", "lmcache_mp"})


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
    """Shared work strategy; all prefix state belongs to PrefixSpreadRouter.

    The adapter keeps the public shared-placement API, not a second sampled
    prefix index or load-threshold algorithm. Production passes the registry's
    complete OffloadTierProfile, including load role and chunk granularity.
    """

    DECISIONS: ClassVar[dict[str, str]] = {
        "share_cache_hit_total": "Requests placed on a prefix the shared tier holds.",
        "share_cache_source_rank_total": "Requests placed on a local prefix owner.",
        "share_cache_cold_total": "Requests placed without a known prefix.",
        "share_cache_stale_total": "Actual reuse reports corrected stale prefix hints.",
    }

    def __init__(self, min_load_tokens: int, req_equiv: int, *, tier=None) -> None:
        super().__init__(
            idle_tokens=0,
            tier=(
                OffloadTierProfile(loads=True, min_load_tokens=int(min_load_tokens))
                if tier is None
                else tier
            ),
            max_seqs=0,
        )
        self.placement = SharedWorkPlacement(req_equiv)

    def select(self, seq, rank_backlog, rank_reqs):
        return self.select_work(seq, rank_backlog, rank_reqs, self.placement)

    def charge(self, seq_id, credit=None) -> int:
        if credit is None:
            credit = self._reuse_pending.get(seq_id, (None, 0, None))[1]
        return super().charge(seq_id, credit)
