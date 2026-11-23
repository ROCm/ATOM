# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DP placement policies and their request-lifecycle hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atom.model_engine.dp_routing.route import Route


class LoadPolicy:
    stateful = False
    router = None

    def __init__(self, lightest: Callable[[], int]) -> None:
        self.lightest = lightest

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        if explicit_rank is not None:
            return Route(explicit_rank, "explicit_total")
        return Route(self.lightest(), "load_balanced_total")

    def charge(self, seq: Any, credit: int) -> int:
        return credit

    def prefilling(self, seq_id: Any) -> bool:
        return False

    def prefill_complete(self, seq_id: Any) -> None:
        pass

    def release(self, seq_id: Any) -> dict:
        return {}

    def reset(self) -> None:
        pass


class SessionPolicy(LoadPolicy):
    stateful = True

    def __init__(
        self,
        lightest: Callable[[], int],
        select_session: Callable[[str], int],
        owners: dict,
        prompts: dict,
        rank_count: int,
        parent_ignored: Callable[[], None],
        *,
        prefix_index: Any = None,
        owner_tokens: int = 0,
        owner_extra_reqs: int = 0,
        rank_reqs: Callable[[], list[int]] | None = None,
        prefix_owner_seen: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__(lightest)
        self.select_session = select_session
        self.owners = owners
        self.prompts = prompts
        self.rank_count = rank_count
        self.parent_ignored = parent_ignored
        # engine.dp_session_prefix_owner_tokens: the prefix index (a
        # `PrefixSpreadRouter` used for its index alone) every request is
        # recorded in, and the rule a new session reads it by; None = off.
        self.prefix_index = prefix_index
        self.owner_tokens = owner_tokens
        self.owner_extra_reqs = owner_extra_reqs
        self.rank_reqs = rank_reqs
        # Told what became of each new session whose prefix is on another rank
        # than the lightest: "follow", "threshold" or "guard", and the tokens
        # that rank holds more.
        self.prefix_owner_seen = prefix_owner_seen

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        session = getattr(seq, "dp_session_id", None)
        index = self.prefix_index
        if not session:
            route = super().select(seq, explicit_rank)
            if index is not None:
                index.record(seq, route.rank)
            return route
        owner = self.owners.get(session)
        credit = self.prompts.get(session, 0)
        if explicit_rank is not None:
            if owner is not None and owner != explicit_rank:
                self.prompts.pop(session, None)
                credit = 0
            owner = explicit_rank
            decision = "explicit_total"
        elif owner is None or not 0 <= owner < self.rank_count:
            owner = self.select_session(session)
            decision = "affinity_new_total"
            if getattr(seq, "dp_parent_session_id", None):
                self.parent_ignored()
            if index is not None:
                prefix_rank, hit = self._prefix_owner(seq, owner)
                if prefix_rank != owner:
                    # Charged its prompt less the prefix it moved for.
                    owner, credit = prefix_rank, hit
                    decision = "affinity_new_prefix_owner_total"
        else:
            decision = "affinity_owner_hit_total"
        self.owners[session] = owner
        if index is not None:
            index.record(seq, owner)
        return Route(owner, decision, credit)

    def _prefix_owner(self, seq: Any, lightest: int) -> tuple[int, int]:
        """The rank a new session takes instead of `lightest`, and the prefix
        it finds there: the rank holding the longest finished prefix of its
        prompt, if that is `owner_tokens` more than `lightest` holds and the
        rank has at most `owner_extra_reqs` more requests in flight than the
        least loaded one. Otherwise `lightest` (and 0)."""
        hits = self.prefix_index.confirmed_hits(seq, self.rank_count)
        if not hits:
            return lightest, 0
        reqs = self.rank_reqs()
        rank = max(hits, key=lambda r: (hits[r], -reqs[r], -r))
        gained = hits[rank] - hits.get(lightest, 0)
        if rank == lightest or gained <= 0:
            return lightest, 0
        if gained < self.owner_tokens:
            outcome = "threshold"
        elif reqs[rank] - min(reqs[: self.rank_count]) > self.owner_extra_reqs:
            outcome = "guard"
        else:
            outcome = "follow"
        if self.prefix_owner_seen is not None:
            self.prefix_owner_seen(outcome, gained)
        if outcome != "follow":
            return lightest, 0
        return rank, hits[rank]

    def charge(self, seq: Any, credit: int) -> int:
        session = getattr(seq, "dp_session_id", None)
        if session:
            self.prompts[session] = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        if self.prefix_index is not None:
            # In flight until its prefill completes; no credit rides on it.
            self.prefix_index.charge(seq.id, 0)
        return credit

    def prefilling(self, seq_id: Any) -> bool:
        return self.prefix_index is not None and self.prefix_index.prefilling(seq_id)

    def prefill_complete(self, seq_id: Any) -> None:
        if self.prefix_index is not None:
            self.prefix_index.prefill_complete(seq_id)

    def release(self, seq_id: Any) -> dict:
        if self.prefix_index is None:
            return {}
        return self.prefix_index.release(seq_id)

    def reset(self) -> None:
        self.owners.clear()
        self.prompts.clear()
        if self.prefix_index is not None:
            self.prefix_index.reset()


class PrefixPolicy(LoadPolicy):
    stateful = True

    def __init__(self, router, rank_tokens, rank_reqs, rank_load, lightest) -> None:
        super().__init__(lightest)
        self.router = router
        self.rank_tokens = rank_tokens
        self.rank_reqs = rank_reqs
        self.rank_load = rank_load

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        rank_tokens = self.rank_tokens()
        rank_reqs = self.rank_reqs()
        if explicit_rank is not None:
            placed = self.router.place(seq, explicit_rank, len(rank_tokens))
            return Route(
                explicit_rank, "explicit_total", placed.credit, placed.moved_tokens
            )
        return self.router.select(
            seq, rank_tokens, rank_reqs, self.rank_load, self.lightest
        )

    def charge(self, seq: Any, credit: int) -> int:
        return self.router.charge(seq.id, credit)

    def prefilling(self, seq_id: Any) -> bool:
        return self.router.prefilling(seq_id)

    def prefill_complete(self, seq_id: Any) -> None:
        self.router.prefill_complete(seq_id)

    def release(self, seq_id: Any) -> dict:
        return self.router.release(seq_id)

    def reset(self) -> None:
        self.router.reset()


class ShareCachePolicy(PrefixPolicy):
    """Shared-tier placement; the router balances on `rank_tokens`, each rank's
    reported prefill backlog."""

    def charge(self, seq: Any, credit: int) -> int:
        seq.dp_est_cached_tokens = hit = super().charge(seq, credit)
        return hit
