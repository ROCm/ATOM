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
    ) -> None:
        super().__init__(lightest)
        self.select_session = select_session
        self.owners = owners
        self.prompts = prompts
        self.rank_count = rank_count
        self.parent_ignored = parent_ignored

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        session = getattr(seq, "dp_session_id", None)
        if not session:
            return super().select(seq, explicit_rank)
        owner = self.owners.get(session)
        if explicit_rank is not None:
            if owner is not None and owner != explicit_rank:
                self.prompts.pop(session, None)
            owner = explicit_rank
            decision = "explicit_total"
        elif owner is None or not 0 <= owner < self.rank_count:
            owner = self.select_session(session)
            decision = "affinity_new_total"
            if getattr(seq, "dp_parent_session_id", None):
                self.parent_ignored()
        else:
            decision = "affinity_owner_hit_total"
        self.owners[session] = owner
        return Route(owner, decision, self.prompts.get(session, 0))

    def charge(self, seq: Any, credit: int) -> int:
        session = getattr(seq, "dp_session_id", None)
        if session:
            self.prompts[session] = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        return credit

    def reset(self) -> None:
        self.owners.clear()
        self.prompts.clear()


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
