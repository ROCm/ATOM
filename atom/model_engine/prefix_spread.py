# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefix-spread DP routing (`--dp-load-balance prefix_spread`).

A request stays on the rank holding its prompt's prefix, unless that rank has a
prefill backlog while another rank is about to run out of prefill work; then it
goes to that idle rank, which loads the prefix from the offload tier.
`CoreManager` owns the per-rank load counters and calls into this router with
them under its lock.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from atom.utils import envs


class PrefixSpreadRouter:
    """Prefix ownership index, in-flight prefill state and the placement rule."""

    # Sampled prefix hashes, `STRIDE_TOKENS` apart: 2^17 entries cover 512M
    # tokens of prefix history.
    STRIDE_TOKENS = 4096
    CAP = 1 << 17

    def __init__(self, idle_tokens: int, tier_loads: bool = True) -> None:
        # A rank with under `idle_tokens` queued prefill is about to idle; a
        # prefix's rank with over twice that has a backlog.
        self.idle_tokens = idle_tokens
        # Whether a rank can load a prefix it does not hold from an offload
        # tier. Without one a moved request recomputes what it does not hold.
        self.tier_loads = tier_loads
        self.counters = dict.fromkeys(
            (
                "prefix_owner_total",
                "prefix_moved_total",
                "prefix_moved_tokens",
                "prefix_moved_partial_total",
                "prefix_cold_total",
                "prefix_source_busy_total",
            ),
            0,
        )
        # Sampled prompt-block hash -> {rank: request whose prefill puts that
        # part on the rank}; most recent last.
        self.index: OrderedDict[int, dict] = OrderedDict()
        # seq id -> (rank, hashes whose entry it relies on before its prefill
        # completes, prompt tokens), and (hash, rank) -> unfinished requests
        # relying on that not-yet-confirmed entry.
        self.seq_hashes: dict = {}
        self.pending: dict = {}
        # Requests routed here whose prefill has not completed (kept apart from
        # the token charge, which is 0 for a fully cached prompt that is
        # nevertheless still prefilling), and the prefix each was credited.
        self._prefilling: set = set()
        self._est_hit: dict = {}

    def reset(self) -> None:
        self.index.clear()
        self.seq_hashes.clear()
        self.pending.clear()
        self._prefilling.clear()
        self._est_hit.clear()

    # -- request lifecycle (called by CoreManager under its lock) --------------

    def charged(self, seq_id) -> int:
        """The prefix `seq_id` was credited at `select`; it is now in flight."""
        self._prefilling.add(seq_id)
        return self._est_hit.pop(seq_id, 0)

    def prefilling(self, seq_id) -> bool:
        return seq_id in self._prefilling

    def prefill_complete(self, seq_id) -> None:
        """`seq_id` completed its prefill: the entries it relied on are real."""
        self._prefilling.discard(seq_id)
        entry = self.seq_hashes.pop(seq_id, None)
        if entry is None:
            return
        rank, hashes, _ = entry
        for h in hashes:
            self.pending.pop((h, rank), None)

    def released(self, seq_id) -> dict:
        """`seq_id` left (finished, aborted, rejected, failed send). Returns
        {request: prompt tokens} for requests that took over computing a part
        it left unprefilled (`_left_unprefilled`); they are charged it."""
        self._est_hit.pop(seq_id, None)
        if seq_id not in self._prefilling:
            self.prefill_complete(seq_id)
            return {}
        self._prefilling.discard(seq_id)
        return {
            heir: prompt
            for heir, prompt in self._left_unprefilled(seq_id).items()
            if heir in self._prefilling
        }

    # -- placement ------------------------------------------------------------

    def select(
        self,
        seq: Any,
        rank_tokens: list[int],
        rank_load: Callable[[int], int],
        lightest: Callable[[], int],
    ) -> tuple[int, str]:
        """(rank, route counter) for `seq`; `charged` returns the prefix it is
        credited.

        - No known prefix: the lightest rank (`least_tokens` score).
        - The prefix's rank (owner) has a backlog (over 2 x idle queued
          prefill) and another rank is about to idle (under idle): that idle
          rank (`_idle_target`). This applies even while the request computing
          the prefix is still prefilling; the tier cannot supply it yet, so
          the request is charged everything but the target's own part.
        - Otherwise the owner. While the prefix's source is still prefilling
          this is what lets the request reuse it from HBM.
        """
        chain = getattr(seq, "prompt_hash_chain", None)
        prompt = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        hits, owner, source = self._rank_hits(chain, len(rank_tokens))
        if owner is None:
            rank = lightest()
            self._record(chain, rank, seq.id, 0, None, prompt)
            self.counters["prefix_cold_total"] += 1
            return rank, "prefix_cold_total"

        hit = hits.get(owner, 0)
        busy = source is not None and source in self._prefilling
        rank = owner
        if rank_tokens[owner] > 2 * self.idle_tokens:
            target = self._idle_target(owner, hits, rank_tokens, rank_load)
            if target is not None:
                rank = target
        if rank == owner and busy:
            self.counters["prefix_source_busy_total"] += 1

        local = hits.get(rank, 0)
        # The tier holds the prefix only once its source finished prefilling,
        # and the connector loads only a rest of at least the load threshold.
        loadable = (
            self.tier_loads and not busy and hit - local >= envs.OFFLOAD_MIN_LOAD_TOKENS
        )
        est_hit = hit if rank == owner or loadable else local
        self._est_hit[seq.id] = est_hit
        if hit <= 0:
            decision = "prefix_cold_total"
        elif rank == owner:
            decision = "prefix_owner_total"
        else:
            decision = "prefix_moved_total"
            self.counters["prefix_moved_tokens"] += max(0, est_hit - local)
            if local > 0:
                self.counters["prefix_moved_partial_total"] += 1
        self.counters[decision] += 1
        self._record(chain, rank, seq.id, est_hit, source, prompt)
        return rank, decision

    def _idle_target(
        self,
        owner: int,
        hits: dict[int, int],
        rank_tokens: list[int],
        rank_load: Callable[[int], int],
    ) -> int | None:
        """The rank a request leaving its prefix's rank goes to, or None.

        Candidates are the ranks other than the owner with under idle queued
        prefill -- chosen by queued prefill, the quantity the rule is about,
        not by the combined load. Among those within idle / 2 of the least
        queued, the one holding the longest part of the prefix (it loads only
        the rest), then the least queued, then the lowest `least_tokens` load.
        """
        idle = self.idle_tokens
        cands = [
            r for r in range(len(rank_tokens)) if r != owner and rank_tokens[r] < idle
        ]
        if not cands:
            return None
        floor = min(rank_tokens[r] for r in cands)
        near = [r for r in cands if rank_tokens[r] <= floor + idle // 2]
        return min(near, key=lambda r: (-hits.get(r, 0), rank_tokens[r], rank_load(r)))

    # -- prefix index ---------------------------------------------------------

    def _rank_hits(self, chain, n: int):
        """Per rank, the longest sampled prefix of this prompt it holds.

        Returns ({rank: tokens}, owner, source): the owner is the rank with the
        longest prefix (the most recent one on a tie), source the request whose
        prefill puts it there. A rank other than the owner counts a part only
        once its source finished prefilling it.
        """
        if chain is None or not self.index:
            return {}, None, None
        hbs = int(chain.hash_block_size)
        stride = max(1, self.STRIDE_TOKENS // hbs)
        hashes = chain.hashes
        hits: dict[int, int] = {}
        owner, source = None, None
        for i in range((len(hashes) // stride) * stride - 1, -1, -stride):
            holders = self.index.get(hashes[i])
            if not holders:
                continue
            for rank, seq_id in reversed(holders.items()):
                if not 0 <= rank < n or rank in hits:
                    continue
                if owner is None:
                    owner, source = rank, seq_id
                elif seq_id in self._prefilling:
                    continue  # a shorter, finished part may still count below
                hits[rank] = (i + 1) * hbs
            if len(hits) == n:
                break
        return hits, owner, source

    def _record(
        self, chain, rank: int, seq_id, hit: int, hit_source, prompt_tokens=0
    ) -> None:
        """Record that `rank` holds this prompt's sampled prefix.

        Each entry names the request whose prefill puts that part on the rank:
        `seq_id` for the part it computes; for the part it reuses (the first
        `hit` tokens) the request that computed it -- the rank's existing
        holder, else `hit_source`. So a follower that only reuses a prefix
        never becomes its busy source. Every entry this request adds or
        rewrites stays provisional until a request relying on it completes its
        prefill (`prefill_complete`); if all of them leave first, it is
        removed (`left_unprefilled`).
        """
        if chain is None:
            return
        hbs = int(chain.hash_block_size)
        stride = max(1, self.STRIDE_TOKENS // hbs)
        index, pending = self.index, self.pending
        relied = []
        for i in range(stride - 1, len(chain.hashes), stride):
            h = chain.hashes[i]
            holders = index.get(h)
            if holders is None:
                holders = index[h] = {}
            else:
                index.move_to_end(h)
            reused = (i + 1) * hbs <= hit
            had = rank in holders
            if reused:
                producer = holders.pop(rank, hit_source)
            else:
                holders.pop(rank, None)
                producer = seq_id
            holders[rank] = producer
            key = (h, rank)
            if not had or not reused:
                pending.setdefault(key, set()).add(seq_id)
                relied.append(h)
            elif key in pending:
                pending[key].add(seq_id)
                relied.append(h)
        if relied:
            self.seq_hashes[seq_id] = (rank, relied, prompt_tokens)
        while len(index) > self.CAP:
            old, holders = index.popitem(last=False)
            for r in holders:
                pending.pop((old, r), None)

    def _left_unprefilled(self, seq_id) -> dict:
        """`seq_id` left before its prefill completed (abort, reject, failed
        send). Remove the provisional entries no other unfinished request on
        that rank still relies on.

        Returns {request: its prompt tokens} for the requests that took over
        as producer of a part `seq_id` was computing: they now compute it, so
        the caller recharges their prefill load."""
        promoted: dict = {}
        entry = self.seq_hashes.pop(seq_id, None)
        if entry is None:
            return promoted
        rank, hashes, _ = entry
        for h in hashes:
            key = (h, rank)
            waiters = self.pending.get(key)
            if waiters is None:
                continue
            waiters.discard(seq_id)
            if waiters:
                # Others on this rank still rely on the entry. If the leaving
                # request was its producer, the KV never arrived: one of them
                # now computes it, so it becomes the (still busy) producer.
                holders = self.index.get(h)
                if holders is not None and holders.get(rank) == seq_id:
                    heir = next(iter(waiters))
                    holders[rank] = heir
                    heir_entry = self.seq_hashes.get(heir)
                    if heir_entry is not None:
                        promoted[heir] = heir_entry[2]
                continue
            del self.pending[key]
            holders = self.index.get(h)
            if holders is not None and rank in holders:
                del holders[rank]
                if not holders:
                    del self.index[h]
        return promoted
