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
from typing import Any, NamedTuple

from atom.kv_transfer.disaggregation.factory import OffloadTierProfile


class Route(NamedTuple):
    """One placement: the rank, its route counter (`CoreManager.
    _record_dp_route_locked`), and for a move the prefix tokens the target
    loads from the tier and whether it already held part of the prefix."""

    rank: int
    decision: str
    moved_tokens: int = 0
    partial: bool = False


class PrefixSpreadRouter:
    """Prefix ownership index, in-flight prefill state and the placement rule."""

    # Sampled prefix hashes, `STRIDE_TOKENS` apart: 2^17 entries cover 512M
    # tokens of prefix history.
    STRIDE_TOKENS = 4096
    CAP = 1 << 17

    # Route counters `select` returns, for `CoreManager`'s route statistics.
    DECISIONS = ("prefix_owner_total", "prefix_moved_total", "prefix_cold_total")

    def __init__(
        self, idle_tokens: int, tier: OffloadTierProfile, max_seqs: int
    ) -> None:
        # A rank with under `idle_tokens` queued prefill is about to idle; a
        # prefix's rank with over twice that has a backlog.
        self.idle_tokens = idle_tokens
        # A rank with this many requests in flight cannot start another
        # (`max_num_seqs`): it is not idle, however little prefill it queues.
        self.max_seqs = max_seqs
        # Whether (and from how many tokens) a rank loads a prefix it does not
        # hold from an offload tier. Without one a moved request recomputes
        # what it does not hold.
        self.tier = tier
        # Sampled prompt-block hash -> {rank: request whose prefill puts that
        # part on the rank}; most recent last.
        self.index: OrderedDict[int, dict] = OrderedDict()
        # seq id -> (rank, hashes whose entry it relies on before its prefill
        # completes), and (hash, rank) -> unfinished requests relying on that
        # not-yet-confirmed entry.
        self.seq_hashes: dict = {}
        self.pending: dict = {}
        # Requests routed here whose prefill has not completed (kept apart from
        # the token charge, which is 0 for a fully cached prompt that is
        # nevertheless still prefilling), and the prefix each was credited.
        self._prefilling: set = set()
        self._est_hit: dict = {}
        # Prefix credit still owned by each in-flight request. A follower
        # taking over an aborted producer loses that credit exactly once.
        self._credited_hit: dict = {}

    def reset(self) -> None:
        """Forget in-flight state. The index stays: it describes the ranks'
        prefix caches, which outlive a batch."""
        for h, rank in self.pending:
            holders = self.index.get(h)
            if holders is not None:
                holders.pop(rank, None)
                if not holders:
                    del self.index[h]
        self.seq_hashes.clear()
        self.pending.clear()
        self._prefilling.clear()
        self._est_hit.clear()
        self._credited_hit.clear()

    # -- request lifecycle (called by CoreManager under its lock) --------------

    def charge(self, seq_id) -> int:
        """Mark `seq_id` in flight; returns the prefix it was credited at
        `select`."""
        self._prefilling.add(seq_id)
        hit = self._est_hit.pop(seq_id, 0)
        self._credited_hit[seq_id] = hit
        return hit

    def prefilling(self, seq_id) -> bool:
        return seq_id in self._prefilling

    def prefill_complete(self, seq_id) -> None:
        """`seq_id` completed its prefill: the entries it relied on are real."""
        self._prefilling.discard(seq_id)
        self._credited_hit.pop(seq_id, None)
        entry = self.seq_hashes.pop(seq_id, None)
        if entry is None:
            return
        rank, hashes = entry
        for h, _start in hashes:
            self.pending.pop((h, rank), None)

    def release(self, seq_id) -> dict:
        """Release a request; return extra prefill charges for its successors."""
        self._est_hit.pop(seq_id, None)
        if seq_id not in self._prefilling:
            self.prefill_complete(seq_id)
            return {}
        self._prefilling.discard(seq_id)
        self._credited_hit.pop(seq_id, None)
        return self._left_unprefilled(seq_id)

    # -- placement ------------------------------------------------------------

    def select(
        self,
        seq: Any,
        rank_tokens: list[int],
        rank_reqs: list[int],
        rank_load: Callable[[int], int],
        lightest: Callable[[], int],
    ) -> Route:
        """Where `seq` goes; `charge` returns the prefix it is credited.

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
        hits, owner, source = self._rank_hits(chain, len(rank_tokens))
        if owner is None:
            rank = lightest()
            self._record(chain, rank, seq.id, 0, None)
            return Route(rank, "prefix_cold_total")

        # An owner is a rank holding a sampled part, so `hit` > 0.
        hit = hits[owner]
        rank = owner
        if rank_tokens[owner] > 2 * self.idle_tokens:
            target = self._idle_target(owner, hits, rank_tokens, rank_reqs, rank_load)
            if target is not None:
                rank = target

        local = hits.get(rank, 0)
        # The tier holds the prefix only once its source finished prefilling,
        # and the connector loads only a rest of at least its load threshold.
        loaded = (
            self.tier.loadable_tokens(local, hit)
            if source not in self._prefilling
            else 0
        )
        est_hit = hit if rank == owner else local + loaded
        self._est_hit[seq.id] = est_hit
        self._record(chain, rank, seq.id, est_hit, source)
        if rank == owner:
            return Route(rank, "prefix_owner_total")
        return Route(rank, "prefix_moved_total", max(0, est_hit - local), local > 0)

    def place(self, seq: Any, rank: int, n_ranks: int) -> None:
        """Index a request placed on `rank` from outside (an explicit
        `data_parallel_rank`): it is credited the part of its prefix that
        rank already holds, and `rank` becomes a holder of the rest."""
        chain = getattr(seq, "prompt_hash_chain", None)
        if chain is None or not 0 <= rank < n_ranks:
            return
        hits, _owner, source = self._rank_hits(chain, n_ranks)
        local = hits.get(rank, 0)
        self._est_hit[seq.id] = local
        self._record(chain, rank, seq.id, local, source)

    def _idle_target(
        self,
        owner: int,
        hits: dict[int, int],
        rank_tokens: list[int],
        rank_reqs: list[int],
        rank_load: Callable[[int], int],
    ) -> int | None:
        """The rank a request leaving its prefix's rank goes to, or None.

        Candidates are the ranks other than the owner with under idle queued
        prefill and a free request slot (a rank full of decodes queues no
        prefill tokens, but cannot start the request) -- chosen by queued
        prefill, the quantity the rule is about,
        not by the combined load. Among those within idle / 2 of the least
        queued, the one holding the longest part of the prefix (it loads only
        the rest), then the least queued, then the lowest `least_tokens` load.
        """
        idle = self.idle_tokens
        cands = [
            r
            for r in range(len(rank_tokens))
            if r != owner and rank_tokens[r] < idle and rank_reqs[r] < self.max_seqs
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
                # A remote restore is provisional even if its source finished.
                # Keep local in-flight producers as owners for HBM followers.
                if (hashes[i], rank) in self.pending and seq_id not in self._prefilling:
                    continue
                if owner is None:
                    owner, source = rank, seq_id
                elif seq_id in self._prefilling:
                    continue  # a shorter, finished part may still count below
                hits[rank] = (i + 1) * hbs
            if len(hits) == n:
                break
        return hits, owner, source

    def _record(self, chain, rank: int, seq_id, hit: int, hit_source) -> None:
        """Record that `rank` holds this prompt's sampled prefix.

        Each entry names the request whose prefill puts that part on the rank:
        `seq_id` for the part it computes; for the part it reuses (the first
        `hit` tokens) the request that computed it -- the rank's existing
        holder, else `hit_source`. So a follower that only reuses a prefix
        never becomes its busy source. Every entry this request adds or
        rewrites stays provisional until a request relying on it completes its
        prefill (`prefill_complete`); if all of them leave first, it is
        removed (`_left_unprefilled`).
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
            if had and holders[rank] in self._prefilling:
                # A newcomer must not erase another request computing this
                # prefix locally. It joins that producer's ordered waiters.
                producer = holders[rank]
            elif reused:
                producer = holders.pop(rank, hit_source)
            else:
                holders.pop(rank, None)
                producer = seq_id
            holders[rank] = producer
            key = (h, rank)
            if not had or not reused:
                pending.setdefault(key, {})[seq_id] = None
                relied.append((h, (i + 1 - stride) * hbs))
            elif key in pending:
                pending[key][seq_id] = None
                relied.append((h, (i + 1 - stride) * hbs))
        if relied:
            self.seq_hashes[seq_id] = (rank, relied)
        while len(index) > self.CAP:
            old, holders = index.popitem(last=False)
            for r in holders:
                pending.pop((old, r), None)

    def _left_unprefilled(self, seq_id) -> dict:
        """`seq_id` left before its prefill completed (abort, reject, failed
        send). Hand provisional entries to the oldest remaining waiter on
        that rank. Its credited prefix above the first lost sampled block
        becomes conservative prefill debt; CoreManager applies it under the
        same lock as this handoff."""
        charges: dict = {}
        entry = self.seq_hashes.pop(seq_id, None)
        if entry is None:
            return charges
        rank, hashes = entry
        for h, start in hashes:
            key = (h, rank)
            waiters = self.pending.get(key)
            if waiters is None:
                continue
            waiters.pop(seq_id, None)
            holders = self.index.get(h)
            produced = holders is not None and holders.get(rank) == seq_id
            if waiters and not produced:
                continue
            if waiters and holders is not None:
                successor = next(iter(waiters))
                holders[rank] = successor
                credited = self._credited_hit.get(successor, 0)
                retained = min(credited, start)
                if retained < credited:
                    self._credited_hit[successor] = retained
                    charges[successor] = charges.get(successor, 0) + credited - retained
                continue
            del self.pending[key]
            if holders is not None and rank in holders:
                del holders[rank]
                if not holders:
                    del self.index[h]
        return charges
