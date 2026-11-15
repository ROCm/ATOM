# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefix-spread DP routing (`--dp-load-balance prefix_spread`).

A request stays on the rank holding its prompt's prefix, unless that rank has a
prefill backlog while another rank is about to run out of prefill work; then it
goes to that idle rank, reusing its local prefix and loading the missing part
from the offload tier only when it clears the tier's minimum load threshold.
`CoreManager` owns the per-rank load counters and calls into this router with
them under its lock.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from atom.model_engine.dp_routing.route import (
    PREFIX_COLD,
    PREFIX_DECISIONS,
    PREFIX_MOVED,
    PREFIX_MOVED_TOKENS,
    PREFIX_OWNER,
    Route,
)

if TYPE_CHECKING:
    from atom.kv_transfer.offload.tier import OffloadTierProfile


class _PrefixInterval:
    """A compressed radix edge of consecutive chained block hashes."""

    __slots__ = ("children", "domain", "hashes", "hbs", "parent", "root_hash", "start")

    def __init__(self, hashes, start, hbs, parent, domain):
        self.hashes = hashes
        self.start = start
        self.hbs = hbs
        self.domain = domain
        self.root_hash = hashes[0] if parent is None else parent.root_hash
        self.parent = parent
        self.children = {}

    @property
    def end(self):
        return self.start + len(self.hashes)


class PrefixSpreadRouter:
    """Prefix ownership index, in-flight prefill state and the placement rule."""

    # Token budget for unique radix edges, matching the former 4096-stride
    # history coverage. Active edges are pinned and may exceed this budget.
    CAP = 1 << 29

    # Route counters `select` returns, for `CoreManager`'s route statistics and
    # the metrics exporter.
    DECISIONS = PREFIX_DECISIONS
    MOVED_TOKENS = PREFIX_MOVED_TOKENS
    # Index-maintenance counters (`index_counters`), cumulative since the router
    # was built; `reset` does not clear them. Plain integer increments made by the
    # code that already does the work, under `CoreManager`'s lock.
    #   over_budget_total   `_trim_history` calls that found the index over `CAP`
    #   tail_trim_*         a leaf shortened from its tail (events, tokens)
    #   leaf_evict_*        a whole leaf dropped by the budget (events, tokens)
    #   trim_blocked_total  over-budget calls that ended still over `CAP`: no
    #                       evictable edge was left, only edges with a pending
    #                       (still-prefilling) claim and edges with children,
    #                       e.g. finished ancestors of a still-prefilling leaf
    # Tail trim / leaf evict are the budget path only; `_prune_empty` drops of
    # emptied edges are ordinary release bookkeeping and are not counted.

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
        # Compressed hash intervals -> {rank: request that produces that
        # interval}; most recent rank last. Each edge owns only its hash slice.
        self.index: OrderedDict[_PrefixInterval, dict] = OrderedDict()
        self._roots: dict = {}
        self._evictable: OrderedDict[_PrefixInterval, None] = OrderedDict()
        self.indexed_tokens = 0
        # One continuous chain per active request. Each provisional interval
        # names just its producer; followers are not copied into every edge.
        self.seq_hashes: dict = {}
        self._active: dict = {}
        self._next_ordinal = 0
        self.pending: dict = {}
        self._produced: dict = {}
        # Requests routed here whose prefill has not completed -> the prefix
        # credit each still owns. Membership is "still prefilling" (kept apart
        # from the token charge, which is 0 for a fully cached prompt that is
        # nevertheless still prefilling). A follower taking over an aborted
        # producer loses its credit exactly once (`_left_unprefilled`).
        self._in_flight: dict = {}
        # Placement-time credit backed independently of any pending producer.
        self._independent_credit: dict = {}
        self._over_budget = 0
        self._tail_trim = 0
        self._tail_trim_tokens = 0
        self._leaf_evict = 0
        self._leaf_evict_tokens = 0
        self._trim_blocked = 0

    def index_counters(self) -> dict:
        """Snapshot of the index-maintenance counters plus the current index size.
        Read-only; call under the same lock as the other router methods."""
        return {
            "over_budget_total": self._over_budget,
            "tail_trim_total": self._tail_trim,
            "tail_trim_tokens_total": self._tail_trim_tokens,
            "leaf_evict_total": self._leaf_evict,
            "leaf_evict_tokens_total": self._leaf_evict_tokens,
            "trim_blocked_total": self._trim_blocked,
            "indexed_tokens": self.indexed_tokens,
        }

    def reset(self) -> None:
        """Forget in-flight state. The index stays: it describes the ranks'
        prefix caches, which outlive a batch."""
        # A dict, not a set: edges hash by address, and the order they are
        # refreshed below decides which history `_trim_history` evicts first.
        affected = dict.fromkeys(edge for edge, _rank in self.pending)
        for edge, rank in self.pending:
            holders = self.index.get(edge)
            if holders is not None:
                holders.pop(rank, None)
        self.seq_hashes.clear()
        self._active.clear()
        self.pending.clear()
        self._produced.clear()
        self._in_flight.clear()
        self._independent_credit.clear()
        self._prune_empty(affected)
        for edge in affected:
            self._refresh_evictable(edge)
        self._trim_history()

    # -- request lifecycle (called by CoreManager under its lock) --------------

    def charge(self, seq_id, credit: int) -> int:
        """Mark `seq_id` in flight; returns the prefix it was credited at
        `select`."""
        self._in_flight[seq_id] = credit
        return credit

    def prefilling(self, seq_id) -> bool:
        return seq_id in self._in_flight

    def prefill_complete(self, seq_id) -> None:
        """`seq_id` completed its prefill: the entries it relied on are real."""
        self._in_flight.pop(seq_id, None)
        entry = self._remove_request(seq_id)
        if entry is None:
            return
        rank, chain, ordinal = entry
        path = self._path(chain)
        for edge, _end in path:
            producer = self.pending.get((edge, rank))
            if producer is None:
                continue  # already-confirmed reuse did not rely on this edge
            producer_entry = self.seq_hashes.get(producer)
            if producer != seq_id and (
                producer_entry is None or producer_entry[2] > ordinal
            ):
                continue  # this producer arrived after the completing request
            del self.pending[(edge, rank)]
            produced = self._produced.get(producer)
            if produced is not None:
                produced.discard(edge)
                if not produced:
                    del self._produced[producer]
            holders = self.index.get(edge)
            if holders is not None and rank in holders:
                # Confirmed history must not retain a request id: recycling
                # that id for a new prefill cannot turn old history provisional.
                holders.pop(rank)
                holders[rank] = None
                self._refresh_evictable(edge)
        # Deepest first: a merge drops only the edge it is called on.
        for edge, _end in reversed(path):
            if edge in self.index:
                self._merge_into_child(edge)
        self._trim_history()

    def release(self, seq_id) -> dict:
        """Release a request; return extra prefill charges for its successors."""
        if seq_id not in self._in_flight:
            self.prefill_complete(seq_id)
            return {}
        del self._in_flight[seq_id]
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
        - The prefix's rank (owner) has a backlog -- over 2 x idle queued
          prefill, or every request slot taken (`max_seqs`), so the request
          could not start there -- and another rank is about to idle (under
          idle, with a free slot): that idle rank (`_idle_target`). It is
          credited what it holds plus what the tier can load: the longest part
          some request finished prefilling (`_tier_hit`). If the request
          computing the prefix is still prefilling and no shorter copy is
          finished, the tier has nothing and the request is charged everything
          but the target's own part. A load below the minimum also keeps this
          target: it reuses its own part and recomputes the missing prefix.
        - Otherwise the owner. While the prefix's source is still prefilling
          this is what lets the request reuse it from HBM.
        """
        chain = getattr(seq, "prompt_hash_chain", None)
        # One read-only walk serves every lookup below; only `_record` walks again.
        path = self._path(chain)
        hits, owner = self._rank_hits(len(rank_tokens), path)
        if owner is None:
            rank = lightest()
            self._record(chain, rank, seq.id, 0)
            return Route(rank, PREFIX_COLD)

        rank = owner
        if (
            rank_tokens[owner] > 2 * self.idle_tokens
            or rank_reqs[owner] >= self.max_seqs
        ):
            target = self._idle_target(owner, hits, rank_tokens, rank_reqs, rank_load)
            if target is not None:
                rank = target

        est_hit, loaded = self._credit(seq, rank, hits, owner, path)
        self._record(
            chain,
            rank,
            seq.id,
            est_hit,
            self._independent_hit(seq, rank, est_hit, path),
        )
        if rank == owner:
            return Route(rank, PREFIX_OWNER, est_hit)
        return Route(rank, PREFIX_MOVED, est_hit, loaded)

    def place(self, seq: Any, rank: int, n_ranks: int) -> Route:
        """Index explicit placement with the same local/tier credit as a move."""
        chain = getattr(seq, "prompt_hash_chain", None)
        if chain is None or not 0 <= rank < n_ranks:
            return Route(rank, "explicit_total")
        path = self._path(chain)
        hits, owner = self._rank_hits(n_ranks, path)
        credit, loaded = self._credit(seq, rank, hits, owner, path)
        self._record(
            chain,
            rank,
            seq.id,
            credit,
            self._independent_hit(seq, rank, credit, path),
        )
        return Route(rank, "explicit_total", credit, loaded)

    def _credit(self, seq, rank, hits, owner, path):
        local = hits.get(rank, 0)
        if rank == owner or not self.tier.loads:
            return local, 0
        in_tier = self._tier_hit(path)
        loaded = self.tier.loadable_tokens(local, in_tier, seq.num_prompt_tokens)
        return local + loaded, loaded

    def _independent_hit(self, seq, rank, credit, path):
        """Credit available without waiting for a same-rank producer.

        Keep the placement-time confirmed local prefix and load policy: an
        unrelated completed tier copy does not disappear when that producer
        aborts. Unfinished tier copies and sub-threshold loads supply no floor.
        """
        local = 0
        for edge, hit in reversed(path):
            holders = self.index[edge]
            if (
                rank in holders
                and (edge, rank) not in self.pending
                and holders[rank] not in self._in_flight
            ):
                local = hit
                break
        if not self.tier.loads:
            return min(credit, local)
        loaded = self.tier.loadable_tokens(
            local, self._tier_hit(path), seq.num_prompt_tokens
        )
        return min(credit, local + loaded)

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

    def _path(self, chain, *, create=False, boundary=0):
        """Walk compressed edges; split only at a branch or credit boundary.

        Chained hashes identify the entire prefix up to their position. Equal
        hashes at an equal depth therefore let us binary-search a common prefix,
        under the same collision assumption used by the block cache.
        """
        if chain is None or not chain.hashes:
            return []
        hashes, hbs = chain.hashes, int(chain.hash_block_size)
        domain = (hbs, chain.cache_seed)
        children = (
            self._roots.setdefault(domain, {})
            if create
            else self._roots.get(domain, {})
        )
        parent = None
        pos, length, path = 0, len(hashes), []
        while pos < length:
            edge = children.get(hashes[pos])
            if edge is None:
                if not create:
                    break
                edge = _PrefixInterval(hashes[pos:], pos, hbs, parent, domain)
                children[hashes[pos]] = edge
                self.index[edge] = {}
                self.indexed_tokens += len(edge.hashes) * hbs
            limit = min(len(edge.hashes), length - pos)
            lo, hi = 0, limit
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if edge.hashes[mid - 1] == hashes[pos + mid - 1]:
                    lo = mid
                else:
                    hi = mid - 1
            matched = lo
            if not matched:
                break
            cut = matched
            if create and pos < boundary < pos + cut:
                cut = boundary - pos
            if create and cut < len(edge.hashes):
                edge = self._split(edge, cut, children)
                matched = cut
            path.append((edge, (pos + matched) * hbs))
            if matched < len(edge.hashes):
                break
            pos += matched
            parent, children = edge, edge.children
        return path

    def _split(self, edge, count, siblings):
        prefix = _PrefixInterval(
            edge.hashes[:count], edge.start, edge.hbs, edge.parent, edge.domain
        )
        siblings[prefix.hashes[0]] = prefix
        edge.hashes = edge.hashes[count:]
        edge.start += count
        edge.parent = prefix
        prefix.children[edge.hashes[0]] = edge
        self.index[prefix] = dict(self.index[edge])
        # Splitting copies interval ownership, not the requests relying on it.
        # Each request continues to retain one chain independent of tree depth.
        for rank in self.index[edge]:
            producer = self.pending.get((edge, rank))
            if producer is None:
                continue
            self.pending[(prefix, rank)] = producer
            self._produced.setdefault(producer, set()).add(prefix)
        return prefix

    def _merge_into_child(self, edge):
        """Undo a `_split` once nothing tells the two halves apart: `edge` has one
        child and both are held by the same ranks in the same recency order with
        no provisional producer. Every lookup answers alike on either side of the
        seam, so the child takes over `edge`'s hashes and place in the tree.
        Without this a session whose prompt grows each turn adds one edge per turn
        and every walk of its chain gets longer.
        """
        if len(edge.children) != 1:
            return
        (child,) = edge.children.values()
        holders = self.index[edge]
        if list(holders.items()) != list(self.index[child].items()) or any(
            (e, rank) in self.pending for e in (edge, child) for rank in holders
        ):
            return
        siblings = (
            self._roots[edge.domain] if edge.parent is None else edge.parent.children
        )
        child.hashes = edge.hashes + child.hashes
        child.start = edge.start
        child.parent = edge.parent
        siblings[child.hashes[0]] = child
        del self.index[edge]

    def _tier_hit(self, path) -> int:
        """Longest matching prefix with a finished holder on any rank."""
        for edge, hit in reversed(path):
            if any(
                seq_id not in self._in_flight for seq_id in self.index[edge].values()
            ):
                return hit
        return 0

    def _rank_hits(self, n: int, path):
        """Longest prefix per rank; equal-length owners use last-recorded order.

        Other ranks' unfinished producers do not supply a local hit. Seed and
        block-size domains are separate even if a supplied hash collides.
        """
        hits = {}
        owner = None
        for edge, hit in reversed(path):
            holders = self.index[edge]
            for rank, seq_id in reversed(holders.items()):
                if not 0 <= rank < n or rank in hits:
                    continue
                if (edge, rank) in self.pending and seq_id not in self._in_flight:
                    continue
                if owner is None:
                    owner = rank
                elif seq_id in self._in_flight:
                    continue
                hits[rank] = hit
            if len(hits) == n:
                break
        return hits, owner

    def _record(
        self, chain, rank: int, seq_id, hit: int, independent_credit: int = 0
    ) -> None:
        """Record one continuous interval, split only where prefixes diverge.

        A request depends on the provisional intervals it uses. History cannot
        evict any interval with active waiters, including its producer.
        """
        if chain is None:
            return
        path = self._path(
            chain, create=True, boundary=hit // int(chain.hash_block_size)
        )
        for edge, end in path:
            holders = self.index[edge]
            key = (edge, rank)
            reused = end <= hit
            had = rank in holders
            if had and key not in self.pending and holders[rank] not in self._in_flight:
                holders[rank] = holders.pop(rank)
                self._refresh_evictable(edge)
                continue
            if had and holders[rank] in self._in_flight:
                # A shared active interval is already pinned. There is no
                # follower table to update and no history LRU to touch here.
                if next(reversed(holders)) != rank:
                    holders[rank] = holders.pop(rank)
                continue
            elif reused:
                producer = holders.pop(rank, seq_id)
            else:
                holders.pop(rank, None)
                producer = seq_id
            holders[rank] = producer
            if key not in self.pending and (not had or not reused):
                self.pending[key] = producer
                self._produced.setdefault(producer, set()).add(edge)
            self._refresh_evictable(edge)
        if path:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            self.seq_hashes[seq_id] = (rank, chain, ordinal)
            self._independent_credit[seq_id] = independent_credit
            key = (
                rank,
                (int(chain.hash_block_size), chain.cache_seed),
                chain.hashes[0],
            )
            self._active.setdefault(key, {})[seq_id] = (chain, ordinal)
        self._trim_history()

    def _remove_request(self, seq_id):
        entry = self.seq_hashes.pop(seq_id, None)
        self._independent_credit.pop(seq_id, None)
        if entry is not None:
            rank, chain, _ordinal = entry
            key = (
                rank,
                (int(chain.hash_block_size), chain.cache_seed),
                chain.hashes[0],
            )
            candidates = self._active[key]
            candidates.pop(seq_id)
            if not candidates:
                del self._active[key]
        return entry

    def _drop_edge(self, edge):
        """Drop a leaf; internal edges remain to locate their descendants."""
        siblings = (
            self._roots[edge.domain] if edge.parent is None else edge.parent.children
        )
        siblings.pop(edge.hashes[0], None)
        if edge.parent is None and not siblings:
            del self._roots[edge.domain]
        self.index.pop(edge)
        self._evictable.pop(edge, None)
        self.indexed_tokens -= len(edge.hashes) * edge.hbs
        if edge.parent is not None:
            self._refresh_evictable(edge.parent)

    def _refresh_evictable(self, edge):
        holders = self.index.get(edge)
        if (
            holders is None
            or edge.children
            or any((edge, rank) in self.pending for rank in holders)
        ):
            self._evictable.pop(edge, None)
        else:
            self._evictable[edge] = None
            self._evictable.move_to_end(edge)

    def _prune_empty(self, edges):
        # Only a released interval and its ancestors can have become empty.
        # Iterate rather than recurse: one-token endpoint extensions can form
        # a deep tree, and other independent histories need no inspection.
        todo = list(edges)
        while todo:
            edge = todo.pop()
            if edge in self.index and not self.index[edge] and not edge.children:
                parent = edge.parent
                self._drop_edge(edge)
                if parent is not None:
                    todo.append(parent)

    def _trim_history(self):
        if self.indexed_tokens <= self.CAP:
            return
        self._over_budget += 1
        # The budget covers unique edges, not replicated rank holders. Pinned
        # active edges may exceed it; releasing them makes them evictable.
        # A small overage shortens the oldest leaf from its tail (its head still
        # names the owner rank) and the leaf then queues behind the others, so
        # repeated small overages spread over many histories. An overage beyond
        # an eighth of the leaf (at least one block), or one that covers the whole
        # leaf, drops it whole, as before: each pass of this loop then removes a
        # whole leaf, so one placement never takes more passes than whole-leaf
        # eviction did. Either way it is only approximately LRU: a recently used
        # leaf can lose a tail share while an older one is still held.
        while self.indexed_tokens > self.CAP and self._evictable:
            edge = next(iter(self._evictable))
            excess = -(-(self.indexed_tokens - self.CAP) // edge.hbs)
            if excess < len(edge.hashes) and excess <= max(1, len(edge.hashes) // 8):
                del edge.hashes[-excess:]
                self.indexed_tokens -= excess * edge.hbs
                self._tail_trim += 1
                self._tail_trim_tokens += excess * edge.hbs
                self._refresh_evictable(edge)  # next victim: another history
            else:
                self._leaf_evict += 1
                self._leaf_evict_tokens += len(edge.hashes) * edge.hbs
                self._drop_edge(edge)
        if self.indexed_tokens > self.CAP:
            self._trim_blocked += 1

    def _left_unprefilled(self, seq_id) -> dict:
        """Transfer an unfinished producer's intervals and credit debt once."""
        charges = {}
        entry = self._remove_request(seq_id)
        if entry is None:
            return charges
        rank, _chain, ordinal = entry
        edges = self._produced.pop(seq_id, ())
        for edge in edges:
            key = (edge, rank)
            if self.pending.get(key) != seq_id:
                continue
            holders = self.index[edge]
            successor = next(
                (
                    sid
                    for sid, (chain, placed_at) in self._active.get(
                        (rank, edge.domain, edge.root_hash), {}
                    ).items()
                    if sid in self._in_flight
                    and placed_at > ordinal
                    and len(chain.hashes) >= edge.end
                    and chain.hashes[edge.end - 1] == edge.hashes[-1]
                ),
                None,
            )
            if successor is not None:
                holders[rank] = successor
                self.pending[key] = successor
                self._produced.setdefault(successor, set()).add(edge)
                credited = self._in_flight.get(successor, 0)
                retained = min(
                    credited,
                    max(
                        edge.start * edge.hbs,
                        self._independent_credit.get(successor, 0),
                    ),
                )
                if retained < credited:
                    self._in_flight[successor] = retained
                    charges[successor] = charges.get(successor, 0) + credited - retained
            else:
                del self.pending[key]
                holders.pop(rank, None)
            self._refresh_evictable(edge)
        self._prune_empty(edges)
        self._trim_history()
        return charges
