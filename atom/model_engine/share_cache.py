# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared-cache DP routing (`--dp-load-balance share_cache`).

With an offload tier every DP rank shares (Mooncake Store, one LMCache MP
server), a prefix one rank computed can be loaded by any rank once its prefill
completed. A request's prefill work then does not depend on where its prefix
was computed, so it can go wherever there is the least prefill left.

Placement: the rank with the lowest ``backlog + cost + req_equiv * reqs``,
where ``backlog`` is the prefill the rank itself reports it has left
(`EngineCoreRequestType.PREFILL_BACKLOG`, plus the charges of requests sent
after that report; `CoreManager` builds it) and ``cost`` the tokens this
request will compute there.

``cost`` comes from a rank-agnostic prefix index: sampled prompt-block hash ->
(rank, request computing it). A prefix whose request finished prefilling is in
the tier, so every rank is charged ``prompt - hit`` (a hit under the
connector's load threshold is recomputed, so only the producer's rank gets it).
A prefix still being prefilled is reusable only on its producer's rank.

The index is a hint: a prefix it lists can be gone from the tier (a save still
in flight or failed, an eviction) or from the producer's HBM. The rank finds
out when it admits the request, and reports how much it really reused
(`note_reuse`); the entries past that are dropped and the request, which
computes them, becomes their producer.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, ClassVar, NamedTuple

# Offload connectors whose tier all DP ranks of one engine share.
SHARED_TIER_CONNECTORS = frozenset({"mooncake_store", "lmcache_mp"})


class StoreRoute(NamedTuple):
    rank: int
    decision: str


class ShareCacheRouter:
    """Rank-agnostic prefix index and the work-based placement rule."""

    STRIDE_TOKENS = 1024
    CAP = 1 << 19

    # Routing decision counters and what each one counts; exported by
    # `atom/metrics/exporter.py` as `atom:dp_<name without _total>`.
    DECISIONS: ClassVar[dict[str, str]] = {
        "share_cache_hit_total": "Requests placed on a prefix the shared tier holds.",
        "share_cache_source_rank_total": (
            "Requests placed on the rank still prefilling their prefix."
        ),
        "share_cache_cold_total": "Requests placed without a known prefix.",
        "share_cache_stale_total": (
            "Placements whose rank reused less of the prefix than the index "
            "listed; the stale entries were dropped."
        ),
    }

    def __init__(self, min_load_tokens: int, req_equiv: int) -> None:
        self.min_load_tokens = int(min_load_tokens)
        self.req_equiv = int(req_equiv)
        # hash -> (rank, producer seq id); most recent last.
        self.index: OrderedDict[int, tuple[int, Any]] = OrderedDict()
        # producer seq id -> hashes it produced (dropped if it leaves unprefilled).
        self.produced: dict[Any, list[int]] = {}
        # A newcomer computing a prefix whose owner is still prefilling it does
        # not take the entry over: hash -> the (rank, seq id) of every such
        # request, in arrival order; the first one left carries the entry if
        # the owner leaves unprefilled (`_backing`: each one's hashes).
        self.backups: dict[int, list[tuple[int, Any]]] = {}
        self._backing: dict[Any, list[int]] = {}
        self._prefilling: set = set()
        self._est_hit: dict = {}
        # seq id -> (prompt hash chain, estimated reuse, rank) of a placement
        # that counted on a prefix, until its rank reports the reuse it found.
        self._placed: dict = {}
        # Rotating tie-break, so fully tied ranks do not all go to rank 0.
        self._cursor = 0

    def reset(self) -> None:
        """Forget in-flight state; the index describes caches that outlive it."""
        self.produced.clear()
        self.backups.clear()
        self._backing.clear()
        self._prefilling.clear()
        self._est_hit.clear()
        self._placed.clear()

    # -- request lifecycle (CoreManager calls these under its lock) -----------

    def charge(self, seq_id) -> int:
        """Mark `seq_id` in flight; returns the prefix it was credited."""
        self._prefilling.add(seq_id)
        return self._est_hit.pop(seq_id, 0)

    def prefilling(self, seq_id) -> bool:
        return seq_id in self._prefilling

    def prefill_complete(self, seq_id) -> None:
        """`seq_id` completed its prefill: what it produced is in the tier."""
        self._prefilling.discard(seq_id)
        self.produced.pop(seq_id, None)
        # What it computed beside an owner still prefilling is in the tier now
        # too: it carries the entry, so the prefix counts as loadable.
        for h in self._backing.pop(seq_id, ()):
            backup = self._drop_backup(h, seq_id)
            if backup is None:
                continue
            owner = self.index.get(h)
            if owner is not None and owner[1] in self._prefilling:
                self.index[h] = backup

    def release(self, seq_id) -> None:
        """`seq_id` left (finished, aborted, rejected, failed send)."""
        self._est_hit.pop(seq_id, None)
        self._placed.pop(seq_id, None)
        if seq_id not in self._prefilling:
            self.prefill_complete(seq_id)
            return
        self._prefilling.discard(seq_id)
        # Left before its prefill completed: what it was producing never
        # reached HBM or the tier. A newcomer computing the same prefix carries
        # the entry on.
        for h in self.produced.pop(seq_id, ()):
            entry = self.index.get(h)
            if entry is not None and entry[1] == seq_id:
                waiting = self.backups.get(h)
                backup = waiting.pop(0) if waiting else None
                if waiting is not None and not waiting:
                    del self.backups[h]
                if backup is not None:
                    self.index[h] = backup
                    # The promoted request now produces this prefix: if it too
                    # leaves unprefilled, its release must drop the entry.
                    self.produced.setdefault(backup[1], []).append(h)
                    carried = self._backing.get(backup[1])
                    if carried is not None and h in carried:
                        carried.remove(h)
                else:
                    del self.index[h]
        for h in self._backing.pop(seq_id, ()):
            self._drop_backup(h, seq_id)

    def _drop_backup(self, h: int, seq_id):
        """Remove `seq_id` from the backups of `h`; its (rank, seq id), or None."""
        waiting = self.backups.get(h)
        if not waiting:
            return None
        backup = next((b for b in waiting if b[1] == seq_id), None)
        if backup is not None:
            waiting.remove(backup)
            if not waiting:
                del self.backups[h]
        return backup

    # -- placement -------------------------------------------------------------

    def select(
        self, seq: Any, rank_backlog: list[int], rank_reqs: list[int]
    ) -> StoreRoute:
        chain = getattr(seq, "prompt_hash_chain", None)
        prompt = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        hit, src_rank, src_seq = self._longest_hit(chain)
        n = len(rank_backlog)
        in_tier = (
            hit > 0 and src_seq not in self._prefilling and hit >= self.min_load_tokens
        )

        def cost(r: int) -> int:
            if hit and (r == src_rank or in_tier):
                return prompt - hit
            return prompt

        cursor = self._cursor
        self._cursor += 1
        best = min(
            range(n),
            key=lambda r: (
                rank_backlog[r] + cost(r) + self.req_equiv * rank_reqs[r],
                r != src_rank,
                (r - cursor) % n,
            ),
        )
        est = prompt - cost(best)
        self._est_hit[seq.id] = est
        if est:
            self._placed[seq.id] = (chain, est, best)
        self._record(chain, best, seq.id, reused=est)
        if hit and best == src_rank:
            return StoreRoute(best, "share_cache_source_rank_total")
        return StoreRoute(
            best, "share_cache_hit_total" if est else "share_cache_cold_total"
        )

    def note_reuse(self, seq_id, reused: int) -> bool:
        """`seq_id`'s rank admitted it and reused `reused` prompt tokens (HBM
        hit plus tier load). Entries of the prefix it was placed on that lie
        past that are stale: drop them, and let `seq_id`, which computes that
        part, produce it. An entry whose producer is still prefilling is left
        alone (its prefix is on its way). Returns whether any was dropped."""
        placed = self._placed.pop(seq_id, None)
        if placed is None:
            return False
        chain, est, rank = placed
        hbs = int(chain.hash_block_size)
        # The resume point can sit one hash block under the reused prefix
        # (replay); that is not a miss.
        if reused + hbs >= est:
            return False
        stride = max(1, self.STRIDE_TOKENS // hbs)
        producing = seq_id in self._prefilling
        dropped = False
        for i in range(stride - 1, len(chain.hashes), stride):
            end = (i + 1) * hbs
            if end > est:
                break
            if end <= reused + hbs:
                continue
            h = chain.hashes[i]
            entry = self.index.get(h)
            if entry is None or entry[1] == seq_id or entry[1] in self._prefilling:
                continue
            dropped = True
            if producing:
                self.index[h] = (rank, seq_id)
                self.index.move_to_end(h)
                self.produced.setdefault(seq_id, []).append(h)
            else:
                del self.index[h]
        return dropped

    # -- prefix index ----------------------------------------------------------

    def _longest_hit(self, chain):
        if chain is None or not self.index:
            return 0, None, None
        hbs = int(chain.hash_block_size)
        stride = max(1, self.STRIDE_TOKENS // hbs)
        hashes = chain.hashes
        for i in range((len(hashes) // stride) * stride - 1, -1, -stride):
            entry = self.index.get(hashes[i])
            if entry is not None:
                return (i + 1) * hbs, entry[0], entry[1]
        return 0, None, None

    def _record(self, chain, rank: int, seq_id, reused: int = 0) -> None:
        """Index this prompt's sampled prefix. The part it reuses keeps its
        producer; the part it computes is produced by `seq_id` on `rank`."""
        if chain is None:
            return
        hbs = int(chain.hash_block_size)
        stride = max(1, self.STRIDE_TOKENS // hbs)
        index = self.index
        mine = []
        backing = []
        for i in range(stride - 1, len(chain.hashes), stride):
            h = chain.hashes[i]
            owner = index.get(h)
            if (i + 1) * hbs <= reused and owner is not None:
                index.move_to_end(h)
                if owner[1] != seq_id and owner[1] in self._prefilling:
                    # Reused from a producer still prefilling it: if that one
                    # leaves unprefilled, this request computes the prefix.
                    self.backups.setdefault(h, []).append((rank, seq_id))
                    backing.append(h)
                continue
            if (
                owner is not None
                and owner[1] != seq_id
                and owner[1] in self._prefilling
            ):
                # The owner is still prefilling it, on another rank. Taking the
                # entry over would lose the prefix if this request left first
                # (its release drops what it produced); rely on the owner and
                # stand by to carry the entry.
                self.backups.setdefault(h, []).append((rank, seq_id))
                backing.append(h)
                continue
            index.pop(h, None)
            index[h] = (rank, seq_id)
            mine.append(h)
        if mine:
            self.produced[seq_id] = mine
        if backing:
            self._backing[seq_id] = backing
        while len(index) > self.CAP:
            index.popitem(last=False)
