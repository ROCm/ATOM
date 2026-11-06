# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Store-aware least-tokens DP routing (`--dp-load-balance store_tokens`).

With an offload tier every DP rank shares (Mooncake Store), a prefix one rank
computed can be loaded by any rank once its prefill completed. So a request's
prefill work does not depend on where its prefix was computed, and the router
can place every request purely by work: the rank with the least prefill left
(plus `req_equiv` per in-flight request, as `least_tokens`), charging the
request only the tokens it will actually compute. `CoreManager` passes each
rank's own report of the prefill it has left (`PREFILL_BACKLOG`) plus the
charges of requests sent after that report, not routing-time estimates.

The prefix index is rank-agnostic: sampled prompt-block hash -> (rank, request
that computes it). A prefix whose producer finished prefilling is in the tier:
every rank is charged ``prompt - hit`` (a rest shorter than the connector's
load threshold is recomputed, so it is charged on ranks other than the
producer's). A prefix still being prefilled is only reusable on its producer's
rank; elsewhere the request is charged the whole prompt.

Same lifecycle interface as `PrefixSpreadRouter`, so `CoreManager` drives both
the same way.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from atom.kv_transfer.disaggregation.factory import OffloadTierProfile
from atom.model_engine.prefix_spread import Route


class StoreAwareRouter:
    """Rank-agnostic prefix index and the work-based placement rule."""

    STRIDE_TOKENS = 1024
    CAP = 1 << 19

    DECISIONS = ("store_hit_total", "store_source_rank_total", "store_cold_total")

    def __init__(self, tier: OffloadTierProfile, req_equiv: int) -> None:
        self.tier = tier
        self.req_equiv = req_equiv
        # hash -> (rank, producer seq id); most recent last.
        self.index: OrderedDict[int, tuple[int, Any]] = OrderedDict()
        # producer seq id -> hashes it produced (dropped if it leaves unprefilled).
        self.produced: dict[Any, list[int]] = {}
        self._prefilling: set = set()
        self._est_hit: dict = {}
        # Rotating tie-break, so fully tied ranks do not all go to rank 0.
        self._cursor = 0

    def reset(self) -> None:
        self.produced.clear()
        self._prefilling.clear()
        self._est_hit.clear()

    # -- request lifecycle -----------------------------------------------------

    def charge(self, seq_id) -> int:
        self._prefilling.add(seq_id)
        return self._est_hit.pop(seq_id, 0)

    def prefilling(self, seq_id) -> bool:
        return seq_id in self._prefilling

    def prefill_complete(self, seq_id) -> None:
        self._prefilling.discard(seq_id)
        self.produced.pop(seq_id, None)

    def release(self, seq_id) -> None:
        self._est_hit.pop(seq_id, None)
        if seq_id not in self._prefilling:
            self.prefill_complete(seq_id)
            return
        self._prefilling.discard(seq_id)
        # Left before its prefill completed: what it was producing never
        # reached HBM or the tier.
        for h in self.produced.pop(seq_id, ()):
            entry = self.index.get(h)
            if entry is not None and entry[1] == seq_id:
                del self.index[h]

    # -- placement ---------------------------------------------------------------

    def select(
        self,
        seq: Any,
        rank_tokens: list[int],
        rank_reqs: list[int],
        rank_load: Callable[[int], int],
        lightest: Callable[[], int],
    ) -> Route:
        chain = getattr(seq, "prompt_hash_chain", None)
        prompt = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        hit, src_rank, src_seq = self._longest_hit(chain)
        n = len(rank_tokens)
        in_tier = hit > 0 and (
            self.tier.loads
            and src_seq not in self._prefilling
            and hit >= self.tier.min_load_tokens
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
                rank_tokens[r] + cost(r) + self.req_equiv * rank_reqs[r],
                r != src_rank,
                (r - cursor) % n,
            ),
        )
        est = prompt - cost(best)
        self._est_hit[seq.id] = est
        self._record(chain, best, seq.id, reused=est)
        if hit and best == src_rank:
            return Route(best, "store_source_rank_total")
        return Route(best, "store_hit_total" if est else "store_cold_total")

    # -- prefix index ------------------------------------------------------------

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
        for i in range(stride - 1, len(chain.hashes), stride):
            h = chain.hashes[i]
            if (i + 1) * hbs <= reused and h in index:
                index.move_to_end(h)
                continue
            index.pop(h, None)
            index[h] = (rank, seq_id)
            mine.append(h)
        if mine:
            self.produced[seq_id] = mine
        while len(index) > self.CAP:
            index.popitem(last=False)
