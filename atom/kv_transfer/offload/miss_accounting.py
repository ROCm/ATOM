# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a prompt's reusable prefix went: the prefill node's miss accounting.

A prefix-cache hit rate alone does not say whether the tokens a request
computed again were new (no tier could have served them) or were computed
before and lost: evicted from HBM and from the tier, or held by the tier but
not loaded (a lookup that did not answer, a hit under the load floor, a load
that failed). Only the second kind is a tier's fault, and only it can be won
back by a bigger or better-managed tier.

:class:`MissAccountant` keeps the store digests of every prompt chunk this
scheduler has seen computed, oldest dropped first past a cap. A chain digest
names its whole prefix, so the chunks of a prompt that were computed before
are a leading run, found by binary search: that run is the request's
*reusable* prefix. At admission each request's reusable tokens are split into

* ``hbm``: served from the HBM prefix cache;
* ``tier``: loaded from the tier;
* ``tier_not_loaded``: the tier held them at the lookup, but they were not
  loaded (under the load floor, or the load failed);
* ``evicted``: neither held them;

and the rest of the prompt is ``new``. That includes a prompt's last chunk
when the prompt ends on a chunk boundary: a prefill computes at least the
prompt's last token, so no source serves that chunk, however often it was
computed before. The tier's part is what its load brings, so a load that
fails moves its tokens to ``tier_not_loaded``.
Counted totals are logged once a minute (``[OFFLOAD-MISS-ACCOUNT]``) and
returned with the scheduler's statistics. Off unless
``ATOM_OFFLOAD_MISS_ACCOUNTING=1``: the digest set costs about 100 bytes a
chunk (``ATOM_OFFLOAD_MISS_ACCOUNTING_CHUNKS``, default 4M chunks).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

logger = logging.getLogger("atom")

_LOG_INTERVAL_S = 60.0


class MissAccountant:
    """Digests of computed prompt chunks, and the reuse totals they classify."""

    CLASSES = ("new", "hbm", "tier", "tier_not_loaded", "evicted")

    def __init__(self, capacity_chunks: int, chunk_tokens: int, label: str) -> None:
        if capacity_chunks <= 0:
            raise ValueError(
                f"miss accounting capacity must be > 0, got {capacity_chunks}"
            )
        self.capacity = int(capacity_chunks)
        self.chunk_tokens = int(chunk_tokens)
        self.label = label
        self._seen: OrderedDict[bytes, None] = OrderedDict()
        self.totals = {name: 0 for name in self.CLASSES}
        self.requests = 0
        # sid -> (tier tokens counted at admission, tier hit at the lookup)
        self._pending_loads: dict[str, tuple[int, int]] = {}
        self._next_log = time.monotonic() + _LOG_INTERVAL_S

    # -- what was computed --------------------------------------------------
    def computed(self, digests) -> None:
        """These chunks' KV now exists (a prefill computed or loaded it)."""
        seen = self._seen
        for digest in digests:
            if digest in seen:
                seen.move_to_end(digest)
            else:
                seen[digest] = None
        while len(seen) > self.capacity:
            seen.popitem(last=False)

    def reusable_chunks(self, digests: list) -> int:
        """The leading run of `digests` computed before. Prefix-closed: a
        chain digest seen implies every earlier one was (up to the cap's
        evictions, which drop the oldest first and so keep the run's head)."""
        lo, hi = 0, len(digests)
        seen = self._seen
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if digests[mid - 1] in seen:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # -- one request ----------------------------------------------------------
    def admitted(
        self,
        sid: str,
        digests: list,
        *,
        prompt_tokens: int,
        servable_tokens: int,
        hbm_tokens: int,
        tier_hit_tokens: int,
        load_tokens: int,
    ) -> None:
        """Classify a request at admission. `servable_tokens` is the most of
        the prompt any source can serve (the prompt less a token, floored to a
        chunk); `tier_hit_tokens` is the lookup's answer counted from the
        prompt's head (0 for none); `load_tokens` the tokens its load will
        bring (0 when it loads nothing)."""
        chunk = self.chunk_tokens
        reusable = min(
            self.reusable_chunks(digests) * chunk,
            max(0, int(servable_tokens)),
            int(prompt_tokens),
        )
        hbm = min(max(0, int(hbm_tokens)), reusable)
        tier = min(max(0, int(load_tokens)), max(0, reusable - hbm))
        held = min(max(0, int(tier_hit_tokens)), reusable)
        not_loaded = max(0, held - hbm - tier)
        evicted = max(0, reusable - hbm - tier - not_loaded)
        totals = self.totals
        totals["new"] += max(0, int(prompt_tokens) - reusable)
        totals["hbm"] += hbm
        totals["tier"] += tier
        totals["tier_not_loaded"] += not_loaded
        totals["evicted"] += evicted
        self.requests += 1
        if tier:
            self._pending_loads[sid] = (tier, held)
        self._maybe_log()

    def load_failed(self, sid: str) -> None:
        """A load counted at admission did not land: its tokens were computed."""
        entry = self._pending_loads.pop(sid, None)
        if entry is None:
            return
        tier, _held = entry
        self.totals["tier"] -= tier
        self.totals["tier_not_loaded"] += tier

    def load_finished(self, sid: str) -> None:
        self._pending_loads.pop(sid, None)

    def forget(self, sid: str) -> None:
        self._pending_loads.pop(sid, None)

    # -- reporting ------------------------------------------------------------
    def statistics(self) -> dict[str, int]:
        return {f"miss_{name}_tokens": value for name, value in self.totals.items()}

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + _LOG_INTERVAL_S
        totals = self.totals
        prompt = sum(totals.values()) or 1
        reusable = prompt - totals["new"] or 1
        logger.info(
            "[OFFLOAD-MISS-ACCOUNT] %s requests=%d prompt_tokens=%d "
            "reusable=%.2f%% of prompt: hbm=%.2f%% tier=%.2f%% "
            "tier_not_loaded=%.2f%% evicted=%.2f%% (of reusable) "
            "seen_chunks=%d",
            self.label,
            self.requests,
            prompt,
            100.0 * (prompt - totals["new"]) / prompt,
            100.0 * totals["hbm"] / reusable,
            100.0 * totals["tier"] / reusable,
            100.0 * totals["tier_not_loaded"] / reusable,
            100.0 * totals["evicted"] / reusable,
            len(self._seen),
        )


__all__ = ["MissAccountant"]
