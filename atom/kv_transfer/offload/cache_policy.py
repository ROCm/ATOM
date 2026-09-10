# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in scan-resistant eviction for LMCache's CPU tier."""

from collections import OrderedDict
from typing import Any

from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy


class SLRUCachePolicy(LRUCachePolicy):
    """Protect reused chunks; single-use chunks compete in probationary LRU.

    Protection covers at most half the resident chunks. ATOM M3 PAGE chunks
    have a fixed byte size, so this also bounds the protected byte budget.
    The CPU backend owns synchronization and calls these methods under its
    cache lock. A pinned or referenced chunk is never an eviction candidate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._probationary: OrderedDict[Any, None] = OrderedDict()
        self._protected: OrderedDict[Any, None] = OrderedDict()

    def update_on_hit(self, key: Any, cache_dict: OrderedDict) -> None:
        """Promote a reused chunk and age out excess protection."""
        super().update_on_hit(key, cache_dict)
        self._probationary.pop(key, None)
        self._protected[key] = None
        self._protected.move_to_end(key)
        while len(self._protected) > len(cache_dict) // 2:
            demoted, _ = self._protected.popitem(last=False)
            self._probationary[demoted] = None

    def update_on_put(self, key: Any) -> None:
        """Admit new data without displacing protected data's priority."""
        super().update_on_put(key)
        self._protected.pop(key, None)
        self._probationary[key] = None
        self._probationary.move_to_end(key)

    def update_on_force_evict(self, key: Any) -> None:
        """Forget a chunk explicitly removed by the storage backend."""
        self._probationary.pop(key, None)
        self._protected.pop(key, None)

    def get_evict_candidates(
        self, cache_dict: OrderedDict, num_candidates: int = 1
    ) -> list:
        """Choose from probation without scanning the protected segment.

        Normal pressure removals do not issue a policy callback. Drop stale
        keys lazily, but retain selected candidates until they are actually
        removed: a caller may retry or decide not to evict them. With evictable
        probationary entries this is amortized O(1) per victim; pinned entries
        still require scanning because pin changes have no policy callback.
        """
        if num_candidates <= 0:
            return []
        candidates = []
        for queue in (self._probationary, self._protected):
            stale = []
            for key in queue:
                cache = cache_dict.get(key)
                if cache is None:
                    stale.append(key)
                elif cache.can_evict:
                    candidates.append(key)
                    if len(candidates) == num_candidates:
                        break
            for key in stale:
                queue.pop(key, None)
            if len(candidates) == num_candidates:
                break
        return candidates


def register_slru_policy() -> None:
    """Register ATOM_SLRU through LMCache's public policy registry."""
    from lmcache.v1.storage_backend.cache_policy import POLICY_MAPPING

    POLICY_MAPPING["ATOM_SLRU"] = SLRUCachePolicy
