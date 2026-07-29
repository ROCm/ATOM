"""DSV4 primary compressed-cache logical ID pool."""

from atom.kv_cache.pools.pooled_free_list import PooledFreeList


class Dsv4CompressedPool(PooledFreeList):
    """Backed/unbacked logical IDs coordinated with the DSV4 chunk arena.

    Hashing, events, and sibling eviction intentionally stay in the manager;
    this class only names the shared ``PooledFreeList`` mechanism for DSV4.
    """
