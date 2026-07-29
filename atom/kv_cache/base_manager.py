"""Shared primary-cache lifecycle used by concrete KV managers."""

from atom.model_engine.block_manager import BlockManager as _PrimaryCacheManagerCore


class BaseKvCacheManager(_PrimaryCacheManagerCore):
    """Chained hash, primary blocks, per-request slots, and KV events.

    The compatibility core remains importable at its historical path while
    concrete managers establish the new factory boundary.
    """
