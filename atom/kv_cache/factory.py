"""KV-cache manager construction without scheduler/backend coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atom.kv_cache.protocol import KvCacheManager


def make_kv_cache_manager(config: Any) -> "KvCacheManager":
    """Construct the current manager implementation.

    Dense/DSV4 dispatch is added once the implementations are split; keeping the
    import local already removes Scheduler's dependency on the compatibility
    ``model_engine.block_manager`` module.
    """
    from atom.model_engine.block_manager import BlockManager

    return BlockManager(config)
