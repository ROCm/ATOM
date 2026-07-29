"""KV-cache manager construction without scheduler/backend coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atom.kv_cache.protocol import KvCacheManager


def make_kv_cache_manager(config: Any) -> "KvCacheManager":
    """Construct the manager selected by the builder-provided pool layout."""
    manager_kind = getattr(config, "kv_manager_kind", None)
    if manager_kind is None:
        # Compatibility for manually-built configs that predate the layout
        # provider. Production Config declares this field explicitly.
        manager_kind = (
            "dsv4"
            if getattr(config, "num_swa_blocks", 0)
            or getattr(config, "v4_arena_group_specs", None)
            else "dense"
        )

    if manager_kind == "dense":
        from atom.kv_cache.dense_manager import DenseKvCacheManager

        return DenseKvCacheManager(config)
    if manager_kind == "dsv4" or str(manager_kind).startswith("dsv4_"):
        from atom.kv_cache.dsv4.manager import Dsv4KvCacheManager

        return Dsv4KvCacheManager(config)
    raise ValueError(f"unknown KV cache manager kind: {manager_kind!r}")
