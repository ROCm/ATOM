"""Log what each kv-cache group offers as a prefix hit, and what wins.

Diagnostic only, env-gated (`ATOM_V41_HIT_PROBE=1`), log-only: it wraps
vLLM's hit finders without changing a value.

It exists because a measured hit of 1024 reached the model while the only
state checkpoint sat at 1280, and the coordinator's own algorithm is a
fixed point over the MINIMUM across groups -- so the state group either did
not take part, or answered 1024 for a boundary nothing was written to.
Those two have opposite fixes, and nothing in the logs told them apart.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_CALLS = [0]


def enabled() -> bool:
    return os.environ.get("ATOM_V41_HIT_PROBE") in ("1", "true", "True")


def install() -> bool:
    """Wrap the per-group finder and the coordinator. Idempotent, log-only."""
    global _INSTALLED
    if _INSTALLED or not enabled():
        return False
    try:
        from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
        from vllm.v1.core.single_type_kv_cache_manager import (
            MambaManager,
            SingleTypeKVCacheManager,
        )
    except Exception:  # noqa: BLE001 - a probe must never take the engine down
        logger.warning("V4.1 hit probe: vLLM internals not where expected")
        return False

    for cls in (MambaManager, SingleTypeKVCacheManager):
        original = cls.__dict__.get("find_longest_cache_hit")
        if original is None or getattr(original, "_atom_probed", False):
            continue
        inner = original.__func__ if hasattr(original, "__func__") else original

        def wrapped(c, *a, _inner=inner, _name=cls.__name__, **kw):
            blocks, hit = _inner(c, *a, **kw)
            spec = kw.get("kv_cache_spec")
            logger.info(
                "V4.1 hit probe: %s(%s) groups=%s max=%s -> hit=%s",
                _name,
                type(spec).__name__,
                kw.get("kv_cache_group_ids"),
                kw.get("max_length"),
                hit,
            )
            return blocks, hit

        wrapped._atom_probed = True
        cls.find_longest_cache_hit = classmethod(wrapped)

    orig = HybridKVCacheCoordinator.find_longest_cache_hit
    if not getattr(orig, "_atom_probed", False):

        def coord(self, *a, _orig=orig, **kw):
            out = _orig(self, *a, **kw)
            _CALLS[0] += 1
            if _CALLS[0] <= 40:
                groups = [type(g.spec).__name__ for g in self.attention_groups]
                logger.info(
                    "V4.1 hit probe: coordinator #%d groups=%s -> combined=%s",
                    _CALLS[0],
                    groups,
                    out[1],
                )
            return out

        coord._atom_probed = True
        HybridKVCacheCoordinator.find_longest_cache_hit = coord

    _INSTALLED = True
    logger.info("V4.1 hit probe installed")
    return True
