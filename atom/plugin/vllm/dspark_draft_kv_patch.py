# SPDX-License-Identifier: Apache-2.0
"""Keep the DSpark draft's sliding-window KV group out of the prefix cache.

Under ``--speculative-config '{"method":"dspark"}'`` there are two KV groups:
ATOM's V4 proxy (``FullAttentionSpec``, block 128) and the draft's three MLA
layers (``SlidingWindowMLASpec``, block 64, window 128).
``find_longest_cache_hit`` reconciles groups to the minimum and
``remove_skipped_blocks`` frees draft blocks while the previous request is still
decoding, so the draft caps the whole request's hit at whatever stale run
survived eviction (17.4% observed, against 84.9% for ATOM native).

Skipping the lookup is safe: ``_roll_back_prefix_hit`` re-forwards the last
512 tokens of every hit and the draft reads only the last 128.

Registered through vLLM's ``@register_kv_cache_spec`` seam rather than a
coordinator monkeypatch, so page size, memory accounting and allocation are
inherited unchanged. The hit-rate recovery is not yet measured on hardware.
"""

import dataclasses
import logging

logger = logging.getLogger("atom")

_spec_cls = None
_registered = False
# Cached so a rebuild cannot hand out a second, unequal type that pickle would
# then resolve inconsistently.
_built = None


def _build_spec_and_manager():
    """Define the spec/manager pair lazily: importing
    ``vllm.v1.core.single_type_kv_cache_manager`` at module import time drags in
    vLLM's scheduler core before the plugin has registered models.
    """
    global _built
    if _built is not None:
        return _built

    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class DSparkDraftSWAMLASpec(SlidingWindowMLASpec):
        """No added fields, so sizing is identical to ``SlidingWindowMLASpec``.
        Exists only so the registry can route it to a different manager."""

    class DSparkDraftSWAManager(SlidingWindowManager):
        """Sliding-window manager that abstains from the prefix cache. Only the
        two prefix-cache entry points are overridden; allocation and
        ``remove_skipped_blocks`` recycling stay on the inherited path."""

        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length: int,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block: bool,
            alignment_tokens: int,
            dcp_world_size: int = 1,
            pcp_world_size: int = 1,
        ):
            """Return the candidate hit unchanged, backed by null blocks.

            The reconciled hit is the minimum over groups, so returning the
            candidate drops this group out of it. ``null_block`` declares the
            tokens computed without claiming storage, and a leading null run is
            the normal sliding-window shape. ``drop_eagle_block`` must still be
            honoured: the coordinator passes ``candidate + block_size`` for it.
            """
            block_size = kv_cache_spec.block_size
            hit_length = max_length
            if drop_eagle_block:
                hit_length = max(0, hit_length - block_size)
            # The coordinator only accepts alignment-aligned hit lengths.
            if alignment_tokens:
                hit_length -= hit_length % alignment_tokens
            hit_length -= hit_length % block_size
            num_blocks = hit_length // block_size
            computed_blocks = tuple(
                [block_pool.null_block] * num_blocks for _ in kv_cache_group_ids
            )
            return computed_blocks, hit_length

        def cache_blocks(self, request, num_tokens, retention_interval=None) -> None:
            """Publish nothing: the lookup above never consults the hash map."""
            return

    # The KVCacheConfig holding these specs is pickled to every worker, and
    # pickle resolves a class by __module__ + __qualname__ -- a `<locals>`
    # qualname resolves nowhere. Rewrite both onto this module; `__getattr__`
    # below rebuilds the pair for workers that never registered.
    for cls in (DSparkDraftSWAMLASpec, DSparkDraftSWAManager):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__

    _built = (DSparkDraftSWAMLASpec, DSparkDraftSWAManager)
    return _built


def __getattr__(name):
    """Resolve the lazily-built classes by name, for pickle (PEP 562)."""
    if name in ("DSparkDraftSWAMLASpec", "DSparkDraftSWAManager"):
        spec_cls, manager_cls = _built or _build_spec_and_manager()
        return {
            "DSparkDraftSWAMLASpec": spec_cls,
            "DSparkDraftSWAManager": manager_cls,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_registered():
    """Register the draft spec/manager pair, once, after vLLM's built-ins.

    ``_ensure_registered`` populates the built-ins only ``if not
    _REGISTRY_KVCACHESPEC_LIST``, so registering into an empty registry would
    short-circuit that pass and leave every built-in unregistered.

    Not hung off ``ATOMPlatform.register_custom_kv_cache_specs`` because
    ATOMPlatform is often not the live platform -- see
    ``platform.install_platform_config_hook``.
    """
    global _spec_cls, _registered
    if _registered:
        return _spec_cls

    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry._ensure_registered()

    spec_cls, manager_cls = _build_spec_and_manager()
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=spec_cls,
        manager_class=manager_cls,
        # Stay grouping-compatible so `is_uniform_type` still accepts the three
        # draft layers as one group.
        uniform_type_base_spec=SlidingWindowMLASpec,
    )
    _spec_cls = spec_cls
    _registered = True
    return spec_cls


def convert_draft_specs(draft_specs):
    """Retype the draft's sliding-window specs, leaving field values intact.

    Returns the input unchanged for a non-DSpark draft, or if the conversion
    fails -- a prefix-cache optimisation must never stop a server starting.
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    if not draft_specs:
        return draft_specs
    if not all(type(spec) is SlidingWindowMLASpec for spec in draft_specs.values()):
        return draft_specs
    try:
        spec_cls = ensure_registered()
        converted = {}
        for name, spec in draft_specs.items():
            # Init fields only: `__post_init__` recomputes `page_size_padded`,
            # so the round-trip is idempotent.
            kwargs = {
                f.name: getattr(spec, f.name)
                for f in dataclasses.fields(spec)
                if f.init
            }
            converted[name] = spec_cls(**kwargs)
    except Exception:
        logger.warning(
            "ATOM DeepSeek-V4: could not retype the DSpark draft SWA specs; "
            "falling back to vLLM's stock sliding-window manager. The draft "
            "group will cap the target's prefix-cache hit.",
            exc_info=True,
        )
        return draft_specs

    logger.info(
        "ATOM DeepSeek-V4: DSpark draft SWA group (%d layers) excluded from "
        "prefix-cache hit reconciliation; its window is rebuilt by the SWA "
        "recompute.",
        len(converted),
    )
    return converted
