# SPDX-License-Identifier: MIT
"""Reserve the CSA2 per-request STATE tail out of vLLM's proxy block pool.

DeepSeek-V4.1 buys its pool in two currencies. The PAGE currency scales with
cached history -- one PAGE of main latent per owner plus that PAGE's index
rows, ``V41PoolGeometry.paged_bytes`` together -- and the STATE currency does
not: every in-flight request owns one fixed-size entry holding its sliding
window ring, the compressor's own rings and its Engram cursor
(``V41PoolGeometry.state_bytes``). The two sit in one contiguous region,
PAGEs first, because ``V41PoolGeometry.window`` states a slot's ring as an
offset past the absolute end of the PAGE currency and the ``_indices`` kernels
dereference page rows and window rows against one base.

vLLM only knows how to buy the first currency. It sizes a pool as
``num_blocks`` uniform blocks of ``page_size_bytes`` and hands every one of
them to its ``BlockPool`` to allocate. ATOM declares the proxy layer's
``page_size_bytes`` as exactly one V4.1 PAGE, which makes block id and PAGE id
the same id -- the property the metadata builder relies on to pass vLLM's
block table straight through as a PAGE table -- and leaves the STATE entries
with nowhere to live.

This patch buys them out of the same allocation. ``get_kv_cache_configs``
decides both how large the tensors are and how many blocks the scheduler may
hand out; those two numbers are independent once the tensors are allocated,
because the worker re-derives its own block count from the tensor it got
(``_reshape_kv_cache_tensors``: ``raw_tensor.numel() // page_size_bytes``).
So reduce ``num_blocks`` by the STATE tail's worth of pages and leave
``kv_cache_tensors[].size`` alone: the profiled memory budget is unchanged,
the arena the worker maps still spans every byte that was bought, and vLLM's
``BlockPool`` simply never issues an id inside the tail. The bind then carves
PAGEs from the head and STATE from the tail it was never asked about.

The tail costs ``max_num_seqs`` STATE entries -- about 900 pages of a BF16
pool and 1,300 of an FP4 one at the shipped 256-token PAGE, against the tens
of thousands a served pool holds. Amortizing STATE into every page instead,
the way the DeepSeek-V4 proxy does, would inflate ``page_size_bytes`` by the
ratio of a whole pool's STATE to a minimum-length request's PAGEs -- roughly
sixfold here, because a V4.1 STATE entry is several PAGEs on its own.

Why ``get_kv_cache_configs`` and not the ``get_kv_cache_config_from_groups``
underneath it: that function's callers finish by shrinking every worker's
config to the smallest block count any worker reported, and that loop asserts
``tensor.size % num_blocks_old == 0`` before rescaling the tensors. A block
count already reduced by a tail that is not a divisor of the tensor trips the
assert. Wrapping the outer call puts the reduction after that arithmetic.

Installed from ``register_model`` -- the ``vllm.general_plugins`` hook --
which vLLM runs in the EngineCore process that later builds the KV cache
config, and early enough that ``get_kv_cache_configs`` has not been called.
``ATOMPlatform.check_and_update_config`` installs it too, but cannot be relied
on: vLLM resolves its platform class from inside ``import vllm`` itself, so
``register_platform`` runs against a partially initialized package and the
loader swallows whatever it raises, leaving the stock ROCm platform active.
The install is idempotent, so both call sites is fine.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger("atom")

# vLLM resolves ``get_kv_cache_configs`` by module attribute at call time, so
# every module that imported the name needs rebinding. ``v1.engine.core`` is
# the only importer in 0.28; ``v1.core.kv_cache_utils`` is where it is defined.
_PATCH_TARGETS = (
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.engine.core",
)

_ATTR = "get_kv_cache_configs"

# The other half of the same reservation. `get_kv_cache_configs` decides
# `max_model_len` for `--max-model-len -1` by binary search against the memory
# it is about to buy, and it does that *before* this patch takes the tail out.
# The fitted length therefore fills the whole pool, the tail comes out of a
# pool already spoken for, and startup fails over a length vLLM chose itself.
# vLLM has the shape for this already: it subtracts the null block from the
# memory auto-fit and the capacity check plan against, while allocation still
# uses the full amount. The STATE tail is subtracted the same way.
_AUTOFIT_ATTR = "_auto_fit_max_model_len"
_POOL_BYTES_ATTR = "_pool_bytes_per_block"

# `BlockPool.__init__` pops one block off the free queue to serve as its null
# placeholder and never gives it back, so this many of any pool's block ids
# are not allocatable.
NULL_BLOCKS = 1


def deepseek_v41_state_reserve_blocks(vllm_config) -> int:
    """Blocks to withhold from vLLM's pool so the CSA2 STATE tail fits.

    Zero for any model that is not DeepSeek-V4.1 on the plugin path, which is
    what makes the patch safe to install unconditionally for the process.
    """
    from atom.plugin.vllm.deepseek_v41_bridge import (
        v41_proxy_state_reserve_blocks,
    )

    return v41_proxy_state_reserve_blocks(vllm_config)


def _how_to_fit(vllm_config) -> str:
    """What would actually make this pool bigger, for the config in hand.

    "Raise --gpu-memory-utilization" is the general answer and the wrong one
    under `--num-gpu-blocks-override`, which pins the block count and ignores
    the utilization entirely: a reader who follows it changes a number and
    sees the identical failure.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    override = getattr(cache_config, "num_gpu_blocks_override", None)
    if override:
        return (
            f"The pool size is pinned by --num-gpu-blocks-override={override}, "
            "so --gpu-memory-utilization will not change it: raise the override, "
            "or lower --max-num-seqs or --max-model-len."
        )
    return "Raise --gpu-memory-utilization, or lower --max-num-seqs or --max-model-len."


def _reserve_state_tail(configs, reserve: int, vllm_config) -> None:
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_BLOCK_SIZE,
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )

    max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)

    # One request must still fit after the tail comes out, or the scheduler
    # can never admit anything -- a clearer failure here than a request that
    # is preempted forever.
    #
    # `+ NULL_BLOCKS`: vLLM's `BlockPool` permanently pops one block off its
    # free queue as the null placeholder, so a pool of N block ids has N-1
    # allocatable. Counting totals made the boundary off by exactly one, and
    # in the direction that passes: with max_model_len=8192 and 32 PAGEs left,
    # this said yes while an 8191-token prompt -- which needs all 32 -- could
    # only ever be given 31.
    #
    # Counted over every group in the pool, not just the proxy's. With DSpark
    # the draft layers are a second KV-cache group drawing on the same
    # BlockPool, so one max-length request needs that group's blocks as well.
    # Against the proxy group alone, `usable` could land between the two
    # figures: startup passes, and the request it was sized for can never be
    # admitted.
    def _blocks_one_request_needs(cfg) -> int:
        total = 0
        for group in cfg.kv_cache_groups:
            # The spec itself is optional, not just its block size: a group is
            # reached here from vLLM's own configs and from tests that describe
            # one by its layer names alone.
            spec = getattr(group, "kv_cache_spec", None)
            block_size = int(
                getattr(spec, "block_size", 0) or ATOM_DEEPSEEK_V41_BLOCK_SIZE
            )
            total += max(1, -(-max_model_len // block_size))
        return total

    applied = False
    seen_groups = []
    for cfg in configs:
        for group in cfg.kv_cache_groups:
            seen_groups.extend(group.layer_names)
        owns_proxy = any(
            ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME in group.layer_names
            for group in cfg.kv_cache_groups
        )
        if not owns_proxy:
            continue
        applied = True
        per_request = _blocks_one_request_needs(cfg)
        min_usable = per_request + NULL_BLOCKS
        usable = cfg.num_blocks - reserve
        if usable < min_usable:
            raise ValueError(
                "DeepSeek-V4.1 plugin mode cannot fit its per-request state: the "
                f"KV pool holds {cfg.num_blocks} PAGEs, of which {reserve} are "
                "reserved for the sliding-window / compressor / Engram state of "
                f"{vllm_config.scheduler_config.max_num_seqs} concurrent requests, "
                f"leaving {usable} for history -- below the "
                f"{min_usable} a single max_model_len={max_model_len} request "
                f"needs ({per_request} across this pool's "
                f"{len(cfg.kv_cache_groups)} KV-cache group(s) plus "
                f"{NULL_BLOCKS} vLLM holds back as its null block). "
                + _how_to_fit(vllm_config)
            )
        logger.info(
            "DeepSeek-V4.1 plugin: reserving %d of %d proxy PAGEs for the CSA2 "
            "STATE tail (%d schedulable)",
            reserve,
            cfg.num_blocks,
            usable,
        )
        cfg.num_blocks = usable

    if not applied:
        # A non-zero reserve with nothing to apply it to is a miss, not a
        # no-op: the pool then gets built at full size and the bind dies
        # later with a message blaming this patch for never running. Name it
        # here, where the layer names that failed to match are still in hand.
        raise RuntimeError(
            f"DeepSeek-V4.1 plugin: computed a {reserve}-block STATE tail "
            f"reserve but no KV-cache config owns "
            f"{ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME!r}. Groups seen: "
            f"{sorted(set(seen_groups))}"
        )


def _patch_auto_fit(module) -> bool:
    """Let ``--max-model-len -1`` fit against the pool minus the STATE tail.

    Returns whether it installed here. A module without the symbol is not an
    error: the length is only auto-fitted on versions that have it.
    """
    original = getattr(module, _AUTOFIT_ATTR, None)
    if original is None or getattr(original, "_atom_v41_autofit_patched", False):
        return False

    @functools.wraps(original)
    def patched(vllm_config, projected_groups_per_worker, available_memory, **kwargs):
        reserve = deepseek_v41_state_reserve_blocks(vllm_config)
        if reserve:
            # Read at call time: `spec_decode_patch` replaces this for a pool
            # whose groups are heterogeneous, and the tail has to be priced in
            # the same block bytes the rest of the sizing uses.
            pool_bytes = getattr(module, _POOL_BYTES_ATTR, None)
            if pool_bytes is not None:
                available_memory = [
                    max(0, memory - reserve * pool_bytes(groups)) if groups else memory
                    for groups, memory in zip(
                        projected_groups_per_worker, available_memory
                    )
                ]
                logger.info(
                    "DeepSeek-V4.1 plugin: auto-fitting max_model_len against the "
                    "pool minus its %d-PAGE STATE tail",
                    reserve,
                )
        return original(
            vllm_config, projected_groups_per_worker, available_memory, **kwargs
        )

    patched._atom_v41_autofit_patched = True
    setattr(module, _AUTOFIT_ATTR, patched)
    return True


def apply_vllm_v41_state_reserve_patch() -> bool:
    """Wrap ``get_kv_cache_configs`` so V4.1 pools keep a STATE tail. Idempotent."""
    import importlib

    modules = []
    for name in _PATCH_TARGETS:
        try:
            modules.append(importlib.import_module(name))
        except Exception as e:  # noqa: BLE001 - optional/version-dependent module
            logger.debug(
                "ATOM V4.1 state-reserve patch: %s unavailable (%s), skip", name, e
            )
    # The auto-fit lives only where it is defined, and is patched whether or
    # not the outer wrap below is still needed.
    for module in modules:
        if _patch_auto_fit(module):
            logger.info(
                "ATOM plugin: patched vLLM %s in %s so an auto-fitted "
                "max_model_len leaves room for the DeepSeek-V4.1 STATE tail",
                _AUTOFIT_ATTR,
                module.__name__,
            )

    modules = [m for m in modules if hasattr(m, _ATTR)]
    if not modules:
        return False
    original = getattr(modules[0], _ATTR)
    if getattr(original, "_atom_v41_state_reserve_patched", False):
        return False

    @functools.wraps(original)
    def patched(*args, **kwargs):
        # Pass through rather than re-declaring the signature: this wraps a
        # vLLM function installed for every model, so a keyword call site --
        # or an argument a later release adds -- would otherwise raise
        # TypeError from inside KV-cache sizing, with the traceback pointing
        # here.
        configs = original(*args, **kwargs)
        vllm_config = kwargs.get("vllm_config") or (args[0] if args else None)
        if vllm_config is None:
            return configs
        reserve = deepseek_v41_state_reserve_blocks(vllm_config)
        if reserve:
            _reserve_state_tail(configs, reserve, vllm_config)
        return configs

    patched._atom_v41_state_reserve_patched = True
    patched.__name__ = original.__name__
    patched.__doc__ = original.__doc__
    for module in modules:
        setattr(module, _ATTR, patched)
    logger.info(
        "ATOM plugin: patched vLLM %s in %s to reserve the DeepSeek-V4.1 STATE tail",
        _ATTR,
        ", ".join(m.__name__ for m in modules),
    )
    return True
