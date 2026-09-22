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


def deepseek_v41_state_reserve_blocks(vllm_config) -> int:
    """Blocks to withhold from vLLM's pool so the CSA2 STATE tail fits.

    Zero for any model that is not DeepSeek-V4.1 on the plugin path, which is
    what makes the patch safe to install unconditionally for the process.
    """
    from atom.plugin.vllm.deepseek_v41_bridge import (
        v41_proxy_state_reserve_blocks,
    )

    return v41_proxy_state_reserve_blocks(vllm_config)


def _reserve_state_tail(configs, reserve: int, vllm_config) -> None:
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_BLOCK_SIZE,
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )

    max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
    # One request must still fit after the tail comes out, or the scheduler
    # can never admit anything -- a clearer failure here than a request that
    # is preempted forever.
    min_usable = max(1, -(-max_model_len // ATOM_DEEPSEEK_V41_BLOCK_SIZE))
    for cfg in configs:
        owns_proxy = any(
            ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME in group.layer_names
            for group in cfg.kv_cache_groups
        )
        if not owns_proxy:
            continue
        usable = cfg.num_blocks - reserve
        if usable < min_usable:
            raise ValueError(
                "DeepSeek-V4.1 plugin mode cannot fit its per-request state: the "
                f"KV pool holds {cfg.num_blocks} PAGEs, of which {reserve} are "
                "reserved for the sliding-window / compressor / Engram state of "
                f"{vllm_config.scheduler_config.max_num_seqs} concurrent requests, "
                f"leaving {usable} for history -- below the "
                f"{min_usable} a single max_model_len={max_model_len} request "
                "needs. Raise --gpu-memory-utilization, or lower --max-num-seqs "
                "or --max-model-len."
            )
        logger.info(
            "DeepSeek-V4.1 plugin: reserving %d of %d proxy PAGEs for the CSA2 "
            "STATE tail (%d schedulable)",
            reserve,
            cfg.num_blocks,
            usable,
        )
        cfg.num_blocks = usable


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
    modules = [m for m in modules if hasattr(m, _ATTR)]
    if not modules:
        return False
    original = getattr(modules[0], _ATTR)
    if getattr(original, "_atom_v41_state_reserve_patched", False):
        return False

    def patched(vllm_config, kv_cache_specs, available_memory):
        configs = original(vllm_config, kv_cache_specs, available_memory)
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
