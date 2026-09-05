"""ATOM vLLM platform integration."""

import logging
import os

from atom.utils import envs

logger = logging.getLogger("atom")

# This flag is used to enable the vLLM plugin mode.
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN

# Largest single-forward token count we allow for DeepSeek-V4 when chunked
# prefill is disabled. Beyond this, a single forward overflows int32 element
# offsets in per-token Triton kernels (num_tokens * hidden > 2**31), surfacing
# as an "illegal memory access". Chunked prefill keeps each forward small and
# is the supported path for long context; this bound only guards the
# non-chunked fallback. Override with the env var below.
_V4_MAX_SINGLE_FORWARD_TOKENS = 131072
_V4_MAX_SINGLE_FORWARD_TOKENS_ENV = "ATOM_V4_MAX_SINGLE_FORWARD_TOKENS"


def _is_deepseek_v4(model_config) -> bool:
    arches = getattr(model_config, "architectures", None) or []
    return any("DeepseekV4" in str(a) for a in arches)


def _chunked_prefill_on(scheduler_config) -> bool:
    return bool(
        getattr(scheduler_config, "chunked_prefill_enabled", False)
        or getattr(scheduler_config, "enable_chunked_prefill", False)
    )


def _demote_piecewise_cudagraph(vllm_config) -> None:
    """Drop the piecewise cudagraph component when there is nothing to split on.

    ``set_splitting_ops_for_v1`` already demotes ``PIECEWISE -> NONE`` and
    ``FULL_AND_PIECEWISE -> FULL`` on empty ``splitting_ops``, but it
    early-returns unless ``mode == VLLM_COMPILE`` -- so on a plugin path, where
    ATOM owns the model and vLLM compiles nothing, the demotion never runs.

    A "piecewise" graph with no splitting ops is the whole model, which vLLM
    dispatches with ``num_reqs=None``, i.e. any batch under the captured token
    count is replayable. V4 attention bakes its grids from the captured batch,
    so replaying an 8-token decode graph for a 6+1 batch faults the GPU.

    ``FULL_DECODE_ONLY`` rather than ``FULL``: V4 prefill is eager, so a full
    graph over mixed batches is not capturable either.
    """
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        return
    from vllm.config import CompilationMode, CUDAGraphMode

    if getattr(cc, "mode", None) == CompilationMode.VLLM_COMPILE:
        return  # vLLM compiles the model itself; its own demotion applies.
    if getattr(cc, "splitting_ops", None):
        return
    mode = getattr(cc, "cudagraph_mode", None)
    if mode is None or not mode.has_piecewise_cudagraphs():
        return
    demoted = (
        CUDAGraphMode.FULL_DECODE_ONLY
        if mode.has_full_cudagraphs()
        else CUDAGraphMode.NONE
    )
    cc.cudagraph_mode = demoted
    logger.warning(
        "ATOM DeepSeek-V4: cudagraph_mode %s requests piecewise cudagraphs, but "
        "the model is ATOM-owned so vLLM compiles nothing and splitting_ops is "
        "empty -- a 'piecewise' graph would be the whole model, replayed for "
        "batches whose query-length structure differs from capture. Demoting to "
        "%s; pass `-O.cudagraph_mode=%s` to select it explicitly.",
        mode.name,
        demoted.name,
        demoted.name,
    )


def _enforce_deepseek_v4_constraints(vllm_config) -> None:
    """Apply V4-specific plugin constraints.

    1. Enable prefix caching via tail recompute: V4's per-request SWA
       sliding-window ring is not carried by vLLM's block-level prefix cache
       (only the CSA/HCA compressed pages are). Rather than disable caching, we
       install a KVCacheManager patch that rolls every prefix hit back by
       ``max(win_with_spec, index_topk)`` tokens so the tail is re-forwarded and
       the ring is repopulated (mirrors native ATOM "fix B'").
       See ``deepseek_v4_prefix_patch``.

    2. Guard the non-chunked oversized forward: with chunked prefill off, vLLM
       couples max_num_batched_tokens to max_model_len, so a native max_model_len
       forces a single ~max_model_len-token forward that overflows int32 element
       offsets in per-token kernels. Fail fast with an actionable error instead
       of crashing with "illegal memory access". Enable chunked prefill for long
       context.

    3. Demote piecewise cudagraph modes, a demotion vLLM skips on a plugin path.
       See ``_demote_piecewise_cudagraph``.
    """
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or not _is_deepseek_v4(mc):
        return

    _demote_piecewise_cudagraph(vllm_config)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and getattr(
        cache_config, "enable_prefix_caching", False
    ):
        from atom.plugin.vllm.deepseek_v4_prefix_patch import (
            apply_vllm_v4_prefix_recompute_patch,
        )

        apply_vllm_v4_prefix_recompute_patch(vllm_config)

    # Unconditional: vLLM's cudagraph memory profiling allocates one block per
    # sequence, fewer than the V4 proxy page amortizes over.
    from atom.plugin.vllm.deepseek_v4_profiling_patch import (
        apply_vllm_v4_profiling_min_blocks_patch,
    )

    apply_vllm_v4_profiling_min_blocks_patch(vllm_config)

    sc = getattr(vllm_config, "scheduler_config", None)
    if sc is None or _chunked_prefill_on(sc):
        return

    try:
        max_single = int(
            os.environ.get(
                _V4_MAX_SINGLE_FORWARD_TOKENS_ENV, _V4_MAX_SINGLE_FORWARD_TOKENS
            )
        )
    except (TypeError, ValueError):
        max_single = _V4_MAX_SINGLE_FORWARD_TOKENS

    mnbt = int(getattr(sc, "max_num_batched_tokens", 0) or 0)
    max_model_len = int(getattr(mc, "max_model_len", 0) or 0)
    if mnbt > max_single:
        msg = (
            "DeepSeek-V4 with chunked prefill disabled requires a single forward "
            f"of up to max_num_batched_tokens={mnbt} tokens (coupled to "
            f"max_model_len={max_model_len}). That exceeds the safe single-forward "
            f"bound ({max_single}); a forward this large overflows int32 element "
            "offsets in per-token kernels and crashes with an illegal memory "
            "access. Enable chunked prefill (enable_chunked_prefill=True) to serve "
            "this context length, or lower max_model_len. Set "
            f"{_V4_MAX_SINGLE_FORWARD_TOKENS_ENV} to override this bound."
        )
        logger.error(msg)
        raise ValueError(msg)


if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform

    class ATOMPlatform(RocmPlatform):
        """ATOM platform wrapper.

        Attention backend selection is owned by ATOM's vLLM attention layers
        (`AttentionForVllm*`). We intentionally do not override
        `get_attn_backend_cls()` here, so any fallback vLLM standard attention
        keeps ROCmPlatform's native backend selection.
        """

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            super().check_and_update_config(vllm_config)
            _enforce_deepseek_v4_constraints(vllm_config)

else:
    ATOMPlatform = None


def install_platform_config_hook() -> None:
    """Run ATOMPlatform's config hook even when the platform plugin lost the race.

    vLLM memoizes `current_platform` on first read. On 0.25.x that read happens
    during `import vllm` itself (`env_override` -> `utils.torch_utils`, whose
    module body runs `is_pin_memory_available()`), while `vllm.model_executor`
    is half-imported: `register_platform()` dies on a circular ImportError,
    vLLM swallows it and caches the builtin platform. Later resolutions log
    "Platform plugin atom is activated" but nothing re-reads
    `_current_platform`, so `check_and_update_config` never runs.

    For DeepSeek-V4 the casualty is `_enforce_deepseek_v4_constraints`: without
    the prefix-cache SWA-recompute patch a cross-request hit reads an empty SWA
    ring and the model emits another request's content. That hook is
    ATOMPlatform's only override, so re-attach just it to whichever platform
    went live. Called from `register_model()`, which runs inside
    `create_engine_config()` before the platform hook. Idempotent.
    """
    if disable_vllm_plugin:
        return
    from vllm.platforms import current_platform

    live_cls = type(current_platform)
    if ATOMPlatform is not None and issubclass(live_cls, ATOMPlatform):
        return  # plugin won the race; the hook is already in the MRO
    original = live_cls.check_and_update_config
    if getattr(original, "_atom_config_hook_installed", False):
        return

    def patched(cls, vllm_config) -> None:
        original(vllm_config)
        _enforce_deepseek_v4_constraints(vllm_config)

    patched._atom_config_hook_installed = True
    live_cls.check_and_update_config = classmethod(patched)
    logger.info(
        "ATOM plugin: re-attached check_and_update_config to the live platform "
        "%s (the platform plugin lost vLLM's memoized current_platform race).",
        live_cls.__name__,
    )
