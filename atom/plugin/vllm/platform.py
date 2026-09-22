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


# DeepSeek-V4.1's architecture string starts with the V4 one, so a substring
# test for V4 matches it too. Both branches below are keyed on these, and V4.1
# must not take V4's.
_DEEPSEEK_V41_ARCHES = ("DeepseekV41ForCausalLM",)


def _is_deepseek_v41(model_config) -> bool:
    arches = getattr(model_config, "architectures", None) or []
    return any(str(a) in _DEEPSEEK_V41_ARCHES for a in arches)


def _is_deepseek_v4(model_config) -> bool:
    if _is_deepseek_v41(model_config):
        return False
    arches = getattr(model_config, "architectures", None) or []
    return any("DeepseekV4" in str(a) for a in arches)


def _chunked_prefill_on(scheduler_config) -> bool:
    return bool(
        getattr(scheduler_config, "chunked_prefill_enabled", False)
        or getattr(scheduler_config, "enable_chunked_prefill", False)
    )


def _enforce_deepseek_v4_constraints(vllm_config) -> None:
    """Apply V4-specific plugin constraints.

    1. Enable prefix caching via SWA recompute: V4's per-request SWA
       sliding-window ring is not carried by vLLM's block-level prefix cache
       (only the CSA/HCA compressed pages are). Rather than disable caching, we
       install a KVCacheManager patch that, on a prefix hit, drops the last
       ``ceil(win_with_spec / block_size)`` cached blocks so the SWA tail is
       re-forwarded and the ring is repopulated (mirrors native ATOM "fix B'").
       See ``deepseek_v4_prefix_patch``.

    2. Guard the non-chunked oversized forward: with chunked prefill off, vLLM
       couples max_num_batched_tokens to max_model_len, so a native max_model_len
       forces a single ~max_model_len-token forward that overflows int32 element
       offsets in per-token kernels. Fail fast with an actionable error instead
       of crashing with "illegal memory access". Enable chunked prefill for long
       context.
    """
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or not _is_deepseek_v4(mc):
        return

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and getattr(
        cache_config, "enable_prefix_caching", False
    ):
        from atom.plugin.vllm.deepseek_v4_prefix_patch import (
            apply_vllm_v4_prefix_swa_patch,
        )

        apply_vllm_v4_prefix_swa_patch(vllm_config)

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


def enforce_deepseek_v41_constraints(vllm_config) -> None:
    """Apply V4.1-specific plugin constraints to one config. Idempotent.

    1. Run eager. A V4.1 step does host-side work every forward -- Engram row
       staging, per-request state reset, the cursor advance -- that no captured
       graph replays, and the plugin's ATOM models are not fx-split, so the
       PIECEWISE mode vLLM falls back to on its own (it reads the proxy
       backend's ``AttentionCGSupport.NEVER`` and only demotes FULL by one
       step) would swallow the whole backbone rather than break around
       attention. Say NONE outright.

    2. Refuse prefix caching. CSA2 blocks are only reusable once the compressor
       has run over a whole PAGE, and the per-request STATE side is not
       replayed by a block-table hit, so vLLM's hash reuse would hand back a
       prefix whose window ring never existed. ``validate_runtime_config``
       refuses it too, but only once the worker is already up.

    3. Refuse speculative decoding, which the proxy bridge does not drive.

    Called from two places, because neither alone covers the process that acts
    on the result: ``ATOMPlatform.check_and_update_config`` is the natural
    config-time site but vLLM may never activate ``ATOMPlatform`` at all (see
    ``deepseek_v41_state_reserve_patch``), and the model wrapper's ``__init__``
    runs in every worker during ``load_model`` -- after the config dump, but
    still ahead of ``initialize_cudagraph_capture`` and of any forward.
    """
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or not _is_deepseek_v41(mc):
        return

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and getattr(
        cache_config, "enable_prefix_caching", False
    ):
        msg = (
            "DeepSeek-V4.1 on the vLLM plugin does not support prefix caching: a "
            "block-table hit restores the compressed pages but not the "
            "per-request window ring, compressor rings or Engram cursor that "
            "CSA2 attention reads alongside them. Pass --no-enable-prefix-caching."
        )
        logger.error(msg)
        raise ValueError(msg)

    if getattr(vllm_config, "speculative_config", None) is not None:
        msg = (
            "DeepSeek-V4.1 on the vLLM plugin does not support speculative "
            "decoding yet; DSpark drafting needs ATOM's tentative staging, which "
            "the proxy bridge does not drive. Run the native ATOM engine for "
            "DSpark, or drop --speculative-config."
        )
        logger.error(msg)
        raise ValueError(msg)

    from vllm.config import CUDAGraphMode

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        compilation_config is not None
        and getattr(compilation_config, "cudagraph_mode", None) != CUDAGraphMode.NONE
    ):
        logger.info(
            "DeepSeek-V4.1 plugin mode runs eager: forcing cudagraph_mode=NONE "
            "(its per-step Engram and state work cannot be replayed by a graph)."
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE


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
            enforce_deepseek_v41_constraints(vllm_config)

else:
    ATOMPlatform = None
