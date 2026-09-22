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


# GLM-5.3-Flash widens its NoPE MLA rope block to this many zero-padded lanes so
# the ROCm MLA kernels see the DeepSeek geometry. It must match
# `atom.models.glm5_next._ROPE_PAD`; the MLA latent/KV entry is
# ``kv_lora_rank + _GLM5_ROPE_PAD``.
_GLM5_ROPE_PAD = 64


def _fix_glm5_hybrid_page_alignment(vllm_config) -> None:
    """Re-align the KDA(mamba)+sparse-MLA KV page for GLM-5.3-Flash.

    GLM-5.3-Flash is a hybrid of KDA (mamba) layers, sparse-MLA attention layers
    (per-token page ``(kv_lora_rank + _GLM5_ROPE_PAD) * dtype`` = 1152 B) and a
    sparse indexer cache (per-token page ``(index_head_dim + 4)`` uint8 = 132 B).
    vLLM's ``_align_hybrid_block_size`` sizes the manager block / mamba page from
    a generic ``MLAAttentionSpec(head_size=model_config.get_head_size())`` that
    over-estimates the attention page, so the mamba page ends up larger than the
    real MLA page at every block size.

    vLLM's hybrid KV manager requires one physical page size across all groups.
    Any attention layer whose natural page is *smaller* than that common page is
    reconciled one of two ways in ``unify_kv_cache_spec_page_size``:
      * block-size increase, when the common page is an exact multiple of the
        layer's natural page -> ``page_size_padded`` stays None -> the cache
        reshapes via the *contiguous* path (works even when the manager block is
        split into 64-token kernel blocks); or
      * physical-page padding otherwise -> ``page_size_padded`` is set -> the
        cache reshapes via a strided view that this vLLM build cannot express
        once the block is split into kernel blocks (it overflows storage).

    So the common page must be an exact multiple of *every* split attention
    layer's per-token page. Set it to ``block * lcm(mla_per_token,
    idx_per_token)`` (the smallest that also covers the block-independent KDA
    state). Then MLA and indexer both take the block-increase/contiguous path and
    no cache is padded+split.
    """
    import math

    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    arches = getattr(model_config, "architectures", None) or []
    if not any("Glm5Next" in str(a) for a in arches):
        return
    if getattr(cache_config, "mamba_page_size_padded", None) is None:
        # No hybrid mamba page to align (e.g. plugin disabled or non-hybrid).
        return

    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    kv_lora_rank = int(getattr(text_config, "kv_lora_rank", 0) or 0)
    if kv_lora_rank <= 0:
        logger.warning(
            "ATOM GLM5 page-align: no kv_lora_rank on config; leaving vLLM's "
            "hybrid block sizing unchanged."
        )
        return

    if cache_config.cache_dtype == "auto":
        kv_dtype = model_config.dtype
    else:
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        kv_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    kv_dtype_size = torch.empty((), dtype=kv_dtype).element_size()

    # Sparse-MLA latent page per token (single latent, not K+V).
    mla_per_token = (kv_lora_rank + _GLM5_ROPE_PAD) * kv_dtype_size
    # Sparse indexer K-cache page per token: index_head_dim + 4 scale bytes,
    # stored as uint8 (1 byte/element). See atom.models.deepseek_v2.Indexer.
    index_head_dim = int(getattr(text_config, "index_head_dim", 128) or 128)
    idx_per_token = index_head_dim + 4

    from vllm.model_executor.models.registry import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture, model_config=model_config
    )
    mamba_state_page = MambaSpec(
        shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=-1,
    ).page_size_bytes

    kernel_align = 64
    block_size = kernel_align
    # Per-token common page that both attention caches divide exactly.
    common_per_token = math.lcm(mla_per_token, idx_per_token)
    page_unit = block_size * common_per_token
    # Smallest multiple of page_unit that also covers the KDA state page.
    common_page = page_unit * max(1, math.ceil(mamba_state_page / page_unit))

    logger.warning(
        "ATOM GLM5 page-align: mla_per_token=%d idx_per_token=%d "
        "mamba_state_page=%d block_size %s -> %d, "
        "mamba_page_size_padded %s -> %d",
        mla_per_token,
        idx_per_token,
        mamba_state_page,
        cache_config.block_size,
        block_size,
        cache_config.mamba_page_size_padded,
        common_page,
    )
    cache_config.block_size = block_size
    if cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = block_size
    cache_config.mamba_page_size_padded = common_page


def _install_glm5_align_patch() -> None:
    """Wrap ``Platform._align_hybrid_block_size`` so the GLM-5.3-Flash page fix
    runs regardless of which concrete platform class vLLM dispatches through.

    vLLM invokes ``_align_hybrid_block_size`` from a platform object that is not
    necessarily ``ATOMPlatform`` (and it runs in each worker's KV-cache setup),
    so a subclass override is not reliably hit. Patch the base method instead.
    """
    from vllm.platforms.interface import Platform

    if getattr(Platform, "_atom_glm5_align_patched", False):
        return

    _orig = Platform._align_hybrid_block_size.__func__

    def _wrapped(cls, vllm_config, backend_cls):
        _orig(cls, vllm_config, backend_cls)
        try:
            _fix_glm5_hybrid_page_alignment(vllm_config)
        except Exception:  # pragma: no cover - never break startup on the fix
            logger.exception("ATOM GLM5 page-align fix failed; using vLLM sizing")

    Platform._align_hybrid_block_size = classmethod(_wrapped)
    Platform._atom_glm5_align_patched = True


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

    _install_glm5_align_patch()

else:
    ATOMPlatform = None
