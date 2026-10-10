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


def _refuse_unsupported_v41_speculation(vllm_config) -> None:
    """Admit DSpark, refuse every other speculative method.

    vLLM drives DSpark itself -- it ships the V4.1 draft model
    (`DSparkV41DraftModel`), proposes from it and decides acceptance with its
    own rejection sampler -- so what the bridge owes is the CSA2 state a
    verify step leaves behind: the window ring, the compressor's incomplete
    group and the Engram cursor, staged tentatively and committed at the
    accepted prefix.

    The other methods have no such owner here. MTP would need a draft the
    proxy does not register, and admitting one silently would not fail at
    configuration time -- it would serve, and be wrong about state, which is
    the failure this path is least able to show.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return
    method = (getattr(spec, "method", None) or "").lower()
    if method == "dspark":
        return
    msg = (
        f"DeepSeek-V4.1 on the vLLM plugin supports DSpark speculation only; "
        f"method={method or 'unset'!r} has no draft this bridge drives. Pass "
        "--speculative-config with method=dspark, drop it, or run the native "
        "ATOM engine."
    )
    logger.error(msg)
    raise ValueError(msg)


def _select_hybrid_aware_scheduler(vllm_config) -> None:
    """Point vLLM at a scheduler whose KV-load-failure recovery knows about
    multiple KV cache groups.

    Gated on a KV connector being configured, because that is what makes the
    path reachable at all: without one no load can fail, and the override would
    change nothing. It is NOT gated on the model being hybrid -- the group
    count is not known yet here (``kv_cache_groups`` are built after memory
    profiling), and the override handles a single group exactly as vLLM does.

    Failing to select the subclass must never be fatal: vLLM's own scheduler
    still runs every model that does not hit a failed tier load, so a problem
    here is logged and stepped over rather than taking the engine down at boot.
    """
    sc = getattr(vllm_config, "scheduler_config", None)
    if sc is None or getattr(vllm_config, "kv_transfer_config", None) is None:
        return
    try:
        from atom.plugin.vllm.scheduler import select_scheduler_cls

        chosen = select_scheduler_cls(sc)
    except Exception:
        logger.warning(
            "ATOM: could not select a hybrid-aware scheduler; vLLM's own "
            "scheduler will be used.",
            exc_info=True,
        )
        return
    if chosen is not None:
        sc.scheduler_cls = chosen


_V41_EXPERIMENTAL_CUDAGRAPH_ENV = "ATOM_V41_EXPERIMENTAL_CUDAGRAPH"


def _breakable_cudagraph_available() -> bool:
    """Whether to let V4.1 be captured. Off unless explicitly asked for.

    Breakable capture removes the *stated* reason V4.1 ran eager: that reason
    was that ATOM's plugin models are not fx-split, so vLLM's PIECEWISE mode
    had nothing to split on, and breakable capture ends the stream capture at
    runtime instead of splitting an fx graph. Everything that follows from
    that is in place -- `v41_stage_step` carries the break, PIECEWISE passes
    vLLM's three `AttentionCGSupport.NEVER` gates untouched (all three test
    FULL), and the dummy-batch cache is reused rather than freed under the
    recorded kernels.

    It is still not enough, and this is measured, not assumed. With the above,
    V4.1 captures and serves without raising -- and answers degenerate into
    noise after the first few tokens. What stays behind is the attention
    itself: its kernels are launched with per-step host values (the batch's
    longest KV extent among them), which a capture freezes at the length it
    happened to record while every decode step grows past it. The ordinary
    remedy -- an eager break on the attention op, as vLLM does for
    `unified_attention_with_output` -- does not apply as a decoration, because
    V4.1's attention returns a fresh tensor and the decorator requires an
    in-place output buffer; applying it anyway faults inside capture.

    So capturing V4.1 needs attention-level work (a persistent per-layer
    output buffer and a device-side length bound), not configuration. Until
    then this returns False by default, so a vLLM that auto-enables
    VLLM_USE_BREAKABLE_CUDAGRAPH cannot quietly turn V4.1's answers to noise.
    Set ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 to pick that work back up; it is
    wrong on purpose and is not a serving configuration.
    """
    if os.environ.get(_V41_EXPERIMENTAL_CUDAGRAPH_ENV, "") not in ("1", "true", "True"):
        return False
    try:
        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )
    except ImportError:
        return False
    return bool(is_breakable_cudagraph_enabled())


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

    _refuse_unsupported_v41_speculation(vllm_config)

    from vllm.config import CUDAGraphMode

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        compilation_config is not None
        and getattr(compilation_config, "cudagraph_mode", None) != CUDAGraphMode.NONE
        and not _breakable_cudagraph_available()
    ):
        logger.info(
            "DeepSeek-V4.1 plugin mode runs eager: forcing cudagraph_mode=NONE. "
            "Its per-step Engram and state work cannot be replayed from a graph, "
            "and the fx splitting vLLM's PIECEWISE mode needs is not done for "
            "ATOM's plugin models, so PIECEWISE would swallow the backbone whole. "
            "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 to capture it instead: that mode "
            "breaks the capture at runtime around the step work, which is what "
            "`v41_stage_step` is marked for."
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif (
        compilation_config is not None
        # An explicit NONE is a request, not an absence: it is how the eager
        # arm of an on/off comparison is held comparable. Setting the env var
        # must not silently turn that arm into a graph arm.
        and getattr(compilation_config, "cudagraph_mode", None) != CUDAGraphMode.NONE
        and _breakable_cudagraph_available()
    ):
        # The requested mode is left alone, including one carrying FULL.
        #
        # It was forced to PIECEWISE for two reasons, and the work that made
        # FULL correct removed both. `eager_break_during_capture` skips the
        # break when the runtime mode is FULL, which used to freeze V4.1's
        # per-step host work into the graph -- but that work no longer runs in
        # the forward at all: the metadata builder stages it, once per step,
        # before the forward and outside anything a graph captures. And the
        # proxy builder's `AttentionCGSupport.NEVER`, which vLLM's three gates
        # in `resolve_cudagraph_mode_and_sizes` test for, is now
        # `UNIFORM_SINGLE_TOKEN_DECODE` -- a claim about where the state is
        # staged, which is the thing that changed.
        #
        # Measured before: asking for FULL_AND_PIECEWISE got
        # "setting cudagraph_mode=NONE because attention is not compiled
        # piecewise", and the server then answered every prompt correctly on
        # no graphs at all. A downgrade to NONE is silent in everything except
        # the capture count.
        logger.info(
            "DeepSeek-V4.1 plugin mode: VLLM_USE_BREAKABLE_CUDAGRAPH=1, leaving "
            "cudagraph_mode=%s for vLLM to resolve. The step's host-side work "
            "is staged by the metadata builder, before the forward, so a FULL "
            "decode graph replays it rather than re-running it.",
            compilation_config.cudagraph_mode,
        )


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
            _select_hybrid_aware_scheduler(vllm_config)
            enforce_deepseek_v41_constraints(vllm_config)

else:
    ATOMPlatform = None
