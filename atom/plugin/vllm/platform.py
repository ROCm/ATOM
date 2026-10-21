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
    # One rule, shared with everything else keyed on this. Reading only
    # `architectures` here while the STATE reserve also accepted `model_type`
    # meant a config carrying just the type had its tail reserved and none of
    # the refusals below applied to it.
    from atom.plugin.vllm.deepseek_v41_bridge import is_deepseek_v41_model_config

    return is_deepseek_v41_model_config(model_config)


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

    1. Prefix caching: V4's per-request SWA ring and compressor tails are not
       in any block vLLM hashes. A hit restores them from a state checkpoint
       vLLM keeps where native ATOM checkpoints (``deepseek_v4_state_checkpoint``). The
       hooks are installed from ``register_model``, which is always reached; calling the
       same idempotent entry here covers a run where this platform is active.

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
        from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
            apply_vllm_v4_prefix_install,
        )

        apply_vllm_v4_prefix_install()

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


def _require_v41_draft_kv_cache_dtype(vllm_config) -> None:
    """DSpark must name its own KV dtype, or it rewrites the target's.

    vLLM drives the DSpark draft with its own DeepSeek-V4.1 attention, which
    only accepts a packed `*_ds_mla` layout. Given `auto` it resolves to
    `fp8_ds_mla` and *writes that back onto the `cache_config` it was handed*
    (`vllm/models/deepseek_v41/attention.py`). Without a draft-level override
    that config is the target's, so ATOM's own pool dtype silently becomes a
    spelling it has no pool for, and the run dies later naming an
    `fp8_ds_mla` the operator never typed.

    `--speculative-config`'s `kv_cache_dtype` is applied to the draft's own
    copy of the config (`_DRAFT_VLLM_CONFIG_OVERRIDES`), which is exactly the
    separation this needs: the draft gets its packed layout, ATOM's target
    pool stays bf16 or fp4.

    Refused here, where the flag to add can be named, rather than left to
    surface as a dtype mismatch two layers down.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return
    if (getattr(spec, "method", None) or "").lower() != "dspark":
        return
    if getattr(spec, "kv_cache_dtype", None):
        return
    msg = (
        "DeepSeek-V4.1 DSpark on the vLLM plugin needs its draft's KV dtype "
        'named explicitly: add "kv_cache_dtype": "fp8_ds_mla" to '
        "--speculative-config. vLLM's V4.1 draft attention takes only a "
        "packed *_ds_mla layout, and without a draft-level override it "
        "rewrites the target's cache_dtype to it -- which ATOM has no pool "
        "for. The target's own --kv-cache-dtype stays auto/bfloat16/nvfp4."
    )
    logger.error(msg)
    raise ValueError(msg)


def _refuse_v41_hf_overrides(vllm_config) -> None:
    """Refuse `--hf-overrides`, which this path cannot honour.

    ATOM's V4.1 schema comes from `normalize_hf_config`, which needs the
    checkpoint's nested `text_config` and `vision_config`. vLLM's resolved
    config does not carry them by the time a worker sees it -- measured
    through `ModelConfig`: `text_config` is None and there is no
    `vision_config` -- so there is nothing to normalize and the bridge reads
    the file instead. An override applied to vLLM's object therefore never
    reaches the model, the pool geometry or the attention topology.

    Loud rather than silent, because the alternative is a server that starts,
    serves, and answers as though the flag had not been passed. Lifting this
    means getting the nested blocks through vLLM's own config path, not
    changing anything here.
    """
    mc = getattr(vllm_config, "model_config", None)
    overrides = getattr(mc, "hf_overrides", None)
    if not overrides:
        return
    msg = (
        "DeepSeek-V4.1 on the vLLM plugin does not support --hf-overrides "
        f"({sorted(overrides) if isinstance(overrides, dict) else overrides}): "
        "ATOM reads this model's config from the checkpoint, because vLLM's "
        "resolved config does not carry the nested text_config/vision_config "
        "its schema is built from. The override would be ignored rather than "
        "applied. Edit the checkpoint's config.json instead."
    )
    logger.error(msg)
    raise ValueError(msg)


def _refuse_v41_multi_axis_parallelism(vllm_config) -> None:
    """Refuse pipeline and context parallel, which V4.1 has no split for.

    The model's own `validate_runtime_config` already refuses both, but in
    plugin mode it cannot see them: `_generate_atom_config_from_vllm_config`
    never carries `pipeline_parallel_size` or the context-parallel sizes into
    the ATOM `Config`, so they read 1 whatever vLLM was told, and the check
    passes because it was never asked the question. `vllm serve -pp 2` is then
    accepted and fails later, somewhere that does not name the cause.

    Asked of vLLM's config directly, which is the only place the truth is, and
    kept here rather than fixed in the translation so that admitting a model
    for a parallelism it does not implement stays one model's decision.
    """
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None:
        return
    unsupported = [
        (name, size)
        for name, size in (
            ("pipeline_parallel_size", getattr(parallel, "pipeline_parallel_size", 1)),
            (
                "prefill_context_parallel_size",
                getattr(parallel, "prefill_context_parallel_size", 1),
            ),
            (
                "decode_context_parallel_size",
                getattr(parallel, "decode_context_parallel_size", 1),
            ),
        )
        if int(size or 1) != 1
    ]
    if not unsupported:
        return
    msg = (
        "DeepSeek-V4.1 on the vLLM plugin supports tensor parallel only; it has "
        "no pipeline or context parallel split. Refused: "
        + ", ".join(f"{name}={size}" for name, size in unsupported)
        + ". Use --tensor-parallel-size for the whole device group."
    )
    logger.error(msg)
    raise ValueError(msg)


def _refuse_v41_kv_transfer(vllm_config) -> None:
    """Refuse every KV connector on this path, including `lmcache_mp`.

    A V4.1 request's recoverable state is not its PAGEs alone: the window
    ring, the compressor rings and the Engram cursor live in the STATE region,
    and a connector that restores PAGE bytes without them hands the forward a
    prefix it cannot continue from.

    The model's check admits `lmcache_mp` because ATOM's own transport
    checkpoints STATE through its PAGE-backed images. That reasoning does not
    carry over here: in plugin mode vLLM resolves the connector through *its*
    registry, where `LMCacheMPConnector` is vLLM's own, and ATOM's STATE
    checkpoint path is not what runs. The name alone is why it was admitted --
    ATOM registers `LMCacheMPConnector` as an alias of `lmcache_mp`.

    Every other name is refused here too, rather than left to
    `KVConnectorFactory.connector_name`, which raises "unknown KV connector"
    for vLLM's own `NixlConnector` / `OffloadingConnector` / `MooncakeConnector`
    -- an error about ATOM's registry, in answer to a question about V4.1.

    ATOM's own `AtomLMCacheOffloadConnector` is refused on this branch for the
    same reason and not for the alias one: it is selected by explicit
    `kv_connector_module_path`, so it never goes through that name resolution,
    but the version here knows nothing about V4.1 -- it carries the recurrent
    mamba state, and `connector.py` has no V4.1 path at all. A V4.1 hit through
    it would restore PAGEs and no STATE.

    This is the condition to relax when that changes: a connector that writes
    and restores V4.1's STATE entries belongs in an allow-list here, added in
    the same change that teaches it to -- not before, since the failure it
    would admit is a silently wrong continuation rather than an error.
    """
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None:
        return
    connector = getattr(transfer, "kv_connector", None) or (
        transfer.get("kv_connector") if isinstance(transfer, dict) else None
    )
    if not connector:
        return
    msg = (
        f"DeepSeek-V4.1 on the vLLM plugin does not support KV transfer "
        f"({connector}): a request's window ring, compressor rings and Engram "
        "cursor live in the STATE region, and no connector on this path "
        "restores them, so a hit would resume from PAGEs whose state is "
        "missing. Run without --kv-transfer-config."
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

    Capture works, and this gate is about who turns it on rather than whether
    it is correct. Three things make a V4.1 forward recordable, each needed:
    the step's host work is staged outside the forward, attention runs in an
    eager break writing through a caller-owned buffer (a fresh tensor would
    change address on every replay), and Engram keeps its side stream with
    the orderings in breaks. Measured on TP4, full GSM8K: 0.9303 +/- 0.0070
    with DSpark and graphs, against 0.9295 +/- 0.0071 eager, 47 FULL and 110
    PIECEWISE graphs captured.

    What this earlier said -- that the answers degenerate into noise and the
    decorator cannot be applied because attention returns a fresh tensor --
    described the branch before `_attn_out_buffer`, which is exactly the
    buffer that was missing.

    Still default-off, and the reason is not doubt about the numbers: vLLM
    auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for some architectures, so
    without a second condition of our own, a vLLM upgrade that adds V4.1 to
    that list would change how this path runs with nothing in the log. Set
    ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 to opt in.
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

    2. Refuse prefix caching. vLLM's prefix cache is bookkeeping: registering a
       cached block is a claim that the backend left recoverable state at that
       block's boundary, and the mamba-style managers that participate only
       report a hit as far as a boundary they hold state for. V4.1's forward
       cannot make that claim -- copying a slot after the forward yields the
       state at the *end* of the query, while the boundary vLLM asks about is
       interior to it. Measured on the sibling branch that tried to serve it:
       setting ``num_prefill_checkpoint_blocks`` to 0 does not opt out, because
       align mode still caches the last complete block's state and the group
       still reports the boundary. So this is a property of the contract, not a
       flag -- any STATE group that joins prefix caching will be handed a
       boundary nobody wrote. ``validate_runtime_config`` refuses it too, but
       only once the worker is already up.

    3. Refuse every speculative method except DSpark, which vLLM drives itself;
       see ``_refuse_unsupported_v41_speculation``.

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
            "CSA2 attention reads alongside them, and a V4.1 forward cannot "
            "leave them at the interior block boundary vLLM's cache assumes. "
            "Pass --no-enable-prefix-caching."
        )
        logger.error(msg)
        raise ValueError(msg)

    _refuse_v41_hf_overrides(vllm_config)
    _refuse_v41_multi_axis_parallelism(vllm_config)
    _refuse_v41_kv_transfer(vllm_config)
    _refuse_unsupported_v41_speculation(vllm_config)
    _require_v41_draft_kv_cache_dtype(vllm_config)

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
            "Set both VLLM_USE_BREAKABLE_CUDAGRAPH=1 and "
            "ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 to capture it instead: that mode "
            "breaks the capture at runtime around the step work, which is what "
            "`v41_stage_step` is marked for. Naming only the first sends a reader "
            "back here with the same message, since this gate needs both."
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
