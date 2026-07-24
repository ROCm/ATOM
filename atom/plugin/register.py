import logging
import os

from atom.config import Config
from atom.models.deepseek_v2 import DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM
from atom.models.glm4_moe import Glm4MoeForCausalLM
from atom.models.minimax_m2 import MiniMaxM2ForCausalLM
from atom.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_5 import (
    Qwen3_5ForConditionalGenerationTextOnly,
    Qwen3_5MoeForConditionalGenerationTextOnly,
)
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.plugin.prepare import is_rtpllm, is_sglang, is_vllm

logger = logging.getLogger("atom")

_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "Glm4MoeForCausalLM": Glm4MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "DeepseekV32ForCausalLM": DeepseekV3ForCausalLM,
    "GlmMoeDsaForCausalLM": GlmMoeDsaForCausalLM,
    "MiniMaxM2ForCausalLM": MiniMaxM2ForCausalLM,
    "MiniMaxM3SparseForCausalLM": MiniMaxM3SparseForCausalLM,
    "MiniMaxM3SparseForConditionalGeneration": MiniMaxM3SparseForConditionalGeneration,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGenerationTextOnly,
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationTextOnly,
}

if is_sglang():
    from atom.models.deepseek_v4 import DeepseekV4ForCausalLM
    from atom.models.kimi_k25 import KimiK25ForCausalLM
    from atom.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )
    from atom.models.qwen3_next import Qwen3NextForCausalLM

    _ATOM_SUPPORTED_MODELS.update(
        {
            "DeepseekV4ForCausalLM": DeepseekV4ForCausalLM,
            "Qwen3NextForCausalLM": Qwen3NextForCausalLM,
            "Qwen3_5ForConditionalGeneration": Qwen3_5ForCausalLM,
            "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForCausalLM,
            # ROCm/ATOM#1078: route Kimi-K2.x through ATOM's quant-aware model
            # path (KimiK25ForCausalLM -> DeepseekV2ForCausalLM). The standalone
            # engine already registers this in atom/model_engine/model_runner.py;
            # the SGLang plugin path was missing it, so launches fell back to
            # sglang's native model and failed weight loading on the excluded
            # (BF16) attention projections.
            "KimiK25ForConditionalGeneration": KimiK25ForCausalLM,
        }
    )


def _register_custom_attention_to_sglang() -> None:
    """Override sglang's built-in "aiter" attention backend with ATOM's implementation.

    sglang only accepts pre-registered backend names, so we reuse the "aiter"
    name to inject ATOMAttnBackendForSgl without modifying sglang source.
    """
    import sglang.srt.layers.attention.aiter_backend as sglang_aiter_backend
    import sglang.srt.layers.attention.dsa_backend as sglang_dsa_backend
    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    from atom.plugin.sglang.attention_backend.deepseek_v4_backend import (
        ATOMDeepseekV4BackendForSgl,
    )
    from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
        ATOMAttnBackendForSgl,
    )
    from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
        ATOMGLM52DSABackendForSgl,
    )
    from atom.plugin.sglang.runtime import is_glm52_dsa_config

    # here register the custom attention backend with the name "aiter"
    # as sglang defines the fixed attention backend choices, which must be
    # in-tree
    logger.info("Register custom attention backend ATOMAttnBackendForSgl to SGLang")

    # Speculative draft paths instantiate AiterAttnBackend directly inside
    # AiterMultiStepDraftBackend, bypassing the attention registry. Rebind the
    # module symbol as well so both registry lookup and direct construction use
    # the plugin backend.
    sglang_aiter_backend.AiterAttnBackend = ATOMAttnBackendForSgl

    @register_attention_backend("aiter")
    def create_atom_backend(runner):
        hf_config = runner.model_config.hf_config
        arches = getattr(hf_config, "architectures", None) or []
        if any("DeepseekV4" in str(arch) for arch in arches):
            logger.info(
                "Use ATOMDeepseekV4BackendForSgl for DeepSeek-V4 through SGLang aiter backend choice"
            )
            return ATOMDeepseekV4BackendForSgl(runner)
        if is_glm52_dsa_config(hf_config):
            logger.info(
                "Use ATOMGLM52DSABackendForSgl for GLM-5.2 through SGLang aiter backend choice"
            )
            return ATOMGLM52DSABackendForSgl(runner)
        return ATOMAttnBackendForSgl(runner)

    @register_attention_backend("dsv4")
    def create_dsv4_backend(runner):
        logger.info(
            "Create ATOMDeepseekV4BackendForSgl through SGLang dsv4 backend choice"
        )
        return ATOMDeepseekV4BackendForSgl(runner)

    @register_attention_backend("dsa")
    def create_atom_dsa_backend(runner):
        hf_config = runner.model_config.hf_config
        if is_glm52_dsa_config(hf_config):
            logger.info(
                "Use ATOMGLM52DSABackendForSgl for GLM-5.2 through SGLang dsa backend choice"
            )
            return ATOMGLM52DSABackendForSgl(runner)
        return sglang_dsa_backend.DeepseekSparseAttnBackend(runner)

    @register_attention_backend("nsa")
    def create_atom_nsa_backend(runner):
        hf_config = runner.model_config.hf_config
        if is_glm52_dsa_config(hf_config):
            logger.info(
                "Use ATOMGLM52DSABackendForSgl for GLM-5.2 through SGLang nsa backend choice"
            )
            return ATOMGLM52DSABackendForSgl(runner)
        from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

        return NativeSparseAttnBackend(runner)


def _patch_sglang_dsv4_draft_backends() -> None:
    """Route SGLang's hard-coded DSV4 speculative factories to ATOM.

    DraftBackendFactory constructs DeepSeek-V4 draft backends directly instead
    of going through the attention registry.  SGLang's native backend asserts a
    native DeepSeekV4TokenToKVPool, while ATOM plugin mode uses a proxy KV pool,
    so patch the factory methods to return the ATOM shim.
    """

    try:
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        from atom.plugin.sglang.attention_backend.deepseek_v4_backend import (
            ATOMDeepseekV4BackendForSgl,
        )
    except Exception as exc:  # noqa: BLE001 - optional SGLang symbols vary by version
        logger.debug("Skip patching SGLang DSV4 draft backends: %s", exc)
        return

    if getattr(DraftBackendFactory, "_atom_dsv4_draft_backend_patched", False):
        return

    def _create_atom_dsv4_decode_backend(self):
        return ATOMDeepseekV4BackendForSgl(
            self.draft_model_runner,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )

    def _create_atom_dsv4_prefill_backend(self):
        return ATOMDeepseekV4BackendForSgl(
            self.draft_model_runner,
            skip_prefill=False,
        )

    DraftBackendFactory._create_dsv4_decode_backend = _create_atom_dsv4_decode_backend
    DraftBackendFactory._create_dsv4_prefill_backend = _create_atom_dsv4_prefill_backend
    DraftBackendFactory._atom_dsv4_draft_backend_patched = True
    logger.info("Patched SGLang DSV4 speculative draft backends to ATOM")


def _patch_sglang_dsv4_spec_cuda_graph() -> None:
    """Patch SGLang speculative CUDA graph handling for ATOM DSV4.

    SGLang's draft graph buffers store hidden states as flattened
    ``spec_hidden_size`` tensors.  ATOM DSV4 keeps the mHC residual as
    ``[tokens, hc, hidden]``.  Flatten just for graph replay input staging, then
    let the ATOM NextN wrapper reshape it back before running the MTP block.
    """

    try:
        try:
            from sglang.srt.model_executor.runner import (
                DecodeCudaGraphRunner as CudaGraphRunner,
            )
        except ImportError:
            from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        from atom.plugin.sglang.runtime.context import is_draft_extend_mode
    except Exception as exc:  # noqa: BLE001 - optional SGLang symbols vary by version
        logger.debug("Skip patching SGLang DSV4 spec cuda graph: %s", exc)
        return

    def _is_dsv4_nextn_runner(runner) -> bool:
        try:
            arches = (
                getattr(
                    getattr(getattr(runner, "model_config", None), "hf_config", None),
                    "architectures",
                    None,
                )
                or []
            )
            return any("DeepseekV4ForCausalLMNextN" in str(arch) for arch in arches)
        except Exception:  # noqa: BLE001 - defensive runner introspection
            return False

    def _is_dsv4_runner(runner) -> bool:
        try:
            arches = (
                getattr(
                    getattr(getattr(runner, "model_config", None), "hf_config", None),
                    "architectures",
                    None,
                )
                or []
            )
            return any("DeepseekV4" in str(arch) for arch in arches)
        except Exception:  # noqa: BLE001 - defensive runner introspection
            return False

    def _flatten_spec_hidden_states(forward_batch):
        spec_info = getattr(forward_batch, "spec_info", None)
        hidden_states = getattr(spec_info, "hidden_states", None)
        if hidden_states is None or getattr(hidden_states, "dim", lambda: 0)() <= 2:
            return None
        flattened = hidden_states.reshape(hidden_states.shape[0], -1)
        input_ids = getattr(forward_batch, "input_ids", None)
        num_tokens = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
        mode = getattr(forward_batch, "forward_mode", None)
        is_draft_extend = is_draft_extend_mode(mode, include_v2=True)
        if is_draft_extend and num_tokens > 0 and flattened.shape[0] != num_tokens:
            if num_tokens % int(flattened.shape[0]) != 0:
                raise RuntimeError(
                    "DSV4 speculative hidden layout cannot be expanded for graph "
                    f"input: hidden={tuple(hidden_states.shape)} "
                    f"flattened={tuple(flattened.shape)} num_tokens={num_tokens}"
                )
            flattened = flattened.repeat_interleave(
                num_tokens // int(flattened.shape[0]), dim=0
            )
        spec_info.hidden_states = flattened
        return hidden_states

    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")

    def _is_dsv4_flash_runner(runner) -> bool:
        model_path = str(
            getattr(getattr(runner, "server_args", None), "model_path", "")
            or getattr(getattr(runner, "model_config", None), "path", "")
        )
        return "DeepSeek-V4-Flash" in model_path

    def _is_dsv4_pro_runner(runner) -> bool:
        model_path = str(
            getattr(getattr(runner, "server_args", None), "model_path", "")
            or getattr(getattr(runner, "model_config", None), "path", "")
        )
        return "DeepSeek-V4-Pro" in model_path

    def _draft_extend_graph_enabled(runner) -> bool:
        if _env_flag("ATOM_SGLANG_V4_DISABLE_DRAFT_EXTEND_CG"):
            return False
        return _env_flag("ATOM_SGLANG_V4_ENABLE_DRAFT_EXTEND_CG") or (
            _is_dsv4_nextn_runner(runner) and _is_dsv4_flash_runner(runner)
        )

    def _target_verify_graph_enabled() -> bool:
        return _env_flag("ATOM_SGLANG_V4_ENABLE_TARGET_VERIFY_CG") and not _env_flag(
            "ATOM_SGLANG_V4_DISABLE_TARGET_VERIFY_CG"
        )

    def _safe_spec_graph_bs(original_bs, env_name: str):
        configured = os.environ.get(env_name)
        if not configured:
            return list(original_bs)
        allowed = {int(x) for x in configured.replace(" ", ",").split(",") if x.strip()}
        return [bs for bs in original_bs if int(bs) in allowed]

    def _get_decode_cuda_graph_bs(server_args):
        if server_args is None:
            return None
        graph_config = getattr(server_args, "cuda_graph_config", None)
        decode_config = getattr(graph_config, "decode", None)
        bs = getattr(decode_config, "bs", None)
        if bs is not None:
            return list(bs)
        for name in ("cuda_graph_bs_decode", "cuda_graph_bs"):
            if hasattr(server_args, name):
                bs = getattr(server_args, name)
                if bs is not None:
                    return list(bs)
        return None

    def _set_decode_cuda_graph_bs(server_args, bs):
        if server_args is None:
            return
        bs = list(bs)
        graph_config = getattr(server_args, "cuda_graph_config", None)
        decode_config = getattr(graph_config, "decode", None)
        if decode_config is not None and hasattr(decode_config, "bs"):
            decode_config.bs = bs
        if hasattr(server_args, "cuda_graph_bs_decode"):
            server_args.cuda_graph_bs_decode = bs
        if hasattr(server_args, "cuda_graph_bs"):
            server_args.cuda_graph_bs = bs

    if not getattr(CudaGraphRunner, "_atom_dsv4_init_patched", False):
        original_target_init = CudaGraphRunner.__init__

        def __init__(self, model_runner, *args, **kwargs):
            should_cap = False
            server_args = getattr(model_runner, "server_args", None)
            original_cuda_graph_bs = _get_decode_cuda_graph_bs(server_args)
            try:
                should_cap = _is_dsv4_runner(model_runner) and bool(
                    getattr(
                        getattr(model_runner, "spec_algorithm", None),
                        "is_speculative",
                        lambda: False,
                    )()
                )
                should_cap = (
                    should_cap
                    and not getattr(model_runner, "is_draft_worker", False)
                    and _target_verify_graph_enabled()
                )
            except Exception:  # noqa: BLE001 - optional graph cap detection
                should_cap = False

            try:
                if should_cap and server_args is not None and original_cuda_graph_bs:
                    _set_decode_cuda_graph_bs(
                        server_args,
                        _safe_spec_graph_bs(
                            original_cuda_graph_bs,
                            "ATOM_SGLANG_V4_TARGET_VERIFY_CG_BS",
                        ),
                    )
                original_target_init(self, model_runner, *args, **kwargs)
            finally:
                if (
                    should_cap
                    and server_args is not None
                    and original_cuda_graph_bs is not None
                ):
                    _set_decode_cuda_graph_bs(server_args, original_cuda_graph_bs)

        CudaGraphRunner.__init__ = __init__
        CudaGraphRunner._atom_dsv4_init_patched = True

    can_run_method = (
        "can_run_graph" if hasattr(CudaGraphRunner, "can_run_graph") else "can_run"
    )

    if not getattr(CudaGraphRunner, "_atom_dsv4_spec_can_run_patched", False):
        original_can_run = getattr(CudaGraphRunner, can_run_method)

        def can_run(self, forward_batch):
            try:
                model_runner = getattr(self, "model_runner", None)
                hf_config = getattr(
                    getattr(model_runner, "model_config", None), "hf_config", None
                )
                arches = getattr(hf_config, "architectures", None) or []
                is_dsv4 = any("DeepseekV4" in str(arch) for arch in arches)
                mode = getattr(forward_batch, "forward_mode", None)
                is_target_verify = bool(
                    getattr(mode, "is_target_verify", lambda: False)()
                )
                is_draft_extend = is_draft_extend_mode(mode, include_v2=True)
                if is_dsv4 and is_target_verify and not _target_verify_graph_enabled():
                    return False
                if is_dsv4 and is_draft_extend:
                    return False
            except Exception:  # noqa: BLE001, S110 - fall back to original can_run
                pass
            return original_can_run(self, forward_batch)

        setattr(CudaGraphRunner, can_run_method, can_run)
        CudaGraphRunner._atom_dsv4_spec_can_run_patched = True

    if not getattr(EAGLEDraftCudaGraphRunner, "_atom_dsv4_replay_patched", False):
        draft_replay_method = (
            "execute" if hasattr(EAGLEDraftCudaGraphRunner, "execute") else "replay"
        )
        original_draft_replay = getattr(EAGLEDraftCudaGraphRunner, draft_replay_method)

        def replay(self, forward_batch):
            if not _is_dsv4_nextn_runner(getattr(self, "model_runner", None)):
                return original_draft_replay(self, forward_batch)
            if _env_flag("ATOM_SGLANG_V4_DISABLE_DRAFT_CG"):
                raise RuntimeError(
                    "DSV4 draft cuda graph replay was disabled after capture; "
                    "disable it before graph initialization instead."
                )
            original_hidden_states = _flatten_spec_hidden_states(forward_batch)
            try:
                return original_draft_replay(self, forward_batch)
            finally:
                if original_hidden_states is not None:
                    forward_batch.spec_info.hidden_states = original_hidden_states

        setattr(EAGLEDraftCudaGraphRunner, draft_replay_method, replay)
        EAGLEDraftCudaGraphRunner._atom_dsv4_replay_patched = True

    if not getattr(
        EAGLEDraftExtendCudaGraphRunner, "_atom_dsv4_execute_patched", False
    ):
        extend_replay_method = (
            "execute"
            if hasattr(EAGLEDraftExtendCudaGraphRunner, "execute")
            else "replay"
        )
        original_extend_replay = getattr(
            EAGLEDraftExtendCudaGraphRunner, extend_replay_method
        )

        def replay(self, forward_batch):
            if not _is_dsv4_nextn_runner(getattr(self, "model_runner", None)):
                return original_extend_replay(self, forward_batch)
            if not _draft_extend_graph_enabled(getattr(self, "model_runner", None)):
                raise RuntimeError(
                    "DSV4 draft-extend cuda graph replay was disabled after capture; "
                    "disable it before graph initialization instead."
                )
            original_hidden_states = _flatten_spec_hidden_states(forward_batch)
            backend = getattr(self, "draft_extend_attn_backend", None)
            previous_runner = (
                getattr(backend, "_atom_dsv4_draft_extend_graph_runner", None)
                if backend is not None
                else None
            )
            try:
                if backend is not None:
                    backend._atom_dsv4_draft_extend_graph_runner = self
                return original_extend_replay(self, forward_batch)
            finally:
                if backend is not None:
                    if previous_runner is None:
                        try:
                            delattr(backend, "_atom_dsv4_draft_extend_graph_runner")
                        except AttributeError:
                            pass
                    else:
                        backend._atom_dsv4_draft_extend_graph_runner = previous_runner
                if original_hidden_states is not None:
                    forward_batch.spec_info.hidden_states = original_hidden_states

        setattr(EAGLEDraftExtendCudaGraphRunner, extend_replay_method, replay)
        EAGLEDraftExtendCudaGraphRunner._atom_dsv4_execute_patched = True

    if not getattr(EagleDraftWorker, "_atom_dsv4_init_cuda_graphs_patched", False):
        original_init_cuda_graphs = EagleDraftWorker.init_cuda_graphs

        def init_cuda_graphs(self):
            ret = original_init_cuda_graphs(self)
            try:
                if _env_flag(
                    "ATOM_SGLANG_V4_DISABLE_DRAFT_CG"
                ) and _is_dsv4_nextn_runner(getattr(self, "draft_runner", None)):
                    self.cuda_graph_runner = None
                if (
                    self.cuda_graph_runner_for_draft_extend is None
                    and _is_dsv4_nextn_runner(getattr(self, "draft_runner", None))
                    and not self.server_args.disable_cuda_graph
                    and _draft_extend_graph_enabled(getattr(self, "draft_runner", None))
                    and self.draft_extend_attn_backend is not None
                ):
                    seq_len_fill = max(
                        1024,
                        int(
                            getattr(self.server_args, "speculative_num_draft_tokens", 1)
                            or 1
                        ),
                    )
                    for backend in (
                        getattr(
                            getattr(self, "draft_runner", None), "attn_backend", None
                        ),
                        getattr(self, "draft_extend_attn_backend", None),
                    ):
                        if backend is not None and hasattr(
                            backend, "_cuda_graph_seq_len_fill_value"
                        ):
                            backend._cuda_graph_seq_len_fill_value = seq_len_fill
                    draft_runner = getattr(self, "draft_runner", None)
                    server_args = getattr(draft_runner, "server_args", None)
                    original_cuda_graph_bs = _get_decode_cuda_graph_bs(server_args)
                    try:
                        if server_args is not None and original_cuda_graph_bs:
                            _set_decode_cuda_graph_bs(
                                server_args,
                                _safe_spec_graph_bs(
                                    original_cuda_graph_bs,
                                    "ATOM_SGLANG_V4_DRAFT_EXTEND_CG_BS",
                                ),
                            )
                        self.cuda_graph_runner_for_draft_extend = (
                            EAGLEDraftExtendCudaGraphRunner(self)
                        )
                    finally:
                        if (
                            server_args is not None
                            and original_cuda_graph_bs is not None
                        ):
                            _set_decode_cuda_graph_bs(
                                server_args, original_cuda_graph_bs
                            )
                elif _is_dsv4_nextn_runner(getattr(self, "draft_runner", None)):
                    self.cuda_graph_runner_for_draft_extend = None
            except Exception as exc:
                logger.warning(
                    "Failed to enable DSV4 draft-extend cuda graph in ATOM plugin: %s",
                    exc,
                )
                if _env_flag("ATOM_SGLANG_V4_ENABLE_DRAFT_EXTEND_CG"):
                    raise RuntimeError(
                        "DSV4 draft-extend CUDA graph was explicitly enabled but "
                        "capture failed"
                    ) from exc
            return ret

        EagleDraftWorker.init_cuda_graphs = init_cuda_graphs
        EagleDraftWorker._atom_dsv4_init_cuda_graphs_patched = True


def register_ops_to_sglang(atom_config: Config) -> None:
    """
    Register custom ops to sglang, including attention
    """
    _register_custom_attention_to_sglang()
    _patch_sglang_dsv4_draft_backends()
    _patch_sglang_dsv4_spec_cuda_graph()


def set_attn_cls() -> None:
    """Keep compatibility with old plugin init hooks.

    FIXME: This is a legacy no-op after attention construction moved to the
    frontend dispatcher. Remove it once downstream plugin init paths stop
    calling ``set_attn_cls`` for side effects.

    Attention selection now happens in ``atom.model_ops.base_attention.Attention``
    at construction time, so plugin init no longer mutates ``atom.model_ops``.
    """
    if is_vllm():
        logger.info("Use Attention dispatcher for vLLM")
    elif is_sglang():
        logger.info("Use Attention dispatcher for SGLang")
    elif is_rtpllm():
        logger.info("Use Attention dispatcher for rtp-llm")


def init_aiter_dist(config: Config) -> None:
    """
    Initialize aiter dist for using aiter custom collective op.

    In vLLM plugin mode, tries to reuse vLLM's TP group and inject aiter's ca_comm
    first (single IPC init, avoids 2x reduce slowdown). For DP+EP, skip the
    reuse fast path and let aiter initialize its own TP/PP/DP/EP groups so EP and
    all2all ownership stays within the ATOM+vLLM stack. Falls back to init_dist_env if
    reuse fails.
    """
    logger.info(
        "Initialize aiter dist for using aiter custom collective op for plugin mode"
    )

    rank = config.plugin_config.rank
    if getattr(config.plugin_config, "is_sglang", False):
        rank = getattr(config.plugin_config, "sglang_aiter_rank_id", rank)
    tensor_parallel_size = config.tensor_parallel_size

    assert (
        config.plugin_config.is_plugin_mode
    ), "Make sure ATOM is running in plugin mode"

    use_vllm_atom_owned_ep = (
        config.plugin_config.is_vllm
        and config.enable_expert_parallel
        and config.parallel_config.data_parallel_size > 1
    )

    if use_vllm_atom_owned_ep:
        logger.info(
            "Skip vLLM TP reuse for OOT DP+EP so aiter owns TP/PP/DP/EP groups."
        )

    if config.plugin_config.is_vllm and not use_vllm_atom_owned_ep:
        from atom.plugin.vllm.tp_group_reuse import init_aiter_dist_from_vllm

        if init_aiter_dist_from_vllm(tensor_parallel_size):
            return

    # Fallback: create aiter's own groups (vLLM reuse failed or non-vLLM plugin)
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.sglang_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = "127.0.0.1"
            dp_master_port = config.plugin_config.sglang_port_args.nccl_port
    elif config.plugin_config.is_rtpllm:
        import os

        dp_master_ip = os.getenv("MASTER_ADDR", "127.0.0.1")
        dp_master_port = int(os.getenv("MASTER_PORT", "29500"))

    distributed_init_method = get_distributed_init_method(dp_master_ip, dp_master_port)

    logger.info(
        f"Initialize aiter dist for using aiter custom collective op for plugin mode, rank:{rank}"
    )
    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )
