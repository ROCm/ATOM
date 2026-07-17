import logging
import os

from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.models.glm4_moe import Glm4MoeForCausalLM
from atom.models.deepseek_v2 import DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM
from atom.models.minimax_m2 import MiniMaxM2ForCausalLM
from atom.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from atom.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGenerationTextOnly,
    Qwen3_5ForConditionalGenerationTextOnly,
)
from atom.config import Config
from atom.plugin.prepare import is_vllm, is_sglang, is_rtpllm

logger = logging.getLogger("atom")


def _is_current_stream_capturing(torch_mod) -> bool:
    try:
        cuda_mod = getattr(torch_mod, "cuda", None)
        is_capturing = getattr(cuda_mod, "is_current_stream_capturing", None)
        return bool(is_capturing()) if is_capturing is not None else False
    except Exception:
        return False

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
    from atom.models.qwen3_next import Qwen3NextForCausalLM
    from atom.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )
    from atom.models.kimi_k25 import KimiK25ForCausalLM

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

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
        ATOMAttnBackendForSgl,
    )
    from atom.plugin.sglang.attention_backend.deepseek_v4_backend import (
        ATOMDeepseekV4BackendForSgl,
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
    except Exception as exc:
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
        import torch
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_worker_v2 import (
            EAGLEWorkerV2,
            EagleDraftWorker,
        )
    except Exception as exc:
        logger.debug("Skip patching SGLang DSV4 spec cuda graph: %s", exc)
        return
    from atom.plugin.sglang.runtime.model_arch import (
        is_glm52_dsa_config as _is_glm52_dsa_config,
    )

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
        except Exception:
            return False

    def _is_glm52_nextn_runner(runner) -> bool:
        try:
            arches = (
                getattr(
                    getattr(getattr(runner, "model_config", None), "hf_config", None),
                    "architectures",
                    None,
                )
                or []
            )
            return any(
                "GlmMoeDsaForCausalLMNextN" in str(arch)
                or "DeepseekV3ForCausalLMNextN" in str(arch)
                for arch in arches
            )
        except Exception:
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
        except Exception:
            return False

    def _is_glm52_runner(runner) -> bool:
        try:
            hf_config = getattr(getattr(runner, "model_config", None), "hf_config", None)
            arches = getattr(hf_config, "architectures", None) or []
            model_path = str(
                getattr(getattr(runner, "server_args", None), "model_path", "")
                or getattr(getattr(runner, "model_config", None), "path", "")
            )
            return (
                _is_glm52_dsa_config(hf_config)
                or any("GlmMoeDsa" in str(arch) for arch in arches)
                or "GLM-5.2" in model_path
                or "glm-5.2" in model_path.lower()
            )
        except Exception:
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
        is_draft_extend = bool(
            getattr(mode, "is_draft_extend", lambda **kwargs: False)(include_v2=True)
        )
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

    def _tensor_head(tensor, limit=12, as_float=False):
        if not torch.is_tensor(tensor):
            return None
        value = tensor.reshape(-1)[: min(limit, int(tensor.numel()))].detach()
        if as_float:
            value = value.float()
        return value.cpu().tolist()

    def _hidden_probe(hidden_states, rows=4):
        if not torch.is_tensor(hidden_states):
            return None
        hidden_rows = hidden_states[: min(rows, int(hidden_states.shape[0]))].detach().float()
        dim = int(hidden_rows.shape[-1])
        checksum_slices = []
        for start in (0, 256, 1024, 2048, 4096, max(0, dim - 256)):
            end = min(dim, start + 256)
            if start < end:
                checksum_slices.append(hidden_rows[:, start:end].sum(dim=-1))
        checksum = (
            torch.stack(checksum_slices, dim=-1)
            if checksum_slices
            else hidden_rows.new_empty((hidden_rows.shape[0], 0))
        )
        return {
            "shape": tuple(hidden_states.shape),
            "norm": hidden_rows.norm(dim=-1).cpu().tolist(),
            "absmax": hidden_rows.abs().amax(dim=-1).cpu().tolist(),
            "mean": hidden_rows.mean(dim=-1).cpu().tolist(),
            "checksum": checksum.cpu().tolist(),
        }

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

    def _runner_probe(obj):
        if obj is None:
            return None
        return {
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "has_load_batch": callable(getattr(obj, "load_batch", None)),
            "has_execute": callable(getattr(obj, "execute", None)),
            "has_replay": callable(getattr(obj, "replay", None)),
            "has_can_run": callable(getattr(obj, "can_run", None)),
            "has_can_run_graph": callable(getattr(obj, "can_run_graph", None)),
            "has_capture_bs": hasattr(obj, "capture_bs"),
            "has_output_buffers": hasattr(obj, "output_buffers"),
            "has_buffers": hasattr(obj, "buffers"),
        }

    def _log_glm52_graph_runner_probe(where: str, model_runner, self_runner=None) -> None:
        if not _env_flag("ATOM_GLM52_GRAPH_RUNNER_PROBE"):
            return
        try:
            attrs = {}
            for name in (
                "graph_runner",
                "cuda_graph_runner",
                "decode_cuda_graph_runner",
                "draft_cuda_graph_runner",
                "cuda_graph_runner_for_draft_extend",
            ):
                if hasattr(model_runner, name):
                    attrs[name] = _runner_probe(getattr(model_runner, name, None))
            logger.warning(
                "GLM52 graph runner probe: where=%s self=%s model_runner=%s attrs=%s",
                where,
                _runner_probe(self_runner),
                _runner_probe(model_runner),
                attrs,
            )
        except Exception:
            logger.exception("Failed to log GLM52 graph runner probe")

    if not getattr(CudaGraphRunner, "_atom_dsv4_init_patched", False):
        original_target_init = CudaGraphRunner.__init__

        def __init__(self, model_runner, *args, **kwargs):
            should_cap = False
            server_args = getattr(model_runner, "server_args", None)
            original_cuda_graph_bs = (
                list(getattr(server_args, "cuda_graph_bs", []))
                if server_args is not None
                else None
            )
            should_force_glm_hidden = False
            original_enable_return_hidden_states = (
                getattr(server_args, "enable_return_hidden_states", None)
                if server_args is not None
                else None
            )
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
                should_force_glm_hidden = (
                    _is_glm52_runner(model_runner)
                    and bool(
                        getattr(
                            getattr(model_runner, "spec_algorithm", None),
                            "is_speculative",
                            lambda: False,
                        )()
                    )
                    and not getattr(model_runner, "is_draft_worker", False)
                )
            except Exception:
                should_cap = False
                should_force_glm_hidden = False

            try:
                if should_cap and server_args is not None and original_cuda_graph_bs:
                    server_args.cuda_graph_bs = _safe_spec_graph_bs(
                        original_cuda_graph_bs,
                        "ATOM_SGLANG_V4_TARGET_VERIFY_CG_BS",
                    )
                if should_force_glm_hidden and server_args is not None:
                    # GLM MTP needs verifier hidden states to seed the next
                    # draft step. Capture target graph in FULL hidden mode
                    # from startup so graph replay matches eager semantics.
                    server_args.enable_return_hidden_states = True
                original_target_init(self, model_runner, *args, **kwargs)
                if should_force_glm_hidden:
                    _log_glm52_graph_runner_probe(
                        "CudaGraphRunner.__init__", model_runner, self
                    )
            finally:
                if (
                    should_cap
                    and server_args is not None
                    and original_cuda_graph_bs is not None
                ):
                    server_args.cuda_graph_bs = original_cuda_graph_bs
                if (
                    should_force_glm_hidden
                    and server_args is not None
                    and original_enable_return_hidden_states is not None
                ):
                    server_args.enable_return_hidden_states = (
                        original_enable_return_hidden_states
                    )

        CudaGraphRunner.__init__ = __init__
        CudaGraphRunner._atom_dsv4_init_patched = True

    if not getattr(CudaGraphRunner, "_atom_dsv4_spec_can_run_patched", False):
        original_can_run = CudaGraphRunner.can_run

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
                is_draft_extend = bool(
                    getattr(mode, "is_draft_extend", lambda **kwargs: False)(
                        include_v2=True
                    )
                )
                if is_dsv4 and is_target_verify and not _target_verify_graph_enabled():
                    return False
                if is_dsv4 and is_draft_extend:
                    return False
            except Exception:
                pass
            return original_can_run(self, forward_batch)

        CudaGraphRunner.can_run = can_run
        CudaGraphRunner._atom_dsv4_spec_can_run_patched = True

    if not getattr(CudaGraphRunner, "_atom_glm52_io_debug_patched", False):
        original_load_batch = getattr(CudaGraphRunner, "load_batch", None)
        original_execute = getattr(CudaGraphRunner, "execute", None)
        original_replay = getattr(CudaGraphRunner, "replay", None)

        def _mode_is_target_verify(forward_batch) -> bool:
            mode = getattr(forward_batch, "forward_mode", None)
            return bool(mode is not None and getattr(mode, "is_target_verify", lambda: False)())

        def _should_log_glm52_target_graph(self, forward_batch) -> bool:
            return (
                (
                    _env_flag("ATOM_GLM52_VERIFY_DEBUG")
                    or _env_flag("ATOM_GLM52_ATTENTION_DEBUG_LOG")
                )
                and _is_glm52_runner(getattr(self, "model_runner", None))
                and _mode_is_target_verify(forward_batch)
            )

        def _log_glm52_target_graph_input(self, forward_batch, where: str) -> None:
            buffers = getattr(self, "buffers", None)
            logger.info(
                "GLM52 target graph input debug: where=%s raw_bs=%s bs=%s "
                "raw_tokens=%s input_ids=%s positions=%s out_cache=%s "
                "seq_lens=%s req_pool=%s spec_positions=%s",
                where,
                getattr(self, "raw_bs", None),
                getattr(self, "bs", None),
                getattr(self, "raw_num_token", None),
                _tensor_head(getattr(buffers, "input_ids", None)),
                _tensor_head(getattr(buffers, "positions", None)),
                _tensor_head(getattr(buffers, "out_cache_loc", None)),
                _tensor_head(getattr(buffers, "seq_lens", None)),
                _tensor_head(getattr(buffers, "req_pool_indices", None)),
                _tensor_head(
                    getattr(getattr(forward_batch, "spec_info", None), "positions", None)
                ),
            )

        def _log_glm52_target_graph_output(out, where: str) -> None:
            logits = getattr(out, "next_token_logits", None)
            logger.info(
                "GLM52 target graph output debug: where=%s hidden=%s logits_shape=%s "
                "logits_head=%s",
                where,
                _hidden_probe(getattr(out, "hidden_states", None)),
                tuple(logits.shape) if torch.is_tensor(logits) else None,
                _tensor_head(logits, limit=8, as_float=True),
            )

        def _log_glm52_attention_debug_buffers(
            model_runner, where: str, forward_batch=None, runner=None
        ) -> None:
            if not _env_flag("ATOM_GLM52_ATTENTION_DEBUG_LOG"):
                return
            try:
                buffers = getattr(runner, "buffers", None)
                spec_info = getattr(forward_batch, "spec_info", None)
                context = {
                    "input_ids": _tensor_head(getattr(forward_batch, "input_ids", None)),
                    "positions": _tensor_head(
                        getattr(spec_info, "positions", None)
                        if getattr(spec_info, "positions", None) is not None
                        else getattr(forward_batch, "positions", None)
                    ),
                    "out_cache_loc": _tensor_head(
                        getattr(forward_batch, "out_cache_loc", None)
                    ),
                    "seq_lens": _tensor_head(getattr(forward_batch, "seq_lens", None)),
                    "req_pool_indices": _tensor_head(
                        getattr(forward_batch, "req_pool_indices", None)
                    ),
                    "buffer_input_ids": _tensor_head(getattr(buffers, "input_ids", None))
                    if buffers is not None
                    else None,
                    "buffer_positions": _tensor_head(getattr(buffers, "positions", None))
                    if buffers is not None
                    else None,
                    "buffer_out_cache_loc": _tensor_head(
                        getattr(buffers, "out_cache_loc", None)
                    )
                    if buffers is not None
                    else None,
                    "buffer_seq_lens": _tensor_head(getattr(buffers, "seq_lens", None))
                    if buffers is not None
                    else None,
                    "buffer_req_pool_indices": _tensor_head(
                        getattr(buffers, "req_pool_indices", None)
                    )
                    if buffers is not None
                    else None,
                    "raw_bs": getattr(runner, "raw_bs", None),
                    "bs": getattr(runner, "bs", None),
                    "raw_tokens": getattr(runner, "raw_num_token", None),
                }
                positions_head = context.get("positions") or []
                input_ids_head = context.get("input_ids") or []
                raw_bs = int(context.get("raw_bs") or 0)
                if raw_bs <= 0 or (
                    positions_head
                    and all(int(x) == 0 for x in positions_head)
                    and input_ids_head
                    and all(int(x) == 0 for x in input_ids_head)
                ):
                    return
                configured = os.environ.get("ATOM_GLM52_ATTENTION_DEBUG_LAYERS", "")
                configured_layers = None
                if configured.strip().lower() not in ("all", "*"):
                    configured_layers = {
                        int(item)
                        for item in configured.replace(" ", ",").split(",")
                        if item.strip()
                    }
                model = getattr(model_runner, "model", None)
                if model is None or not hasattr(model, "modules"):
                    return
                collected = []
                for module in model.modules():
                    buf = getattr(module, "_atom_glm52_attn_debug", None)
                    if not torch.is_tensor(buf):
                        continue
                    layer = getattr(module, "layer_num", None)
                    if (
                        configured_layers is not None
                        and int(layer) not in configured_layers
                    ):
                        continue
                    sparse_buf = getattr(module, "sparse_kv_indices_buffer", None)
                    sparse_info = None
                    if torch.is_tensor(sparse_buf):
                        flat_sparse = sparse_buf.reshape(-1)
                        sparse_info = {
                            "shape": tuple(sparse_buf.shape),
                            "head": flat_sparse[: min(16, int(flat_sparse.numel()))]
                            .detach()
                            .cpu()
                            .tolist(),
                            "tail": flat_sparse[
                                max(0, int(flat_sparse.numel()) - 16) :
                            ]
                            .detach()
                            .cpu()
                            .tolist(),
                        }
                    collected.append(
                        {
                            "layer": int(layer) if layer is not None else None,
                            "values": buf.detach().cpu().tolist(),
                            "sparse": sparse_info,
                        }
                    )
                if collected:
                    logger.info(
                        "GLM52 attention layer debug: where=%s context=%s values=%s",
                        where,
                        context,
                        collected,
                    )
            except Exception:
                logger.exception("Failed to log GLM52 attention layer debug")

        def load_batch(self, forward_batch, *args, **kwargs):
            ret = original_load_batch(self, forward_batch, *args, **kwargs)
            try:
                if _should_log_glm52_target_graph(self, forward_batch):
                    _log_glm52_target_graph_input(self, forward_batch, "load_batch")
            except Exception:
                logger.exception("Failed to log GLM52 target graph input debug")
            return ret

        def execute(self, forward_batch, *args, **kwargs):
            out = original_execute(self, forward_batch, *args, **kwargs)
            try:
                if _should_log_glm52_target_graph(self, forward_batch):
                    _log_glm52_target_graph_output(out, "execute")
            except Exception:
                logger.exception("Failed to log GLM52 target graph output debug")
            return out

        def replay(self, forward_batch, *args, **kwargs):
            should_log = False
            try:
                should_log = _should_log_glm52_target_graph(self, forward_batch)
                if should_log:
                    _log_glm52_graph_runner_probe(
                        "CudaGraphRunner.replay",
                        getattr(self, "model_runner", None),
                        self,
                    )
                    _log_glm52_target_graph_input(self, forward_batch, "replay")
            except Exception:
                logger.exception("Failed to log GLM52 target graph replay input debug")
            out = original_replay(self, forward_batch, *args, **kwargs)
            try:
                if should_log:
                    _log_glm52_target_graph_output(out, "replay")
                    _log_glm52_attention_debug_buffers(
                        getattr(self, "model_runner", None),
                        "replay",
                        forward_batch=forward_batch,
                        runner=self,
                    )
            except Exception:
                logger.exception("Failed to log GLM52 target graph replay output debug")
            return out

        if original_load_batch is not None and original_execute is not None:
            CudaGraphRunner.load_batch = load_batch
            CudaGraphRunner.execute = execute
        if original_replay is not None:
            CudaGraphRunner.replay = replay
        if (
            (original_load_batch is not None and original_execute is not None)
            or original_replay is not None
        ):
            CudaGraphRunner._atom_glm52_io_debug_patched = True

    if not getattr(EAGLEDraftCudaGraphRunner, "_atom_dsv4_replay_patched", False):
        original_draft_replay = EAGLEDraftCudaGraphRunner.replay
        original_draft_execute = getattr(EAGLEDraftCudaGraphRunner, "execute", None)

        def replay(self, forward_batch):
            if not (
                _is_dsv4_nextn_runner(getattr(self, "model_runner", None))
                or _is_glm52_nextn_runner(getattr(self, "model_runner", None))
            ):
                return original_draft_replay(self, forward_batch)
            if _env_flag("ATOM_SGLANG_V4_DISABLE_DRAFT_CG"):
                raise RuntimeError(
                    "DSV4 draft cuda graph replay was disabled after capture; "
                    "disable it before graph initialization instead."
                )
            original_hidden_states = _flatten_spec_hidden_states(forward_batch)
            try:
                try:
                    if os.path.exists("/tmp/atom_glm52_draft_debug_on"):
                        import torch

                        spec_info = getattr(forward_batch, "spec_info", None)
                        logger.info(
                            "GLM52 draft graph seed debug: bs=%s topk_index_head=%s "
                            "topk_p_head=%s hidden_probe=%s out_cache_loc_head=%s "
                            "positions_head=%s",
                            getattr(forward_batch, "batch_size", None),
                            _tensor_head(getattr(spec_info, "topk_index", None)),
                            _tensor_head(getattr(spec_info, "topk_p", None), as_float=True),
                            _hidden_probe(getattr(spec_info, "hidden_states", None)),
                            _tensor_head(getattr(forward_batch, "out_cache_loc", None)),
                            _tensor_head(getattr(forward_batch, "positions", None)),
                        )
                except Exception:
                    logger.exception("Failed to log GLM52 draft graph seed debug")
                out = original_draft_replay(self, forward_batch)
                try:
                    if os.path.exists("/tmp/atom_glm52_draft_debug_on"):
                        if len(out) == 4:
                            parent_list, top_scores_index, draft_tokens, draft_probs = out
                        else:
                            parent_list, top_scores_index, draft_tokens = out
                            draft_probs = None

                        def _head(tensor, rows=4):
                            if not torch.is_tensor(tensor):
                                return None
                            return (
                                tensor[: min(rows, int(tensor.shape[0]))]
                                .detach()
                                .cpu()
                                .tolist()
                            )

                        logger.info(
                            "GLM52 draft graph replay debug: parent_head=%s "
                            "top_scores_index_head=%s draft_tokens_head=%s "
                            "draft_probs_head=%s",
                            _head(parent_list),
                            _head(top_scores_index),
                            _head(draft_tokens),
                            _head(draft_probs),
                        )
                except Exception:
                    logger.exception("Failed to log GLM52 draft graph replay debug")
                return out
            finally:
                if original_hidden_states is not None:
                    forward_batch.spec_info.hidden_states = original_hidden_states

        EAGLEDraftCudaGraphRunner.replay = replay
        if original_draft_execute is not None:

            def execute(self, forward_batch):
                out = original_draft_execute(self, forward_batch)
                try:
                    if os.path.exists("/tmp/atom_glm52_draft_debug_on"):
                        parent_list, top_scores_index, draft_tokens, draft_probs = out

                        def _head(tensor, rows=4):
                            if not torch.is_tensor(tensor):
                                return None
                            return (
                                tensor[: min(rows, int(tensor.shape[0]))]
                                .detach()
                                .cpu()
                                .tolist()
                            )

                        logger.info(
                            "GLM52 draft graph execute debug: raw_bs=%s bs=%s "
                            "parent_head=%s top_scores_index_head=%s "
                            "draft_tokens_head=%s draft_probs_head=%s",
                            getattr(self, "raw_bs", None),
                            getattr(self, "bs", None),
                            _head(parent_list),
                            _head(top_scores_index),
                            _head(draft_tokens),
                            _head(draft_probs),
                        )
                except Exception:
                    logger.exception("Failed to log GLM52 draft graph execute debug")
                return out

            EAGLEDraftCudaGraphRunner.execute = execute
        EAGLEDraftCudaGraphRunner._atom_dsv4_replay_patched = True

    try:
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
    except Exception:
        EagleVerifyInput = None

    if EagleVerifyInput is not None and not getattr(
        EagleVerifyInput, "_atom_glm52_sample_debug_patched", False
    ):
        original_sample = EagleVerifyInput.sample

        def sample(self, batch, logits_output, vocab_mask=None):
            if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in (
                "1",
                "true",
                "True",
            ):
                try:
                    import torch

                    bs = len(batch.seq_lens)
                    next_token_logits = logits_output.next_token_logits
                    target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
                        bs, self.draft_token_num
                    )
                    candidates = self.draft_token.reshape(bs, self.draft_token_num)
                    capturing = _is_current_stream_capturing(torch)
                    logits_probe = "<cuda_graph_capture>"
                    cand_logits_probe = "<cuda_graph_capture>"
                    hidden_probe = "<cuda_graph_capture>"
                    metadata_probe = "<cuda_graph_capture>"
                    if not capturing:
                        probe_rows = min(
                            int(next_token_logits.shape[0]),
                            max(1, min(2, bs)) * int(self.draft_token_num),
                        )
                        top_vals, top_ids = torch.topk(
                            next_token_logits[:probe_rows], k=3, dim=-1
                        )
                        cand_flat = candidates.reshape(-1)[:probe_rows].to(
                            next_token_logits.device
                        )
                        cand_logits = next_token_logits[:probe_rows].gather(
                            1, cand_flat[:, None]
                        )
                        logits_probe = {
                            "top_ids": top_ids.detach().cpu().tolist(),
                            "top_vals": top_vals.detach().float().cpu().tolist(),
                        }
                        cand_logits_probe = cand_logits.detach().float().cpu().tolist()
                        hidden_states = getattr(logits_output, "hidden_states", None)
                        if torch.is_tensor(hidden_states):
                            hidden_rows = hidden_states[:probe_rows].detach().float()
                            dim = int(hidden_rows.shape[-1])
                            checksum_slices = []
                            for start in (0, 256, 1024, 2048, 4096, max(0, dim - 256)):
                                end = min(dim, start + 256)
                                if start < end:
                                    checksum_slices.append(
                                        hidden_rows[:, start:end].sum(dim=-1)
                                    )
                            checksum = (
                                torch.stack(checksum_slices, dim=-1)
                                if checksum_slices
                                else hidden_rows.new_empty((hidden_rows.shape[0], 0))
                            )
                            sample_cols = [
                                c for c in (0, 1, 2, 3, 7, 31, 127, 511, 1023, 2047, 4095, dim - 1)
                                if 0 <= c < dim
                            ]
                            hidden_probe = {
                                "shape": tuple(hidden_states.shape),
                                "norm": hidden_rows.norm(dim=-1).cpu().tolist(),
                                "absmax": hidden_rows.abs().amax(dim=-1).cpu().tolist(),
                                "mean": hidden_rows.mean(dim=-1).cpu().tolist(),
                                "checksum": checksum.cpu().tolist(),
                                "sample_cols": sample_cols,
                                "sample_vals": hidden_rows[:, sample_cols]
                                .cpu()
                                .tolist()
                                if sample_cols
                                else [],
                            }
                        else:
                            hidden_probe = None
                        metadata_probe = {
                            "counter": getattr(
                                self, "_atom_glm52_verify_counter", None
                            ),
                            "row_probe": getattr(self, "_atom_glm52_row_probe", None),
                        }
                    logger.info(
                        "GLM52 verify sample debug: bs=%s draft_token_num=%s "
                        "logits_shape=%s candidates_head=%s target_predict_head=%s "
                        "seq_lens_head=%s top3_probe=%s cand_logits_probe=%s "
                        "hidden_probe=%s metadata_probe=%s",
                        bs,
                        self.draft_token_num,
                        tuple(next_token_logits.shape),
                        "<cuda_graph_capture>"
                        if capturing
                        else candidates[: min(2, bs)].detach().cpu().tolist(),
                        "<cuda_graph_capture>"
                        if capturing
                        else target_predict[: min(2, bs)].detach().cpu().tolist(),
                        (
                            "<cuda_graph_capture>"
                            if capturing
                            else batch.seq_lens[: min(8, int(batch.seq_lens.numel()))]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        if torch.is_tensor(batch.seq_lens)
                        else None,
                        logits_probe,
                        cand_logits_probe,
                        hidden_probe,
                        metadata_probe,
                    )
                except Exception:
                    logger.exception("Failed to log GLM52 verify sample debug")
            return original_sample(self, batch, logits_output, vocab_mask)

        EagleVerifyInput.sample = sample
        EagleVerifyInput._atom_glm52_sample_debug_patched = True

    if not getattr(EAGLEWorkerV2, "_atom_glm52_verify_kv_debug_patched", False):
        original_verify = EAGLEWorkerV2.verify

        def verify(self, batch):
            forced_tokens = os.environ.get("ATOM_GLM52_FORCE_VERIFY_DRAFT_TOKENS", "")
            if forced_tokens.strip():
                try:
                    target_runner = getattr(
                        getattr(self, "target_worker", None), "model_runner", None
                    )
                    if _is_glm52_runner(target_runner):
                        tokens = [
                            int(item)
                            for item in forced_tokens.replace(" ", ",").split(",")
                            if item.strip()
                        ]
                        verify_input = getattr(batch, "spec_info", None)
                        draft_token = getattr(verify_input, "draft_token", None)
                        draft_token_num = int(
                            getattr(verify_input, "draft_token_num", 0)
                            or len(tokens)
                        )
                        bs = int(getattr(batch, "batch_size", 0) or len(batch.seq_lens))
                        if (
                            tokens
                            and torch.is_tensor(draft_token)
                            and draft_token_num > 0
                            and int(draft_token.numel()) >= bs * draft_token_num
                        ):
                            row = torch.tensor(
                                tokens[:draft_token_num],
                                dtype=draft_token.dtype,
                                device=draft_token.device,
                            )
                            if int(row.numel()) < draft_token_num:
                                row = torch.nn.functional.pad(
                                    row,
                                    (0, draft_token_num - int(row.numel())),
                                    value=int(row[-1].item()),
                                )
                            draft_token.view(bs, draft_token_num)[:, :].copy_(
                                row[None, :]
                            )
                            logger.info(
                                "GLM52 forced verify draft tokens: bs=%s "
                                "draft_token_num=%s tokens=%s",
                                bs,
                                draft_token_num,
                                row.detach().cpu().tolist(),
                            )
                except Exception:
                    logger.exception("Failed to force GLM52 verify draft tokens")
            out = original_verify(self, batch)
            if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in (
                "1",
                "true",
                "True",
            ):
                try:
                    req_pool_indices = getattr(batch, "req_pool_indices", None)
                    req_to_token_pool = getattr(self, "req_to_token_pool", None)
                    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
                    seq_lens = getattr(batch, "seq_lens", None)
                    new_seq_lens = getattr(out, "new_seq_lens", None)
                    tail_probe = []
                    if (
                        torch.is_tensor(req_pool_indices)
                        and torch.is_tensor(req_to_token)
                        and torch.is_tensor(seq_lens)
                    ):
                        probe_bs = min(4, int(req_pool_indices.numel()))
                        for row in range(probe_bs):
                            req_idx = int(req_pool_indices[row].detach().cpu())
                            old_len = int(seq_lens[row].detach().cpu())
                            new_len = (
                                int(new_seq_lens[row].detach().cpu())
                                if torch.is_tensor(new_seq_lens)
                                else old_len
                            )
                            start = max(0, old_len - 8)
                            end = min(req_to_token.shape[1], new_len + 4)
                            tail_probe.append(
                                {
                                    "row": row,
                                    "req": req_idx,
                                    "old_len": old_len,
                                    "new_len": new_len,
                                    "tokens": req_to_token[req_idx, start:end]
                                    .detach()
                                    .cpu()
                                    .tolist(),
                                }
                            )
                    logits_output = getattr(out, "logits_output", None)
                    next_draft_input = getattr(out, "next_draft_input", None)
                    logger.info(
                        "GLM52 verify kv debug: accept_lens=%s new_seq_lens=%s "
                        "next_token_ids_head=%s tail_probe=%s logits_hidden=%s "
                        "next_draft_hidden=%s next_draft_topk=%s next_draft_p=%s "
                        "can_run_cuda_graph=%s",
                        _tensor_head(getattr(out, "accept_lens", None)),
                        _tensor_head(new_seq_lens),
                        _tensor_head(getattr(out, "next_token_ids", None)),
                        tail_probe,
                        _hidden_probe(getattr(logits_output, "hidden_states", None)),
                        _hidden_probe(getattr(next_draft_input, "hidden_states", None)),
                        _tensor_head(getattr(next_draft_input, "topk_index", None)),
                        _tensor_head(
                            getattr(next_draft_input, "topk_p", None), as_float=True
                        ),
                        getattr(out, "can_run_cuda_graph", None),
                    )
                except Exception:
                    logger.exception("Failed to log GLM52 verify kv debug")
            return out

        EAGLEWorkerV2.verify = verify
        EAGLEWorkerV2._atom_glm52_verify_kv_debug_patched = True

    if not getattr(EAGLEDraftExtendCudaGraphRunner, "_atom_dsv4_replay_patched", False):
        original_extend_replay = EAGLEDraftExtendCudaGraphRunner.replay
        original_extend_can_run = EAGLEDraftExtendCudaGraphRunner.can_run

        def _dsv4_draft_extend_graph_layout_ok(runner, forward_batch=None):
            try:
                num_draft_tokens = int(getattr(runner, "num_tokens_per_bs", 0) or 0)
                if num_draft_tokens <= 0:
                    return False
                raw_bs = int(getattr(forward_batch, "batch_size", 0) or 0)
                if raw_bs <= 0:
                    raw_bs = min(getattr(runner, "capture_bs", [0]) or [0])
                if raw_bs <= 0:
                    return False
                if forward_batch is not None and getattr(
                    runner, "require_mlp_tp_gather", False
                ):
                    max_num_tokens = max(forward_batch.global_num_tokens_cpu)
                    max_batch_size = max_num_tokens // num_draft_tokens
                else:
                    max_batch_size = raw_bs
                import bisect

                index = bisect.bisect_left(runner.capture_bs, max_batch_size)
                if index >= len(runner.capture_bs):
                    return False
                bs = runner.capture_bs[index]
                output = runner.output_buffers.get(bs)
                logits = getattr(output, "next_token_logits", None)
                expected = bs * num_draft_tokens
                if logits is None or int(logits.shape[0]) < expected:
                    return False
                return True
            except Exception:
                return False

        def can_run(self, forward_batch):
            if not _is_dsv4_nextn_runner(getattr(self, "model_runner", None)):
                return original_extend_can_run(self, forward_batch)
            if not original_extend_can_run(self, forward_batch):
                return False
            return _dsv4_draft_extend_graph_layout_ok(self, forward_batch)

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
            previous_replay_batch = (
                getattr(backend, "_replay_forward_batch", None)
                if backend is not None
                else None
            )
            try:
                if backend is not None:
                    backend._atom_dsv4_draft_extend_graph_runner = self
                    buffers = getattr(self, "buffers", None)
                    input_ids = getattr(forward_batch, "input_ids", None)
                    num_tokens = (
                        int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
                    )
                    if buffers is not None and num_tokens > 0:
                        from types import SimpleNamespace

                        backend._replay_forward_batch = SimpleNamespace(
                            forward_mode=getattr(forward_batch, "forward_mode", None),
                            positions=getattr(buffers, "positions", None)[:num_tokens],
                            out_cache_loc=getattr(buffers, "out_cache_loc", None)[
                                :num_tokens
                            ],
                        )
                out = original_extend_replay(self, forward_batch)
                try:
                    # EAGLE V2 consumes draft-extend logits with a fixed
                    # `seq * speculative_num_draft_tokens + offset` layout.
                    # SGLang's runner trims to the actual compact token count,
                    # which makes that indexing OOB when fewer than the padded
                    # graph tokens were materialized.  Return the captured
                    # padded output buffer for DSV4 so downstream indexing stays
                    # within the fixed graph layout.
                    if bool(
                        getattr(
                            getattr(self, "forward_mode", None),
                            "is_draft_extend_v2",
                            lambda: False,
                        )()
                    ):
                        padded_out = getattr(self, "output_buffers", {}).get(
                            getattr(self, "bs", None)
                        )
                        if padded_out is not None:
                            out = padded_out
                except Exception:
                    logger.exception(
                        "Failed to restore padded DSV4 draft-extend graph output"
                    )
                return out
            finally:
                if backend is not None:
                    if previous_runner is None:
                        try:
                            delattr(backend, "_atom_dsv4_draft_extend_graph_runner")
                        except AttributeError:
                            pass
                    else:
                        backend._atom_dsv4_draft_extend_graph_runner = previous_runner
                    if previous_replay_batch is None:
                        try:
                            delattr(backend, "_replay_forward_batch")
                        except AttributeError:
                            pass
                    else:
                        backend._replay_forward_batch = previous_replay_batch
                if original_hidden_states is not None:
                    forward_batch.spec_info.hidden_states = original_hidden_states

        EAGLEDraftExtendCudaGraphRunner.can_run = can_run
        EAGLEDraftExtendCudaGraphRunner.replay = replay
        EAGLEDraftExtendCudaGraphRunner._atom_dsv4_replay_patched = True

    if not getattr(EagleDraftWorker, "_atom_dsv4_draft_extend_accept_patched", False):
        original_draft_extend_for_decode = EagleDraftWorker._draft_extend_for_decode

        def _draft_extend_for_decode(self, batch, batch_result):
            try:
                is_fixed_nextn = _is_dsv4_nextn_runner(
                    getattr(self, "draft_runner", None)
                ) or _is_glm52_nextn_runner(getattr(self, "draft_runner", None))
                if (
                    not is_fixed_nextn
                ):
                    return original_draft_extend_for_decode(self, batch, batch_result)

                import torch
                from sglang.srt.speculative.eagle_info import EagleDraftInput
                from sglang.srt.speculative.spec_utils import fast_topk

                num_draft_tokens = int(
                    getattr(self, "speculative_num_draft_tokens", 0)
                    or getattr(self.server_args, "speculative_num_draft_tokens", 0)
                    or 0
                )
                if num_draft_tokens <= 0:
                    return original_draft_extend_for_decode(self, batch, batch_result)

                draft_extend_graph_runner = getattr(
                    self, "cuda_graph_runner_for_draft_extend", None
                )
                if draft_extend_graph_runner is not None and not _dsv4_draft_extend_graph_layout_ok(
                    draft_extend_graph_runner
                ):
                    runner = draft_extend_graph_runner
                    self.cuda_graph_runner_for_draft_extend = None
                    try:
                        return original_draft_extend_for_decode(
                            self, batch, batch_result
                        )
                    finally:
                        self.cuda_graph_runner_for_draft_extend = runner

                accept_lens = getattr(batch_result, "accept_lens", None)
                if not torch.is_tensor(accept_lens):
                    return original_draft_extend_for_decode(self, batch, batch_result)

                # DRAFT_EXTEND_V2 materializes a fixed `num_draft_tokens` slots
                # per sequence.  SGLang's default compact `cumsum(accept_lens)-1`
                # index aliases rows from neighboring requests in this layout.
                # `accept_lens` includes the target bonus token, so clamp before
                # converting it to a fixed-layout per-request row offset.
                graph_accept_lens = accept_lens.clamp(min=1, max=num_draft_tokens)

                draft_input = EagleDraftInput(
                    hidden_states=batch_result.logits_output.hidden_states,
                    num_tokens_per_req=self.speculative_num_steps + 1,
                    num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
                )
                select_index = (
                    torch.arange(len(batch.seq_lens), device=self.device)
                    * num_draft_tokens
                    + graph_accept_lens
                    - 1
                )

                with self.plan_stream_ctx:
                    forward_batch = (
                        draft_input.prepare_for_extend_to_fill_draft_kvcache(
                            batch,
                            batch_result.next_token_ids,
                            num_draft_tokens,
                            self.draft_runner,
                            draft_extend_graph_runner,
                        )
                    )

                if self.plan_stream:
                    torch.get_device_module(self.device).current_stream().wait_stream(
                        self.plan_stream
                    )

                # The graph only fills draft slots.  Keep the scheduler-facing
                # `batch_result.accept_lens` untouched, but make the graph's
                # per-sequence counts match the fixed draft-token layout.
                forward_batch.spec_info.num_correct_drafts = graph_accept_lens - 1
                forward_batch.spec_info.num_accept_tokens = graph_accept_lens

                can_cuda_graph = (
                    draft_extend_graph_runner
                    and draft_extend_graph_runner.can_run(forward_batch)
                )
                if can_cuda_graph:
                    draft_logits_output = (
                        draft_extend_graph_runner.replay(forward_batch)
                    )
                else:
                    draft_logits_output = self.draft_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output

                output_len = int(draft_logits_output.next_token_logits.shape[0])
                max_index = (
                    int(select_index.max().detach().cpu())
                    if select_index.numel()
                    else -1
                )
                if max_index >= output_len and can_cuda_graph:
                    draft_logits_output = self.draft_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output
                    can_cuda_graph = False
                    output_len = int(draft_logits_output.next_token_logits.shape[0])
                if max_index >= output_len:
                    raise RuntimeError(
                        "ATOM DRAFT_EXTEND_V2 output/index layout mismatch: "
                        f"max_index={max_index}, output_len={output_len}, "
                        f"batch={len(batch.seq_lens)}, "
                        f"num_draft_tokens={num_draft_tokens}, "
                        f"can_cuda_graph={bool(can_cuda_graph)}"
                    )

                selected_logits = draft_logits_output.next_token_logits.index_select(
                    0, select_index
                )
                selected_hidden_states = draft_logits_output.hidden_states
                if draft_logits_output.hidden_states is not None:
                    selected_hidden_states = (
                        draft_logits_output.hidden_states.index_select(0, select_index)
                    )

                probs = torch.softmax(selected_logits, dim=-1)
                ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)

                if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in (
                    "1",
                    "true",
                    "True",
                ):
                    try:
                        logger.info(
                            "GLM52 draft_extend fixed debug: is_glm=%s "
                            "accept_lens=%s graph_accept_lens=%s select_index=%s "
                            "target_hidden=%s next_token_ids_head=%s "
                            "output_shape=%s selected_hidden=%s ret_topk=%s ret_p=%s "
                            "can_cuda_graph=%s",
                            _is_glm52_nextn_runner(
                                getattr(self, "draft_runner", None)
                            ),
                            accept_lens.detach().cpu().tolist(),
                            graph_accept_lens.detach().cpu().tolist(),
                            select_index.detach().cpu().tolist(),
                            _hidden_probe(batch_result.logits_output.hidden_states),
                            _tensor_head(batch_result.next_token_ids),
                            tuple(draft_logits_output.next_token_logits.shape),
                            _hidden_probe(selected_hidden_states),
                            _tensor_head(ret_topk_index),
                            _tensor_head(ret_topk_p, as_float=True),
                            bool(can_cuda_graph),
                        )
                    except Exception:
                        logger.exception("Failed to log GLM52 draft_extend fixed debug")

                next_draft_input = batch_result.next_draft_input
                (
                    next_draft_input.topk_p,
                    next_draft_input.topk_index,
                    next_draft_input.hidden_states,
                ) = (
                    ret_topk_p,
                    ret_topk_index,
                    selected_hidden_states,
                )
                return None
            except Exception:
                raise

        EagleDraftWorker._draft_extend_for_decode = _draft_extend_for_decode
        EagleDraftWorker._atom_dsv4_draft_extend_accept_patched = True

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
                    original_cuda_graph_bs = (
                        list(getattr(server_args, "cuda_graph_bs", []))
                        if server_args is not None
                        else None
                    )
                    try:
                        if server_args is not None and original_cuda_graph_bs:
                            server_args.cuda_graph_bs = _safe_spec_graph_bs(
                                original_cuda_graph_bs,
                                "ATOM_SGLANG_V4_DRAFT_EXTEND_CG_BS",
                            )
                        self.cuda_graph_runner_for_draft_extend = (
                            EAGLEDraftExtendCudaGraphRunner(self)
                        )
                    finally:
                        if (
                            server_args is not None
                            and original_cuda_graph_bs is not None
                        ):
                            server_args.cuda_graph_bs = original_cuda_graph_bs
                elif _is_dsv4_nextn_runner(getattr(self, "draft_runner", None)):
                    self.cuda_graph_runner_for_draft_extend = None
            except Exception as exc:
                logger.warning(
                    "Failed to enable DSV4 draft-extend cuda graph in ATOM plugin: %s",
                    exc,
                )
            return ret

        EagleDraftWorker.init_cuda_graphs = init_cuda_graphs
        EagleDraftWorker._atom_dsv4_init_cuda_graphs_patched = True


def _patch_sglang_eagle_v2_draft_argmax() -> None:
    """Use ATOM draft distributed argmax for SGLang EAGLE topk=1 drafting."""
    if os.getenv("ATOM_SGLANG_DRAFT_ARGMAX", "1").lower() in ("0", "false", "no"):
        return
    try:
        import torch
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
        from sglang.srt.speculative.spec_utils import (
            maybe_detect_nan,
            maybe_detect_oob,
            select_top_k_tokens,
        )
    except Exception as exc:
        logger.debug("Skip patching SGLang EAGLE draft argmax: %s", exc)
        return

    if getattr(EagleDraftWorker, "_atom_sglang_draft_argmax_patched", False):
        return

    def draft_forward(self, forward_batch):
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        draft_debug = os.environ.get("ATOM_GLM52_DRAFT_DEBUG", "0") in (
            "1",
            "true",
            "True",
        ) or os.path.exists("/tmp/atom_glm52_draft_debug_on")
        if draft_debug and not _is_current_stream_capturing(torch):
            try:
                hidden_probe = None
                if torch.is_tensor(hidden_states):
                    hidden_rows = hidden_states[
                        : min(4, int(hidden_states.shape[0]))
                    ].detach().float()
                    dim = int(hidden_rows.shape[-1])
                    checksum_slices = []
                    for start in (
                        0,
                        256,
                        1024,
                        2048,
                        4096,
                        max(0, dim - 256),
                    ):
                        end = min(dim, start + 256)
                        if start < end:
                            checksum_slices.append(
                                hidden_rows[:, start:end].sum(dim=-1)
                            )
                    checksum = (
                        torch.stack(checksum_slices, dim=-1)
                        if checksum_slices
                        else hidden_rows.new_empty((hidden_rows.shape[0], 0))
                    )
                    hidden_probe = {
                        "shape": tuple(hidden_states.shape),
                        "norm": hidden_rows.norm(dim=-1).cpu().tolist(),
                        "absmax": hidden_rows.abs().amax(dim=-1).cpu().tolist(),
                        "mean": hidden_rows.mean(dim=-1).cpu().tolist(),
                        "checksum": checksum.cpu().tolist(),
                    }
                logger.info(
                    "GLM52 draft_forward debug: bs=%s topk=%s steps=%s "
                    "topk_index_shape=%s topk_index_head=%s topk_p_head=%s "
                    "hidden_shape=%s hidden_probe=%s out_cache_loc_shape=%s",
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                    tuple(topk_index.shape),
                    topk_index.reshape(-1)[: min(12, int(topk_index.numel()))]
                    .detach()
                    .cpu()
                    .tolist(),
                    topk_p.reshape(-1)[: min(12, int(topk_p.numel()))]
                    .detach()
                    .cpu()
                    .tolist(),
                    tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None,
                    hidden_probe,
                    tuple(out_cache_loc.shape) if torch.is_tensor(out_cache_loc) else None,
                )
            except Exception:
                logger.exception("Failed to log GLM52 draft_forward debug")

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        score_list = []
        token_list = []
        parents_list = []
        scores = None

        use_argmax = self.topk == 1
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            forward_batch._atom_use_draft_argmax = use_argmax
            spec_info.hidden_states = hidden_states

            if draft_debug and not _is_current_stream_capturing(torch):
                try:
                    logger.info(
                        "GLM52 draft step debug: step=%s input_ids_shape=%s "
                        "input_ids_head=%s hidden_shape=%s scores_shape=%s",
                        i,
                        tuple(input_ids.shape),
                        input_ids.reshape(-1)[: min(12, int(input_ids.numel()))]
                        .detach()
                        .cpu()
                        .tolist(),
                        tuple(hidden_states.shape)
                        if torch.is_tensor(hidden_states)
                        else None,
                        tuple(scores.shape) if torch.is_tensor(scores) else None,
                    )
                except Exception:
                    logger.exception("Failed to log GLM52 draft step debug")

            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output

            draft_token_ids = None
            customized_info = getattr(logits_output, "customized_info", None) or {}
            if use_argmax:
                draft_token_ids = customized_info.get("draft_token_ids")

            if draft_token_ids is not None:
                topk_index = draft_token_ids.reshape(-1, 1)
                topk_p = torch.ones(
                    (topk_index.shape[0], 1),
                    dtype=torch.float32,
                    device=topk_index.device,
                )
            else:
                maybe_detect_nan(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                from sglang.srt.utils.common import fast_topk

                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )

            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            if draft_debug and not _is_current_stream_capturing(torch):
                try:
                    logger.info(
                        "GLM52 draft step output debug: step=%s draft_token_ids_head=%s "
                        "topk_index_head=%s topk_p_head=%s logits_shape=%s hidden_shape=%s",
                        i,
                        draft_token_ids.reshape(-1)[
                            : min(12, int(draft_token_ids.numel()))
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(draft_token_ids)
                        else None,
                        topk_index.reshape(-1)[: min(12, int(topk_index.numel()))]
                        .detach()
                        .cpu()
                        .tolist(),
                        topk_p.reshape(-1)[: min(12, int(topk_p.numel()))]
                        .detach()
                        .cpu()
                        .tolist(),
                        tuple(logits_output.next_token_logits.shape)
                        if torch.is_tensor(logits_output.next_token_logits)
                        else None,
                        tuple(logits_output.hidden_states.shape)
                        if torch.is_tensor(logits_output.hidden_states)
                        else None,
                    )
                except Exception:
                    logger.exception("Failed to log GLM52 draft step output debug")
            hidden_states = logits_output.hidden_states
            forward_batch.positions.add_(1)

        score_list = torch.cat(score_list, dim=1).flatten(1)
        ss_token_list = torch.cat(token_list, dim=1)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = torch.sort(top_scores.indices).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
        if draft_debug and not _is_current_stream_capturing(torch):
            try:
                logger.info(
                    "GLM52 draft final debug: parent_shape=%s top_scores_index_head=%s "
                    "draft_tokens_head=%s score_head=%s",
                    tuple(parent_list.shape) if "parent_list" in locals() else None,
                    top_scores_index[: min(4, int(top_scores_index.shape[0]))]
                    .detach()
                    .cpu()
                    .tolist(),
                    draft_tokens[: min(4, int(draft_tokens.shape[0]))]
                    .detach()
                    .cpu()
                    .tolist(),
                    score_list[: min(4, int(score_list.shape[0]))]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                )
            except Exception:
                logger.exception("Failed to log GLM52 draft final debug")

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    EagleDraftWorker.draft_forward = draft_forward
    EagleDraftWorker._atom_sglang_draft_argmax_patched = True
    logger.info("Patched SGLang EAGLE draft_forward for ATOM distributed argmax")


def _patch_sglang_glm52_logits_norm_debug() -> None:
    if os.environ.get("ATOM_GLM52_LOGITS_NORM_DEBUG", "0") not in (
        "1",
        "true",
        "True",
    ):
        return
    try:
        import torch
        from sglang.srt.layers.logits_processor import LogitsProcessor
    except Exception as exc:
        logger.debug("Skip patching GLM52 logits norm debug: %s", exc)
        return

    if getattr(LogitsProcessor, "_atom_glm52_logits_norm_debug_patched", False):
        return

    original_get_logits = LogitsProcessor._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata):
        should_log = False
        try:
            forward_mode = getattr(logits_metadata, "forward_mode", None)
            should_log = bool(
                forward_mode is not None
                and getattr(forward_mode, "is_target_verify", lambda: False)()
                and not _is_current_stream_capturing(torch)
            )
            if should_log:
                probe = hidden_states[: min(8, int(hidden_states.shape[0]))].detach()
                logger.info(
                    "GLM52 logits norm pre: mode=%s hidden_shape=%s "
                    "hidden_norm=%s hidden_absmax=%s hidden_mean=%s",
                    forward_mode,
                    tuple(hidden_states.shape),
                    probe.float().norm(dim=-1).cpu().tolist(),
                    probe.float().abs().amax(dim=-1).cpu().tolist(),
                    probe.float().mean(dim=-1).cpu().tolist(),
                )
        except Exception:
            logger.exception("Failed to log GLM52 logits norm pre debug")

        logits = original_get_logits(self, hidden_states, lm_head, logits_metadata)

        if should_log:
            try:
                probe = logits[: min(8, int(logits.shape[0]))].detach().float()
                top_vals, top_ids = torch.topk(probe, k=3, dim=-1)
                logger.info(
                    "GLM52 logits norm post: logits_shape=%s logits_norm=%s "
                    "logits_absmax=%s logits_mean=%s top_ids=%s top_vals=%s",
                    tuple(logits.shape),
                    probe.norm(dim=-1).cpu().tolist(),
                    probe.abs().amax(dim=-1).cpu().tolist(),
                    probe.mean(dim=-1).cpu().tolist(),
                    top_ids.cpu().tolist(),
                    top_vals.cpu().tolist(),
                )
            except Exception:
                logger.exception("Failed to log GLM52 logits norm post debug")
        return logits

    LogitsProcessor._get_logits = _get_logits
    LogitsProcessor._atom_glm52_logits_norm_debug_patched = True
    logger.info("Patched SGLang LogitsProcessor for GLM52 norm debug")


def register_ops_to_sglang(atom_config: Config) -> None:
    """
    Register custom ops to sglang, including attention
    """
    _register_custom_attention_to_sglang()
    _patch_sglang_dsv4_draft_backends()
    _patch_sglang_dsv4_spec_cuda_graph()
    _patch_sglang_eagle_v2_draft_argmax()
    _patch_sglang_glm52_logits_norm_debug()


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
