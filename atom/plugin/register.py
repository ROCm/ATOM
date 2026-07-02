import logging
import os

from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.models.glm4_moe import Glm4MoeForCausalLM
from atom.models.deepseek_v2 import DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM
from atom.models.minimax_m2 import MiniMaxM2ForCausalLM
from atom.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGenerationTextOnly,
    Qwen3_5ForConditionalGenerationTextOnly,
)
from atom.config import Config
from atom.plugin.prepare import is_vllm, is_sglang, is_rtpllm

logger = logging.getLogger("atom")

_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "Glm4MoeForCausalLM": Glm4MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "DeepseekV32ForCausalLM": DeepseekV3ForCausalLM,
    "GlmMoeDsaForCausalLM": GlmMoeDsaForCausalLM,
    "MiniMaxM2ForCausalLM": MiniMaxM2ForCausalLM,
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
        arches = getattr(runner.model_config.hf_config, "architectures", None) or []
        if any("DeepseekV4" in str(arch) for arch in arches):
            logger.info(
                "Use ATOMDeepseekV4BackendForSgl for DeepSeek-V4 through SGLang aiter backend choice"
            )
            return ATOMDeepseekV4BackendForSgl(runner)
        return ATOMAttnBackendForSgl(runner)

    @register_attention_backend("dsv4")
    def create_dsv4_backend(runner):
        logger.info(
            "Create ATOMDeepseekV4BackendForSgl through SGLang dsv4 backend choice"
        )
        return ATOMDeepseekV4BackendForSgl(runner)


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
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    except Exception as exc:
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
            except Exception:
                should_cap = False

            try:
                if should_cap and server_args is not None and original_cuda_graph_bs:
                    server_args.cuda_graph_bs = _safe_spec_graph_bs(
                        original_cuda_graph_bs,
                        "ATOM_SGLANG_V4_TARGET_VERIFY_CG_BS",
                    )
                original_target_init(self, model_runner, *args, **kwargs)
            finally:
                if (
                    should_cap
                    and server_args is not None
                    and original_cuda_graph_bs is not None
                ):
                    server_args.cuda_graph_bs = original_cuda_graph_bs

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

    if not getattr(EAGLEDraftCudaGraphRunner, "_atom_dsv4_replay_patched", False):
        original_draft_replay = EAGLEDraftCudaGraphRunner.replay

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

        EAGLEDraftCudaGraphRunner.replay = replay
        EAGLEDraftCudaGraphRunner._atom_dsv4_replay_patched = True

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
                if (
                    not _is_dsv4_nextn_runner(getattr(self, "draft_runner", None))
                    or getattr(self, "cuda_graph_runner_for_draft_extend", None) is None
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

                if not _dsv4_draft_extend_graph_layout_ok(
                    self.cuda_graph_runner_for_draft_extend
                ):
                    runner = self.cuda_graph_runner_for_draft_extend
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

                # DRAFT_EXTEND_V2 materializes exactly `num_draft_tokens` slots
                # per sequence.  `accept_lens` includes the target bonus token,
                # so the value can be `num_draft_tokens + 1`; using that directly
                # in the fixed-layout index points one slot past the graph output.
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
                            self.cuda_graph_runner_for_draft_extend,
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
                    self.cuda_graph_runner_for_draft_extend
                    and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
                )
                if can_cuda_graph:
                    draft_logits_output = (
                        self.cuda_graph_runner_for_draft_extend.replay(forward_batch)
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
                        "DSV4 DRAFT_EXTEND_V2 output/index layout mismatch: "
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
