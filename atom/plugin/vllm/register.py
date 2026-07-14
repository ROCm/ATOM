import logging
from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.spec_decode_patch import apply_vllm_spec_decode_patch

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "LlamaForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GptOssForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV32ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GlmMoeDsaForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepSeekMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepSeekV4MTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.plugin.vllm.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3NextMTP": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3_5ForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
    "MiniMaxM2ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV4ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "MiniMaxM3SparseForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "MiniMaxM3SparseForConditionalGeneration": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Eagle3LlamaForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "LlamaForCausalLMEagle3": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Eagle3DeepseekV2ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Eagle3DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
}


class MiniMaxM3Config(PretrainedConfig):
    """Minimal local config shim for MiniMax-M3 VL checkpoints."""

    model_type = "minimax_m3_vl"
    text_config_override_attrs = {
        "use_index_cache",
        "index_topk_freq",
        "index_topk_pattern",
        "index_skip_topk_offset",
    }

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | None = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.hidden_size = getattr(text_config, "hidden_size", None)

        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in self.text_config_override_attrs:
            return
        text_config = self.__dict__.get("text_config")
        if text_config is not None and text_config is not self:
            setattr(text_config, name, value)


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def _register_hf_configs() -> None:
    try:
        AutoConfig.register(MiniMaxM3Config.model_type, MiniMaxM3Config)
    except ValueError as exc:
        if "already used by a Transformers config" not in str(exc):
            raise


def _register_mxfp8_quantization_config() -> None:
    """Let ATOM-owned MXFP8 checkpoints pass vLLM config validation.

    vLLM uses the same name, "mxfp8", for an online-quant shorthand. MiniMax-M3
    MXFP8 checkpoints store "quant_method": "mxfp8" in config.json, and ATOM
    parses/loads those weights itself. Registering this no-op config prevents
    vLLM from routing the checkpoint config through OnlineQuantizationConfig.
    """
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )

    @register_quantization_config("mxfp8")
    class AtomMxfp8Config(QuantizationConfig):
        @classmethod
        def from_config(cls, config):
            return cls()

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @classmethod
        def get_name(cls):
            return "mxfp8"

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16, torch.float16]

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            return []

        def get_quant_method(
            self, layer: torch.nn.Module, prefix: str
        ) -> QuantizeMethodBase | None:
            return None


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    # Do not call _set_plugin_mode() here. SGLang (and other stacks) discover
    # vllm.platform_plugins and would set atom's backbone to "vllm" before
    # importing SGLang plugin modules — then atom.models.qwen3_5's ``if is_vllm():``
    # branch runs and requires vllm.model_executor.models.qwen3_5, which may be
    # absent. Backbone is set in register_model() for real vLLM runs.

    _register_hf_configs()
    _register_mxfp8_quantization_config()

    # return the ATOM platform to vllm
    return "atom.plugin.vllm.platform.ATOMPlatform"


def _patch_vllm_attention_process_weights_after_loading(attention) -> None:
    orig = attention.process_weights_after_loading

    if getattr(orig, "_atom_default_act_dtype_patched", False):
        return

    try:
        import inspect

        sig = inspect.signature(orig)
        act_dtype_param = sig.parameters.get("act_dtype")
        if (
            act_dtype_param is not None
            and act_dtype_param.default is not inspect._empty
        ):
            return
    except Exception:
        pass

    import functools

    @functools.wraps(orig)
    def wrapped(self, act_dtype: "torch.dtype" = torch.bfloat16):
        return orig(self, act_dtype)

    setattr(wrapped, "_atom_default_act_dtype_patched", True)
    attention.process_weights_after_loading = wrapped


def _patch_vllm_harmony_parser_manager() -> None:
    """Keep GPT-OSS harmony parsing enabled on vLLM builds with parser manager.

    Some vLLM versions route chat-completions parsing through
    ``ParserManager.get_parser``. Its contract says ``is_harmony=True`` should
    always return ``HarmonyParser``, but the implementation can return ``None``
    first when no explicit tool/reasoning parser is configured. That exposes
    raw GPT-OSS analysis/final channel text as ``message.content`` and breaks
    accuracy evaluation.
    """
    try:
        from vllm.parser import ParserManager
        from vllm.parser.harmony import HarmonyParser
    except ImportError:
        return

    original = ParserManager.get_parser.__func__
    if getattr(original, "_atom_harmony_parser_patched", False):
        return

    def get_parser(
        cls,
        tool_parser_name=None,
        reasoning_parser_name=None,
        enable_auto_tools=False,
        model_name=None,
        is_harmony=False,
    ):
        if is_harmony and not tool_parser_name and not reasoning_parser_name:
            from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

            HarmonyParser.reasoning_parser_cls = GptOssReasoningParser
            HarmonyParser.tool_parser_cls = None
            return HarmonyParser
        return original(
            cls,
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            is_harmony=is_harmony,
        )

    setattr(get_parser, "_atom_harmony_parser_patched", True)
    ParserManager.get_parser = classmethod(get_parser)

    original_parse = HarmonyParser.parse
    if getattr(original_parse, "_atom_harmony_parse_fallback_patched", False):
        return

    def parse_with_raw_content_fallback(
        self,
        model_output,
        request,
        enable_auto_tools=False,
        model_output_token_ids=(),
    ):
        try:
            reasoning, content, tool_calls = original_parse(
                self,
                model_output,
                request,
                enable_auto_tools=enable_auto_tools,
                model_output_token_ids=model_output_token_ids,
            )
        except Exception as exc:
            if exc.__class__.__name__ != "HarmonyError":
                raise
            return None, model_output or None, None

        if reasoning:
            return reasoning, reasoning, tool_calls
        if content is None and model_output:
            return reasoning, model_output, tool_calls
        return reasoning, content, tool_calls

    setattr(
        parse_with_raw_content_fallback,
        "_atom_harmony_parse_fallback_patched",
        True,
    )
    HarmonyParser.parse = parse_with_raw_content_fallback


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _set_plugin_mode()

    from atom.plugin.vllm.gdn_backend import register_gdn_attention_backend

    register_gdn_attention_backend()
    _patch_vllm_harmony_parser_manager()

    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM with {qual}")
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()

    # patch attention process weights after loading
    # to avoid the specific handle in ATOM loader
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)
    # vLLM's speculative decoder keeps an allow-list of attention metadata
    # classes. ATOM-vLLM uses its own metadata classes after attention
    # isolation, so extend that allow-list before MTP/Eagle proposal runs.
    apply_vllm_spec_decode_patch()

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

    # The native MORI MoE path is frontend-agnostic; inject atom-vllm-specific
    # launch-config selection and dispatch-buffer trimming via plugin patches.
    from atom.plugin.vllm.mori_patch import apply_vllm_mori_patch

    apply_vllm_mori_patch()
    # Expose batch-ordered req_ids to ATOM metadata builders so the DeepSeek-V4
    # proxy can key state-slot allocation on the request id (host-resident)
    # instead of a D2H copy of the first block id.
    from atom.plugin.vllm.req_id_passthrough_patch import (
        apply_vllm_req_id_passthrough_patch,
    )

    apply_vllm_req_id_passthrough_patch()
