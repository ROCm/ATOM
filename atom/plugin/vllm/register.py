import logging

import torch
from transformers import AutoConfig, PretrainedConfig

from atom.plugin.prepare import _set_framework_backbone
from atom.plugin.vllm.spec_decode_patch import apply_vllm_spec_decode_patch
from atom.utils import envs

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
    # GLM-5.3-Flash (glm5_next). Native ATOM already has this class; without
    # this entry `vllm serve` never sees Glm5NextForConditionalGeneration on
    # vLLM 0.28.x and the plugin never wraps ATOM kernels.
    "Glm5NextForConditionalGeneration": (
        "atom.plugin.vllm.models.glm5_next:Glm5NextForConditionalGenerationVllm"
    ),
    # GLM-5.3-Flash MTP draft (checkpoint layer 45 through ATOM's NextN runtime).
    "Glm5NextMTPModel": (
        "atom.plugin.vllm.models.glm5_next:Glm5NextMTPVllm"
    ),
    "DeepSeekMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepSeekV4MTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.plugin.vllm.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3NextMTP": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3_5ForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
    "KimiK3ForConditionalGeneration": (
        "atom.plugin.vllm.models.kimi_k3:KimiK3ForConditionalGenerationVllm"
    ),
    # vLLM registers this arch too, but only to its NVIDIA implementation.
    "K3DSparkModel": "atom.plugin.vllm.models.kimi_k3_dspark:KimiK3DSparkVllm",
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


def _register_glm5_archs_early() -> None:
    """Register Glm5Next{,MTP} archs in the main process ModelRegistry.

    register_model() registers all ATOM archs, but runs inside worker processes
    after the main process has already validated speculative-config (which builds
    a draft ModelConfig and checks its architecture against the registry).
    Call this from register_platform() so the two GLM-5.3 archs are available
    before EngineArgs.create_engine_config() runs.
    """
    try:
        from vllm.model_executor.models import registry as vllm_model_registry
    except Exception:
        return

    _guard = "_atom_glm5_archs_registered"
    if getattr(vllm_model_registry, _guard, False):
        return

    glm5_archs = {
        k: v for k, v in _VLLM_MODEL_REGISTRY_OVERRIDES.items()
        if k.startswith("Glm5Next")
    }
    for arch, qual in glm5_archs.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None and (
            getattr(existing, "module_name", None) == module_name
            and getattr(existing, "class_name", None) == class_name
        ):
            continue
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        logger.info("ATOM plugin (early): registered %s for MTP config validation", arch)

    try:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()
    except Exception:
        pass

    setattr(vllm_model_registry, _guard, True)


def register_platform() -> str | None:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    from atom.plugin.vllm.rocm_dcp_full_graph_patch import (
        apply_vllm_rocm_dcp_full_graph_patch,
    )

    apply_vllm_rocm_dcp_full_graph_patch()

    # aiter's custom all-reduce needs vLLM's rendezvous to stay on a TCPStore.
    from atom.plugin.vllm.dist_store_patch import apply_vllm_tcp_store_patch

    apply_vllm_tcp_store_patch()

    # Do not call _set_plugin_mode() here. SGLang (and other stacks) discover
    # vllm.platform_plugins and would set atom's backbone to "vllm" before
    # importing SGLang plugin modules — then atom.models.qwen3_5's ``if is_vllm():``
    # branch runs and requires vllm.model_executor.models.qwen3_5, which may be
    # absent. Backbone is set in register_model() for real vLLM runs.

    _register_hf_configs()
    _register_mxfp8_quantization_config()
    # DeepSeek-V4's packed proxy arena cannot immediately recycle block ids;
    # install the targeted scheduler-side queue-order compatibility patch before
    # any KVCacheManager is constructed.
    from atom.plugin.vllm.deepseek_v4_prefix_patch import (
        apply_vllm_v4_block_reuse_patch,
    )

    apply_vllm_v4_block_reuse_patch()

    _register_kv_connectors()

    # Patch SpeculativeConfig early (before EngineArgs validates --speculative-config)
    # so that method=mtp resolves correctly for glm5_next. register_model() also
    # calls this, but that runs in worker processes after arg validation has already
    # raised NotImplementedError for unknown MTP model types.
    _patch_vllm_glm5_next_mtp_speculative_config()

    # Register GLM-5.3-Flash archs early so ModelConfig (called during
    # create_speculative_config) can resolve Glm5NextMTPModel. register_model()
    # registers all archs but runs in worker processes; the draft ModelConfig
    # validation happens in the main process before any workers start.
    _register_glm5_archs_early()

    # return the ATOM platform to vllm
    return "atom.plugin.vllm.platform.ATOMPlatform"


def _register_kv_connectors() -> None:
    """Expose ATOM's byte-level LMCache offload to vLLM's connector factory.

    Convenience only. vLLM validates ``kv_transfer_config`` while building
    VllmConfig, which happens BEFORE platform plugins are invoked, so a run
    that names the connector by bare name fails config validation before this
    ever runs. The supported way to select it is vLLM's out-of-tree entry
    point, which takes priority over the registry and needs no registration:

        --kv-transfer-config '{"kv_connector": "AtomLMCacheOffloadConnector",
          "kv_connector_module_path": "atom.plugin.vllm.kv_transfer.connector",
          "kv_role": "kv_both"}'

    Registered by module path so importing the plugin does not drag in the
    offload stack (and LMCache) for every run.
    """
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    name = "AtomLMCacheOffloadConnector"
    if name in getattr(KVConnectorFactory, "_registry", {}):
        return
    KVConnectorFactory.register_connector(
        name,
        "atom.plugin.vllm.kv_transfer.connector",
        name,
    )
    logger.info("Registered ATOM KV connector: %s", name)


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

    wrapped._atom_default_act_dtype_patched = True
    attention.process_weights_after_loading = wrapped


def _patch_vllm_harmony_parser_manager() -> None:
    """Restore vLLM's documented Harmony parser selection contract."""
    try:
        from vllm.parser import ParserManager
        from vllm.parser.harmony import HarmonyParser
    except ImportError:
        return

    original = ParserManager.get_parser.__func__
    if getattr(original, "_atom_harmony_parser_patched", False):
        return

    # Do nothing after upstream fixes the early return for is_harmony=True.
    try:
        if original(ParserManager, is_harmony=True) is not None:
            return
    except TypeError:
        return

    def get_parser(
        cls,
        tool_parser_name=None,
        reasoning_parser_name=None,
        enable_auto_tools=False,
        model_name=None,
        is_harmony=False,
    ):
        parser_cls = original(
            cls,
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            is_harmony=is_harmony,
        )
        if parser_cls is not None or not is_harmony:
            return parser_cls

        from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

        HarmonyParser.reasoning_parser_cls = GptOssReasoningParser
        HarmonyParser.tool_parser_cls = None
        return HarmonyParser

    get_parser._atom_harmony_parser_patched = True
    ParserManager.get_parser = classmethod(get_parser)


def _patch_vllm_glm5_next_mtp_speculative_config() -> None:
    """Teach vLLM's SpeculativeConfig to resolve method=mtp for glm5_next.

    vLLM's hf_config_override() rewrites the target's hf_config so the MTP
    draft model config gets the right arch and model_type. GLM-5.3-Flash
    (model_type='glm5_next') is not in the upstream dispatch table; patch it in
    here without touching the installed vLLM wheel.

    The num_nextn_predict_layers field lives in text_config on this checkpoint.
    """
    try:
        import vllm.config.speculative as vllm_spec_cfg
        from vllm.config.speculative import SpeculativeConfig
    except Exception:
        logger.warning("ATOM plugin: could not import vllm.config.speculative; GLM-5.3 MTP skipped")
        return

    if getattr(vllm_spec_cfg, "_atom_glm5_mtp_patched", False):
        return

    # 1. Widen MTPModelTypes so the draft model_type passes the isinstance gate.
    import typing
    orig_literal = vllm_spec_cfg.MTPModelTypes
    orig_args = set(typing.get_args(orig_literal))
    if "glm5_next_mtp" not in orig_args:
        new_args = tuple(orig_args | {"glm5_next_mtp"})
        vllm_spec_cfg.MTPModelTypes = typing.Literal[new_args]  # type: ignore[assignment]

    # 2. Wrap hf_config_override to insert the GLM-5.3-Flash → MTP rewrite.
    original_override = staticmethod(SpeculativeConfig.__dict__["hf_config_override"])

    def _glm5_hf_config_override(hf_config):
        # GLM-5.3-Flash top-level model_type is "glm5_next"; the MTP layer
        # count is in text_config.num_nextn_predict_layers.
        if getattr(hf_config, "model_type", None) in ("glm5_next", "glm5_next_text"):
            text_config = getattr(hf_config, "text_config", hf_config)
            n_predict = int(getattr(text_config, "num_nextn_predict_layers", 1) or 1)
            hf_config.model_type = "glm5_next_mtp"
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["Glm5NextMTPModel"],
            })
            logger.info(
                "ATOM plugin: rewrote glm5_next hf_config to Glm5NextMTPModel "
                "(n_predict=%d).", n_predict
            )
            return hf_config
        return original_override.__func__(hf_config)

    SpeculativeConfig.hf_config_override = staticmethod(_glm5_hf_config_override)
    vllm_spec_cfg._atom_glm5_mtp_patched = True
    logger.info("ATOM plugin: patched SpeculativeConfig.hf_config_override for glm5_next MTP.")


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _set_plugin_mode()

    # Install the GLM-5.3-Flash MLA/mamba KV page-alignment fix. register_model
    # runs in each worker before its KV-cache setup calls
    # `_align_hybrid_block_size`, and (unlike register_platform) reliably runs in
    # the worker process, so patch the base Platform method from here.
    from atom.plugin.vllm.platform import _install_glm5_align_patch

    _install_glm5_align_patch()

    # The general-plugin hook runs in the EngineCore process that owns the
    # scheduler/KVCacheManager; install this here as well as in the platform hook.
    from atom.plugin.vllm.deepseek_v4_prefix_patch import (
        apply_vllm_v4_block_reuse_patch,
    )

    apply_vllm_v4_block_reuse_patch()

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

    # vLLM rejects Kimi-K3 DSpark under DCP while validating the speculative
    # config. That validation runs later than this hook but earlier than any
    # model is built, and unlike register_platform -- which vLLM invokes from
    # inside its own import -- here vllm.engine is safe to import.
    from atom.plugin.vllm.dspark_dcp_patch import (
        apply_vllm_dspark_dcp_config_patch,
    )

    apply_vllm_dspark_dcp_config_patch()

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

    # vLLM 0.26 profiles CUDA graph memory on ROCm by temporarily capturing
    # and destroying every graph. Skip that pass before it can leave stale
    # AITER graph-owned state.
    from atom.plugin.vllm.cudagraph_memory_profiler_patch import (
        apply_vllm_cudagraph_memory_profiler_patch,
    )

    apply_vllm_cudagraph_memory_profiler_patch()

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

    # The native MORI MoE path is frontend-agnostic; inject atom-vllm-specific
    # launch-config selection and dispatch-buffer trimming via plugin patches.
    from atom.plugin.vllm.mori_patch import apply_vllm_mori_patch

    apply_vllm_mori_patch()

    from atom.plugin.vllm.qwen35_attention_patch import (
        apply_qwen35_vllm_attention_patch,
    )

    apply_qwen35_vllm_attention_patch()
    # Expose batch-ordered req_ids to ATOM metadata builders so the DeepSeek-V4
    # proxy can key state-slot allocation on the request id (host-resident)
    # instead of a D2H copy of the first block id.
    from atom.plugin.vllm.req_id_passthrough_patch import (
        apply_vllm_req_id_passthrough_patch,
    )

    apply_vllm_req_id_passthrough_patch()

    _patch_vllm_glm5_next_mtp_speculative_config()
