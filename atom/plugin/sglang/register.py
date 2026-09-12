import logging
import os

from transformers import AutoConfig, PretrainedConfig

from atom.plugin.sglang.models.kimi_k3_processor import (
    register_kimi_k3_text_only_processor,
)
from atom.plugin.sglang.models.qwen3_8_flash_next_processor import (
    register_qwen4_exp_text_only_processor,
)
from atom.plugin.sglang.patches.prefill_compile_only_patch import (
    apply_prefill_compile_only_patch,
)
from atom.plugin.sglang.patches.triton_kernel_retention_patch import (
    apply_triton_kernel_retention_patch,
)

logger = logging.getLogger("atom.plugin.sglang.register")


def _ensure_aiter_gpu_archs_env() -> None:
    """Bridge ATOM image arch env names to aiter's runtime JIT env."""

    if os.environ.get("GPU_ARCHS"):
        return
    for env_name in ("GPU_ARCH_LIST", "PYTORCH_ROCM_ARCH"):
        archs = os.environ.get(env_name)
        if archs:
            os.environ["GPU_ARCHS"] = archs
            return


def _is_atom_external_model_enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get() == "atom.plugin.sglang.models"
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return False


def _hf_quant_method(model_config) -> str:
    try:
        quant_cfg = model_config._parse_quant_hf_config()
    except Exception:  # noqa: BLE001 - tolerate absent or incompatible HF config
        quant_cfg = None
    if not quant_cfg:
        return ""
    return str(quant_cfg.get("quant_method", "")).lower()


def _install_model_config_quant_patch() -> None:
    from sglang.srt.configs.model_config import ModelConfig

    if getattr(ModelConfig, "_atom_sglang_quant_patch", False):
        return

    original_verify_quantization = ModelConfig._verify_quantization

    def verify_quantization_with_atom_external_bypass(self):
        try:
            return original_verify_quantization(self)
        except ValueError as exc:
            if (
                _is_atom_external_model_enabled()
                and _hf_quant_method(self) == "mxfp8"
                and "quantization is currently not supported in ROCm" in str(exc)
            ):
                logger.info(
                    "Skipping SGLang server-args quantization gate for ATOM "
                    "external MXFP8 model; ATOM owns quantized weight loading."
                )
                self.quantization = None
                return None
            raise

    ModelConfig._verify_quantization = verify_quantization_with_atom_external_bypass
    ModelConfig._atom_sglang_quant_patch = True


def _install_loader_quant_patch() -> None:
    from sglang.srt.model_loader import loader

    if getattr(loader, "_atom_sglang_quant_patch", False):
        return

    original_get_quantization_config = loader._get_quantization_config

    def get_quantization_config_with_atom_external_bypass(model_config, load_config):
        model_class, _ = loader.get_model_architecture(model_config)
        if getattr(model_class, "sglang_skip_quant_config", False):
            logger.info(
                "Skipping SGLang native quant_config for external model %s; "
                "the model wrapper owns quantized weight loading.",
                model_class.__name__,
            )
            return None
        return original_get_quantization_config(model_config, load_config)

    loader._get_quantization_config = get_quantization_config_with_atom_external_bypass
    loader._atom_sglang_quant_patch = True


def _install_decode_graph_forward_context_patch() -> None:
    try:
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            has_forward_context,
        )
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    if getattr(DecodeCudaGraphRunner, "_atom_forward_context_patched", False):
        return

    original_capture_one_shape = DecodeCudaGraphRunner.capture_one_shape

    def capture_one_shape_with_forward_context(self, *args, **kwargs):
        if has_forward_context():
            return original_capture_one_shape(self, *args, **kwargs)

        attn_backend = self.model_runner.attn_backend
        attn_backend.token_to_kv_pool = self.model_runner.token_to_kv_pool
        attn_backend.req_to_token_pool = self.model_runner.req_to_token_pool
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            return original_capture_one_shape(self, *args, **kwargs)

    DecodeCudaGraphRunner.capture_one_shape = capture_one_shape_with_forward_context
    DecodeCudaGraphRunner._atom_forward_context_patched = True


def _register_tc_piecewise_attention_split_ops() -> None:
    """Keep ATOM attention kernels outside captured piecewise subgraphs."""

    from sglang.srt.compilation.compilation_config import SPLIT_OPS

    # Qwen3.5 uses native returning ops for dynamic compile-only prefill and
    # decode CUDA Graphs. Padded piecewise prefill keeps graph-stable mutating
    # ops. Both variants must remain split boundaries so per-batch attention
    # metadata stays live.
    for op_name in (
        "aiter.unified_attention_with_output_base",
        "aiter.unified_attention_with_output_base.default",
        "aiter.linear_attention_with_output_base",
        "aiter.linear_attention_with_output_base.default",
        "aiter.sglang_qwen35_attention_with_stable_output",
        "aiter.sglang_qwen35_attention_with_stable_output.default",
        "aiter.sglang_qwen35_linear_attention_with_stable_output",
        "aiter.sglang_qwen35_linear_attention_with_stable_output.default",
    ):
        if op_name not in SPLIT_OPS:
            SPLIT_OPS.append(op_name)


try:
    # Qwen3_5TextConfig → Qwen3NextConfig brings mamba2_cache_params /
    # linear_layer_ids that SGLang's hybrid GDN memory pool requires.
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig as _Qwen4ExpTextBase
except Exception:  # pragma: no cover - register only loads under SGLang
    _Qwen4ExpTextBase = PretrainedConfig


class Qwen4ExpTextConfig(_Qwen4ExpTextBase):
    """Shim for nested ``qwen4_exp_text`` until transformers>=5.16.1.

    Subclasses SGLang's Qwen3.5 text config so ``hybrid_gdn_config`` /
    MambaPool sizing see ``mamba2_cache_params``. Without that, SGLang
    allocates no mamba slots and ATOM GDN/PLE zero out → greedy garbage.
    """

    model_type = "qwen4_exp_text"

    @property
    def layers_block_type(self):
        """Prefer checkpoint ``layer_types``; map Flash QSA names to GDN pool ids."""
        layer_types = getattr(self, "layer_types", None)
        if layer_types:
            out = []
            for layer_type in layer_types:
                if layer_type in (
                    "full_attention",
                    "qwen_sparse_attention",
                    "attention",
                ):
                    out.append("attention")
                else:
                    out.append("linear_attention")
            return out
        return super().layers_block_type  # type: ignore[misc]


class Qwen4ExpConfig(PretrainedConfig):
    """Shim for ``qwen4_exp`` (Qwen3.8-Flash-Next). Must be module-level.

    SGLang spawn-pickles ServerArgs.model_config.hf_config; a nested class
    inside ``_register_qwen4_exp_hf_configs`` is not picklable.
    """

    model_type = "qwen4_exp"

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_kwargs = dict(text_config)
            text_kwargs.pop("model_type", None)
            text_config = Qwen4ExpTextConfig(**text_kwargs)
        if isinstance(vision_config, dict):
            vision_kwargs = dict(vision_config)
            vision_kwargs.pop("model_type", None)
            vision_config = PretrainedConfig(**vision_kwargs)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)
        src = self.text_config
        if src is None:
            return
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "intermediate_size",
        ):
            if getattr(self, key, None) is None and hasattr(src, key):
                setattr(self, key, getattr(src, key))


def _register_qwen4_exp_hf_configs() -> None:
    """Let this image's Transformers parse Qwen3.8-Flash-Next checkpoints.

    The FP8 checkpoint is ``model_type=qwen4_exp`` with nested
    ``qwen4_exp_text``. That class ships in transformers>=5.16.1; this
    image is older, so AutoConfig.from_pretrained fails in ServerArgs
    before ATOM's Native get_hf_config can run.
    """

    def _register(model_type: str, config_cls: type) -> None:
        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError as exc:
                if "already used by a Transformers config" not in str(exc):
                    raise

    _register("qwen4_exp_text", Qwen4ExpTextConfig)
    _register("qwen4_exp", Qwen4ExpConfig)


def _patch_qwen4_exp_mrope() -> None:
    """Map Flash-Next ``qwen4_exp`` onto Qwen3.5 M-RoPE index math.

    Text warmup still goes through TransformersAutoMultimodalProcessor, which
    calls SGLang's get_rope_index. That helper only knows qwen3_5 / qwen2_vl.
    """
    try:
        from sglang.srt.layers.rotary_embedding import mrope as mrope_mod
        from sglang.srt.layers.rotary_embedding import mrope_rope_index as mri
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    orig = mri.get_rope_index
    if getattr(orig, "_atom_qwen4_exp", False):
        return

    def get_rope_index_with_qwen4_exp(*args, **kwargs):
        if len(args) >= 5 and args[4] == "qwen4_exp":
            args = (*args[:4], "qwen3_5", *args[5:])
        if kwargs.get("model_type") == "qwen4_exp":
            kwargs = {**kwargs, "model_type": "qwen3_5"}
        return orig(*args, **kwargs)

    get_rope_index_with_qwen4_exp._atom_qwen4_exp = True
    mri.get_rope_index = get_rope_index_with_qwen4_exp
    if getattr(mrope_mod, "get_rope_index", None) is orig:
        mrope_mod.get_rope_index = get_rope_index_with_qwen4_exp


def _patch_hybrid_gdn_config_for_qwen4_exp() -> None:
    """Teach SGLang that Qwen3.8-Flash-Next is a hybrid GDN model.

    Stock ``hybrid_gdn_config`` only matches Qwen3-Next / Qwen3.5 configs.
    Without this, ``is_hybrid_ssm`` stays False, no MambaPool / ``mamba_map``
    is allocated, ``SGLangGDNForwardContext.build`` returns None, every GDN
    layer zeros its output, and PLE is skipped — greedy text becomes garbage
    even when PLE ``weight_scale`` loaded correctly.
    """
    import sys

    try:
        from sglang.srt.configs import hybrid_arch as ha
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        logger.warning("hybrid_gdn_config patch skipped: hybrid_arch import failed")
        return

    orig = ha.hybrid_gdn_config
    if getattr(orig, "_atom_qwen4_exp", False):
        return

    def hybrid_gdn_config(model_config):
        cfg = orig(model_config)
        if cfg is not None:
            return cfg
        hf = getattr(model_config, "hf_config", None)
        if hf is None:
            return None
        text = None
        getter = getattr(hf, "get_text_config", None)
        if callable(getter):
            try:
                text = getter()
            except Exception:  # noqa: BLE001
                text = None
        if text is None:
            text = getattr(hf, "text_config", None) or hf
        mt = getattr(text, "model_type", None) or getattr(hf, "model_type", None)
        if mt in ("qwen4_exp_text", "qwen4_exp") or isinstance(
            text, Qwen4ExpTextConfig
        ) or isinstance(hf, Qwen4ExpConfig):
            return text
        return None

    hybrid_gdn_config._atom_qwen4_exp = True  # type: ignore[attr-defined]
    ha.hybrid_gdn_config = hybrid_gdn_config
    # ``from hybrid_arch import hybrid_gdn_config`` aliases must be rebound.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "hybrid_gdn_config", None) is orig:
                mod.hybrid_gdn_config = hybrid_gdn_config
        except Exception:  # noqa: BLE001
            continue
    logger.info(
        "Patched hybrid_gdn_config to recognize qwen4_exp / Qwen3.8-Flash-Next"
    )


def register_plugin() -> None:
    """Install ATOM patches that must run before SGLang parses server args."""

    _ensure_aiter_gpu_archs_env()
    _register_qwen4_exp_hf_configs()
    _patch_qwen4_exp_mrope()
    _patch_hybrid_gdn_config_for_qwen4_exp()
    _install_model_config_quant_patch()
    _install_loader_quant_patch()
    _register_tc_piecewise_attention_split_ops()
    _install_decode_graph_forward_context_patch()
    apply_prefill_compile_only_patch()
    apply_triton_kernel_retention_patch()
    register_kimi_k3_text_only_processor()
    register_qwen4_exp_text_only_processor()
    try:
        from atom.plugin.sglang.runtime import apply_load_config_patch

        apply_load_config_patch()
    except Exception:
        logger.exception("Failed to install ATOM SGLang load-config patch")
