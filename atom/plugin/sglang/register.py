import logging
import os

from atom.plugin.sglang.models.kimi_k3_processor import (
    register_kimi_k3_text_only_processor,
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


def register_plugin() -> None:
    """Install ATOM patches that must run before SGLang parses server args."""

    _ensure_aiter_gpu_archs_env()
    _install_model_config_quant_patch()
    _install_loader_quant_patch()
    _register_tc_piecewise_attention_split_ops()
    _install_decode_graph_forward_context_patch()
    apply_prefill_compile_only_patch()
    apply_triton_kernel_retention_patch()
    register_kimi_k3_text_only_processor()

    try:
        from atom.plugin.sglang.runtime import apply_load_config_patch

        apply_load_config_patch()
    except Exception:
        logger.exception("Failed to install ATOM SGLang load-config patch")
