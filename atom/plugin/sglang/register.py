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
            os.environ["GPU_ARCHS"] = _native_arch_in(archs)
            return


def _native_arch_in(archs: str) -> str:
    """Narrow a fat-binary arch list to the arch actually present.

    AITER's `get_gfx()` reads `GPU_ARCHS` and, for a list such as
    `gfx942;gfx950`, reports the LAST entry. On MI308X that makes every
    gfx-keyed tuned table (GEMM, MoE, GDN) miss and selects gfx950 defaults.
    """
    listed = [a for a in archs.replace(",", ";").split(";") if a]
    if len(listed) <= 1:
        return archs
    try:
        from aiter.jit.utils.chip_info import _detect_native

        native = [a for a in _detect_native() if a in listed]
    except Exception:  # noqa: BLE001 - no GPU / rocminfo in this process
        native = []
    return native[0] if native else archs


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


def _keep_atom_full_attn_for_native_qwen4_exp() -> None:
    """0.5.20 replaces hybrid full-attn with QwenSparseAttnBackend for Flash.

    Native ATOM still owns QSA compute and allocates CUDA-graph buffers in
    ATOMAttnBackendForSgl.out_graph. Skip the upstream swap.
    """
    try:
        from sglang.srt.layers.attention.qsa import config as qsa_config
    except Exception:  # noqa: BLE001 - 0.5.17 has no qsa.config
        return
    original = getattr(qsa_config, "is_qwen_qsa", None)
    if original is None or getattr(qsa_config, "_atom_keep_full_attn", False):
        return

    def is_qwen_qsa_for_atom(hf_config):
        arches = getattr(hf_config, "architectures", None) or []
        model_type = str(getattr(hf_config, "model_type", "") or "")
        text = getattr(hf_config, "text_config", None)
        text_mt = str(getattr(text, "model_type", "") or "")
        if (
            any("Qwen4Exp" in str(a) for a in arches)
            or model_type.startswith("qwen4_exp")
            or text_mt.startswith("qwen4_exp")
        ):
            from atom.plugin.sglang.patches.qwen4_exp_rocm_patch import (
                note_qwen4_exp_loaded,
            )

            note_qwen4_exp_loaded()
            logger.info(
                "Keep ATOMAttnBackendForSgl as hybrid full-attn for Native "
                "Qwen4Exp; skip SGLang QwenSparseAttnBackend"
            )
            return False
        return original(hf_config)

    qsa_config.is_qwen_qsa = is_qwen_qsa_for_atom
    qsa_config._atom_keep_full_attn = True


def register_plugin() -> None:
    """Install ATOM patches that must run before SGLang parses server args."""

    _ensure_aiter_gpu_archs_env()
    _install_model_config_quant_patch()
    _install_loader_quant_patch()
    _register_tc_piecewise_attention_split_ops()
    _install_decode_graph_forward_context_patch()
    apply_prefill_compile_only_patch()
    apply_triton_kernel_retention_patch()
    from atom.plugin.sglang.attention_backend.gdn_replayssm import (
        install_sglang_replayssm_commit,
    )
    from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
        apply_qwen4_exp_recognition_patch,
    )
    from atom.plugin.sglang.patches.qwen4_exp_rocm_patch import (
        apply_qwen4_exp_rocm_patch,
    )

    # 0.5.20 recognizes Flash. This call only installs MTP draft-arch and
    # HC hidden-width adapters that upstream still does not provide.
    apply_qwen4_exp_recognition_patch()
    # EP decode asm MoE and HIP topk=1 Triton tree/verify. Plugin-only;
    # keeps Ling's Flash MTP adapter off atom/model_ops and eagle3_llama.
    apply_qwen4_exp_rocm_patch()
    _keep_atom_full_attn_for_native_qwen4_exp()
    install_sglang_replayssm_commit()
    register_kimi_k3_text_only_processor()

    try:
        from atom.plugin.sglang.runtime import apply_load_config_patch

        apply_load_config_patch()
    except Exception:
        logger.exception("Failed to install ATOM SGLang load-config patch")
