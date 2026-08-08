"""SGLang plugin model adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

GLM52_DSA_ARCH = "GlmMoeDsaForCausalLM"
GLM52_DSA_MODEL_TYPE = "glm_moe_dsa"


def is_glm52_dsa_config(config: Any) -> bool:
    """Return whether an HF config describes GLM-5.2 DSA."""

    archs = getattr(config, "architectures", None) or []
    return (
        any(GLM52_DSA_ARCH in str(arch) for arch in archs)
        or getattr(config, "model_type", None) == GLM52_DSA_MODEL_TYPE
    )


@dataclass(frozen=True)
class SGLangModelAdapterSpec:
    """Adapter hooks for one SGLang plugin model architecture.

    The first version keeps the existing runtime flags while adding function
    hooks for config preparation and install-time model adaptation. This avoids
    growing a long list of booleans in the generic wrapper as new models arrive.
    """

    wrapper_binds_gdn_context: bool = False
    uses_context_only_forward: bool = False
    # SGLang initializes atom_config from the target ServerArgs but passes a
    # draft-local HF config to the external draft model. This early hook lets
    # adapters preserve target metadata and switch to the draft config before
    # normal config preparation.
    prepare_draft_model_config: Callable[[Any, Any], None] | None = None
    prepare_config: Callable[[Any, str], None] | None = None
    construction_context: Callable[[], AbstractContextManager[Any]] | None = None
    install_adapters: Callable[[Any], None] | None = None
    bind_cache_views: Callable[[Any, Any], None] | None = None


def _prepare_qwen35_config(atom_config: Any, model_arch: str) -> None:
    from atom.plugin.sglang.models.qwen3_5 import apply_prepare_model_adaptations

    apply_prepare_model_adaptations(atom_config, model_arch)


def _prepare_minimax_m2_config(atom_config: Any, model_arch: str) -> None:
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return

    from atom.models.minimax_m2 import MiniMaxM2ForCausalLM

    quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=MiniMaxM2ForCausalLM.packed_modules_mapping,
    )


def _prepare_kimi_k25_config(atom_config: Any, model_arch: str) -> None:
    from atom.plugin.sglang.models.kimi_k25 import (
        remap_kimi_k25_quant_config_for_sglang_plugin,
    )

    remap_kimi_k25_quant_config_for_sglang_plugin(atom_config, model_arch)


def _prepare_minimax_m3_config(atom_config: Any, model_arch: str) -> None:
    from atom.models.minimax_m3 import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )

    # SGLang applies JSON model overrides to the root HF config. Mirror the
    # sparse-attention options into the text config consumed by MiniMax-M3.
    hf_config = atom_config.hf_config
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        for name in (
            "use_index_cache",
            "index_topk_freq",
            "index_topk_pattern",
            "index_skip_topk_offset",
        ):
            if hasattr(hf_config, name):
                setattr(text_config, name, getattr(hf_config, name))

    # MiniMax-M3 native sparse attention is block-sparse at 128-token granularity.
    # The SGLang recipe must use --page-size 128; keep ATOM's config aligned so
    # sparse metadata and SHUFFLE cache views speak the same page ABI.
    atom_config.kv_cache_block_size = 128
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return

    model_cls = (
        MiniMaxM3SparseForConditionalGeneration
        if model_arch == "MiniMaxM3SparseForConditionalGeneration"
        else MiniMaxM3SparseForCausalLM
    )
    quant_config.remap_layer_name(
        hf_config,
        packed_modules_mapping=model_cls.packed_modules_mapping,
        quant_exclude_name_mapping=getattr(model_cls, "quant_exclude_name_mapping", {}),
    )


def _prepare_glm52_dsa_config(atom_config: Any, model_arch: str) -> None:
    from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM

    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is not None:
        quant_config.remap_layer_name(
            atom_config.hf_config,
            packed_modules_mapping=getattr(
                GlmMoeDsaForCausalLM, "packed_modules_mapping", {}
            ),
            weights_mapper=getattr(GlmMoeDsaForCausalLM, "hf_to_atom_mapper", {}),
            quant_exclude_name_mapping=getattr(
                GlmMoeDsaForCausalLM, "quant_exclude_name_mapping", {}
            ),
        )
        default_excludes = getattr(
            GlmMoeDsaForCausalLM, "quant_default_exclude_layers", []
        )
        if default_excludes:
            quant_config.apply_default_exclude_layers(default_excludes)

    # SGLang's DSA pool uses page64/preshuffle for GLM/DeepSeek-family DSA.
    # Keep ATOM's config aligned for the native GLM indexer, while
    # ATOM_MLA_PAGE_SIZE can remain 1 so sparse MLA reads selected physical ids.
    atom_config.kv_cache_block_size = 64


def _install_deepseek_mla_adapters(model: Any) -> None:
    from atom.plugin.sglang.models.deepseek_mla import setup_deepseek_for_sglang

    setup_deepseek_for_sglang(model)


def _glm52_dsa_construction_context():
    from atom.plugin.sglang.models.glm52_dsa_attention import (
        glm52_native_mla_attention_construction,
    )

    return glm52_native_mla_attention_construction()


def _install_glm52_dsa_native_adapters(model: Any) -> None:
    from atom.plugin.sglang.models.glm52_dsa import setup_glm52_dsa_for_sglang

    setup_glm52_dsa_for_sglang(model)


def _install_deepseek_v4_adapters(model: Any) -> None:
    # DeepSeek-V4 in SGLang plugin mode follows the proxy-KV bridge path:
    # SGLang owns scheduling/allocation, while ATOM owns the model, cache views,
    # forward metadata, and attention kernels.  We still patch forward_impl to
    # reconcile SGLang padded prefill tensors with real-token ATOM metadata.
    from atom.models.deepseek_v4 import DeepseekV4Attention
    from atom.plugin.sglang.models.deepseek_v4_attention import (
        patch_deepseek_v4_attention_for_sglang,
    )

    for module in model.modules():
        if isinstance(module, DeepseekV4Attention):
            patch_deepseek_v4_attention_for_sglang(module)


def _bind_deepseek_v4_cache_views(model: Any, runtime: Any) -> None:
    del runtime
    from atom.plugin.sglang.deepseek_v4_bridge import (
        bind_deepseek_v4_proxy_cache_views,
        maybe_get_proxy_pool_from_sglang_backend,
        reset_deepseek_v4_state_slots,
    )

    proxy_pool, _ = maybe_get_proxy_pool_from_sglang_backend()
    if not bind_deepseek_v4_proxy_cache_views(model, proxy_pool):
        raise RuntimeError("DeepSeek-V4 SGLang proxy KV pool is not initialized")

    from atom.utils.forward_context import get_forward_context

    reset_slots = getattr(get_forward_context().attn_metadata, "reset_slots", None)
    reset_deepseek_v4_state_slots(model, reset_slots)


def _bind_glm52_dsa_cache_views(model: Any, runtime: Any) -> None:
    if getattr(runtime.forward_batch.forward_mode, "is_idle", lambda: False)():
        return

    from atom.plugin.sglang.glm52_dsa_bridge import (
        bind_glm52_dsa_cache_views,
    )
    from atom.plugin.sglang.runtime.attention_backend_resolver import (
        resolve_sglang_runtime,
    )

    token_to_kv_pool = resolve_sglang_runtime(runtime.forward_batch).token_to_kv_pool
    if not bind_glm52_dsa_cache_views(model, token_to_kv_pool):
        raise RuntimeError("GLM-5.2 SGLang DSA KV pool is not initialized")


def _install_minimax_m3_adapters(model: Any) -> None:
    from atom.plugin.sglang.models.minimax_m3 import setup_minimax_m3_for_sglang

    setup_minimax_m3_for_sglang(model)


def _minimax_m3_construction_context():
    from atom.plugin.sglang.models.minimax_m3 import (
        minimax_m3_native_sparse_attention_construction,
    )

    return minimax_m3_native_sparse_attention_construction()


def _bind_minimax_m3_cache_views(model: Any, runtime: Any) -> None:
    if getattr(runtime.forward_batch.forward_mode, "is_idle", lambda: False)():
        return

    from atom.plugin.sglang.minimax_m3_bridge import (
        bind_minimax_m3_sparse_cache_views,
        maybe_get_minimax_m3_pools_from_sglang_batch,
    )

    token_to_kv_pool, _ = maybe_get_minimax_m3_pools_from_sglang_batch(
        runtime.forward_batch
    )
    if not bind_minimax_m3_sparse_cache_views(model, token_to_kv_pool):
        raise RuntimeError("MiniMax-M3 SGLang sparse KV pool is not initialized")


def _prepare_eagle3_llama_draft_model_config(
    atom_config: Any, draft_config: Any
) -> None:
    from atom.config import QuantizationConfig

    num_draft_layers = int(getattr(draft_config, "num_hidden_layers", 0) or 0)
    if num_draft_layers != 1:
        raise ValueError("ATOM SGLang EAGLE3 supports exactly one draft layer")

    target_config = getattr(atom_config.hf_config, "text_config", atom_config.hf_config)
    layer_offset = int(getattr(target_config, "num_hidden_layers", 0) or 0)
    # SGLang allocates the independent one-layer draft KV pool at physical
    # layer [0, 1), while ATOM numbers the draft attention logically after the
    # target layers (for example, layer 60). Save the target layer count here so
    # SGLangATOMEagle3Attention can assign ATOM's logical layer_num=60; the
    # EAGLE3 cache bridge maps that logical id back to SGLang's physical slot 0.
    atom_config.sgl_atom_eagle3_layer_offset = layer_offset

    atom_config.hf_config = draft_config
    model_path = getattr(draft_config, "_name_or_path", None) or getattr(
        draft_config, "name_or_path", None
    )
    if model_path:
        atom_config.model = model_path
    atom_config.quant_config = QuantizationConfig(
        draft_config,
        online_quant_config=getattr(atom_config, "online_quant_config", None),
    )


def _eagle3_llama_construction_context():
    from atom.plugin.sglang.eagle3_llama_bridge import (
        eagle3_llama_native_attention_construction,
    )

    return eagle3_llama_native_attention_construction()


def _bind_eagle3_llama_cache_views(model: Any, runtime: Any) -> None:
    if getattr(runtime.forward_batch.forward_mode, "is_idle", lambda: False)():
        return

    from atom.plugin.sglang.eagle3_llama_bridge import (
        bind_eagle3_llama_cache_views,
        maybe_get_eagle3_pools_from_sglang_batch,
    )

    token_to_kv_pool, _ = maybe_get_eagle3_pools_from_sglang_batch(
        runtime.forward_batch
    )
    if not bind_eagle3_llama_cache_views(model, token_to_kv_pool):
        raise RuntimeError("EAGLE3 SGLang draft KV pool is not initialized")


MODEL_ADAPTER_SPECS = {
    "DeepseekV3ForCausalLM": SGLangModelAdapterSpec(
        install_adapters=_install_deepseek_mla_adapters,
        uses_context_only_forward=True,
    ),
    "DeepseekV32ForCausalLM": SGLangModelAdapterSpec(
        install_adapters=_install_deepseek_mla_adapters,
        uses_context_only_forward=True,
    ),
    GLM52_DSA_ARCH: SGLangModelAdapterSpec(
        prepare_config=_prepare_glm52_dsa_config,
        construction_context=_glm52_dsa_construction_context,
        install_adapters=_install_glm52_dsa_native_adapters,
        bind_cache_views=_bind_glm52_dsa_cache_views,
        uses_context_only_forward=True,
    ),
    "KimiK25ForConditionalGeneration": SGLangModelAdapterSpec(
        prepare_config=_prepare_kimi_k25_config,
        install_adapters=_install_deepseek_mla_adapters,
        uses_context_only_forward=True,
    ),
    "Qwen3ForCausalLM": SGLangModelAdapterSpec(),
    "Qwen3MoeForCausalLM": SGLangModelAdapterSpec(),
    "Qwen3NextForCausalLM": SGLangModelAdapterSpec(
        wrapper_binds_gdn_context=True,
    ),
    "Qwen3_5ForConditionalGeneration": SGLangModelAdapterSpec(
        prepare_config=_prepare_qwen35_config,
    ),
    "Qwen3_5MoeForConditionalGeneration": SGLangModelAdapterSpec(
        prepare_config=_prepare_qwen35_config,
    ),
    "MiniMaxM2ForCausalLM": SGLangModelAdapterSpec(
        uses_context_only_forward=True,
        prepare_config=_prepare_minimax_m2_config,
    ),
    "DeepseekV4ForCausalLM": SGLangModelAdapterSpec(
        install_adapters=_install_deepseek_v4_adapters,
        bind_cache_views=_bind_deepseek_v4_cache_views,
    ),
    "MiniMaxM3SparseForCausalLM": SGLangModelAdapterSpec(
        uses_context_only_forward=True,
        prepare_config=_prepare_minimax_m3_config,
        construction_context=_minimax_m3_construction_context,
        install_adapters=_install_minimax_m3_adapters,
        bind_cache_views=_bind_minimax_m3_cache_views,
    ),
    "MiniMaxM3SparseForConditionalGeneration": SGLangModelAdapterSpec(
        uses_context_only_forward=True,
        prepare_config=_prepare_minimax_m3_config,
        construction_context=_minimax_m3_construction_context,
        install_adapters=_install_minimax_m3_adapters,
        bind_cache_views=_bind_minimax_m3_cache_views,
    ),
    "LlamaForCausalLMEagle3": SGLangModelAdapterSpec(
        uses_context_only_forward=True,
        prepare_draft_model_config=_prepare_eagle3_llama_draft_model_config,
        construction_context=_eagle3_llama_construction_context,
        bind_cache_views=_bind_eagle3_llama_cache_views,
    ),
}

# Architectures whose SGLang EntryClass is generated by base_model_wrapper.
# Custom outer-wrapper modules, such as Qwen3.5 multimodal wrappers, keep their
# own EntryClass and should not appear here or SGLang will see duplicate classes.
MODEL_ARCH_SPECS = {
    key: MODEL_ADAPTER_SPECS[key]
    for key in (
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        GLM52_DSA_ARCH,
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "MiniMaxM2ForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "LlamaForCausalLMEagle3",
        "DeepseekV4ForCausalLM",
    )
}


def get_model_arch_spec(model_arch: str) -> SGLangModelAdapterSpec:
    return MODEL_ADAPTER_SPECS.get(model_arch, SGLangModelAdapterSpec())
