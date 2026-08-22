from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES


def test_qwen35_text_moe_uses_atom_vllm_wrapper():
    arch = "Qwen3_5MoeForCausalLM"
    assert (
        _VLLM_MODEL_REGISTRY_OVERRIDES[arch]
        == "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForCausalLMVllm"
    )
    assert (
        _ATOM_MODEL_CLASSES[arch]
        == "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForCausalLM"
    )


def test_qwen35_text_moe_wrapper_exposes_hybrid_cache_interface():
    from vllm.model_executor.models.interfaces import IsHybrid

    from atom.plugin.vllm.models.qwen3_5 import Qwen3_5MoeForCausalLMVllm

    assert IsHybrid in Qwen3_5MoeForCausalLMVllm.__mro__
    assert callable(Qwen3_5MoeForCausalLMVllm.get_mamba_state_dtype_from_config)
    assert callable(Qwen3_5MoeForCausalLMVllm.get_mamba_state_shape_from_config)
    assert callable(Qwen3_5MoeForCausalLMVllm.get_mamba_state_copy_func)
