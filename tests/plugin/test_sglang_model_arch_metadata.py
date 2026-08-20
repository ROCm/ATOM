from types import SimpleNamespace

import pytest

from atom.plugin.sglang.runtime.model_arch import resolve_model_arch_spec


@pytest.mark.parametrize(
    ("model_arch", "builder_name"),
    (
        ("KimiK3ForConditionalGeneration", "_build_kimi_k3_forward_metadata"),
        ("GlmMoeDsaForCausalLM", "_build_glm52_dsa_forward_metadata"),
        ("DeepseekV4ForCausalLM", "_build_deepseek_v4_forward_metadata"),
        (
            "MiniMaxM3SparseForCausalLM",
            "_build_minimax_m3_forward_metadata",
        ),
        (
            "MiniMaxM3SparseForConditionalGeneration",
            "_build_minimax_m3_forward_metadata",
        ),
        ("LlamaForCausalLMEagle3", "_build_eagle3_llama_forward_metadata"),
    ),
)
def test_resolve_model_arch_spec_selects_metadata_builder(
    model_arch: str, builder_name: str
):
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(architectures=[model_arch])
    )

    assert resolved_arch == model_arch
    assert model_spec.build_forward_metadata is not None
    assert model_spec.build_forward_metadata.__name__ == builder_name


@pytest.mark.parametrize(
    "model_arch",
    (
        "Qwen3ForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "UnknownForCausalLM",
    ),
)
def test_resolve_model_arch_spec_uses_generic_metadata(model_arch: str):
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(architectures=[model_arch])
    )

    assert resolved_arch == model_arch
    assert model_spec.build_forward_metadata is None


def test_resolve_model_arch_spec_supports_glm_model_type_fallback():
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(architectures=[], model_type="glm_moe_dsa")
    )

    assert resolved_arch == "GlmMoeDsaForCausalLM"
    assert model_spec.build_forward_metadata is not None
    assert (
        model_spec.build_forward_metadata.__name__
        == "_build_glm52_dsa_forward_metadata"
    )


@pytest.mark.parametrize(
    ("model_type", "expected_arch", "builder_name"),
    (
        ("glm_moe_dsa", "GlmMoeDsaForCausalLM", "_build_glm52_dsa_forward_metadata"),
        (
            "kimi_k3",
            "KimiK3ForConditionalGeneration",
            "_build_kimi_k3_forward_metadata",
        ),
        (
            "minimax_m3_vl",
            "MiniMaxM3SparseForConditionalGeneration",
            "_build_minimax_m3_forward_metadata",
        ),
        (
            "deepseek_v4",
            "DeepseekV4ForCausalLM",
            "_build_deepseek_v4_forward_metadata",
        ),
    ),
)
def test_resolve_model_arch_spec_supports_family_version_architectures(
    model_type: str, expected_arch: str, builder_name: str
):
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(
            architectures=["FutureVersionForCausalLM"], model_type=model_type
        )
    )

    assert resolved_arch == expected_arch
    assert model_spec.build_forward_metadata is not None
    assert model_spec.build_forward_metadata.__name__ == builder_name


@pytest.mark.parametrize(
    ("model_type", "expected_arch"),
    (
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen3_next", "Qwen3NextForCausalLM"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    ),
)
def test_resolve_qwen_family_versions_use_their_own_adapter(
    model_type: str, expected_arch: str
):
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(architectures=["FutureQwenArchitecture"], model_type=model_type)
    )

    assert resolved_arch == expected_arch
    assert model_spec is not None


def test_resolve_model_arch_spec_uses_nested_text_model_type():
    resolved_arch, _ = resolve_model_arch_spec(
        SimpleNamespace(
            architectures=["FutureQwenConditionalGeneration"],
            model_type="future_qwen_vl",
            text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
    )

    assert resolved_arch == "Qwen3_5MoeForConditionalGeneration"


def test_resolve_model_arch_spec_does_not_treat_plain_llama_as_eagle3():
    resolved_arch, model_spec = resolve_model_arch_spec(
        SimpleNamespace(architectures=["LlamaForCausalLM"], model_type="llama")
    )

    assert resolved_arch == "LlamaForCausalLM"
    assert model_spec.build_forward_metadata is None
