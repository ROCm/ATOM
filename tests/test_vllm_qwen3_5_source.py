import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "atom/plugin/vllm/models/qwen3_5.py"


def _wrapper_class() -> ast.ClassDef:
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MoeForCausalLMVllm"
    )


def test_qwen35_text_moe_registers_the_atom_vllm_wrapper():
    vllm_target = "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForCausalLMVllm"
    atom_target = "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForCausalLM"
    register_source = (ROOT / "atom/plugin/vllm/register.py").read_text(
        encoding="utf-8"
    )
    wrapper_source = (ROOT / "atom/plugin/vllm/model_wrapper.py").read_text(
        encoding="utf-8"
    )

    assert vllm_target in register_source
    assert atom_target in wrapper_source


def test_qwen35_text_moe_wrapper_exposes_hybrid_cache_contract():
    wrapper = _wrapper_class()
    bases = {ast.unparse(base) for base in wrapper.bases}
    methods = {node.name for node in wrapper.body if isinstance(node, ast.FunctionDef)}

    assert {"ATOMMoEForCausalLM", "IsHybrid"} <= bases
    assert {
        "get_mamba_state_dtype_from_config",
        "get_mamba_state_shape_from_config",
        "get_mamba_state_copy_func",
    } <= methods


def test_qwen35_text_moe_wires_the_fused_expert_loader():
    wrapper = _wrapper_class()
    load_weights = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    source = ast.unparse(load_weights)

    assert "load_fused_expert_weights_fn=self.model.load_fused_expert_weights" in source
