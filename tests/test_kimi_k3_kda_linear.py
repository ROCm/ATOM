import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

KIMI_K3_PATH = Path(__file__).resolve().parents[1] / "atom" / "models" / "kimi_k3.py"


def _kda_method(tree: ast.Module, method_name: str) -> ast.FunctionDef:
    kda_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiKDAAttention"
    )
    return next(
        node
        for node in kda_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_kda_fused_input_projection_uses_atom_linear_module():
    tree = ast.parse(KIMI_K3_PATH.read_text())
    init = _kda_method(tree, "__init__")
    forward = _kda_method(tree, "_forward_impl")

    constructs_merged_linear = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MergedColumnParallelLinear"
        for node in ast.walk(init)
    )
    initialized_attributes = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    calls_in_proj_module = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "in_proj"
        for node in ast.walk(forward)
    )
    calls_functional_linear = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "F"
        and node.func.attr == "linear"
        for node in ast.walk(forward)
    )

    assert constructs_merged_linear
    assert {"in_proj", "b_proj"}.issubset(initialized_attributes)
    assert initialized_attributes.isdisjoint({"q_proj", "k_proj", "v_proj", "g_proj"})
    assert calls_in_proj_module
    assert not calls_functional_linear


def test_kda_checkpoint_projections_map_only_selected_layers_to_in_proj():
    tree = ast.parse(KIMI_K3_PATH.read_text())
    mapping_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kda_packed_modules_mapping"
    )
    namespace = {}
    module = ast.Module(body=[mapping_function], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(KIMI_K3_PATH), "exec"), namespace
    )

    mapping = namespace["_kda_packed_modules_mapping"]([1, 3])
    for layer_idx in (1, 3):
        base = f".layers.{layer_idx}.self_attn."
        for shard_id, source_name in enumerate(
            ("q_proj", "k_proj", "v_proj", "g_proj")
        ):
            assert mapping[f"{base}{source_name}"] == (f"{base}in_proj", shard_id)
        assert f"{base}b_proj" not in mapping

    assert ".layers.0.self_attn.g_proj" not in mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_kda_atom_linear_matches_functional_linear_numerically():
    script = textwrap.dedent("""
        from types import SimpleNamespace

        import torch
        import torch.nn.functional as F

        import atom.model_ops.linear as linear_module
        from atom.model_ops.linear import ColumnParallelLinear, MergedColumnParallelLinear

        torch.manual_seed(20260722)
        device = torch.device("cuda:0")
        linear_module.get_tp_group = lambda: SimpleNamespace(
            rank_in_group=0,
            world_size=8,
        )

        hidden_size = 7168
        proj_size = 96 * 128
        num_heads = 96
        output_sizes = [proj_size, proj_size, proj_size, proj_size]

        with torch.device(device):
            in_proj = MergedColumnParallelLinear(
                hidden_size,
                output_sizes,
                bias=False,
                quant_config=None,
                source_quant_dtype=torch.bfloat16,
                prefix="test.kda.in_proj",
            )
            b_proj = ColumnParallelLinear(
                hidden_size,
                num_heads,
                bias=False,
                quant_config=None,
                source_quant_dtype=torch.bfloat16,
                prefix="test.kda.b_proj",
            )

        reference_shards = []
        with torch.no_grad():
            for shard_id, output_size in enumerate(output_sizes):
                checkpoint_weight = (
                    torch.randn(
                        output_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.02
                )
                in_proj.weight_loader(
                    in_proj.weight,
                    checkpoint_weight,
                    loaded_shard_id=shard_id,
                )
                reference_shards.append(checkpoint_weight.chunk(8, dim=0)[0].clone())
                del checkpoint_weight
            checkpoint_b_weight = (
                torch.randn(
                    num_heads,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.02
            )
            b_proj.weight_loader(b_proj.weight, checkpoint_b_weight)
            reference_b_weight = checkpoint_b_weight.chunk(8, dim=0)[0].clone()
        reference_weight = torch.cat(reference_shards, dim=0)

        for num_tokens in (1, 16):
            hidden_states = (
                torch.randn(
                    num_tokens,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.02
            )
            expected = F.linear(hidden_states, reference_weight)
            actual = in_proj(hidden_states)
            expected_beta = F.linear(hidden_states, reference_b_weight)
            actual_beta = b_proj(hidden_states)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(
                actual_beta, expected_beta, rtol=2e-2, atol=2e-2
            )
            max_abs = (actual.float() - expected.float()).abs().max().item()
            beta_max_abs = (
                (actual_beta.float() - expected_beta.float()).abs().max().item()
            )
            print(
                f"tokens={num_tokens} max_abs_diff={max_abs:.8f} "
                f"beta_max_abs_diff={beta_max_abs:.8f}"
            )
        """)
    env = os.environ.copy()
    env.setdefault("AITER_LOG_LEVEL", "WARNING")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=KIMI_K3_PATH.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")
