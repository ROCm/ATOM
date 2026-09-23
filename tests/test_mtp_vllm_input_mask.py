"""Contract tests for the vLLM plugin's MTP position-0 input mask.

The mask used to be installed by rebinding every draft layer's ``forward`` to a
wrapper with a hardcoded five-argument signature. When the fused FP8 prologue
gave ``DeepSeekMultiTokenPredictorLayer.forward`` a sixth argument, the wrapper
kept advertising the old arity and every vLLM-plugin MTP run died during
``profile_run``::

    TypeError: DeepSeekMultiTokenPredictorLayer.forward() takes from 5 to 6
    positional arguments but 7 were given

The mask now lives on the predictors, which own the embedding lookup on both
the fused and unfused paths. These tests pin that down: the flag exists and is
honoured, and no one reintroduces a signature-shaped wrapper.

Parsed rather than imported -- ``atom.plugin.vllm.model_wrapper`` needs vLLM and
``atom.models.deepseek_mtp`` needs an AITER build, neither of which a plain CPU
test runner has. That also lets the file live outside ``tests/plugin/``, which
``run_unit_tests.sh`` skips: a guard against this regression is worth nothing in
a directory CI never runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from atom.models.utils import mask_pos0_inputs_embeds

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_WRAPPER = REPO_ROOT / "atom" / "plugin" / "vllm" / "model_wrapper.py"

# arch -> (model module, predictor class that must carry the flag)
MASKED_PREDICTORS = {
    "DeepSeekMTPModel": ("deepseek_mtp.py", "DeepSeekMultiTokenPredictor"),
    "Glm4MoeMTPModel": ("glm4_moe_mtp.py", "Glm4MoeMultiTokenPredictor"),
}

FLAG = "mask_pos0_inputs_embeds"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls.name}.{name} not found")


def test_mask_helper_zeroes_only_position_zero_rows():
    embeds = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    positions = torch.tensor([0, 1, 0, 2])

    masked = mask_pos0_inputs_embeds(embeds, positions)

    assert torch.equal(masked[0], torch.zeros(3))
    assert torch.equal(masked[2], torch.zeros(3))
    assert torch.equal(masked[1], embeds[1])
    assert torch.equal(masked[3], embeds[3])


def test_masked_archs_are_exactly_the_predictors_that_support_the_flag():
    tree = _parse(MODEL_WRAPPER)
    archs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == (
            "_MTP_MASK_INPUT_ARCH"
        ):
            archs = {elt.value for elt in node.value.elts}
    assert archs is not None, "_MTP_MASK_INPUT_ARCH not found"
    assert archs == set(MASKED_PREDICTORS), (
        "an arch was added to _MTP_MASK_INPUT_ARCH without teaching its "
        f"predictor about {FLAG}: {archs ^ set(MASKED_PREDICTORS)}"
    )


def test_predictors_declare_and_apply_the_flag():
    for module_name, predictor_name in MASKED_PREDICTORS.values():
        cls = _class(
            _parse(REPO_ROOT / "atom" / "models" / module_name), predictor_name
        )

        init_src = ast.dump(_method(cls, "__init__"))
        assert f"attr='{FLAG}'" in init_src, (
            f"{predictor_name}.__init__ must default {FLAG} to False so the "
            "plugin has something to switch on"
        )

        forward_src = ast.dump(_method(cls, "forward"))
        assert FLAG in forward_src, (
            f"{predictor_name}.forward must honour {FLAG}; the plugin sets it "
            "and never inspects the result"
        )


def test_plugin_switches_the_flag_instead_of_rebinding_layer_forwards():
    source = MODEL_WRAPPER.read_text(encoding="utf-8")

    assert (
        f"{FLAG} = True" in source
    ), "the vLLM plugin must enable the mask by setting the predictor flag"
    assert "layer.forward = " not in source, (
        "rebinding an MTP layer's forward hardcodes its signature -- that is "
        "exactly the drift that broke every plugin MTP run"
    )
