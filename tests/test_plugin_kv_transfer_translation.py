# SPDX-License-Identifier: MIT
"""What the plugin reports as vLLM's KV transport, to every model.

This translation is not V4.1's. The ATOM `Config` it fills is built for every
plugin model, and several gates read the field as a plain truth value:
`qwen4_exp_attn` refuses outright when it is non-empty, `EngineCore` sets
`kv_transfer_enabled` from it, and `simulated_tp` rejects on it. So "no
transport" has to translate to something falsy, which is what the function's
docstring promises.

It did not. The pinned vLLM's `KVTransferConfig` is a pydantic *dataclass*
with no `model_dump`, so the field-by-field branch is the one that runs, and
its defaults alone -- a load-failure policy and an empty extra-config --
produced a truthy dict for a configuration naming no connector at all.

Executed by AST extraction so CPU CI collects it with no vLLM present.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "atom/plugin/config.py"


def _translate():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_vllm_kv_transfer_config"
    )
    scope = {"Any": object}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        scope,
    )
    return scope["_vllm_kv_transfer_config"]


def _vllm_config(transfer):
    return SimpleNamespace(kv_transfer_config=transfer)


class _PydanticDataclassLike:
    """vLLM's shape: attributes, defaults, and no `model_dump`."""

    def __init__(self, **fields):
        self.kv_connector = fields.get("kv_connector")
        self.kv_role = fields.get("kv_role")
        self.kv_connector_module_path = fields.get("kv_connector_module_path")
        self.kv_load_failure_policy = fields.get("kv_load_failure_policy", "fail")
        self.kv_connector_extra_config = fields.get("kv_connector_extra_config", {})


def test_no_transfer_config_at_all_is_off():
    assert _translate()(_vllm_config(None)) == {}
    assert _translate()(SimpleNamespace()) == {}


def test_a_config_naming_no_connector_is_off():
    """The regression: vLLM builds the object before deciding what goes in it.

    Its defaults alone used to come back as a truthy dict, which every reader
    takes as "a transport is configured" -- and `qwen4_exp_attn` refuses on it.
    """
    translated = _translate()(_vllm_config(_PydanticDataclassLike()))
    assert translated == {}
    assert not translated


@pytest.mark.parametrize(
    "empty", [None, ""], ids=["connector-None", "connector-empty-string"]
)
def test_an_empty_connector_name_is_off(empty):
    assert _translate()(_vllm_config(_PydanticDataclassLike(kv_connector=empty))) == {}
    assert (
        _translate()(_vllm_config({"kv_connector": empty, "kv_role": "kv_both"})) == {}
    )


def test_a_configured_connector_is_carried_through():
    """The field exists so a model's own gate can see the name and decide."""
    translated = _translate()(
        _vllm_config(
            _PydanticDataclassLike(kv_connector="LMCacheMPConnector", kv_role="kv_both")
        )
    )
    assert translated["kv_connector"] == "LMCacheMPConnector"
    assert translated["kv_role"] == "kv_both"
    assert translated


def test_a_dict_is_carried_through_and_copied():
    original = {"kv_connector": "NixlConnector", "kv_role": "kv_producer"}
    translated = _translate()(_vllm_config(original))
    assert translated == original
    translated["kv_connector"] = "mutated"
    assert original["kv_connector"] == "NixlConnector"


def test_model_dump_is_used_when_the_object_offers_one():
    """A newer pydantic would take this branch; it must honour the same rule."""
    dumped = {"kv_connector": "MooncakeConnector", "kv_role": "kv_both"}
    translate = _translate()
    assert translate(_vllm_config(SimpleNamespace(model_dump=lambda: dumped))) == dumped
    assert (
        translate(
            _vllm_config(
                SimpleNamespace(model_dump=lambda: {"kv_load_failure_policy": "fail"})
            )
        )
        == {}
    )
