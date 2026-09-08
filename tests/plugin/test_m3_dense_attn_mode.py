# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``_m3_dense_attn_mode`` must exclude exactly what the backend choice does.

``_mha_backend_for_layer`` decides the backend before it consults the mode, and
``AttentionForVllmMHA.__init__`` decides ``kv_separated`` from the mode. If the
two disagree a layer asks vLLM for a KV cache shape its own backend never voted
for -- which does not fail at startup, it fails at the first forward.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[2] / "atom/plugin/vllm/attention/layer_mha.py"
)


def _load_mode_helper():
    """Execute the real helper without importing its module.

    layer_mha imports aiter and vllm at module scope, neither of which a CPU
    runner has. Lifting the function out by AST keeps the test running the
    shipped code rather than a copy of it, and raises loudly if it is renamed
    or moved instead of quietly testing nothing.
    """
    tree = ast.parse(_SOURCE.read_text())
    wanted = {"_M3_MODEL_TYPES", "_m3_dense_attn_mode"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        )
    ]
    assert len(nodes) == len(wanted), (
        f"expected {sorted(wanted)} in {_SOURCE.name}, found "
        f"{[getattr(n, 'name', None) for n in nodes]}"
    )
    namespace: dict = {}
    # Only the two nodes selected above, out of a file in this repo.
    exec(  # noqa: S102
        compile(ast.Module(body=nodes, type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    return namespace["_m3_dense_attn_mode"]


_m3_dense_attn_mode = _load_mode_helper()

NUM_HIDDEN_LAYERS = 60


class _Cfg:
    def __init__(self, model_type, num_hidden_layers=NUM_HIDDEN_LAYERS):
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers


@pytest.fixture
def gluon_env(monkeypatch):
    monkeypatch.setenv("ATOM_M3_DENSE_ATTN_BACKEND", "gluon")
    from atom.utils import envs

    importlib.reload(envs)
    yield
    monkeypatch.delenv("ATOM_M3_DENSE_ATTN_BACKEND", raising=False)
    importlib.reload(envs)


@pytest.mark.parametrize("layer_num", [0, 2, NUM_HIDDEN_LAYERS - 1])
def test_m3_target_layers_take_the_mode(gluon_env, layer_num):
    assert _m3_dense_attn_mode(_Cfg("minimax_m3"), layer_num) == "gluon"


@pytest.mark.parametrize("layer_num", [NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS + 1])
def test_the_spec_decode_draft_is_excluded(gluon_env, layer_num):
    """layer_num >= num_hidden_layers is the EAGLE3 draft.

    It is routed to AiterMhaFlexibleBlockBackendForVllm before the mode is
    consulted, and its bf16 KV goes through _forward_vllm_native_combined_kv.
    Letting the mode reach its KV spec would request K/V separation on a layer
    whose backend never asked for it.
    """
    assert _m3_dense_attn_mode(_Cfg("minimax_m3"), layer_num) == "triton"


@pytest.mark.parametrize("model_type", ["llama", "qwen3_moe", "glm4_moe", ""])
def test_other_models_never_see_the_mode(gluon_env, model_type):
    assert _m3_dense_attn_mode(_Cfg(model_type), 0) == "triton"


def test_the_default_is_the_untouched_path(monkeypatch):
    monkeypatch.delenv("ATOM_M3_DENSE_ATTN_BACKEND", raising=False)
    from atom.utils import envs

    importlib.reload(envs)

    assert _m3_dense_attn_mode(_Cfg("minimax_m3"), 0) == "triton"
