# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the actual QKV gate on CPU, without importing the GPU-backed model.

Only the function AST is compiled: no copy of the gate, no module-level skip,
and no replacement of ATOM modules. The aiter import is controlled per test;
kernel correctness and decode performance still require gfx1250 hardware.
"""

import ast
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, path, "exec"), namespace)  # noqa: S102 - test repository code
    return namespace[name]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("ATOM_USE_MODEL_SENSITIVE_RMSNORM", "0")
    spec = importlib.util.spec_from_file_location(
        "qkv_test_envs", ROOT / "atom/utils/envs.py"
    )
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    real_import = builtins.__import__
    backend = SimpleNamespace(qkv_a_rmsnorm_fp4=object())

    def controlled_import(name, *args, **kwargs):
        if name == "aiter.ops.rmsnorm":
            if backend.qkv_a_rmsnorm_fp4 is None:
                raise ImportError("aiter predates qkv_a_rmsnorm_fp4")
            return backend
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    function = _load_function(
        "atom/models/deepseek_v2.py",
        "_qkv_a_norm_fp4_supported",
        {"nn": SimpleNamespace(Module=object), "envs": envs},
    )
    return function, backend, envs


def _attn(**overrides):
    base = {
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "indexer": None,
        "q_a_layernorm": SimpleNamespace(x_pad_to_multiple=0),
        "kv_a_layernorm": SimpleNamespace(x_pad_to_multiple=0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("q_lora_rank", [1152, 1280, 1408, 1536])
@pytest.mark.parametrize("kv_lora_rank", [0, 8, 504, 512])
def test_supported_shapes(gate, q_lora_rank, kv_lora_rank):
    supported, _, _ = gate
    assert supported(_attn(q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank))


@pytest.mark.parametrize("q_lora_rank", [None, 896, 1024, 1025, 1153, 1537, 1664])
def test_refuses_unsupported_q_rank(gate, q_lora_rank):
    supported, _, _ = gate
    assert not supported(_attn(q_lora_rank=q_lora_rank))


@pytest.mark.parametrize("kv_lora_rank", [1, 511, 513, 1024])
def test_refuses_unsupported_kv_rank(gate, kv_lora_rank):
    supported, _, _ = gate
    assert not supported(_attn(kv_lora_rank=kv_lora_rank))


def test_refuses_an_indexer(gate):
    supported, _, _ = gate
    assert not supported(_attn(indexer=object()))


@pytest.mark.parametrize("norm", ["q_a_layernorm", "kv_a_layernorm"])
def test_refuses_a_padded_norm(gate, norm):
    supported, _, _ = gate
    assert not supported(_attn(**{norm: SimpleNamespace(x_pad_to_multiple=8)}))


def test_missing_aiter_op_falls_back(gate):
    supported, backend, _ = gate
    backend.qkv_a_rmsnorm_fp4 = None
    assert not supported(_attn())


def test_model_sensitive_rmsnorm_falls_back(gate, monkeypatch):
    supported, _, _ = gate
    monkeypatch.setenv("ATOM_USE_MODEL_SENSITIVE_RMSNORM", "1")
    assert not supported(_attn())


def test_env_flag_defaults_off(gate, monkeypatch):
    _, _, envs = gate
    monkeypatch.delenv("ATOM_DS_QKV_A_NORM_FP4", raising=False)
    assert envs.ATOM_DS_QKV_A_NORM_FP4 is False


@pytest.mark.parametrize("value, expected", [("0", False), ("1", True)])
def test_env_flag_is_explicit(gate, monkeypatch, value, expected):
    _, _, envs = gate
    monkeypatch.setenv("ATOM_DS_QKV_A_NORM_FP4", value)
    assert envs.ATOM_DS_QKV_A_NORM_FP4 is expected
