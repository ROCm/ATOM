# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU coverage of the optional Gluon TDM API and its dispatch fallback."""

import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def modules():
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            attn_res=load("atom.model_ops.kimi_k3.attention_residual"),
            arch=load("aiter.ops.triton.utils._triton.arch_info"),
            gluon=load("triton.experimental.gluon"),
            load=load,
        )


@pytest.fixture
def runtime(modules, monkeypatch):
    # Plain namespaces are intentional: MagicMock would manufacture missing APIs.
    language = ModuleType("triton.experimental.gluon.language")
    language.amd = SimpleNamespace(
        cdna5=SimpleNamespace(
            buffer_load=Mock(),
            buffer_store=Mock(),
            tdm=SimpleNamespace(
                make_tensor_descriptor=Mock(), async_load=Mock(), async_wait=Mock()
            ),
        )
    )
    monkeypatch.setitem(sys.modules, language.__name__, language)
    monkeypatch.setattr(modules.gluon, "language", language, raising=False)
    monkeypatch.setattr(modules.gluon, "jit", lambda function: function)
    monkeypatch.setattr(modules.arch, "get_arch", Mock(return_value="gfx1250"))
    monkeypatch.setitem(
        modules.attn_res.envs.environment_variables,
        "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
        lambda: True,
    )
    modules.attn_res._attn_res_tdm_module.cache_clear()
    try:
        yield SimpleNamespace(**vars(modules), language=language)
    finally:
        modules.attn_res._attn_res_tdm_module.cache_clear()


def _intercept_import(monkeypatch, target, error):
    original = builtins.__import__
    parent, _, attribute = target.rpartition(".")

    def import_module(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target or (name == parent and attribute in (fromlist or ())):
            raise error
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_module)


@pytest.mark.parametrize(
    "path",
    [
        "amd",
        "amd.cdna5",
        "amd.cdna5.tdm",
        "amd.cdna5.buffer_load",
        "amd.cdna5.buffer_store",
        "amd.cdna5.tdm.make_tensor_descriptor",
        "amd.cdna5.tdm.async_load",
        "amd.cdna5.tdm.async_wait",
    ],
)
@pytest.mark.parametrize(
    "value", ["absent", None, 0], ids=["absent", "none", "not-callable"]
)
def test_incomplete_api_falls_back_before_kernel_import(
    runtime, monkeypatch, path, value
):
    owner = runtime.language
    *parents, attribute = path.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    if value == "absent":
        delattr(owner, attribute)
    else:
        setattr(owner, attribute, value)
    _intercept_import(
        monkeypatch,
        "atom.model_ops.kimi_k3.attention_residual_gluon",
        AssertionError("unsupported Gluon must not import the kernel module"),
    )

    assert runtime.attn_res._attn_res_tdm_module() is None


def test_complete_api_returns_real_kernel_module(runtime):
    selected = runtime.attn_res._attn_res_tdm_module()

    assert selected is runtime.load("atom.model_ops.kimi_k3.attention_residual_gluon")
    assert callable(selected.attn_res_tdm_supported)
    assert callable(selected.launch_attn_res_tdm)


@pytest.mark.parametrize("gate", ["disabled", "other-arch"])
def test_inapplicable_runtime_does_not_import_gluon(runtime, monkeypatch, gate):
    if gate == "disabled":
        monkeypatch.setitem(
            runtime.attn_res.envs.environment_variables,
            "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
            lambda: False,
        )
    else:
        runtime.arch.get_arch.return_value = "gfx950"
    _intercept_import(
        monkeypatch,
        "triton.experimental.gluon",
        AssertionError("inapplicable runtime must not import Gluon"),
    )

    assert runtime.attn_res._attn_res_tdm_module() is None
    if gate == "disabled":
        runtime.arch.get_arch.assert_not_called()


@pytest.mark.parametrize(
    "target",
    [
        "triton.experimental.gluon.language",
        "atom.model_ops.kimi_k3.attention_residual_gluon",
    ],
)
def test_missing_import_keeps_triton_fallback(runtime, monkeypatch, target):
    _intercept_import(monkeypatch, target, ImportError("optional module unavailable"))

    assert runtime.attn_res._attn_res_tdm_module() is None


def test_unexpected_import_error_is_not_swallowed(runtime, monkeypatch):
    _intercept_import(
        monkeypatch,
        "atom.model_ops.kimi_k3.attention_residual_gluon",
        RuntimeError("broken module initialization"),
    )

    with pytest.raises(RuntimeError, match="broken module initialization"):
        runtime.attn_res._attn_res_tdm_module()


def test_kernel_failure_does_not_silently_fall_back(runtime, monkeypatch):
    selected = runtime.attn_res._attn_res_tdm_module()
    monkeypatch.setattr(selected, "attn_res_tdm_supported", lambda tensor: True)
    monkeypatch.setattr(
        selected, "launch_attn_res_tdm", Mock(side_effect=RuntimeError("kernel failed"))
    )
    fallback = Mock()
    monkeypatch.setattr(runtime.attn_res, "_launch_attn_res_triton", fallback)

    with pytest.raises(RuntimeError, match="kernel failed"):
        runtime.attn_res._apply_attn_res_impl(
            torch.zeros(1, 8, dtype=torch.bfloat16),
            torch.zeros(1, 1, 8, dtype=torch.bfloat16),
            torch.ones(8),
            1e-6,
        )

    fallback.assert_not_called()
