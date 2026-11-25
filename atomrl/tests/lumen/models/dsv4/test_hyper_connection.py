# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HYPER_CONNECTION_PATH = (
    Path(__file__).parents[3].parent
    / "atomrl/lumen/models/dsv4/ops/hyper_connection.py"
)


def _load_hyper_connection(monkeypatch, calls):
    einops = types.ModuleType("einops")
    einops.rearrange = lambda tensor, _pattern: tensor
    einops.repeat = lambda tensor, _pattern, **_axes: tensor
    monkeypatch.setitem(sys.modules, "einops", einops)

    fusions = types.ModuleType("aiter.ops.triton.fusions.mhc")

    def mhc_pre_dsv4(residual, fn, scale, base, **kwargs):
        calls["pre"] = (residual, fn, scale, base, kwargs)
        m, n, c = residual.shape
        post = torch.arange(m * n, dtype=torch.float32).reshape(m, n, 1)
        comb = torch.arange(m * n * n, dtype=torch.float32).reshape(m, n, n)
        layer_input = torch.arange(m * c, dtype=residual.dtype).reshape(m, c)
        return post, comb, layer_input

    def mhc_post_dsv4(layer_input, residual, post, comb):
        calls["post"] = (layer_input, residual, post, comb)
        return residual + layer_input[:, None, :]

    def mhc_head_dsv4(residual, fn, scale, base, **kwargs):
        calls["head"] = (residual, fn, scale, base, kwargs)
        return residual.sum(dim=1)

    fusions.mhc_pre_dsv4 = mhc_pre_dsv4
    fusions.mhc_post_dsv4 = mhc_post_dsv4
    fusions.mhc_head_dsv4 = mhc_head_dsv4

    for name in ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.fusions"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.fusions.mhc", fusions)

    module_mod = types.ModuleType("megatron.core.transformer.module")

    class MegatronModule(torch.nn.Module):
        def __init__(self, config):
            super().__init__()

    module_mod.MegatronModule = MegatronModule
    config_mod = types.ModuleType("megatron.core.transformer.transformer_config")
    config_mod.TransformerConfig = object
    for name in ("megatron", "megatron.core", "megatron.core.transformer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.module", module_mod)
    monkeypatch.setitem(
        sys.modules, "megatron.core.transformer.transformer_config", config_mod
    )

    legacy_backend = types.ModuleType("atomrl.lumen.models.dsv4.ops.mhc_backend")

    def get_mhc_op(_name):
        raise AssertionError("legacy MHC backend was invoked")

    legacy_backend.get_mhc_op = get_mhc_op
    for name in (
        "atomrl.lumen",
        "atomrl.lumen.models",
        "atomrl.lumen.models.dsv4",
        "atomrl.lumen.models.dsv4.ops",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules, "atomrl.lumen.models.dsv4.ops.mhc_backend", legacy_backend
    )

    module_name = "_test_dsv4_hyper_connection"
    spec = importlib.util.spec_from_file_location(module_name, HYPER_CONNECTION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _config():
    return SimpleNamespace(
        layernorm_epsilon=1e-5,
        dsv4_hc_mult=4,
        dsv4_hc_sinkhorn_iters=7,
        dsv4_hc_eps=2e-6,
    )


def test_hc_pre_calls_aiter_with_flattened_batch_sequence(monkeypatch):
    calls = {}
    module = _load_hyper_connection(monkeypatch, calls)
    util = module.DeepSeekV4HyperConnectionUtil(_config())
    b, s, n, c = 2, 3, 4, 5
    x = torch.randn(b, s, n, c, dtype=torch.float32)
    fn = torch.randn(2 * n + n * n, n * c, dtype=torch.float64)
    scale = torch.randn(3, dtype=torch.float64)
    base = torch.randn(2 * n + n * n, dtype=torch.float64)

    layer_input, post, comb = util.hc_pre_raw(x, fn, scale, base)

    residual_arg, fn_arg, scale_arg, base_arg, kwargs = calls["pre"]
    assert residual_arg.shape == (b * s, n, c)
    assert residual_arg.dtype == torch.bfloat16
    assert residual_arg.is_contiguous()
    assert fn_arg.dtype == scale_arg.dtype == base_arg.dtype == torch.float32
    assert kwargs == {
        "rms_eps": 1e-5,
        "hc_pre_eps": 2e-6,
        "hc_sinkhorn_eps": 2e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 7,
    }
    assert layer_input.shape == (b, s, c)
    assert layer_input.dtype == x.dtype
    assert post.shape == (b, s, n, 1)
    assert comb.shape == (b, s, n, n)
    assert post.dtype == comb.dtype == torch.float32


def test_hc_post_calls_aiter_with_flattened_batch_sequence(monkeypatch):
    calls = {}
    module = _load_hyper_connection(monkeypatch, calls)
    util = module.DeepSeekV4HyperConnectionUtil(_config())
    b, s, n, c = 2, 3, 4, 5
    x = torch.randn(b, s, c, dtype=torch.float32)
    residual = torch.randn(b, s, n, c, dtype=torch.float32)
    post = torch.randn(b, s, n, 1, dtype=torch.float32)
    comb = torch.randn(b, s, n, n, dtype=torch.float32)

    output = util.hc_post_raw(x, residual, post, comb)

    layer_arg, residual_arg, post_arg, comb_arg = calls["post"]
    assert layer_arg.shape == (b * s, c)
    assert residual_arg.shape == (b * s, n, c)
    assert post_arg.shape == (b * s, n, 1)
    assert comb_arg.shape == (b * s, n, n)
    assert layer_arg.dtype == residual_arg.dtype == torch.bfloat16
    assert post_arg.dtype == comb_arg.dtype == torch.float32
    assert output.shape == (b, s, n, c)
    assert output.dtype == x.dtype


def test_hc_head_calls_aiter_with_flattened_batch_sequence(monkeypatch):
    calls = {}
    module = _load_hyper_connection(monkeypatch, calls)
    util = module.DeepSeekV4HyperConnectionUtil(_config())
    b, s, n, c = 2, 3, 4, 5
    x = torch.randn(b, s, n, c, dtype=torch.float32)
    fn = torch.randn(n, n * c, dtype=torch.float64)
    scale = torch.randn(1, dtype=torch.float64)
    base = torch.randn(n, dtype=torch.float64)

    output = util.hc_head_raw(x, fn, scale, base)

    residual_arg, fn_arg, scale_arg, base_arg, kwargs = calls["head"]
    assert residual_arg.shape == (b * s, n, c)
    assert residual_arg.dtype == torch.bfloat16
    assert fn_arg.dtype == scale_arg.dtype == base_arg.dtype == torch.float32
    assert kwargs == {"rms_eps": 1e-5, "hc_pre_eps": 2e-6}
    assert output.shape == (b, s, c)
    assert output.dtype == x.dtype


def test_hyper_connection_has_no_legacy_mhc_dependency():
    source = HYPER_CONNECTION_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("mhc_backend", "tile_kernels", "tilelang"):
        assert forbidden not in source

    tree_source = inspect.cleandoc(source)
    assert "from aiter.ops.triton.fusions.mhc import" in tree_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lumen_hc_chain_matches_direct_aiter_outputs_and_gradients():
    from aiter.ops.triton.fusions.mhc import (
        mhc_head_dsv4,
        mhc_post_dsv4,
        mhc_pre_dsv4,
    )
    from atomrl.lumen.models.dsv4.ops.hyper_connection import (
        DeepSeekV4HyperConnectionUtil,
    )

    torch.manual_seed(47)
    b, s, n, c = 2, 2, 4, 64
    config = _config()
    util = DeepSeekV4HyperConnectionUtil(config)
    tensors = (
        torch.randn(b, s, n, c, device="cuda", dtype=torch.bfloat16),
        torch.randn(b, s, c, device="cuda", dtype=torch.bfloat16),
        torch.randn(2 * n + n * n, n * c, device="cuda", dtype=torch.float32),
        torch.randn(3, device="cuda", dtype=torch.float32),
        torch.randn(2 * n + n * n, device="cuda", dtype=torch.float32),
        torch.randn(n, n * c, device="cuda", dtype=torch.float32),
        torch.randn(1, device="cuda", dtype=torch.float32),
        torch.randn(n, device="cuda", dtype=torch.float32),
    )
    lumen_inputs = tuple(t.detach().clone().requires_grad_() for t in tensors)
    direct_inputs = tuple(t.detach().clone().requires_grad_() for t in tensors)

    residual, layer_update, fn, scale, base, head_fn, head_scale, head_base = (
        lumen_inputs
    )
    _layer_input, post, comb = util.hc_pre_raw(residual, fn, scale, base)
    residual_out = util.hc_post_raw(layer_update, residual, post, comb)
    lumen_output = util.hc_head_raw(residual_out, head_fn, head_scale, head_base)

    residual, layer_update, fn, scale, base, head_fn, head_scale, head_base = (
        direct_inputs
    )
    direct_post, direct_comb, _direct_layer = mhc_pre_dsv4(
        residual.reshape(b * s, n, c),
        fn,
        scale,
        base,
        rms_eps=config.layernorm_epsilon,
        hc_pre_eps=config.dsv4_hc_eps,
        hc_sinkhorn_eps=config.dsv4_hc_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=config.dsv4_hc_sinkhorn_iters,
    )
    direct_residual = mhc_post_dsv4(
        layer_update.reshape(b * s, c),
        residual.reshape(b * s, n, c),
        direct_post,
        direct_comb,
    )
    direct_output = mhc_head_dsv4(
        direct_residual,
        head_fn,
        head_scale,
        head_base,
        rms_eps=config.layernorm_epsilon,
        hc_pre_eps=config.dsv4_hc_eps,
    ).reshape(b, s, c)

    torch.testing.assert_close(lumen_output, direct_output)
    grad = torch.randn_like(lumen_output)
    lumen_output.backward(grad)
    direct_output.backward(grad)
    for lumen_input, direct_input in zip(lumen_inputs, direct_inputs):
        assert lumen_input.grad is not None
        assert direct_input.grad is not None
        torch.testing.assert_close(lumen_input.grad, direct_input.grad)
