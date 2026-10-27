# SPDX-License-Identifier: MIT
"""Availability is resolved at import; execution errors must remain visible."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="the dispatch under test is AITER's")

from atom.model_ops import blockscale


@pytest.mark.parametrize("interface", ["missing", "legacy", "native"])
def test_fp8_interface_detection_at_module_load(monkeypatch, interface):
    def legacy(x, w, xs, ws, dtype=torch.bfloat16, isBpreshuffled=False):
        raise AssertionError("Availability detection must not execute a GEMM")

    def native(*args, **kwargs):
        raise AssertionError("Availability detection must not execute a GEMM")

    if interface == "missing":
        monkeypatch.delattr(aiter, "gemm_a8w8_blockscale", raising=False)
    else:
        monkeypatch.setattr(
            aiter,
            "gemm_a8w8_blockscale",
            native if interface == "native" else legacy,
            raising=False,
        )
    parameters = ["XQ", "WQ", "x_scale", "w_scale", "dtype", "isBpreshuffled"]
    if interface == "native":
        parameters.append("split_k")
    schema = SimpleNamespace(arguments=[SimpleNamespace(name=n) for n in parameters])
    monkeypatch.setattr(
        torch.ops.aiter,
        "gemm_a8w8_blockscale",
        SimpleNamespace(default=SimpleNamespace(_schema=schema)),
        raising=False,
    )
    spec = importlib.util.spec_from_file_location(
        "atom.model_ops._blockscale_compat_test", blockscale.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._aiter_fp8_gemm is (native if interface == "native" else None)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_aiter_execution_errors_are_not_hidden(monkeypatch, error_type):
    error = error_type("configured backend failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(blockscale, "_aiter_fp8_gemm", fail)
    with pytest.raises(error_type) as raised:
        blockscale.native_quant_linear(
            torch.empty((3, 64), dtype=torch.float8_e4m3fn),
            torch.empty((65, 64), dtype=torch.float8_e4m3fn),
            torch.empty((3, 2), dtype=torch.float8_e8m0fnu),
            x_scale=torch.empty((3, 2), dtype=torch.float8_e8m0fnu),
        )
    assert raised.value is error
