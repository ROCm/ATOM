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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="runs the FlyDSL and Triton GEMMs"
)
@pytest.mark.parametrize(
    "n,k",
    [(2304, 6144), (6144, 2048), (6144, 6144), (6144, 3072)],  # DSpark draft, TP4
)
@pytest.mark.parametrize("m", [1, 9, 200])
def test_preshuffled_1x32_matches_the_row_major_gemm_bit_for_bit(n, k, m):
    """The MiniMax-M3.1 DSpark draft's MXFP8 linears on AITER's preshuffled
    group32 GEMM: same bytes, scales and products as the row-major (Triton)
    path, so the output is identical."""
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.shuffle import shuffle_weight

    if get_gfx() != "gfx950":
        pytest.skip("the preshuffled 1x32 GEMM is gfx950's")
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda") * 0.05
    exp = torch.ceil(
        torch.log2(w.view(n, k // 32, 32).abs().amax(-1).clamp_min(1e-12) / 448.0)
    )
    wq = (w.view(n, k // 32, 32) / torch.pow(2.0, exp)[..., None]).view(n, k)
    wq = wq.clamp(-448, 448).to(torch.float8_e4m3fn)
    ws = (exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    row_major = blockscale.native_quant_linear(x, wq, ws, weight_group_rows=1)
    preshuffled = blockscale.native_quant_linear(
        x,
        shuffle_weight(wq, (16, 16)),
        ws,
        weight_group_rows=1,
        weight_preshuffled=True,
    )
    torch.testing.assert_close(preshuffled, row_major, rtol=0, atol=0)
