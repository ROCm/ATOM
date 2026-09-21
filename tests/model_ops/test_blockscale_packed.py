# SPDX-License-Identifier: MIT
"""Numerical and replay contracts for native group32 decode projections."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops.blockscale import native_quant_linear

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="CDNA4 packed MFMA")


def _operands(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.randint(122, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    ws = torch.randint(
        122, 131, ((n + 31) // 32, k // 32), device="cuda", dtype=torch.uint8
    )
    return x, weight, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)


def _reference(x, weight, xs, ws):
    # FP64 is independent of the MFMA's internal block accumulation and of
    # either implementation's K reduction order.
    a = x.double() * xs.double().repeat_interleave(32, -1)
    b = weight.double() * ws.double().repeat_interleave(32, 0)[
        : weight.shape[0]
    ].repeat_interleave(32, -1)
    return a @ b.T


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 5120, 576),
        (3, 2053, 1280),
        (4, 8192, 1280),
        (8, 5120, 1152),
        (16, 4096, 1280),
        (31, 4096, 1280),
        (3, 5120, 2048),
        (8, 5120, 2304),
        (4, 5120, 4096),
        (3, 2304, 5120),
        (1, 5120, 8192),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_group32_projection_panel_scales_and_tails(m, n, k, dtype):
    x, weight, xs, ws = _operands(m, n, k)
    actual = native_quant_linear(x, weight, ws, x_scale=xs, dtype=dtype)
    expected = _reference(x, weight, xs, ws)
    peak = expected.abs().max().item()
    # Keep the established native-MFMA FP32 bound. BF16 additionally rounds
    # output; near cancellation still uses the same peak-relative floor.
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016 if dtype == torch.bfloat16 else 3e-5,
        atol=5e-5 * peak,
    )


@pytest.mark.parametrize(
    "a_code,b_code", [(0, 254), (254, 0), (128, 0), (255, 127), (127, 255)]
)
def test_group32_projection_extreme_scale_codes(a_code, b_code):
    x = torch.ones(3, 1280, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.ones(2048, 1280, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.full((3, 40), a_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    ws = torch.full((64, 40), b_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    actual = native_quant_linear(x, weight, ws, x_scale=xs, dtype=torch.float32)
    if 255 in (a_code, b_code):
        assert actual.isnan().all()
    else:
        expected = torch.full_like(actual, 1280 * 2.0 ** (a_code + b_code - 254))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_group32_projection_graph_reads_live_inputs_and_scales():
    x, weight, xs, ws = _operands(6, 4096, 1280)
    x = x.view(2, 3, 1280)
    xs = xs.view(2, 3, 40)
    native_quant_linear(x, weight, ws, x_scale=xs, dtype=torch.float32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = native_quant_linear(x, weight, ws, x_scale=xs, dtype=torch.float32)
    for factor in (2, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        expected = _reference(x.flatten(0, 1), weight, xs.flatten(0, 1), ws)
        torch.testing.assert_close(
            actual.flatten(0, 1).float(),
            expected.float(),
            rtol=3e-5,
            atol=5e-5 * expected.abs().max().item(),
        )
