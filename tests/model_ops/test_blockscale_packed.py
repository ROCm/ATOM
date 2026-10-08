# SPDX-License-Identifier: MIT
"""Numerical and replay contracts for native group32 FP8 projections."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import gemm_op_a8w8

from atom.model_ops import blockscale
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
    [(m, 5120, k) for m in (1, 3, 4) for k in (1152, 1312, 1664)] + [(3, 2053, 1280)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compat_packed_partial_k_panels_remain_finite_on_graph_replay(
    monkeypatch, m, n, k, dtype
):
    """Short shared-expert batches must not read undefined pipelined K tails.

    In Triton 3.7 the two-stage PACK=4 kernel produced NaNs for the TP2
    shared-expert down projection (N=5120, K=1152), even with finite operands.
    Exercise the compatibility kernel even when AITER has its own backend.
    """
    monkeypatch.setattr(blockscale, "_aiter_fp8_gemm", None)
    x, weight, xs, ws = _operands(m, n, k)
    native_quant_linear(x, weight, ws, x_scale=xs, dtype=dtype)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = native_quant_linear(x, weight, ws, x_scale=xs, dtype=dtype)
    for factor in (1.0, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        expected = _reference(x, weight, xs, ws)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(),
            expected.to(dtype).float(),
            rtol=0.016 if dtype == torch.bfloat16 else 3e-5,
            atol=5e-5 * expected.abs().max().item(),
        )


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
        # N-first scheduling: token/column tails and short K tails.
        (63, 8193, 1280),
        (129, 4097, 576),
        (255, 16385, 1152),
        (1023, 4097, 1280),
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
@pytest.mark.parametrize("n,k", [(2048, 1280), (2053, 1152)])
def test_group32_projection_extreme_scale_codes(a_code, b_code, n, k):
    x = torch.ones(3, k, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.ones(n, k, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.full((3, k // 32), a_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    ws = torch.full(
        ((n + 31) // 32, k // 32), b_code, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    actual = native_quant_linear(x, weight, ws, x_scale=xs, dtype=torch.float32)
    if 255 in (a_code, b_code):
        assert actual.isnan().all()
    else:
        expected = torch.full_like(actual, k * 2.0 ** (a_code + b_code - 254))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,n,k,split_k", [(3, 4096, 1280, None), (33, 8193, 576, 3)]
)
def test_group32_projection_graph_reads_live_inputs_and_scales(rows, n, k, split_k):
    x, weight, xs, ws = _operands(2 * rows, n, k)
    x = x.view(2, rows, k)
    xs = xs.view(2, rows, k // 32)
    native_quant_linear(x, weight, ws, x_scale=xs, dtype=torch.float32, split_k=split_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = native_quant_linear(
            x, weight, ws, x_scale=xs, dtype=torch.float32, split_k=split_k
        )
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_row_scaled_fp8_projection_with_token_and_column_tails(dtype):
    x, weight, xs, ws = _operands(97, 8193, 576)
    # Per-row scales share the scheduling kernel but not the compact weight grid.
    ws = ws.repeat_interleave(32, 0)[: weight.shape[0]].contiguous()
    actual = native_quant_linear(
        x, weight, ws, x_scale=xs, weight_group_rows=1, dtype=dtype, split_k=3
    )
    expected = (x.double() * xs.double().repeat_interleave(32, -1)) @ (
        weight.double() * ws.double().repeat_interleave(32, -1)
    ).T
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016 if dtype == torch.bfloat16 else 3e-5,
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize("shape", [(3, 1280), (2, 3, 1280), (0, 1280)])
@pytest.mark.parametrize("group_rows", [1, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp8_projection_uses_aiter_backend_config(
    monkeypatch, shape, group_rows, dtype
):
    if blockscale._aiter_fp8_gemm is None:
        pytest.skip("AITER native group32 public interface is unavailable")
    backend = pytest.importorskip(
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32"
    ).gemm_a8w8_blockscale_group32
    m = 1
    for dim in shape[:-1]:
        m *= dim
    x, weight, xs, ws = _operands(m, 4096, shape[-1])
    if group_rows == 1:
        ws = ws.repeat_interleave(32, 0).contiguous()
    expected = backend(x, weight, xs, ws, dtype=dtype, weight_group_rows=group_rows)
    lookup = gemm_op_a8w8.get_CKGEMM_config
    calls = []

    def record_lookup(m, n, k, tuned_file):
        config = lookup(m, n, k, tuned_file)
        calls.append((m, n, k, tuned_file, config))
        return config

    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", record_lookup)
    actual = native_quant_linear(
        x.view(shape),
        weight,
        ws,
        x_scale=xs.view(*shape[:-1], shape[-1] // 32),
        weight_group_rows=group_rows,
        dtype=dtype,
    )
    assert actual.shape == (*shape[:-1], weight.shape[0])
    torch.testing.assert_close(actual, expected.view(actual.shape), rtol=0, atol=0)
    assert len(calls) == 1
    rows, n, k, tuned_file, config = calls[0]
    assert (rows, n, k) == (m, 4096, 1280)
    assert tuned_file.endswith("a8w8_blockscale_group32_tuned_gemm.csv")
    assert config is not None and config["libtype"] == "triton"


@pytest.mark.parametrize("declared_group_rows", [1, 32])
def test_fp8_projection_rejects_mismatched_weight_scale_group(declared_group_rows):
    x, weight, xs, ws = _operands(3, 65, 64)
    if declared_group_rows == 32:
        ws = ws.repeat_interleave(32, 0)[:65].contiguous()
    with pytest.raises(ValueError, match="Weight scale shape"):
        native_quant_linear(
            x, weight, ws, x_scale=xs, weight_group_rows=declared_group_rows
        )


def test_fp8_projection_compile_dynamic_batched_rows():
    def forward(x, weight, xs, ws):
        return native_quant_linear(x, weight, ws, x_scale=xs)

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for rows in (3, 7, 33):
        x, weight, xs, ws = _operands(2 * rows, 4096, 1280)
        x = x.view(2, rows, 1280)
        xs = xs.view(2, rows, 40)
        torch.testing.assert_close(
            compiled(x, weight, xs, ws), forward(x, weight, xs, ws), rtol=0, atol=0
        )
